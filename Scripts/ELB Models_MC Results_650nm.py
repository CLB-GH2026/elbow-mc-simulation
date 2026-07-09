"""
Elbow STL Mesh → 3D Voxel Volume + pmcx Fluence Overlay  (650 nm)
----------------------------------------------------------------------
Pipeline (shared logic lives in pbm_mc_core; see that package's README for
the full stage list and the tissue-label convention this script's `tissues`
dict follows). Mirrors the knee OKS batch pipeline exactly.

Tissue hierarchy (highest label wins when meshes overlap):
  1  humerus-bone         Distal humerus (capitellum + trochlea)
  2  radius-bone          Radial head + shaft
  3  ulna-bone            Olecranon + proximal ulna
  7  capitellum-cart      Capitellum articular cartilage
  8  radhead-cart         Radial head articular cartilage
  9  trochlear-cart       Trochlear articular cartilage
  5  annular-lig          Annular ligament (fibrocartilage)
  11 muscle               Synthesised — concentric dilation (extensor/flexor origin)
  12 adipose              Synthesised — concentric dilation
  13 skin                 Synthesised — concentric dilation
  14 synovial             Synthesised — dilation of cartilage/annular-lig gap
  15 epidermis            Synthesised — outermost 1-voxel skin ring

Wrapping note:
  The elbow has the shallowest target depth of all four Kineon joints.
  MUSCLE_THICK_MM = 10 mm models the thin brachioradialis/triceps/extensor
  origin covering the lateral epicondyle and radiocapitellar joint.

Source positions (default):
  +Y = anterior,  −Y = posterior,  +X = lateral (radial),  +Z = superior
  Lateral source targets the lateral epicondyle / radiocapitellar joint
  (tennis elbow); posterior targets the olecranon fossa; medial targets the
  medial epicondyle. All three Z values are auto-set to the joint-line height.

Dependencies:
    pip install numpy trimesh pmcx plotly scipy
    pip install git+https://github.com/CLB-GH2026/pbm-mc-core.git@v0.1.0
"""

import time
from pathlib import Path
from datetime import datetime

import numpy as np
import plotly.graph_objects as go

from pbm_mc_core import (
    opt, EPIDERMIS_LABEL, build_melanin_conditions,
    build_label_volume,
    add_synovial_fluid, add_wrapping_layers, add_epidermis_layer,
    find_joint_line_z, find_surface_source_positions,
    optimize_source_positions_reciprocity,
    run_pmcx,
    analyze_fluence_absorption, analyze_penetration_depth,
    results_to_csv, melanin_comparison_to_csv,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

start_time = time.perf_counter()

WAVELENGTH_M  = 650e-9
WAVELENGTH_NM = 650

# Epidermal optical properties by melanin condition at 650 nm.
# True (unscaled) values; build_melanin_conditions() applies the epidermis
# thickness-correction scale (0.2 mm physical / 1 mm voxel).
_MELANIN_RAW_650NM = {
    #        µa      µs'    g     n
    'fair':  (0.020, 1.80, 0.80, 1.40),  # Fitzpatrick I-II
    'olive': (0.070, 1.90, 0.80, 1.40),  # Fitzpatrick III-IV
    'dark':  (0.200, 2.00, 0.80, 1.40),  # Fitzpatrick V-VI
}

# ── Source optimiser ──────────────────────────────────────────────────────────
OPTIMIZE_SOURCES = False   # True → per-subject reciprocity scan before main run
OPT_N_SOURCES    = 3
OPT_MIN_SEP_MM   = 25.0
OPT_NPHOTON      = 1e6

# ── Grid / voxel ─────────────────────────────────────────────────────────────
VOXEL_SIZE    = 1.0               # mm per voxel
GRID_DIMS_MM  = (120, 110, 200)   # x, y, z — elbow is smaller than knee/shoulder
VOXEL_RES     = tuple(int(round(d / VOXEL_SIZE)) for d in GRID_DIMS_MM)
AUTO_ORIENT   = True              # auto-correct Z-axis inversion (humerus above radius)
FLUENCE_OUTPUT = None             # None = run pmcx; path string = load saved .npy

# ── Soft-tissue wrapping (mm) ─────────────────────────────────────────────────
# Extensor/flexor origin at the elbow is thin laterally — see CLAUDE.md
# "Key Differences from Knee Pipeline".
MUSCLE_THICK_MM  = 10   # brachioradialis / triceps at elbow
ADIPOSE_THICK_MM =  3
SKIN_THICK_MM    =  2

# ── Source power ──────────────────────────────────────────────────────────────
SOURCE_POWER_MW   = 120
SOURCE_DUTY_CYCLE = 0.75
SOURCE_OPT_EFF    = 0.85
CONE_ANGLE_DEG    = 20     # source cone full angle

MELANIN_CONDITIONS = build_melanin_conditions(_MELANIN_RAW_650NM, voxel_size_mm=VOXEL_SIZE)

# ─────────────────────────────────────────────────────────────────────────────
# TISSUE GROUPS — passed into analyze_fluence_absorption / results_to_csv /
# melanin_comparison_to_csv, which are anatomy-agnostic in pbm_mc_core.
# ─────────────────────────────────────────────────────────────────────────────
GROUPS = {
    'Bone':       lambda n: 'bone'     in n,
    'Cartilage':  lambda n: 'cart'     in n,
    'AnnularLig': lambda n: 'annular'  in n,
    'Synovial':   lambda n: 'synovial' in n,
    'Muscle':     lambda n: 'muscle'   in n,
    'Adipose':    lambda n: 'adipose'  in n,
    'Skin':       lambda n: 'skin'     in n,
}
DOSE_GROUPS = {
    'Cartilage':      lambda n: 'cart'     in n,
    'Muscle':         lambda n: 'muscle'   in n,
    'Synovial Fluid': lambda n: 'synovial' in n,
}
COMP_GROUPS = {
    'Cartilage':      lambda n: 'cart'     in n,
    'AnnularLig':     lambda n: 'annular'  in n,
    'Synovial Fluid': lambda n: 'synovial' in n,
    'Muscle':         lambda n: 'muscle'   in n,
    'Bone':           lambda n: 'bone'     in n,
    'Skin+Epidermis': lambda n: 'skin' in n or 'epidermis' in n,
}

# Elbow's "target" tissue for joint-line detection / source-position
# optimisation includes the annular ligament (fibrocartilage) in addition to
# cartilage and synovial fluid — unlike the knee/shoulder default predicate.
_TARGET_MATCH_FN = lambda n: ('cart' in n) or ('annular' in n) or ('synovial' in n)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PER-SUBJECT RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_subject(subject_id, mesh_dir_base, output_dir, melanin_condition='fair'):
    """Run the full pipeline for a single elbow subject."""

    mesh_dir = Path(mesh_dir_base) / f"Raw_Mesh_Files_{subject_id}"
    if not mesh_dir.exists():
        print(f"  Skipping {subject_id} — directory not found: {mesh_dir}")
        return None

    print(f"\n{'=' * 60}")
    print(f"  Processing {subject_id}  [{melanin_condition}]")
    print(f"{'=' * 60}")

    # ── Tissue table ─────────────────────────────────────────────────────────
    # Optical properties at 650 nm (µa, µs', g, n).
    # Cartilage, annular ligament, bone values are the same as the knee pipeline.
    # Annular ligament is fibrocartilage — same optical class as knee meniscus.
    tissues = {
        "synovial":       (None,                                              14, opt(0.0005, 0.01,  0.90, 1.36)),
        "skin":           (None,                                              13, opt(0.011,  1.50,  0.80, 1.40)),
        "adipose":        (None,                                              12, opt(0.003,  1.20,  0.90, 1.44)),
        "muscle":         (None,                                              11, opt(0.0280, 0.60,  0.93, 1.37)),
        "annular-lig":    (mesh_dir / "annular_lig_raw.stl",                   5, opt(0.014,  2.00,  0.90, 1.37)),  # fibrocartilage/ligament
        "trochlear-cart": (mesh_dir / "trochlear_cartilage_raw.stl",           9, opt(0.025,  1.20,  0.90, 1.37)),  # hyaline
        "radhead-cart":   (mesh_dir / "radial_head_cartilage_raw.stl",         8, opt(0.025,  1.20,  0.90, 1.37)),  # hyaline
        "capitellum-cart":(mesh_dir / "capitellum_cartilage_raw.stl",          7, opt(0.025,  1.20,  0.90, 1.37)),  # hyaline
        "ulna-bone":      (mesh_dir / "ulna_raw.stl",                          3, opt(0.068,  2.80,  0.92, 1.37)),
        "radius-bone":    (mesh_dir / "radius_raw.stl",                       2, opt(0.068,  2.80,  0.92, 1.37)),
        "humerus-bone":   (mesh_dir / "humerus_distal_raw.stl",               1, opt(0.068,  2.80,  0.92, 1.37)),
    }
    tissues["epidermis"] = (None, EPIDERMIS_LABEL, MELANIN_CONDITIONS[melanin_condition])

    try:
        # ── Step 1: Build label volume ────────────────────────────────────
        vol, origin, mesh_center = build_label_volume(
            tissues, VOXEL_RES, VOXEL_SIZE,
            auto_orient=AUTO_ORIENT,
            orient_ref_a='humerus-bone', orient_ref_b='radius-bone',
        )

        bone_labels      = [t[1] for name, t in tissues.items() if "bone"    in name]
        cartilage_labels = [t[1] for name, t in tissues.items() if "cart"    in name]
        labrum_labels    = [t[1] for name, t in tissues.items() if "annular" in name]

        vol = add_synovial_fluid(
            vol,
            cartilage_labels=cartilage_labels + labrum_labels,
            bone_labels=bone_labels,
            fluid_label=tissues["synovial"][1],
            dilation_vox=3
        )

        layer_configs_vox = [
            (tissues["muscle"][1],  int(round(MUSCLE_THICK_MM  / VOXEL_SIZE))),
            (tissues["adipose"][1], int(round(ADIPOSE_THICK_MM / VOXEL_SIZE))),
            (tissues["skin"][1],    int(round(SKIN_THICK_MM    / VOXEL_SIZE))),
        ]
        vol = add_wrapping_layers(vol, layer_configs_vox)
        vol = add_epidermis_layer(vol, skin_label=tissues["skin"][1],
                                   epidermis_label=EPIDERMIS_LABEL)

        # ── Step 2b: Locate joint line Z ─────────────────────────────────
        jl_z = find_joint_line_z(vol, tissues, origin, VOXEL_SIZE, mesh_center,
                                  target_match_fn=_TARGET_MATCH_FN)

        _colors = ['red', 'green', 'blue', 'orange', 'purple']
        if OPTIMIZE_SOURCES:
            print("\n--- Reciprocity source position optimisation ---")
            opt_positions = optimize_source_positions_reciprocity(
                vol, tissues, origin, mesh_center, VOXEL_SIZE,
                OPT_N_SOURCES, OPT_MIN_SEP_MM, OPT_NPHOTON,
                epidermis_label=EPIDERMIS_LABEL,
                target_match_fn=_TARGET_MATCH_FN,
            )
            if opt_positions:
                src_configs = [
                    {'name': f'Opt-{i+1}', 'world_pos': pos, 'color': _colors[i % len(_colors)]}
                    for i, pos in enumerate(opt_positions)
                ]
            else:
                print("  [OPT] Falling back to default positions")
                src_configs = _default_src_configs(jl_z)
        else:
            src_configs = _default_src_configs(jl_z)

        for cfg in src_configs:
            d = np.array([0, 0, jl_z]) - np.array(cfg['world_pos'])
            cfg['srcdir'] = (d / np.linalg.norm(d)).tolist()

        pmcx_source_plus = find_surface_source_positions(
            vol, origin, VOXEL_SIZE, mesh_center, src_configs
        )
        pmcx_source = [{'srcpos': s['srcpos'], 'srcdir': s['srcdir']}
                       for s in pmcx_source_plus]

        # ── Step 4: Run pmcx ──────────────────────────────────────────────
        fluence_combined, fluence_list = run_pmcx(
            vol, tissues, pmcx_source,
            wavelength_m=WAVELENGTH_M,
            source_power_mw=SOURCE_POWER_MW,
            duty_cycle=SOURCE_DUTY_CYCLE,
            opt_eff=SOURCE_OPT_EFF,
            cone_angle_deg=CONE_ANGLE_DEG,
            voxel_size_mm=VOXEL_SIZE,
        )

        # ── Step 6: Absorption analysis ───────────────────────────────────
        results = analyze_fluence_absorption(
            fluence_combined, vol, tissues, VOXEL_SIZE,
            pmcx_source=pmcx_source,
            groups=GROUPS,
            source_power_mw=SOURCE_POWER_MW,
            duty_cycle=SOURCE_DUTY_CYCLE,
            opt_eff=SOURCE_OPT_EFF,
        )

        subj_dir = Path(output_dir) / melanin_condition / subject_id
        subj_dir.mkdir(parents=True, exist_ok=True)

        cart_names  = [n for n in results if 'cart'     in n]
        cart_vox    = sum(results[n]['n_voxels'] for n in cart_names)
        cart_flu_mw = (sum(results[n]['mean_flu'] * results[n]['n_voxels']
                           for n in cart_names) / cart_vox) if cart_vox > 0 else 0.0

        syn_names   = [n for n in results if 'synovial' in n]
        syn_vox     = sum(results[n]['n_voxels'] for n in syn_names)
        syn_flu_mw  = (sum(results[n]['mean_flu'] * results[n]['n_voxels']
                           for n in syn_names) / syn_vox) if syn_vox > 0 else 0.0

        print("\n=== Penetration depth analysis ===")
        bin_centers, mean_flu, max_depth = analyze_penetration_depth(
            fluence_combined, vol, VOXEL_SIZE, mesh_center, origin
        )
        fig_depth = plot_depth_histogram(
            bin_centers, mean_flu, subject_id, WAVELENGTH_NM,
            cartilage_flu_mw=cart_flu_mw,
            synovial_flu_mw=syn_flu_mw,
        )
        depth_html = str(subj_dir / f"depth_histogram_{subject_id}_{melanin_condition}.html")
        fig_depth.write_html(depth_html)
        print(f"  Saved: {depth_html}")

        np.save(subj_dir / "label_volume.npy", vol)
        np.save(subj_dir / "fluence_combined.npy", fluence_combined)
        for i, flu in enumerate(fluence_list):
            np.save(subj_dir / f"fluence_src{i + 1}.npy", flu)

        return subject_id, results

    except Exception as e:
        print(f"  ERROR processing {subject_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _default_src_configs(jl_z):
    """
    Default source positions for the elbow at 650 nm.

    Coordinate convention:
      +Y = anterior,  −Y = posterior,  +X = lateral (radial),  +Z = superior

    Elbow anatomy:
      Lateral source:    over the lateral epicondyle (target for lateral
                         epicondylitis / tennis elbow, X ≈ +40 mm).
      Posterior:         over the olecranon fossa (Y ≈ −35 mm).
      Medial:            over the medial epicondyle (X ≈ −35 mm).

    All Z values are auto-set to jl_z (radiocapitellar joint height).
    """
    return [
        {'name': 'Lateral',   'world_pos': [ 40,   0, jl_z], 'color': 'red'  },
        {'name': 'Posterior', 'world_pos': [  0, -35, jl_z], 'color': 'green'},
        {'name': 'Medial',    'world_pos': [-35,   0, jl_z], 'color': 'blue' },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# NOTE: plot_depth_histogram is kept LOCAL rather than imported from
# pbm_mc_core. The shared library's analysis.plot_depth_histogram hardcodes
# knee-anatomy depth references (DEPTH_REFS at 0.8/2.0/3.5 cm, dose zone
# 2.0-3.5 cm) that are not exposed as parameters. The elbow is explicitly the
# shallowest of the four joints (see CLAUDE.md: skin ~0.5 cm, muscle ~1 cm,
# joint ~2 cm) — using the library version verbatim would silently replace
# elbow's correct depth references/dose zone with knee's. Only the
# `wavelength_nm` parameter (the one genuine bug: the title previously read
# "... Shoulder" and this script hardcoded melanin_comparison_to_csv's
# wavelength_nm to 808 even though this is the 650 nm script) has been
# adopted from the library's fix — WAVELENGTH_NM is now a single module
# constant threaded through both the title and the CSV header.
# ─────────────────────────────────────────────────────────────────────────────

def plot_depth_histogram(bin_centers, mean_flu, subject_id, wavelength_nm,
                          bin_width_cm=0.25, treatment_times_s=(300, 600, 900),
                          cartilage_flu_mw=0.0, synovial_flu_mw=0.0):
    # Elbow anatomy depth references (approximate, lateral access):
    #   ~0.5 cm  skin + adipose (very thin laterally)
    #   ~1.0 cm  muscle / extensor origin
    #   ~2.0 cm  lateral elbow joint space
    DEPTH_REFS  = [(0.5, 'Skin/Adipose'), (1.0, 'Muscle'), (2.0, 'Elbow Joint')]
    ZONE_LO, ZONE_HI = 1.0, 2.5

    bin_centers = np.asarray(bin_centers)
    mean_flu    = np.asarray(mean_flu)
    zone_mask   = (bin_centers >= ZONE_LO) & (bin_centers <= ZONE_HI)
    zone_width  = ZONE_HI - ZONE_LO
    n_zone      = zone_mask.sum()
    if n_zone >= 2:
        zone_integral = float(np.trapezoid(mean_flu[zone_mask], bin_centers[zone_mask]))
    elif n_zone == 1:
        zone_integral = float(mean_flu[zone_mask][0] * bin_width_cm)
    else:
        zone_integral = 0.0
    zone_norm_mw  = zone_integral / zone_width
    dose_lines    = [f"  {t // 60:.0f} min:  {zone_norm_mw * 1e-3 * t:.4f} J/cm²"
                     for t in treatment_times_s]
    cart_doses    = [f"  {t // 60:.0f} min:  {cartilage_flu_mw * 1e-3 * t:.4f} J/cm²"
                     for t in treatment_times_s]
    syn_doses     = [f"  {t // 60:.0f} min:  {synovial_flu_mw * 1e-3 * t:.4f} J/cm²"
                     for t in treatment_times_s]
    annot_text = (
        f"<b>Zone {ZONE_LO}–{ZONE_HI} cm  ∫F·dz / Δz</b><br>"
        f"Norm. fluence rate: {zone_norm_mw:.4f} mW/cm²<br>"
        + "<br>".join(dose_lines)
        + f"<br><br><b>Cartilage (vol-weighted)</b><br>"
        + f"Fluence rate: {cartilage_flu_mw:.4f} mW/cm²<br>"
        + "<br>".join(cart_doses)
        + f"<br><br><b>Synovial Fluid</b><br>"
        + f"Fluence rate: {synovial_flu_mw:.4f} mW/cm²<br>"
        + "<br>".join(syn_doses)
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bin_centers, y=mean_flu, width=bin_width_cm * 0.85,
        marker=dict(color=mean_flu, colorscale='Hot', reversescale=True,
                    showscale=True,
                    colorbar=dict(title=dict(text='mW/cm²', side='right'),
                                  thickness=15, len=0.6)),
        name='Mean Fluence Rate',
    ))
    max_depth = float(bin_centers[-1]) + bin_width_cm / 2 if len(bin_centers) else 6.0
    for depth, label in DEPTH_REFS:
        if depth <= max_depth:
            fig.add_shape(type='line', x0=depth, x1=depth, y0=0, y1=1,
                          xref='x', yref='paper',
                          line=dict(color='rgba(100,200,255,0.55)', width=1, dash='dash'))
            fig.add_annotation(x=depth, y=1, xref='x', yref='paper',
                                text=label, showarrow=False,
                                font=dict(size=9, color='#8b949e'),
                                xanchor='left', yanchor='bottom', xshift=3)
    fig.add_annotation(x=0.98, y=0.98, xref='paper', yref='paper',
                        text=annot_text, showarrow=False, align='left',
                        xanchor='right', yanchor='top',
                        font=dict(size=10, color='#e6edf3'),
                        bgcolor='rgba(22,27,34,0.85)', bordercolor='#30363d',
                        borderwidth=1, borderpad=6)
    fig.update_layout(
        title=dict(text=f'Fluence Rate vs Penetration Depth — {subject_id} ({wavelength_nm} nm) Elbow',
                   font=dict(size=14)),
        xaxis=dict(title='Penetration Depth from Skin Surface (cm)',
                   gridcolor='#30363d', zeroline=False, dtick=0.25),
        yaxis=dict(title='Mean Fluence Rate (mW/cm²)', type='log',
                   gridcolor='#30363d', zeroline=False),
        paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
        font_color='#e6edf3',
        legend=dict(bgcolor='#161b22', bordercolor='#30363d', borderwidth=1),
        margin=dict(l=70, r=20, t=55, b=55), bargap=0.05,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Subject list ─────────────────────────────────────────────────────────
    # Populate once STL files are available.
    # Expected directory name format: Raw_Mesh_Files_ELB001, ELB002, …
    # Required STL files per subject (see TISSUE TABLE above):
    #   humerus_distal_raw.stl, radius_raw.stl, ulna_raw.stl,
    #   capitellum_cartilage_raw.stl, radial_head_cartilage_raw.stl,
    #   trochlear_cartilage_raw.stl, annular_lig_raw.stl
    SUBJECT_IDS = []   # e.g. ["ELB001", "ELB002"]

    BASE_DIR   = Path(".")
    RUN_ID     = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = Path(f"results_elbow_650nm_{RUN_ID}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Elbow MC Simulation — 650 nm")
    print(f"Subjects: {SUBJECT_IDS if SUBJECT_IDS else '(none configured — add to SUBJECT_IDS)'}")
    print(f"Output:   {OUTPUT_DIR}")

    if not SUBJECT_IDS:
        print("\n⚠  No subjects configured.  Add subject IDs to SUBJECT_IDS and place "
              "STL files in Raw_Mesh_Files_ELB### directories.")
        raise SystemExit(0)

    all_condition_results = {}
    for condition in MELANIN_CONDITIONS:
        print(f"\n{'=' * 60}\n  Melanin: {condition.upper()}\n{'=' * 60}")
        (OUTPUT_DIR / condition).mkdir(exist_ok=True)
        cond_results = []
        for subject_id in SUBJECT_IDS:
            result = run_subject(subject_id, BASE_DIR, OUTPUT_DIR,
                                 melanin_condition=condition)
            if result is not None:
                cond_results.append(result)
        all_condition_results[condition] = cond_results
        if cond_results:
            results_to_csv(
                cond_results,
                groups=GROUPS,
                dose_groups=DOSE_GROUPS,
                source_power_mw=SOURCE_POWER_MW,
                duty_cycle=SOURCE_DUTY_CYCLE,
                opt_eff=SOURCE_OPT_EFF,
                n_sources=3,
                output_path=str(OUTPUT_DIR / f"MC_Elbow_650nm_{condition}.csv"),
            )

    melanin_comparison_to_csv(
        all_condition_results,
        groups=COMP_GROUPS,
        output_path=str(OUTPUT_DIR / "MC_Elbow_Melanin_Comparison_650nm.csv"),
        wavelength_nm=WAVELENGTH_NM,
    )
    print(f"\nDone.")
