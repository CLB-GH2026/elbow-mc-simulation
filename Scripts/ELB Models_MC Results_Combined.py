"""
Elbow MC — Combined 650 nm + 808 nm (Move+ device model)
---------------------------------------------------------
Models the Move+ device with both wavelengths emitting SIMULTANEOUSLY from
co-located pads. The wavelengths are physically independent at PBM intensities
and MC transport is linear, so the tissue sees the SUM of the two fluence
fields. This builds the geometry once, runs pmcx at each wavelength (own optics
and source power, same co-located source positions), sums the fields, and
reports the combined per-tissue fluence / absorption / illumination-zone
coverage. The separate 808/650 scripts remain for single-wavelength studies.

Dependencies:
    pip install numpy trimesh pmcx plotly scipy scikit-image
    pip install git+https://github.com/CLB-GH2026/pbm-mc-core.git@v0.1.1
"""

import numpy as np
import time
from pathlib import Path
from datetime import datetime

from pbm_mc_core import (
    opt, EPIDERMIS_LABEL, build_melanin_conditions,
    build_label_volume,
    add_synovial_fluid, add_wrapping_layers, add_epidermis_layer,
    find_joint_line_z, find_surface_source_positions,
    run_pmcx,
    analyze_combined_absorption, analyze_penetration_depth, plot_depth_histogram,
    target_depth_zone, results_to_csv, melanin_comparison_to_csv,
)

start_time = time.perf_counter()

TARGET_MATCH_FN = lambda n: ('cart' in n) or ('annular' in n) or ('synovial' in n)

_MELANIN_RAW_808NM = {
    'fair':  (0.008, 1.50, 0.80, 1.40),
    'olive': (0.025, 1.60, 0.80, 1.40),
    'dark':  (0.075, 1.70, 0.80, 1.40),
}
_MELANIN_RAW_650NM = {
    'fair':  (0.020, 1.80, 0.80, 1.40),
    'olive': (0.070, 1.90, 0.80, 1.40),
    'dark':  (0.200, 2.00, 0.80, 1.40),
}

VOXEL_SIZE    = 1.0
GRID_DIMS_MM  = (120, 110, 200)
VOXEL_RES     = tuple(int(round(d / VOXEL_SIZE)) for d in GRID_DIMS_MM)
AUTO_ORIENT   = True

MUSCLE_THICK_MM  = 10   # brachioradialis / triceps at elbow
ADIPOSE_THICK_MM = 3
SKIN_THICK_MM    = 2
CONE_ANGLE_DEG   = 20

POWER_808 = dict(mw=50,  duty=0.75, eff=0.85)
POWER_650 = dict(mw=120, duty=0.75, eff=0.85)
N_SOURCES = 3
AVG_808   = POWER_808['mw'] * POWER_808['duty'] * POWER_808['eff']
AVG_650   = POWER_650['mw'] * POWER_650['duty'] * POWER_650['eff']
AVG_PAD   = AVG_808 + AVG_650
TOTAL_INPUT_MW = N_SOURCES * AVG_PAD

MELANIN_808 = build_melanin_conditions(_MELANIN_RAW_808NM, voxel_size_mm=VOXEL_SIZE)
MELANIN_650 = build_melanin_conditions(_MELANIN_RAW_650NM, voxel_size_mm=VOXEL_SIZE)

GROUPS = {
    'Bone':           lambda n: 'bone'     in n,
    'Cartilage':      lambda n: 'cart'     in n,
    'AnnularLig':     lambda n: 'annular'  in n,
    'Synovial':       lambda n: 'synovial' in n,
    'Muscle':         lambda n: 'muscle'   in n,
    'Adipose':        lambda n: 'adipose'  in n,
    'Skin+Epidermis': lambda n: ('skin' in n) or ('epidermis' in n),
}
DOSE_GROUPS = {
    'Cartilage':      lambda n: 'cart'     in n,
    'AnnularLig':     lambda n: 'annular'  in n,
    'Synovial Fluid': lambda n: 'synovial' in n,
    'Muscle':         lambda n: 'muscle'   in n,
}
COMP_GROUPS = {
    'Cartilage':      lambda n: 'cart'     in n,
    'AnnularLig':     lambda n: 'annular'  in n,
    'Synovial Fluid': lambda n: 'synovial' in n,
    'Muscle':         lambda n: 'muscle'   in n,
    'Bone':           lambda n: 'bone'     in n,
    'Skin+Epidermis': lambda n: 'skin' in n or 'epidermis' in n,
}


def _tissues(mesh_dir, wl, epidermis_opts):
    if wl == '808':
        t = {
            "synovial":        (None,                                      14, opt(0.0005, 0.01, 0.90, 1.36)),
            "skin":            (None,                                      13, opt(0.003,  1.22, 0.79, 1.40)),
            "adipose":         (None,                                      12, opt(0.0013, 1.00, 0.90, 1.44)),
            "muscle":          (None,                                      11, opt(0.0180, 0.55, 0.93, 1.37)),
            "annular-lig":     (mesh_dir / "annular_lig_raw.stl",           5, opt(0.006,  1.80, 0.90, 1.37)),
            "trochlear-cart":  (mesh_dir / "trochlear_cartilage_raw.stl",   9, opt(0.015,  1.00, 0.90, 1.37)),
            "radhead-cart":    (mesh_dir / "radial_head_cartilage_raw.stl", 8, opt(0.015,  1.00, 0.90, 1.37)),
            "capitellum-cart": (mesh_dir / "capitellum_cartilage_raw.stl",  7, opt(0.015,  1.00, 0.90, 1.37)),
            "ulna-bone":       (mesh_dir / "ulna_raw.stl",                  3, opt(0.040,  2.50, 0.92, 1.37)),
            "radius-bone":     (mesh_dir / "radius_raw.stl",                2, opt(0.040,  2.50, 0.92, 1.37)),
            "humerus-bone":    (mesh_dir / "humerus_distal_raw.stl",        1, opt(0.040,  2.50, 0.92, 1.37)),
        }
    else:  # 650
        t = {
            "synovial":        (None,                                      14, opt(0.0005, 0.01, 0.90, 1.36)),
            "skin":            (None,                                      13, opt(0.011,  1.50, 0.80, 1.40)),
            "adipose":         (None,                                      12, opt(0.003,  1.20, 0.90, 1.44)),
            "muscle":          (None,                                      11, opt(0.0280, 0.60, 0.93, 1.37)),
            "annular-lig":     (mesh_dir / "annular_lig_raw.stl",           5, opt(0.014,  2.00, 0.90, 1.37)),
            "trochlear-cart":  (mesh_dir / "trochlear_cartilage_raw.stl",   9, opt(0.025,  1.20, 0.90, 1.37)),
            "radhead-cart":    (mesh_dir / "radial_head_cartilage_raw.stl", 8, opt(0.025,  1.20, 0.90, 1.37)),
            "capitellum-cart": (mesh_dir / "capitellum_cartilage_raw.stl",  7, opt(0.025,  1.20, 0.90, 1.37)),
            "ulna-bone":       (mesh_dir / "ulna_raw.stl",                  3, opt(0.068,  2.80, 0.92, 1.37)),
            "radius-bone":     (mesh_dir / "radius_raw.stl",                2, opt(0.068,  2.80, 0.92, 1.37)),
            "humerus-bone":    (mesh_dir / "humerus_distal_raw.stl",        1, opt(0.068,  2.80, 0.92, 1.37)),
        }
    t["epidermis"] = (None, EPIDERMIS_LABEL, epidermis_opts)
    return t


def _default_src_configs(jl_z):
    return [
        {'name': 'Lateral',   'world_pos': [ 40,   0, jl_z], 'color': 'red'  },
        {'name': 'Posterior', 'world_pos': [  0, -35, jl_z], 'color': 'green'},
        {'name': 'Medial',    'world_pos': [-35,   0, jl_z], 'color': 'blue' },
    ]


def run_subject(subject_id, mesh_dir_base, output_dir, melanin_condition='fair'):
    mesh_dir = Path(mesh_dir_base) / f"Raw_Mesh_Files_{subject_id}"
    if not mesh_dir.exists():
        print(f"  Skipping {subject_id} — directory not found: {mesh_dir}")
        return None

    print(f"\n{'=' * 60}\n  Processing {subject_id}  [{melanin_condition}]  (combined 650+808)\n{'=' * 60}")

    tissues_808 = _tissues(mesh_dir, '808', MELANIN_808[melanin_condition])
    tissues_650 = _tissues(mesh_dir, '650', MELANIN_650[melanin_condition])

    try:
        vol, origin, mesh_center = build_label_volume(
            tissues_808, VOXEL_RES, VOXEL_SIZE, auto_orient=AUTO_ORIENT,
            orient_ref_a='humerus-bone', orient_ref_b='radius-bone',
        )
        bone_labels = [t[1] for n, t in tissues_808.items() if "bone" in n]
        cart_labels = [t[1] for n, t in tissues_808.items() if "cart" in n]
        ann_labels  = [t[1] for n, t in tissues_808.items() if "annular" in n]
        vol = add_synovial_fluid(vol, cartilage_labels=cart_labels + ann_labels,
                                 bone_labels=bone_labels,
                                 fluid_label=tissues_808["synovial"][1], dilation_vox=3)
        vol = add_wrapping_layers(vol, [
            (tissues_808["muscle"][1],  int(round(MUSCLE_THICK_MM / VOXEL_SIZE))),
            (tissues_808["adipose"][1], int(round(ADIPOSE_THICK_MM / VOXEL_SIZE))),
            (tissues_808["skin"][1],    int(round(SKIN_THICK_MM / VOXEL_SIZE))),
        ])
        vol = add_epidermis_layer(vol, skin_label=tissues_808["skin"][1],
                                  epidermis_label=EPIDERMIS_LABEL)

        jl_z = find_joint_line_z(vol, tissues_808, origin, VOXEL_SIZE, mesh_center,
                                 target_match_fn=TARGET_MATCH_FN)
        src_configs = _default_src_configs(jl_z)
        for cfg in src_configs:
            d = np.array([0, 0, jl_z]) - np.array(cfg['world_pos'])
            cfg['srcdir'] = (d / np.linalg.norm(d)).tolist()
        pmcx_source_plus = find_surface_source_positions(vol, origin, VOXEL_SIZE,
                                                         mesh_center, src_configs)
        pmcx_source = [{'srcpos': s['srcpos'], 'srcdir': s['srcdir']}
                       for s in pmcx_source_plus]

        print("\n--- 808 nm pass ---")
        flu_808, _ = run_pmcx(vol, tissues_808, pmcx_source, wavelength_m=808e-9,
                              source_power_mw=POWER_808['mw'], duty_cycle=POWER_808['duty'],
                              opt_eff=POWER_808['eff'], cone_angle_deg=CONE_ANGLE_DEG,
                              voxel_size_mm=VOXEL_SIZE)
        print("\n--- 650 nm pass ---")
        flu_650, _ = run_pmcx(vol, tissues_650, pmcx_source, wavelength_m=650e-9,
                              source_power_mw=POWER_650['mw'], duty_cycle=POWER_650['duty'],
                              opt_eff=POWER_650['eff'], cone_angle_deg=CONE_ANGLE_DEG,
                              voxel_size_mm=VOXEL_SIZE)

        results = analyze_combined_absorption(
            flu_808, flu_650, vol, tissues_808, tissues_650, VOXEL_SIZE,
            groups=GROUPS, total_input_mw=TOTAL_INPUT_MW,
            label_a='808nm', label_b='650nm')

        subj_dir = Path(output_dir) / melanin_condition / subject_id
        subj_dir.mkdir(parents=True, exist_ok=True)

        flu_comb = flu_808 + flu_650
        bin_centers, mean_flu, _ = analyze_penetration_depth(
            flu_comb, vol, VOXEL_SIZE, mesh_center, origin)
        z_lo, z_hi, z_med = target_depth_zone(vol, tissues_808, VOXEL_SIZE, TARGET_MATCH_FN)
        if z_lo is None:
            z_lo, z_hi, z_med = 1.0, 2.5, 1.75

        def grp_flu(match):
            names = [n for n in results if match(n)]
            vox = sum(results[n]['n_voxels'] for n in names)
            return (sum(results[n]['mean_flu'] * results[n]['n_voxels']
                        for n in names) / vox) if vox else 0.0

        fig = plot_depth_histogram(
            bin_centers, mean_flu, f"{subject_id} (combined 650+808)", "650+808",
            depth_refs=[(z_med, 'Cartilage/annular-lig/synovial (targets)')],
            zone_lo=z_lo, zone_hi=z_hi,
            group_flu_mw={'Cartilage': grp_flu(lambda n: 'cart' in n),
                          'Annular Ligament': grp_flu(lambda n: 'annular' in n),
                          'Synovial Fluid': grp_flu(lambda n: 'synovial' in n)},
        )
        fig.write_html(str(subj_dir / f"depth_histogram_{subject_id}_{melanin_condition}_combined.html"))

        np.save(subj_dir / "label_volume.npy", vol)
        np.save(subj_dir / "fluence_808.npy", flu_808)
        np.save(subj_dir / "fluence_650.npy", flu_650)
        np.save(subj_dir / "fluence_combined.npy", flu_comb)
        return subject_id, results

    except Exception as e:
        print(f"  ERROR processing {subject_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    SUBJECT_IDS = ["ELB001"]

    BASE_DIR   = Path(".")
    RUN_ID     = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = Path(f"results_elbow_combined_{RUN_ID}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Elbow MC Simulation — Combined 650 + 808 nm (Move+ device)")
    print(f"Per-pad avg power: {AVG_PAD:.1f} mW (808: {AVG_808:.1f} + 650: {AVG_650:.1f}); "
          f"total input {TOTAL_INPUT_MW:.1f} mW over {N_SOURCES} pads")
    print(f"Subjects: {SUBJECT_IDS}\nOutput: {OUTPUT_DIR}")

    if not SUBJECT_IDS:
        raise SystemExit(0)

    all_condition_results = {}
    for condition in MELANIN_808:
        print(f"\n{'=' * 60}\n  Melanin: {condition.upper()}\n{'=' * 60}")
        (OUTPUT_DIR / condition).mkdir(exist_ok=True)
        cond_results = []
        for subject_id in SUBJECT_IDS:
            r = run_subject(subject_id, BASE_DIR, OUTPUT_DIR, melanin_condition=condition)
            if r is not None:
                cond_results.append(r)
        all_condition_results[condition] = cond_results
        if cond_results:
            results_to_csv(
                cond_results, groups=GROUPS, dose_groups=DOSE_GROUPS,
                source_power_mw=AVG_PAD, duty_cycle=1.0, opt_eff=1.0, n_sources=N_SOURCES,
                total_power_mw_override=TOTAL_INPUT_MW,
                power_label='808nm(50mW)+650nm(120mW) co-located, 0.75 duty / 0.85 opt',
                output_path=str(OUTPUT_DIR / f"MC_Elbow_Combined_{condition}.csv"),
            )

    melanin_comparison_to_csv(
        all_condition_results, groups=COMP_GROUPS,
        output_path=str(OUTPUT_DIR / "MC_Elbow_Combined_Melanin_Comparison.csv"),
        wavelength_nm="650+808",
    )
    print(f"\nDone in {time.perf_counter() - start_time:.1f} s.")
