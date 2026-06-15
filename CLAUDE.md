# CLAUDE.md — Elbow PBM Monte Carlo Simulation

This file provides guidance to Claude Code when working with this repository.
Mirrors the structure of the knee `knee-mc-simulation` repo.

## Active Scripts

| File | Purpose |
|---|---|
| `ELB Models_MC Results_808nm.py` | Batch pipeline — all subjects at 808 nm |
| `ELB Models_MC Results_650nm.py` | Batch pipeline — all subjects at 650 nm |

## Project Overview

Monte Carlo photon transport simulation of PBM laser delivery to the elbow joint.
Primary clinical targets: lateral epicondylitis (tennis elbow) — shallow, ~1–2 cm;
radiocapitellar joint space — ~2–3 cm via lateral approach.

The elbow has the **shallowest target depth** of all four Kineon joints.
Tissue layers lateral to the joint are thin (skin + adipose ~5 mm, extensor
muscle origin ~10 mm), making cartilage fluence substantially higher than
knee or shoulder for the same source power.

## Key Configuration Constants

```python
VOXEL_SIZE       = 1.0
GRID_DIMS_MM     = (120, 110, 200)   # smallest grid — elbow is compact
AUTO_ORIENT      = True              # humerus distal above radius check
MUSCLE_THICK_MM  = 10   # extensor/flexor origin — thin laterally
ADIPOSE_THICK_MM =  3
SKIN_THICK_MM    =  2
SOURCE_POWER_MW  = 50   # 808nm  (120 for 650nm)
```

## Coordinate System

- **+Z = superior** (humerus distal above radius/ulna)
- **+Y = anterior** (volar / cubital fossa)
- **+X = lateral** (radial side)

## Tissue Labels

| Label | Tissue |
|---|---|
| 1 | Humerus distal (capitellum + trochlea) |
| 2 | Radius (radial head + shaft) |
| 3 | Ulna (olecranon + proximal) |
| 5 | Annular ligament (fibrocartilage) |
| 7 | Capitellum articular cartilage |
| 8 | Radial head articular cartilage |
| 9 | Trochlear articular cartilage |
| 11 | Muscle |
| 12 | Adipose |
| 13 | Skin |
| 14 | Synovial fluid (synthesised) |
| 15 | Epidermis |

## Default Source Placement

Three sources targeting the radiocapitellar joint from lateral and posterior:
- Lateral   (+40, 0, jl_z)
- Posterior (0, −35, jl_z)
- Medial    (−35, 0, jl_z)

For lateral epicondylitis, set `OPTIMIZE_SOURCES = True` or manually
bias sources toward (+45, +10, jl_z).

## Required STL Files Per Subject

Place in `Scripts/Raw_Mesh_Files_ELB###/`:
```
humerus_distal_raw.stl
radius_raw.stl
ulna_raw.stl
capitellum_cartilage_raw.stl
radial_head_cartilage_raw.stl
trochlear_cartilage_raw.stl
annular_lig_raw.stl
```

## Recommended STL Sources

| Tissue | Source | Notes |
|---|---|---|
| Humerus, radius, ulna | [BodyParts3D GitHub](https://github.com/Kevin-Mattheus-Moerman/BodyParts3D) | CC-BY-SA |
| Cartilage, annular lig | 3D Slicer from MRI | Elbow MRI available on OpenNeuro |
| All tissues (auto) | TotalSegmentator on elbow CT | May need `--fast` flag for small FOV |

## Key Differences from Knee Pipeline

- `GRID_DIMS_MM` smallest of all joints (120×110×200 mm)
- `MUSCLE_THICK_MM = 10` — thinnest muscle covering of all targets
- Orientation uses `humerus-bone` vs `radius-bone` centroids
- Annular ligament replaces meniscus (fibrocartilage class, same optical props)
- Depth histogram zone: skin ~0.5 cm, muscle ~1 cm, joint ~2 cm
- Capitellum cartilage is the primary fluence reporting tissue (not patella/meniscus)
