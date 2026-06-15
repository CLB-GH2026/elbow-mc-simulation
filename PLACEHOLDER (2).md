# ELB002 — Awaiting STL Files

Place the following STL files in this directory to run the simulation:

- `humerus_distal_raw.stl`
- `radius_raw.stl`
- `ulna_raw.stl`
- `capitellum_cartilage_raw.stl`
- `radial_head_cartilage_raw.stl`
- `trochlear_cartilage_raw.stl`
- `annular_lig_raw.stl`

## Sourcing

See the repository CLAUDE.md for recommended sources (BodyParts3D, TotalSegmentator, SpineWeb, SimTK).

## Coordinate Convention

All meshes must share a common coordinate system:
- **+Z = superior** (cranial)
- **+Y = anterior** (ventral)
- **+X = lateral** (right side of body)

The pipeline auto-corrects Z-axis inversion via `AUTO_ORIENT = True`.
Meshes from BodyParts3D are already in this convention.
