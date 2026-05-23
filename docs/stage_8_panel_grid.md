# Stage 8 — Panel Grid and Five-Point Control Sampling

## Purpose

This stage introduces the panel-discrete computational surface required for the scientific version of the solar model.

The purpose is to stop treating the reflector as only a plotted continuous disk and to create a reproducible panel-level calculation layer.

## Generated objects

The stage generates:

1. `outputs/geometry/panel_grid_3738.csv`
2. `outputs/geometry/panel_control_points_3738.csv`
3. `outputs/geometry/panel_grid_summary.csv`
4. `outputs/figures/geometry/panel_grid_tripod_bases.png`

## Panel grid

The projected reflector aperture is represented by 3738 calculation cells.

The projected aperture area is:

S_ap = pi * 27^2

The projected cell area is:

S_cell = S_ap / 3738

The equivalent projected cell size is:

L_eq = sqrt(S_cell)

These values are used because the RMS budget is later calculated over the projected aperture surface.

## Five control points

Each panel receives five control points:

1. center;
2. +x projected offset;
3. -x projected offset;
4. +y projected offset;
5. -y projected offset.

This is a computational approximation of the four-corner-plus-center idea used in the manuscript.

The exact geometric panel boundaries are not reconstructed at this stage. The control-point layer is a reproducible numerical approximation that will later be checked by convergence tests.

## Tripod correction

The figure and configuration show the corrected support-base radius:

r_base = 15.5 m

The tripod bases are internal and must not be placed at the 27 m outer rim.

## Next stage

Stage 9 will use these control points for the numerical visibility function chi(r,t,N):

solar vector
→ control point
→ ray toward Sun
→ intersection with internal structures
→ illuminated or shadowed state.
