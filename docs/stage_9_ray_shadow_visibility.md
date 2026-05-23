# Stage 9 — Numerical Visibility Function chi(r,t,N)

## Purpose

This stage creates the first numerical ray-based visibility model for the ROT-54/2.6 solar calculation.

The purpose is to replace decorative shadow drawing with a computable visibility function:

chi(r,t,N) = 1 if the control point is directly illuminated;

chi(r,t,N) = 0 if the control point is shadowed by an internal structural element.

## Input

The input is the Stage 8 five-point panel grid:

- `outputs/geometry/panel_grid_3738.csv`
- `outputs/geometry/panel_control_points_3738.csv`

The control-point layer contains:

3738 panels × 5 control points = 18690 evaluated surface points.

## Structural correction

The tripod legs are modelled as internal supports. They do not extend to the outer rim.

The corrected support base radius is:

r_base = 15.5 m

with the allowed working range:

15 m <= r_base <= 16 m

The outer rim radius remains:

R = 27 m

This correction must propagate to all shadow maps, absorbed flux maps, temperature maps, panel response maps, and RMS outputs.

## Method

For each selected scenario and time:

1. Compute the solar declination.
2. Compute the active front-side hour angle.
3. Select morning, noon, and evening solar directions.
4. Transform the Sun vector to reflector-local coordinates.
5. For every control point, cast a ray toward the Sun.
6. Check intersection with:
   - central hub;
   - secondary mirror;
   - optical reflector;
   - internal tripod legs.
7. Write chi = 0 for shadowed points and chi = 1 for illuminated points.

## Current model status

This is a first numerical visibility layer. It is not yet the final measured structural model.

The exact tripod azimuths remain placeholders until measured/passport geometry is inserted.

## Generated outputs

- `outputs/shadow_ray/control_point_visibility.csv`
- `outputs/shadow_ray/panel_shadow_summary.csv`
- `outputs/shadow_ray/scenario_shadow_summary.csv`
- `outputs/figures/shadow_ray/*.png`

## Next stage

Stage 10 will convert visibility into absorbed solar flux:

q_abs = alpha_s * I_sun * cos(theta_i) * chi

## Tripod azimuth convention

The reflector-local aperture-plane convention is:

- 0 deg = +x = east;
- 90 deg = +y = north;
- 180 deg = west;
- 270 deg = south.

The tripod geometry therefore uses:

angles_deg = [270.0, 30.0, 150.0]

This gives one support leg strictly along the south direction and two remaining legs separated by 120 degrees from it.

The earlier [90.0, 210.0, 330.0] set is invalid for this coordinate convention because it places one support leg strictly northward.


## Tripod azimuth correction

The reflector-local aperture-plane convention is:

- 0 deg = +x = east;
- 90 deg = +y = north;
- 180 deg = west;
- 270 deg = south.

The tripod geometry therefore uses:

angles_deg = [270.0, 30.0, 150.0]

This gives one support leg strictly along the south direction and two remaining legs separated by 120 degrees from it.

The earlier [90.0, 210.0, 330.0] set is invalid for this coordinate convention because it places one support leg strictly northward.

## Secondary mirror central-shadow correction

For finite vertical cylinders, the ray-intersection test must check the full cylinder volume, not only side-wall crossing.

This is critical near solar noon, where the Sun direction may be almost axial in reflector-local coordinates. In that case, dx and dy can be close to zero, while dz is positive. A ray starting below the 5 m secondary mirror footprint must be classified as shadowed.

