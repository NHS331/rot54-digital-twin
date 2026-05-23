# Stage 10 — Absorbed Solar Flux

## Purpose

Stage 10 converts the ray-based visibility function from Stage 9 into an absorbed solar flux field over the ROT-54/2.6 main reflector surface.

Stage 9 produced:

chi(r,t,N) = 1 for directly illuminated control points;

chi(r,t,N) = 0 for shadowed or non-front-side control points.

Stage 10 computes:

q_abs(r,t,N) = alpha_s * DNI(t) * mu_front(r,t,N) * chi(r,t,N)

where:

- alpha_s is the solar absorptivity of the reflector surface;
- DNI is the direct normal irradiance in W/m²;
- mu_front = cos(theta_i) is the local incidence factor;
- chi is the numerical visibility function;
- q_abs is the absorbed solar flux density in W/m².

## Important limitation

This stage does not yet compute temperature.

It computes only the absorbed solar flux density. Temperature requires a heat-balance model including convection, long-wave radiation, material thickness, heat capacity, and possibly transient thermal inertia.

## Inputs

- `outputs/shadow_ray/control_point_visibility.csv`
- `outputs/geometry/panel_grid_3738.csv`

## Outputs

- `outputs/solar_flux/control_point_absorbed_flux.csv`
- `outputs/solar_flux/panel_absorbed_flux_summary.csv`
- `outputs/solar_flux/scenario_absorbed_flux_summary.csv`
- `outputs/figures/solar_flux/*.png`

## Next stage

Stage 11 will convert absorbed flux into a first-order surface temperature increment:

Delta_T_s = q_abs / h_eff

or, in the transient version:

rho * c * dT/dt = q_abs - q_loss

The steady-state version should be implemented first because it is easier to validate.
