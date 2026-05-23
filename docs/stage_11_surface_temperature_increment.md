# Stage 11 — First-Order Surface Temperature Increment

## Purpose

Stage 11 converts absorbed solar flux into a first-order steady-state surface temperature increment.

The input from Stage 10 is:

q_abs(r,t,N)

The computed output is:

Delta_T_s(r,t,N) = q_abs(r,t,N) / h_eff(t)

and:

T_surface_C(r,t,N) = T_ambient_C(t) + Delta_T_s(r,t,N)

## Physical meaning

This stage is not yet a full transient thermal model.

It is a first engineering thermal layer that converts spatially non-uniform absorbed solar flux into spatially non-uniform temperature increments.

## Heat-transfer model

The current first-order model uses:

h_eff = h_conv + h_rad

where:

h_conv = 5.7 + 3.8 * v

and:

h_rad = 4 * epsilon_lw * sigma * T_ambient_K^3

## Important limitation

The model is steady-state and local.

It does not yet include:

- thermal inertia;
- panel thickness;
- conduction between neighbouring panels;
- transient heating/cooling;
- rear-side radiation;
- measured coating-specific emissivity;
- measured weather input.

Those belong to later stages.

## Inputs

- `outputs/solar_flux/control_point_absorbed_flux.csv`
- `outputs/geometry/panel_grid_3738.csv`

## Outputs

- `outputs/surface_temperature/control_point_surface_temperature.csv`
- `outputs/surface_temperature/panel_temperature_summary.csv`
- `outputs/surface_temperature/scenario_temperature_summary.csv`
- `outputs/figures/surface_temperature/*.png`

## Next stage

Stage 12 will convert surface temperature increment into panel-level thermal deformation.
