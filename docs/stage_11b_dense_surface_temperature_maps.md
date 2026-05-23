# Stage 11b — Dense Interpolated Surface Temperature Maps

## Purpose

Stage 11 computed surface temperature increments at 18,690 control points.

That is physically useful, but visually it produces point-like maps.

Stage 11b creates dense interpolated heatmaps on a regular aperture grid.

The default grid is:

1600 x 1600

Inside the circular 27 m aperture this produces approximately 2,000,000 valid coordinate points per scenario/time case.

This is more than 100 times denser than the original 18,690 control points.

## Important distinction

This stage does not add new physics.

It creates a dense visualization and post-processing layer from the validated Stage 11 point field.

The physical calculation remains traceable to:

chi → q_abs → Delta_T_s

## Inputs

- `outputs/surface_temperature/control_point_surface_temperature.csv`

## Outputs

- `outputs/surface_temperature_dense/dense_temperature_grid_summary.csv`
- `outputs/surface_temperature_dense/dense_temperature_*.npz`
- `outputs/figures/surface_temperature_dense/dense_temperature_delta_*.png`

## Why NPZ instead of CSV

A dense CSV with about 2 million rows per case and 9 cases would become unnecessarily large and slow.

The NPZ files store the dense coordinate arrays directly:

- `x_m`
- `y_m`
- `delta_t_s_C`
- `surface_temperature_C`

This is the correct format for numerical post-processing.

## Next stage

After the dense map layer is accepted, Stage 12 can use the panel-level temperature summaries for thermal deformation.
