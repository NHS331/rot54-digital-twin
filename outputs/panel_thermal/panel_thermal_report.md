# ROT-54/2.6 Panel Thermal Nonuniformity Report

## Purpose

This step converts pointwise transient thermal snapshots into an equivalent panel-level temperature nonuniformity map.

## Worst panel-level snapshot

- Source snapshot: `transient_snapshot_spring_equinox_evening_v00.csv`
- Case: `spring_equinox`
- Snapshot code: `evening`
- Wind code: `v00`
- Local time: `2026-03-20T18:10:00+04:00`
- Maximum panel RMS temperature nonuniformity: `9.986754 °C`
- 95th percentile panel RMS temperature nonuniformity: `0.907874 °C`
- Maximum panel peak-to-peak temperature difference: `24.998614 °C`

## Validation

All panel thermal validation checks passed.

## Interpretation

The output `panel_delta_rms_C` is the direct thermal bridge to the later reduced mechanical response model. The next step will use this quantity to estimate normal panel displacement through the coefficient k_u.

This is still an equivalent panel discretization. It must later be replaced or refined by the real ROT-54/2.6 panel layout if exact CAD or survey panel coordinates become available.
