# ROT-54/2.6 Panel Thermal Nonuniformity Report

## Purpose

This step converts pointwise transient thermal snapshots into an equivalent panel-level temperature nonuniformity map.

## Worst panel-level snapshot

- Source snapshot: `transient_snapshot_winter_solstice_evening_v00.csv`
- Case: `winter_solstice`
- Snapshot code: `evening`
- Wind code: `v00`
- Local time: `2026-12-22T16:40:00+04:00`
- Maximum panel RMS temperature nonuniformity: `9.084490 °C`
- 95th percentile panel RMS temperature nonuniformity: `1.079769 °C`
- Maximum panel peak-to-peak temperature difference: `25.048111 °C`

## Validation

All panel thermal validation checks passed.

## Interpretation

The output `panel_delta_rms_C` is the direct thermal bridge to the later reduced mechanical response model. The next step will use this quantity to estimate normal panel displacement through the coefficient k_u.

This is still an equivalent panel discretization. It must later be replaced or refined by the real ROT-54/2.6 panel layout if exact CAD or survey panel coordinates become available.
