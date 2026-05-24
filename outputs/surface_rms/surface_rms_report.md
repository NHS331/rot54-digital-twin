# ROT-54/2.6 Surface RMS and Ruze Efficiency Report

## Purpose

This step aggregates panel-level normal thermomechanical response into aperture-level additional RMS surface error and combines it with the baseline surface RMS.

## Model

- Additional thermally induced RMS: `σ_T = RMS(u_n over panels)`
- Total RMS: `σ_Σ = sqrt(σ_0^2 + σ_T^2)`
- Baseline RMS: `σ_0 = 0.070000 mm`
- Ruze efficiency: `η_R = exp[-(4πσ/λ)^2]`

## Worst total-RMS case

- Source: `panel_response_winter_solstice_peak_v00.csv`
- Case: `winter_solstice`
- Snapshot: `peak`
- Wind: `v00`
- Local time: ``
- `σ_T upper`: `0.145711 mm`
- `σ_Σ upper`: `0.161653 mm`
- `f10 upper total`: `47.903 GHz`
- `η_R total upper at 4.5 GHz`: `0.999071`
- `η_R total upper at 30 GHz`: `0.959519`
- `η_R total upper at 100 GHz`: `0.631826`

## Validation

All surface RMS validation checks passed.

## Interpretation

The primary aggregation uses panel grid-point count as an approximate area weight. Unweighted RMS values are also saved for comparison.

The next step can generate final article-style tables for selected seasonal scenarios and wind speeds.
