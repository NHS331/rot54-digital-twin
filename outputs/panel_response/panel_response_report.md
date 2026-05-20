# ROT-54/2.6 Panel Normal Thermomechanical Response Report

## Purpose

This step converts equivalent panel-level thermal nonuniformity into normal thermomechanical response using the reduced relation `u_n = k_u · ΔT_panel`.

## Worst response snapshot

- Source: `panel_thermal_spring_equinox_evening_v00.csv`
- Case: `spring_equinox`
- Snapshot: `evening`
- Wind: `v00`
- Maximum upper-bound panel RMS response: `1.597881 mm`
- 95th percentile upper-bound panel RMS response: `0.145260 mm`
- Maximum central-max panel RMS response: `0.958728 mm`
- Maximum peak-to-peak upper response: `3.999778 mm`

## Worst panel

- Panel ID: `P_0066_0022`
- x: `25.070 m`
- y: `-9.383 m`
- Panel ΔT RMS: `9.986754 °C`
- Upper RMS response: `1.597881 mm`

## Validation

All panel response validation checks passed.

## Interpretation

The response values are magnitude estimates. They are suitable for RMS-budget calculations but are not yet a signed deformation field.

The next step will aggregate panel response over the aperture to obtain `σ_T`, then combine it with the baseline surface RMS `σ_0 = 0.070 mm`.
