# ROT-54/2.6 Steady Thermal Wind Sensitivity Report

## Purpose

This report checks whether the first steady-state thermal maps respond physically to wind-speed changes before moving to transient thermal modeling.

## Worst steady thermal case

- Case: `winter_solstice`
- Time code: `morning`
- Local time: `2026-12-22 09:24:00 Asia/Yerevan`
- Wind speed: `0.00 m/s`
- Air temperature: `-5.00 °C`
- Sky temperature: `-18.00 °C`
- Maximum absorbed flux: `237.765 W/m²`
- Maximum surface temperature: `28.721 °C`
- Maximum ΔT relative to air: `33.721 °C`

## Wind monotonicity checks

All wind-sensitivity monotonicity checks passed.

## Interpretation

The key expected behavior is that increasing wind speed increases the convective heat-transfer coefficient and does not increase the maximum surface temperature or maximum ΔT relative to air for the same solar case.

The full-aperture mean temperature is not used as a strict monotonicity criterion because shaded zones can be radiatively cooled below air temperature and then pulled back toward air temperature by stronger convection.
