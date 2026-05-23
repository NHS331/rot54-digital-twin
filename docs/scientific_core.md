# Scientific Core of the ROT-54/2.6 Solar Model

## Working scientific core

This project is not a solar-trajectory plotting package. The target scientific model is a visibility-driven, panel-discrete thermomechanical RMS model for solar-induced surface error in a fixed spherical panel reflector.

The computational chain is:

solar geometry
→ 3D reflector geometry
→ panel discretization
→ solar-vector projection
→ ray-based self-shadowing
→ absorbed solar flux
→ surface temperature anomaly
→ panel-level thermal non-uniformity
→ thermomechanical RMS contribution
→ total RMS budget
→ optional frequency-domain interpretation.

## Main hypothesis

For a fixed-axis spherical panel reflector with permanent internal self-shadowing, the dominant solar-induced operational surface-error contribution is governed by panel-scale temperature non-uniformity, not only by the absolute mean temperature of the reflector.

## Scientific contribution

The model must provide:

1. A reproducible front-side solar exposure calculation.
2. A numerical visibility function chi(r,t,N).
3. A panel-discrete representation of the ROT-54/2.6 reflector.
4. A five-point intra-panel sampling model.
5. A bounded thermomechanical conversion from temperature non-uniformity to normal displacement.
6. A reproducible RMS budget.
7. Sensitivity, convergence, and ablation checks.

## Current development priority

The immediate priority is to move from SolarGraph-level geometry to numerical solar statistics:

annual front-side exposure
→ panel grid
→ five control points per panel
→ numerical shadow function
→ shadow statistics per panel.
