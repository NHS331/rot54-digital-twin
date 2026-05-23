# Model Validation Plan

## Purpose

This document defines the validation logic for the ROT-54/2.6 solar-thermomechanical model.

The purpose is not experimental validation at this stage. The purpose is numerical reproducibility, internal consistency, parameter traceability, and model sensitivity control.

## Required validation layers

### 1. Astronomical validation

The code must reproduce the front-side solar exposure values used in the article:

- Summer solstice: approximately 13.58 h
- Winter solstice: approximately 9.29 h
- Annual front-side exposure for 2026: approximately 4275.93 h

These values are treated as control targets. If a future implementation produces significantly different values, the discrepancy must be explained by a documented change in assumptions.

### 2. Geometry validation

The code must preserve the fixed ROT-54/2.6 parameters:

- Main reflector diameter: 54 m
- Main reflector radius: 27 m
- Effective aperture: 32 m
- Optical channel: 2.6 m
- Secondary mirror diameter: 5 m
- Axis tilt: 15 degrees southward from local zenith
- Panel count: 3738
- Passport surface smoothness: 70 micrometres

### 3. Panel-discretization validation

The panel model must provide:

- 3738 panel elements
- projected cell area
- equivalent cell size
- five control points per panel
- panel-level illumination statistics
- panel-level shadow statistics

The five-point panel model must later be checked against denser local sampling.

### 4. Shadow-function validation

The visibility function chi(r,t,N) must be numerical rather than illustrative.

For every evaluated control point, the model must determine:

- whether the point is illuminated;
- whether the point is shadowed;
- the shadowing source if available;
- the local solar incidence factor.

### 5. Thermomechanical validation

The model must show that k_u is a bounded sensitivity coefficient, not a fitted constant.

The required control values are:

- central interval: 0.080–0.096 mm/degC
- upper estimate: 0.160 mm/degC

The coefficient must be derived from panel-scale thermal bending assumptions and then used as a sensitivity parameter.

### 6. RMS validation

The model must reproduce the article-level RMS target values within a declared tolerance after the thermal and mechanical modules are implemented.

The RMS chain must include:

temperature non-uniformity
→ panel displacement response
→ added solar-induced RMS
→ total RMS budget.

### 7. Sensitivity, convergence, and ablation

The mature codebase must include:

- sensitivity to wind speed;
- sensitivity to k_u;
- sensitivity to optical and thermal surface parameters;
- convergence of panel sampling;
- ablation of self-shadowing, wind cooling, and incidence-angle projection.

## Immediate validation priority

The next technical goal is not thermal modeling yet. The next goal is verified solar statistics:

annual front-side exposure
→ panel grid
→ five control points per panel
→ numerical visibility function
→ shadow statistics per panel.
