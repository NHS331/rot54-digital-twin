# Stage 6 — Front-Side Solar Exposure Verification

## Purpose

This stage verifies the annual front-side solar exposure calculation for the ROT-54/2.6 main reflector.

The goal is to separate three quantities that must not be confused:

1. daylight duration above the local apparent horizon;
2. front-side visibility of the fixed reflector axis;
3. effective front-side exposure limited by both conditions.

The calculation is intentionally analytical and lightweight. It is used as a control layer before moving to panel-level shadowing.

## Geometry used

The reflector axis is tilted by 15 degrees toward the south from the local zenith.

For the Orgov / Aragats site:

- latitude: 40.35 deg
- southward reflector-axis tilt: 15.00 deg
- equivalent front-side normal declination: 25.35 deg

The model uses the standard engineering solar-declination approximation:

delta_sun(N) = 23.45 deg * sin(360 deg * (284 + N) / 365)

## Conditions

The Sun contributes to front-side exposure only if both conditions are true:

1. the Sun is above the apparent horizon;
2. the solar vector lies on the front side of the reflector.

The active hour angle is the smaller of:

- the horizon-limited half-day angle;
- the front-side-normal half-angle.

## Important diagnostic

The manuscript target values are:

- summer solstice: about 13.58 h;
- winter solstice: about 9.29 h;
- annual front-side exposure: about 4275.93 h.

The first two values are expected to be consistent with a standard refraction horizon near -0.833 deg.

The annual value must be rechecked carefully, because it is sensitive to the apparent-horizon convention and to the exact implementation of the front-side condition.

This stage therefore reports both:

- the standard-refraction result;
- the apparent horizon that would be required to reproduce the manuscript annual value.

The code must not silently force the annual target. Any mismatch must be documented.
