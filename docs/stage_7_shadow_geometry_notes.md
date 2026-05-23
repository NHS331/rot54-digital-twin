# Stage 7 Geometry Note — Tripod Support Bases

## Critical correction

The tripod support bases must not be placed at the outer rim of the 54 m main reflector.

The working geometric correction is:

- the bases are located inside the reflector area;
- each base is approximately 15–16 m from the center;
- the current modelling value is 15.5 m from the center;
- the exact azimuths must be confirmed from drawings, photographs, or passport geometry before the final ray-shadowing implementation.

## Why this matters

If the tripod bases are placed at the outer edge of the main reflector, the shadowing model becomes physically wrong:

1. support-leg shadows become too long or incorrectly positioned;
2. shadow origins are displaced outward;
3. the calculated shadow fraction per panel becomes biased;
4. later thermal maps and RMS maps inherit this geometric error.

## Current modelling decision

For the next numerical stage, the tripod bases are treated as internal support bases at radius:

r_base = 15.5 m

with an allowed working range:

15.0 m <= r_base <= 16.0 m

The initial azimuth values in the configuration file are placeholders for a symmetric first-pass model. They are not final measured geometry.

## Use in the next stage

The ray-based self-shadowing module must use these internal base points as the lower/support-side anchor points of the tripod legs.

The shadowing chain should be:

solar vector
→ candidate surface/control point
→ ray toward the Sun
→ intersection test with secondary reflector, central hub, tripod legs, and support bases
→ visibility chi(r,t,N).
