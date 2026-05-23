# Stage 7 — Internal Tripod Base Correction

## Correction applied

The tripod support legs are no longer modelled as radial structures extending to the outer rim of the 54 m reflector.

The corrected engineering model uses internal tripod bases located at:

r_base = 15.50 m

with the accepted working range:

15.0 m <= r_base <= 16.0 m

The outer rim radius remains:

R = 27.0 m

## Updated first-stage shadow model

| Parameter | Value |
|---|---:|
| support_arms.start_radius_m | 2.000 |
| support_arms.end_radius_m | 15.500 |
| support_arms.outer_rim_radius_m | 27.000 |

## Updated Shadow V2 model

| Parameter | Value |
|---|---:|
| support_arm_prisms.start_radius_m | 2.200 |
| support_arm_prisms.end_radius_m | 15.500 |
| support_arm_prisms.outer_rim_radius_m | 27.000 |

## Consequence

All previously generated shadow figures and downstream plots based on the old end_radius_m = 27.0 m must be treated as stale.

The following layers must be regenerated:

1. shadow maps;
2. Shadow V2 maps;
3. absorbed solar flux maps;
4. steady temperature maps;
5. thermal sensitivity maps;
6. transient thermal maps;
7. panel temperature non-uniformity;
8. panel displacement response;
9. surface RMS / efficiency tables.

## Remaining uncertainty

The base radius is corrected. The exact azimuths of the three bases are still treated as first-pass symmetric placeholders until measured/passport azimuths are entered.
