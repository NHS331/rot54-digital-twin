# Front-Side Solar Exposure Verification — 2026

## Constants

| Parameter | Value |
|---|---:|
| Latitude | 40.350000 deg |
| Reflector southward tilt | 15.000000 deg |
| Equivalent front-side normal declination | 25.350000 deg |
| Solar angular speed | 15.000000 deg/h |

## Manuscript control targets

| Quantity | Target |
|---|---:|
| Summer solstice exposure | 13.580000 h |
| Winter solstice exposure | 9.290000 h |
| Annual exposure | 4275.930000 h |

### Geometric horizon model

| Quantity | Value |
|---|---:|
| Apparent horizon altitude | 0.000000 deg |
| Equinox exposure, 2026-03-20 | 11.908558 h |
| Summer solstice exposure, 2026-06-22 | 13.581081 h |
| Winter solstice exposure, 2026-12-22 | 9.117542 h |
| Annual exposure, 2026 | 4233.940078 h |


### Standard-refraction horizon model

| Quantity | Value |
|---|---:|
| Apparent horizon altitude | -0.833000 deg |
| Equinox exposure, 2026-03-20 | 11.949007 h |
| Summer solstice exposure, 2026-06-22 | 13.581081 h |
| Winter solstice exposure, 2026-12-22 | 9.287670 h |
| Annual exposure, 2026 | 4261.600121 h |


### Back-fit horizon required to match the manuscript annual target

| Quantity | Value |
|---|---:|
| Apparent horizon altitude | -1.293799 deg |
| Equinox exposure, 2026-03-20 | 11.949007 h |
| Summer solstice exposure, 2026-06-22 | 13.581081 h |
| Winter solstice exposure, 2026-12-22 | 9.381161 h |
| Annual exposure, 2026 | 4275.930000 h |


## Diagnostic differences for the standard-refraction model

| Quantity | Difference |
|---|---:|
| Summer solstice minus manuscript target | 0.001081 h |
| Winter solstice minus manuscript target | -0.002330 h |
| Annual exposure minus manuscript target | -14.329879 h |

## Interpretation

The summer and winter control-day values are consistent with the manuscript targets under the standard-refraction horizon convention.

The annual value is more sensitive. In this analytical implementation, the standard-refraction model does not silently reproduce the manuscript annual target of 4275.93 h.

To reproduce the annual target exactly with the same analytical assumptions, the apparent horizon would need to be approximately -1.293799 deg.

This does not automatically prove that the manuscript value is wrong. It means that the annual value must be traced to the exact previous implementation: horizon convention, front-side condition, day sampling, and possible time-integration assumptions.

## Decision for the next stage

Do not use the annual value as an unquestioned constant.

Use this report as the control file before moving to panel-grid and ray-based shadowing.
