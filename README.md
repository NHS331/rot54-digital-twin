# ROT-54/2.6 Digital Scientific Twin

Python-first reproducible scientific and engineering digital twin of the ROT-54/2.6 radio-optical telescope.

## Core principle

Python is the single source of truth.

Downstream tools such as Blender, Godot, FreeCAD, Gmsh, CalculiX, and Code_Aster are consumers of validated exported data. They must not independently compute astronomy, geometry, solar vectors, shadowing, thermal loading, or deformation fields.

## Initial stage

Current stage:

- Stage 0: project skeleton
- Stage 1: SimulationClock
- Stage 2: Observer / Orgov site
- Stage 3: Sun AltAz calculation
- Stage 4: SolarGraph

## Current fixed parameters

- Site: Orgov / Aragats Scientific Center, Armenia
- Latitude: 40.3508609 deg N
- Longitude: 44.2417924 deg E
- Altitude: 1711 m
- Main reflector diameter: 54 m
- Effective aperture: 32 m
- Optical channel: 2.6 m
- Secondary mirror: 5 m
- Main axis tilt: 15 deg toward south
- Working computational panel count: 3738
- Passport surface smoothness: 70 micrometers
- Internal time standard: UTC