# Anatomy template map

Each module in this folder owns one editable anatomical template or motion
model. `drosophila_vpython.py` assembles these pieces but should not be the place
to tune their proportions.

| File | Edit this to change |
|---|---|
| `leg_geometry.py` | Hip locations, foot reach, knee/ankle geometry, stride and tripod phase |
| `legs.py` | Leg segment radii, colours, muscle/tendon/ligament appearance |
| `wings.py` | Wing outline dimensions, shoulder placement, span, sweep and opacity |
| `wing_motion.py` | Wing oscillator, stroke amplitude, activation-to-rate mapping and mirrored stroke geometry |
| `head.py` | Head size, position and colour |
| `thorax.py` | Thorax dimensions and individual flight-muscle fibre paths |
| `abdomen.py` | Every abdomen segment's size and position, independently |
| `eyes.py` | Compound-eye size and ommatidia patch spacing |
| `antennae.py` | Antenna base, shaft, club and movement dimensions |
| `halteres.py` | Haltere stalk and knob geometry |
| `nervous_system.py` | Display-only ganglion, nerve-cord and branch dimensions |

## Side convention

The body-local coordinate system is `+X = forward`, `+Y = up`, `-Z =
anatomical left`, and `+Z = anatomical right`. The camera currently views from
approximately `+Z`; consequently the near-side (right) appendages may appear
on the opposite visual side of the screen from the anatomical label. The label
is relative to the fly, not to the viewer.

**Do not make left and right hip offsets negative and then multiply them by a
side sign.** Template offsets are stored as positive magnitudes and multiplied
by `side` exactly once. `tests/test_geometry.py` checks that all leg joints stay
on their assigned side and that corresponding left/right geometry is mirrored.

## Swapping a leg template

```python
from templates.leg_geometry import LegGeometryTemplate
from templates.legs import LegAssembly, LegVisualTemplate

my_leg_geometry = LegGeometryTemplate(
    front_attach_z=0.27,
    foot_base_z=0.88,
    stride=0.30,
)
my_leg_visual = LegVisualTemplate(femur_radius=0.082)
leg = LegAssembly("front_left", 0, -1, my_leg_geometry, my_leg_visual)
```

## Swapping a wing shape or motion model

Edit `WingTemplate` in `wings.py` or create another instance and pass it to
`WingAssembly`. The kinematics are independent: replace `WingMotionModel` in
`wing_motion.py` with another object implementing `step(dt, activation)` and
`wing_frame(side, activation, span, sweep_x)`.

The default wing oscillator is only a visual placeholder. Its rates are in
radians/second and are not a measured biological wingbeat frequency.
