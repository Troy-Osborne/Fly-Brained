# Drosophila Embodied Body — VPython Prototype

A runnable visual sensorimotor sandbox for embodied-cognition experiments. It includes:

- A stylised articulated fly body: head, thorax, segmented abdomen, compound-eye patches, antennae, wings, halteres, and six three-segment legs.
- Visible actuator and tissue proxies: leg flexor/extensor muscle bellies, thoracic flight-muscle fibres, tendon links, passive compliant joint restraints, and bristle-like mechanosensors.
- Stylised nervous anatomy: brain ganglion, ventral nerve cord, segmental ganglia, leg motor branches, and eye/antenna sensory afferents.
- A simple ground plane, obstacle, airflow field, wind particles, and a moving second agent.
- Sensor channels for proprioception, contact/tactile events, airflow, pressure, sound, visual-stream features, and internal-state variables.
- A swappable controller contract and per-channel expected-versus-actual prediction errors.
- Separate, replaceable anatomical templates for head, thorax, abdomen, eyes, antennae, halteres, legs, wings, wing kinematics and neural display.
- Optional webcam and multichannel microphone input, plus JSONL telemetry recording.

## Install

Python 3.10–3.12 is recommended. In a virtual environment (Windows PowerShell shown):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Then install the required packages:

```bash
python -m pip install -r requirements.txt
```

Launch:

```bash
python drosophila_vpython.py
```

Optional live inputs:

```bash
python -m pip install opencv-python sounddevice
python drosophila_vpython.py --camera --microphone
```

Record a run:

```bash
python drosophila_vpython.py --record experiment_001.jsonl
```

## Controller interface

Implement the `Controller` protocol in `drosophila_vpython.py`:

```python
class MyController:
    def step(self, sensory, prediction_errors, dt):
        return ControlPacket(
            muscle_activations={
                "front_left": 0.4,
                "front_right": 0.4,
                "middle_left": 0.5,
                "middle_right": 0.5,
                "hind_left": 0.7,
                "hind_right": 0.7,
            },
            forward_drive=0.5,
            steering=0.0,
            wing_activation=0.0,
            antennae_activation=0.5,
            expected_sensory={
                "proprio.forward_speed": 0.3,
                "airflow.speed": 1.2,
                "pressure.pa": 101325.0,
            },
            metadata={"controller": "MyController"},
        )
```

Then install it with `simulation.install_controller(MyController())` after constructing `FlySimulation`.

External streams can be injected with:

```python
simulation.push_audio(audio_frames, sample_rate=48000)  # shape: (frames, channels)
simulation.push_visual_frame(image_array)                # HxW or HxWxC
```

The controller receives scalar / derived channels through the mapping interface,
and can access the latest raw buffers without forcing them into the telemetry JSON:

```python
audio_window = sensory.streams["audio"]       # NumPy array: (frames, channels)
visual_frame = sensory.streams["visual_frame"] # full-resolution input frame, or None
```

## Sensory keys

Examples include `proprio.forward_speed`, `proprio.joint_angle_mean`, `airflow.speed`, `airflow.lateral`, `pressure.pa`, `tactile.contact_count`, `tactile.force_proxy`, `audio.ch0.rms`, `audio.ch1.rms`, `vision.mean_luma`, `vision.motion`, `intero.hunger`, `intero.glucose_proxy`, `intero.octopamine_proxy`, `intero.nociception`, `intero.temperature_c`, `intero.hydration`, and `intero.fatigue`.

`prediction_errors[key]` is **actual minus expected** for channels for which the controller supplied a prediction on the prior update. The aggregate telemetry RMS is dimensionless and uses documented characteristic scales; raw per-channel deltas remain available and are passed to the controller.

Custom scalar internal-state variables can be added without modifying the sensor classes:

```python
simulation.sensors.set_custom_variable("hydration_drive", 0.35)
simulation.sensors.set_custom_variable("reproductive_state", 0.10)
```

They appear as `intero.custom.hydration_drive` and `intero.custom.reproductive_state`.

## Scope and scientific limitations

This is a first-pass VPython prototype, not a validated fruit-fly biomechanical model. Its leg motion and body translation are kinematic / simplified-dynamics approximations; the visible nerve pathways are anatomical placeholders, and the muscle, tendon, passive-restraint, nociception and internal-state models are proxies for interface and experiment development. The pressure and airflow values are synthetic environment measurements. Audio and image inputs are made available as raw buffers and also summarised into simple features; they are not yet transformed into biological receptor or connectome inputs.

It does **not** currently load the Male CNS connectome, solve detailed aerodynamic fluid dynamics, implement a biologically complete muscle model, or claim subjective pain / human-like emotion. The controller boundary is deliberately explicit so the connectome adapter and more rigorous physics can be added without rewriting the visual body.

## Next technical extensions

1. Replace the current analytic limb trajectory with joint-space dynamics and contact constraints.
2. Implement calibrated muscle-tendon force-length / force-velocity models.
3. Add mechanosensory transduction from local deformation and force.
4. Add receptor-specific visual, auditory, olfactory and gustatory encoding.
5. Add a Male CNS v1.0 connectome import + neuron/synapse simulation adapter.
6. Validate morphology and motor mappings against the relevant male-fly anatomy and datasets.

## Anatomical template modules (v0.2)

The fly is now assembled from independent templates under `templates/`. This
keeps anatomy changes out of the sensor/controller/physics loop:

| Module | What it controls |
|---|---|
| `templates/head.py` | Head dimensions, position and colour |
| `templates/thorax.py` | Thorax dimensions and each flight-muscle path |
| `templates/abdomen.py` | Each abdomen segment's size, position and colour |
| `templates/legs.py` | VPython leg assembly and visible muscle/tendon/ligament parts |
| `templates/leg_geometry.py` | Hip, knee, ankle and foot locations, stride, lift and gait phase |
| `templates/leg_visuals.py` | Leg segment radii and tissue colours |
| `templates/wings.py` | Wing mesh/ellipsoid assembly |
| `templates/wing_geometry.py` | Wing size, chord, span, shoulder placement, sweep and appearance |
| `templates/wing_motion.py` | Replaceable wing oscillator and mirrored stroke geometry |
| `templates/eyes.py` | Compound-eye and ommatidia patch dimensions |
| `templates/antennae.py` | Antenna base, shaft and club dimensions |
| `templates/halteres.py` | Haltere stalk and knob dimensions |
| `templates/nervous_system.py` | Display-only brain, cord, ganglia and branch dimensions |
| `templates/fly_templates.py` | Bundles all templates into one swappable body configuration |

### Coordinate convention and corrected leg laterality

The body-local coordinate system is:

- `+X`: forward, toward the head
- `+Y`: upward
- `-Z`: the fly's anatomical left
- `+Z`: the fly's anatomical right

The previous leg bug came from assigning the hip attachment a negative
lateral offset and then multiplying it by the side sign, while the foot used a
positive offset multiplied by that sign. The hip and foot therefore occupied
opposite sides and the leg crossed the midline. The revised geometry stores
lateral offsets as positive magnitudes and multiplies by the side sign exactly
once. The hip, knee, ankle and foot now stay on their assigned side.

The camera currently views approximately from `+Z`, so the near-side (the
fly's anatomical right) can appear on the opposite visual side of the screen
from what a viewer casually calls "left". The names `front_left` etc. are
anatomical labels in the fly's local coordinate frame.

### Swap templates without editing the body assembler

The body accepts a `FlyAnatomyTemplates` bundle. Use `dataclasses.replace()`
to alter only the component you are experimenting with:

```python
from dataclasses import replace
from templates import DEFAULT_FLY_TEMPLATES
from templates.leg_geometry import LegGeometryTemplate
from templates.wing_geometry import WingTemplate

my_templates = replace(
    DEFAULT_FLY_TEMPLATES,
    legs=replace(DEFAULT_FLY_TEMPLATES.legs,
                 front_attach_z=0.27, foot_base_z=0.88, stride=0.30),
    wings=replace(DEFAULT_FLY_TEMPLATES.wings,
                  length=1.18, chord=0.42, span=1.12),
)
```

Then supply that bundle when constructing the simulation:

```python
simulation = FlySimulation(args, body_templates=my_templates)
```

For an independent wing-stroke experiment, implement an object with
`step(dt, activation)` and `wing_frame(side, activation, span, sweep_x)`, then
supply it as `wing_motion_model=MyWingMotionModel()`. The default oscillator is
a visual kinematic placeholder; its rates are radians per second and are not a
measured biological wingbeat frequency.

### Geometry regression tests

Run the VPython-free checks from the project root:

```bash
python -m unittest discover -s tests -v
```

The tests check that every leg joint remains on its named side, that left/right
leg geometry mirrors when gait phase is aligned, and that wing axes are mirrored
rather than crossing the body midline.
