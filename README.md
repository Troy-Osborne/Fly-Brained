# Fly-Brained

Fly-Brained combines a controller-independent fruit fly body, a configurable environment and closed-loop neural experiments. The first integration checks whether front-foot contact produces a measurable turn away from an obstacle.

From the repository root, install the headless dependencies and run:

```sh
python -m pip install -r requirements-headless.txt
python -m unittest discover -s tests -v
python -m experiments.touch_reflex
```

To use a separately obtained Spiking_Microcircuit_Composites checkout, add its root to your Python path (for example, `export PYTHONPATH=/path/to/Spiking_Microcircuit_Composites` on Linux or macOS), then run:

```sh
python -m experiments.touch_reflex --controller mcf
python -m unittest discover -s tests -v
```

The adapter was exercised with MCF revision `6212fd1e3a6b3d7185d265dd46f1150eeffabe2f`. Each 20 ms body tick thresholds the two front-foot object loads, advances one two-channel relay step and maps the output to leg stroke rates and efforts. The motor mapping is hand-written. Each JSON line records the configuration, contact and response ticks, and final yaw. Negative yaw turns right. The initial gait phases place the two front feet at different forward positions, so obstacles are placed under each actual foot. These trials test response direction and timing. Learning, phase-balanced comparisons and resource scaling are future experiments.

## Original v0.3 body guide

# Drosophila embodied cognition testbed — v0.3

**Research prototype, not a validated male fly biomechanics or neural simulation.**
The plant, environment, transducers and renderer are separate from any decision-making
controller. No connectome, network, reward maximizer or learning policy is required.
The original v0.2 visual sandbox remains available (`drosophila_vpython.py`), but
use `run_experiment.py` for the new experimental engine. Old `ControlPacket` and
`controller_template.py` apply only to the legacy sandbox.

## Quick start (Windows PowerShell)

From the extracted directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-headless.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe run_experiment.py --scenario wind --controller reflex --out runs/wind_01 --raw
.\.venv\Scripts\python.exe run_experiment.py --replay runs/wind_01 --out runs/wind_replay
```

To enable VPython, install `requirements.txt` using the same Python, then:

```powershell
.\.venv\Scripts\python.exe run_experiment.py --scenario motor --controller open --out runs/motor_view --view
```

Output paths must not already exist. Headless mode is the default, executes without
VPython, and is preferable for data collection. The optional view is paced at the
simulation step rate and receives the exact leg positions and wing phase from the
engine. Browser rendering has not been tested in the delivery environment.

## Architecture

```
experiment config -> world fields / bodies -> plant (body.py)
                          |                        ^
                          v                        | commands
                     sensors.py -> delivery -> Controller.step(observation)
                                      ^                |
                                      |                | optional next-tick prediction
                                  comparator <---------+
                       engine.py -> recordings, optional renderer
```

- `research/interface.py`: public observation/motor contract, no VPython.
- `research/body.py`: independent leg effort and signed stroke rate, activation
  dynamics, tendon/force proxies, gait geometry, planar locomotion, contacts,
  energy, feeding and wing oscillator. No hidden high-level drive or steering.
- `research/world.py`: arena, deterministic moving object, food, fields,
  interventions and config validation.
- `research/sensors.py`: local-frame transduction, synthetic binocular vision,
  stereo audio, tactile channels, proprioception, interoception, optional external
  stream replacement and sensory masks.
- `research/engine.py`: fixed-tick causal coordinator, controller-independent.
- `research/controllers.py`: explicit idle, open-loop, reflex and replay controls.
- `research/recording.py`: manifest, per-tick logs, optional raw stream files and metrics.
- `research/render.py`: VPython view of the research engine, no simulation authority.
- `templates/`: original separate anatomical and wing/leg motion templates.
- `example_controller.py`: minimal brain-independent adapter example.
- `compare_experiments.py`: paired-seed baseline/intervention batch CSV.
- `tests/`: headless correctness and geometry tests.

### Replaceable anatomy

`Engine(experiment, controller, templates=...)` accepts `FlyAnatomyTemplates`.
Use `dataclasses.replace(DEFAULT_FLY_TEMPLATES, legs=replace(...), wings=replace(...))`.
Leg and wing motion parameter templates feed both physics proxies and graphics;
head, eye, antenna, thorax, abdomen, and nerve geometry are presently primarily
visual. See `templates/README.md` for the v0.2 organization. Changing visual wing
shape does **not** yet change lift or aerodynamics. The renderer uses each leg's
explicit points rather than recalculating its gait. The fly frame remains +X forward,
+Y up, -Z left, +Z right. VPython yaw rotates +X toward -Z.

## Experiments

All named interventions are active over the middle third of a run, with pre/post
phases. Default dt=0.02 s, 600 steps = 12 simulated seconds.

| Scenario | Intervention |
|---|---|
| baseline | No imposed perturbation |
| wind | Lateral wind field |
| motor | Front-left actuator gain set to zero; restored at end |
| blind | Both eye streams and all vision scalar channels withheld |
| deaf | Stereo audio and audio features withheld |
| touch | Tactile channels withheld |
| conflict | Proprioception withheld while lateral wind applied (not fabricated sensory conflict) |
| obstacle | Obstacle repositioned into the route |
| temperature | Ambient temperature raised |
| pressure | Ambient pressure lowered |
| hunger | Food repositioned; hunger evolves endogenously (not forced hunger) |

Examples:

```powershell
python run_experiment.py --scenario blind --controller reflex --out runs/blind --raw
python run_experiment.py --scenario motor --controller idle --out runs/idle
python run_experiment.py --config examples/custom_experiment.json --controller open --out runs/custom
python compare_experiments.py --scenario motor --seeds 42 43 44 --controllers idle open reflex --out motor_comparison.csv
```

`compare_experiments.py` pairs each seed/controller across an event-free control and
an intervention. The bundled example uses seed 42, 300 steps, zero noise. Distinct
seeds only vary stochastic sensory noise when nonzero; use `--noise` for a simple
noise experiment. Noise is a single absolute Gaussian standard deviation in each
channel's own units, not calibrated biological noise; avoid interpreting it as
comparable across modalities. Pairing alone does not guarantee matched random
variates after a mask changes which channels are sampled. Metrics are descriptive,
not evidence of cognition or biological realism.

Custom JSON: `name`, `seed`, `dt`, `steps`, optional `arena`, `noise`, `custom` and
`events`. Each event has `start`, `stop` (exclusive integer ticks), `kind`, `target`,
`value`. Kinds: `wind` target x/z, `motor_gain` target leg name [0,1], `mask` target
exact channel or dot-separated prefix, `temperature`, `pressure`, `custom` target
variable name, `obstacle`/`food` target x/z. Later overlapping assignments win.
Events apply on observation and transition tick k; contact during transition k is
available at k+1. Custom scalar channels are externally controlled, not physiological
ODEs. No automatic recovery policy is installed in the body.

## Controller contract

```python
from research.interface import MotorCommand, LEG_NAMES
from research.engine import Engine
from research.world import scenario

class MyBrain:
    def step(self, obs):
        channels = obs.channels       # delivered measurements only
        valid = obs.valid             # availability, not a masked value
        images = obs.streams          # vision.left/right, audio when available
        error = obs.prediction_error  # observation minus prior prediction
        return MotorCommand(
            rate={n: 0.5 for n in LEG_NAMES},
            effort={n: 0.8 for n in LEG_NAMES},
            wing=0.0, antenna=0.3, feed=0.2,
            expected={'pressure.pa': channels['pressure.pa']}
                       if 'pressure.pa' in channels else {},
        )

engine = Engine(scenario('wind'), MyBrain())
for tick in engine.run():
    pass  # researcher-only log; never feed it to the controller as an observation
```

Or `python run_experiment.py --controller example_controller:Controller --out runs/custom_brain`.
`Engine(experiment)` uses idle control. All unspecified motor commands default to
zero. Rates are signed [-1,1]; efforts, wing, antenna and feed are [0,1]. Invalid
names/nonfinite values are rejected; out-of-range finite inputs are clipped.
A neural runtime can keep its own internal state in its controller instance. It
can perform multiple neural substeps per observation; the body step is fixed.

At tick k: sample current body/environment -> apply mask/noise -> compare available
channels against predictions from k-1 -> call controller -> validate and apply
command -> advance body -> record. A first observation has no previous prediction.
Missing channels produce no error; `prediction_unavailable` records omissions.
Predictions are optional and are not computed by the body. No aggregate mixes
pressure, speed or other unlike units. The controller sees read-only channel maps
and copied arrays; researcher logs contain truth and must not be used as controller
inputs. This is a software interface, not a security sandbox for untrusted Python.

### Sensory/actuator details and units

| Channels | Interpretation |
|---|---|
| `proprio.forward_speed`, `lateral_speed`, accelerations | local frame, model units/s, model units/s² |
| `proprio.yaw_rate`, `haltere.yaw_rate` | radians/s, proxy |
| `proprio.<leg>.angle` | knee interior angle in radians |
| `proprio.<leg>.activation`, `fatigue` | normalized [0,1] proxies |
| `proprio.<leg>.tendon_length` | model units |
| `proprio.<leg>.tension`, `force` | arbitrary force units (force signed traction) |
| `tactile.<leg>.ground` | stance phase proxy (not solved ground contact) |
| `tactile.<leg>.object_load`, `tactile.contact_load` | planar overlap proxy in model units |
| `tactile.body_contact` | contact flag from previous transition |
| `airflow.*`, `antenna.*` | relative local air motion/deflection proxies, model units/s |
| `pressure.pa` | ambient Pascal value, not computational fluid dynamics |
| `thermal.ambient_c`, `intero.temperature_c` | degrees Celsius |
| `intero.*` | hunger, energy, glucose, octopamine, nociception, hydration, stress, fatigue: normalized proxies |
| `vision.left`, `vision.right` streams | 8×48 grayscale, simplified planar ray projection |
| `audio` stream | 4 kHz stereo samples × 2, synthetic amplitude |
| `vision.*.brightness/contrast`, `audio.*.rms` | scalar features derived from corresponding buffers |
| `wing.activation`, `wing.phase_sin` | oscillator state, no flight dynamics |

Antenna command currently animates antenna posture; does not yet alter sensory
transfer. Wing commands affect activation, oscillator and energy expenditure,
not lift. No olfactory, gustatory, photoreceptor spectral, or fluid solver is
implemented. Food is an inexhaustible region; feeding requires proximity + command.
All uncalibrated spatial quantities are model units, not millimetres or metres.

### External streams

Pass `stream_source=` to `Engine`; it is a function `(tick, time, dt) -> dict`.
Keys `audio`, `vision.left`, `vision.right` replace corresponding synthetic buffers.
Resample audio to 4 kHz stereo in advance; exactly round(4000*dt) samples per tick.
Images are 2D normalized float grayscale. Omitted keys retain synthetic data.
Features are recomputed before masking. A live camera/microphone adapter or recorded
source can implement this callable; synchronized live capture is not provided by
the new runner (legacy v0.2 still has its original optional live adapters).
Replay of external streams requires supplying the same source in Python; CLI replay
explicitly rejects external-input runs. Raw recordings contain delivered streams,
not masked-away buffers.

## Data and reproducibility

Each run directory contains `manifest.json` (experiment, anatomy, seed, versions,
source hashes, controller name, external-source flag), `ticks.jsonl` (delivered
channels, valid flags, previous prediction, signed errors, command, researcher-only
truth, before/after state, active interventions), `summary.json` (path length,
contact ticks, food, per-channel prediction RMSE/counts), and optional `streams/`
with one compressed NumPy archive per tick. The manifest hashes bundled research
and anatomy modules, not arbitrary third-party controller code; archive your own
controller code/config separately. Keep the entire folder plus code for provenance.
A partial run leaves partial logs/summary. Replay requires a completed run,
default anatomy, synthetic streams, same compatible code/runtime; it verifies every
scalar log row, not bitwise raw buffers. No guarantees across Python/NumPy versions.

## Evidence and limitations

17 unit tests passed (see `examples/test_report.txt`); a 120-tick wind/reflex run
was replayed exactly; all 11 scenarios were exercised in short headless tests.
`examples/motor_comparison.csv` contains a six-second paired test with one seed,
three explicit controllers. Actual VPython browser rendering was not tested here.

Physics is intentionally reduced order: planar translation and yaw, circle contact,
kinematic limb poses and stance-conditioned traction. It is not mass/inertia-calibrated,
not a rigid-link constant-length mechanical solver, and does not resolve joint torques,
true tendon dynamics, individual antagonistic muscle fibers, adhesion or flight.
Motor failure removes commanded traction, not a validated lesion. Neural pathways are
visual, not mapped receptors or motor neurons. No claims about real fly cognition,
learning or male-specific physiology follow from these tests.

Next scientific milestone: replace the plant's proxy traction model with articulated
contact mechanics (e.g. MuJoCo) and explicit antagonist musculotendon dynamics while
preserving the public contract. Add calibrated receptor transfer functions and
receptor/muscle mapping before connecting any real connectome. For cognition tests,
add task-based metrics (goal acquisition, adaptation time, retained performance),
matched perturbation controls, independent held-out tasks, and learning controllers
whose mechanisms are explicitly separate from the body.
