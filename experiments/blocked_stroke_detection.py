"""Causal, model-based blocked-stroke detection on the standalone hinge."""
import json
import math
from dataclasses import asdict, dataclass

from research.joint_bench import DT, JointState, advance

# Fixed before evaluation. Threshold is a one-step velocity error in rad/s.
THRESHOLD = .01
DWELL = 3
DURATION = 700


@dataclass(frozen=True)
class Protocol:
    initial: float
    effort: float
    onset: int
    width: int
    clearance: float


PROTOCOLS = (
    Protocol(0., .8, 100, 450, .18),
    Protocol(-.07, .65, 75, 380, .14),
    Protocol(.06, .9, 140, 400, .23),
    Protocol(.02, .72, 55, 470, .20),
)


class Observer:
    """Consume the issued command and two consecutive proprioceptive samples."""
    def __init__(self, mode):
        self.mode = mode
        self.activation = 0.
        self.streak = 0

    def predict(self, angle, velocity, command):
        if self.mode == 'persistence':
            return velocity
        if self.mode == 'no_command':
            command = 0.
        state = JointState(angle, velocity, self.activation)
        # Free motion ignores the stop argument. Activation is reconstructed
        # from issued commands; measured actuator state is never delivered.
        expected, _ = advance(state, command, 0., 'free')
        self.activation = expected.activation
        return expected.velocity

    def observe(self, expected, measured):
        residual = abs(expected - measured)
        self.streak = self.streak + 1 if residual > THRESHOLD else 0
        return residual, self.streak >= DWELL


def evaluate(protocol, condition, mode):
    state = JointState(angle=protocol.initial)
    observer = Observer(mode)
    first_contact = first_alarm = first_contact_alarm = None
    tp = fp = positives = 0
    squared_error = 0.
    # Intervals use zero-based indices. Release is matched across modes.
    release = protocol.onset + 260
    for tick in range(DURATION):
        command = protocol.effort if protocol.onset <= tick < protocol.onset + protocol.width else 0.
        expected = observer.predict(state.angle, state.velocity, command)
        contact = condition
        if condition == 'release':
            contact = 'rigid' if tick < release else 'free'
        state, reaction = advance(state, command, protocol.initial + protocol.clearance, contact)
        residual, alarm = observer.observe(expected, state.velocity)
        loaded = reaction > 1e-9
        squared_error += residual**2
        positives += loaded
        tp += alarm and loaded
        fp += alarm and not loaded
        if loaded and first_contact is None:
            first_contact = tick
        if alarm and first_alarm is None:
            first_alarm = tick
        if loaded and alarm and first_contact_alarm is None:
            first_contact_alarm = tick
    return dict(protocol=asdict(protocol), condition=condition, observer=mode,
                first_contact_tick=first_contact, first_alarm_tick=first_alarm,
                first_loaded_alarm_tick=first_contact_alarm,
                loaded_ticks=positives, true_alarm_ticks=tp, false_alarm_ticks=fp,
                unloaded_ticks=DURATION-positives, velocity_rmse=math.sqrt(squared_error/DURATION))


def benchmark():
    return [evaluate(protocol, condition, mode)
            for protocol in PROTOCOLS
            for condition in ('free', 'compliant', 'rigid', 'release')
            for mode in ('command', 'no_command', 'persistence')]


if __name__ == '__main__':
    print(json.dumps(dict(dt=DT, threshold=THRESHOLD, dwell_ticks=DWELL, rows=benchmark()),
                     allow_nan=False))
