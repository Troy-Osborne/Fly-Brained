"""Print matched phase, actuator and contact conditions as JSON lines."""
import json
import math
from experiments.touch_reflex import trial
from nervous_system.adapters.mcf import TouchController


def comparison():
    for phase_index in range(8):
        phase = phase_index * math.tau / 8
        for side in ("left", "right", "sham"):
            for actuator in ("legacy", "series_elastic"):
                yield trial(side, TouchController, actuator, phase)


if __name__ == "__main__":
    for result in comparison():
        print(json.dumps(result, allow_nan=False))
