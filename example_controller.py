"""Replace this class with an arbitrary neural, symbolic, or hybrid controller."""
from research.interface import MotorCommand, LEG_NAMES

class Controller:
    def step(self, observation):
        s=observation.channels
        # Independent signals for each leg. Prediction is optional.
        return MotorCommand(rate={n:.6 for n in LEG_NAMES}, effort={n:.7 for n in LEG_NAMES},
                            expected={'pressure.pa':s['pressure.pa']} if 'pressure.pa' in s else {})
