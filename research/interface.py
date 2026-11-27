"""Brain-independent, VPython-free public interface (schema v1)."""
from dataclasses import dataclass, field
from typing import Mapping, Protocol
import math
import numpy as np

LEG_NAMES = tuple(f'{region}_{side}' for region in ('front','middle','hind') for side in ('left','right'))

@dataclass(frozen=True)
class MotorCommand:
    # Signed stroke rate: -1 reverse, 0 hold, +1 forward. Separate effort controls muscle activation.
    rate: Mapping[str, float] = field(default_factory=dict)
    effort: Mapping[str, float] = field(default_factory=dict)
    wing: float = 0.0
    antenna: float = 0.0
    feed: float = 0.0
    expected: Mapping[str, float] = field(default_factory=dict) # predictions for tick k+1

    def checked(self):
        def finite(x):
            x = float(x)
            if not math.isfinite(x): raise ValueError('non-finite command')
            return x
        def bound(x, low=0): return max(low, min(1., finite(x)))
        if (set(self.rate)|set(self.effort)) - set(LEG_NAMES): raise ValueError('unknown leg name')
        return MotorCommand({k:bound(self.rate.get(k,0),-1) for k in LEG_NAMES},
                            {k:bound(self.effort.get(k,0)) for k in LEG_NAMES},
                            bound(self.wing),bound(self.antenna),bound(self.feed),
                            {str(k):finite(v) for k,v in self.expected.items()})

@dataclass(frozen=True)
class Observation:
    tick: int
    time: float
    dt: float
    channels: Mapping[str,float]  # ONLY delivered sensory channels, not world truth
    valid: Mapping[str,bool]
    streams: Mapping[str,np.ndarray]  # synthetic camera/audio arrays, read-only copies
    stream_valid: Mapping[str,bool]
    prediction_error: Mapping[str,float] # actual delivered sample minus previous prediction

class Controller(Protocol):
    def step(self, observation: Observation) -> MotorCommand: ...
