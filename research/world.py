"""Deterministic planar arena, scheduled interventions, and physical fields."""
from dataclasses import dataclass, field
import math
import numpy as np

@dataclass(frozen=True)
class Event:
    start: int
    stop: int  # exclusive; zero-based transition tick
    kind: str
    target: str = ''
    value: float = 0.

@dataclass(frozen=True)
class Arena:
    half_size: float = 6.
    obstacle: tuple = (0.7, 0., 0.55) # x,z,radius in model units
    food: tuple = (2.7, 1.6, 0.5)
    moving_body: bool = True
    wind: tuple = (0.,0.)
    temperature: float = 24.
    pressure: float = 101325.

@dataclass(frozen=True)
class Experiment:
    name: str = 'baseline'
    seed: int = 42
    dt: float = 0.02
    steps: int = 600
    arena: Arena = field(default_factory=Arena)
    events: tuple[Event,...] = ()
    noise: float = 0.
    custom: dict = field(default_factory=dict)

    def validate(self):
        if not (0 < self.dt <= .05 and self.steps > 0 and self.noise >= 0): raise ValueError('invalid simulation parameters')
        if not math.isclose(self.dt*4000, round(self.dt*4000), abs_tol=1e-8): raise ValueError('dt must align to 4 kHz audio samples')
        from .interface import LEG_NAMES
        for e in self.events:
            if not (0 <= e.start < e.stop <= self.steps) or not math.isfinite(e.value): raise ValueError(f'invalid event {e}')
            if e.kind not in {'wind','motor_gain','mask','temperature','pressure','custom','obstacle','food'}: raise ValueError(f'unknown intervention {e.kind}')
            if e.kind == 'motor_gain' and (e.target not in LEG_NAMES or not 0<=e.value<=1): raise ValueError(f'invalid motor gain {e}')
            if e.kind == 'wind' and e.target not in {'x','z'}: raise ValueError(f'invalid wind axis {e}')
            if e.kind == 'mask' and not e.target: raise ValueError('empty mask')
            if e.kind in {'food','obstacle'} and e.target not in {'x','z'}: raise ValueError('invalid object axis')
        return self

    def active(self,k): return [e for e in self.events if e.start <= k < e.stop]

    def fields(self,k):
        wind = np.array(self.arena.wind,dtype=float); temp=self.arena.temperature; pressure=self.arena.pressure
        obstacle=list(self.arena.obstacle); food=list(self.arena.food)
        for e in self.active(k):
            if e.kind=='wind': wind[0 if e.target=='x' else 1]=e.value
            elif e.kind=='temperature': temp=e.value
            elif e.kind=='pressure': pressure=e.value
            elif e.kind in {'obstacle','food'}: (obstacle if e.kind=='obstacle' else food)[0 if e.target=='x' else 1]=e.value
        return wind,temp,pressure,tuple(obstacle),tuple(food)

    def bodies(self,k):
        _,_,_,obstacle,food=self.fields(k)
        t=k*self.dt
        bodies={'obstacle':obstacle,'food':food}
        if self.arena.moving_body: bodies['agent']=(1.3+1.2*math.sin(t*.7),1.8*math.cos(t*.7),.3)
        return bodies


def scenario(name, seed=42, steps=600, dt=.02):
    a,b=steps//3,2*steps//3
    events={
      'baseline':(),
      'wind':(Event(a,b,'wind','z',3.),),
      'motor':(Event(a,b,'motor_gain','front_left',0.),),
      'blind':(Event(a,b,'mask','vision',1.),),
      'deaf':(Event(a,b,'mask','audio',1.),),
      'touch':(Event(a,b,'mask','tactile',1.),),
      'conflict':(Event(a,b,'mask','proprio',1.),Event(a,b,'wind','z',2.)),
      'obstacle':(Event(a,b,'obstacle','x',-1.3),),
      'temperature':(Event(a,b,'temperature','',35.),),
      'pressure':(Event(a,b,'pressure','',101285.),),
      'hunger':(Event(a,b,'food','x',-2.),Event(a,b,'food','z',0.)),
    }
    if name not in events: raise ValueError(f'unknown scenario: {name}')
    if steps<3: raise ValueError('at least 3 steps required for named scenarios')
    return Experiment(name,seed,dt,steps,events=events[name]).validate()
