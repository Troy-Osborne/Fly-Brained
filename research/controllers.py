"""Explicit baseline policies, not part of body mechanics."""
from .interface import MotorCommand, LEG_NAMES

class Idle:
    def step(self, obs): return MotorCommand()

class OpenLoop:
    def step(self, obs):
        return MotorCommand(rate={n:.85 for n in LEG_NAMES}, effort={n:.8 for n in LEG_NAMES},feed=.8)

class Reflex:
    def step(self,obs):
        s=obs.channels
        # Simple sensory steering: turn away from stronger contrast and contact;
        # heading itself and hidden object coordinates are never supplied.
        steer=max(-.5,min(.5,2*(s.get('vision.left.contrast',0)-s.get('vision.right.contrast',0))))
        if s.get('tactile.body_contact',0): steer=.6
        rates={n:max(-1,min(1,.65+steer*(1 if n.endswith('left') else -1))) for n in LEG_NAMES}
        return MotorCommand(rates,{n:.8 for n in LEG_NAMES},feed=.8,
                            expected={k:s[k] for k in ('proprio.forward_speed','airflow.speed','pressure.pa') if k in s})

class Replay:
    def __init__(self,commands): self.commands=commands
    def step(self,obs): return MotorCommand(**self.commands[obs.tick])

CONTROLLERS={'idle':Idle,'open':OpenLoop,'reflex':Reflex}
