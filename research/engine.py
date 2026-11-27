"""Fixed-step experiment runner with explicit causal timing and separated truth."""
from dataclasses import asdict
from types import MappingProxyType
import numpy as np
from .body import Body
from .interface import Observation, MotorCommand
from .sensors import sample, deliver
from .controllers import Idle
from templates import DEFAULT_FLY_TEMPLATES

class Engine:
    def __init__(self, experiment, controller=None, templates=DEFAULT_FLY_TEMPLATES, stream_source=None):
        self.experiment=experiment.validate(); self.controller=controller if controller is not None else Idle()
        self.stream_source=stream_source
        self.body=Body(templates, experiment.leg_actuator, experiment.initial_phase_rad); self.rng=np.random.default_rng(experiment.seed); self.tick=0
        self.pending={}; self.last_streams={}; self.last_observation=None

    def step(self):
        k=self.tick; exp=self.experiment
        if k>=exp.steps: raise StopIteration
        truth,raw=sample(self.body,exp,k)
        if self.stream_source is not None:
            # Explicit external source; same simulation timestamp as synthetic inputs.
            from .sensors import replace_streams
            replace_streams(truth,raw,self.stream_source(k,k*exp.dt,exp.dt))
        channels,valid,streams,stream_valid=deliver(truth,raw,exp,k,self.rng)
        errors={key:channels[key]-val for key,val in self.pending.items() if key in channels}
        missing=sorted(set(self.pending)-set(errors))
        obs=Observation(k,k*exp.dt,exp.dt,MappingProxyType(channels),MappingProxyType(valid),
                        MappingProxyType(streams),MappingProxyType(stream_valid),MappingProxyType(errors))
        self.last_observation=obs; self.last_streams=streams
        cmd=self.controller.step(obs)
        if not isinstance(cmd,MotorCommand): raise TypeError('controller must return MotorCommand')
        cmd=cmd.checked()
        row={'tick':k,'time':obs.time,'observation':channels,'valid':valid,'stream_valid':stream_valid,
             'expected_previous':dict(self.pending),'prediction_error':errors,'prediction_unavailable':missing,
             'command':asdict(cmd),'truth':truth,'state':{'position':self.body.pos.tolist(),'velocity':self.body.vel.tolist(),
                    'yaw':self.body.yaw,'food_consumed':self.body.food_consumed,'contacts':[c[0] for c in self.body.contacts]},
             'events':[asdict(e) for e in exp.active(k)]}
        self.body.step(cmd,exp,k); self.pending=dict(cmd.expected); self.tick+=1
        row['state_after']={'position':self.body.pos.tolist(),'velocity':self.body.vel.tolist(),'yaw':self.body.yaw,
                            'food_consumed':self.body.food_consumed,'contacts':[c[0] for c in self.body.contacts],
                            'motor_gain':self.body.motor_gain.tolist()}
        return row

    def run(self):
        while self.tick<self.experiment.steps: yield self.step()
