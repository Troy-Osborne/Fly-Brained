#!/usr/bin/env python3
"""CLI: deterministic headless runs, optional VPython, or exact command replay."""
import argparse, importlib, json
from dataclasses import asdict
from research.world import scenario, Experiment, Arena, Event
from research.engine import Engine
from research.controllers import CONTROLLERS, Replay
from research.recording import Recorder, load_run
from templates import DEFAULT_FLY_TEMPLATES

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario',default='baseline',choices=['baseline','wind','motor','blind','deaf','touch','conflict','obstacle','temperature','pressure','hunger'])
    p.add_argument('--controller',default='idle',help='idle, open, reflex, or module:Class (zero-arg constructor)')
    p.add_argument('--steps',type=int,default=600); p.add_argument('--seed',type=int,default=42); p.add_argument('--dt',type=float,default=.02)
    p.add_argument('--out',required=True,help='new run directory'); p.add_argument('--view',action='store_true'); p.add_argument('--raw',action='store_true')
    p.add_argument('--config',help='JSON experiment configuration (overrides scenario, seed, dt, steps)')
    p.add_argument('--replay',help='existing run directory; same seed, experiment and default anatomy')
    a=p.parse_args(); original=None
    if a.replay:
        exp,m,original=load_run(a.replay)
        if m.get("external_stream_source"): p.error("external-source runs require replaying the original external input in Python")
        if json.dumps(m['templates'],sort_keys=True)!=json.dumps(asdict(DEFAULT_FLY_TEMPLATES),sort_keys=True): p.error('CLI replay only supports default anatomy; use Engine with original custom templates')
        if len(original)!=exp.steps: p.error('incomplete source run')
        ctrl=Replay([r['command'] for r in original]); name='replay'
    else:
        if a.config:
            with open(a.config,encoding='utf-8') as f: config=json.load(f)
            config['arena']=Arena(**config.get('arena',{})); config['events']=tuple(Event(**e) for e in config.get('events',[]))
            exp=Experiment(**config).validate()
        else: exp=scenario(a.scenario,a.seed,a.steps,a.dt)
        name=a.controller
        if name in CONTROLLERS: ctrl=CONTROLLERS[name]()
        else:
            module,cls=name.split(':',1); ctrl=getattr(importlib.import_module(module),cls)()
    engine=Engine(exp,ctrl); renderer=None
    if a.view:
        from research.render import Renderer
        renderer=Renderer(engine)
    rec=Recorder(a.out,engine,name,a.raw)
    try:
        for row in engine.run():
            if original is not None and row!=original[row['tick']]: raise RuntimeError(f'replay divergence at tick {row["tick"]}')
            rec.write(row,engine.last_streams)
            if renderer: renderer.draw(engine,row)
    finally: rec.close()
    print(f'Completed {engine.tick} ticks. Results: {a.out}'+(' (exact replay verified)' if original is not None else ''))
if __name__=='__main__': main()
