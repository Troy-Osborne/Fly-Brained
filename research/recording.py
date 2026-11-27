"""Self-contained run folders; replay uses recorded commands, not a live controller."""
from dataclasses import asdict
from pathlib import Path
import json, platform, hashlib
import numpy as np
from .world import Experiment, Event, Arena


def dump(path, obj): Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def manifest(engine, controller_name):
    return {'schema_version':1,'experiment':asdict(engine.experiment),'templates':asdict(engine.body.templates),
            'controller':controller_name,'external_stream_source':engine.stream_source is not None,'python':platform.python_version(),'numpy':np.__version__,
            'source_sha256':{str(p.relative_to(Path(__file__).parent.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for folder in ('research','templates') for p in sorted((Path(__file__).parent.parent/folder).glob('*.py'))}}

class Recorder:
    def __init__(self,path,engine,controller_name,raw=False):
        self.path=Path(path); self.path.mkdir(parents=True,exist_ok=False); self.raw=raw
        dump(self.path/'manifest.json',manifest(engine,controller_name))
        self.file=(self.path/'ticks.jsonl').open('w',encoding='utf-8')
        if raw: (self.path/'streams').mkdir()
        self.count=0; self.distance=0.; self.contacts=0; self.errsum={}; self.errcount={}; self.final=None
    def write(self,row,streams):
        self.file.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n'); self.count+=1
        if self.raw: np.savez_compressed(self.path/'streams'/f"{row['tick']:06d}.npz",**streams)
        self.distance+=float(np.linalg.norm(np.array(row['state_after']['position'])-row['state']['position']))
        self.contacts+=bool(row['state_after']['contacts']); self.final=row['state_after']
        for k,v in row['prediction_error'].items(): self.errsum[k]=self.errsum.get(k,0)+v*v; self.errcount[k]=self.errcount.get(k,0)+1
    def close(self):
        self.file.close(); dump(self.path/'summary.json',{'steps':self.count,'path_length_model_units':self.distance,
            'contact_ticks':self.contacts,'final_state':self.final,'prediction_rmse_by_channel':{k:(v/self.errcount[k])**.5 for k,v in self.errsum.items()},
            'prediction_counts':self.errcount})

def load_run(path):
    path=Path(path); m=json.loads((path/'manifest.json').read_text()); e=m['experiment']; e['arena']=Arena(**e['arena']); e['events']=tuple(Event(**v) for v in e['events'])
    rows=[json.loads(l) for l in (path/'ticks.jsonl').read_text().splitlines()]
    return Experiment(**e),m,rows
