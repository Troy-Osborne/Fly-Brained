"""Paired-seed baseline vs intervention summaries, no VPython required."""
import argparse,csv
from dataclasses import replace
import numpy as np
from research.engine import Engine
from research.world import scenario
from research.controllers import CONTROLLERS

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--scenario',default='motor'); p.add_argument('--steps',type=int,default=600); p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44]); p.add_argument('--controllers',nargs='+',choices=CONTROLLERS,default=['idle','open','reflex']); p.add_argument('--noise',type=float,default=0.); p.add_argument('--out',required=True); a=p.parse_args()
    results=[]
    for seed in a.seeds:
        exp=replace(scenario(a.scenario,seed,a.steps),noise=a.noise)
        for name in a.controllers:
            for condition,config in [('control',replace(exp,events=())),('intervention',exp)]:
                engine=Engine(config,CONTROLLERS[name]()); distance=0.; contacts=0; sq=0.; count=0
                for row in engine.run():
                    distance+=float(np.linalg.norm(np.array(row['state_after']['position'])-row['state']['position']))
                    contacts+=bool(row['state_after']['contacts']); error=row['prediction_error'].get('proprio.forward_speed')
                    if error is not None: sq+=error**2; count+=1
                results.append(dict(seed=seed,controller=name,condition=condition,scenario=a.scenario,steps=a.steps,noise=a.noise,path_length=distance,final_x=engine.body.pos[0],final_z=engine.body.pos[1],contact_ticks=contacts,food_consumed=engine.body.food_consumed,forward_speed_rmse=(sq/count)**.5 if count else ''))
    with open(a.out,'x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
if __name__=='__main__': main()
