"""Synthetic egocentric transducers. Numeric units are documented in README."""
import math
import numpy as np
from .interface import LEG_NAMES

SAMPLE_RATE=4000

def sample(body, exp, tick):
    wind,temp,pressure,_,_=exp.fields(tick)
    rot=body.rotation(); relative=rot.T@(wind-body.vel); vel=rot.T@body.vel; acc=rot.T@body.accel
    s={'proprio.forward_speed':float(vel[0]),'proprio.lateral_speed':float(vel[1]),'proprio.yaw_rate':body.omega,
       'proprio.forward_accel':float(acc[0]),'proprio.lateral_accel':float(acc[1]),'airflow.forward':float(relative[0]),
       'airflow.lateral':float(relative[1]),'airflow.speed':float(np.linalg.norm(relative)), 'pressure.pa':pressure,
       'thermal.ambient_c':temp, 'tactile.body_contact':float(bool(body.contacts)),
       'tactile.contact_load':sum(overlap for _,overlap,_ in body.contacts),
       'antenna.left.deflection':float(relative[1]+.25*relative[0]),'antenna.right.deflection':float(-relative[1]+.25*relative[0]),
       'haltere.yaw_rate':body.omega,'wing.activation':body.last_command.wing,'wing.phase_sin':math.sin(body.wing.phase)}
    for n in LEG_NAMES:
        leg=body.legs[n]
        for key in ('angle','activation','tension','tendon_length','fatigue','force'):
            s[f'proprio.{n}.{key}']=float(leg[key])
        s[f'tactile.{n}.ground']=float(leg['stance'])
        s[f'tactile.{n}.object_load']=float(leg['object_contact'])
    s.update({f'intero.{k}':float(v) for k,v in body.internal.items()})
    custom=dict(exp.custom)
    for e in exp.active(tick):
        if e.kind=='custom': custom[e.target]=e.value
    s.update({f'intero.custom.{k}':float(v) for k,v in custom.items()})
    bodies=exp.bodies(tick); eye_images=[]; n=48
    # Two 120-degree fields, centred +/- 60 degrees. Ray-circle intersection,
    # with occlusion, proximity brightness and distinguishable food contrast.
    for side in (-1,1):
        angles=np.linspace(-math.pi/3,math.pi/3,n)+side*math.pi/3
        dirs=np.stack([np.cos(angles),np.sin(angles)],axis=1)
        nearest=np.full(n,10.); intensity=np.full(n,.08)
        for name,(x,z,r) in bodies.items():
            delta=rot.T@(np.array([x,z])-body.pos); proj=dirs@delta
            disc=r*r-(delta@delta-proj*proj); hit=(disc>=0)&(proj>0)
            distance=np.maximum(0,proj-np.sqrt(np.maximum(0,disc)))
            update=hit&(distance<nearest); nearest[update]=distance[update]
            intensity[update]=(.9 if name=='food' else .5)/(1+.25*distance[update])
        eye_images.append(np.tile(intensity,(8,1)).astype(np.float32))
        tag='left' if side==-1 else 'right'
        s[f'vision.{tag}.brightness']=float(intensity.mean())
        s[f'vision.{tag}.contrast']=float(intensity.std())
    # Synthetic stereo particle-motion-like signal, not a solved acoustic field.
    count=round(exp.dt*SAMPLE_RATE); t=(tick*count+np.arange(count))/SAMPLE_RATE
    audio=np.zeros((count,2),dtype=np.float32)
    if 'agent' in bodies:
        x,z,_=bodies['agent']; delta=rot.T@(np.array([x,z])-body.pos); dist=np.linalg.norm(delta)
        pan=float(np.clip(delta[1]/max(dist,.01),-1,1))
        for i,side in enumerate((-1,1)):
            audio[:,i]=(.1/(1+dist))*(1+.5*side*pan)*np.sin(2*math.pi*180*t+side*pan*.5)
    audio+= (.01*min(5,np.linalg.norm(relative))*np.sin(2*math.pi*45*t))[:,None]
    for i,tag in enumerate(('left','right')): s[f'audio.{tag}.rms']=float(np.sqrt(np.mean(audio[:,i]**2)))
    streams={'vision.left':eye_images[0],'vision.right':eye_images[1],'audio':audio}
    return s,streams


def deliver(s,streams,exp,tick,rng):
    masks=[e.target for e in exp.active(tick) if e.kind=='mask' and e.value!=0]
    def valid(k): return not any(k==m or k.startswith(m+'.') for m in masks)
    flags={k:valid(k) for k in s}; sf={k:valid(k) for k in streams}
    channels={k:float(v+ (rng.normal(0,exp.noise) if exp.noise else 0)) for k,v in s.items() if flags[k]}
    out={k:np.array(v,copy=True) for k,v in streams.items() if sf[k]}
    for arr in out.values(): arr.setflags(write=False)
    return channels,flags,out,sf


def replace_streams(channels, streams, supplied):
    """Replace named synthetic streams with externally supplied normalized buffers.
    Audio: samples x 2, 4 kHz; eyes: 2D grayscale, [0,1]. Caller owns resampling.
    """
    for name,value in supplied.items():
        if name not in streams: raise ValueError(f'unknown stream {name}')
        arr=np.array(value,dtype=np.float32,copy=True)
        if arr.ndim!=2 or not arr.size or not np.isfinite(arr).all(): raise ValueError(f'invalid buffer {name}')
        if name=='audio':
            if arr.shape!=streams[name].shape: raise ValueError('audio must match tick sample count and stereo channels')
            for i,side in enumerate(('left','right')): channels[f'audio.{side}.rms']=float(np.sqrt(np.mean(arr[:,i]**2)))
        else:
            if arr.min()<0 or arr.max()>1: raise ValueError('eye intensity must be in [0,1]')
            channels[name+'.brightness']=float(arr.mean()); channels[name+'.contrast']=float(arr.std())
        streams[name]=arr
