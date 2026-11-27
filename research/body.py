"""Approximate planar sensorimotor plant; no GUI, policy, or neural dependencies."""
import math
import numpy as np
from templates import DEFAULT_FLY_TEMPLATES
from templates.leg_geometry import local_leg_points
from templates.wing_motion import WingMotionModel
from .interface import LEG_NAMES, MotorCommand

class Body:
    def __init__(self, templates=DEFAULT_FLY_TEMPLATES):
        self.templates=templates
        self.pos=np.array([-2.8,0.]); self.vel=np.zeros(2); self.yaw=0.; self.omega=0.
        self.phase=np.zeros(6); self.activation=np.zeros(6); self.fatigue=np.zeros(6)
        self.wing=WingMotionModel(templates.wing_motion)
        self.internal=dict(hunger=.30,energy=.9,glucose_proxy=.78,octopamine_proxy=.22,nociception=0.,temperature_c=24.,hydration=.9,stress=0.,fatigue=0.)
        self.contacts=[]; self.legs={}; self.accel=np.zeros(2); self.food_consumed=0.
        self.last_command=MotorCommand().checked(); self.motor_gain=np.ones(6)
        self.refresh_legs()

    def rotation(self):
        c,s=math.cos(self.yaw),math.sin(self.yaw)
        return np.array([[c,s],[-s,c]]) # matches VPython rotation about +Y

    def refresh_legs(self):
        t=self.templates.legs
        for i,name in enumerate(LEG_NAMES):
            side=-1 if i%2==0 else 1
            p=local_leg_points(i//2,side,self.phase[i]/t.gait_angular_rate,template=t)
            phase=self.phase[i]+(math.pi if (i//2+i%2)%2 else 0.)
            a=np.asarray(p['attach']); knee=np.asarray(p['knee']); ankle=np.asarray(p['ankle']); foot=np.asarray(p['foot'])
            v1=a-knee; v2=ankle-knee
            angle=math.acos(float(np.clip(v1@v2/max(1e-12,np.linalg.norm(v1)*np.linalg.norm(v2)),-1,1)))
            stance=math.sin(phase)<=1e-9
            # Tension proxy: activated muscle + passive extension relative to rest.
            length=float(np.linalg.norm(ankle-foot)); rest=.28
            tension=max(0.,(length-rest)*2.)+.75*self.activation[i]*(1-self.fatigue[i])
            self.legs[name]=dict(points=p,phase=phase,stance=stance,angle=angle,tendon_length=length,tension=tension,
                                 activation=float(self.activation[i]),fatigue=float(self.fatigue[i]),force=0.,object_contact=0.)

    def step(self, command, experiment, k):
        dt=experiment.dt; self.last_command=command.checked(); cmd=self.last_command
        wind,temp,_,_,food=experiment.fields(k); old_vel=self.vel.copy()
        self.motor_gain=np.ones(6)
        for e in experiment.active(k):
            if e.kind=='motor_gain': self.motor_gain[LEG_NAMES.index(e.target)]=e.value
        rates=np.array([cmd.rate[n] for n in LEG_NAMES])*self.motor_gain
        efforts=np.array([cmd.effort[n] for n in LEG_NAMES])*self.motor_gain
        self.activation+=(efforts-self.activation)*(1-math.exp(-dt*15))
        self.fatigue=np.clip(self.fatigue+dt*(.06*self.activation-.025*(1-self.activation)),0,.95)
        self.phase=(self.phase+dt*self.templates.legs.gait_angular_rate*rates)%(2*math.pi)
        self.refresh_legs()
        # Ground traction from stance stroke; no separate global drive or steering.
        # These are phenomenological forces, not a rigid-body or muscle solver.
        # Stance phase offsets are peripheral mechanics, not a decision policy.
        forces=np.array([1.5*rates[i]*self.activation[i]*(1-self.fatigue[i])*self.motor_gain[i]*float(self.legs[n]['stance']) for i,n in enumerate(LEG_NAMES)])
        # A foot striking a solid object loses its nominal traction for this tick.
        # This is a planar contact proxy (not a resolved 3D contact impulse).
        for i,n in enumerate(LEG_NAMES):
            foot=self.legs[n]['points']['foot']; xy=self.pos+self.rotation()@np.array([foot[0],foot[2]])
            overlap=max([0.]+[r+.05-float(np.linalg.norm(xy-np.array([x,z]))) for name,(x,z,r) in experiment.bodies(k).items() if name!='food'])
            self.legs[n]['object_contact']=overlap
            if overlap>0: forces[i]=0.
            self.legs[n]['force']=float(forces[i])
        forward=self.rotation()@np.array([1.,0.]); traction=forward*forces.sum()/3.
        torque=(forces[1::2].sum()-forces[::2].sum())*.35
        self.omega+=dt*(torque-2.5*self.omega); self.yaw=(self.yaw+dt*self.omega+math.pi)%(2*math.pi)-math.pi
        self.vel+=dt*(traction+0.08*(wind-self.vel)-1.4*self.vel)
        self.pos+=self.vel*dt; self.contacts=[]
        radius=.43
        for name,(x,z,r) in experiment.bodies(k).items():
            if name=='food': continue
            delta=self.pos-np.array([x,z]); dist=float(np.linalg.norm(delta)); overlap=radius+r-dist
            if overlap>0:
                normal=delta/dist if dist>1e-9 else np.array([-1.,0.])
                self.pos+=normal*overlap
                vn=float(self.vel@normal)
                if vn<0: self.vel-=vn*normal
                self.contacts.append((name, overlap, normal.copy()))
        limit=experiment.arena.half_size-radius
        for j in range(2):
            if abs(self.pos[j])>limit:
                sign=np.sign(self.pos[j]); overlap=abs(self.pos[j])-limit; self.pos[j]=sign*limit
                if sign*self.vel[j]>0:self.vel[j]=0.
                normal=np.zeros(2); normal[j]=-sign; self.contacts.append(('wall',overlap,normal))
        self.accel=(self.vel-old_vel)/dt
        self.wing.step(dt,cmd.wing)
        work=float(np.mean(self.activation*np.abs(rates)))+.2*cmd.wing
        ix=self.internal; consumed=0.
        if np.linalg.norm(self.pos-np.array(food[:2]))<=food[2]+radius: consumed=dt*.15*cmd.feed
        self.food_consumed+=consumed
        ix['energy']=float(np.clip(ix['energy']-dt*(.002+.015*work)+consumed,0,1))
        ix['hunger']=float(np.clip(ix['hunger']+dt*(.004+.012*work)-consumed,0,1))
        ix['glucose_proxy']=float(np.clip(ix['glucose_proxy']-dt*.009*work+consumed,0,1))
        ix['nociception']=float(np.clip(ix['nociception']*math.exp(-dt*2)+sum(v for _,v,_ in self.contacts)*3,0,1))
        ix['stress']=float(np.clip(ix['stress']+dt*(ix['nociception']-.3*ix['stress']),0,1))
        ix['octopamine_proxy']=float(np.clip(.2+.5*ix['stress']+.15*ix['hunger'],0,1))
        ix['temperature_c']+=dt*.3*(temp-ix['temperature_c']+.3*work)
        ix['hydration']=max(0.,ix['hydration']-dt*.0005*(1+work))
        ix['fatigue']=float(self.fatigue.mean())
