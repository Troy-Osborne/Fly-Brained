"""VPython is a view only. State and exact leg points come from research.Body."""
class Renderer:
    def __init__(self,engine):
        from vpython import canvas, vector, box, sphere, label, color
        from drosophila_vpython import DrosophilaBody
        self.vector=vector; self.scene=canvas(title='Drosophila research v0.3',width=1100,height=700,background=vector(.10,.13,.16))
        self.scene.range=6; self.scene.forward=vector(-.6,-.9,-1.4)
        self.model=DrosophilaBody(engine.body.templates)
        size=engine.experiment.arena.half_size*2
        box(pos=vector(0,-.12,0),size=vector(size,.2,size),color=vector(.23,.29,.25))
        self.objects={n:sphere(radius=r,color=(color.green if n=='food' else color.cyan if n=='agent' else color.gray(.5))) for n,(_,_,r) in engine.experiment.bodies(0).items()}
        self.text=label(pos=vector(0,3,0),text='',box=False,height=14)
    def draw(self,engine,row):
        from drosophila_vpython import ControlPacket
        from vpython import rate
        rate(round(1/engine.experiment.dt)); b=engine.body; v=self.vector; m=self.model
        m.root=v(b.pos[0],-b.templates.legs.foot_y,b.pos[1]); m.yaw=b.yaw; m.phase_time=engine.tick*engine.experiment.dt
        m.wing_motion.phase=b.wing.phase
        c=ControlPacket(muscle_activations={n:l['activation'] for n,l in b.legs.items()},forward_drive=0.,wing_activation=b.last_command.wing,antennae_activation=b.last_command.antenna)
        m.update_visuals(c,v(0,0,0),0.,leg_points={n:l['points'] for n,l in b.legs.items()})
        for n,(x,z,r) in engine.experiment.bodies(engine.tick).items(): self.objects[n].pos=v(x,r if n!='food' else .05,z)
        self.text.text=f"{engine.experiment.name} | t={engine.tick*engine.experiment.dt:.2f} | hunger={b.internal['hunger']:.2f} | contacts={len(b.contacts)}"
