import unittest
from dataclasses import replace
import numpy as np
from research.world import Experiment, Arena, Event, scenario
from research.engine import Engine
from research.controllers import Idle, OpenLoop, Reflex, Replay
from research.interface import MotorCommand, LEG_NAMES

class Predict:
    def step(self,obs):
        return MotorCommand(expected={'pressure.pa':101325.,'nonexistent':1.})

class ResearchTests(unittest.TestCase):
    def test_idle_has_no_automatic_gait_or_displacement(self):
        e=Engine(Experiment(steps=100,arena=Arena(moving_body=False)),Idle())
        initial=e.body.pos.copy(); list(e.run())
        np.testing.assert_array_equal(initial,e.body.pos)
        np.testing.assert_array_equal(e.body.phase,np.zeros(6))
        np.testing.assert_array_equal(e.body.activation,np.zeros(6))
    def test_seed_and_command_replay(self):
        exp=replace(scenario('wind',steps=50),noise=.002)
        a=list(Engine(exp,Reflex()).run())
        self.assertEqual(a,list(Engine(exp,Reflex()).run()))
        self.assertEqual(a,list(Engine(exp,Replay([r['command'] for r in a])).run()))
        self.assertNotEqual(a,list(Engine(replace(exp,seed=1),Reflex()).run()))
    def test_prediction_alignment_and_missing(self):
        exp=Experiment(steps=5,events=(Event(2,3,'pressure','',101300.),Event(3,4,'mask','pressure',1.)))
        rows=list(Engine(exp,Predict()).run())
        self.assertEqual(rows[0]['prediction_error'],{})
        self.assertEqual(rows[1]['prediction_error']['pressure.pa'],0.)
        self.assertEqual(rows[2]['prediction_error']['pressure.pa'],-25.)
        self.assertNotIn('pressure.pa',rows[3]['prediction_error'])
        self.assertIn('pressure.pa',rows[3]['prediction_unavailable'])
        self.assertIn('nonexistent',rows[4]['prediction_unavailable'])
    def test_mask_does_not_leak_raw_vision(self):
        e=Engine(scenario('blind',steps=9),OpenLoop())
        for row in e.run():
            masked=3<=row['tick']<6
            self.assertEqual('vision.left' in e.last_streams,not masked)
            self.assertEqual('vision.left.brightness' in row['observation'],not masked)
            self.assertIn('vision.left.brightness',row['truth'])
    def test_motor_loss_changes_motion_and_recovers(self):
        exp=Experiment(steps=180,arena=Arena(moving_body=False),events=(Event(20,120,'motor_gain','front_left',0.),))
        e=Engine(exp,OpenLoop()); rows=list(e.run())
        self.assertEqual(rows[19]['state_after']['motor_gain'][0],1.)
        self.assertEqual(rows[20]['state_after']['motor_gain'][0],0.)
        self.assertEqual(rows[120]['state_after']['motor_gain'][0],1.)
        base=Engine(replace(exp,events=()),OpenLoop()); list(base.run())
        self.assertGreater(np.linalg.norm(e.body.pos-base.body.pos),.05)
    def test_stream_shapes_and_read_only(self):
        e=Engine(Experiment(steps=1)); e.step(); s=e.last_streams
        self.assertEqual(s['audio'].shape,(80,2)); self.assertEqual(s['vision.left'].shape,(8,48))
        self.assertFalse(s['audio'].flags.writeable)
    def test_motion_laterality(self):
        e=Engine(scenario('motor',steps=60),OpenLoop())
        for row in e.run():
            for n,leg in e.body.legs.items():
                side=-1 if n.endswith('left') else 1
                self.assertTrue(all(side*p[2]>0 for p in leg['points'].values()))
    def test_custom_internal_channel(self):
        e=Engine(Experiment(steps=3,custom={'test':.2},events=(Event(1,2,'custom','test',.9),)))
        self.assertEqual([r['observation']['intero.custom.test'] for r in e.run()],[.2,.9,.2])
    def test_invalid_motor_values(self):
        with self.assertRaises(ValueError): MotorCommand(rate={'front_left':float('nan')}).checked()
        with self.assertRaises(ValueError): MotorCommand(effort={'fake':1.}).checked()
    def test_food_requires_command_and_contact(self):
        exp=Experiment(steps=20,arena=Arena(food=(-2.8,0.,.5),moving_body=False))
        idle=Engine(exp,Idle()); list(idle.run()); self.assertEqual(idle.body.food_consumed,0.)
        class Feed:
            def step(self,obs): return MotorCommand(feed=1.)
        active=Engine(exp,Feed()); list(active.run()); self.assertGreater(active.body.food_consumed,0.)
        self.assertLess(active.body.internal['hunger'],idle.body.internal['hunger'])
    def test_wind_sensor_local_frame(self):
        e=Engine(Experiment(steps=1,arena=Arena(wind=(0.,2.))))
        e.body.yaw=np.pi/2; row=e.step(); self.assertAlmostEqual(row['observation']['airflow.forward'],-2.)
    def test_external_streams_obey_ablation(self):
        def source(k,t,dt): return {'vision.left':np.ones((4,7)), 'audio':np.ones((80,2))*.1}
        exp=Experiment(steps=2,events=(Event(1,2,'mask','vision',1.),))
        e=Engine(exp,stream_source=source); row=e.step()
        self.assertEqual(row['observation']['vision.left.brightness'],1.)
        self.assertEqual(e.last_streams['vision.left'].shape,(4,7))
        self.assertAlmostEqual(row['observation']['audio.left.rms'],.1,places=6)
        e.step(); self.assertNotIn('vision.left',e.last_streams)
    def test_all_scenarios_finite(self):
        import json
        for name in ('baseline','wind','motor','blind','deaf','touch','conflict','obstacle','temperature','pressure','hunger'):
            for row in Engine(scenario(name,steps=12),Reflex()).run(): json.dumps(row,allow_nan=False)

if __name__=='__main__': unittest.main()
