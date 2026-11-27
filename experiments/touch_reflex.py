"""Run left, right and sham contact trials with a stationary fly at onset."""
import argparse
from dataclasses import asdict
import json

from nervous_system.adapters.mcf import MCFTouchController, TouchController
from research.body import Body
from research.engine import Engine
from research.world import Arena, Experiment


def trial(side, controller_type, actuator='legacy', phase=0.):
    body = Body(initial_phase_rad=phase)
    obstacle = (4, 4, 0.08)
    if side != "sham":
        foot = body.legs[f"front_{side}"]["points"]["foot"]
        obstacle = (float(body.pos[0] + foot[0]), float(body.pos[1] + foot[2]), 0.08)
    exp = Experiment(name=f"touch-{side}", arena=Arena(obstacle=obstacle, moving_body=False),
                     steps=50, leg_actuator=actuator, initial_phase_rad=phase)
    engine = Engine(exp, controller_type())
    peak_tension = 0.
    first_contact = None
    first_command = None
    for row in engine.run():
        peak_tension = max(peak_tension, *(leg['tension'] for leg in engine.body.legs.values()))
        if first_contact is None and any(row["observation"][f"tactile.front_{s}.object_load"] > 0.01
                                         for s in ("left", "right")):
            first_contact = row["tick"]
        if first_command is None and any(row["command"]["rate"].values()):
            first_command = row["tick"]
    return dict(condition=side, experiment=asdict(exp), first_contact_tick=first_contact,
                peak_leg_tension=peak_tension, first_command_tick=first_command, final_yaw_rad=engine.body.yaw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=["reference", "mcf"], default="reference")
    args = parser.parse_args()
    ctrl = MCFTouchController if args.controller == "mcf" else TouchController
    for side in ("left", "right", "sham"):
        print(json.dumps({"controller": args.controller, **trial(side, ctrl)}))
