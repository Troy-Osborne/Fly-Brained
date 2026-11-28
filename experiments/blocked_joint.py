"""Run matched trials; optionally save a trace or display it in VPython."""
import argparse
import json
from pathlib import Path

from research.joint_bench import DT, JointState, advance, tensions

RELEASE = 400
END = 800
CONDITIONS = ('free', 'compliant', 'rigid-held', 'rigid-release')


def simulate(initial=0., effort=.8, step=DT):
    """All times are seconds. The default pulse lasts from 0.1 to 0.55 s."""
    stride = round(.001 / step)
    if stride < 1 or abs(stride * step - .001) > 1e-12:
        raise ValueError('step must divide the recorded millisecond interval')
    stop = initial + .18
    traces = {}
    for condition in CONDITIONS:
        state = JointState(angle=initial)
        rows = []
        for ms in range(END + 1):
            contact = condition.split('-')[0]
            # Row t records the preceding interval; the first released interval ends at 401 ms.
            if condition == 'rigid-release' and ms > RELEASE:
                contact = 'free'
            command = effort if 100 < ms <= 550 else 0.
            reaction = 0.
            if ms:
                for _ in range(stride):
                    state, load = advance(state, command, stop, contact, step)
                    reaction += load / stride
            forward, reverse = tensions(state)
            rows.append(dict(t=ms/1000, angle=state.angle, velocity=state.velocity,
                             command=command, activation=state.activation,
                             forward=forward, reverse=reverse, reaction=reaction,
                             obstacle=contact != 'free'))
        traces[condition] = rows
    return dict(schema=1, step_s=step, stop_rad=stop, release_s=RELEASE/1000, trials=traces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, help='save a new JSON trace')
    parser.add_argument('--replay', type=Path, help='read a previously saved trace')
    parser.add_argument('--view', action='store_true', help='open the interactive 3D view')
    args = parser.parse_args()
    trace = json.loads(args.replay.read_text()) if args.replay else simulate()
    if trace['schema'] != 1:
        parser.error('unsupported trace version')
    if args.out:
        with args.out.open('x') as f:
            json.dump(trace, f, allow_nan=False)
    print('At 400 ms: angle (rad), forward tendon tension, reaction torque (model units)')
    for name, rows in trace['trials'].items():
        at_stop = rows[RELEASE]
        print(f"{name:14} {at_stop['angle']:8.4f} {at_stop['forward']:8.4f} {at_stop['reaction']:8.4f}")
    if args.view:
        from research.joint_view import show
        show(trace)


if __name__ == '__main__':
    main()
