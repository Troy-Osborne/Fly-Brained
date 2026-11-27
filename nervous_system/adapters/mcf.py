"""Convert delivered front-foot contacts into circuit inputs and signed leg commands."""
from research.interface import LEG_NAMES, MotorCommand

SIDES = ("left", "right")


class TouchController:
    """One circuit tick per observation. Input/output order is left, right."""
    def relay(self, contacts):
        return contacts

    def step(self, observation):
        contacts = []
        for side in SIDES:
            channel = f"tactile.front_{side}.object_load"
            contacts.append(int(observation.channels.get(channel, 0.0) > 0.01))
        left, right = self.relay(contacts)
        if left and right:
            rates = dict.fromkeys(LEG_NAMES, -1)
        else:
            turn = int(left) - int(right)
            rates = {leg: turn if leg.endswith("_left") else -turn for leg in LEG_NAMES}
        return MotorCommand(rate=rates, effort={leg: abs(rate) for leg, rate in rates.items()})


class MCFTouchController(TouchController):
    """Requires an independently installed MCF checkout on the Python path."""
    def __init__(self):
        from mcf_parts import relay
        self.circuit = relay(2)

    def relay(self, contacts):
        return self.circuit.step({"in": contacts})["out"]
