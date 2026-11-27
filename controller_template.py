"""Template for replacing DemoController with a connectome-backed controller.

This file intentionally does not pretend to simulate the Male CNS connectome.
It defines the small adapter seam where a neural runtime can be connected.
"""
from __future__ import annotations

from typing import Mapping

from drosophila_vpython import ControlPacket, LEG_NAMES


class ConnectomeRuntime:
    """Replace methods with calls into your chosen neuron/synapse simulator."""

    def encode_sensory(self, sensory: Mapping[str, float], dt: float) -> None:
        # TODO: map receptor-level variables to sensory-neuron input currents,
        # spikes, or conductances using verified Male CNS annotations.
        raise NotImplementedError

    def advance(self, dt: float) -> None:
        # TODO: advance the selected connectome subnetwork.
        raise NotImplementedError

    def read_motor_outputs(self) -> Mapping[str, float]:
        # TODO: return calibrated motor-neuron / muscle-target activations.
        raise NotImplementedError


class ConnectomeController:
    """Controller adapter. Supply an implemented ConnectomeRuntime instance."""

    def __init__(self, runtime: ConnectomeRuntime):
        self.runtime = runtime

    def step(self, sensory: Mapping[str, float], prediction_errors: Mapping[str, float], dt: float) -> ControlPacket:
        self.runtime.encode_sensory(sensory, dt)
        # Prediction errors are kept available to a higher-level experimental
        # pathway; do not silently inject them into arbitrary neurons.
        self.runtime.advance(dt)
        output = self.runtime.read_motor_outputs()
        activations = {name: float(output.get(name, 0.0)) for name in LEG_NAMES}
        return ControlPacket(
            muscle_activations=activations,
            forward_drive=sum(activations.values()) / len(activations),
            steering=(activations["front_left"] + activations["middle_left"] + activations["hind_left"])
                     - (activations["front_right"] + activations["middle_right"] + activations["hind_right"]),
            wing_activation=float(output.get("wing_activation", 0.0)),
            antennae_activation=float(output.get("antennae_activation", 0.5)),
            expected_sensory=dict(output.get("expected_sensory", {})) if isinstance(output.get("expected_sensory", {}), dict) else {},
            metadata={"controller": "ConnectomeController"},
        )
