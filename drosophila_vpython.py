#!/usr/bin/env python3
"""Drosophila embodied-body VPython prototype.

A self-contained, interactive sensorimotor sandbox for embodied-cognition research.
This is a kinematic / simplified-dynamics prototype, not a validated anatomical
or connectome simulator.  A custom controller can be supplied by implementing
Controller.step(sensory, prediction_errors, dt) -> ControlPacket.

Run:
    python -m pip install -r requirements.txt
    python drosophila_vpython.py

Optional live inputs:
    python -m pip install opencv-python sounddevice
    python drosophila_vpython.py --camera --microphone

Optional recording:
    python drosophila_vpython.py --record run.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol

import numpy as np

from templates import DEFAULT_FLY_TEMPLATES, FlyAnatomyTemplates
from templates.legs import LegAssembly
from templates.leg_geometry import SIDE_LEFT, SIDE_RIGHT
from templates.wings import WingAssembly
from templates.wing_motion import WingMotionModel

from vpython import (
    arrow,
    box,
    button,
    canvas,
    checkbox,
    color,
    curve,
    cylinder,
    ellipsoid,
    label,
    mag,
    norm,
    rate,
    rotate,
    slider,
    sphere,
    vector,
    wtext,
)


# -----------------------------------------------------------------------------
# Small utility functions
# -----------------------------------------------------------------------------

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def safe_unit(v, fallback=None):
    if mag(v) < 1e-9:
        return fallback if fallback is not None else vector(1, 0, 0)
    return norm(v)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def smooth_step(current: float, target: float, gain: float, dt: float) -> float:
    return current + (target - current) * (1.0 - math.exp(-gain * dt))


# -----------------------------------------------------------------------------
# Stream adapters. External systems can call push_audio() / push_frame().
# -----------------------------------------------------------------------------

class MultichannelAudioInput:
    """Thread-safe latest-window audio feature extractor.

    Accepts samples shaped (frames, channels), or a mono 1-D array. The sensor
    publishes per-channel RMS and zero-crossing rate. It does not retain audio
    indefinitely.
    """

    def __init__(self, window_seconds: float = 0.20):
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._samples = np.zeros((0, 2), dtype=np.float32)
        self.sample_rate = 48000
        self.channel_count = 2

    def push_audio(self, samples: np.ndarray, sample_rate: int) -> None:
        data = np.asarray(samples, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        if data.ndim != 2 or data.shape[0] == 0:
            return
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        with self._lock:
            self.sample_rate = int(sample_rate)
            self.channel_count = int(data.shape[1])
            max_frames = max(128, int(self.sample_rate * self.window_seconds))
            if self._samples.shape[1] != data.shape[1]:
                self._samples = np.zeros((0, data.shape[1]), dtype=np.float32)
            self._samples = np.concatenate((self._samples, data), axis=0)[-max_frames:]

    def latest_window(self) -> np.ndarray:
        """Return a copy of the latest (frames, channels) audio window."""
        with self._lock:
            return self._samples.copy()

    def features(self) -> Dict[str, float]:
        with self._lock:
            data = self._samples.copy()
        result: Dict[str, float] = {"audio.channel_count": float(self.channel_count)}
        if data.size == 0:
            for ch in range(max(1, self.channel_count)):
                result[f"audio.ch{ch}.rms"] = 0.0
                result[f"audio.ch{ch}.zcr"] = 0.0
            result["audio.rms"] = 0.0
            return result
        for ch in range(data.shape[1]):
            channel = data[:, ch]
            rms = float(np.sqrt(np.mean(channel * channel)))
            zcr = float(np.mean(np.abs(np.diff(np.signbit(channel))))) if len(channel) > 1 else 0.0
            result[f"audio.ch{ch}.rms"] = rms
            result[f"audio.ch{ch}.zcr"] = zcr
        result["audio.rms"] = float(np.mean([result[f"audio.ch{ch}.rms"] for ch in range(data.shape[1])]))
        return result


class VisualStreamInput:
    """Thread-safe image-stream adapter with simple luminance / motion features."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._raw_frame: Optional[np.ndarray] = None
        self._previous: Optional[np.ndarray] = None
        self.timestamp = 0.0

    def push_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        raw = np.asarray(frame)
        if raw.ndim not in (2, 3) or raw.size == 0:
            return
        raw = np.nan_to_num(raw).copy()
        if raw.ndim == 3:
            gray = raw[..., :3].astype(np.float32).mean(axis=2)
        else:
            gray = raw.astype(np.float32)
        gray = gray[:: max(1, gray.shape[0] // 48), :: max(1, gray.shape[1] // 64)]
        with self._lock:
            self._previous = self._frame
            self._frame = gray
            self._raw_frame = raw
            self.timestamp = time.time() if timestamp is None else float(timestamp)

    def latest_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the most recently supplied full-resolution image."""
        with self._lock:
            return None if self._raw_frame is None else self._raw_frame.copy()

    def features(self) -> Dict[str, float]:
        with self._lock:
            current = None if self._frame is None else self._frame.copy()
            previous = None if self._previous is None else self._previous.copy()
        if current is None:
            return {
                "vision.frame_present": 0.0,
                "vision.mean_luma": 0.0,
                "vision.contrast": 0.0,
                "vision.motion": 0.0,
            }
        scale = 255.0 if float(np.nanmax(current)) > 1.5 else 1.0
        normed = current / scale
        motion = 0.0
        if previous is not None and previous.shape == current.shape:
            motion = float(np.mean(np.abs(current - previous)) / scale)
        return {
            "vision.frame_present": 1.0,
            "vision.mean_luma": float(np.mean(normed)),
            "vision.contrast": float(np.std(normed)),
            "vision.motion": motion,
        }


class OptionalLiveInputs:
    """Optional OpenCV camera and sounddevice microphone adapters."""

    def __init__(self, vision: VisualStreamInput, audio: MultichannelAudioInput):
        self.vision = vision
        self.audio = audio
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._streams: list[Any] = []

    def start_camera(self, device: int = 0) -> None:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Camera input requires: pip install opencv-python") from exc

        def worker():
            cap = cv2.VideoCapture(device)
            try:
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if ok:
                        self.vision.push_frame(frame)
                    else:
                        time.sleep(0.05)
            finally:
                cap.release()

        thread = threading.Thread(target=worker, daemon=True, name="fly-camera")
        thread.start()
        self._threads.append(thread)

    def start_microphone(self, channels: int = 2, sample_rate: int = 48000) -> None:
        try:
            import sounddevice as sd  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Microphone input requires: pip install sounddevice") from exc

        def callback(indata, frames, timing, status):
            if not self._stop.is_set():
                self.audio.push_audio(indata.copy(), sample_rate)

        stream = sd.InputStream(
            channels=channels,
            samplerate=sample_rate,
            callback=callback,
            dtype="float32",
            blocksize=1024,
        )
        stream.start()
        self._streams.append(stream)

    def stop(self) -> None:
        self._stop.set()
        for stream in self._streams:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Controller contract and a deliberately simple demonstration controller
# -----------------------------------------------------------------------------

class SensoryPacket(dict):
    """Numeric sensory channels plus access to the corresponding raw streams.

    Scalar features are dictionary entries. Raw audio and image buffers are
    available on ``packet.streams`` and are deliberately excluded from JSONL
    telemetry unless a researcher explicitly records them separately.
    """

    def __init__(self, values: Mapping[str, float], streams: Optional[Dict[str, Any]] = None):
        super().__init__(values)
        self.streams = streams or {}

@dataclass
class ControlPacket:
    """One control update from a controller to the body.

    muscle_activations keys are front_left, front_right, middle_left,
    middle_right, hind_left, hind_right. Values are clamped to [0, 1].
    """

    muscle_activations: Dict[str, float] = field(default_factory=dict)
    forward_drive: float = 0.55
    steering: float = 0.0
    wing_activation: float = 0.0
    antennae_activation: float = 0.5
    expected_sensory: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Controller(Protocol):
    def step(
        self,
        sensory: Mapping[str, float],
        prediction_errors: Mapping[str, float],
        dt: float,
    ) -> ControlPacket:
        """Return desired motor activations and predictions for the next step."""
        ...


class DemoController:
    """A replaceable heuristic gait controller used to prove the body interface."""

    LEG_KEYS = (
        "front_left", "front_right", "middle_left", "middle_right", "hind_left", "hind_right"
    )

    def __init__(self):
        self.drive = 0.48
        self.steering = 0.0
        self._last_command = ControlPacket()

    def step(self, sensory: Mapping[str, float], prediction_errors: Mapping[str, float], dt: float) -> ControlPacket:
        obstacle = as_float(sensory.get("world.obstacle_proximity", 1.0), 1.0)
        contact = as_float(sensory.get("tactile.contact_count", 0.0))
        wind_side = as_float(sensory.get("airflow.lateral", 0.0))
        nociception = as_float(sensory.get("intero.nociception", 0.0))

        # Retreat / slow down when an obstacle or unexpected contact is close.
        target_drive = 0.52
        if obstacle < 0.8 or contact > 0:
            target_drive = 0.12
        if nociception > 0.55:
            target_drive = 0.06
        self.drive = smooth_step(self.drive, target_drive, 1.8, dt)
        self.steering = clamp(-0.18 * wind_side, -0.25, 0.25)

        activations = {}
        for key in self.LEG_KEYS:
            # Front legs respond slightly more to proximity; hind legs provide push.
            value = self.drive
            if key.startswith("front"):
                value *= 0.9 if obstacle < 1.0 else 1.0
            if key.startswith("hind"):
                value = min(1.0, value * 1.12)
            activations[key] = clamp(value + (0.04 if key.endswith("left") else -0.04) * self.steering, 0.0, 1.0)

        # A simple wing response to arousal / high perturbation; not a flight model.
        arousal = as_float(sensory.get("intero.octopamine_proxy", 0.2), 0.2)
        wing = clamp((arousal - 0.48) * 1.2 + as_float(sensory.get("airflow.speed", 0.0)) * 0.04, 0.0, 0.75)

        # Lightweight one-step predictions. The comparator reports raw deltas.
        speed = as_float(sensory.get("proprio.forward_speed", 0.0))
        predicted_speed = speed + (self.drive * 0.85 - speed) * min(1.0, dt * 2.0)
        predicted = {
            "proprio.forward_speed": predicted_speed,
            "airflow.speed": as_float(sensory.get("airflow.speed", 0.0)),
            "pressure.pa": as_float(sensory.get("pressure.pa", 101325.0)),
        }
        packet = ControlPacket(
            muscle_activations=activations,
            forward_drive=self.drive,
            steering=self.steering,
            wing_activation=wing,
            antennae_activation=clamp(0.55 + obstacle * 0.15, 0.2, 1.0),
            expected_sensory=predicted,
            metadata={"controller": "DemoController"},
        )
        self._last_command = packet
        return packet


class PredictionComparator:
    """Preserves per-channel actual-minus-expected deltas."""

    def __init__(self):
        self.pending_expected: Dict[str, float] = {}
        self.last_errors: Dict[str, float] = {}
        self.last_normalised_errors: Dict[str, float] = {}
        self.rms_error = 0.0
        # Characteristic scales make the aggregate dimensionless. Raw errors
        # remain untouched and are what the controller receives.
        self.error_scales = {
            "proprio.forward_speed": 0.85,
            "airflow.speed": 5.0,
            "pressure.pa": 30.0,
        }

    def compare(self, actual: Mapping[str, float]) -> Dict[str, float]:
        errors: Dict[str, float] = {}
        for key, expected in self.pending_expected.items():
            if key in actual:
                errors[key] = as_float(actual[key]) - as_float(expected)
        self.last_errors = errors
        self.last_normalised_errors = {
            key: value / self.error_scales.get(key, max(1.0, abs(value)))
            for key, value in errors.items()
        }
        normed = list(self.last_normalised_errors.values())
        self.rms_error = math.sqrt(sum(v * v for v in normed) / max(1, len(normed)))
        return errors

    def set_expected(self, expected: Mapping[str, float]) -> None:
        self.pending_expected = {k: as_float(v) for k, v in expected.items()}


# -----------------------------------------------------------------------------
# Individual appendage assemblies are implemented in templates/ so they can
# be replaced without editing the simulation/controller code.
# -----------------------------------------------------------------------------

LEG_NAMES = (
    "front_left", "front_right", "middle_left", "middle_right", "hind_left", "hind_right"
)


# -----------------------------------------------------------------------------
# Main fly body
# -----------------------------------------------------------------------------

class DrosophilaBody:
    def __init__(self, templates: FlyAnatomyTemplates = DEFAULT_FLY_TEMPLATES,
                 wing_motion_model=None):
        self.templates = templates
        self.root = vector(-2.8, 1.18, 0.0)
        self.velocity = 0.0
        self.yaw = 0.0
        self.phase_time = 0.0
        self.stride = 0.26
        self.lift = 0.17
        self.max_forward_speed = 0.85
        self.wing_motion = wing_motion_model or WingMotionModel(templates.wing_motion)
        self.muscle_energy = 1.0
        self.injury = 0.0
        leg_specs = (
            ("front_left", 0, SIDE_LEFT), ("front_right", 0, SIDE_RIGHT),
            ("middle_left", 1, SIDE_LEFT), ("middle_right", 1, SIDE_RIGHT),
            ("hind_left", 2, SIDE_LEFT), ("hind_right", 2, SIDE_RIGHT),
        )
        self.legs = [
            LegAssembly(name, index, side, templates.legs, templates.leg_visual)
            for name, index, side in leg_specs
        ]

        # BODY PART TEMPLATES: these can be changed independently in templates/.
        thorax = templates.thorax
        head = templates.head
        eyes = templates.eyes
        antenna = templates.antenna
        haltere = templates.haltere
        nervous = templates.nervous_system

        self.thorax = ellipsoid(pos=self.root, length=thorax.length, height=thorax.height,
                                width=thorax.width, color=vector(*thorax.colour))
        self.head = ellipsoid(pos=self.root, length=head.length, height=head.height,
                              width=head.width, color=vector(*head.colour))
        self.abdomen = []
        for segment in templates.abdomen:
            self.abdomen.append(ellipsoid(
                pos=self.root, length=segment.length, height=segment.height,
                width=segment.width, color=vector(*segment.colour)
            ))

        self.eyes = [sphere(pos=self.root, radius=eyes.radius, color=vector(*eyes.colour))
                     for _ in (SIDE_LEFT, SIDE_RIGHT)]
        self.ommatidia = []
        for _eye_side in (SIDE_LEFT, SIDE_RIGHT):
            for _i in range(eyes.columns):
                for _j in range(eyes.rows):
                    self.ommatidia.append(sphere(
                        pos=self.root, radius=eyes.ommatidia_radius,
                        color=vector(*eyes.ommatidia_colour)
                    ))

        self.wings = [WingAssembly(side, templates.wings) for side in (SIDE_LEFT, SIDE_RIGHT)]
        self.halteres = []
        self.halter_stalks = []
        for side in (SIDE_LEFT, SIDE_RIGHT):
            self.halteres.append(sphere(
                pos=self.root, radius=haltere.knob_radius, color=vector(*haltere.knob_colour)
            ))
            self.halter_stalks.append(cylinder(
                pos=self.root, axis=vector(-0.2, 0, side * 0.3), radius=haltere.stalk_radius,
                color=vector(*haltere.stalk_colour)
            ))
        self.antennae = []
        self.antenna_clubs = []
        for side in (SIDE_LEFT, SIDE_RIGHT):
            self.antennae.append(cylinder(
                pos=self.root, axis=vector(0.25, 0.2, side * 0.1), radius=antenna.shaft_radius,
                color=vector(*antenna.shaft_colour)
            ))
            self.antenna_clubs.append(sphere(
                pos=self.root, radius=antenna.club_radius, color=vector(*antenna.club_colour)
            ))

        # Stylised neural display; anatomical placeholders are configured separately.
        nerve_color = vector(*nervous.nerve_colour)
        self.brain_ganglion = sphere(
            pos=self.root, radius=nervous.brain_radius, color=vector(*nervous.brain_colour), opacity=0.95
        )
        self.ventral_nerve_cord = cylinder(
            pos=self.root, axis=vector(-1.2, 0, 0), radius=nervous.cord_radius, color=nerve_color
        )
        self.segmental_ganglia = [
            sphere(pos=self.root, radius=nervous.ganglion_radius, color=vector(0.62, 0.83, 0.52))
            for _ in nervous.ganglion_x
        ]
        self.motor_nerves = [
            cylinder(pos=self.root, axis=vector(0, 0.1, 0), radius=nervous.motor_nerve_radius,
                     color=nerve_color) for _ in range(6)
        ]
        self.sensory_nerves = [
            cylinder(pos=self.root, axis=vector(0.1, 0, 0), radius=nervous.sensory_nerve_radius,
                     color=vector(*nervous.sensory_nerve_colour)) for _ in range(4)
        ]
        self.flight_muscles = [cylinder(
            pos=self.root, axis=vector(0.2, 0.1, 0), radius=thorax.flight_muscle_radius,
            color=vector(*thorax.flight_muscle_colour)
        ) for _ in thorax.flight_muscle_paths]
        self.leg_label = label(pos=self.root, text="", height=10, box=False, opacity=0, color=color.white)

    def local_to_world(self, local):
        return self.root + rotate(local, angle=self.yaw, axis=vector(0, 1, 0))

    def update_visuals(self, command: ControlPacket, wind: Any, dt: float, leg_points=None):
        self.phase_time += dt
        self.wing_motion.step(dt, command.wing_activation)
        self.thorax.pos = self.root
        self.thorax.axis = rotate(vector(1, 0, 0), angle=self.yaw, axis=vector(0, 1, 0))
        self.head.pos = self.local_to_world(vector(*self.templates.head.local_position))
        self.head.axis = rotate(vector(1, 0, 0), angle=self.yaw, axis=vector(0, 1, 0))
        for segment, abdomen in zip(self.templates.abdomen, self.abdomen):
            local = vector(segment.x, segment.y, 0)
            abdomen.pos = self.local_to_world(local)
            abdomen.axis = rotate(vector(-1, 0, 0), angle=self.yaw, axis=vector(0, 1, 0))
        eye_template = self.templates.eyes
        for idx, side in enumerate((SIDE_LEFT, SIDE_RIGHT)):
            centre = vector(*eye_template.centre)
            centre.z = side * abs(eye_template.centre[2])
            self.eyes[idx].pos = self.local_to_world(centre)
        # Each compound eye gets its own patch of ommatidia.
        n = 0
        for side in (SIDE_LEFT, SIDE_RIGHT):
            centre = vector(*eye_template.centre)
            centre.z = side * abs(eye_template.centre[2])
            for i in range(eye_template.columns):
                for j in range(eye_template.rows):
                    local = centre + vector(
                        eye_template.column_spacing * (i - (eye_template.columns - 1) / 2),
                        eye_template.row_spacing * (j - (eye_template.rows - 1) / 2),
                        side * eye_template.lateral_offset,
                    )
                    self.ommatidia[n].pos = self.local_to_world(local)
                    n += 1
        for idx, side in enumerate((SIDE_LEFT, SIDE_RIGHT)):
            self.wings[idx].update(self.root, self.yaw, self.wing_motion,
                                   command.wing_activation, self.local_to_world)

            haltere = self.templates.haltere
            hbase_local = vector(haltere.base[0], haltere.base[1], side * abs(haltere.base[2]))
            htip_local = vector(haltere.tip[0], haltere.tip[1] + 0.07 * math.sin(self.wing_motion.phase * 0.42),
                                side * abs(haltere.tip[2]))
            hbase = self.local_to_world(hbase_local)
            htip = self.local_to_world(htip_local)
            self.halter_stalks[idx].pos = hbase
            self.halter_stalks[idx].axis = htip - hbase
            self.halteres[idx].pos = htip

            antenna = self.templates.antenna
            abase_local = vector(antenna.base[0], antenna.base[1], side * abs(antenna.base[2]))
            aend_local = vector(
                antenna.end_x + 0.10 * command.antennae_activation,
                antenna.end_y + 0.06 * math.sin(self.phase_time * 3),
                side * (antenna.lateral_end + 0.04 * math.sin(self.phase_time * 2)),
            )
            abase = self.local_to_world(abase_local)
            aend = self.local_to_world(aend_local)
            self.antennae[idx].pos = abase
            self.antennae[idx].axis = aend - abase
            self.antenna_clubs[idx].pos = aend

        brain = self.local_to_world(vector(0.46, -0.035, 0))
        self.brain_ganglion.pos = brain
        nervous = self.templates.nervous_system
        cord_front = self.local_to_world(vector(nervous.cord_front_x, nervous.cord_y, 0))
        cord_back = self.local_to_world(vector(nervous.cord_back_x, nervous.cord_y, 0))
        self.ventral_nerve_cord.pos = cord_front
        self.ventral_nerve_cord.axis = cord_back - cord_front
        for ganglion, gx in zip(self.segmental_ganglia, nervous.ganglion_x):
            ganglion.pos = self.local_to_world(vector(gx, nervous.cord_y, 0))

        # Flight-muscle fibres cross the thorax in antagonistic directions.
        thorax_template = self.templates.thorax
        flight_level = 0.35 + 0.65 * command.wing_activation
        for muscle, (p0_values, delta_values) in zip(
            self.flight_muscles, thorax_template.flight_muscle_paths
        ):
            muscle.pos = self.local_to_world(vector(*p0_values))
            muscle.axis = rotate(vector(*delta_values), angle=self.yaw, axis=vector(0, 1, 0))
            muscle.radius = 0.018 + 0.028 * flight_level

        foot_positions = []
        for leg in self.legs:
            activation = command.muscle_activations.get(leg.name, 0.45)
            foot_positions.append(leg.update(self.root, self.yaw, self.phase_time,
                                             activation, self.stride, self.lift,
                                             points=None if leg_points is None else leg_points[leg.name]))
        for index, leg in enumerate(self.legs):
            cord_node = self.local_to_world(vector(leg.attach.x, nervous.cord_y, 0))
            self.motor_nerves[index].pos = cord_node
            self.motor_nerves[index].axis = leg.hip.pos - cord_node
        left_eye = self.eyes[0].pos
        right_eye = self.eyes[1].pos
        left_antenna = self.antenna_clubs[0].pos
        right_antenna = self.antenna_clubs[1].pos
        for nerve, endpoint in zip(self.sensory_nerves, (left_eye, right_eye, left_antenna, right_antenna)):
            nerve.pos = endpoint
            nerve.axis = brain - endpoint
        return foot_positions

    def step_dynamics(self, command: ControlPacket, wind_vector, obstacle_x: float, dt: float):
        drive = clamp(command.forward_drive, 0.0, 1.0)
        # Approximate body translation from a tripod gait, drag and airflow.
        target_speed = drive * self.max_forward_speed
        wind_drag = 0.035 * wind_vector.x
        target_speed += wind_drag
        front_clearance = obstacle_x - (self.root.x + 1.05)
        obstacle_contact = front_clearance < 0.04 and abs(self.root.z) < 0.68
        if obstacle_contact:
            target_speed = min(target_speed, 0.02)
        target_speed = clamp(target_speed, -0.25, self.max_forward_speed)
        self.velocity = smooth_step(self.velocity, target_speed, 2.4, dt)
        self.root.x += self.velocity * dt
        if obstacle_contact:
            self.root.x = min(self.root.x, obstacle_x - 1.05)
            self.velocity = min(self.velocity, 0.02)
        self.root.z += (command.steering * 0.6 + wind_vector.z * 0.025) * dt
        self.root.z = clamp(self.root.z, -1.4, 1.4)
        self.yaw = clamp(self.yaw + (command.steering * 0.12 + wind_vector.z * 0.003) * dt, -0.55, 0.55)
        self.root.y = 1.18 + 0.025 * math.sin(self.phase_time * 16.0) - min(0.10, abs(wind_vector.y) * 0.01)
        return obstacle_contact


# -----------------------------------------------------------------------------
# Environment and sensory transduction
# -----------------------------------------------------------------------------

class Environment:
    def __init__(self):
        self.wind = vector(1.2, 0.0, 0.0)
        self.pressure_base = 101325.0
        self.obstacle_x = 3.6
        self.obstacle = box(pos=vector(self.obstacle_x, 0.43, 0), size=vector(0.35, 0.86, 1.0),
                            color=vector(0.55, 0.58, 0.62), opacity=0.72)
        self.moving_agent = sphere(pos=vector(1.3, 0.36, 1.8), radius=0.25,
                                   color=vector(0.26, 0.68, 0.83), make_trail=True, retain=60)
        self.agent_head = sphere(pos=self.moving_agent.pos + vector(0.18, 0.12, 0), radius=0.13,
                                 color=vector(0.36, 0.74, 0.86))
        self.wind_particles = [sphere(pos=vector(-3 + i * 0.75, 0.45 + 0.15 * (i % 3), -1.7),
                                      radius=0.025, color=vector(0.45, 0.74, 0.88), opacity=0.50)
                               for i in range(12)]
        self.wind_arrows = [arrow(pos=vector(-2.7, 2.0, z), axis=vector(0.5, 0, 0),
                                  shaftwidth=0.025, color=vector(0.36, 0.68, 0.83), opacity=0.55)
                            for z in (-1.4, 0.0, 1.4)]
        self.ground = box(pos=vector(0, -0.12, 0), size=vector(40, 0.18, 12),
                          color=vector(0.20, 0.25, 0.22))
        self.ground_grid = []
        for x in range(-12, 17, 2):
            self.ground_grid.append(curve(pos=[vector(x, -0.025, -5), vector(x, -0.025, 5)],
                                          color=vector(0.29, 0.34, 0.31), radius=0.004))
        for z in range(-5, 6, 2):
            self.ground_grid.append(curve(pos=[vector(-12, -0.025, z), vector(16, -0.025, z)],
                                          color=vector(0.29, 0.34, 0.31), radius=0.004))

    def update(self, t: float, dt: float):
        self.moving_agent.pos = vector(1.1 + 0.9 * math.sin(t * 0.43),
                                       0.36 + 0.10 * math.sin(t * 1.7),
                                       1.8 * math.cos(t * 0.43))
        self.agent_head.pos = self.moving_agent.pos + vector(0.18, 0.12, 0)
        for i, particle in enumerate(self.wind_particles):
            particle.pos.x += self.wind.x * dt * (0.8 + 0.025 * i)
            particle.pos.z += self.wind.z * dt * 0.08
            if particle.pos.x > 5.0:
                particle.pos.x = -5.0
            if particle.pos.x < -5.0:
                particle.pos.x = 5.0
        for a in self.wind_arrows:
            a.axis = vector(0.38 + 0.12 * self.wind.x, self.wind.y * 0.04, self.wind.z * 0.08)

    def obstacle_proximity(self, body: DrosophilaBody) -> float:
        return max(0.0, self.obstacle_x - (body.root.x + 1.05))


class SensorSuite:
    def __init__(self, audio: MultichannelAudioInput, vision: VisualStreamInput):
        self.audio = audio
        self.vision = vision
        self.hunger = 0.30
        self.glucose = 0.78
        self.octopamine = 0.22
        self.nociception = 0.0
        self.body_temperature = 25.0
        self.hydration = 0.88
        self.stress = 0.08
        self.fatigue = 0.05
        self.energy = 0.92
        self._last_tactile = 0.0
        self.custom_internal: Dict[str, float] = {}

    def set_custom_variable(self, name: str, value: float) -> None:
        """Register/update a custom scalar interoceptive variable in [0, 1]."""
        key = "_".join(str(name).strip().lower().split())
        if not key:
            raise ValueError("Custom internal variable needs a non-empty name")
        self.custom_internal[key] = clamp(value, 0.0, 1.0)

    def update_internal_state(self, dt: float, body: DrosophilaBody, contact: bool, wind_speed: float):
        self.hunger = clamp(self.hunger + 0.0015 * dt - 0.0005 * body.velocity * dt, 0, 1)
        self.glucose = clamp(self.glucose - 0.0012 * body.velocity * dt + 0.0002 * dt, 0, 1)
        self.energy = clamp(self.energy - 0.0008 * body.velocity * dt + 0.00012 * dt, 0, 1)
        self.fatigue = clamp(self.fatigue + 0.0015 * body.velocity * dt - 0.0008 * dt, 0, 1)
        self.stress = smooth_step(self.stress, 0.6 if contact else 0.08 + wind_speed * 0.015, 1.3, dt)
        self.octopamine = smooth_step(self.octopamine, 0.18 + 0.55 * self.stress + 0.18 * self.hunger, 0.8, dt)
        self.nociception = smooth_step(self.nociception, 0.75 if contact else 0.0, 3.0, dt)
        self.body_temperature = smooth_step(self.body_temperature, 25.0 + 0.7 * wind_speed, 0.1, dt)
        self.hydration = clamp(self.hydration - 0.00015 * dt, 0, 1)

    def sample(self, t: float, body: DrosophilaBody, env: Environment,
               feet, obstacle_contact: bool, agent_contact: bool = False) -> Dict[str, float]:
        contacts = sum(1 for leg in body.legs if leg.contact)
        tactile_force = sum(leg.contact_force for leg in body.legs) + (1.2 if obstacle_contact else 0.0)
        wind = env.wind
        wind_speed = mag(wind)
        pressure = env.pressure_base + 11.0 * math.sin(t * 0.38) + 0.5 * 1.20 * wind_speed * wind_speed
        pressure_gradient = 1.6 * wind.x + 0.3 * wind.z
        joint_angles = [leg.joint_angle for leg in body.legs]
        muscle_force = sum(leg.muscle_force for leg in body.legs)
        features: Dict[str, float] = {
            "proprio.forward_speed": body.velocity,
            "proprio.body_yaw": body.yaw,
            "proprio.joint_angle_mean": float(np.mean(joint_angles)),
            "proprio.joint_angle_std": float(np.std(joint_angles)),
            "proprio.muscle_force_sum": muscle_force,
            "proprio.leg_contact_fraction": contacts / 6.0,
            "airflow.speed": wind_speed,
            "airflow.forward": wind.x,
            "airflow.vertical": wind.y,
            "airflow.lateral": wind.z,
            "airflow.dynamic_pressure": 0.5 * 1.20 * wind_speed * wind_speed,
            "pressure.pa": pressure,
            "pressure.gradient_proxy": pressure_gradient,
            "tactile.contact_count": float(contacts + (1 if obstacle_contact else 0) + (1 if agent_contact else 0)),
            "tactile.force_proxy": tactile_force + (0.9 if agent_contact else 0.0),
            "tactile.other_body_contact": float(agent_contact),
            "world.obstacle_proximity": env.obstacle_proximity(body),
            "world.moving_body_distance": mag(env.moving_agent.pos - body.root),
            "intero.hunger": self.hunger,
            "intero.energy": self.energy,
            "intero.glucose_proxy": self.glucose,
            "intero.octopamine_proxy": self.octopamine,
            "intero.nociception": self.nociception,
            "intero.temperature_c": self.body_temperature,
            "intero.hydration": self.hydration,
            "intero.stress": self.stress,
            "intero.fatigue": self.fatigue,
        }
        features.update(self.audio.features())
        features.update(self.vision.features())
        for name, value in self.custom_internal.items():
            features[f"intero.custom.{name}"] = value
        # A simple scene-derived visual channel remains available without a camera.
        features["vision.scene_obstacle_bearing"] = math.atan2(
            env.obstacle.pos.z - body.root.z, env.obstacle.pos.x - body.root.x
        )
        features["vision.scene_obstacle_distance"] = env.obstacle_proximity(body)
        return SensoryPacket(features, streams={
            "audio": self.audio.latest_window(),
            "visual_frame": self.vision.latest_frame(),
        })


# -----------------------------------------------------------------------------
# Telemetry, user interface and simulation coordinator
# -----------------------------------------------------------------------------

class JsonlRecorder:
    def __init__(self, path: Optional[str]):
        self.path = Path(path) if path else None
        self.handle = self.path.open("a", encoding="utf-8") if self.path else None

    def write(self, record: Mapping[str, Any]):
        if self.handle:
            self.handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            self.handle.flush()

    def close(self):
        if self.handle:
            self.handle.close()


class FlySimulation:
    def __init__(self, args, body_templates: FlyAnatomyTemplates = DEFAULT_FLY_TEMPLATES,
                 wing_motion_model=None):
        self.args = args
        self.scene = canvas(
            title="Drosophila Embodied Body | VPython Sensorimotor Prototype",
            width=1180,
            height=760,
            background=vector(0.035, 0.055, 0.075),
            center=vector(0.0, 0.85, 0.0),
            forward=vector(-0.12, -0.12, -1.0),
            range=5.8,
        )
        self.scene.caption = (
            "\n<b>Controls:</b> wind slider changes airflow; forward slider overrides gait drive; "
            "press Space to pause, R to reset, W/S to alter airflow.\n"
            "<span style='color:#9db4c5'>Blue = environment / moving body · Orange = muscle · Gold = tendon · Teal = passive restraint</span>\n\n"
        )
        self.environment = Environment()
        self.body = DrosophilaBody(body_templates, wing_motion_model=wing_motion_model)
        self.audio = MultichannelAudioInput()
        self.vision = VisualStreamInput()
        self.sensors = SensorSuite(self.audio, self.vision)
        self.controller: Controller = DemoController()
        self.comparator = PredictionComparator()
        self.live_inputs = OptionalLiveInputs(self.vision, self.audio)
        self.recorder = JsonlRecorder(args.record)
        self.paused = False
        self.sim_time = 0.0
        self.last_sensory: Dict[str, float] = {}
        self.last_errors: Dict[str, float] = {}
        self.last_command = ControlPacket()
        self.obstacle_contact = False
        self.agent_contact = False
        self._make_controls()
        self._make_overlays()

        if args.camera:
            self.live_inputs.start_camera(args.camera_device)
        if args.microphone:
            self.live_inputs.start_microphone(channels=args.audio_channels)

    def _make_controls(self):
        self.scene.append_to_caption("\n<b>Environment</b>  Wind X ")
        self.wind_slider = slider(min=-4, max=4, value=self.environment.wind.x,
                                  bind=self._wind_changed)
        self.scene.append_to_caption("  Wind Z ")
        self.wind_z_slider = slider(min=-3, max=3, value=0, bind=self._wind_z_changed)
        self.scene.append_to_caption("  Forward drive ")
        self.drive_slider = slider(min=0, max=1, value=0.48, bind=self._drive_changed)
        self.manual_drive = checkbox(text="Manual drive", checked=False, bind=self._manual_changed)
        self.scene.append_to_caption("  ")
        button(text="Reset body", bind=self._reset)
        button(text="Pause / resume", bind=self._toggle_pause)
        self.scene.append_to_caption("\n")
        self.status_text = wtext(text="Starting simulation…")

    def _make_overlays(self):
        self.telemetry = label(pos=vector(-3.7, 3.15, 0), text="", height=10, box=True,
                               line=False, opacity=0.20, color=vector(0.84, 0.91, 0.95))
        self.error_label = label(pos=vector(3.0, 2.8, 0), text="", height=10, box=True,
                                 line=False, opacity=0.20, color=vector(1.0, 0.78, 0.45))
        self.wind_label = label(pos=vector(0, 2.8, 3.6), text="", height=10, box=False,
                                color=vector(0.55, 0.82, 0.95))

    def _wind_changed(self, control):
        self.environment.wind.x = float(control.value)

    def _wind_z_changed(self, control):
        self.environment.wind.z = float(control.value)

    def _drive_changed(self, control):
        # The value is consumed by _make_command() while Manual drive is enabled.
        pass

    def _manual_changed(self, control):
        # VPython widgets are kept enabled for broad version compatibility.
        pass

    def _toggle_pause(self, _):
        self.paused = not self.paused

    def _reset(self, _):
        self.body.root = vector(-2.8, 1.18, 0)
        self.body.velocity = 0.0
        self.body.yaw = 0.0
        self.sensors.nociception = 0.0
        self.obstacle_contact = False
        self.agent_contact = False

    def _key_handler(self, event):
        key = event.key.lower()
        if key == "space":
            self.paused = not self.paused
        elif key == "r":
            self._reset(None)
        elif key == "w":
            self.environment.wind.x = clamp(self.environment.wind.x + 0.25, -4, 4)
            self.wind_slider.value = self.environment.wind.x
        elif key == "s":
            self.environment.wind.x = clamp(self.environment.wind.x - 0.25, -4, 4)
            self.wind_slider.value = self.environment.wind.x

    def install_controller(self, controller: Controller):
        """Replace the demonstration controller with a user implementation."""
        self.controller = controller

    def push_audio(self, samples: np.ndarray, sample_rate: int = 48000):
        """Public input hook for external multichannel audio streams."""
        self.audio.push_audio(samples, sample_rate)

    def push_visual_frame(self, frame: np.ndarray, timestamp: Optional[float] = None):
        """Public input hook for external visual streams (H x W or H x W x C)."""
        self.vision.push_frame(frame, timestamp)

    def _make_command(self, sensory, errors, dt):
        command = self.controller.step(sensory, errors, dt)
        if self.manual_drive.checked:
            manual = float(self.drive_slider.value)
            command.forward_drive = manual
            command.muscle_activations = {name: manual for name in LEG_NAMES}
            command.metadata = {**command.metadata, "manual_drive_override": True}
        return command

    def step(self, dt: float):
        self.sim_time += dt
        self.environment.update(self.sim_time, dt)
        self.obstacle_contact = self.body.step_dynamics(
            self.last_command, self.environment.wind, self.environment.obstacle_x, dt
        )
        self.agent_contact = mag(self.environment.moving_agent.pos - self.body.root) < 1.10
        feet = self.body.update_visuals(self.last_command, self.environment.wind, dt)
        any_world_contact = self.obstacle_contact or self.agent_contact
        self.sensors.update_internal_state(dt, self.body, any_world_contact,
                                          mag(self.environment.wind))
        sensory = self.sensors.sample(self.sim_time, self.body, self.environment,
                                      feet, self.obstacle_contact, self.agent_contact)
        errors = self.comparator.compare(sensory)
        command = self._make_command(sensory, errors, dt)
        # Predictions from this command are compared with measurements next iteration.
        self.comparator.set_expected(command.expected_sensory)
        self.last_sensory = sensory
        self.last_errors = errors
        self.last_command = command
        self._update_telemetry(sensory, errors, command)
        self.recorder.write({
            "time": self.sim_time,
            "sensory": sensory,
            "prediction_error": errors,
            "prediction_error_rms_normalised": self.comparator.rms_error,
            "prediction_error_normalised": self.comparator.last_normalised_errors,
            "motor": {
                "muscles": command.muscle_activations,
                "forward_drive": command.forward_drive,
                "steering": command.steering,
                "wing_activation": command.wing_activation,
            },
        })

    def _update_telemetry(self, s, errors, command):
        self.telemetry.text = (
            f"BODY / INTERNAL STATE\n"
            f"forward speed     {s['proprio.forward_speed']:+.2f} m/s\n"
            f"body yaw          {s['proprio.body_yaw']:+.2f} rad\n"
            f"joint angle       {s['proprio.joint_angle_mean']:+.2f} rad\n"
            f"muscle force      {s['proprio.muscle_force_sum']:.2f} a.u.\n"
            f"contact events    {int(s['tactile.contact_count'])}\n"
            f"hunger            {s['intero.hunger']:.2f}\n"
            f"octopamine proxy  {s['intero.octopamine_proxy']:.2f}\n"
            f"nociception       {s['intero.nociception']:.2f}\n"
            f"temperature       {s['intero.temperature_c']:.1f} °C\n"
            f"fatigue           {s['intero.fatigue']:.2f}"
        )
        tracked = ("proprio.forward_speed", "airflow.speed", "pressure.pa")
        lines = ["PREDICTION / REALITY DELTAS"]
        for key in tracked:
            value = errors.get(key, 0.0)
            lines.append(f"{key.split('.')[-1]:18s} {value:+.4f}")
        lines.append(f"normalised RMS      {self.comparator.rms_error:.4f}")
        lines.append(f"audio RMS           {s.get('audio.rms', 0.0):.4f}")
        lines.append(f"visual motion       {s.get('vision.motion', 0.0):.4f}")
        lines.append(f"controller          {command.metadata.get('controller', 'custom')}")
        self.error_label.text = "\n".join(lines)
        self.wind_label.text = f"AIRFLOW  x={self.environment.wind.x:+.1f}  z={self.environment.wind.z:+.1f} m/s"
        self.status_text.text = ("  PAUSED" if self.paused else "  RUNNING") + (
            " | live camera" if self.args.camera else " | camera adapter ready"
        ) + (" | live microphone" if self.args.microphone else " | audio adapter ready")

    def run(self):
        self.scene.bind("keydown", self._key_handler)
        last = time.perf_counter()
        try:
            while True:
                rate(60)
                now = time.perf_counter()
                dt = clamp(now - last, 1.0 / 240.0, 1.0 / 20.0)
                last = now
                if not self.paused:
                    self.step(dt)
        except KeyboardInterrupt:
            pass
        finally:
            self.live_inputs.stop()
            self.recorder.close()


def parse_args():
    parser = argparse.ArgumentParser(description="VPython Drosophila sensorimotor body prototype")
    parser.add_argument("--camera", action="store_true", help="read frames from the default OpenCV camera")
    parser.add_argument("--camera-device", type=int, default=0)
    parser.add_argument("--microphone", action="store_true", help="read multichannel audio from sounddevice")
    parser.add_argument("--audio-channels", type=int, default=2)
    parser.add_argument("--record", type=str, default=None, help="append sensor / motor telemetry to JSONL")
    return parser.parse_args()


def main():
    args = parse_args()
    simulation = FlySimulation(args)
    simulation.run()


if __name__ == "__main__":
    main()
