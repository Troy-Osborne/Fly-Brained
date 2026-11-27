"""Composable template set for swapping the complete visible body anatomy."""
from dataclasses import dataclass
from typing import Tuple

from .abdomen import AbdomenSegmentTemplate, DEFAULT_ABDOMEN_SEGMENTS
from .antennae import AntennaTemplate, DEFAULT_ANTENNA
from .eyes import EyeTemplate, DEFAULT_EYES
from .halteres import HaltereTemplate, DEFAULT_HALTERE
from .head import HeadTemplate, DEFAULT_HEAD
from .leg_visuals import LegVisualTemplate, DEFAULT_LEG_VISUAL
from .leg_geometry import LegGeometryTemplate, DEFAULT_LEG_TEMPLATE
from .nervous_system import NervousSystemTemplate, DEFAULT_NERVOUS_SYSTEM
from .thorax import ThoraxTemplate, DEFAULT_THORAX
from .wing_geometry import WingTemplate, DEFAULT_WING_TEMPLATE
from .wing_motion import WingMotionTemplate, DEFAULT_WING_MOTION


@dataclass(frozen=True)
class FlyAnatomyTemplates:
    head: HeadTemplate = DEFAULT_HEAD
    thorax: ThoraxTemplate = DEFAULT_THORAX
    abdomen: Tuple[AbdomenSegmentTemplate, ...] = DEFAULT_ABDOMEN_SEGMENTS
    eyes: EyeTemplate = DEFAULT_EYES
    antenna: AntennaTemplate = DEFAULT_ANTENNA
    haltere: HaltereTemplate = DEFAULT_HALTERE
    legs: LegGeometryTemplate = DEFAULT_LEG_TEMPLATE
    leg_visual: LegVisualTemplate = DEFAULT_LEG_VISUAL
    wings: WingTemplate = DEFAULT_WING_TEMPLATE
    wing_motion: WingMotionTemplate = DEFAULT_WING_MOTION
    nervous_system: NervousSystemTemplate = DEFAULT_NERVOUS_SYSTEM


DEFAULT_FLY_TEMPLATES = FlyAnatomyTemplates()
