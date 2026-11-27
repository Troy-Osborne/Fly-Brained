"""Segment-by-segment abdomen template, ordered thorax-to-tail."""
from dataclasses import dataclass

@dataclass(frozen=True)
class AbdomenSegmentTemplate:
    length: float
    height: float
    width: float
    x: float
    y: float
    colour: tuple = (0.25, 0.18, 0.14)

DEFAULT_ABDOMEN_SEGMENTS = (
    AbdomenSegmentTemplate(0.48, 0.31, 0.43, -0.36, -0.015),
    AbdomenSegmentTemplate(0.48 * 0.95, 0.31 * 0.95, 0.43 * 0.95, -0.64, -0.025),
    AbdomenSegmentTemplate(0.48 * 0.83, 0.31 * 0.83, 0.43 * 0.83, -0.92, -0.035),
    AbdomenSegmentTemplate(0.48 * 0.65, 0.31 * 0.65, 0.43 * 0.65, -1.20, -0.045),
)
