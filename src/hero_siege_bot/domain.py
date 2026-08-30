from dataclasses import dataclass
from enum import StrEnum, auto

import numpy as np
from numpy.typing import NDArray


def normalize_pixel_index(index: float, dimension: int) -> float:
    """Map an image pixel index to edge-inclusive [0, 1] coordinates.

    Pixel index zero maps to 0 and index ``dimension - 1`` maps to 1. A
    single-pixel dimension maps to 0 because it has no distinct far edge.
    """
    if dimension <= 0:
        raise ValueError("pixel dimension must be positive")
    if dimension == 1:
        return 0.0
    return float(np.clip(index / (dimension - 1), 0.0, 1.0))


class BotState(StrEnum):
    CALIBRATING = auto()
    EXPLORING = auto()
    COMBAT = auto()
    LOOTING = auto()
    RECOVERING = auto()
    DEAD = auto()
    RESTARTING = auto()
    PAUSED = auto()


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class Detection:
    kind: str
    center: Point
    confidence: float
    bbox: Rect | None = None


@dataclass(frozen=True)
class MapMasks:
    explored: NDArray[np.bool_]
    fog: NDArray[np.bool_]
    walkable: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if (
            self.explored.shape != self.fog.shape
            or self.explored.shape != self.walkable.shape
        ):
            raise ValueError("map masks must have matching shapes")
        if self.explored.ndim != 2:
            raise ValueError("map masks must be two-dimensional")
        if any(
            mask.dtype != np.bool_ for mask in (self.explored, self.fog, self.walkable)
        ):
            raise TypeError("map masks must use boolean arrays")


@dataclass(frozen=True)
class Observation:
    timestamp: float
    calibrated: bool
    calibration_confidence: float
    focused: bool
    health_ratio: float | None
    resource_ratio: float | None
    player_map_position: Point | None
    enemies: tuple[Detection, ...]
    loot: tuple[Detection, ...]
    dead: bool
    restart_visible: bool
    movement_progress: float
    map_masks: MapMasks | None = None
    restart_target: Point | None = None
    yolo: tuple[Detection, ...] = ()


@dataclass(frozen=True)
class Action:
    kind: str
    key: str | None = None
    target: Point | None = None
    duration_s: float = 0.0
