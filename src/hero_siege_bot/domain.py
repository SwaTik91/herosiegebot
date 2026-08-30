from dataclasses import dataclass
from enum import StrEnum, auto


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


@dataclass(frozen=True)
class Action:
    kind: str
    key: str | None = None
    target: Point | None = None
    duration_s: float = 0.0
