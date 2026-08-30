from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

import yaml  # type: ignore[import-untyped]


def _ratio(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")


def _positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class WindowConfig:
    title: str


@dataclass(frozen=True)
class ControlsConfig:
    movement: Mapping[str, str]
    skills: tuple[str, ...]
    potions: tuple[str, ...]
    emergency_stop: str


@dataclass(frozen=True)
class CalibrationConfig:
    confidence_threshold: float
    min_stable_frames: int
    min_scale: float
    max_scale: float
    scale_step: float

    def __post_init__(self) -> None:
        _ratio("confidence_threshold", self.confidence_threshold)
        if self.min_stable_frames <= 0:
            raise ValueError("min_stable_frames must be positive")
        _positive("min_scale", self.min_scale)
        _positive("max_scale", self.max_scale)
        _positive("scale_step", self.scale_step)
        if self.min_scale > self.max_scale:
            raise ValueError("min_scale must not exceed max_scale")


@dataclass(frozen=True)
class SurvivalConfig:
    health_threshold: float
    resource_threshold: float
    potion_cooldown_s: float

    def __post_init__(self) -> None:
        _ratio("health_threshold", self.health_threshold)
        _ratio("resource_threshold", self.resource_threshold)
        _positive("potion_cooldown_s", self.potion_cooldown_s)


@dataclass(frozen=True)
class CombatConfig:
    detection_confidence: float
    attack_hold_s: float
    skill_cooldowns_s: Mapping[str, float]
    combat_timeout_s: float
    loot_timeout_s: float

    def __post_init__(self) -> None:
        _ratio("detection_confidence", self.detection_confidence)
        _positive("attack_hold_s", self.attack_hold_s)
        _positive("combat_timeout_s", self.combat_timeout_s)
        _positive("loot_timeout_s", self.loot_timeout_s)
        for key, duration in self.skill_cooldowns_s.items():
            _positive(f"skill_cooldowns_s.{key}", duration)


@dataclass(frozen=True)
class ExplorationConfig:
    movement_pulse_s: float
    max_movement_pulse_s: float
    stuck_timeout_s: float

    def __post_init__(self) -> None:
        _positive("movement_pulse_s", self.movement_pulse_s)
        _positive("max_movement_pulse_s", self.max_movement_pulse_s)
        _positive("stuck_timeout_s", self.stuck_timeout_s)
        if self.movement_pulse_s > self.max_movement_pulse_s:
            raise ValueError("movement_pulse_s must not exceed max_movement_pulse_s")


@dataclass(frozen=True)
class RecordingConfig:
    enabled: bool
    overlay: bool
    frame_interval_s: float

    def __post_init__(self) -> None:
        _positive("frame_interval_s", self.frame_interval_s)


@dataclass(frozen=True)
class BotConfig:
    window: WindowConfig
    controls: ControlsConfig
    calibration: CalibrationConfig
    survival: SurvivalConfig
    combat: CombatConfig
    exploration: ExplorationConfig
    recording: RecordingConfig


_DEFAULTS: dict[str, object] = {
    "window": {"title": "Hero Siege"},
    "controls": {
        "movement": {"up": "W", "left": "A", "down": "S", "right": "D"},
        "skills": ["Q", "E"],
        "potions": ["1", "2"],
        "emergency_stop": "F12",
    },
    "calibration": {
        "confidence_threshold": 0.9,
        "min_stable_frames": 3,
        "min_scale": 0.5,
        "max_scale": 1.5,
        "scale_step": 0.05,
    },
    "survival": {
        "health_threshold": 0.35,
        "resource_threshold": 0.25,
        "potion_cooldown_s": 1.0,
    },
    "combat": {
        "detection_confidence": 0.7,
        "attack_hold_s": 0.25,
        "skill_cooldowns_s": {"Q": 5.0, "E": 8.0},
        "combat_timeout_s": 10.0,
        "loot_timeout_s": 3.0,
    },
    "exploration": {
        "movement_pulse_s": 0.15,
        "max_movement_pulse_s": 0.3,
        "stuck_timeout_s": 1.5,
    },
    "recording": {"enabled": True, "overlay": True, "frame_interval_s": 1.0},
}


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return cast(dict[str, object], value)


def _merged(raw: dict[str, object]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, default in _DEFAULTS.items():
        section = dict(_mapping(default, name))
        if name in raw:
            section.update(_mapping(raw[name], name))
        result[name] = section
    return result


def _float(section: Mapping[str, object], name: str) -> float:
    value = section[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    return float(value)


def _int(section: Mapping[str, object], name: str) -> int:
    value = section[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of strings")
    return tuple(value)


def load_config(path: Path) -> BotConfig:
    loaded = yaml.safe_load(path.read_text())
    if loaded is None:
        raw: dict[str, object] = {}
    else:
        raw = _mapping(loaded, "configuration")
    sections = _merged(raw)

    window = sections["window"]
    controls = sections["controls"]
    calibration = sections["calibration"]
    survival = sections["survival"]
    combat = sections["combat"]
    exploration = sections["exploration"]
    recording = sections["recording"]

    movement = _mapping(controls["movement"], "movement")
    cooldowns = _mapping(combat["skill_cooldowns_s"], "skill_cooldowns_s")
    return BotConfig(
        window=WindowConfig(title=str(window["title"])),
        controls=ControlsConfig(
            movement=MappingProxyType({key: str(value) for key, value in movement.items()}),
            skills=_strings(controls["skills"], "skills"),
            potions=_strings(controls["potions"], "potions"),
            emergency_stop=str(controls["emergency_stop"]),
        ),
        calibration=CalibrationConfig(
            confidence_threshold=_float(calibration, "confidence_threshold"),
            min_stable_frames=_int(calibration, "min_stable_frames"),
            min_scale=_float(calibration, "min_scale"),
            max_scale=_float(calibration, "max_scale"),
            scale_step=_float(calibration, "scale_step"),
        ),
        survival=SurvivalConfig(
            health_threshold=_float(survival, "health_threshold"),
            resource_threshold=_float(survival, "resource_threshold"),
            potion_cooldown_s=_float(survival, "potion_cooldown_s"),
        ),
        combat=CombatConfig(
            detection_confidence=_float(combat, "detection_confidence"),
            attack_hold_s=_float(combat, "attack_hold_s"),
            skill_cooldowns_s=MappingProxyType(
                {key: _float(cooldowns, key) for key in cooldowns}
            ),
            combat_timeout_s=_float(combat, "combat_timeout_s"),
            loot_timeout_s=_float(combat, "loot_timeout_s"),
        ),
        exploration=ExplorationConfig(
            movement_pulse_s=_float(exploration, "movement_pulse_s"),
            max_movement_pulse_s=_float(exploration, "max_movement_pulse_s"),
            stuck_timeout_s=_float(exploration, "stuck_timeout_s"),
        ),
        recording=RecordingConfig(
            enabled=bool(recording["enabled"]),
            overlay=bool(recording["overlay"]),
            frame_interval_s=_float(recording, "frame_interval_s"),
        ),
    )
