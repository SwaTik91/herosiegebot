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

    def __post_init__(self) -> None:
        required_movement = {"up": "W", "left": "A", "down": "S", "right": "D"}
        if self.movement != required_movement:
            raise ValueError("movement must use the required WASD bindings")
        if self.skills != ("Q", "E"):
            raise ValueError("skills must use the required Q/E bindings")
        if self.potions != ("1", "2"):
            raise ValueError("potions must use the required 1/2 bindings")
        if self.emergency_stop != "F12":
            raise ValueError("emergency_stop must remain F12")


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
        if set(self.skill_cooldowns_s) != {"Q", "E"}:
            raise ValueError("skill_cooldowns_s must contain exactly Q/E")
        for key, duration in self.skill_cooldowns_s.items():
            _positive(f"skill_cooldowns_s.{key}", duration)


@dataclass(frozen=True)
class DetectorConfig:
    template_confidence: float
    nms_iou_threshold: float
    motion_threshold: int
    min_candidate_area: int
    bar_min_border_confidence: float
    health_fill_hsv_lower: tuple[int, int, int]
    health_fill_hsv_upper: tuple[int, int, int]
    resource_fill_hsv_lower: tuple[int, int, int]
    resource_fill_hsv_upper: tuple[int, int, int]
    bar_border_hsv_lower: tuple[int, int, int]
    bar_border_hsv_upper: tuple[int, int, int]
    player_marker_hsv_lower: tuple[int, int, int]
    player_marker_hsv_upper: tuple[int, int, int]
    player_marker_min_area: int

    def __post_init__(self) -> None:
        for name in (
            "template_confidence",
            "nms_iou_threshold",
            "bar_min_border_confidence",
        ):
            _ratio(name, cast(float, getattr(self, name)))
        if not 0 <= self.motion_threshold <= 255:
            raise ValueError("motion_threshold must be between 0 and 255")
        if self.min_candidate_area <= 0:
            raise ValueError("min_candidate_area must be positive")
        if self.player_marker_min_area <= 0:
            raise ValueError("player_marker_min_area must be positive")
        pairs = (
            ("health_fill", self.health_fill_hsv_lower, self.health_fill_hsv_upper),
            (
                "resource_fill",
                self.resource_fill_hsv_lower,
                self.resource_fill_hsv_upper,
            ),
            ("bar_border", self.bar_border_hsv_lower, self.bar_border_hsv_upper),
            (
                "player_marker",
                self.player_marker_hsv_lower,
                self.player_marker_hsv_upper,
            ),
        )
        for name, lower, upper in pairs:
            _validate_hsv(f"{name}_hsv_lower", lower)
            _validate_hsv(f"{name}_hsv_upper", upper)
            if any(low > high for low, high in zip(lower, upper, strict=True)):
                raise ValueError(f"{name}_hsv_lower must not exceed {name}_hsv_upper")


@dataclass(frozen=True)
class ExplorationConfig:
    movement_pulse_s: float
    max_movement_pulse_s: float
    stuck_timeout_s: float
    fog_hsv_lower: tuple[int, int, int] = (0, 0, 0)
    fog_hsv_upper: tuple[int, int, int] = (179, 255, 45)
    explored_hsv_lower: tuple[int, int, int] = (0, 0, 140)
    explored_hsv_upper: tuple[int, int, int] = (179, 120, 255)
    morphology_kernel_size: int = 3
    morphology_iterations: int = 1
    frontier_reveal_radius: int = 4
    frontier_path_weight: float = 1.0
    frontier_reveal_weight: float = 1.0
    frontier_failure_penalty: float = 100.0
    frontier_blacklist_radius: float = 0.08
    progress_min_distance: float = 0.01
    progress_min_reveal_pixels: int = 2
    no_progress_sample_limit: int = 3

    def __post_init__(self) -> None:
        _positive("movement_pulse_s", self.movement_pulse_s)
        _positive("max_movement_pulse_s", self.max_movement_pulse_s)
        _positive("stuck_timeout_s", self.stuck_timeout_s)
        if self.movement_pulse_s > self.max_movement_pulse_s:
            raise ValueError("movement_pulse_s must not exceed max_movement_pulse_s")
        for name, hsv in (
            ("fog_hsv_lower", self.fog_hsv_lower),
            ("fog_hsv_upper", self.fog_hsv_upper),
            ("explored_hsv_lower", self.explored_hsv_lower),
            ("explored_hsv_upper", self.explored_hsv_upper),
        ):
            if len(hsv) != 3:
                raise ValueError(f"{name} must contain three HSV channels")
            limits = (179, 255, 255)
            if any(value < 0 or value > limit for value, limit in zip(hsv, limits, strict=True)):
                raise ValueError(f"{name} contains an out-of-range HSV channel")
        if any(
            low > high
            for low, high in zip(self.fog_hsv_lower, self.fog_hsv_upper, strict=True)
        ):
            raise ValueError("fog_hsv_lower must not exceed fog_hsv_upper")
        if any(
            low > high
            for low, high in zip(
                self.explored_hsv_lower, self.explored_hsv_upper, strict=True
            )
        ):
            raise ValueError("explored_hsv_lower must not exceed explored_hsv_upper")
        if self.morphology_kernel_size <= 0 or self.morphology_kernel_size % 2 == 0:
            raise ValueError("morphology_kernel_size must be a positive odd integer")
        if self.morphology_iterations <= 0:
            raise ValueError("morphology_iterations must be positive")
        if self.frontier_reveal_radius <= 0:
            raise ValueError("frontier_reveal_radius must be positive")
        _positive("frontier_path_weight", self.frontier_path_weight)
        _positive("frontier_reveal_weight", self.frontier_reveal_weight)
        _positive("frontier_failure_penalty", self.frontier_failure_penalty)
        _positive("frontier_blacklist_radius", self.frontier_blacklist_radius)
        _positive("progress_min_distance", self.progress_min_distance)
        if self.progress_min_reveal_pixels <= 0:
            raise ValueError("progress_min_reveal_pixels must be positive")
        if self.no_progress_sample_limit <= 0:
            raise ValueError("no_progress_sample_limit must be positive")


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
    detectors: DetectorConfig
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
    "detectors": {
        "template_confidence": 0.8,
        "nms_iou_threshold": 0.3,
        "motion_threshold": 25,
        "min_candidate_area": 12,
        "bar_min_border_confidence": 0.7,
        "health_fill_hsv_lower": [0, 120, 100],
        "health_fill_hsv_upper": [10, 255, 255],
        "resource_fill_hsv_lower": [105, 120, 100],
        "resource_fill_hsv_upper": [130, 255, 255],
        "bar_border_hsv_lower": [0, 0, 180],
        "bar_border_hsv_upper": [179, 80, 255],
        "player_marker_hsv_lower": [80, 100, 50],
        "player_marker_hsv_upper": [115, 255, 255],
        "player_marker_min_area": 12,
    },
    "exploration": {
        "movement_pulse_s": 0.15,
        "max_movement_pulse_s": 0.3,
        "stuck_timeout_s": 1.5,
        "fog_hsv_lower": [0, 0, 0],
        "fog_hsv_upper": [179, 255, 45],
        "explored_hsv_lower": [0, 0, 140],
        "explored_hsv_upper": [179, 120, 255],
        "morphology_kernel_size": 3,
        "morphology_iterations": 1,
        "frontier_reveal_radius": 4,
        "frontier_path_weight": 1.0,
        "frontier_reveal_weight": 1.0,
        "frontier_failure_penalty": 100.0,
        "frontier_blacklist_radius": 0.08,
        "progress_min_distance": 0.01,
        "progress_min_reveal_pixels": 2,
        "no_progress_sample_limit": 3,
    },
    "recording": {"enabled": True, "overlay": True, "frame_interval_s": 1.0},
}


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return cast(dict[str, object], value)


def _merge_mapping(
    defaults: Mapping[str, object], overrides: Mapping[str, object]
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, default in defaults.items():
        override = overrides.get(key, default)
        if isinstance(default, dict) and isinstance(override, dict):
            result[key] = _merge_mapping(
                cast(dict[str, object], default), cast(dict[str, object], override)
            )
        else:
            result[key] = override
    for key, override in overrides.items():
        if key not in result:
            result[key] = override
    return result


def _merged(raw: dict[str, object]) -> dict[str, dict[str, object]]:
    merged = _merge_mapping(_DEFAULTS, raw)
    return {name: _mapping(value, name) for name, value in merged.items()}


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


def _bool(section: Mapping[str, object], name: str) -> bool:
    value = section[name]
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of strings")
    return tuple(value)


def _hsv(value: object, name: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        raise ValueError(f"{name} must be a list of three integers")
    return cast(tuple[int, int, int], tuple(value))


def _validate_hsv(name: str, hsv: tuple[int, int, int]) -> None:
    limits = (179, 255, 255)
    if any(value < 0 or value > limit for value, limit in zip(hsv, limits, strict=True)):
        raise ValueError(f"{name} contains an out-of-range HSV channel")


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
    detectors = sections["detectors"]
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
        detectors=DetectorConfig(
            template_confidence=_float(detectors, "template_confidence"),
            nms_iou_threshold=_float(detectors, "nms_iou_threshold"),
            motion_threshold=_int(detectors, "motion_threshold"),
            min_candidate_area=_int(detectors, "min_candidate_area"),
            bar_min_border_confidence=_float(
                detectors, "bar_min_border_confidence"
            ),
            health_fill_hsv_lower=_hsv(
                detectors["health_fill_hsv_lower"], "health_fill_hsv_lower"
            ),
            health_fill_hsv_upper=_hsv(
                detectors["health_fill_hsv_upper"], "health_fill_hsv_upper"
            ),
            resource_fill_hsv_lower=_hsv(
                detectors["resource_fill_hsv_lower"], "resource_fill_hsv_lower"
            ),
            resource_fill_hsv_upper=_hsv(
                detectors["resource_fill_hsv_upper"], "resource_fill_hsv_upper"
            ),
            bar_border_hsv_lower=_hsv(
                detectors["bar_border_hsv_lower"], "bar_border_hsv_lower"
            ),
            bar_border_hsv_upper=_hsv(
                detectors["bar_border_hsv_upper"], "bar_border_hsv_upper"
            ),
            player_marker_hsv_lower=_hsv(
                detectors["player_marker_hsv_lower"], "player_marker_hsv_lower"
            ),
            player_marker_hsv_upper=_hsv(
                detectors["player_marker_hsv_upper"], "player_marker_hsv_upper"
            ),
            player_marker_min_area=_int(detectors, "player_marker_min_area"),
        ),
        exploration=ExplorationConfig(
            movement_pulse_s=_float(exploration, "movement_pulse_s"),
            max_movement_pulse_s=_float(exploration, "max_movement_pulse_s"),
            stuck_timeout_s=_float(exploration, "stuck_timeout_s"),
            fog_hsv_lower=_hsv(exploration["fog_hsv_lower"], "fog_hsv_lower"),
            fog_hsv_upper=_hsv(exploration["fog_hsv_upper"], "fog_hsv_upper"),
            explored_hsv_lower=_hsv(
                exploration["explored_hsv_lower"], "explored_hsv_lower"
            ),
            explored_hsv_upper=_hsv(
                exploration["explored_hsv_upper"], "explored_hsv_upper"
            ),
            morphology_kernel_size=_int(exploration, "morphology_kernel_size"),
            morphology_iterations=_int(exploration, "morphology_iterations"),
            frontier_reveal_radius=_int(exploration, "frontier_reveal_radius"),
            frontier_path_weight=_float(exploration, "frontier_path_weight"),
            frontier_reveal_weight=_float(exploration, "frontier_reveal_weight"),
            frontier_failure_penalty=_float(exploration, "frontier_failure_penalty"),
            frontier_blacklist_radius=_float(exploration, "frontier_blacklist_radius"),
            progress_min_distance=_float(exploration, "progress_min_distance"),
            progress_min_reveal_pixels=_int(
                exploration, "progress_min_reveal_pixels"
            ),
            no_progress_sample_limit=_int(exploration, "no_progress_sample_limit"),
        ),
        recording=RecordingConfig(
            enabled=_bool(recording, "enabled"),
            overlay=_bool(recording, "overlay"),
            frame_interval_s=_float(recording, "frame_interval_s"),
        ),
    )
