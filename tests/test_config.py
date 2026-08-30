from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from hero_siege_bot.config import load_config
from hero_siege_bot.domain import BotState, Detection, Point


def test_load_config_rejects_out_of_range_threshold(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("window:\n  title: Hero Siege\nsurvival:\n  health_threshold: 1.2\n")
    with pytest.raises(ValueError, match="health_threshold"):
        load_config(path)


def test_default_config_uses_expected_controls() -> None:
    config = load_config(Path("config/default.yaml"))
    assert config.controls.movement == {"up": "W", "left": "A", "down": "S", "right": "D"}
    assert config.controls.skills == ("Q", "E")
    assert config.controls.potions == ("1", "2")
    assert config.controls.emergency_stop == "F12"


def test_load_config_rejects_non_positive_duration(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("combat:\n  attack_hold_s: 0\n")
    with pytest.raises(ValueError, match="attack_hold_s"):
        load_config(path)


def test_configuration_and_domain_values_are_immutable() -> None:
    config = load_config(Path("config/default.yaml"))
    detection = Detection("enemy", Point(0.5, 0.25), 0.9)

    with pytest.raises(FrozenInstanceError):
        detection.confidence = 0.1  # type: ignore[misc]
    with pytest.raises(TypeError):
        config.controls.movement["up"] = "X"  # type: ignore[index]


def test_bot_state_values_use_lowercase_names() -> None:
    assert BotState.CALIBRATING.value == "calibrating"
    assert BotState.PAUSED.value == "paused"
