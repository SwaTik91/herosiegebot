import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from hero_siege_bot.config import CalibrationConfig, load_config
from hero_siege_bot.domain import BotState, Detection, Point


def _project_version(path: Path) -> str:
    with path.open("rb") as project_file:
        return tomllib.load(project_file)["project"]["version"]


def _package_data(path: Path) -> list[str]:
    with path.open("rb") as project_file:
        return tomllib.load(project_file)["tool"]["setuptools"]["package-data"][
            "hero_siege_bot"
        ]


def test_package_version_is_a5() -> None:
    assert _project_version(Path("pyproject.toml")) == "0.1.0a5"


def test_package_data_includes_only_shipped_calibration_anchors() -> None:
    patterns = _package_data(Path("pyproject.toml"))

    assert patterns == ["assets/anchors/*.png"]
    assert not any("templates" in pattern for pattern in patterns)


def test_clean_build_excludes_templates_and_retains_every_shipped_anchor(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    shutil.copytree(
        Path.cwd(),
        source,
        ignore=shutil.ignore_patterns(
            ".git",
            ".venv",
            ".worktrees",
            ".superpowers",
            "build",
            "dist",
            "*.egg-info",
            "__pycache__",
        ),
    )
    synthetic = source / "src/hero_siege_bot/assets/templates/synthetic.png"
    synthetic.parent.mkdir(parents=True, exist_ok=True)
    synthetic.write_bytes(b"not-a-real-png")
    output = tmp_path / "dist"

    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(output), str(source)],
        check=True,
        capture_output=True,
        text=True,
    )

    wheel = next(output.glob("*.whl"))
    sdist = next(output.glob("*.tar.gz"))
    with zipfile.ZipFile(wheel) as archive:
        wheel_entries = archive.namelist()
    with tarfile.open(sdist, "r:gz") as archive:
        sdist_entries = archive.getnames()

    for entries in (wheel_entries, sdist_entries):
        assert not any("/assets/templates/" in f"/{entry}" for entry in entries)
    shipped_anchors = {
        path.name for path in (source / "src/hero_siege_bot/assets/anchors").glob("*.png")
    }
    assert shipped_anchors
    for anchor in shipped_anchors:
        assert any(entry.endswith(f"/assets/anchors/{anchor}") for entry in wheel_entries)
        assert any(entry.endswith(f"/assets/anchors/{anchor}") for entry in sdist_entries)


def write_config(tmp_path: Path, contents: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(contents)
    return path


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
    assert config.controls.emergency_stop == "CTRL+SHIFT+F10"


def test_default_yolo_is_observe_only_with_user_weights() -> None:
    config = load_config(Path("config/default.yaml"))
    assert config.yolo.enabled is True
    assert config.yolo.observe_only is True
    assert config.yolo.weights == "models/hero-siege-yolo11n-user-v1.pt"
    assert config.yolo.confidence == 0.35
    assert config.recording.live_overlay is True


@pytest.mark.parametrize("contents", ["{}", "calibration: {}"])
def test_default_calibration_supports_two_x_profile_scale(
    tmp_path: Path, contents: str
) -> None:
    config = load_config(write_config(tmp_path, contents))

    assert config.calibration.max_scale == 2.0


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


@pytest.mark.parametrize(
    ("controls", "field"),
    [
        (
            "movement: {up: UP, left: A, down: S, right: D}",
            "movement",
        ),
        ("skills: [Q, R]", "skills"),
        ('potions: ["1", "3"]', "potions"),
        ("emergency_stop: F12", "emergency_stop"),
        ("emergency_stop: Ctrl+Shift+F10", "emergency_stop"),
        ("emergency_stop: CTRL+SHIFT+F11", "emergency_stop"),
    ],
)
def test_load_config_rejects_changed_safety_controls(
    tmp_path: Path, controls: str, field: str
) -> None:
    path = write_config(tmp_path, f"controls:\n  {controls}\n")

    with pytest.raises(ValueError, match=field):
        load_config(path)


def test_nested_cooldown_override_preserves_other_defaults(tmp_path: Path) -> None:
    path = write_config(tmp_path, "combat:\n  skill_cooldowns_s:\n    Q: 6.0\n")

    config = load_config(path)

    assert config.combat.skill_cooldowns_s == {"Q": 6.0, "E": 8.0}


@pytest.mark.parametrize("field", ["enabled", "overlay"])
def test_recording_flags_require_boolean_values(tmp_path: Path, field: str) -> None:
    path = write_config(tmp_path, f'recording:\n  {field}: "false"\n')

    with pytest.raises(TypeError, match=field):
        load_config(path)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("survival", "health_threshold"),
        ("survival", "resource_threshold"),
        ("combat", "detection_confidence"),
    ],
)
@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_ratio_boundaries_are_accepted(
    tmp_path: Path, section: str, field: str, threshold: float
) -> None:
    path = write_config(tmp_path, f"{section}:\n  {field}: {threshold}\n")

    assert getattr(getattr(load_config(path), section), field) == threshold


@pytest.mark.parametrize(
    "threshold",
    [0.0, 0.899, float("nan"), float("inf"), float("-inf")],
)
def test_calibration_confidence_rejects_values_below_mandatory_floor(
    threshold: float,
) -> None:
    with pytest.raises(ValueError, match="confidence_threshold"):
        CalibrationConfig(
            confidence_threshold=threshold,
            min_stable_frames=3,
            min_scale=0.5,
            max_scale=2.0,
            scale_step=0.05,
        )


@pytest.mark.parametrize("threshold", [0.9, 1.0])
def test_calibration_confidence_accepts_mandatory_range(threshold: float) -> None:
    config = CalibrationConfig(
        confidence_threshold=threshold,
        min_stable_frames=3,
        min_scale=0.5,
        max_scale=2.0,
        scale_step=0.05,
    )

    assert config.confidence_threshold == threshold


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_ratio_values_outside_boundaries_are_rejected(
    tmp_path: Path, threshold: float
) -> None:
    path = write_config(tmp_path, f"combat:\n  detection_confidence: {threshold}\n")

    with pytest.raises(ValueError, match="detection_confidence"):
        load_config(path)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("survival", "potion_cooldown_s"),
        ("combat", "combat_timeout_s"),
        ("combat", "loot_timeout_s"),
        ("exploration", "movement_pulse_s"),
        ("exploration", "max_movement_pulse_s"),
        ("exploration", "stuck_timeout_s"),
        ("recording", "frame_interval_s"),
    ],
)
def test_all_configured_durations_must_be_positive(
    tmp_path: Path, section: str, field: str
) -> None:
    path = write_config(tmp_path, f"{section}:\n  {field}: 0\n")

    with pytest.raises(ValueError, match=field):
        load_config(path)


def test_skill_cooldowns_must_be_positive(tmp_path: Path) -> None:
    path = write_config(tmp_path, "combat:\n  skill_cooldowns_s:\n    Q: 0\n")

    with pytest.raises(ValueError, match="skill_cooldowns_s.Q"):
        load_config(path)
