from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray

from hero_siege_bot.calibration import Calibration
from hero_siege_bot.capture import CapturedFrame
from hero_siege_bot.cli import _load_calibrator
from hero_siege_bot.config import load_config
from hero_siege_bot.detectors import (
    MotionColorDetector,
    ScreenStateDetector,
    TemplateDetector,
)
from hero_siege_bot.domain import Detection, Rect
from hero_siege_bot.exploration import segment_minimap
from hero_siege_bot.perception import Perception


def _pattern(seed: int) -> NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    pattern = rng.integers(0, 256, (7, 9, 3), dtype=np.uint8)
    cv2.circle(pattern, (4, 3), 2, (255, 255, 255), 1)
    return pattern


LOOT = _pattern(10)
DEATH = _pattern(20)
RESTART = _pattern(30)
FRAME_FIXTURE_DIRECTORY = Path("tests/fixtures/frames")
RECORDED_FRAMES = tuple(sorted(FRAME_FIXTURE_DIRECTORY.glob("*.png")))


def _image(enemy_left: int, marker_left: int) -> NDArray[np.uint8]:
    image = np.zeros((100, 160, 3), dtype=np.uint8)

    health = image[0:10, 0:50]
    cv2.rectangle(health, (0, 0), (49, 9), (255, 255, 255), 1)
    health[1:9, 1:25] = (0, 0, 255)

    resource = image[10:20, 0:50]
    cv2.rectangle(resource, (0, 0), (49, 9), (255, 255, 255), 1)
    resource[1:9, 1:13] = (255, 0, 0)

    minimap = image[0:40, 120:160]
    minimap[:] = (180, 180, 180)
    minimap[16:21, marker_left : marker_left + 5] = (255, 255, 0)

    gameplay = image[20:100, 0:120]
    gameplay[25:35, enemy_left : enemy_left + 10] = (0, 0, 255)
    gameplay[55:62, 80:89] = LOOT

    state = image[40:100, 120:160]
    state[5:12, 4:13] = DEATH
    state[35:42, 25:34] = RESTART
    return image


def _frame(image: NDArray[np.uint8], timestamp: float = 12.5) -> CapturedFrame:
    return CapturedFrame(
        image=image,
        client_rect=Rect(50, 70, image.shape[1], image.shape[0]),
        focused=True,
        timestamp=timestamp,
    )


def _calibration(**regions: Rect) -> Calibration:
    defaults = {
        "health": Rect(0, 0, 50, 10),
        "resource": Rect(0, 10, 50, 10),
        "minimap": Rect(120, 0, 40, 40),
        "gameplay": Rect(0, 20, 120, 80),
        "screen_state": Rect(120, 40, 40, 60),
    }
    defaults.update(regions)
    return Calibration(regions=defaults, scale=1.0, confidence=0.97)


def _perception() -> Perception:
    config = load_config(Path("config/default.yaml"))
    return Perception(
        config=config,
        enemy_detector=MotionColorDetector(
            "enemy",
            hsv_lower=(0, 240, 240),
            hsv_upper=(5, 255, 255),
            motion_threshold=20,
            min_area=20,
        ),
        loot_detector=TemplateDetector(
            {"loot": LOOT},
            confidence_threshold=0.99,
            nms_iou_threshold=0.3,
        ),
        screen_state_detector=ScreenStateDetector(
            DEATH,
            RESTART,
            confidence_threshold=0.99,
            nms_iou_threshold=0.3,
        ),
    )


def test_perception_composes_calibrated_crops_into_immutable_observation() -> None:
    perception = _perception()
    perception.observe(_frame(_image(enemy_left=5, marker_left=3)), _calibration())
    image = _image(enemy_left=35, marker_left=24)

    observation = perception.observe(
        _frame(image, timestamp=13.0),
        _calibration(),
    )

    assert observation.timestamp == 13.0
    assert observation.calibrated is True
    assert observation.calibration_confidence == 0.97
    assert observation.focused is True
    assert observation.health_ratio == pytest.approx(0.5)
    assert observation.resource_ratio == pytest.approx(0.25)
    assert observation.player_map_position is not None
    assert observation.player_map_position.x == pytest.approx(26 / 39, abs=0.02)
    assert observation.player_map_position.y == pytest.approx(18 / 39, abs=0.02)
    assert len(observation.enemies) == 1
    assert len(observation.loot) == 1
    assert observation.dead is True
    assert observation.restart_visible is True
    assert observation.restart_target is not None
    assert observation.restart_target.x == pytest.approx(149 / 159)
    assert observation.restart_target.y == pytest.approx(78 / 99)
    assert observation.movement_progress > 0.0
    assert observation.map_masks is not None
    expected_masks = segment_minimap(
        image[0:40, 120:160],
        load_config(Path("config/default.yaml")).exploration,
    )
    np.testing.assert_array_equal(
        observation.map_masks.explored, expected_masks.explored
    )
    np.testing.assert_array_equal(observation.map_masks.fog, expected_masks.fog)
    np.testing.assert_array_equal(
        observation.map_masks.walkable, expected_masks.walkable
    )


@pytest.mark.parametrize(
    ("name", "rect"),
    [
        ("health", Rect(-1, 0, 50, 10)),
        ("resource", Rect(0, 10, 0, 10)),
        ("minimap", Rect(140, 0, 40, 40)),
        ("gameplay", Rect(0, 90, 120, 80)),
        ("screen_state", Rect(120, 40, 50, 60)),
    ],
)
def test_perception_returns_uncalibrated_observation_for_invalid_required_crop(
    name: str, rect: Rect
) -> None:
    observation = _perception().observe(
        _frame(_image(enemy_left=5, marker_left=3)),
        _calibration(**{name: rect}),
    )

    assert observation.calibrated is False
    assert observation.health_ratio is None
    assert observation.resource_ratio is None
    assert observation.player_map_position is None
    assert observation.enemies == ()
    assert observation.loot == ()
    assert observation.dead is False
    assert observation.restart_visible is False
    assert observation.movement_progress == 0.0
    assert observation.map_masks is None


def test_perception_returns_uncalibrated_observation_for_missing_required_crop() -> None:
    calibration = _calibration()
    calibration = Calibration(
        regions={
            name: rect
            for name, rect in calibration.regions.items()
            if name != "minimap"
        },
        scale=calibration.scale,
        confidence=calibration.confidence,
    )

    observation = _perception().observe(
        _frame(_image(enemy_left=5, marker_left=3)), calibration
    )

    assert observation.calibrated is False


def test_scaled_calibration_feeds_perception_without_invalid_crop_pause() -> None:
    image = np.zeros((1024, 1600, 3), dtype=np.uint8)
    anchor_dir = Path("src/hero_siege_bot/assets/anchors")
    hud = cv2.imread(
        str(anchor_dir / "hud_status_right_cap_v2.png"),
        cv2.IMREAD_COLOR,
    )
    minimap = cv2.imread(
        str(anchor_dir / "minimap_top_left_corner.png"),
        cv2.IMREAD_COLOR,
    )
    assert hud is not None
    assert minimap is not None
    image[12:79, 226:266] = cv2.resize(
        hud,
        (40, 67),
        interpolation=cv2.INTER_AREA,
    )
    image[0:38, 1403:1441] = cv2.resize(
        minimap,
        (38, 38),
        interpolation=cv2.INTER_AREA,
    )
    image[100:105, 1450:1455] = (255, 255, 0)
    captured = CapturedFrame(
        image=image,
        client_rect=Rect(0, 0, 1600, 1024),
        focused=True,
        timestamp=12.5,
    )
    config = load_config(Path("config/default.yaml"))
    calibration = _load_calibrator(config).calibrate([captured] * 5)
    assert calibration is not None
    assert calibration.regions["minimap"] == Rect(1403, 0, 197, 226)

    class NoDetections:
        def detect(self, crop: NDArray[np.uint8]) -> tuple[Detection, ...]:
            assert crop.size > 0
            return ()

    perception = Perception(
        config=config,
        enemy_detector=NoDetections(),
        loot_detector=NoDetections(),
        screen_state_detector=NoDetections(),
    )

    observation = perception.observe(captured, calibration)

    assert observation.calibrated is True
    assert observation.player_map_position is not None
    assert observation.map_masks is not None


def test_detector_thresholds_and_colors_are_configurable(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "detectors:\n"
        "  template_confidence: 0.88\n"
        "  nms_iou_threshold: 0.25\n"
        "  motion_threshold: 31\n"
        "  min_candidate_area: 17\n"
        "  bar_min_border_confidence: 0.75\n"
        "  player_marker_hsv_lower: [80, 100, 120]\n"
        "  player_marker_hsv_upper: [100, 255, 255]\n"
    )

    detectors = load_config(path).detectors

    assert detectors.template_confidence == 0.88
    assert detectors.nms_iou_threshold == 0.25
    assert detectors.motion_threshold == 31
    assert detectors.min_candidate_area == 17
    assert detectors.bar_min_border_confidence == 0.75
    assert detectors.player_marker_hsv_lower == (80, 100, 120)
    assert detectors.player_marker_hsv_upper == (100, 255, 255)


def _assert_status_annotation(field: str, annotation: object) -> None:
    assert isinstance(annotation, dict)
    assert set(annotation) == {"status", "value"}
    status = annotation["status"]
    value = annotation["value"]
    assert status in {"verified", "unknown"}
    if status == "unknown":
        assert value is None
        return

    if field == "player_map_point":
        if value is None:
            return
        assert isinstance(value, dict)
        assert set(value) == {"x", "y"}
        assert all(
            isinstance(value[axis], (int, float))
            and not isinstance(value[axis], bool)
            and 0.0 <= value[axis] <= 1.0
            for axis in ("x", "y")
        )
    elif field == "bar_ratios":
        assert isinstance(value, dict)
        assert set(value) == {"health", "resource"}
        assert all(
            value[name] is None
            or (
                isinstance(value[name], (int, float))
                and not isinstance(value[name], bool)
                and 0.0 <= value[name] <= 1.0
            )
            for name in ("health", "resource")
        )
    elif field in {"enemy_boxes", "loot_boxes"}:
        assert isinstance(value, list)
        for box in value:
            assert isinstance(box, dict)
            assert set(box) == {"x", "y", "width", "height"}
            assert all(isinstance(box[name], int) for name in box)
            assert box["width"] > 0
            assert box["height"] > 0
    elif field in {"death", "restart_visible"}:
        assert isinstance(value, bool)
    else:
        raise AssertionError(f"unknown annotation field: {field}")


@pytest.mark.parametrize("image_path", RECORDED_FRAMES or (None,))
def test_recorded_frame_has_complete_status_aware_sidecar(
    image_path: Path | None,
) -> None:
    if image_path is None:
        pytest.skip("no private recorded frame fixtures have been added")

    sidecar = image_path.with_suffix(".yaml")
    assert sidecar.exists(), f"missing sidecar for {image_path.name}"
    expected = yaml.safe_load(sidecar.read_text())
    assert set(expected) == {
        "frame_size",
        "hud_rectangles",
        "player_map_point",
        "bar_ratios",
        "enemy_boxes",
        "loot_boxes",
        "death",
        "restart_visible",
    }
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image is not None
    assert expected["frame_size"] == {
        "width": image.shape[1],
        "height": image.shape[0],
    }
    assert set(expected["hud_rectangles"]) == {
        "health",
        "resource",
        "minimap",
        "gameplay",
        "screen_state",
    }
    for field in (
        "player_map_point",
        "bar_ratios",
        "enemy_boxes",
        "loot_boxes",
        "death",
        "restart_visible",
    ):
        _assert_status_annotation(field, expected[field])


@pytest.mark.parametrize(
    ("field", "annotation"),
    [
        ("enemy_boxes", {"status": "unknown", "value": None}),
        ("enemy_boxes", {"status": "verified", "value": []}),
        (
            "loot_boxes",
            {
                "status": "verified",
                "value": [{"x": 10, "y": 20, "width": 5, "height": 6}],
            },
        ),
        ("player_map_point", {"status": "verified", "value": None}),
        (
            "player_map_point",
            {"status": "verified", "value": {"x": 0.25, "y": 0.75}},
        ),
        (
            "bar_ratios",
            {
                "status": "verified",
                "value": {"health": 1.0, "resource": None},
            },
        ),
        ("death", {"status": "verified", "value": False}),
    ],
)
def test_status_annotation_schema_accepts_unknown_and_typed_verified_values(
    field: str, annotation: dict[str, object]
) -> None:
    _assert_status_annotation(field, annotation)


def test_unknown_annotation_rejects_verified_absence_value() -> None:
    with pytest.raises(AssertionError):
        _assert_status_annotation(
            "enemy_boxes",
            {"status": "unknown", "value": []},
        )
