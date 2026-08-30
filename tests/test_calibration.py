import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

import hero_siege_bot.capture as capture_module
from hero_siege_bot.calibration import (
    AnchorRegion,
    AutoCalibrator,
    CalibrationProfile,
)
from hero_siege_bot.capture import CapturedFrame, WindowCapture
from hero_siege_bot.cli import _load_calibrator
from hero_siege_bot.config import CalibrationConfig, load_config
from hero_siege_bot.domain import Rect


def _anchor(width: int, height: int, seed: int) -> NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    cv2.rectangle(image, (2, 2), (width - 3, height - 3), (255, 255, 255), 2)
    cv2.line(image, (0, height - 1), (width - 1, 0), (0, 0, 0), 2)
    return image


HUD_ANCHOR = _anchor(60, 30, 10)
MINIMAP_ANCHOR = _anchor(60, 60, 20)


def _image(minimap_x: int = 1800) -> NDArray[np.uint8]:
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    image[45:75, 60:120] = HUD_ANCHOR
    image[60:120, minimap_x : minimap_x + 60] = MINIMAP_ANCHOR
    return cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)


def _frames(images: Sequence[NDArray[np.uint8]]) -> list[CapturedFrame]:
    return [
        CapturedFrame(
            image=image,
            client_rect=Rect(200, 100, 1280, 720),
            focused=True,
            timestamp=float(index),
        )
        for index, image in enumerate(images)
    ]


def _calibrator(min_stable_frames: int = 3) -> AutoCalibrator:
    config = CalibrationConfig(
        confidence_threshold=0.9,
        min_stable_frames=min_stable_frames,
        min_scale=0.5,
        max_scale=1.0,
        scale_step=0.01,
    )
    return AutoCalibrator(
        config=config,
        anchors={"hud": HUD_ANCHOR, "minimap": MINIMAP_ANCHOR},
        regions={
            "health": AnchorRegion("hud", x=0.0, y=1.0, width=3.0, height=0.5),
            "minimap": AnchorRegion(
                "minimap", x=-2.0, y=-0.5, width=3.0, height=3.0
            ),
        },
    )


def _rect_iou(left: Rect, right: Rect) -> float:
    intersection_left = max(left.x, right.x)
    intersection_top = max(left.y, right.y)
    intersection_right = min(left.x + left.width, right.x + right.width)
    intersection_bottom = min(left.y + left.height, right.y + right.height)
    intersection = max(0, intersection_right - intersection_left) * max(
        0, intersection_bottom - intersection_top
    )
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union


def _real_frame_variants(
    image: NDArray[np.uint8],
    *,
    scale: float = 1.0,
    offset_x: int = 0,
    offset_y: int = 0,
) -> list[NDArray[np.uint8]]:
    variants: list[NDArray[np.uint8]] = []
    perturb_x = offset_x + round(300 * scale)
    perturb_y = offset_y + round(400 * scale)
    perturb_width = max(8, round(40 * scale))
    perturb_height = max(8, round(30 * scale))
    for index, (contrast, brightness) in enumerate(
        ((0.97, -4.0), (1.0, 0.0), (1.03, 4.0))
    ):
        variant = np.clip(
            image.astype(np.float32) * contrast + brightness,
            0,
            255,
        ).astype(np.uint8)
        noise = np.random.default_rng(800 + index).integers(
            0,
            32,
            (perturb_height, perturb_width, 3),
            dtype=np.uint8,
        )
        variant[
            perturb_y : perturb_y + perturb_height,
            perturb_x : perturb_x + perturb_width,
        ] = noise
        variants.append(variant)
    assert not np.array_equal(variants[0], variants[1])
    assert not np.array_equal(variants[1], variants[2])
    return variants


def test_calibrates_scaled_hud_and_minimap_regions_across_three_frames() -> None:
    calibrator = _calibrator()

    result = calibrator.calibrate(_frames([_image(), _image(), _image()]))

    assert result is not None
    assert result.confidence >= 0.9
    assert _rect_iou(result.regions["minimap"], Rect(1120, 20, 120, 120)) >= 0.9
    assert _rect_iou(result.regions["health"], Rect(40, 50, 120, 10)) >= 0.9


def test_real_anchors_calibrate_verified_fixture_regions() -> None:
    image = cv2.imread(
        "tests/fixtures/frames/highland_graveyard_1024x655.png",
        cv2.IMREAD_COLOR,
    )
    assert image is not None
    frames = _frames(_real_frame_variants(image))

    result = _load_calibrator(load_config(Path("config/default.yaml"))).calibrate(frames)

    assert result is not None
    assert _rect_iou(result.regions["health"], Rect(58, 17, 102, 12)) >= 0.9
    assert _rect_iou(result.regions["resource"], Rect(58, 34, 102, 10)) >= 0.9
    assert _rect_iou(result.regions["minimap"], Rect(898, 0, 126, 143)) >= 0.9
    assert result.regions["gameplay"] == Rect(0, 0, 1024, 655)
    assert result.regions["screen_state"] == Rect(0, 0, 1024, 655)


def test_current_hud_profile_calibrates_verified_fixture_regions() -> None:
    image = cv2.imread(
        "tests/fixtures/frames/satanic_black_hole_1024x655.png",
        cv2.IMREAD_COLOR,
    )
    assert image is not None
    frames = _frames(_real_frame_variants(image))

    result = _load_calibrator(load_config(Path("config/default.yaml"))).calibrate(frames)

    assert result is not None
    assert result.confidence >= 0.9
    assert _rect_iou(result.regions["health"], Rect(58, 17, 102, 12)) >= 0.9
    assert _rect_iou(result.regions["resource"], Rect(58, 34, 102, 10)) >= 0.9
    assert _rect_iou(result.regions["minimap"], Rect(898, 0, 126, 143)) >= 0.9
    assert result.regions["gameplay"] == Rect(0, 0, 1024, 655)
    assert result.regions["screen_state"] == Rect(0, 0, 1024, 655)


def test_selects_highest_confidence_stable_profile() -> None:
    strong_anchor = _anchor(30, 20, 31)
    weak_anchor = strong_anchor.copy()
    weak_anchor[5, 5] = 255 - weak_anchor[5, 5]
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[60:80, 100:130] = strong_anchor
    config = CalibrationConfig(
        confidence_threshold=0.9,
        min_stable_frames=3,
        min_scale=1.0,
        max_scale=1.0,
        scale_step=0.01,
    )
    calibrator = AutoCalibrator(
        config,
        profiles=(
            CalibrationProfile(
                "lower-confidence",
                {"hud": weak_anchor},
                {"health": AnchorRegion("hud", 0.0, 0.0, 1.0, 1.0)},
            ),
            CalibrationProfile(
                "highest-confidence",
                {"hud": strong_anchor},
                {"health": AnchorRegion("hud", 1.0, 0.0, 1.0, 1.0)},
            ),
        ),
    )

    result = calibrator.calibrate(_frames([image, image.copy(), image.copy()]))

    assert result is not None
    assert result.regions["health"] == Rect(130, 60, 30, 20)


@pytest.mark.parametrize(
    ("scale", "offset_x", "offset_y", "brightness"),
    [
        (0.8, 31, 17, 18),
        (1.2, 43, 29, -12),
    ],
)
def test_real_anchors_localize_deterministic_offline_variants(
    scale: float,
    offset_x: int,
    offset_y: int,
    brightness: int,
) -> None:
    source = cv2.imread(
        "tests/fixtures/frames/highland_graveyard_1024x655.png",
        cv2.IMREAD_COLOR,
    )
    assert source is not None
    scaled_size = (
        round(source.shape[1] * scale),
        round(source.shape[0] * scale),
    )
    scaled = cv2.resize(source, scaled_size, interpolation=cv2.INTER_LINEAR)
    adjusted = np.clip(
        scaled.astype(np.int16) + brightness,
        0,
        255,
    ).astype(np.uint8)
    transformed = np.zeros(
        (
            adjusted.shape[0] + offset_y + 11,
            adjusted.shape[1] + offset_x + 13,
            3,
        ),
        dtype=np.uint8,
    )
    transformed[
        offset_y : offset_y + adjusted.shape[0],
        offset_x : offset_x + adjusted.shape[1],
    ] = adjusted

    result = _load_calibrator(load_config(Path("config/default.yaml"))).calibrate(
        _frames(
            _real_frame_variants(
                transformed,
                scale=scale,
                offset_x=offset_x,
                offset_y=offset_y,
            )
        )
    )

    assert result is not None
    assert result.confidence >= 0.9
    base_regions = {
        "health": Rect(58, 17, 102, 12),
        "resource": Rect(58, 34, 102, 10),
        "minimap": Rect(898, 0, 126, 143),
        "gameplay": Rect(0, 0, 1024, 655),
        "screen_state": Rect(0, 0, 1024, 655),
    }
    for name, base in base_regions.items():
        expected = Rect(
            x=round(base.x * scale) + offset_x,
            y=round(base.y * scale) + offset_y,
            width=round(base.width * scale),
            height=round(base.height * scale),
        )
        actual = result.regions[name]
        assert abs(actual.x - expected.x) <= 2
        assert abs(actual.y - expected.y) <= 2
        assert _rect_iou(actual, expected) >= 0.8


def test_rejects_sequence_when_one_frame_has_no_anchors() -> None:
    calibrator = _calibrator()
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)

    result = calibrator.calibrate(_frames([_image(), blank, _image()]))

    assert result is None


def test_rejects_anchors_that_disagree_across_frame_sequence() -> None:
    calibrator = _calibrator()

    result = calibrator.calibrate(
        _frames([_image(), _image(minimap_x=1500), _image()])
    )

    assert result is None


@pytest.mark.parametrize("configured_minimum", [1, 2])
def test_requires_three_frames_even_when_configured_minimum_is_lower(
    configured_minimum: int,
) -> None:
    calibrator = _calibrator(min_stable_frames=configured_minimum)

    result = calibrator.calibrate(_frames([_image(), _image()]))

    assert result is None


def test_window_bounds_are_read_after_enabling_dpi_awareness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class User32:
        def SetProcessDpiAwarenessContext(self, context: object) -> bool:
            events.append("dpi")
            return True

    def enum_windows(callback: object, context: object) -> None:
        callback(42, context)  # type: ignore[operator]

    def client_to_screen(hwnd: int, point: tuple[int, int]) -> tuple[int, int]:
        events.append("coordinates")
        return point[0] + 100, point[1] + 200

    fake_ctypes = SimpleNamespace(
        c_void_p=lambda value: value,
        windll=SimpleNamespace(user32=User32()),
    )
    fake_win32gui = SimpleNamespace(
        EnumWindows=enum_windows,
        IsWindowVisible=lambda hwnd: True,
        GetWindowText=lambda hwnd: "Hero Siege",
        GetClientRect=lambda hwnd: (0, 0, 640, 480),
        ClientToScreen=client_to_screen,
    )
    monkeypatch.setattr(capture_module.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)
    monkeypatch.setitem(sys.modules, "win32gui", fake_win32gui)

    result = WindowCapture("Hero Siege").find()

    assert result == Rect(100, 200, 640, 480)
    assert events == ["dpi", "coordinates", "coordinates"]
