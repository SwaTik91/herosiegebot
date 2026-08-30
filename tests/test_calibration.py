import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

import hero_siege_bot.capture as capture_module
from hero_siege_bot.calibration import AnchorRegion, AutoCalibrator
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
    frames = _frames([image, image, image])

    result = _load_calibrator(load_config(Path("config/default.yaml"))).calibrate(frames)

    assert result is not None
    assert _rect_iou(result.regions["health"], Rect(58, 17, 102, 12)) >= 0.9
    assert _rect_iou(result.regions["resource"], Rect(58, 34, 102, 10)) >= 0.9
    assert _rect_iou(result.regions["minimap"], Rect(898, 0, 126, 143)) >= 0.9
    assert result.regions["gameplay"] == Rect(0, 0, 1024, 655)
    assert result.regions["screen_state"] == Rect(0, 0, 1024, 655)


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
