from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from hero_siege_bot.calibration import AnchorRegion, AutoCalibrator
from hero_siege_bot.capture import CapturedFrame
from hero_siege_bot.config import CalibrationConfig
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


def _calibrator() -> AutoCalibrator:
    config = CalibrationConfig(
        confidence_threshold=0.9,
        min_stable_frames=3,
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
