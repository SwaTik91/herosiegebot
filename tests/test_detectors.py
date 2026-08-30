from collections.abc import Iterator

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from hero_siege_bot.detectors import (
    BarReader,
    MotionColorDetector,
    ScreenStateDetector,
    TemplateDetector,
    YoloBox,
    YoloDetector,
)
from hero_siege_bot.domain import Rect, normalize_pixel_index


def test_normalized_pixel_coordinates_are_edge_inclusive_with_single_pixel_guard() -> None:
    assert normalize_pixel_index(0.0, 5) == 0.0
    assert normalize_pixel_index(4.0, 5) == 1.0
    assert normalize_pixel_index(0.0, 1) == 0.0


def _bar(ratio: float, *, damaged_border: bool = False) -> NDArray[np.uint8]:
    image = np.zeros((14, 104, 3), dtype=np.uint8)
    cv2.rectangle(image, (1, 1), (102, 12), (255, 255, 255), 1)
    fill_width = round(100 * ratio)
    if fill_width:
        image[2:12, 2 : 2 + fill_width] = (0, 255, 0)
    if damaged_border:
        image[1, 1:80] = 0
    return image


@pytest.mark.parametrize("ratio", [0.0, 0.25, 0.5, 1.0])
def test_bar_reader_returns_generated_fill_ratio(ratio: float) -> None:
    reader = BarReader(
        fill_hsv_lower=(55, 240, 240),
        fill_hsv_upper=(65, 255, 255),
        border_hsv_lower=(0, 0, 240),
        border_hsv_upper=(179, 20, 255),
        min_border_confidence=0.8,
    )

    measured, confidence = reader.read_ratio(_bar(ratio))

    assert measured == pytest.approx(ratio, abs=0.01)
    assert confidence == pytest.approx(1.0)


def test_bar_reader_rejects_crop_with_insufficient_border_confidence() -> None:
    reader = BarReader(
        fill_hsv_lower=(55, 240, 240),
        fill_hsv_upper=(65, 255, 255),
        border_hsv_lower=(0, 0, 240),
        border_hsv_upper=(179, 20, 255),
        min_border_confidence=0.8,
    )

    measured, confidence = reader.read_ratio(_bar(0.5, damaged_border=True))

    assert measured is None
    assert confidence < 0.8


def _template() -> NDArray[np.uint8]:
    rng = np.random.default_rng(42)
    template = rng.integers(0, 256, (9, 11, 3), dtype=np.uint8)
    cv2.line(template, (0, 0), (10, 8), (255, 255, 255), 1)
    return template


def test_template_detector_returns_two_non_overlapping_normalized_matches() -> None:
    template = _template()
    image = np.zeros((60, 100, 3), dtype=np.uint8)
    image[5:14, 10:21] = template
    image[35:44, 70:81] = template
    detector = TemplateDetector(
        {"loot": template},
        confidence_threshold=0.99,
        nms_iou_threshold=0.3,
    )

    detections = detector.detect(image)

    assert len(detections) == 2
    assert detections[0].kind == detections[1].kind == "loot"
    assert sorted(detection.center.x for detection in detections) == pytest.approx(
        [15 / 99, 75 / 99]
    )
    assert sorted(detection.center.y for detection in detections) == pytest.approx(
        [9 / 59, 39 / 59]
    )
    assert all(0.99 <= detection.confidence <= 1.0 for detection in detections)


def test_template_detector_suppresses_overlapping_response_peaks() -> None:
    template = _template()
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    image[12:21, 20:31] = template
    detector = TemplateDetector(
        {"restart": template},
        confidence_threshold=0.8,
        nms_iou_threshold=0.2,
    )

    detections = detector.detect(image)

    assert len(detections) == 1
    assert detections[0].center.x == pytest.approx(0.5, abs=0.02)


def _moving_frames() -> Iterator[NDArray[np.uint8]]:
    for left in (8, 34):
        image = np.zeros((50, 80, 3), dtype=np.uint8)
        image[18:28, left : left + 10] = (0, 0, 255)
        yield image


def test_motion_color_detector_requires_colored_sprite_to_move() -> None:
    first, second = _moving_frames()
    detector = MotionColorDetector(
        kind="enemy",
        hsv_lower=(0, 240, 240),
        hsv_upper=(5, 255, 255),
        motion_threshold=20,
        min_area=20,
    )

    assert detector.detect(first) == ()
    detections = detector.detect(second)

    assert len(detections) == 1
    assert detections[0].kind == "enemy"
    assert detections[0].center.x == pytest.approx(38.5 / 79)
    assert detections[0].center.y == pytest.approx(22.5 / 49)
    assert 0.0 <= detections[0].confidence <= 1.0


def test_screen_state_detector_finds_generated_death_and_restart_templates() -> None:
    death = _template()
    restart = cv2.flip(death, 1)
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[10:19, 15:26] = death
    image[50:59, 85:96] = restart
    detector = ScreenStateDetector(
        death_template=death,
        restart_template=restart,
        confidence_threshold=0.99,
        nms_iou_threshold=0.3,
    )

    detections = detector.detect(image)

    assert {detection.kind for detection in detections} == {"death", "restart"}


class _FakeYoloBackend:
    def __init__(self, boxes: tuple[YoloBox, ...]) -> None:
        self.boxes = boxes
        self.images: list[NDArray[np.uint8]] = []

    def predict(self, image: NDArray[np.uint8]) -> tuple[YoloBox, ...]:
        self.images.append(image)
        return self.boxes


def test_yolo_detector_maps_backend_boxes_to_normalized_detections() -> None:
    backend = _FakeYoloBackend(
        (
            YoloBox("enemy", 10, 20, 30, 40, 0.91),
            YoloBox("loot", 80, 5, 90, 15, 0.44),
        )
    )
    detector = YoloDetector(backend, confidence_threshold=0.35)
    image = np.zeros((50, 100, 3), dtype=np.uint8)

    detections = detector.detect(image)

    assert [item.kind for item in detections] == ["enemy", "loot"]
    assert detections[0].center.x == pytest.approx(19.5 / 99)
    assert detections[0].center.y == pytest.approx(29.5 / 49)
    assert detections[0].confidence == pytest.approx(0.91)
    assert detections[0].bbox == Rect(10, 20, 20, 20)
    assert detections[1].bbox == Rect(80, 5, 10, 10)
    assert backend.images[0] is image


def test_yolo_detector_drops_boxes_below_confidence() -> None:
    backend = _FakeYoloBackend((YoloBox("enemy", 0, 0, 8, 8, 0.2),))
    detector = YoloDetector(backend, confidence_threshold=0.35)
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    assert detector.detect(image) == ()
