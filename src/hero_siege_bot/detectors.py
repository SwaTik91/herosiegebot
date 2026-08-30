from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from hero_siege_bot.domain import Detection, Point, normalize_pixel_index

HSV = tuple[int, int, int]


class Detector(Protocol):
    def detect(self, image: NDArray[np.uint8]) -> tuple[Detection, ...]: ...


def _validate_image(image: NDArray[np.uint8]) -> None:
    if image.dtype != np.uint8:
        raise TypeError("detector image must use uint8 pixels")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("detector image must be a BGR image")


def _hsv_mask(image: NDArray[np.uint8], lower: HSV, upper: HSV) -> NDArray[np.uint8]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cast(
        NDArray[np.uint8],
        cv2.inRange(
            hsv,
            np.asarray(lower, dtype=np.uint8),
            np.asarray(upper, dtype=np.uint8),
        ),
    )


class BarReader:
    def __init__(
        self,
        *,
        fill_hsv_lower: HSV,
        fill_hsv_upper: HSV,
        border_hsv_lower: HSV,
        border_hsv_upper: HSV,
        min_border_confidence: float,
    ) -> None:
        if not 0.0 <= min_border_confidence <= 1.0:
            raise ValueError("min_border_confidence must be between 0.0 and 1.0")
        self._fill_hsv_lower = fill_hsv_lower
        self._fill_hsv_upper = fill_hsv_upper
        self._border_hsv_lower = border_hsv_lower
        self._border_hsv_upper = border_hsv_upper
        self._min_border_confidence = min_border_confidence

    def read_ratio(self, image: NDArray[np.uint8]) -> tuple[float | None, float]:
        _validate_image(image)
        border = _hsv_mask(
            image, self._border_hsv_lower, self._border_hsv_upper
        )
        points = cv2.findNonZero(border)
        if points is None:
            return None, 0.0

        x, y, width, height = cv2.boundingRect(points)
        if width < 3 or height < 3:
            return None, 0.0
        top = border[y, x : x + width]
        bottom = border[y + height - 1, x : x + width]
        left = border[y + 1 : y + height - 1, x]
        right = border[y + 1 : y + height - 1, x + width - 1]
        border_pixels = np.concatenate((top, bottom, left, right))
        confidence = float(np.count_nonzero(border_pixels) / border_pixels.size)
        if confidence < self._min_border_confidence:
            return None, confidence

        interior = image[y + 1 : y + height - 1, x + 1 : x + width - 1]
        if interior.size == 0:
            return None, confidence
        fill = _hsv_mask(interior, self._fill_hsv_lower, self._fill_hsv_upper)
        filled_columns = np.mean(fill > 0, axis=0) >= 0.5
        ratio = float(np.count_nonzero(filled_columns) / filled_columns.size)
        return ratio, confidence


@dataclass(frozen=True)
class _Candidate:
    kind: str
    x: int
    y: int
    width: int
    height: int
    confidence: float


def _intersection_over_union(first: _Candidate, second: _Candidate) -> float:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.x + first.width, second.x + second.width)
    bottom = min(first.y + first.height, second.y + second.height)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union else 0.0


def _to_detection(
    candidate: _Candidate, image_width: int, image_height: int
) -> Detection:
    return Detection(
        kind=candidate.kind,
        center=Point(
            x=normalize_pixel_index(
                candidate.x + (candidate.width - 1) / 2.0, image_width
            ),
            y=normalize_pixel_index(
                candidate.y + (candidate.height - 1) / 2.0, image_height
            ),
        ),
        confidence=float(np.clip(candidate.confidence, 0.0, 1.0)),
    )


class TemplateDetector:
    def __init__(
        self,
        templates: Mapping[str, NDArray[np.uint8]],
        *,
        confidence_threshold: float,
        nms_iou_threshold: float,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        if not 0.0 <= nms_iou_threshold <= 1.0:
            raise ValueError("nms_iou_threshold must be between 0.0 and 1.0")
        if not templates:
            raise ValueError("at least one template is required")
        for template in templates.values():
            _validate_image(template)
        self._templates = {kind: template.copy() for kind, template in templates.items()}
        self._confidence_threshold = confidence_threshold
        self._nms_iou_threshold = nms_iou_threshold

    def detect(self, image: NDArray[np.uint8]) -> tuple[Detection, ...]:
        _validate_image(image)
        candidates: list[_Candidate] = []
        for kind, template in self._templates.items():
            height, width = template.shape[:2]
            if height > image.shape[0] or width > image.shape[1]:
                continue
            scores = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            rows, columns = np.nonzero(scores >= self._confidence_threshold)
            candidates.extend(
                _Candidate(
                    kind=kind,
                    x=int(column),
                    y=int(row),
                    width=width,
                    height=height,
                    confidence=float(scores[row, column]),
                )
                for row, column in zip(rows, columns, strict=True)
            )

        kept: list[_Candidate] = []
        for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
            if any(
                candidate.kind == selected.kind
                and _intersection_over_union(candidate, selected)
                > self._nms_iou_threshold
                for selected in kept
            ):
                continue
            kept.append(candidate)
        kept.sort(key=lambda item: (item.kind, item.y, item.x))
        return tuple(
            _to_detection(candidate, image.shape[1], image.shape[0])
            for candidate in kept
        )


class MotionColorDetector:
    def __init__(
        self,
        kind: str,
        *,
        hsv_lower: HSV,
        hsv_upper: HSV,
        motion_threshold: int,
        min_area: int,
    ) -> None:
        if not 0 <= motion_threshold <= 255:
            raise ValueError("motion_threshold must be between 0 and 255")
        if min_area <= 0:
            raise ValueError("min_area must be positive")
        self._kind = kind
        self._hsv_lower = hsv_lower
        self._hsv_upper = hsv_upper
        self._motion_threshold = motion_threshold
        self._min_area = min_area
        self._previous: NDArray[np.uint8] | None = None

    def detect(self, image: NDArray[np.uint8]) -> tuple[Detection, ...]:
        _validate_image(image)
        previous = self._previous
        self._previous = image.copy()
        if previous is None or previous.shape != image.shape:
            return ()

        color = _hsv_mask(image, self._hsv_lower, self._hsv_upper)
        difference = cv2.absdiff(image, previous)
        motion = np.max(difference, axis=2) >= self._motion_threshold
        candidate_mask = ((color > 0) & motion).astype(np.uint8)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(
            candidate_mask, connectivity=8
        )
        detections: list[Detection] = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self._min_area:
                continue
            x = normalize_pixel_index(float(centroids[label, 0]), image.shape[1])
            y = normalize_pixel_index(float(centroids[label, 1]), image.shape[0])
            bbox_area = int(
                stats[label, cv2.CC_STAT_WIDTH] * stats[label, cv2.CC_STAT_HEIGHT]
            )
            detections.append(
                Detection(
                    kind=self._kind,
                    center=Point(x=x, y=y),
                    confidence=float(np.clip(area / bbox_area, 0.0, 1.0)),
                )
            )
        return tuple(detections)


class ScreenStateDetector:
    def __init__(
        self,
        death_template: NDArray[np.uint8],
        restart_template: NDArray[np.uint8],
        *,
        confidence_threshold: float,
        nms_iou_threshold: float,
    ) -> None:
        self._detector = TemplateDetector(
            {"death": death_template, "restart": restart_template},
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
        )

    def detect(self, image: NDArray[np.uint8]) -> tuple[Detection, ...]:
        return self._detector.detect(image)
