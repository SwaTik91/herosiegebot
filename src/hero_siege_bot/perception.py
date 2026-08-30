from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from hero_siege_bot.calibration import Calibration
from hero_siege_bot.capture import CapturedFrame
from hero_siege_bot.config import BotConfig
from hero_siege_bot.detectors import BarReader, Detector
from hero_siege_bot.domain import Detection, Observation, Point, Rect, normalize_pixel_index
from hero_siege_bot.exploration import segment_minimap

_REQUIRED_REGIONS = ("health", "resource", "minimap", "gameplay", "screen_state")


class Perception:
    def __init__(
        self,
        *,
        config: BotConfig,
        enemy_detector: Detector,
        loot_detector: Detector,
        screen_state_detector: Detector,
        yolo_detector: Detector | None = None,
    ) -> None:
        self._config = config
        detector_config = config.detectors
        self._health_reader = BarReader(
            fill_hsv_lower=detector_config.health_fill_hsv_lower,
            fill_hsv_upper=detector_config.health_fill_hsv_upper,
            border_hsv_lower=detector_config.bar_border_hsv_lower,
            border_hsv_upper=detector_config.bar_border_hsv_upper,
            min_border_confidence=detector_config.bar_min_border_confidence,
        )
        self._resource_reader = BarReader(
            fill_hsv_lower=detector_config.resource_fill_hsv_lower,
            fill_hsv_upper=detector_config.resource_fill_hsv_upper,
            border_hsv_lower=detector_config.bar_border_hsv_lower,
            border_hsv_upper=detector_config.bar_border_hsv_upper,
            min_border_confidence=detector_config.bar_min_border_confidence,
        )
        self._enemy_detector = enemy_detector
        self._loot_detector = loot_detector
        self._screen_state_detector = screen_state_detector
        self._yolo_detector = yolo_detector
        self._previous_player: Point | None = None

    def observe(
        self, frame: CapturedFrame, calibration: Calibration
    ) -> Observation:
        crops = self._crops(frame.image, calibration.regions)
        if crops is None:
            return self._empty_observation(frame, calibration)

        health_ratio, _ = self._health_reader.read_ratio(crops["health"])
        resource_ratio, _ = self._resource_reader.read_ratio(crops["resource"])
        map_masks = segment_minimap(crops["minimap"], self._config.exploration)
        player = self._locate_player(crops["minimap"])
        movement_progress = self._movement_progress(player)

        confidence = self._config.combat.detection_confidence
        enemies = self._frame_detections(
            tuple(
                detection
                for detection in self._enemy_detector.detect(crops["gameplay"])
                if detection.confidence >= confidence
            ),
            calibration.regions["gameplay"],
            frame.image.shape,
        )
        loot = self._frame_detections(
            tuple(
                detection
                for detection in self._loot_detector.detect(crops["gameplay"])
                if detection.confidence >= confidence
            ),
            calibration.regions["gameplay"],
            frame.image.shape,
        )
        states = self._screen_state_detector.detect(crops["screen_state"])
        restart_detections = self._frame_detections(
            tuple(detection for detection in states if detection.kind == "restart"),
            calibration.regions["screen_state"],
            frame.image.shape,
        )
        restart = max(
            restart_detections,
            key=lambda detection: detection.confidence,
            default=None,
        )
        yolo = (
            self._yolo_detector.detect(frame.image)
            if self._yolo_detector is not None
            else ()
        )
        return Observation(
            timestamp=frame.timestamp,
            calibrated=True,
            calibration_confidence=calibration.confidence,
            focused=frame.focused,
            health_ratio=health_ratio,
            resource_ratio=resource_ratio,
            player_map_position=player,
            enemies=enemies,
            loot=loot,
            dead=any(detection.kind == "death" for detection in states),
            restart_visible=restart is not None,
            movement_progress=movement_progress,
            map_masks=map_masks,
            restart_target=restart.center if restart is not None else None,
            yolo=yolo,
        )

    def _locate_player(self, minimap: NDArray[np.uint8]) -> Point | None:
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        detector_config = self._config.detectors
        mask = cast(
            NDArray[np.uint8],
            cv2.inRange(
                hsv,
                np.asarray(
                    detector_config.player_marker_hsv_lower, dtype=np.uint8
                ),
                np.asarray(
                    detector_config.player_marker_hsv_upper, dtype=np.uint8
                ),
            ),
        )
        count, _, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        candidates = [
            label
            for label in range(1, count)
            if stats[label, cv2.CC_STAT_AREA]
            >= detector_config.player_marker_min_area
        ]
        if not candidates:
            return None
        label = max(candidates, key=lambda item: stats[item, cv2.CC_STAT_AREA])
        return Point(
            x=normalize_pixel_index(float(centroids[label, 0]), minimap.shape[1]),
            y=normalize_pixel_index(float(centroids[label, 1]), minimap.shape[0]),
        )

    @staticmethod
    def _frame_detections(
        detections: tuple[Detection, ...],
        region: Rect,
        frame_shape: tuple[int, ...],
    ) -> tuple[Detection, ...]:
        frame_height, frame_width = frame_shape[:2]
        return tuple(
            Detection(
                kind=detection.kind,
                center=Point(
                    normalize_pixel_index(
                        region.x + detection.center.x * max(0, region.width - 1),
                        frame_width,
                    ),
                    normalize_pixel_index(
                        region.y + detection.center.y * max(0, region.height - 1),
                        frame_height,
                    ),
                ),
                confidence=detection.confidence,
            )
            for detection in detections
        )

    def _movement_progress(self, player: Point | None) -> float:
        previous = self._previous_player
        self._previous_player = player
        if previous is None or player is None:
            return 0.0
        return float(np.hypot(player.x - previous.x, player.y - previous.y))

    @staticmethod
    def _crops(
        image: NDArray[np.uint8], regions: Mapping[str, Rect]
    ) -> dict[str, NDArray[np.uint8]] | None:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            return None
        height, width = image.shape[:2]
        crops: dict[str, NDArray[np.uint8]] = {}
        for name in _REQUIRED_REGIONS:
            rect = regions.get(name)
            if (
                rect is None
                or rect.x < 0
                or rect.y < 0
                or rect.width <= 0
                or rect.height <= 0
                or rect.x + rect.width > width
                or rect.y + rect.height > height
            ):
                return None
            crops[name] = image[
                rect.y : rect.y + rect.height,
                rect.x : rect.x + rect.width,
            ]
        return crops

    @staticmethod
    def _empty_observation(
        frame: CapturedFrame, calibration: Calibration
    ) -> Observation:
        return Observation(
            timestamp=frame.timestamp,
            calibrated=False,
            calibration_confidence=calibration.confidence,
            focused=frame.focused,
            health_ratio=None,
            resource_ratio=None,
            player_map_position=None,
            enemies=(),
            loot=(),
            dead=False,
            restart_visible=False,
            movement_progress=0.0,
            restart_target=None,
        )
