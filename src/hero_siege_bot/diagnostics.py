from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from hero_siege_bot.calibration import Calibration
from hero_siege_bot.capture import CapturedFrame
from hero_siege_bot.domain import Action, BotState, Detection, Observation, Point, Rect

_DEFAULT_OVERLAY = object()
CHROMA_KEY_BGR = (255, 0, 255)


def compose_live_overlay(
    observation: Observation,
    height: int,
    width: int,
) -> NDArray[np.uint8]:
    """Boxes on a chroma-key background so a click-through overlay stays transparent."""
    rendered = np.full((height, width, 3), CHROMA_KEY_BGR, dtype=np.uint8)
    painter = DiagnosticsOverlay()
    for detection in observation.yolo:
        painter._yolo_box(rendered, detection)
    status = f"yolo {len(observation.yolo)}"
    cv2.putText(
        rendered,
        status,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_8,
    )
    return rendered


class DiagnosticsOverlay:
    """Render decision evidence without mutating the captured frame."""

    def render(
        self,
        image: NDArray[np.uint8],
        calibration: Calibration,
        observation: Observation,
        state: BotState,
        actions: Sequence[Action],
        *,
        frontier: Point | None = None,
        target: Point | None = None,
    ) -> NDArray[np.uint8]:
        rendered = image.copy()
        for name, region in calibration.regions.items():
            self._rectangle(rendered, region, (255, 180, 0))
            cv2.putText(
                rendered,
                name,
                (region.x, max(10, region.y - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 180, 0),
                1,
                cv2.LINE_AA,
            )

        gameplay = Rect(0, 0, rendered.shape[1], rendered.shape[0])
        for detection in observation.enemies:
            self._marker(rendered, gameplay, detection.center, (0, 0, 255), "enemy")
        for detection in observation.loot:
            self._marker(rendered, gameplay, detection.center, (0, 255, 255), "loot")
        for detection in observation.yolo:
            self._yolo_box(rendered, detection)

        minimap = calibration.regions.get("minimap")
        if minimap is not None:
            self._frontier_mask(rendered, minimap, observation)
            if frontier is not None:
                self._marker(rendered, minimap, frontier, (255, 0, 255), "frontier")
            if target is not None:
                self._marker(rendered, minimap, target, (0, 255, 0), "target")

        action_text = ", ".join(
            f"{action.kind}:{action.key or '-'}" for action in actions
        ) or "none"
        lines = (
            f"state={state.value}",
            f"calibration={observation.calibration_confidence:.2f}",
            (
                f"enemy={self._best_confidence(observation.enemies):.2f} "
                f"loot={self._best_confidence(observation.loot):.2f} "
                f"yolo={len(observation.yolo)}"
            ),
            f"actions={action_text}",
        )
        for index, line in enumerate(lines):
            cv2.putText(
                rendered,
                line,
                (4, 12 + index * 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return rendered

    _YOLO_COLORS = {
        "player": (0, 255, 0),
        "companion": (255, 180, 0),
        "enemy": (0, 0, 255),
        "loot": (0, 255, 255),
        "vein": (0, 215, 255),
        "chest": (180, 105, 255),
        "stash": (255, 0, 255),
        "waypoint": (255, 255, 0),
    }

    def _yolo_box(self, image: NDArray[np.uint8], detection: Detection) -> None:
        color = self._YOLO_COLORS.get(detection.kind, (200, 200, 200))
        box = detection.bbox
        if box is None:
            gameplay = Rect(0, 0, image.shape[1], image.shape[0])
            self._marker(image, gameplay, detection.center, color, detection.kind)
            return
        self._rectangle(image, box, color)
        cv2.putText(
            image,
            f"{detection.kind} {detection.confidence:.2f}",
            (box.x, max(10, box.y - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _rectangle(
        image: NDArray[np.uint8], region: Rect, color: tuple[int, int, int]
    ) -> None:
        cv2.rectangle(
            image,
            (region.x, region.y),
            (region.x + region.width - 1, region.y + region.height - 1),
            color,
            1,
        )

    @staticmethod
    def _marker(
        image: NDArray[np.uint8],
        region: Rect,
        point: Point,
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        x = region.x + round(point.x * max(0, region.width - 1))
        y = region.y + round(point.y * max(0, region.height - 1))
        cv2.circle(image, (x, y), 3, color, 1)
        cv2.putText(
            image,
            label,
            (x + 4, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            color,
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _best_confidence(detections: Sequence[Detection]) -> float:
        return max(
            (item.confidence for item in detections),
            default=0.0,
        )

    @staticmethod
    def _frontier_mask(
        image: NDArray[np.uint8],
        region: Rect,
        observation: Observation,
    ) -> None:
        masks = observation.map_masks
        if masks is None or region.width <= 0 or region.height <= 0:
            return
        adjacent_fog = cv2.dilate(
            masks.fog.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
        ).astype(np.bool_)
        frontier = masks.explored & masks.walkable & adjacent_fog
        resized = cv2.resize(
            frontier.astype(np.uint8),
            (region.width, region.height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.bool_)
        height, width = image.shape[:2]
        left = max(0, region.x)
        top = max(0, region.y)
        right = min(width, region.x + region.width)
        bottom = min(height, region.y + region.height)
        if left >= right or top >= bottom:
            return
        mask = resized[
            top - region.y : bottom - region.y,
            left - region.x : right - region.x,
        ]
        crop = image[top:bottom, left:right]
        crop[mask] = (255, 0, 255)


class JsonlRecorder:
    def __init__(
        self,
        root: Path,
        *,
        frame_interval_s: float,
        overlay: DiagnosticsOverlay | None | object = _DEFAULT_OVERLAY,
        now: datetime | None = None,
    ) -> None:
        if frame_interval_s <= 0.0:
            raise ValueError("frame_interval_s must be positive")
        session_time = now or datetime.now(UTC)
        self.session_dir = root / session_time.strftime("%Y%m%dT%H%M%S.%fZ")
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self._events_path = self.session_dir / "events.jsonl"
        self._evidence_dir = self.session_dir / "evidence"
        self._evidence_dir.mkdir()
        self._frame_interval_s = frame_interval_s
        self._overlay = (
            DiagnosticsOverlay()
            if overlay is _DEFAULT_OVERLAY
            else cast(DiagnosticsOverlay | None, overlay)
        )
        self._last_frame_timestamp: float | None = None
        self._frame_number = 0

    def record(
        self,
        observation: Observation,
        state: BotState,
        actions: Sequence[Action],
    ) -> None:
        event = {
            "timestamp": observation.timestamp,
            "state": state.value,
            "focused": observation.focused,
            "calibrated": observation.calibrated,
            "calibration_confidence": observation.calibration_confidence,
            "health_ratio": observation.health_ratio,
            "resource_ratio": observation.resource_ratio,
            "movement_progress": observation.movement_progress,
            "player_map_position": self._point(observation.player_map_position),
            "enemies": [asdict(item) for item in observation.enemies],
            "loot": [asdict(item) for item in observation.loot],
            "yolo": [asdict(item) for item in observation.yolo],
            "dead": observation.dead,
            "restart_visible": observation.restart_visible,
            "restart_target": self._point(observation.restart_target),
            "actions": [asdict(action) for action in actions],
        }
        with self._events_path.open("a", encoding="utf-8") as events:
            events.write(json.dumps(event, separators=(",", ":")) + "\n")

    def record_frame(
        self,
        frame: CapturedFrame,
        observation: Observation,
        state: BotState,
        actions: Sequence[Action],
        calibration: Calibration,
        *,
        frontier: Point | None = None,
        target: Point | None = None,
    ) -> None:
        previous = self._last_frame_timestamp
        if previous is not None and frame.timestamp - previous < self._frame_interval_s:
            return
        self._last_frame_timestamp = frame.timestamp
        rendered = (
            frame.image.copy()
            if self._overlay is None
            else self._overlay.render(
                frame.image,
                calibration,
                observation,
                state,
                actions,
                frontier=frontier,
                target=target,
            )
        )
        path = self._evidence_dir / f"{self._frame_number:06d}.png"
        self._frame_number += 1
        if not cv2.imwrite(str(path), rendered):
            raise OSError(f"failed to write evidence frame: {path}")

    @staticmethod
    def _point(point: Point | None) -> dict[str, float] | None:
        return cast(dict[str, float], asdict(point)) if point is not None else None
