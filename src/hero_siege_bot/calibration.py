from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from hero_siege_bot.capture import CapturedFrame
from hero_siege_bot.config import CalibrationConfig
from hero_siege_bot.domain import Rect

_FALLBACK_REGION_NAMES = frozenset(
    {"health", "resource", "minimap", "gameplay", "screen_state"}
)


@dataclass(frozen=True)
class AnchorRegion:
    anchor: str
    x: float
    y: float
    width: float
    height: float
    clip_to_frame: bool = False
    edge_clip_tolerance: float | None = None


@dataclass(frozen=True)
class NormalizedRegion:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("normalized region values must be finite")
        if not 0.0 <= self.x < 1.0 or not 0.0 <= self.y < 1.0:
            raise ValueError("normalized region origins must be in [0.0, 1.0)")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("normalized region extents must be positive")
        if self.x + self.width > 1.0 or self.y + self.height > 1.0:
            raise ValueError("normalized region must fit within the frame")


@dataclass(frozen=True)
class Calibration:
    regions: Mapping[str, Rect]
    scale: float
    confidence: float
    method: str = "template"


@dataclass(frozen=True)
class CalibrationProfile:
    name: str
    anchors: Mapping[str, NDArray[np.uint8]]
    regions: Mapping[str, AnchorRegion]


@dataclass(frozen=True)
class _Match:
    rect: Rect
    scale: float
    confidence: float


@dataclass(frozen=True)
class _Profile:
    name: str
    anchors: Mapping[str, NDArray[np.uint8]]
    regions: Mapping[str, AnchorRegion]


class AutoCalibrator:
    def __init__(
        self,
        config: CalibrationConfig,
        anchors: Mapping[str, NDArray[np.uint8]] | None = None,
        regions: Mapping[str, AnchorRegion] | None = None,
        *,
        profiles: Sequence[CalibrationProfile] | None = None,
        fallback_regions: Mapping[str, NormalizedRegion] | None = None,
        fallback_confidence: float = 0.9,
    ) -> None:
        if (
            not math.isfinite(fallback_confidence)
            or not 0.9 <= fallback_confidence <= 1.0
        ):
            raise ValueError("fallback_confidence must be finite and between 0.9 and 1.0")
        self._config = config
        if profiles is not None:
            if anchors is not None or regions is not None:
                raise ValueError("use either profiles or anchors and regions")
            definitions = tuple(profiles)
        else:
            if anchors is None or regions is None:
                raise ValueError("anchors and regions are required")
            definitions = (CalibrationProfile("default", anchors, regions),)
        if not definitions:
            raise ValueError("at least one calibration profile is required")
        self._profiles = tuple(self._prepare_profile(profile) for profile in definitions)
        if (
            fallback_regions is not None
            and fallback_regions.keys() != _FALLBACK_REGION_NAMES
        ):
            raise ValueError(
                "fallback regions must contain exactly: "
                + ", ".join(sorted(_FALLBACK_REGION_NAMES))
            )
        self._fallback_regions = (
            MappingProxyType(dict(fallback_regions))
            if fallback_regions is not None
            else None
        )
        self._fallback_confidence = fallback_confidence

    @classmethod
    def _prepare_profile(cls, profile: CalibrationProfile) -> _Profile:
        if not profile.anchors:
            raise ValueError(f"profile {profile.name!r} requires at least one anchor")
        unknown_anchors = {
            region.anchor
            for region in profile.regions.values()
            if region.anchor not in profile.anchors
        }
        if unknown_anchors:
            names = ", ".join(sorted(unknown_anchors))
            raise ValueError(
                f"profile {profile.name!r} regions refer to unknown anchors: {names}"
            )
        return _Profile(
            name=profile.name,
            anchors=MappingProxyType(
                {
                    name: cls._grayscale(template)
                    for name, template in profile.anchors.items()
                }
            ),
            regions=MappingProxyType(dict(profile.regions)),
        )

    @staticmethod
    def _grayscale(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] == 3:
            return cast(
                NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            )
        raise ValueError("anchor and frame images must be grayscale or BGR")

    def _scales(self) -> list[float]:
        count = round(
            (self._config.max_scale - self._config.min_scale)
            / self._config.scale_step
        )
        scales = [
            self._config.min_scale + index * self._config.scale_step
            for index in range(count + 1)
        ]
        if not scales or scales[-1] < self._config.max_scale:
            scales.append(self._config.max_scale)
        return scales

    def _match(
        self, frame: NDArray[np.uint8], template: NDArray[np.uint8]
    ) -> _Match | None:
        best: _Match | None = None
        seen_sizes: set[tuple[int, int]] = set()
        template_height, template_width = template.shape

        for candidate_scale in self._scales():
            width = max(1, round(template_width * candidate_scale))
            height = max(1, round(template_height * candidate_scale))
            if (width, height) in seen_sizes:
                continue
            seen_sizes.add((width, height))
            if width > frame.shape[1] or height > frame.shape[0]:
                continue

            scaled = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)
            scores = cv2.matchTemplate(frame, scaled, cv2.TM_CCOEFF_NORMED)
            _, confidence, _, location = cv2.minMaxLoc(scores)
            scale = ((width / template_width) + (height / template_height)) / 2.0
            match = _Match(
                rect=Rect(location[0], location[1], width, height),
                scale=scale,
                confidence=float(confidence),
            )
            if best is None or match.confidence > best.confidence:
                best = match

        if best is None or best.confidence < self._config.confidence_threshold:
            return None
        return best

    def _detect(
        self, frame: CapturedFrame, profile: _Profile
    ) -> dict[str, _Match] | None:
        grayscale = self._grayscale(frame.image)
        matches: dict[str, _Match] = {}
        for name, template in profile.anchors.items():
            match = self._match(grayscale, template)
            if match is None:
                return None
            matches[name] = match
        return matches

    def _stable(
        self,
        frames: Sequence[CapturedFrame],
        detections: Sequence[dict[str, _Match]],
        profile: _Profile,
    ) -> bool:
        first_shape = frames[0].image.shape[:2]
        if any(frame.image.shape[:2] != first_shape for frame in frames[1:]):
            return False

        frame_height, frame_width = first_shape
        x_tolerance = max(2.0, frame_width * 0.005)
        y_tolerance = max(2.0, frame_height * 0.005)
        scale_tolerance = max(0.02, self._config.scale_step * 1.5)
        for name in profile.anchors:
            matches = [frame_matches[name] for frame_matches in detections]
            x_values = [match.rect.x for match in matches]
            y_values = [match.rect.y for match in matches]
            scale_values = [match.scale for match in matches]
            if max(x_values) - min(x_values) > x_tolerance:
                return False
            if max(y_values) - min(y_values) > y_tolerance:
                return False
            if max(scale_values) - min(scale_values) > scale_tolerance:
                return False
        return True

    def calibrate(self, frames: Sequence[CapturedFrame]) -> Calibration | None:
        required = max(3, self._config.min_stable_frames)
        if len(frames) < required:
            return None

        candidates: list[Calibration] = []
        for profile in self._profiles:
            detections = [self._detect(frame, profile) for frame in frames]
            for start in range(len(frames) - required + 1):
                detected_window = detections[start : start + required]
                if any(matches is None for matches in detected_window):
                    continue
                stable_detections = [
                    matches for matches in detected_window if matches is not None
                ]
                frame_window = frames[start : start + required]
                if not self._stable(frame_window, stable_detections, profile):
                    continue
                frame_height, frame_width = frame_window[0].image.shape[:2]
                candidates.append(
                    self._build(
                        stable_detections,
                        profile,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )
                )
        template = max(
            candidates,
            key=lambda candidate: candidate.confidence,
            default=None,
        )
        if template is not None:
            return template
        return self._build_fallback(frames[-required:])

    def _build_fallback(
        self, frames: Sequence[CapturedFrame]
    ) -> Calibration | None:
        if self._fallback_regions is None or not frames:
            return None
        if any(not frame.focused for frame in frames):
            return None
        first_shape = frames[0].image.shape[:2]
        first_client_rect = frames[0].client_rect
        if any(frame.image.shape[:2] != first_shape for frame in frames[1:]):
            return None
        if any(frame.client_rect != first_client_rect for frame in frames[1:]):
            return None

        frame_height, frame_width = first_shape
        regions = {
            name: self._clip_to_frame(
                Rect(
                    x=round(region.x * frame_width),
                    y=round(region.y * frame_height),
                    width=round(region.width * frame_width),
                    height=round(region.height * frame_height),
                ),
                frame_width,
                frame_height,
            )
            for name, region in self._fallback_regions.items()
        }
        return Calibration(
            regions=MappingProxyType(regions),
            scale=1.0,
            confidence=self._fallback_confidence,
            method="proportional",
        )

    def _build(
        self,
        detections: Sequence[dict[str, _Match]],
        profile: _Profile,
        *,
        frame_width: int,
        frame_height: int,
    ) -> Calibration:
        anchor_rects: dict[str, Rect] = {}
        all_matches: list[_Match] = []
        for name in profile.anchors:
            matches = [frame_matches[name] for frame_matches in detections]
            all_matches.extend(matches)
            anchor_rects[name] = Rect(
                x=round(sum(match.rect.x for match in matches) / len(matches)),
                y=round(sum(match.rect.y for match in matches) / len(matches)),
                width=round(sum(match.rect.width for match in matches) / len(matches)),
                height=round(sum(match.rect.height for match in matches) / len(matches)),
            )

        regions: dict[str, Rect] = {}
        for name, definition in profile.regions.items():
            anchor = anchor_rects[definition.anchor]
            region = Rect(
                x=round(anchor.x + definition.x * anchor.width),
                y=round(anchor.y + definition.y * anchor.height),
                width=round(definition.width * anchor.width),
                height=round(definition.height * anchor.height),
            )
            if definition.clip_to_frame:
                region = self._clip_to_frame(region, frame_width, frame_height)
            elif definition.edge_clip_tolerance is not None:
                region = self._clip_tolerated_edges(
                    region,
                    frame_width,
                    frame_height,
                    definition.edge_clip_tolerance,
                )
            regions[name] = region

        return Calibration(
            regions=MappingProxyType(regions),
            scale=sum(match.scale for match in all_matches) / len(all_matches),
            confidence=min(match.confidence for match in all_matches),
        )

    @staticmethod
    def _clip_to_frame(region: Rect, frame_width: int, frame_height: int) -> Rect:
        left = min(frame_width, max(0, region.x))
        top = min(frame_height, max(0, region.y))
        right = min(frame_width, max(0, region.x + region.width))
        bottom = min(frame_height, max(0, region.y + region.height))
        return Rect(
            x=left,
            y=top,
            width=max(0, right - left),
            height=max(0, bottom - top),
        )

    @classmethod
    def _clip_tolerated_edges(
        cls,
        region: Rect,
        frame_width: int,
        frame_height: int,
        tolerance: float,
    ) -> Rect:
        overscan = (
            (max(0, -region.x), region.width),
            (max(0, region.x + region.width - frame_width), region.width),
            (max(0, -region.y), region.height),
            (max(0, region.y + region.height - frame_height), region.height),
        )
        if any(pixels > dimension * tolerance for pixels, dimension in overscan):
            return region
        return cls._clip_to_frame(region, frame_width, frame_height)
