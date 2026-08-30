from collections import deque
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from hero_siege_bot.config import ExplorationConfig
from hero_siege_bot.domain import Action, MapMasks, Point, normalize_pixel_index


def _clean_mask(
    mask: NDArray[np.uint8],
    kernel: NDArray[np.uint8],
    iterations: int,
    border_value: int,
) -> NDArray[np.bool_]:
    closed = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=iterations,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    opened = cv2.morphologyEx(
        closed,
        cv2.MORPH_OPEN,
        kernel,
        iterations=iterations,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return opened.astype(np.bool_)


def segment_minimap(
    image: NDArray[np.uint8], config: ExplorationConfig
) -> MapMasks:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("mini-map image must be a BGR image")
    if image.dtype != np.uint8:
        raise TypeError("mini-map image must use uint8 pixels")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = np.ones(
        (config.morphology_kernel_size, config.morphology_kernel_size),
        dtype=np.uint8,
    )
    explored = _clean_mask(
        cast(
            NDArray[np.uint8],
            cv2.inRange(
                hsv,
                np.asarray(config.explored_hsv_lower, dtype=np.uint8),
                np.asarray(config.explored_hsv_upper, dtype=np.uint8),
            ),
        ),
        kernel,
        config.morphology_iterations,
        0,
    )
    fog = _clean_mask(
        cast(
            NDArray[np.uint8],
            cv2.inRange(
                hsv,
                np.asarray(config.fog_hsv_lower, dtype=np.uint8),
                np.asarray(config.fog_hsv_upper, dtype=np.uint8),
            ),
        ),
        kernel,
        config.morphology_iterations,
        255,
    )
    fog &= ~explored
    return MapMasks(explored=explored, fog=fog, walkable=explored.copy())


class FrontierExplorer:
    def __init__(self, config: ExplorationConfig) -> None:
        self._config = config
        self._current_target: Point | None = None
        self._previous_player: Point | None = None
        self._previous_explored: NDArray[np.bool_] | None = None
        self._no_progress_samples = 0
        self._failed_targets: list[Point] = []

    def frontier_mask(self, masks: MapMasks) -> NDArray[np.bool_]:
        adjacent_fog = cv2.dilate(
            masks.fog.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
        ).astype(np.bool_)
        return masks.explored & masks.walkable & adjacent_fog

    def choose_target(self, masks: MapMasks, player: Point) -> Point | None:
        distances = self._reachable_distances(masks.walkable, player)
        reachable_frontier = self.frontier_mask(masks) & (distances >= 0)
        component_count, labels = cv2.connectedComponents(
            reachable_frontier.astype(np.uint8), connectivity=8
        )

        best_score = float("-inf")
        best_target: Point | None = None
        for label in range(1, component_count):
            rows, columns = np.nonzero(labels == label)
            if rows.size == 0:
                continue
            candidate_index = int(np.argmin(distances[rows, columns]))
            candidate = self._normalized_point(
                int(rows[candidate_index]),
                int(columns[candidate_index]),
                masks.walkable.shape,
            )
            if self._is_blacklisted(candidate):
                continue

            revealable = self._revealable_fog(labels == label, masks.fog)
            path_length = int(distances[rows[candidate_index], columns[candidate_index]])
            failure_penalty = self._failure_penalty(candidate)
            score = (
                self._config.frontier_reveal_weight * revealable
                - self._config.frontier_path_weight * path_length
                - self._config.frontier_failure_penalty * failure_penalty
            )
            if score > best_score:
                best_score = score
                best_target = candidate

        self._current_target = best_target
        self._previous_player = player
        self._previous_explored = masks.explored.copy()
        self._no_progress_samples = 0
        return best_target

    def record_progress(self, player: Point, masks: MapMasks) -> bool:
        if self._previous_player is None or self._previous_explored is None:
            self._previous_player = player
            self._previous_explored = masks.explored.copy()
            return False
        if self._previous_explored.shape != masks.explored.shape:
            self._previous_player = player
            self._previous_explored = masks.explored.copy()
            self._no_progress_samples = 0
            return False

        player_distance = self._point_distance(player, self._previous_player)
        newly_explored = int(np.count_nonzero(masks.explored & ~self._previous_explored))
        progressed = (
            player_distance >= self._config.progress_min_distance
            or newly_explored >= self._config.progress_min_reveal_pixels
        )
        self._previous_player = player
        self._previous_explored = masks.explored.copy()
        if progressed:
            self._no_progress_samples = 0
            return True

        self._no_progress_samples += 1
        if (
            self._no_progress_samples >= self._config.no_progress_sample_limit
            and self._current_target is not None
        ):
            self._failed_targets.append(self._current_target)
            self._current_target = None
            self._no_progress_samples = 0
        return False

    def blacklist_current_target(self) -> None:
        if self._current_target is not None:
            self._failed_targets.append(self._current_target)
            self._current_target = None
        self._no_progress_samples = 0

    def movement_action(self, player: Point, target: Point) -> tuple[Action, ...]:
        """Return one or two bounded WASD actions, or none for a zero vector."""
        pulse = min(
            self._config.movement_pulse_s,
            self._config.max_movement_pulse_s,
        )
        actions: list[Action] = []
        delta_y = target.y - player.y
        delta_x = target.x - player.x
        if delta_y < 0:
            actions.append(Action(kind="key_hold", key="W", duration_s=pulse))
        elif delta_y > 0:
            actions.append(Action(kind="key_hold", key="S", duration_s=pulse))
        if delta_x < 0:
            actions.append(Action(kind="key_hold", key="A", duration_s=pulse))
        elif delta_x > 0:
            actions.append(Action(kind="key_hold", key="D", duration_s=pulse))
        return tuple(actions)

    def _reachable_distances(
        self, walkable: NDArray[np.bool_], player: Point
    ) -> NDArray[np.int32]:
        height, width = walkable.shape
        start_column = min(width - 1, max(0, round(player.x * (width - 1))))
        start_row = min(height - 1, max(0, round(player.y * (height - 1))))
        distances = np.full(walkable.shape, -1, dtype=np.int32)
        if not walkable[start_row, start_column]:
            return distances

        distances[start_row, start_column] = 0
        pending = deque([(start_row, start_column)])
        while pending:
            row, column = pending.popleft()
            for row_delta, column_delta in (
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ):
                next_row = row + row_delta
                next_column = column + column_delta
                if (
                    0 <= next_row < height
                    and 0 <= next_column < width
                    and walkable[next_row, next_column]
                    and distances[next_row, next_column] < 0
                ):
                    distances[next_row, next_column] = distances[row, column] + 1
                    pending.append((next_row, next_column))
        return distances

    def _revealable_fog(
        self, frontier_component: NDArray[np.bool_], fog: NDArray[np.bool_]
    ) -> int:
        radius = self._config.frontier_reveal_radius
        kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
        reveal_region = cv2.dilate(
            frontier_component.astype(np.uint8), kernel
        ).astype(np.bool_)
        return int(np.count_nonzero(reveal_region & fog))

    def _is_blacklisted(self, point: Point) -> bool:
        return any(
            self._point_distance(point, failed)
            <= self._config.frontier_blacklist_radius
            for failed in self._failed_targets
        )

    def _failure_penalty(self, point: Point) -> float:
        influence_radius = self._config.frontier_blacklist_radius * 2.0
        return sum(
            max(
                0.0,
                1.0 - self._point_distance(point, failed) / influence_radius,
            )
            for failed in self._failed_targets
        )

    @staticmethod
    def _normalized_point(row: int, column: int, shape: tuple[int, int]) -> Point:
        height, width = shape
        return Point(
            x=normalize_pixel_index(column, width),
            y=normalize_pixel_index(row, height),
        )

    @staticmethod
    def _point_distance(first: Point, second: Point) -> float:
        return float(np.hypot(first.x - second.x, first.y - second.y))
