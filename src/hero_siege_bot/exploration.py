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
    structure = _clean_mask(
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
    fog_seed = _clean_mask(
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
    sealed = cv2.dilate(structure.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(
        np.bool_
    )
    inner = (~sealed) & ~_border_connected(~sealed)
    grown = inner | (
        cv2.dilate(inner.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.bool_)
        & ~structure
    )
    openings = _four_adjacent(grown) & ~structure
    interiors = grown | openings
    explored = structure | interiors
    fog = fog_seed & ~explored
    walkable = interiors.copy()
    if not walkable.any():
        walkable = structure.copy()
    return MapMasks(explored=explored, fog=fog, walkable=walkable)


def _border_connected(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.bool_)
    _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    border = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    border = border[border != 0]
    if border.size == 0:
        return np.zeros(mask.shape, dtype=np.bool_)
    return np.isin(labels, border)


def _four_adjacent(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    adjacent = np.zeros(mask.shape, dtype=np.bool_)
    adjacent[1:, :] |= mask[:-1, :]
    adjacent[:-1, :] |= mask[1:, :]
    adjacent[:, 1:] |= mask[:, :-1]
    adjacent[:, :-1] |= mask[:, 1:]
    return adjacent


class FrontierExplorer:
    def __init__(self, config: ExplorationConfig) -> None:
        self._config = config
        self._current_target: Point | None = None
        self._previous_player: Point | None = None
        self._previous_explored: NDArray[np.bool_] | None = None
        self._no_progress_samples = 0
        self._failed_targets: list[Point] = []

    def frontier_mask(self, masks: MapMasks) -> NDArray[np.bool_]:
        return masks.walkable & _four_adjacent(masks.fog)

    def target_is_valid(self, masks: MapMasks, target: Point) -> bool:
        frontier = self.frontier_mask(masks)
        height, width = frontier.shape
        column = min(width - 1, max(0, round(target.x * (width - 1))))
        row = min(height - 1, max(0, round(target.y * (height - 1))))
        return bool(frontier[row, column])

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

    def movement_action(
        self,
        player: Point,
        target: Point,
        walkable: NDArray[np.bool_] | None = None,
        fog: NDArray[np.bool_] | None = None,
    ) -> tuple[Action, ...]:
        """Return one or two bounded WASD actions, or none for a zero vector."""
        aim = target
        if walkable is not None:
            waypoint = self._first_path_step(walkable, player, target)
            if waypoint is not None:
                aim = waypoint
            elif fog is not None:
                into_fog = self._adjacent_fog_step(walkable, fog, player)
                if into_fog is not None:
                    aim = into_fog
                else:
                    nearest_fog = self._nearest_mask_point(fog, player)
                    if nearest_fog is not None:
                        aim = nearest_fog
        pulse = min(
            self._config.movement_pulse_s,
            self._config.max_movement_pulse_s,
        )
        actions: list[Action] = []
        delta_y = aim.y - player.y
        delta_x = aim.x - player.x
        if delta_y < 0:
            actions.append(Action(kind="key_hold", key="W", duration_s=pulse))
        elif delta_y > 0:
            actions.append(Action(kind="key_hold", key="S", duration_s=pulse))
        if delta_x < 0:
            actions.append(Action(kind="key_hold", key="A", duration_s=pulse))
        elif delta_x > 0:
            actions.append(Action(kind="key_hold", key="D", duration_s=pulse))
        return tuple(actions)

    def _first_path_step(
        self,
        walkable: NDArray[np.bool_],
        player: Point,
        target: Point,
    ) -> Point | None:
        start = self._snapped_cell(walkable, player)
        goal = self._snapped_cell(walkable, target)
        if start is None or goal is None or start == goal:
            return None
        parents = self._cardinal_parents(walkable, start)
        if goal not in parents:
            return None
        step = goal
        while parents[step] != start:
            step = parents[step]
        return self._normalized_point(step[0], step[1], walkable.shape)

    def _adjacent_fog_step(
        self,
        walkable: NDArray[np.bool_],
        fog: NDArray[np.bool_],
        player: Point,
    ) -> Point | None:
        start = self._snapped_cell(walkable, player)
        if start is None:
            return None
        height, width = fog.shape
        row, column = start
        for row_delta, column_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_row = row + row_delta
            next_column = column + column_delta
            if (
                0 <= next_row < height
                and 0 <= next_column < width
                and fog[next_row, next_column]
            ):
                return self._normalized_point(next_row, next_column, fog.shape)
        return None

    def _nearest_mask_point(
        self, mask: NDArray[np.bool_], point: Point
    ) -> Point | None:
        if not mask.any():
            return None
        height, width = mask.shape
        column = min(width - 1, max(0, round(point.x * (width - 1))))
        row = min(height - 1, max(0, round(point.y * (height - 1))))
        cells = np.argwhere(mask)
        offset = cells - np.array((row, column))
        nearest = cells[int(np.argmin(np.sum(offset * offset, axis=1)))]
        return self._normalized_point(int(nearest[0]), int(nearest[1]), mask.shape)

    def _snapped_cell(
        self, walkable: NDArray[np.bool_], point: Point
    ) -> tuple[int, int] | None:
        height, width = walkable.shape
        column = min(width - 1, max(0, round(point.x * (width - 1))))
        row = min(height - 1, max(0, round(point.y * (height - 1))))
        if walkable[row, column]:
            return row, column
        nearby = np.argwhere(walkable)
        if nearby.size == 0:
            return None
        offset = nearby - np.array((row, column))
        nearest = nearby[int(np.argmin(np.sum(offset * offset, axis=1)))]
        return int(nearest[0]), int(nearest[1])

    @staticmethod
    def _cardinal_parents(
        walkable: NDArray[np.bool_], start: tuple[int, int]
    ) -> dict[tuple[int, int], tuple[int, int]]:
        height, width = walkable.shape
        parents: dict[tuple[int, int], tuple[int, int]] = {}
        pending = deque([start])
        seen = {start}
        while pending:
            row, column = pending.popleft()
            for row_delta, column_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                next_row = row + row_delta
                next_column = column + column_delta
                if (
                    0 <= next_row < height
                    and 0 <= next_column < width
                    and walkable[next_row, next_column]
                    and (next_row, next_column) not in seen
                ):
                    seen.add((next_row, next_column))
                    parents[(next_row, next_column)] = (row, column)
                    pending.append((next_row, next_column))
        return parents

    def _reachable_distances(
        self, walkable: NDArray[np.bool_], player: Point
    ) -> NDArray[np.int32]:
        height, width = walkable.shape
        start_column = min(width - 1, max(0, round(player.x * (width - 1))))
        start_row = min(height - 1, max(0, round(player.y * (height - 1))))
        distances = np.full(walkable.shape, -1, dtype=np.int32)
        if not walkable[start_row, start_column]:
            nearby = np.argwhere(walkable)
            if nearby.size == 0:
                return distances
            offset = nearby - np.array((start_row, start_column))
            nearest = nearby[int(np.argmin(np.sum(offset * offset, axis=1)))]
            start_row, start_column = int(nearest[0]), int(nearest[1])

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
