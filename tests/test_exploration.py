import cv2
import numpy as np
from numpy.typing import NDArray

from hero_siege_bot.config import ExplorationConfig
from hero_siege_bot.domain import Point
from hero_siege_bot.exploration import FrontierExplorer, MapMasks, segment_minimap


def exploration_config(**overrides: object) -> ExplorationConfig:
    values: dict[str, object] = {
        "movement_pulse_s": 0.15,
        "max_movement_pulse_s": 0.15,
        "stuck_timeout_s": 1.5,
        "fog_hsv_lower": (0, 0, 0),
        "fog_hsv_upper": (179, 255, 40),
        "explored_hsv_lower": (0, 0, 180),
        "explored_hsv_upper": (179, 80, 255),
        "morphology_kernel_size": 1,
        "morphology_iterations": 1,
        "frontier_reveal_radius": 2,
        "frontier_path_weight": 1.0,
        "frontier_reveal_weight": 1.0,
        "frontier_failure_penalty": 100.0,
        "frontier_blacklist_radius": 0.2,
        "progress_min_distance": 0.02,
        "progress_min_reveal_pixels": 1,
        "no_progress_sample_limit": 3,
    }
    values.update(overrides)
    return ExplorationConfig(**values)  # type: ignore[arg-type]


def masks(
    explored: NDArray[np.bool_],
    fog: NDArray[np.bool_],
    walkable: NDArray[np.bool_] | None = None,
) -> MapMasks:
    return MapMasks(explored=explored, fog=fog, walkable=explored if walkable is None else walkable)


def test_segment_minimap_classifies_synthetic_room_and_corridor_exactly() -> None:
    image = np.full((9, 9, 3), 10, dtype=np.uint8)
    expected_explored = np.zeros((9, 9), dtype=np.bool_)
    expected_explored[2:7, 2:7] = True
    expected_explored[4, 7:9] = True
    image[expected_explored] = (220, 220, 220)

    result = segment_minimap(image, exploration_config())

    np.testing.assert_array_equal(result.explored, expected_explored)
    np.testing.assert_array_equal(result.fog, ~expected_explored)
    np.testing.assert_array_equal(result.walkable, expected_explored)


def test_segment_minimap_applies_configured_open_and_close_morphology() -> None:
    image = np.full((11, 11, 3), 10, dtype=np.uint8)
    image[3:8, 3:8] = (220, 220, 220)
    image[5, 5] = (10, 10, 10)
    image[1, 1] = (220, 220, 220)
    expected_explored = np.zeros((11, 11), dtype=np.bool_)
    expected_explored[3:8, 3:8] = True

    result = segment_minimap(
        image,
        exploration_config(morphology_kernel_size=3),
    )

    np.testing.assert_array_equal(result.explored, expected_explored)
    np.testing.assert_array_equal(result.fog, ~expected_explored)


def test_frontier_pixels_are_walkable_explored_pixels_next_to_fog() -> None:
    explored = np.zeros((9, 9), dtype=np.bool_)
    explored[2:7, 2:7] = True
    fog = ~explored
    explorer = FrontierExplorer(exploration_config())

    frontier = explorer.frontier_mask(masks(explored, fog))
    adjacent_fog = cv2.dilate(
        fog.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
    ).astype(np.bool_)

    assert frontier.any()
    assert np.all(frontier <= explored)
    assert np.all(frontier <= adjacent_fog)
    assert not frontier[4, 4]


def test_choose_target_ignores_closer_frontier_behind_wall() -> None:
    explored = np.zeros((15, 15), dtype=np.bool_)
    explored[7, 1:7] = True
    explored[6:9, 8:12] = True
    fog = np.zeros((15, 15), dtype=np.bool_)
    fog[6, 1] = True
    fog[5, 8:12] = True
    player = Point(6 / 14, 0.5)
    explorer = FrontierExplorer(exploration_config())

    target = explorer.choose_target(masks(explored, fog), player)

    assert target is not None
    target_pixel = (
        round(target.y * (explored.shape[0] - 1)),
        round(target.x * (explored.shape[1] - 1)),
    )
    assert target_pixel[1] <= 2


def test_choose_target_returns_none_when_player_has_no_reachable_frontier() -> None:
    explored = np.zeros((7, 7), dtype=np.bool_)
    explored[1:3, 1:3] = True
    explored[5, 5] = True
    fog = np.zeros((7, 7), dtype=np.bool_)
    fog[4, 5] = True
    explorer = FrontierExplorer(exploration_config())

    assert explorer.choose_target(masks(explored, fog), Point(0.2, 0.2)) is None


def test_three_no_progress_samples_exclude_current_frontier() -> None:
    explored = np.zeros((9, 13), dtype=np.bool_)
    explored[4, 1:12] = True
    fog = np.zeros((9, 13), dtype=np.bool_)
    fog[3, 1] = True
    fog[3, 11] = True
    player = Point(0.5, 0.5)
    explorer = FrontierExplorer(exploration_config(frontier_blacklist_radius=0.3))
    map_masks = masks(explored, fog)
    first_target = explorer.choose_target(map_masks, player)
    assert first_target is not None

    assert not explorer.record_progress(player, map_masks)
    assert not explorer.record_progress(player, map_masks)
    assert not explorer.record_progress(player, map_masks)
    second_target = explorer.choose_target(map_masks, player)

    assert second_target is not None
    assert abs(second_target.x - first_target.x) > 0.3


def test_explicit_blacklist_excludes_current_frontier_immediately() -> None:
    explored = np.zeros((9, 13), dtype=np.bool_)
    explored[4, 1:12] = True
    fog = np.zeros((9, 13), dtype=np.bool_)
    fog[3, 1] = True
    fog[3, 11] = True
    player = Point(0.5, 0.5)
    explorer = FrontierExplorer(exploration_config(frontier_blacklist_radius=0.3))
    map_masks = masks(explored, fog)
    first_target = explorer.choose_target(map_masks, player)
    assert first_target is not None

    explorer.blacklist_current_target()
    second_target = explorer.choose_target(map_masks, player)

    assert second_target is not None
    assert abs(second_target.x - first_target.x) > 0.3


def test_recent_failure_penalty_ranks_down_nearby_reachable_frontier() -> None:
    explored = np.zeros((11, 21), dtype=np.bool_)
    explored[5, 1:20] = True
    player = Point(0.5, 0.5)
    explorer = FrontierExplorer(
        exploration_config(
            frontier_blacklist_radius=0.1,
            frontier_failure_penalty=10.0,
        )
    )
    initial_fog = np.zeros_like(explored)
    initial_fog[4, 4] = True
    initial_masks = masks(explored, initial_fog)
    assert explorer.choose_target(initial_masks, player) == Point(0.25, 0.5)
    for _ in range(3):
        assert not explorer.record_progress(player, initial_masks)

    equivalent_fog = np.zeros_like(explored)
    equivalent_fog[4, 7] = True
    equivalent_fog[4, 13] = True
    target = explorer.choose_target(masks(explored, equivalent_fog), player)

    assert target == Point(0.6, 0.5)


def test_revealed_area_counts_as_progress_without_player_motion() -> None:
    explored = np.zeros((7, 7), dtype=np.bool_)
    explored[3, 1:6] = True
    fog = np.zeros((7, 7), dtype=np.bool_)
    fog[2, 1] = True
    explorer = FrontierExplorer(exploration_config())
    player = Point(0.5, 0.5)
    initial = masks(explored, fog)
    assert explorer.choose_target(initial, player) is not None
    revealed = explored.copy()
    revealed[2, 1] = True

    assert explorer.record_progress(player, masks(revealed, fog & ~revealed))


def test_movement_actions_allow_diagonal_wasd_and_clamp_pulses() -> None:
    explorer = FrontierExplorer(exploration_config())

    actions = explorer.movement_action(Point(0.5, 0.5), Point(0.8, 0.2))

    assert tuple(action.key for action in actions) == ("W", "D")
    assert all(action.kind == "key_hold" for action in actions)
    assert all(action.duration_s == 0.15 for action in actions)


def test_movement_action_safely_returns_no_keys_for_zero_vector() -> None:
    explorer = FrontierExplorer(exploration_config())

    assert explorer.movement_action(Point(0.5, 0.5), Point(0.5, 0.5)) == ()
