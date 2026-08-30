from types import MappingProxyType

import pytest

from hero_siege_bot.config import CombatConfig, SurvivalConfig
from hero_siege_bot.controllers import CombatController, LootController, SurvivalController
from hero_siege_bot.domain import Action, Detection, Observation, Point


def combat_config() -> CombatConfig:
    return CombatConfig(
        detection_confidence=0.7,
        attack_hold_s=0.25,
        skill_cooldowns_s=MappingProxyType({"Q": 5.0, "E": 8.0}),
        combat_timeout_s=10.0,
        loot_timeout_s=3.0,
    )


def observation(**overrides: object) -> Observation:
    values: dict[str, object] = {
        "timestamp": 0.0,
        "calibrated": True,
        "calibration_confidence": 1.0,
        "focused": True,
        "health_ratio": 1.0,
        "resource_ratio": 1.0,
        "player_map_position": Point(0.5, 0.5),
        "enemies": (),
        "loot": (),
        "dead": False,
        "restart_visible": False,
        "movement_progress": 1.0,
    }
    values.update(overrides)
    return Observation(**values)  # type: ignore[arg-type]


def detection(kind: str, x: float, confidence: float) -> Detection:
    return Detection(kind=kind, center=Point(x, 0.5), confidence=confidence)


def test_combat_targets_highest_confidence_enemy_and_bounds_attack_hold() -> None:
    controller = CombatController(combat_config())
    strongest = detection("enemy", 0.8, 0.95)
    observed = observation(
        enemies=(
            detection("enemy", 0.4, 0.8),
            strongest,
            detection("enemy", 0.2, 0.6),
        )
    )

    actions = controller.actions(observed, now=0.0)

    assert actions[:2] == (
        Action(kind="mouse_move", target=strongest.center),
        Action(kind="mouse_hold", key="left", duration_s=0.25),
    )


def test_combat_skills_respect_independent_cooldowns() -> None:
    controller = CombatController(combat_config())
    observed = observation(enemies=(detection("enemy", 0.5, 0.9),))

    first = controller.actions(observed, now=0.0)
    too_soon = controller.actions(observed, now=4.9)
    q_ready = controller.actions(observed, now=5.0)
    e_ready = controller.actions(observed, now=8.0)

    assert tuple(action.key for action in first[2:]) == ("Q", "E")
    assert tuple(action.key for action in too_soon[2:]) == ()
    assert tuple(action.key for action in q_ready[2:]) == ("Q",)
    assert tuple(action.key for action in e_ready[2:]) == ("E",)


def test_combat_stops_emitting_actions_at_encounter_timeout() -> None:
    controller = CombatController(combat_config())
    observed = observation(enemies=(detection("enemy", 0.5, 0.9),))

    assert controller.actions(observed, now=20.0)
    assert controller.actions(observed, now=29.9)
    assert controller.actions(observed, now=30.0) == ()
    assert controller.abandoned


def test_combat_timeout_resets_after_enemies_clear() -> None:
    controller = CombatController(combat_config())
    observed = observation(enemies=(detection("enemy", 0.5, 0.9),))

    assert controller.actions(observed, now=20.0)
    assert controller.actions(observed, now=30.0) == ()
    assert controller.actions(observation(), now=31.0) == ()
    assert not controller.abandoned
    assert controller.actions(observed, now=40.0)


@pytest.mark.parametrize(
    "cooldowns",
    [
        {"Q": 5.0},
        {"Q": 5.0, "E": 8.0, "R": 1.0},
    ],
)
def test_combat_config_requires_exact_q_and_e_skills(
    cooldowns: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match="Q/E"):
        CombatConfig(
            detection_confidence=0.7,
            attack_hold_s=0.25,
            skill_cooldowns_s=MappingProxyType(cooldowns),
            combat_timeout_s=10.0,
            loot_timeout_s=3.0,
        )


def test_survival_uses_health_potion_once_per_cooldown() -> None:
    controller = SurvivalController(
        SurvivalConfig(
            health_threshold=0.35,
            resource_threshold=0.25,
            potion_cooldown_s=1.0,
        )
    )
    low_health = observation(health_ratio=0.2)

    first = controller.actions(low_health, now=10.0)
    repeated = controller.actions(low_health, now=10.9)
    cooled_down = controller.actions(low_health, now=11.0)

    assert tuple(action.key for action in first) == ("1",)
    assert repeated == ()
    assert tuple(action.key for action in cooled_down) == ("1",)


def test_loot_times_out_at_configured_deadline() -> None:
    controller = LootController(combat_config())
    loot = detection("loot", 0.7, 0.9)
    observed = observation(loot=(loot,))

    initial = controller.actions(observed, now=20.0)
    before_deadline = controller.actions(observed, now=22.9)
    at_deadline = controller.actions(observed, now=23.0)

    assert initial[0].target == loot.center
    assert initial[1].duration_s == 0.25
    assert before_deadline
    assert at_deadline == ()
    assert controller.abandoned


def test_loot_timeout_resets_after_loot_disappears() -> None:
    controller = LootController(combat_config())
    observed = observation(loot=(detection("loot", 0.7, 0.9),))

    assert controller.actions(observed, now=1.0)
    assert controller.actions(observation(), now=2.0) == ()
    assert not controller.abandoned
    assert controller.actions(observed, now=10.0)
