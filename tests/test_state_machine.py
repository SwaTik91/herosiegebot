from hero_siege_bot.domain import BotState, Detection, Observation, Point
from hero_siege_bot.state_machine import BotStateMachine


def observation(**overrides: object) -> Observation:
    defaults: dict[str, object] = {
        "timestamp": 0.0,
        "calibrated": True,
        "calibration_confidence": 0.95,
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
    defaults.update(overrides)
    return Observation(**defaults)  # type: ignore[arg-type]


def detection(kind: str, confidence: float = 0.9) -> Detection:
    return Detection(kind, Point(0.5, 0.5), confidence)


def test_calibration_enters_exploration() -> None:
    machine = BotStateMachine()

    assert (
        machine.update(observation(calibrated=True, calibration_confidence=0.95))
        is BotState.EXPLORING
    )


def test_calibration_waits_for_configured_confidence() -> None:
    machine = BotStateMachine(calibration_confidence=0.96)

    assert machine.update(observation(calibration_confidence=0.95)) is BotState.CALIBRATING


def test_enemy_interrupts_exploration() -> None:
    machine = BotStateMachine(BotState.EXPLORING)
    enemy = Detection("enemy", Point(0.5, 0.5), 0.9)

    assert machine.update(observation(enemies=(enemy,))) is BotState.COMBAT


def test_enemy_below_configured_confidence_does_not_interrupt_exploration() -> None:
    machine = BotStateMachine(BotState.EXPLORING, detection_confidence=0.95)

    assert (
        machine.update(observation(enemies=(detection("enemy", 0.9),)))
        is BotState.EXPLORING
    )


def test_focus_loss_always_pauses() -> None:
    for state in BotState:
        machine = BotStateMachine(state)
        assert machine.update(observation(focused=False)) is BotState.PAUSED


def test_lost_calibration_preempts_non_paused_states() -> None:
    for state in BotState:
        if state is BotState.PAUSED:
            continue
        machine = BotStateMachine(state)
        assert machine.update(observation(calibrated=False)) is BotState.CALIBRATING


def test_death_preempts_active_states() -> None:
    machine = BotStateMachine(BotState.COMBAT)

    assert machine.update(observation(dead=True)) is BotState.DEAD


def test_combat_enters_looting_when_enemies_clear() -> None:
    machine = BotStateMachine(BotState.COMBAT)

    assert machine.update(observation()) is BotState.LOOTING


def test_combat_continues_while_confident_enemy_remains() -> None:
    machine = BotStateMachine(BotState.COMBAT)

    assert (
        machine.update(observation(enemies=(detection("enemy"),))) is BotState.COMBAT
    )


def test_looting_returns_to_exploration_when_loot_clears() -> None:
    machine = BotStateMachine(BotState.LOOTING)

    assert machine.update(observation()) is BotState.EXPLORING


def test_looting_continues_while_confident_loot_remains() -> None:
    machine = BotStateMachine(BotState.LOOTING, detection_confidence=0.8)

    assert machine.update(observation(loot=(detection("loot", 0.8),))) is BotState.LOOTING


def test_loot_below_configured_confidence_is_treated_as_cleared() -> None:
    machine = BotStateMachine(BotState.LOOTING, detection_confidence=0.91)

    assert (
        machine.update(observation(loot=(detection("loot", 0.9),)))
        is BotState.EXPLORING
    )


def test_no_progress_enters_recovering() -> None:
    machine = BotStateMachine(BotState.EXPLORING)

    assert machine.update(observation(movement_progress=0.0)) is BotState.RECOVERING


def test_recovery_returns_to_exploration_after_progress() -> None:
    machine = BotStateMachine(BotState.RECOVERING)

    assert machine.update(observation(movement_progress=0.1)) is BotState.EXPLORING


def test_dead_enters_restarting_when_restart_is_visible() -> None:
    machine = BotStateMachine(BotState.DEAD)

    assert machine.update(observation(restart_visible=True)) is BotState.RESTARTING


def test_restarting_enters_calibrating_after_restart() -> None:
    machine = BotStateMachine(BotState.RESTARTING)

    assert machine.update(observation()) is BotState.CALIBRATING


def test_update_returns_and_stores_the_same_transition() -> None:
    machine = BotStateMachine(BotState.EXPLORING)

    result = machine.update(observation(enemies=(detection("enemy"),)))

    assert result is BotState.COMBAT
    assert machine.state is result
