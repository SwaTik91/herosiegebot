from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType

import cv2
import numpy as np
import pytest

from hero_siege_bot import cli
from hero_siege_bot.calibration import Calibration
from hero_siege_bot.capture import CapturedFrame
from hero_siege_bot.config import CombatConfig, ExplorationConfig
from hero_siege_bot.diagnostics import DiagnosticsOverlay, JsonlRecorder
from hero_siege_bot.domain import (
    Action,
    BotState,
    Detection,
    MapMasks,
    Observation,
    Point,
    Rect,
)
from hero_siege_bot.runtime import BotRuntime


def frame(
    *,
    focused: bool = True,
    timestamp: float = 1.0,
    client_rect: Rect | None = None,
    image_size: tuple[int, int] = (20, 20),
) -> CapturedFrame:
    return CapturedFrame(
        image=np.zeros((*image_size, 3), dtype=np.uint8),
        client_rect=client_rect or Rect(0, 0, 20, 20),
        focused=focused,
        timestamp=timestamp,
    )


def masks() -> MapMasks:
    explored = np.ones((5, 5), dtype=np.bool_)
    fog = np.zeros((5, 5), dtype=np.bool_)
    fog[:, -1] = True
    return MapMasks(explored=explored, fog=fog, walkable=explored.copy())


def observation(**overrides: object) -> Observation:
    values: dict[str, object] = {
        "timestamp": 1.0,
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
        "movement_progress": 0.1,
        "map_masks": masks(),
    }
    values.update(overrides)
    return Observation(**values)  # type: ignore[arg-type]


class CaptureFake:
    def __init__(self) -> None:
        self.next_frame: CapturedFrame | None = frame()

    def grab(self) -> CapturedFrame | None:
        return self.next_frame


class CalibratorFake:
    def __init__(self) -> None:
        self.result: Calibration | None = Calibration(
            MappingProxyType({"gameplay": Rect(0, 0, 20, 20)}),
            1.0,
            0.95,
        )
        self.calls = 0

    def calibrate(self, frames: Sequence[CapturedFrame]) -> Calibration | None:
        self.calls += 1
        return self.result


class DiagnosticCalibratorFake(CalibratorFake):
    def __init__(self) -> None:
        super().__init__()
        self.last_diagnostic: str | None = None


class ThreeFrameCalibratorFake(CalibratorFake):
    def __init__(self) -> None:
        super().__init__()
        self.frame_counts: list[int] = []

    def calibrate(self, frames: Sequence[CapturedFrame]) -> Calibration | None:
        self.calls += 1
        self.frame_counts.append(len(frames))
        if len(frames) < 3:
            return None
        return self.result


class PerceptionFake:
    def __init__(self) -> None:
        self.observations: list[Observation] = [observation()]
        self.calls = 0

    def observe(self, captured: CapturedFrame, calibration: Calibration) -> Observation:
        del captured, calibration
        self.calls += 1
        if not self.observations:
            return observation()
        return self.observations.pop(0)


class ExplorerFake:
    def __init__(self) -> None:
        self.target = Point(0.8, 0.5)
        self.blacklisted = False
        self.progress_calls = 0
        self.progressed = True
        self.progress_results: list[bool] = []

    def choose_target(self, map_masks: MapMasks, player: Point) -> Point | None:
        del map_masks, player
        return self.target

    def movement_action(self, player: Point, target: Point) -> tuple[Action, ...]:
        del player, target
        return (Action("key_hold", key="D", duration_s=0.1),)

    def record_progress(self, player: Point, map_masks: MapMasks) -> bool:
        del player, map_masks
        self.progress_calls += 1
        if self.progress_results:
            return self.progress_results.pop(0)
        return self.progressed

    def blacklist_current_target(self) -> None:
        self.blacklisted = True


class ControllerFake:
    def __init__(self, actions: tuple[Action, ...] = ()) -> None:
        self.result = actions
        self.calls = 0
        self.abandoned = False

    def actions(self, observed: Observation, now: float) -> tuple[Action, ...]:
        del observed, now
        self.calls += 1
        return self.result


class RecorderSpy:
    def __init__(self) -> None:
        self.records: list[tuple[Observation, BotState, tuple[Action, ...]]] = []

    def record(
        self, observed: Observation, state: BotState, actions: Sequence[Action]
    ) -> None:
        self.records.append((observed, state, tuple(actions)))


class InputSpy:
    def __init__(self) -> None:
        self.executed: list[tuple[Action, ...]] = []
        self.release_all_calls = 0
        self.geometries: list[Rect] = []
        self.closed = False

    def execute(self, actions: Sequence[Action]) -> None:
        self.executed.append(tuple(actions))
        for action in actions:
            if action.kind == "release_all":
                self.release_all()

    def release_all(self) -> None:
        self.release_all_calls += 1

    def update_geometry(self, client_rect: Rect) -> bool:
        changed = bool(self.geometries and self.geometries[-1] != client_rect)
        self.geometries.append(client_rect)
        return changed

    def close(self) -> None:
        self.closed = True
        self.release_all()


@pytest.fixture
def runtime_parts() -> dict[str, object]:
    return {
        "capture": CaptureFake(),
        "calibrator": CalibratorFake(),
        "perception": PerceptionFake(),
        "explorer": ExplorerFake(),
        "combat": ControllerFake((Action("key_hold", key="Q", duration_s=0.05),)),
        "survival": ControllerFake(),
        "loot": ControllerFake((Action("mouse_hold", key="left", duration_s=0.05),)),
        "recorder": RecorderSpy(),
        "input_controller": InputSpy(),
        "calibration_confidence": 0.9,
        "no_progress_sample_limit": 3,
        "movement_pulse_s": 0.1,
        "detection_confidence": 0.7,
    }


@pytest.fixture
def runtime(runtime_parts: dict[str, object]) -> BotRuntime:
    return BotRuntime(**runtime_parts)  # type: ignore[arg-type]


def test_focus_loss_releases_input_and_pauses(
    runtime: BotRuntime, runtime_parts: dict[str, object]
) -> None:
    capture = runtime_parts["capture"]
    assert isinstance(capture, CaptureFake)
    capture.next_frame = frame(focused=False)

    assert runtime.step() is BotState.PAUSED
    input_spy = runtime_parts["input_controller"]
    assert isinstance(input_spy, InputSpy)
    assert input_spy.release_all_calls == 1


def test_returning_focus_requires_fresh_calibration(
    runtime: BotRuntime, runtime_parts: dict[str, object]
) -> None:
    capture = runtime_parts["capture"]
    calibrator = runtime_parts["calibrator"]
    assert isinstance(capture, CaptureFake)
    assert isinstance(calibrator, CalibratorFake)
    assert runtime.step() is BotState.EXPLORING
    assert calibrator.calls == 1

    capture.next_frame = frame(focused=False)
    assert runtime.step() is BotState.PAUSED

    capture.next_frame = frame(focused=True)
    assert runtime.step() is BotState.EXPLORING
    assert calibrator.calls == 2


def test_capture_outage_invalidates_cached_calibration(
    runtime: BotRuntime, runtime_parts: dict[str, object]
) -> None:
    capture = runtime_parts["capture"]
    calibrator = runtime_parts["calibrator"]
    assert isinstance(capture, CaptureFake)
    assert isinstance(calibrator, CalibratorFake)
    assert runtime.step() is BotState.EXPLORING
    assert calibrator.calls == 1

    capture.next_frame = None
    assert runtime.step() is BotState.PAUSED

    capture.next_frame = frame()
    assert runtime.step() is BotState.EXPLORING
    assert calibrator.calls == 2


def test_capture_outage_discards_partial_calibration_frames(
    runtime_parts: dict[str, object],
) -> None:
    capture = runtime_parts["capture"]
    assert isinstance(capture, CaptureFake)
    calibrator = ThreeFrameCalibratorFake()
    runtime_parts["calibrator"] = calibrator
    runtime = BotRuntime(**runtime_parts)  # type: ignore[arg-type]

    assert runtime.step() is BotState.CALIBRATING
    capture.next_frame = None
    assert runtime.step() is BotState.PAUSED
    capture.next_frame = frame(timestamp=2.0)
    assert runtime.step() is BotState.CALIBRATING
    capture.next_frame = frame(timestamp=3.0)
    assert runtime.step() is BotState.CALIBRATING
    capture.next_frame = frame(timestamp=4.0)
    assert runtime.step() is BotState.EXPLORING

    assert calibrator.frame_counts == [1, 1, 2, 3]


def test_runtime_reports_initial_state_and_changes_without_repeating(
    runtime_parts: dict[str, object],
) -> None:
    reported: list[BotState] = []
    calibrator = runtime_parts["calibrator"]
    capture = runtime_parts["capture"]
    assert isinstance(calibrator, CalibratorFake)
    assert isinstance(capture, CaptureFake)
    calibrator.result = None
    runtime_parts["state_reporter"] = reported.append
    runtime = BotRuntime(**runtime_parts)  # type: ignore[arg-type]

    assert runtime.step() is BotState.CALIBRATING
    assert runtime.step() is BotState.CALIBRATING
    calibrator.result = Calibration(
        MappingProxyType({"gameplay": Rect(0, 0, 20, 20)}),
        1.0,
        0.95,
    )
    assert runtime.step() is BotState.EXPLORING
    assert runtime.step() is BotState.EXPLORING
    capture.next_frame = frame(focused=False)
    assert runtime.step() is BotState.PAUSED

    assert reported == [
        BotState.CALIBRATING,
        BotState.EXPLORING,
        BotState.PAUSED,
    ]


def test_runtime_reports_changed_calibration_diagnostic_once(
    runtime_parts: dict[str, object],
) -> None:
    reported: list[str] = []
    calibrator = DiagnosticCalibratorFake()
    runtime_parts["calibrator"] = calibrator
    calibrator.result = None
    calibrator.last_diagnostic = "waiting for 3 stable frames (1/3)"
    runtime_parts["calibration_reporter"] = reported.append
    runtime = BotRuntime(**runtime_parts)  # type: ignore[arg-type]

    assert runtime.step() is BotState.CALIBRATING
    assert runtime.step() is BotState.CALIBRATING
    calibrator.last_diagnostic = "waiting for 3 stable frames (2/3)"
    assert runtime.step() is BotState.CALIBRATING
    assert runtime.step() is BotState.CALIBRATING

    assert reported == [
        "waiting for 3 stable frames (1/3)",
        "waiting for 3 stable frames (2/3)",
    ]


def test_calibration_reporter_accepts_calibrator_without_diagnostic(
    runtime_parts: dict[str, object],
) -> None:
    reported: list[str] = []
    runtime_parts["calibration_reporter"] = reported.append
    runtime = BotRuntime(**runtime_parts)  # type: ignore[arg-type]

    assert runtime.step() is BotState.EXPLORING
    assert reported == []


def test_capture_unavailable_reports_runtime_diagnostic_once(
    runtime_parts: dict[str, object],
) -> None:
    reported: list[str] = []
    capture = runtime_parts["capture"]
    assert isinstance(capture, CaptureFake)
    capture.next_frame = None
    runtime_parts["calibration_reporter"] = reported.append
    runtime = BotRuntime(**runtime_parts)  # type: ignore[arg-type]

    assert runtime.step() is BotState.PAUSED
    assert runtime.step() is BotState.PAUSED

    assert reported == ["capture unavailable"]


def test_focus_loss_reports_runtime_diagnostic_once(
    runtime_parts: dict[str, object],
) -> None:
    reported: list[str] = []
    capture = runtime_parts["capture"]
    assert isinstance(capture, CaptureFake)
    capture.next_frame = frame(focused=False)
    runtime_parts["calibration_reporter"] = reported.append
    runtime = BotRuntime(**runtime_parts)  # type: ignore[arg-type]

    assert runtime.step() is BotState.PAUSED
    assert runtime.step() is BotState.PAUSED

    assert reported == ["window focus lost"]


def test_focus_loss_reports_again_after_diagnosticless_recovery(
    runtime_parts: dict[str, object],
) -> None:
    reported: list[str] = []
    capture = runtime_parts["capture"]
    assert isinstance(capture, CaptureFake)
    runtime_parts["calibration_reporter"] = reported.append
    runtime = BotRuntime(**runtime_parts)  # type: ignore[arg-type]

    capture.next_frame = frame(focused=False)
    assert runtime.step() is BotState.PAUSED
    capture.next_frame = frame(focused=True)
    assert runtime.step() is BotState.EXPLORING
    capture.next_frame = frame(focused=False)
    assert runtime.step() is BotState.PAUSED

    assert reported == ["window focus lost", "window focus lost"]


def test_geometry_change_reports_recalibration_diagnostic_once(
    runtime_parts: dict[str, object],
) -> None:
    reported: list[str] = []
    capture = runtime_parts["capture"]
    assert isinstance(capture, CaptureFake)
    runtime_parts["calibration_reporter"] = reported.append
    runtime = BotRuntime(**runtime_parts)  # type: ignore[arg-type]
    assert runtime.step() is BotState.EXPLORING
    capture.next_frame = frame(
        client_rect=Rect(100, 50, 30, 20),
        image_size=(20, 30),
    )

    assert runtime.step() is BotState.CALIBRATING
    assert runtime.step() is BotState.EXPLORING

    assert reported == ["capture geometry changed; recalibrating"]


def test_reporter_failure_releases_input_and_does_not_repeat_diagnostic(
    runtime_parts: dict[str, object],
) -> None:
    emitted: list[str] = []
    calibrator = DiagnosticCalibratorFake()
    calibrator.last_diagnostic = "calibrated with proportional geometry"
    runtime_parts["calibrator"] = calibrator
    input_spy = runtime_parts["input_controller"]
    assert isinstance(input_spy, InputSpy)

    def fail_after_emit(message: str) -> None:
        emitted.append(message)
        raise RuntimeError("reporter failed")

    runtime_parts["calibration_reporter"] = fail_after_emit
    runtime = BotRuntime(**runtime_parts)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="reporter failed"):
        runtime.step()

    assert input_spy.release_all_calls == 1
    assert runtime.step() is BotState.EXPLORING
    assert emitted == ["calibrated with proportional geometry"]


def test_reporter_error_remains_primary_when_input_release_fails(
    runtime_parts: dict[str, object],
) -> None:
    class FailingReleaseInput(InputSpy):
        def release_all(self) -> None:
            super().release_all()
            raise OSError("release failed")

    calibrator = DiagnosticCalibratorFake()
    calibrator.last_diagnostic = "calibrated with proportional geometry"
    runtime_parts["calibrator"] = calibrator
    runtime_parts["input_controller"] = FailingReleaseInput()

    def fail_after_emit(message: str) -> None:
        del message
        raise RuntimeError("reporter failed")

    runtime_parts["calibration_reporter"] = fail_after_emit
    runtime = BotRuntime(**runtime_parts)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="reporter failed") as captured:
        runtime.step()

    assert captured.value.__notes__ == [
        "release_all also failed: OSError('release failed')"
    ]


def test_cli_console_status_is_concise_and_uppercase(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._print_state(BotState.CALIBRATING)

    assert capsys.readouterr().out == "CALIBRATING\n"


def test_runtime_orders_initial_state_diagnostic_and_exploring_output(
    runtime_parts: dict[str, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    calibrator = DiagnosticCalibratorFake()
    calibrator.last_diagnostic = "calibrated with proportional geometry"
    runtime_parts["calibrator"] = calibrator
    runtime_parts["state_reporter"] = cli._print_state
    runtime_parts["calibration_reporter"] = cli._print_calibration_diagnostic
    runtime = BotRuntime(**runtime_parts)  # type: ignore[arg-type]

    assert runtime.step() is BotState.EXPLORING

    assert capsys.readouterr().out == (
        "CALIBRATING\n"
        "calibration: calibrated with proportional geometry\n"
        "EXPLORING\n"
    )


def test_focus_loss_precedes_simultaneous_geometry_change(
    runtime: BotRuntime, runtime_parts: dict[str, object]
) -> None:
    capture = runtime_parts["capture"]
    assert isinstance(capture, CaptureFake)
    assert runtime.step() is BotState.EXPLORING
    capture.next_frame = frame(
        focused=False,
        client_rect=Rect(100, 50, 30, 20),
        image_size=(20, 30),
    )

    assert runtime.step() is BotState.PAUSED


def test_calibration_completes_before_any_input(
    runtime: BotRuntime, runtime_parts: dict[str, object]
) -> None:
    assert runtime.step() is BotState.EXPLORING

    calibrator = runtime_parts["calibrator"]
    input_spy = runtime_parts["input_controller"]
    assert isinstance(calibrator, CalibratorFake)
    assert isinstance(input_spy, InputSpy)
    assert calibrator.calls == 1
    assert input_spy.executed == [
        (Action("key_hold", key="D", duration_s=0.1),)
    ]


def test_geometry_change_releases_updates_input_and_requires_recalibration(
    runtime: BotRuntime, runtime_parts: dict[str, object]
) -> None:
    capture = runtime_parts["capture"]
    calibrator = runtime_parts["calibrator"]
    input_spy = runtime_parts["input_controller"]
    assert isinstance(capture, CaptureFake)
    assert isinstance(calibrator, CalibratorFake)
    assert isinstance(input_spy, InputSpy)
    assert runtime.step() is BotState.EXPLORING
    capture.next_frame = frame(client_rect=Rect(100, 50, 30, 20), image_size=(20, 30))

    assert runtime.step() is BotState.CALIBRATING
    assert input_spy.release_all_calls == 1
    assert input_spy.geometries[-1] == Rect(100, 50, 30, 20)
    assert len(input_spy.executed) == 1
    assert runtime.step() is BotState.EXPLORING
    assert calibrator.calls == 2


def test_frame_dimension_change_invalidates_cached_calibration(
    runtime: BotRuntime, runtime_parts: dict[str, object]
) -> None:
    capture = runtime_parts["capture"]
    input_spy = runtime_parts["input_controller"]
    assert isinstance(capture, CaptureFake)
    assert isinstance(input_spy, InputSpy)
    assert runtime.step() is BotState.EXPLORING
    capture.next_frame = frame(client_rect=Rect(0, 0, 20, 20), image_size=(21, 20))

    assert runtime.step() is BotState.CALIBRATING
    assert input_spy.release_all_calls == 1
    assert len(input_spy.executed) == 1


def test_low_confidence_observation_releases_input_and_pauses(
    runtime: BotRuntime, runtime_parts: dict[str, object]
) -> None:
    perception = runtime_parts["perception"]
    assert isinstance(perception, PerceptionFake)
    perception.observations = [observation(calibration_confidence=0.2)]

    assert runtime.step() is BotState.PAUSED
    input_spy = runtime_parts["input_controller"]
    assert isinstance(input_spy, InputSpy)
    assert input_spy.release_all_calls == 1
    assert input_spy.executed == []


def test_no_progress_enters_recovery_after_three_samples(
    runtime: BotRuntime, runtime_parts: dict[str, object]
) -> None:
    perception = runtime_parts["perception"]
    explorer = runtime_parts["explorer"]
    assert isinstance(perception, PerceptionFake)
    assert isinstance(explorer, ExplorerFake)
    explorer.progressed = False
    perception.observations = [observation(movement_progress=0.0)] * 3

    states = [runtime.step() for _ in range(3)]

    assert states == [BotState.EXPLORING, BotState.EXPLORING, BotState.RECOVERING]
    assert explorer.progress_calls == 3
    assert not explorer.blacklisted


def test_fog_reveal_progress_prevents_recovery_without_position_change(
    runtime: BotRuntime, runtime_parts: dict[str, object]
) -> None:
    perception = runtime_parts["perception"]
    explorer = runtime_parts["explorer"]
    assert isinstance(perception, PerceptionFake)
    assert isinstance(explorer, ExplorerFake)
    explorer.record_progress = lambda player, map_masks: True  # type: ignore[method-assign]
    perception.observations = [observation(movement_progress=0.0)] * 4

    states = [runtime.step() for _ in range(4)]

    assert states == [BotState.EXPLORING] * 4


def test_missing_exploration_geometry_releases_input_and_pauses(
    runtime: BotRuntime, runtime_parts: dict[str, object]
) -> None:
    perception = runtime_parts["perception"]
    assert isinstance(perception, PerceptionFake)
    perception.observations = [observation(map_masks=None)]

    assert runtime.step() is BotState.PAUSED
    input_spy = runtime_parts["input_controller"]
    assert isinstance(input_spy, InputSpy)
    assert input_spy.release_all_calls == 1
    assert input_spy.executed == []


def test_recovery_releases_then_pulses_orthogonal_and_reverse(
    runtime: BotRuntime, runtime_parts: dict[str, object]
) -> None:
    perception = runtime_parts["perception"]
    input_spy = runtime_parts["input_controller"]
    assert isinstance(perception, PerceptionFake)
    assert isinstance(input_spy, InputSpy)
    explorer = runtime_parts["explorer"]
    assert isinstance(explorer, ExplorerFake)
    explorer.progressed = False
    perception.observations = [observation(movement_progress=0.0)] * 6

    states = [runtime.step() for _ in range(6)]

    assert states[2:6] == [
        BotState.RECOVERING,
        BotState.RECOVERING,
        BotState.RECOVERING,
        BotState.EXPLORING,
    ]
    assert input_spy.release_all_calls == 1
    recovery_pulses = [
        actions
        for actions in input_spy.executed[2:]
        if actions
    ]
    assert recovery_pulses == [
        (Action("release_all"),),
        (Action("key_hold", key="S", duration_s=0.1),),
        (Action("key_hold", key="A", duration_s=0.1),),
    ]
    explorer = runtime_parts["explorer"]
    assert isinstance(explorer, ExplorerFake)
    assert explorer.blacklisted


def test_recovery_progress_resumes_without_blacklisting(
    runtime: BotRuntime, runtime_parts: dict[str, object]
) -> None:
    perception = runtime_parts["perception"]
    explorer = runtime_parts["explorer"]
    assert isinstance(perception, PerceptionFake)
    assert isinstance(explorer, ExplorerFake)
    explorer.progress_results = [False, False, False, False, True]
    perception.observations = [
        observation(movement_progress=0.0),
        observation(movement_progress=0.0),
        observation(movement_progress=0.0),
        observation(movement_progress=0.0),
        observation(movement_progress=0.1),
    ]

    states = [runtime.step() for _ in range(5)]

    assert states[-1] is BotState.EXPLORING
    assert not explorer.blacklisted


def test_recovery_release_is_recorded_before_normal_execution(
    runtime_parts: dict[str, object]
) -> None:
    order: list[str] = []

    class OrderedRecorder(RecorderSpy):
        def record(
            self, observed: Observation, state: BotState, actions: Sequence[Action]
        ) -> None:
            order.append(f"record:{actions[0].kind if actions else 'none'}")
            super().record(observed, state, actions)

    class OrderedInput(InputSpy):
        def execute(self, actions: Sequence[Action]) -> None:
            order.append(f"execute:{actions[0].kind if actions else 'none'}")
            super().execute(actions)

        def release_all(self) -> None:
            order.append("release")
            super().release_all()

    recorder = OrderedRecorder()
    runtime_parts["recorder"] = recorder
    runtime_parts["input_controller"] = OrderedInput()
    runtime = BotRuntime(**runtime_parts)  # type: ignore[arg-type]
    perception = runtime_parts["perception"]
    assert isinstance(perception, PerceptionFake)
    explorer = runtime_parts["explorer"]
    assert isinstance(explorer, ExplorerFake)
    explorer.progressed = False
    perception.observations = [observation(movement_progress=0.0)] * 3

    runtime.step()
    runtime.step()
    order.clear()
    runtime.step()

    assert order == ["record:release_all", "execute:release_all", "release"]
    assert recorder.records[-1][2] == (Action("release_all"),)


def test_combat_preempts_exploration_and_combines_survival(
    runtime: BotRuntime, runtime_parts: dict[str, object]
) -> None:
    perception = runtime_parts["perception"]
    survival = runtime_parts["survival"]
    input_spy = runtime_parts["input_controller"]
    assert isinstance(perception, PerceptionFake)
    assert isinstance(survival, ControllerFake)
    assert isinstance(input_spy, InputSpy)
    enemy = Detection("enemy", Point(0.4, 0.5), 0.95)
    perception.observations = [observation(enemies=(enemy,))]
    survival.result = (Action("key_hold", key="1", duration_s=0.05),)

    assert runtime.step() is BotState.COMBAT
    assert input_spy.executed == [
        (
            Action("key_hold", key="1", duration_s=0.05),
            Action("key_hold", key="Q", duration_s=0.05),
        )
    ]


def test_runtime_passes_configured_detection_confidence_to_state_machine(
    runtime_parts: dict[str, object],
) -> None:
    runtime_parts["detection_confidence"] = 0.95
    runtime = BotRuntime(**runtime_parts)  # type: ignore[arg-type]
    perception = runtime_parts["perception"]
    assert isinstance(perception, PerceptionFake)
    perception.observations = [
        observation(enemies=(Detection("enemy", Point(0.4, 0.5), 0.9),))
    ]

    assert runtime.step() is BotState.EXPLORING


def test_loot_timeout_abandons_persistent_detection(runtime_parts: dict[str, object]) -> None:
    combat_config = CombatConfig(
        detection_confidence=0.7,
        attack_hold_s=0.1,
        skill_cooldowns_s=MappingProxyType({"Q": 1.0, "E": 1.0}),
        combat_timeout_s=2.0,
        loot_timeout_s=1.0,
    )
    from hero_siege_bot.controllers import LootController

    runtime_parts["loot"] = LootController(combat_config)
    runtime = BotRuntime(**runtime_parts)  # type: ignore[arg-type]
    perception = runtime_parts["perception"]
    assert isinstance(perception, PerceptionFake)
    item = Detection("loot", Point(0.5, 0.5), 0.9)
    perception.observations = [
        observation(enemies=(Detection("enemy", Point(0.5, 0.5), 0.9),)),
        observation(loot=(item,), timestamp=2.0),
        observation(loot=(item,), timestamp=3.0),
    ]

    assert runtime.step() is BotState.COMBAT
    assert runtime.step() is BotState.LOOTING
    assert runtime.step() is not BotState.LOOTING
    input_spy = runtime_parts["input_controller"]
    assert isinstance(input_spy, InputSpy)
    assert input_spy.release_all_calls >= 1


def test_combat_timeout_abandons_persistent_detection(
    runtime_parts: dict[str, object],
) -> None:
    from hero_siege_bot.controllers import CombatController

    combat_config = CombatConfig(
        detection_confidence=0.7,
        attack_hold_s=0.1,
        skill_cooldowns_s=MappingProxyType({"Q": 1.0, "E": 1.0}),
        combat_timeout_s=1.0,
        loot_timeout_s=1.0,
    )
    runtime_parts["combat"] = CombatController(combat_config)
    runtime = BotRuntime(**runtime_parts)  # type: ignore[arg-type]
    perception = runtime_parts["perception"]
    assert isinstance(perception, PerceptionFake)
    enemy = Detection("enemy", Point(0.5, 0.5), 0.9)
    perception.observations = [
        observation(enemies=(enemy,), timestamp=1.0),
        observation(enemies=(enemy,), timestamp=2.0, movement_progress=0.0),
    ]

    assert runtime.step() is BotState.COMBAT
    assert runtime.step() is not BotState.COMBAT


def test_persistent_abandoned_enemy_does_not_reset_recovery_sequence(
    runtime_parts: dict[str, object],
) -> None:
    from hero_siege_bot.controllers import CombatController

    combat_config = CombatConfig(
        detection_confidence=0.7,
        attack_hold_s=0.1,
        skill_cooldowns_s=MappingProxyType({"Q": 1.0, "E": 1.0}),
        combat_timeout_s=1.0,
        loot_timeout_s=1.0,
    )
    runtime_parts["combat"] = CombatController(combat_config)
    runtime = BotRuntime(**runtime_parts)  # type: ignore[arg-type]
    perception = runtime_parts["perception"]
    explorer = runtime_parts["explorer"]
    input_spy = runtime_parts["input_controller"]
    assert isinstance(perception, PerceptionFake)
    assert isinstance(explorer, ExplorerFake)
    assert isinstance(input_spy, InputSpy)
    explorer.progressed = False
    enemy = Detection("enemy", Point(0.5, 0.5), 0.9)
    perception.observations = [
        observation(timestamp=0.0),
        *[
            observation(
                enemies=(enemy,),
                timestamp=float(timestamp),
                movement_progress=0.0,
            )
            for timestamp in range(1, 7)
        ],
    ]

    states = [runtime.step() for _ in range(7)]

    assert states == [
        BotState.EXPLORING,
        BotState.COMBAT,
        BotState.RECOVERING,
        BotState.RECOVERING,
        BotState.RECOVERING,
        BotState.EXPLORING,
        BotState.EXPLORING,
    ]
    post_timeout_actions = input_spy.executed[2:]
    assert post_timeout_actions.count((Action("release_all"),)) == 1
    assert (Action("key_hold", key="S", duration_s=0.1),) in post_timeout_actions
    assert (Action("key_hold", key="A", duration_s=0.1),) in post_timeout_actions


def test_combat_can_trigger_again_after_enemy_clear_resets_abandonment(
    runtime_parts: dict[str, object],
) -> None:
    from hero_siege_bot.controllers import CombatController

    combat_config = CombatConfig(
        detection_confidence=0.7,
        attack_hold_s=0.1,
        skill_cooldowns_s=MappingProxyType({"Q": 1.0, "E": 1.0}),
        combat_timeout_s=1.0,
        loot_timeout_s=1.0,
    )
    runtime_parts["combat"] = CombatController(combat_config)
    runtime = BotRuntime(**runtime_parts)  # type: ignore[arg-type]
    perception = runtime_parts["perception"]
    explorer = runtime_parts["explorer"]
    assert isinstance(perception, PerceptionFake)
    assert isinstance(explorer, ExplorerFake)
    enemy = Detection("enemy", Point(0.5, 0.5), 0.9)
    explorer.progress_results = [True, False, True]
    perception.observations = [
        observation(timestamp=0.0),
        observation(enemies=(enemy,), timestamp=1.0),
        observation(enemies=(enemy,), timestamp=2.0, movement_progress=0.0),
        observation(enemies=(), timestamp=3.0),
        observation(enemies=(enemy,), timestamp=4.0),
    ]

    states = [runtime.step() for _ in range(5)]

    assert states == [
        BotState.EXPLORING,
        BotState.COMBAT,
        BotState.RECOVERING,
        BotState.EXPLORING,
        BotState.COMBAT,
    ]


def test_different_confident_enemy_preempts_while_old_target_is_suppressed(
    runtime_parts: dict[str, object],
) -> None:
    from hero_siege_bot.controllers import CombatController

    combat_config = CombatConfig(
        detection_confidence=0.7,
        attack_hold_s=0.1,
        skill_cooldowns_s=MappingProxyType({"Q": 1.0, "E": 1.0}),
        combat_timeout_s=1.0,
        loot_timeout_s=1.0,
    )
    runtime_parts["combat"] = CombatController(combat_config)
    runtime = BotRuntime(**runtime_parts)  # type: ignore[arg-type]
    perception = runtime_parts["perception"]
    input_spy = runtime_parts["input_controller"]
    assert isinstance(perception, PerceptionFake)
    assert isinstance(input_spy, InputSpy)
    old_enemy = Detection("enemy", Point(0.25, 0.5), 0.95)
    new_enemy = Detection("enemy", Point(0.8, 0.5), 0.9)
    perception.observations = [
        observation(timestamp=0.0),
        observation(enemies=(old_enemy,), timestamp=1.0),
        observation(enemies=(old_enemy,), timestamp=2.0, movement_progress=0.0),
        observation(enemies=(old_enemy, new_enemy), timestamp=3.0),
    ]

    states = [runtime.step() for _ in range(4)]

    assert states[-1] is BotState.COMBAT
    assert Action("mouse_move", target=new_enemy.center) in input_spy.executed[-1]


def test_low_confidence_noise_does_not_keep_old_target_suppressed(
    runtime_parts: dict[str, object],
) -> None:
    from hero_siege_bot.controllers import CombatController

    combat_config = CombatConfig(
        detection_confidence=0.7,
        attack_hold_s=0.1,
        skill_cooldowns_s=MappingProxyType({"Q": 1.0, "E": 1.0}),
        combat_timeout_s=1.0,
        loot_timeout_s=1.0,
    )
    runtime_parts["combat"] = CombatController(combat_config)
    runtime = BotRuntime(**runtime_parts)  # type: ignore[arg-type]
    perception = runtime_parts["perception"]
    assert isinstance(perception, PerceptionFake)
    old_enemy = Detection("enemy", Point(0.25, 0.5), 0.95)
    noise = Detection("enemy", Point(0.25, 0.5), 0.2)
    new_enemy = Detection("enemy", Point(0.8, 0.5), 0.9)
    perception.observations = [
        observation(timestamp=0.0),
        observation(enemies=(old_enemy,), timestamp=1.0),
        observation(enemies=(old_enemy,), timestamp=2.0, movement_progress=0.0),
        observation(enemies=(noise,), timestamp=3.0),
        observation(enemies=(new_enemy,), timestamp=4.0),
    ]

    states = [runtime.step() for _ in range(5)]

    assert states[-2:] == [BotState.EXPLORING, BotState.COMBAT]


def test_abandoned_target_suppression_expires_while_detection_persists(
    runtime_parts: dict[str, object],
) -> None:
    from hero_siege_bot.controllers import CombatController
    from hero_siege_bot.runtime import ABANDONED_TARGET_SUPPRESSION_MAX_STEPS

    combat_config = CombatConfig(
        detection_confidence=0.7,
        attack_hold_s=0.1,
        skill_cooldowns_s=MappingProxyType({"Q": 1.0, "E": 1.0}),
        combat_timeout_s=1.0,
        loot_timeout_s=1.0,
    )
    runtime_parts["combat"] = CombatController(combat_config)
    runtime = BotRuntime(**runtime_parts)  # type: ignore[arg-type]
    perception = runtime_parts["perception"]
    explorer = runtime_parts["explorer"]
    assert isinstance(perception, PerceptionFake)
    assert isinstance(explorer, ExplorerFake)
    explorer.progressed = False
    enemy = Detection("enemy", Point(0.25, 0.5), 0.95)
    persistent_steps = ABANDONED_TARGET_SUPPRESSION_MAX_STEPS + 1
    perception.observations = [
        observation(timestamp=0.0),
        observation(enemies=(enemy,), timestamp=1.0),
        observation(enemies=(enemy,), timestamp=2.0, movement_progress=0.0),
        *[
            observation(
                enemies=(enemy,),
                timestamp=float(3 + offset),
                movement_progress=0.0,
            )
            for offset in range(persistent_steps)
        ],
    ]

    states = [runtime.step() for _ in perception.observations.copy()]

    assert BotState.RECOVERING in states
    assert states[-1] is BotState.COMBAT


def test_death_restart_click_invalidates_calibration(
    runtime: BotRuntime, runtime_parts: dict[str, object]
) -> None:
    perception = runtime_parts["perception"]
    input_spy = runtime_parts["input_controller"]
    calibrator = runtime_parts["calibrator"]
    assert isinstance(perception, PerceptionFake)
    assert isinstance(input_spy, InputSpy)
    assert isinstance(calibrator, CalibratorFake)
    perception.observations = [
        observation(dead=True),
        observation(restart_visible=True, restart_target=Point(0.72, 0.61)),
    ]

    assert runtime.step() is BotState.DEAD
    assert runtime.step() is BotState.RESTARTING
    assert input_spy.executed[-1] == (
        Action("mouse_move", target=Point(0.72, 0.61)),
        Action("mouse_hold", key="left", duration_s=0.05),
    )
    assert runtime.step() is BotState.EXPLORING
    assert calibrator.calls == 2


def test_restarting_without_verified_target_emits_no_click(
    runtime: BotRuntime, runtime_parts: dict[str, object]
) -> None:
    perception = runtime_parts["perception"]
    input_spy = runtime_parts["input_controller"]
    assert isinstance(perception, PerceptionFake)
    assert isinstance(input_spy, InputSpy)
    perception.observations = [
        observation(dead=True),
        observation(restart_visible=True, restart_target=None),
    ]

    assert runtime.step() is BotState.DEAD
    assert runtime.step() is BotState.DEAD
    assert input_spy.executed[-1] == ()


def test_decision_is_recorded_before_actions_are_executed(
    runtime_parts: dict[str, object]
) -> None:
    order: list[str] = []

    class OrderedRecorder(RecorderSpy):
        def record(
            self, observed: Observation, state: BotState, actions: Sequence[Action]
        ) -> None:
            order.append("record")
            super().record(observed, state, actions)

    class OrderedInput(InputSpy):
        def execute(self, actions: Sequence[Action]) -> None:
            order.append("execute")
            super().execute(actions)

    runtime_parts["recorder"] = OrderedRecorder()
    runtime_parts["input_controller"] = OrderedInput()
    runtime = BotRuntime(**runtime_parts)  # type: ignore[arg-type]

    runtime.step()

    assert order == ["record", "execute"]


def test_exception_releases_all_input_and_propagates(
    runtime: BotRuntime, runtime_parts: dict[str, object]
) -> None:
    perception = runtime_parts["perception"]
    assert isinstance(perception, PerceptionFake)

    def fail(captured: CapturedFrame, calibration: Calibration) -> Observation:
        del captured, calibration
        raise RuntimeError("vision failed")

    perception.observe = fail  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="vision failed"):
        runtime.step()

    input_spy = runtime_parts["input_controller"]
    assert isinstance(input_spy, InputSpy)
    assert input_spy.release_all_calls == 1


def test_run_releases_input_when_stopped(
    runtime: BotRuntime, runtime_parts: dict[str, object]
) -> None:
    stop = threading.Event()
    stop.set()

    runtime.run(stop)

    input_spy = runtime_parts["input_controller"]
    assert isinstance(input_spy, InputSpy)
    assert input_spy.closed


def test_jsonl_recorder_writes_serializable_event_and_evidence(tmp_path: Path) -> None:
    recorder = JsonlRecorder(tmp_path, frame_interval_s=0.5)
    observed = observation(
        enemies=(Detection("enemy", Point(0.25, 0.5), 0.9),)
    )
    actions = (Action("key_hold", key="Q", duration_s=0.05),)

    recorder.record(observed, BotState.COMBAT, actions)
    recorder.record_frame(
        frame(timestamp=1.0),
        observed,
        BotState.COMBAT,
        actions,
        Calibration(MappingProxyType({"gameplay": Rect(1, 1, 10, 10)}), 1.0, 0.95),
    )

    event = json.loads((recorder.session_dir / "events.jsonl").read_text().strip())
    assert event["state"] == "combat"
    assert event["actions"][0]["key"] == "Q"
    evidence = list((recorder.session_dir / "evidence").glob("*.png"))
    assert len(evidence) == 1
    assert cv2.imread(str(evidence[0])) is not None


def test_jsonl_records_release_all_action_evidence(tmp_path: Path) -> None:
    recorder = JsonlRecorder(tmp_path, frame_interval_s=0.5)

    recorder.record(
        observation(movement_progress=0.0),
        BotState.RECOVERING,
        (Action("release_all"),),
    )

    event = json.loads((recorder.session_dir / "events.jsonl").read_text())
    assert event["actions"] == [
        {
            "kind": "release_all",
            "key": None,
            "target": None,
            "duration_s": 0.0,
        }
    ]


def test_explicitly_disabled_overlay_saves_unmodified_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("overlay rendered")

    monkeypatch.setattr(DiagnosticsOverlay, "render", forbidden)
    recorder = JsonlRecorder(tmp_path, frame_interval_s=0.5, overlay=None)
    captured = frame(timestamp=1.0)
    recorder.record_frame(
        captured,
        observation(),
        BotState.EXPLORING,
        (),
        Calibration(MappingProxyType({"gameplay": Rect(1, 1, 10, 10)}), 1.0, 0.95),
    )

    evidence = next((recorder.session_dir / "evidence").glob("*.png"))
    saved = cv2.imread(str(evidence))
    assert saved is not None
    assert np.array_equal(saved, captured.image)


def test_overlay_draws_on_copy_without_modifying_perception_image() -> None:
    source = np.zeros((20, 20, 3), dtype=np.uint8)
    before = source.copy()
    observed = observation(
        enemies=(Detection("enemy", Point(0.25, 0.5), 0.9),),
        loot=(Detection("loot", Point(0.75, 0.5), 0.8),),
    )
    calibration = Calibration(
        MappingProxyType({"gameplay": Rect(1, 1, 18, 18)}), 1.0, 0.95
    )

    rendered = DiagnosticsOverlay().render(
        source,
        calibration,
        observed,
        BotState.COMBAT,
        (Action("key_hold", key="Q", duration_s=0.05),),
        frontier=Point(0.8, 0.5),
        target=Point(0.7, 0.5),
    )

    assert np.array_equal(source, before)
    assert not np.array_equal(rendered, source)


def exploration_config() -> ExplorationConfig:
    return ExplorationConfig(
        movement_pulse_s=0.1,
        max_movement_pulse_s=0.2,
        stuck_timeout_s=1.0,
    )


def test_cli_dry_run_never_constructs_real_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}")
    used_backend: list[object] = []

    class RuntimeFake:
        def run(self, stop: threading.Event) -> None:
            del stop

    def build(config: object, backend: object, **kwargs: object) -> RuntimeFake:
        del config, kwargs
        used_backend.append(backend)
        return RuntimeFake()

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("real input was constructed")

    monkeypatch.setattr(cli, "build_runtime", build)
    monkeypatch.setattr(cli, "SendInputBackend", forbidden)

    assert cli.main(["dry-run", "--config", str(config_path)]) == 0
    assert len(used_backend) == 1
    assert used_backend[0].__class__.__name__ == "DryRunInputBackend"


def test_cli_does_not_create_overlay_when_config_disables_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("recording:\n  overlay: false\n")
    config = cli.load_config(config_path)
    received_overlays: list[object] = []

    def forbidden_overlay() -> None:
        raise AssertionError("overlay was created")

    def recorder(
        root: Path, *, frame_interval_s: float, overlay: object
    ) -> object:
        del root, frame_interval_s
        received_overlays.append(overlay)
        return object()

    monkeypatch.setattr(cli, "DiagnosticsOverlay", forbidden_overlay)
    monkeypatch.setattr(cli, "JsonlRecorder", recorder)

    assert cli._build_recorder(config, tmp_path) is not None
    assert received_overlays == [None]


def test_cli_run_refuses_non_windows_before_building_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}")
    monkeypatch.setattr(cli.sys, "platform", "darwin")

    with pytest.raises(SystemExit, match="Windows"):
        cli.main(["run", "--config", str(config_path), "--enable-input"])


def test_cli_run_requires_explicit_input_enablement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}")
    monkeypatch.setattr(cli.sys, "platform", "win32")

    with pytest.raises(SystemExit, match="enable-input"):
        cli.main(["run", "--config", str(config_path)])


def test_runtime_construction_aborts_when_mandatory_hotkey_registration_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hero_siege_bot.input import DryRunInputBackend

    class FailingHotkey:
        def register(self, callback: object) -> None:
            del callback
            raise RuntimeError("RegisterHotKey failed")

        def unregister(self) -> None:
            raise AssertionError("unregister should not run after failed registration")

    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}")
    monkeypatch.setattr(cli, "_load_template", lambda name: np.zeros((2, 2, 3), np.uint8))
    monkeypatch.setattr(cli, "_load_calibrator", lambda config: CalibratorFake())
    monkeypatch.setattr(cli, "_build_recorder", lambda config, root: None)

    with pytest.raises(RuntimeError, match="RegisterHotKey failed"):
        cli.build_runtime(
            cli.load_config(config_path),
            DryRunInputBackend(),
            capture=CaptureFake(),
            diagnostics_root=tmp_path,
            hotkey=FailingHotkey(),  # type: ignore[arg-type]
        )
