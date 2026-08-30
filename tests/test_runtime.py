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


def frame(*, focused: bool = True, timestamp: float = 1.0) -> CapturedFrame:
    return CapturedFrame(
        image=np.zeros((20, 20, 3), dtype=np.uint8),
        client_rect=Rect(0, 0, 20, 20),
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

    def choose_target(self, map_masks: MapMasks, player: Point) -> Point | None:
        del map_masks, player
        return self.target

    def movement_action(self, player: Point, target: Point) -> tuple[Action, ...]:
        del player, target
        return (Action("key_hold", key="D", duration_s=0.1),)

    def record_progress(self, player: Point, map_masks: MapMasks) -> bool:
        del player, map_masks
        self.progress_calls += 1
        return False

    def blacklist_current_target(self) -> None:
        self.blacklisted = True


class ControllerFake:
    def __init__(self, actions: tuple[Action, ...] = ()) -> None:
        self.result = actions
        self.calls = 0

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

    def execute(self, actions: Sequence[Action]) -> None:
        self.executed.append(tuple(actions))
        for action in actions:
            if action.kind == "release_all":
                self.release_all()

    def release_all(self) -> None:
        self.release_all_calls += 1


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
    assert isinstance(perception, PerceptionFake)
    perception.observations = [observation(movement_progress=0.0)] * 3

    states = [runtime.step() for _ in range(3)]

    assert states == [BotState.EXPLORING, BotState.EXPLORING, BotState.RECOVERING]
    explorer = runtime_parts["explorer"]
    assert isinstance(explorer, ExplorerFake)
    assert explorer.progress_calls == 0
    assert not explorer.blacklisted


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


def test_loot_timeout_emits_no_input_actions(runtime_parts: dict[str, object]) -> None:
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
    assert runtime.step() is BotState.LOOTING
    input_spy = runtime_parts["input_controller"]
    assert isinstance(input_spy, InputSpy)
    assert input_spy.executed[-1] == ()


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
        observation(restart_visible=True),
    ]

    assert runtime.step() is BotState.DEAD
    assert runtime.step() is BotState.RESTARTING
    assert input_spy.executed[-1] == (
        Action("mouse_move", target=Point(0.5, 0.5)),
        Action("mouse_hold", key="left", duration_s=0.05),
    )
    assert runtime.step() is BotState.EXPLORING
    assert calibrator.calls == 2


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
    assert input_spy.release_all_calls == 1


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
