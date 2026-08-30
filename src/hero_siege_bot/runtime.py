from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from hero_siege_bot.calibration import Calibration
from hero_siege_bot.capture import CapturedFrame
from hero_siege_bot.controllers import COMBAT_TARGET_MATCH_RADIUS_NORMALIZED
from hero_siege_bot.domain import (
    Action,
    BotState,
    Detection,
    MapMasks,
    Observation,
    Point,
    Rect,
)
from hero_siege_bot.state_machine import BotStateMachine

ABANDONED_TARGET_SUPPRESSION_MAX_STEPS = 8


@dataclass(frozen=True)
class _CombatSuppression:
    target: Detection
    expires_after_step: int


class Capture(Protocol):
    def grab(self) -> CapturedFrame | None: ...


class Calibrator(Protocol):
    def calibrate(self, frames: Sequence[CapturedFrame]) -> Calibration | None: ...


class PerceptionEngine(Protocol):
    def observe(
        self, frame: CapturedFrame, calibration: Calibration
    ) -> Observation: ...


class Explorer(Protocol):
    def choose_target(self, masks: MapMasks, player: Point) -> Point | None: ...

    def movement_action(
        self, player: Point, target: Point
    ) -> tuple[Action, ...]: ...

    def record_progress(self, player: Point, masks: MapMasks) -> bool: ...

    def blacklist_current_target(self) -> None: ...


class Controller(Protocol):
    def actions(
        self, observation: Observation, now: float
    ) -> tuple[Action, ...]: ...


class Recorder(Protocol):
    def record(
        self,
        observation: Observation,
        state: BotState,
        actions: Sequence[Action],
    ) -> None: ...


class InputController(Protocol):
    def execute(self, actions: Sequence[Action]) -> None: ...

    def release_all(self) -> None: ...

    def update_geometry(self, client_rect: Rect) -> bool: ...

    def close(self) -> None: ...


class BotRuntime:
    """Coordinates perception and bounded actions behind a fail-safe boundary."""

    def __init__(
        self,
        *,
        capture: Capture,
        calibrator: Calibrator,
        perception: PerceptionEngine,
        explorer: Explorer,
        combat: Controller,
        survival: Controller,
        loot: Controller,
        recorder: Recorder | None,
        input_controller: InputController,
        calibration_confidence: float,
        no_progress_sample_limit: int,
        movement_pulse_s: float,
        detection_confidence: float,
        state_machine: BotStateMachine | None = None,
        state_reporter: Callable[[BotState], None] | None = None,
        calibration_reporter: Callable[[str], None] | None = None,
    ) -> None:
        if not 0.0 <= calibration_confidence <= 1.0:
            raise ValueError("calibration_confidence must be between 0.0 and 1.0")
        if no_progress_sample_limit <= 0:
            raise ValueError("no_progress_sample_limit must be positive")
        if movement_pulse_s <= 0.0:
            raise ValueError("movement_pulse_s must be positive")
        self.capture = capture
        self.calibrator = calibrator
        self.perception = perception
        self.explorer = explorer
        self.combat = combat
        self.survival = survival
        self.loot = loot
        self.recorder = recorder
        self.input = input_controller
        self._machine = state_machine or BotStateMachine(
            calibration_confidence=calibration_confidence,
            detection_confidence=detection_confidence,
        )
        self._calibration_confidence = calibration_confidence
        self._detection_confidence = detection_confidence
        self._no_progress_sample_limit = no_progress_sample_limit
        self._movement_pulse_s = movement_pulse_s
        self._calibration: Calibration | None = None
        self._calibration_source: tuple[Rect, int, int] | None = None
        self._last_frame_geometry: tuple[Rect, int, int] | None = None
        self._calibration_frames: deque[CapturedFrame] = deque(maxlen=30)
        self._target: Point | None = None
        self._last_movement_key: str | None = None
        self._no_progress_samples = 0
        self._recovery_phase = 0
        self._decision_step = 0
        self._combat_suppression: _CombatSuppression | None = None
        self._state_reporter = state_reporter
        self._last_reported_state: BotState | None = None
        self._calibration_reporter = calibration_reporter
        self._last_reported_calibration_diagnostic: str | None = None

    def step(self) -> BotState:
        state = self._step()
        if state is BotState.CALIBRATING:
            self._report_state(state)
            self._report_calibration_diagnostic()
        else:
            self._report_calibration_diagnostic()
            self._report_state(state)
        return state

    def _report_state(self, state: BotState) -> None:
        if self._state_reporter is not None and state is not self._last_reported_state:
            self._state_reporter(state)
            self._last_reported_state = state

    def _report_calibration_diagnostic(self) -> None:
        diagnostic = getattr(self.calibrator, "last_diagnostic", None)
        if (
            self._calibration_reporter is not None
            and isinstance(diagnostic, str)
            and diagnostic != self._last_reported_calibration_diagnostic
        ):
            self._calibration_reporter(diagnostic)
            self._last_reported_calibration_diagnostic = diagnostic

    def _step(self) -> BotState:
        try:
            captured = self.capture.grab()
            if captured is None:
                self.input.release_all()
                self._machine.state = BotState.PAUSED
                return BotState.PAUSED
            geometry = self._frame_geometry(captured)
            backend_changed = self.input.update_geometry(captured.client_rect)
            geometry_changed = (
                self._last_frame_geometry is not None
                and geometry != self._last_frame_geometry
            )
            self._last_frame_geometry = geometry
            if not captured.focused:
                self.input.release_all()
                self._invalidate_calibration()
                self._machine.state = BotState.PAUSED
                return BotState.PAUSED
            if backend_changed or geometry_changed:
                self.input.release_all()
                self._invalidate_calibration()
                self._machine.state = BotState.CALIBRATING
                return BotState.CALIBRATING

            calibration = self._ensure_calibration(captured, geometry)
            if calibration is None:
                self.input.release_all()
                self._machine.state = BotState.CALIBRATING
                return BotState.CALIBRATING

            observed = self.perception.observe(captured, calibration)
            if not self._safe_observation(observed):
                self.input.release_all()
                self._machine.state = BotState.PAUSED
                if self.recorder is not None:
                    self.recorder.record(observed, BotState.PAUSED, ())
                    self._record_frame(
                        captured,
                        observed,
                        BotState.PAUSED,
                        (),
                        calibration,
                    )
                self._invalidate_calibration()
                return BotState.PAUSED

            state, actions, release = self._decide(observed)
            if release:
                self.input.release_all()
            if self.recorder is not None:
                self.recorder.record(observed, state, actions)
                self._record_frame(captured, observed, state, actions, calibration)
            self.input.execute(actions)
            if state is BotState.RESTARTING:
                self._invalidate_calibration()
            return state
        except BaseException as error:
            try:
                self.input.release_all()
            except Exception as release_error:  # noqa: BLE001 - preserve primary failure
                error.add_note(f"release_all also failed: {release_error!r}")
            raise

    def run(self, stop: threading.Event) -> None:
        try:
            while not stop.is_set():
                self.step()
        finally:
            self.input.close()

    def _ensure_calibration(
        self,
        captured: CapturedFrame,
        geometry: tuple[Rect, int, int],
    ) -> Calibration | None:
        if self._calibration is not None and self._calibration_source == geometry:
            return self._calibration
        if self._calibration is not None:
            self.input.release_all()
            self._invalidate_calibration()
        self._calibration_frames.append(captured)
        candidate = self.calibrator.calibrate(tuple(self._calibration_frames))
        if (
            candidate is not None
            and candidate.confidence >= self._calibration_confidence
        ):
            self._calibration = candidate
            self._calibration_source = geometry
            self._calibration_frames.clear()
            self._machine.state = BotState.EXPLORING
        return self._calibration

    def _safe_observation(self, observation: Observation) -> bool:
        geometry_available = (
            observation.player_map_position is not None
            and observation.map_masks is not None
        )
        return (
            observation.focused
            and observation.calibrated
            and observation.calibration_confidence >= self._calibration_confidence
            and (
                geometry_available
                or observation.dead
                or observation.restart_visible
            )
        )

    def _decide(
        self, observation: Observation
    ) -> tuple[BotState, tuple[Action, ...], bool]:
        self._decision_step += 1
        state_before = self._machine.state
        state_observation = observation
        if (
            state_before in (BotState.CALIBRATING, BotState.PAUSED)
            and self._calibration is not None
        ):
            self._machine.state = BotState.CALIBRATING
        state_observation = self._apply_combat_suppression(state_observation)
        if self._machine.state in (BotState.EXPLORING, BotState.RECOVERING):
            player = observation.player_map_position
            masks = observation.map_masks
            progressed = bool(
                player is not None
                and masks is not None
                and self.explorer.record_progress(player, masks)
            )
            state_observation = replace(
                state_observation, movement_progress=1.0 if progressed else 0.0
            )
            if self._machine.state is BotState.EXPLORING:
                if progressed:
                    self._no_progress_samples = 0
                else:
                    self._no_progress_samples += 1
                    if self._no_progress_samples < self._no_progress_sample_limit:
                        state_observation = replace(
                            state_observation, movement_progress=1.0
                        )

        state = self._machine.update(state_observation)
        if state is BotState.RECOVERING:
            return self._recover(state_observation)
        self._recovery_phase = 0

        release = state in (BotState.DEAD, BotState.CALIBRATING, BotState.PAUSED)
        actions: list[Action] = []
        if state in (BotState.EXPLORING, BotState.COMBAT, BotState.LOOTING):
            actions.extend(self.survival.actions(observation, observation.timestamp))
        if state is BotState.EXPLORING:
            actions.extend(self._exploration_actions(observation))
        elif state is BotState.COMBAT:
            actions.extend(
                self.combat.actions(state_observation, observation.timestamp)
            )
            if bool(getattr(self.combat, "abandoned", False)):
                abandoned_target = getattr(
                    self.combat, "abandoned_target", None
                )
                if not isinstance(abandoned_target, Detection):
                    abandoned_target = self._best_confident_enemy(
                        state_observation.enemies
                    )
                if abandoned_target is not None:
                    self._combat_suppression = _CombatSuppression(
                        target=abandoned_target,
                        expires_after_step=(
                            self._decision_step
                            + ABANDONED_TARGET_SUPPRESSION_MAX_STEPS
                        ),
                    )
                self._machine.state = BotState.RECOVERING
                recovered_state, recovered_actions, _ = self._recover(
                    state_observation
                )
                return recovered_state, recovered_actions, True
        elif state is BotState.LOOTING:
            actions.extend(self.loot.actions(observation, observation.timestamp))
            if bool(getattr(self.loot, "abandoned", False)):
                self._machine.state = BotState.RECOVERING
                recovered_state, recovered_actions, _ = self._recover(
                    state_observation
                )
                return recovered_state, recovered_actions, True
        elif (
            state is BotState.RESTARTING
            and observation.restart_target is not None
        ):
            actions.extend(
                (
                    Action("mouse_move", target=observation.restart_target),
                    Action("mouse_hold", key="left", duration_s=0.05),
                )
            )
        return state, tuple(actions), release

    def _apply_combat_suppression(
        self, observation: Observation
    ) -> Observation:
        suppression = self._combat_suppression
        if suppression is None:
            return observation
        if self._decision_step > suppression.expires_after_step:
            self._clear_combat_suppression()
            return observation

        confident_matches = tuple(
            detection
            for detection in observation.enemies
            if detection.confidence >= self._detection_confidence
            and self._detection_distance(detection, suppression.target)
            <= COMBAT_TARGET_MATCH_RADIUS_NORMALIZED
        )
        if not confident_matches:
            self._clear_combat_suppression()
            return observation
        return replace(
            observation,
            enemies=tuple(
                detection
                for detection in observation.enemies
                if detection not in confident_matches
            ),
        )

    def _clear_combat_suppression(self) -> None:
        self._combat_suppression = None
        reset_abandonment = getattr(self.combat, "reset_abandonment", None)
        if callable(reset_abandonment):
            reset_abandonment()

    def _best_confident_enemy(
        self, detections: tuple[Detection, ...]
    ) -> Detection | None:
        return max(
            (
                detection
                for detection in detections
                if detection.confidence >= self._detection_confidence
            ),
            key=lambda detection: detection.confidence,
            default=None,
        )

    @staticmethod
    def _detection_distance(first: Detection, second: Detection) -> float:
        return math.hypot(
            first.center.x - second.center.x,
            first.center.y - second.center.y,
        )

    def _exploration_actions(self, observation: Observation) -> tuple[Action, ...]:
        player = observation.player_map_position
        masks = observation.map_masks
        if player is None or masks is None:
            return ()
        if self._target is None:
            self._target = self.explorer.choose_target(masks, player)
        if self._target is None:
            return ()
        actions = self.explorer.movement_action(player, self._target)
        movement = next(
            (
                action.key
                for action in reversed(actions)
                if action.kind == "key_hold" and action.key in {"W", "A", "S", "D"}
            ),
            None,
        )
        if movement is not None:
            self._last_movement_key = movement
        return actions

    def _recover(
        self, observation: Observation
    ) -> tuple[BotState, tuple[Action, ...], bool]:
        if observation.movement_progress > 0.0:
            self._machine.state = BotState.EXPLORING
            self._no_progress_samples = 0
            self._recovery_phase = 0
            return (
                BotState.EXPLORING,
                self._exploration_actions(observation),
                False,
            )
        self._recovery_phase += 1
        if self._recovery_phase == 1:
            return BotState.RECOVERING, (Action("release_all"),), False
        if self._recovery_phase == 2:
            key = self._orthogonal_key(self._last_movement_key)
            return BotState.RECOVERING, self._pulse(key), False
        if self._recovery_phase == 3:
            key = self._reverse_key(self._last_movement_key)
            return BotState.RECOVERING, self._pulse(key), False

        self.explorer.blacklist_current_target()
        self._machine.state = BotState.EXPLORING
        self._target = None
        self._no_progress_samples = 0
        self._recovery_phase = 0
        return BotState.EXPLORING, (), False

    def _pulse(self, key: str | None) -> tuple[Action, ...]:
        if key is None:
            return ()
        return (Action("key_hold", key=key, duration_s=self._movement_pulse_s),)

    @staticmethod
    def _orthogonal_key(key: str | None) -> str | None:
        if key is None:
            return None
        return {"W": "D", "D": "S", "S": "A", "A": "W"}.get(key)

    @staticmethod
    def _reverse_key(key: str | None) -> str | None:
        if key is None:
            return None
        return {"W": "S", "S": "W", "A": "D", "D": "A"}.get(key)

    def _record_frame(
        self,
        captured: CapturedFrame,
        observation: Observation,
        state: BotState,
        actions: Sequence[Action],
        calibration: Calibration,
    ) -> None:
        if self.recorder is None:
            return
        record_frame = getattr(self.recorder, "record_frame", None)
        if callable(record_frame):
            record_frame(
                captured,
                observation,
                state,
                actions,
                calibration,
                target=self._target,
            )

    def _invalidate_calibration(self) -> None:
        self._calibration = None
        self._calibration_source = None
        self._calibration_frames.clear()
        self._target = None
        self._clear_combat_suppression()

    @staticmethod
    def _frame_geometry(captured: CapturedFrame) -> tuple[Rect, int, int]:
        height, width = captured.image.shape[:2]
        return captured.client_rect, width, height
