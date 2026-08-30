import importlib
import sys

import pytest

from hero_siege_bot.domain import Action, Point, Rect
from hero_siege_bot.input import DryRunInputBackend, SafeInput, SendInputBackend


def test_bounded_key_hold_always_emits_down_then_up() -> None:
    backend = DryRunInputBackend()
    sleeps: list[float] = []
    safe_input = SafeInput(
        backend,
        max_key_hold_s=0.2,
        max_mouse_hold_s=0.3,
        sleep=sleeps.append,
    )

    safe_input.execute((Action(kind="key_hold", key="W", duration_s=1.0),))

    assert backend.events == [("key_down", "W"), ("key_up", "W")]
    assert sleeps == [0.2]


def test_mouse_hold_is_clamped_and_released_when_sleep_fails() -> None:
    backend = DryRunInputBackend()

    def fail_sleep(duration: float) -> None:
        assert duration == 0.25
        raise RuntimeError("clock failed")

    safe_input = SafeInput(
        backend,
        max_key_hold_s=0.2,
        max_mouse_hold_s=0.25,
        sleep=fail_sleep,
    )

    with pytest.raises(RuntimeError, match="clock failed"):
        safe_input.execute(
            (Action(kind="mouse_hold", key="left", duration_s=0.8),)
        )

    assert backend.events == [
        ("mouse_down", "left"),
        ("mouse_up", "left"),
    ]


def test_release_all_releases_every_tracked_key_and_button() -> None:
    backend = DryRunInputBackend()
    safe_input = SafeInput(backend, max_key_hold_s=0.2, max_mouse_hold_s=0.3)
    safe_input.execute(
        (
            Action(kind="key_down", key="W"),
            Action(kind="key_down", key="Q"),
            Action(kind="mouse_down", key="left"),
        )
    )

    safe_input.release_all()

    assert backend.events[-3:] == [
        ("key_up", "Q"),
        ("key_up", "W"),
        ("mouse_up", "left"),
    ]


def test_release_all_action_releases_tracked_input_in_sequence() -> None:
    backend = DryRunInputBackend()
    safe_input = SafeInput(backend, max_key_hold_s=0.2, max_mouse_hold_s=0.3)

    safe_input.execute(
        (
            Action(kind="key_down", key="W"),
            Action(kind="release_all"),
            Action(kind="key_hold", key="Q", duration_s=0.0),
        )
    )

    assert backend.events == [
        ("key_down", "W"),
        ("key_up", "W"),
        ("key_down", "Q"),
        ("key_up", "Q"),
    ]


def test_failed_releases_remain_tracked_for_retry_while_all_are_attempted() -> None:
    class FaultInjectingBackend(DryRunInputBackend):
        def __init__(self) -> None:
            super().__init__()
            self.failed_key = False
            self.failed_button = False

        def key_up(self, key: str) -> None:
            self.events.append(("key_up", key))
            if not self.failed_key:
                self.failed_key = True
                raise RuntimeError("key release failed")

        def mouse_up(self, button: str) -> None:
            self.events.append(("mouse_up", button))
            if not self.failed_button:
                self.failed_button = True
                raise RuntimeError("mouse release failed")

    backend = FaultInjectingBackend()
    safe_input = SafeInput(backend, max_key_hold_s=0.2, max_mouse_hold_s=0.3)
    safe_input.execute(
        (
            Action(kind="key_down", key="Q"),
            Action(kind="mouse_down", key="left"),
        )
    )

    with pytest.raises(RuntimeError, match="key release failed"):
        safe_input.release_all()

    assert backend.events[-2:] == [
        ("key_up", "Q"),
        ("mouse_up", "left"),
    ]

    safe_input.release_all()

    assert backend.events.count(("key_up", "Q")) == 2
    assert backend.events.count(("mouse_up", "left")) == 2


def test_emergency_stop_releases_input_and_rejects_actions_until_reset() -> None:
    backend = DryRunInputBackend()
    safe_input = SafeInput(backend, max_key_hold_s=0.2, max_mouse_hold_s=0.3)
    safe_input.execute((Action(kind="key_down", key="W"),))

    safe_input.emergency_stop()

    assert backend.events[-1] == ("key_up", "W")
    with pytest.raises(RuntimeError, match="emergency stop"):
        safe_input.execute((Action(kind="key_hold", key="Q", duration_s=0.1),))

    safe_input.reset()
    safe_input.execute((Action(kind="key_hold", key="Q", duration_s=0.0),))
    assert backend.events[-2:] == [("key_down", "Q"), ("key_up", "Q")]


def test_mouse_move_preserves_normalized_target_for_backend() -> None:
    backend = DryRunInputBackend()
    safe_input = SafeInput(backend, max_key_hold_s=0.2, max_mouse_hold_s=0.3)
    target = Point(0.25, 0.75)

    safe_input.execute((Action(kind="mouse_move", target=target),))

    assert backend.events == [("mouse_move", target)]


def test_execute_releases_tracked_input_after_action_failure() -> None:
    backend = DryRunInputBackend()
    safe_input = SafeInput(backend, max_key_hold_s=0.2, max_mouse_hold_s=0.3)

    with pytest.raises(ValueError, match="unsupported action"):
        safe_input.execute(
            (
                Action(kind="key_down", key="W"),
                Action(kind="invalid"),
            )
        )

    assert backend.events == [("key_down", "W"), ("key_up", "W")]


def test_send_input_maps_normalized_target_into_virtual_desktop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class User32:
        def GetSystemMetrics(self, metric: int) -> int:
            return {76: -1920, 77: 0, 78: 3840, 79: 1080}[metric]

    monkeypatch.setattr(sys, "platform", "win32")
    backend = SendInputBackend(Rect(100, 200, 401, 201))
    sent: list[tuple[int, int, int]] = []
    monkeypatch.setattr(backend, "_user32", lambda: User32())
    monkeypatch.setattr(
        backend,
        "_send_mouse",
        lambda flags, x=0, y=0: sent.append((flags, x, y)),
    )

    backend.mouse_move(Point(0.5, 0.5))

    expected_x = round((300 + 1920) * 65535 / 3839)
    expected_y = round(300 * 65535 / 1079)
    assert sent == [(0x0001 | 0x8000 | 0x4000, expected_x, expected_y)]


def test_input_module_is_import_safe_without_windows_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    sys.modules.pop("hero_siege_bot.input", None)

    imported = importlib.import_module("hero_siege_bot.input")

    assert imported.SendInputBackend is not None
