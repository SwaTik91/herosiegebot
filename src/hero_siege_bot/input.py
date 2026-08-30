from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, Protocol, cast

from hero_siege_bot.domain import Action, Point, Rect

_LOGGER = logging.getLogger(__name__)


class InputBackend(Protocol):
    def key_down(self, key: str) -> None: ...

    def key_up(self, key: str) -> None: ...

    def mouse_move(self, target: Point) -> None: ...

    def mouse_down(self, button: str) -> None: ...

    def mouse_up(self, button: str) -> None: ...


class EmergencyHotkey(Protocol):
    def register(self, callback: Callable[[], None]) -> None: ...

    def unregister(self) -> None: ...


class DryRunInputBackend:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def key_down(self, key: str) -> None:
        self.events.append(("key_down", key))

    def key_up(self, key: str) -> None:
        self.events.append(("key_up", key))

    def mouse_move(self, target: Point) -> None:
        self.events.append(("mouse_move", target))

    def mouse_down(self, button: str) -> None:
        self.events.append(("mouse_down", button))

    def mouse_up(self, button: str) -> None:
        self.events.append(("mouse_up", button))


class SafeInput:
    def __init__(
        self,
        backend: InputBackend,
        *,
        max_key_hold_s: float,
        max_mouse_hold_s: float,
        sleep: Callable[[float], None] = time.sleep,
        hotkey: EmergencyHotkey | None = None,
    ) -> None:
        if max_key_hold_s <= 0.0 or max_mouse_hold_s <= 0.0:
            raise ValueError("maximum hold durations must be positive")
        self._backend = backend
        self._max_key_hold_s = max_key_hold_s
        self._max_mouse_hold_s = max_mouse_hold_s
        self._sleep = sleep
        self._pressed_keys: set[str] = set()
        self._pressed_buttons: set[str] = set()
        self._stopped = False
        self._closed = False
        self._lock = threading.RLock()

        self._hotkey = hotkey
        if hotkey is not None:
            hotkey.register(self.emergency_stop)

    def execute(self, actions: Sequence[Action]) -> None:
        try:
            for action in actions:
                with self._lock:
                    if self._stopped:
                        raise RuntimeError("emergency stop is active")
                self._execute_one(action)
        except BaseException:
            self.release_all()
            raise

    def release_all(self) -> None:
        first_error: Exception | None = None
        with self._lock:
            for key in sorted(self._pressed_keys):
                try:
                    self._backend.key_up(key)
                except Exception as error:  # noqa: BLE001 - continue releasing all input
                    if first_error is None:
                        first_error = error
                else:
                    self._pressed_keys.discard(key)
            for button in sorted(self._pressed_buttons):
                try:
                    self._backend.mouse_up(button)
                except Exception as error:  # noqa: BLE001 - continue releasing all input
                    if first_error is None:
                        first_error = error
                else:
                    self._pressed_buttons.discard(button)
        if first_error is not None:
            raise first_error

    def emergency_stop(self) -> None:
        with self._lock:
            self._stopped = True
        self.release_all()

    def reset(self) -> None:
        with self._lock:
            self._stopped = False

    def update_geometry(self, client_rect: Rect) -> bool:
        update = getattr(self._backend, "update_geometry", None)
        if not callable(update):
            return False
        return bool(update(client_rect))

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        first_error: Exception | None = None
        try:
            self.release_all()
        except Exception as error:  # noqa: BLE001 - unregister must still run
            first_error = error
        if self._hotkey is not None:
            try:
                self._hotkey.unregister()
            except Exception as error:  # noqa: BLE001 - preserve release failure
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def _execute_one(self, action: Action) -> None:
        if action.kind == "release_all":
            self.release_all()
        elif action.kind == "key_down":
            self._press_key(self._required_key(action))
        elif action.kind == "key_up":
            self._release_key(self._required_key(action))
        elif action.kind == "key_hold":
            self._hold_key(
                self._required_key(action),
                min(max(0.0, action.duration_s), self._max_key_hold_s),
            )
        elif action.kind == "mouse_move":
            if action.target is None:
                raise ValueError("mouse_move action requires a target")
            with self._lock:
                self._ensure_active()
                self._backend.mouse_move(action.target)
        elif action.kind == "mouse_down":
            self._press_button(self._required_key(action))
        elif action.kind == "mouse_up":
            self._release_button(self._required_key(action))
        elif action.kind == "mouse_hold":
            self._hold_button(
                self._required_key(action),
                min(max(0.0, action.duration_s), self._max_mouse_hold_s),
            )
        else:
            raise ValueError(f"unsupported action kind: {action.kind}")

    @staticmethod
    def _required_key(action: Action) -> str:
        if action.key is None:
            raise ValueError(f"{action.kind} action requires a key or button")
        return action.key

    def _press_key(self, key: str) -> None:
        with self._lock:
            self._ensure_active()
            if key in self._pressed_keys:
                return
            self._backend.key_down(key)
            self._pressed_keys.add(key)

    def _release_key(self, key: str) -> None:
        with self._lock:
            if key not in self._pressed_keys:
                return
            self._backend.key_up(key)
            self._pressed_keys.discard(key)

    def _hold_key(self, key: str, duration_s: float) -> None:
        self._press_key(key)
        try:
            self._sleep(duration_s)
        finally:
            self._release_key(key)

    def _press_button(self, button: str) -> None:
        with self._lock:
            self._ensure_active()
            if button in self._pressed_buttons:
                return
            self._backend.mouse_down(button)
            self._pressed_buttons.add(button)

    def _release_button(self, button: str) -> None:
        with self._lock:
            if button not in self._pressed_buttons:
                return
            self._backend.mouse_up(button)
            self._pressed_buttons.discard(button)

    def _hold_button(self, button: str, duration_s: float) -> None:
        self._press_button(button)
        try:
            self._sleep(duration_s)
        finally:
            self._release_button(button)

    def _ensure_active(self) -> None:
        if self._stopped:
            raise RuntimeError("emergency stop is active")


class WindowsEmergencyHotkey:
    _HOTKEY_ID = 0x4853
    _WM_HOTKEY = 0x0312
    _PM_NOREMOVE = 0x0000
    _PM_REMOVE = 0x0001
    _POLL_INTERVAL_S = 0.01
    _MOD_CONTROL = 0x0002
    _MOD_SHIFT = 0x0004
    _MOD_NOREPEAT = 0x4000
    _VK_F10 = 0x79

    def __init__(self, *, registration_timeout_s: float = 2.0) -> None:
        if registration_timeout_s <= 0.0:
            raise ValueError("registration_timeout_s must be positive")
        self._registration_timeout_s = registration_timeout_s
        self._ready = threading.Event()
        self._shutdown = threading.Event()
        self._registered = False
        self._registration_error_code: int | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def register(self, callback: Callable[[], None]) -> None:
        if sys.platform != "win32":
            raise OSError("Windows hotkeys are only available on Windows")
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("Ctrl+Shift+F10 hotkey is already registered")
            self._ready.clear()
            self._shutdown.clear()
            self._registered = False
            self._registration_error_code = None
            thread = threading.Thread(
                target=self._message_loop,
                args=(callback,),
                name="hero-siege-emergency-hotkey",
                daemon=True,
            )
            self._thread = thread
            thread.start()
        if not self._ready.wait(self._registration_timeout_s):
            self.unregister()
            raise TimeoutError("timed out waiting for Ctrl+Shift+F10 RegisterHotKey")
        if not self._registered:
            thread.join(self._registration_timeout_s)
            if thread.is_alive():
                raise RuntimeError(
                    "Ctrl+Shift+F10 hotkey thread did not shut down"
                )
            with self._lock:
                self._thread = None
            raise RuntimeError(
                "RegisterHotKey failed for mandatory Ctrl+Shift+F10 emergency stop "
                f"(Win32 error {self._registration_error_code})"
            )

    def unregister(self) -> None:
        with self._lock:
            thread = self._thread
            if thread is None:
                return
            self._shutdown.set()
        thread.join(self._registration_timeout_s)
        if thread.is_alive():
            raise RuntimeError("Ctrl+Shift+F10 hotkey thread did not shut down")
        with self._lock:
            self._thread = None
            self._registered = False

    def _report_registration(
        self, registered: bool, error_code: int | None = None
    ) -> None:
        self._registered = registered
        self._registration_error_code = error_code
        self._ready.set()

    def _message_loop(self, callback: Callable[[], None]) -> None:
        import ctypes
        from ctypes import wintypes

        user32 = cast(Any, ctypes).windll.user32
        kernel32 = cast(Any, ctypes).windll.kernel32
        message = wintypes.MSG()
        user32.PeekMessageW(
            ctypes.byref(message), None, 0, 0, self._PM_NOREMOVE
        )
        if self._shutdown.is_set():
            return
        registered = bool(
            user32.RegisterHotKey(
                None,
                self._HOTKEY_ID,
                self._MOD_CONTROL | self._MOD_SHIFT | self._MOD_NOREPEAT,
                self._VK_F10,
            )
        )
        error_code = None if registered else int(kernel32.GetLastError())
        self._report_registration(registered, error_code)
        if not registered:
            return
        try:
            while not self._shutdown.is_set():
                while (
                    not self._shutdown.is_set()
                    and user32.PeekMessageW(
                        ctypes.byref(message),
                        None,
                        self._WM_HOTKEY,
                        self._WM_HOTKEY,
                        self._PM_REMOVE,
                    )
                ):
                    try:
                        callback()
                    except Exception:
                        _LOGGER.exception(
                            "Ctrl+Shift+F10 emergency-stop callback failed"
                        )
                self._shutdown.wait(self._POLL_INTERVAL_S)
        finally:
            user32.UnregisterHotKey(None, self._HOTKEY_ID)


class SendInputBackend:
    _SCAN_CODES: ClassVar[Mapping[str, int]] = {
        "1": 0x02,
        "2": 0x03,
        "Q": 0x10,
        "W": 0x11,
        "E": 0x12,
        "A": 0x1E,
        "S": 0x1F,
        "D": 0x20,
    }
    _MOUSE_DOWN_FLAGS: ClassVar[Mapping[str, int]] = {
        "left": 0x0002,
        "right": 0x0008,
        "middle": 0x0020,
    }
    _MOUSE_UP_FLAGS: ClassVar[Mapping[str, int]] = {
        "left": 0x0004,
        "right": 0x0010,
        "middle": 0x0040,
    }

    def __init__(self, client_rect: Rect) -> None:
        if sys.platform != "win32":
            raise OSError("SendInput is only available on Windows")
        if client_rect.width <= 0 or client_rect.height <= 0:
            raise ValueError("client rectangle must have positive dimensions")
        self._client_rect: Rect = client_rect

    def update_geometry(self, client_rect: Rect) -> bool:
        if client_rect.width <= 0 or client_rect.height <= 0:
            raise ValueError("client rectangle must have positive dimensions")
        changed = client_rect != self._client_rect
        self._client_rect = client_rect
        return changed

    def key_down(self, key: str) -> None:
        self._send_keyboard(key, key_up=False)

    def key_up(self, key: str) -> None:
        self._send_keyboard(key, key_up=True)

    def mouse_move(self, target: Point) -> None:
        x, y = self._client_coordinates(target)
        user32 = self._user32()
        virtual_x = user32.GetSystemMetrics(76)
        virtual_y = user32.GetSystemMetrics(77)
        virtual_width = user32.GetSystemMetrics(78)
        virtual_height = user32.GetSystemMetrics(79)
        absolute_x = round((x - virtual_x) * 65535 / max(1, virtual_width - 1))
        absolute_y = round((y - virtual_y) * 65535 / max(1, virtual_height - 1))
        self._send_mouse(
            0x0001 | 0x8000 | 0x4000,
            min(65535, max(0, absolute_x)),
            min(65535, max(0, absolute_y)),
        )

    def mouse_down(self, button: str) -> None:
        self._send_mouse(self._mouse_flag(button, self._MOUSE_DOWN_FLAGS))

    def mouse_up(self, button: str) -> None:
        self._send_mouse(self._mouse_flag(button, self._MOUSE_UP_FLAGS))

    def _client_coordinates(self, target: Point) -> tuple[int, int]:
        normalized_x = min(1.0, max(0.0, target.x))
        normalized_y = min(1.0, max(0.0, target.y))
        return (
            self._client_rect.x
            + round(normalized_x * (self._client_rect.width - 1)),
            self._client_rect.y
            + round(normalized_y * (self._client_rect.height - 1)),
        )

    @staticmethod
    def _user32() -> Any:
        import ctypes

        return cast(Any, ctypes).windll.user32

    @classmethod
    def _mouse_flag(cls, button: str, flags: Mapping[str, int]) -> int:
        try:
            return flags[button]
        except KeyError as error:
            raise ValueError(f"unsupported mouse button: {button}") from error

    def _send_keyboard(self, key: str, *, key_up: bool) -> None:
        try:
            scan_code = self._SCAN_CODES[key.upper()]
        except KeyError as error:
            raise ValueError(f"unsupported key: {key}") from error
        self._send_input(
            input_type=1,
            keyboard=(scan_code, 0x0008 | (0x0002 if key_up else 0)),
        )

    def _send_mouse(self, flags: int, x: int = 0, y: int = 0) -> None:
        self._send_input(input_type=0, mouse=(x, y, flags))

    @staticmethod
    def _send_input(
        *,
        input_type: int,
        keyboard: tuple[int, int] | None = None,
        mouse: tuple[int, int, int] | None = None,
    ) -> None:
        import ctypes
        from ctypes import wintypes

        class MouseInput(ctypes.Structure):
            _fields_ = [
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_size_t),
            ]

        class KeyboardInput(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_size_t),
            ]

        class InputValue(ctypes.Union):
            _fields_ = [("mi", MouseInput), ("ki", KeyboardInput)]

        class Input(ctypes.Structure):
            _anonymous_ = ("value",)
            _fields_ = [("type", wintypes.DWORD), ("value", InputValue)]

        if keyboard is not None:
            scan_code, flags = keyboard
            value = InputValue(
                ki=KeyboardInput(0, scan_code, flags, 0, 0),
            )
        elif mouse is not None:
            x, y, flags = mouse
            value = InputValue(
                mi=MouseInput(x, y, 0, flags, 0, 0),
            )
        else:
            raise ValueError("SendInput event data is required")

        event = Input(type=input_type, value=value)
        sent = cast(Any, ctypes).windll.user32.SendInput(
            1,
            ctypes.byref(event),
            ctypes.sizeof(Input),
        )
        if sent != 1:
            raise cast(Any, ctypes).WinError()
