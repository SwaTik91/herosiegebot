from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hero_siege_bot.domain import Rect


@dataclass(frozen=True)
class CapturedFrame:
    image: NDArray[np.uint8]
    client_rect: Rect
    focused: bool
    timestamp: float


class WindowCapture:
    def __init__(self, title: str) -> None:
        self._title = title.casefold()
        self._hwnd: int | None = None
        self._camera: Any | None = None

    def _find_handle(self) -> int | None:
        if sys.platform != "win32":
            return None

        import win32gui  # type: ignore[import-not-found, import-untyped]

        matches: list[int] = []

        def collect(hwnd: int, _: object) -> None:
            if not win32gui.IsWindowVisible(hwnd):
                return
            if self._title in win32gui.GetWindowText(hwnd).casefold():
                matches.append(hwnd)

        win32gui.EnumWindows(collect, None)
        return matches[0] if matches else None

    def find(self) -> Rect | None:
        hwnd = self._find_handle()
        if hwnd is None:
            self._hwnd = None
            return None

        import win32gui  # type: ignore[import-not-found, import-untyped]

        client_left, client_top, client_right, client_bottom = win32gui.GetClientRect(
            hwnd
        )
        left, top = win32gui.ClientToScreen(hwnd, (client_left, client_top))
        right, bottom = win32gui.ClientToScreen(hwnd, (client_right, client_bottom))
        if right <= left or bottom <= top:
            self._hwnd = None
            return None

        self._hwnd = hwnd
        return Rect(left, top, right - left, bottom - top)

    def grab(self) -> CapturedFrame | None:
        client_rect = self.find()
        if client_rect is None or self._hwnd is None:
            return None

        import dxcam  # type: ignore[import-not-found]
        import win32gui  # type: ignore[import-not-found, import-untyped]

        if self._camera is None:
            self._camera = dxcam.create(output_color="BGR")

        image = self._camera.grab(
            region=(
                client_rect.x,
                client_rect.y,
                client_rect.x + client_rect.width,
                client_rect.y + client_rect.height,
            )
        )
        if image is None:
            return None

        return CapturedFrame(
            image=np.asarray(image, dtype=np.uint8),
            client_rect=client_rect,
            focused=win32gui.GetForegroundWindow() == self._hwnd,
            timestamp=time.time(),
        )
