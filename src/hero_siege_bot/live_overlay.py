from __future__ import annotations

import sys
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from hero_siege_bot.diagnostics import CHROMA_KEY_BGR
from hero_siege_bot.domain import Rect


class LiveOverlay(Protocol):
    def show(self, image: NDArray[np.uint8], client_rect: Rect) -> None: ...

    def close(self) -> None: ...


class NullLiveOverlay:
    def show(self, image: NDArray[np.uint8], client_rect: Rect) -> None:
        del image, client_rect

    def close(self) -> None:
        return None


class Win32LiveOverlay:
    """Click-through chroma-key window pinned to the game client area."""

    def __init__(self) -> None:
        self._hwnd: int | None = None
        self._class_atom: int | None = None

    def show(self, image: NDArray[np.uint8], client_rect: Rect) -> None:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            return
        height, width = image.shape[:2]
        if width <= 0 or height <= 0 or client_rect.width <= 0 or client_rect.height <= 0:
            return
        self._ensure_window(client_rect)
        hwnd = self._hwnd
        if hwnd is None:
            return
        scaled = cv2.resize(
            image,
            (client_rect.width, client_rect.height),
            interpolation=cv2.INTER_NEAREST,
        )
        self._blit(hwnd, scaled, client_rect)

    def close(self) -> None:
        hwnd = self._hwnd
        self._hwnd = None
        if hwnd is None:
            return
        import win32gui  # type: ignore[import-not-found, import-untyped]

        if win32gui.IsWindow(hwnd):
            win32gui.DestroyWindow(hwnd)

    def _ensure_window(self, client_rect: Rect) -> None:
        import win32con  # type: ignore[import-not-found, import-untyped]
        import win32gui  # type: ignore[import-not-found, import-untyped]

        if self._hwnd is not None and win32gui.IsWindow(self._hwnd):
            win32gui.SetWindowPos(
                self._hwnd,
                win32con.HWND_TOPMOST,
                client_rect.x,
                client_rect.y,
                client_rect.width,
                client_rect.height,
                win32con.SWP_NOACTIVATE | win32con.SWP_SHOWWINDOW,
            )
            return

        if self._class_atom is None:
            window_class = win32gui.WNDCLASS()
            window_class.lpfnWndProc = win32gui.DefWindowProc
            window_class.lpszClassName = "HeroSiegeBotLiveOverlay"
            window_class.hInstance = win32gui.GetModuleHandle(None)
            self._class_atom = win32gui.RegisterClass(window_class)

        ex_style = (
            win32con.WS_EX_LAYERED
            | win32con.WS_EX_TRANSPARENT
            | win32con.WS_EX_TOPMOST
            | win32con.WS_EX_NOACTIVATE
            | win32con.WS_EX_TOOLWINDOW
        )
        self._hwnd = win32gui.CreateWindowEx(
            ex_style,
            self._class_atom,
            "Hero Siege Bot Overlay",
            win32con.WS_POPUP,
            client_rect.x,
            client_rect.y,
            client_rect.width,
            client_rect.height,
            0,
            0,
            win32gui.GetModuleHandle(None),
            None,
        )
        key = (
            CHROMA_KEY_BGR[0]
            | (CHROMA_KEY_BGR[1] << 8)
            | (CHROMA_KEY_BGR[2] << 16)
        )
        win32gui.SetLayeredWindowAttributes(
            self._hwnd, key, 0, win32con.LWA_COLORKEY
        )
        win32gui.ShowWindow(self._hwnd, win32con.SW_SHOWNOACTIVATE)

    def _blit(self, hwnd: int, image: NDArray[np.uint8], client_rect: Rect) -> None:
        import win32con  # type: ignore[import-not-found, import-untyped]
        import win32gui  # type: ignore[import-not-found, import-untyped]
        import win32ui  # type: ignore[import-not-found, import-untyped]

        height, width = image.shape[:2]
        # Windows DIBs are BGR bottom-up.
        pixels = np.ascontiguousarray(np.flipud(image))
        hdc = win32gui.GetDC(hwnd)
        dst = win32ui.CreateDCFromHandle(hdc)
        src = dst.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(dst, width, height)
        src.SelectObject(bitmap)
        bitmap.SetBitmapBits(pixels.tobytes())
        dst.BitBlt((0, 0), (width, height), src, (0, 0), win32con.SRCCOPY)
        src.DeleteDC()
        win32gui.ReleaseDC(hwnd, hdc)
        win32gui.DeleteObject(bitmap.GetHandle())
        del client_rect


def create_live_overlay(*, enabled: bool) -> LiveOverlay:
    if not enabled:
        return NullLiveOverlay()
    if sys.platform == "win32":
        return Win32LiveOverlay()
    return NullLiveOverlay()
