from __future__ import annotations

import sys
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from hero_siege_bot.diagnostics import CHROMA_KEY_BGR
from hero_siege_bot.domain import Rect

OVERLAY_CLASS_NAME = "HeroSiegeBotLiveOverlay"


def is_game_focused(game_hwnd: int, foreground_hwnd: int, foreground_class: str) -> bool:
    return foreground_hwnd == game_hwnd or foreground_class == OVERLAY_CLASS_NAME


def chroma_colorref() -> int:
    return (
        CHROMA_KEY_BGR[0]
        | (CHROMA_KEY_BGR[1] << 8)
        | (CHROMA_KEY_BGR[2] << 16)
    )


def pack_dib_bgra(image: NDArray[np.uint8]) -> bytes:
    """Pack a top-down BGR frame as a bottom-up 32-bit DIB."""
    height, width = image.shape[:2]
    bgra = np.empty((height, width, 4), dtype=np.uint8)
    bgra[:, :, :3] = image
    bgra[:, :, 3] = 255
    return np.ascontiguousarray(np.flipud(bgra)).tobytes()


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
        self._wndproc: object | None = None

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
        self._pump_messages()

    def close(self) -> None:
        hwnd = self._hwnd
        self._hwnd = None
        if hwnd is None:
            return
        import win32gui  # type: ignore[import-not-found, import-untyped]

        if win32gui.IsWindow(hwnd):
            win32gui.DestroyWindow(hwnd)

    def _pump_messages(self) -> None:
        import win32gui  # type: ignore[import-not-found, import-untyped]

        win32gui.PumpWaitingMessages()

    def _wnd_proc(self, hwnd: int, msg: int, wparam: int, lparam: int) -> int:
        import win32con  # type: ignore[import-not-found, import-untyped]
        import win32gui  # type: ignore[import-not-found, import-untyped]

        if msg == win32con.WM_PAINT:
            win32gui.BeginPaint(hwnd)
            win32gui.EndPaint(hwnd)
            return 0
        if msg == win32con.WM_ERASEBKGND:
            return 1
        return int(win32gui.DefWindowProc(hwnd, msg, wparam, lparam))

    def _ensure_window(self, client_rect: Rect) -> None:
        import win32con  # type: ignore[import-not-found, import-untyped]
        import win32gui  # type: ignore[import-not-found, import-untyped]

        if self._hwnd is not None and win32gui.IsWindow(self._hwnd):
            return

        if self._class_atom is None:
            self._wndproc = self._wnd_proc
            window_class = win32gui.WNDCLASS()
            window_class.lpfnWndProc = self._wndproc
            window_class.lpszClassName = OVERLAY_CLASS_NAME
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
        win32gui.ShowWindow(self._hwnd, win32con.SW_SHOWNOACTIVATE)

    def _blit(self, hwnd: int, image: NDArray[np.uint8], client_rect: Rect) -> None:
        import ctypes
        from ctypes import wintypes

        import win32con  # type: ignore[import-not-found, import-untyped]
        import win32gui  # type: ignore[import-not-found, import-untyped]

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD),
            ]

        class BITMAPINFO(ctypes.Structure):
            _fields_ = [
                ("bmiHeader", BITMAPINFOHEADER),
                ("bmiColors", wintypes.DWORD * 3),
            ]

        class POINT(ctypes.Structure):
            _fields_ = (("x", wintypes.LONG), ("y", wintypes.LONG))

        class SIZE(ctypes.Structure):
            _fields_ = (("cx", wintypes.LONG), ("cy", wintypes.LONG))

        height, width = image.shape[:2]
        bits = pack_dib_bgra(image)
        info = BITMAPINFO()
        info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        info.bmiHeader.biWidth = width
        info.bmiHeader.biHeight = height
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        info.bmiHeader.biSizeImage = len(bits)
        gdi32 = ctypes.windll.gdi32
        user32 = ctypes.windll.user32
        gdi32.CreateCompatibleDC.restype = wintypes.HDC
        gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
        gdi32.CreateDIBSection.restype = wintypes.HBITMAP
        gdi32.CreateDIBSection.argtypes = [
            wintypes.HDC,
            ctypes.c_void_p,
            wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p),
            wintypes.HANDLE,
            wintypes.DWORD,
        ]
        gdi32.SelectObject.restype = wintypes.HGDIOBJ
        gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
        gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
        gdi32.DeleteDC.argtypes = [wintypes.HDC]
        user32.UpdateLayeredWindow.argtypes = [
            wintypes.HWND,
            wintypes.HDC,
            ctypes.POINTER(POINT),
            ctypes.POINTER(SIZE),
            wintypes.HDC,
            ctypes.POINTER(POINT),
            wintypes.COLORREF,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        user32.UpdateLayeredWindow.restype = wintypes.BOOL
        screen_hdc = win32gui.GetDC(0)
        bits_ptr = ctypes.c_void_p()
        mem_dc = None
        bitmap = None
        old = None
        try:
            mem_dc = gdi32.CreateCompatibleDC(screen_hdc)
            bitmap = gdi32.CreateDIBSection(
                screen_hdc,
                ctypes.byref(info),
                0,
                ctypes.byref(bits_ptr),
                None,
                0,
            )
            if not bitmap or not bits_ptr.value:
                raise OSError(f"CreateDIBSection failed ({ctypes.GetLastError()})")
            ctypes.memmove(bits_ptr, bits, len(bits))
            old = gdi32.SelectObject(mem_dc, bitmap)
            dest = POINT(client_rect.x, client_rect.y)
            size = SIZE(width, height)
            origin = POINT(0, 0)
            updated = user32.UpdateLayeredWindow(
                int(hwnd),
                screen_hdc,
                ctypes.byref(dest),
                ctypes.byref(size),
                mem_dc,
                ctypes.byref(origin),
                chroma_colorref(),
                None,
                win32con.ULW_COLORKEY,
            )
            if not updated:
                raise OSError(f"UpdateLayeredWindow failed ({ctypes.GetLastError()})")
        finally:
            if mem_dc and old:
                gdi32.SelectObject(mem_dc, old)
            if bitmap:
                gdi32.DeleteObject(bitmap)
            if mem_dc:
                gdi32.DeleteDC(mem_dc)
            win32gui.ReleaseDC(0, screen_hdc)


def create_live_overlay(*, enabled: bool) -> LiveOverlay:
    if not enabled:
        return NullLiveOverlay()
    if sys.platform == "win32":
        return Win32LiveOverlay()
    return NullLiveOverlay()
