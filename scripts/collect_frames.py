from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

import cv2

from hero_siege_bot.capture import CapturedFrame, WindowCapture


class FrameCapture(Protocol):
    def grab(self) -> CapturedFrame | None: ...


def collect_frames(
    capture: FrameCapture,
    output_dir: Path,
    *,
    count: int,
    interval_s: float,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Path, ...]:
    """Capture client-area frames without constructing an input backend."""
    if count <= 0:
        raise ValueError("count must be positive")
    if interval_s < 0:
        raise ValueError("interval_s must not be negative")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for index in range(1, count + 1):
        frame = capture.grab()
        if frame is None:
            raise RuntimeError("window was not found or could not be captured")

        height, width = frame.image.shape[:2]
        path = output_dir / f"frame_{index:04d}_{width}x{height}.png"
        written = cv2.imwrite(
            str(path),
            frame.image,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )
        if not written:
            raise RuntimeError(f"failed to write frame: {path}")
        saved.append(path)
        if index < count:
            sleep(interval_s)
    return tuple(saved)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save lossless Hero Siege client frames without emitting input."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("calibration-frames"),
        help="directory for captured PNG files (default: calibration-frames)",
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--interval", type=float, default=1.0, dest="interval_s")
    parser.add_argument("--window-title", default="Hero Siege")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        saved = collect_frames(
            WindowCapture(arguments.window_title),
            arguments.output_dir,
            count=arguments.count,
            interval_s=arguments.interval_s,
        )
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    for path in saved:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
