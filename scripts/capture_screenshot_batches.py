from __future__ import annotations

import argparse
import random
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

Grab = Callable[[], NDArray[np.uint8]]
Sleep = Callable[[float], None]
Clock = Callable[[], datetime]
Continue = Callable[[], bool]


def _write_shot(pending_dir: Path, image: NDArray[np.uint8], index: int) -> Path:
    path = pending_dir / f"shot_{index:05d}.png"
    written = cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not written:
        raise RuntimeError(f"failed to write screenshot: {path}")
    return path


def _archive_batch(
    paths: Sequence[Path],
    archives_dir: Path,
    *,
    batch_index: int,
    stamped: datetime,
) -> Path:
    archives_dir.mkdir(parents=True, exist_ok=True)
    zip_path = archives_dir / (
        f"screenshots_{batch_index:03d}_{stamped.strftime('%Y%m%d-%H%M%S')}.zip"
    )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, arcname=path.name)
    for path in paths:
        path.unlink()
    return zip_path


def grab_screen() -> NDArray[np.uint8]:
    try:
        import mss
    except ImportError:
        mss = None
    if mss is not None:
        with mss.mss() as sct:
            raw = np.asarray(sct.grab(sct.monitors[1]))
        if raw.ndim != 3 or raw.shape[2] < 3:
            raise RuntimeError("screen capture returned an unexpected image")
        return np.ascontiguousarray(raw[:, :, :3])

    if sys.platform == "darwin":
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            completed = subprocess.run(
                ["screencapture", "-x", "-t", "png", str(temp_path)],
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError("screencapture failed")
            image = cv2.imread(str(temp_path), cv2.IMREAD_COLOR)
        finally:
            temp_path.unlink(missing_ok=True)
        if image is None:
            raise RuntimeError("failed to read macOS screenshot")
        return np.asarray(image, dtype=np.uint8)

    try:
        from PIL import ImageGrab
    except ImportError as error:
        raise RuntimeError(
            "install mss (`pip install mss`) or Pillow to capture the screen"
        ) from error
    grabbed = ImageGrab.grab()
    return cv2.cvtColor(np.asarray(grabbed), cv2.COLOR_RGB2BGR)


def run_capture_loop(
    grab: Grab,
    pending_dir: Path,
    archives_dir: Path,
    *,
    batch_size: int = 50,
    interval_s: float = 4.5,
    sleep: Sleep = time.sleep,
    should_continue: Continue = lambda: True,
    now: Clock = datetime.now,
) -> tuple[Path, ...]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if interval_s < 0:
        raise ValueError("interval_s must not be negative")

    pending_dir.mkdir(parents=True, exist_ok=True)
    archives_dir.mkdir(parents=True, exist_ok=True)

    archives: list[Path] = []
    pending: list[Path] = []
    index = 0

    def flush() -> None:
        if not pending:
            return
        archives.append(
            _archive_batch(
                pending,
                archives_dir,
                batch_index=len(archives) + 1,
                stamped=now(),
            )
        )
        pending.clear()

    try:
        while should_continue():
            index += 1
            pending.append(_write_shot(pending_dir, grab(), index))
            print(
                f"saved {pending[-1].name} ({len(pending)}/{batch_size} in current batch)",
                flush=True,
            )
            if len(pending) >= batch_size:
                flush()
                print(f"archived {archives[-1]}", flush=True)
            sleep(interval_s)
    except KeyboardInterrupt:
        print("\nstop requested", flush=True)
    leftover = bool(pending)
    flush()
    if leftover:
        print(f"archived leftover shots as {archives[-1]}", flush=True)
    return tuple(archives)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Take screenshots every 4-5 seconds, zip each 50 shots, "
            "and keep going until Ctrl+C."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/screenshot-batches"),
        help="root folder for pending shots and zip archives",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--interval-min", type=float, default=4.0)
    parser.add_argument("--interval-max", type=float, default=5.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.interval_min < 0 or arguments.interval_max < arguments.interval_min:
        raise SystemExit("interval-min must be >= 0 and <= interval-max")

    pending_dir = arguments.output_dir / "pending"
    archives_dir = arguments.output_dir / "archives"

    def sleep_jitter(_: float) -> None:
        time.sleep(random.uniform(arguments.interval_min, arguments.interval_max))

    print(
        f"capturing every {arguments.interval_min:g}-{arguments.interval_max:g}s; "
        f"zip every {arguments.batch_size} shots; stop with Ctrl+C",
        flush=True,
    )
    print(f"pending: {pending_dir.resolve()}", flush=True)
    print(f"archives: {archives_dir.resolve()}", flush=True)
    try:
        archives = run_capture_loop(
            grab_screen,
            pending_dir,
            archives_dir,
            batch_size=arguments.batch_size,
            interval_s=arguments.interval_min,
            sleep=sleep_jitter,
        )
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    if archives:
        print("archives ready to send:")
        for path in archives:
            print(path.resolve())
    else:
        print("no screenshots were saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
