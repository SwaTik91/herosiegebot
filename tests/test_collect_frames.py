import importlib.util
from pathlib import Path
from types import ModuleType
from typing import cast

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from hero_siege_bot.capture import CapturedFrame
from hero_siege_bot.domain import Rect


def _load_collector() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "collect_frames", Path("scripts/collect_frames.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collect_frames = cast("object", _load_collector().collect_frames)


class FakeCapture:
    def __init__(self, images: list[NDArray[np.uint8] | None]) -> None:
        self._images = iter(images)

    def grab(self) -> CapturedFrame | None:
        image = next(self._images)
        if image is None:
            return None
        height, width = image.shape[:2]
        return CapturedFrame(
            image=image,
            client_rect=Rect(25, 50, width, height),
            focused=True,
            timestamp=1234.5,
        )


def test_collect_frames_writes_lossless_numbered_pngs(tmp_path: Path) -> None:
    first = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    second = np.flip(first, axis=1).copy()

    saved = collect_frames(
        FakeCapture([first, second]),
        tmp_path,
        count=2,
        interval_s=0.0,
        sleep=lambda _: None,
    )

    assert [path.name for path in saved] == [
        "frame_0001_6x4.png",
        "frame_0002_6x4.png",
    ]
    assert np.array_equal(cv2.imread(str(saved[0]), cv2.IMREAD_COLOR), first)
    assert np.array_equal(cv2.imread(str(saved[1]), cv2.IMREAD_COLOR), second)


def test_collect_frames_stops_when_window_capture_is_unavailable(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="window was not found or could not be captured"):
        collect_frames(
            FakeCapture([None]),
            tmp_path,
            count=1,
            interval_s=0.0,
            sleep=lambda _: None,
        )

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(("count", "interval_s"), [(0, 0.0), (1, -0.1)])
def test_collect_frames_rejects_invalid_collection_settings(
    tmp_path: Path, count: int, interval_s: float
) -> None:
    with pytest.raises(ValueError):
        collect_frames(
            FakeCapture([]),
            tmp_path,
            count=count,
            interval_s=interval_s,
            sleep=lambda _: None,
        )
