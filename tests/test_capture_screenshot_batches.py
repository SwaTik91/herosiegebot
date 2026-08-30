import importlib.util
import zipfile
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import cast

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "capture_screenshot_batches",
        Path("scripts/capture_screenshot_batches.py"),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def recorder_module() -> ModuleType:
    return _load_module()


def _image(value: int) -> NDArray[np.uint8]:
    return np.full((4, 6, 3), value, dtype=np.uint8)


def test_run_loop_archives_every_full_batch_and_leaves_remainder(
    tmp_path: Path, recorder_module: ModuleType
) -> None:
    pending = tmp_path / "pending"
    archives = tmp_path / "archives"
    frames = [_image(index) for index in range(7)]
    grabs = iter(frames)
    continue_flags = iter([True] * 7 + [False])

    zips = recorder_module.run_capture_loop(
        lambda: next(grabs),
        pending,
        archives,
        batch_size=3,
        interval_s=0.0,
        sleep=lambda _: None,
        should_continue=lambda: next(continue_flags),
        now=lambda: datetime(2026, 8, 30, 10, 6, 0),
    )

    assert [path.name for path in zips] == [
        "screenshots_001_20260830-100600.zip",
        "screenshots_002_20260830-100600.zip",
        "screenshots_003_20260830-100600.zip",
    ]
    assert list(pending.iterdir()) == []
    with zipfile.ZipFile(zips[0]) as archive:
        assert archive.namelist() == [
            "shot_00001.png",
            "shot_00002.png",
            "shot_00003.png",
        ]
        payload = np.frombuffer(archive.read("shot_00001.png"), dtype=np.uint8)
        assert np.array_equal(cv2.imdecode(payload, cv2.IMREAD_COLOR), frames[0])
    with zipfile.ZipFile(zips[2]) as archive:
        assert archive.namelist() == ["shot_00007.png"]


def test_flush_archives_leftover_shots_on_stop(
    tmp_path: Path, recorder_module: ModuleType
) -> None:
    pending = tmp_path / "pending"
    archives = tmp_path / "archives"
    continue_flags = iter([True, True, False])
    grabs = iter([_image(1), _image(2)])

    zips = recorder_module.run_capture_loop(
        lambda: next(grabs),
        pending,
        archives,
        batch_size=50,
        interval_s=0.0,
        sleep=lambda _: None,
        should_continue=lambda: next(continue_flags),
        now=lambda: datetime(2026, 8, 30, 10, 6, 5),
    )

    assert [path.name for path in zips] == ["screenshots_001_20260830-100605.zip"]
    assert list(pending.iterdir()) == []
    with zipfile.ZipFile(zips[0]) as archive:
        assert archive.namelist() == ["shot_00001.png", "shot_00002.png"]


@pytest.mark.parametrize(("batch_size", "interval_s"), [(0, 1.0), (50, -0.1)])
def test_run_loop_rejects_invalid_settings(
    tmp_path: Path,
    recorder_module: ModuleType,
    batch_size: int,
    interval_s: float,
) -> None:
    with pytest.raises(ValueError):
        recorder_module.run_capture_loop(
            lambda: _image(1),
            tmp_path / "pending",
            tmp_path / "archives",
            batch_size=batch_size,
            interval_s=interval_s,
            sleep=lambda _: None,
            should_continue=lambda: False,
        )
