from __future__ import annotations

import argparse
import signal
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

import cv2
import numpy as np
from numpy.typing import NDArray

from hero_siege_bot.calibration import AnchorRegion, AutoCalibrator
from hero_siege_bot.capture import WindowCapture
from hero_siege_bot.config import BotConfig, load_config
from hero_siege_bot.controllers import CombatController, LootController, SurvivalController
from hero_siege_bot.detectors import ScreenStateDetector, TemplateDetector
from hero_siege_bot.diagnostics import DiagnosticsOverlay, JsonlRecorder
from hero_siege_bot.exploration import FrontierExplorer
from hero_siege_bot.input import DryRunInputBackend, InputBackend, SafeInput, SendInputBackend
from hero_siege_bot.perception import Perception
from hero_siege_bot.runtime import BotRuntime, Capture


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hero-siege-bot")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("dry-run", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name == "run":
            command.add_argument("--enable-input", action="store_true")
    return parser


def _abort(message: str) -> NoReturn:
    raise SystemExit(message)


def _load_calibrator(config: BotConfig) -> AutoCalibrator:
    anchor_dir = Path(__file__).with_name("assets") / "anchors"
    anchors: dict[str, NDArray[np.uint8]] = {}
    for path in sorted(anchor_dir.glob("*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            anchors[path.stem] = np.asarray(image, dtype=np.uint8)
    if not anchors:
        raise RuntimeError(
            "no calibrated anchor assets are installed; add validated assets before running"
        )

    # Task 8 replaces these conservative relative regions with measured annotations.
    anchor = next(iter(anchors))
    regions = {
        "gameplay": AnchorRegion(anchor, -8.0, -4.0, 16.0, 12.0),
        "health": AnchorRegion(anchor, 0.0, 0.0, 1.0, 0.25),
        "resource": AnchorRegion(anchor, 0.0, 0.25, 1.0, 0.25),
        "minimap": AnchorRegion(anchor, 0.0, 0.5, 1.0, 1.0),
        "screen_state": AnchorRegion(anchor, -8.0, -4.0, 16.0, 12.0),
    }
    return AutoCalibrator(config.calibration, anchors, regions)


def _load_template(name: str) -> NDArray[np.uint8]:
    path = Path(__file__).with_name("assets") / "templates" / f"{name}.png"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(
            f"validated detector template is not installed: {path.name}"
        )
    return np.asarray(image, dtype=np.uint8)


def build_runtime(
    config: BotConfig,
    backend: InputBackend,
    *,
    capture: Capture | None = None,
    diagnostics_root: Path = Path("diagnostics"),
) -> BotRuntime:
    window_capture = capture or WindowCapture(config.window.title)
    detector_config = config.detectors
    enemy_detector = TemplateDetector(
        {"enemy": _load_template("enemy")},
        confidence_threshold=detector_config.template_confidence,
        nms_iou_threshold=detector_config.nms_iou_threshold,
    )
    loot_detector = TemplateDetector(
        {"loot": _load_template("loot")},
        confidence_threshold=detector_config.template_confidence,
        nms_iou_threshold=detector_config.nms_iou_threshold,
    )
    screen_state_detector = ScreenStateDetector(
        _load_template("death"),
        _load_template("restart"),
        confidence_threshold=detector_config.template_confidence,
        nms_iou_threshold=detector_config.nms_iou_threshold,
    )
    perception = Perception(
        config=config,
        enemy_detector=enemy_detector,
        loot_detector=loot_detector,
        screen_state_detector=screen_state_detector,
    )
    recorder = (
        JsonlRecorder(
            diagnostics_root,
            frame_interval_s=config.recording.frame_interval_s,
            overlay=DiagnosticsOverlay() if config.recording.overlay else None,
        )
        if config.recording.enabled
        else None
    )
    safe_input = SafeInput(
        backend,
        max_key_hold_s=config.exploration.max_movement_pulse_s,
        max_mouse_hold_s=config.combat.attack_hold_s,
    )
    return BotRuntime(
        capture=window_capture,
        calibrator=_load_calibrator(config),
        perception=perception,
        explorer=FrontierExplorer(config.exploration),
        combat=CombatController(config.combat),
        survival=SurvivalController(config.survival),
        loot=LootController(config.combat),
        recorder=recorder,
        input_controller=safe_input,
        calibration_confidence=config.calibration.confidence_threshold,
        no_progress_sample_limit=config.exploration.no_progress_sample_limit,
        movement_pulse_s=config.exploration.movement_pulse_s,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "run":
        if sys.platform != "win32":
            _abort("run is only supported on Windows")
        if not arguments.enable_input:
            _abort("run requires explicit --enable-input")

    config = load_config(arguments.config)
    capture: WindowCapture | None = None
    if arguments.command == "dry-run":
        backend: InputBackend = DryRunInputBackend()
    else:
        capture = WindowCapture(config.window.title)
        client_rect = capture.find()
        if client_rect is None:
            _abort("Hero Siege window was not found")
        backend = SendInputBackend(client_rect)

    runtime = build_runtime(config, backend, capture=capture)
    stop = threading.Event()

    def request_stop(signum: int, frame: object) -> None:
        del signum, frame
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    runtime.run(stop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
