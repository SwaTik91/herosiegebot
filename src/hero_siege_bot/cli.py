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

from hero_siege_bot.calibration import (
    AnchorRegion,
    AutoCalibrator,
    CalibrationProfile,
    NormalizedRegion,
)
from hero_siege_bot.capture import WindowCapture
from hero_siege_bot.config import BotConfig, load_config
from hero_siege_bot.controllers import CombatController, LootController, SurvivalController
from hero_siege_bot.detectors import ScreenStateDetector, TemplateDetector
from hero_siege_bot.diagnostics import DiagnosticsOverlay, JsonlRecorder
from hero_siege_bot.domain import BotState
from hero_siege_bot.exploration import FrontierExplorer
from hero_siege_bot.input import (
    DryRunInputBackend,
    EmergencyHotkey,
    InputBackend,
    SafeInput,
    SendInputBackend,
    WindowsEmergencyHotkey,
)
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


def _print_state(state: BotState) -> None:
    print(state.name, flush=True)


def _print_calibration_diagnostic(message: str) -> None:
    print(f"calibration: {message}", flush=True)


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

    required = {
        "hud_status_right_cap",
        "hud_status_right_cap_v2",
        "minimap_top_left_corner",
    }
    missing = required - anchors.keys()
    if missing:
        names = ", ".join(sorted(missing))
        raise RuntimeError(f"required calibrated anchors are missing: {names}")

    old_regions = {
        "health": AnchorRegion(
            "hud_status_right_cap", -5.1, 0.15625, 5.1, 0.375
        ),
        "resource": AnchorRegion(
            "hud_status_right_cap", -5.1, 0.6875, 5.1, 0.3125
        ),
        "minimap": AnchorRegion(
            "minimap_top_left_corner",
            0.0,
            0.0,
            5.25,
            143 / 24,
            edge_clip_tolerance=0.05,
        ),
        "gameplay": AnchorRegion(
            "hud_status_right_cap",
            -8.0,
            -0.375,
            51.2,
            20.46875,
            clip_to_frame=True,
        ),
        "screen_state": AnchorRegion(
            "hud_status_right_cap",
            -8.0,
            -0.375,
            51.2,
            20.46875,
            clip_to_frame=True,
        ),
    }
    current_regions = {
        "health": AnchorRegion(
            "hud_status_right_cap_v2", -3.48, 9 / 42, 4.08, 12 / 42
        ),
        "resource": AnchorRegion(
            "hud_status_right_cap_v2", -3.48, 26 / 42, 4.08, 10 / 42
        ),
        "minimap": AnchorRegion(
            "minimap_top_left_corner",
            0.0,
            0.0,
            5.25,
            143 / 24,
            edge_clip_tolerance=0.05,
        ),
        "gameplay": AnchorRegion(
            "hud_status_right_cap_v2",
            -5.8,
            -8 / 42,
            40.96,
            655 / 42,
            clip_to_frame=True,
        ),
        "screen_state": AnchorRegion(
            "hud_status_right_cap_v2",
            -5.8,
            -8 / 42,
            40.96,
            655 / 42,
            clip_to_frame=True,
        ),
    }
    minimap = anchors["minimap_top_left_corner"]
    profiles = (
        CalibrationProfile(
            "hud-v1",
            {
                "hud_status_right_cap": anchors["hud_status_right_cap"],
                "minimap_top_left_corner": minimap,
            },
            old_regions,
        ),
        CalibrationProfile(
            "hud-v2",
            {
                "hud_status_right_cap_v2": anchors["hud_status_right_cap_v2"],
                "minimap_top_left_corner": minimap,
            },
            current_regions,
        ),
    )
    fallback_regions = {
        "health": NormalizedRegion(
            87 / 1600, 26 / 1024, 163 / 1600, 19 / 1024
        ),
        "resource": NormalizedRegion(
            87 / 1600, 53 / 1024, 163 / 1600, 16 / 1024
        ),
        "minimap": NormalizedRegion(
            1403 / 1600, 0.0, 197 / 1600, 226 / 1024
        ),
        "gameplay": NormalizedRegion(0.0, 0.0, 1.0, 1.0),
        "screen_state": NormalizedRegion(0.0, 0.0, 1.0, 1.0),
    }
    return AutoCalibrator(
        config.calibration,
        profiles=profiles,
        fallback_regions=fallback_regions,
    )


def _load_template(name: str) -> NDArray[np.uint8]:
    path = Path(__file__).with_name("assets") / "templates" / f"{name}.png"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(
            f"validated detector template is not installed: {path.name}"
        )
    return np.asarray(image, dtype=np.uint8)


def _build_recorder(
    config: BotConfig, diagnostics_root: Path
) -> JsonlRecorder | None:
    if not config.recording.enabled:
        return None
    overlay = DiagnosticsOverlay() if config.recording.overlay else None
    return JsonlRecorder(
        diagnostics_root,
        frame_interval_s=config.recording.frame_interval_s,
        overlay=overlay,
    )


def build_runtime(
    config: BotConfig,
    backend: InputBackend,
    *,
    capture: Capture | None = None,
    diagnostics_root: Path = Path("diagnostics"),
    hotkey: EmergencyHotkey | None = None,
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
    recorder = _build_recorder(config, diagnostics_root)
    safe_input = SafeInput(
        backend,
        max_key_hold_s=config.exploration.max_movement_pulse_s,
        max_mouse_hold_s=config.combat.attack_hold_s,
        hotkey=hotkey,
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
        detection_confidence=config.combat.detection_confidence,
        state_reporter=_print_state,
        calibration_reporter=_print_calibration_diagnostic,
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
    hotkey: EmergencyHotkey | None = None
    if arguments.command == "dry-run":
        backend: InputBackend = DryRunInputBackend()
    else:
        capture = WindowCapture(config.window.title)
        client_rect = capture.find()
        if client_rect is None:
            _abort("Hero Siege window was not found")
        backend = SendInputBackend(client_rect)
        hotkey = WindowsEmergencyHotkey()

    runtime = build_runtime(config, backend, capture=capture, hotkey=hotkey)
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
