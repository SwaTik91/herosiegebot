# Hero Siege Vision Farm Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Windows Python MVP that automatically calibrates itself and farms one procedurally generated Hero Siege location for 30 minutes using only screen capture and normal input.

**Architecture:** A capture and perception pipeline produces immutable observations for a deterministic state machine. Separate exploration, combat, survival, loot, and input modules turn those observations into bounded actions; all computer-vision modules support recorded-frame tests and the real input backend can be replaced with a dry-run backend.

**Tech Stack:** Python 3.12, NumPy, OpenCV, dxcam, pywin32, PyYAML, pytest, Ruff, mypy

## Global Constraints

- Target platform is Windows with Hero Siege in borderless-window mode.
- Never read process memory or use DLL injection, hooks, or code injection.
- Emit only normal keyboard and mouse events through Windows `SendInput`.
- Movement uses WASD; potions use `1` and `2`; skills use `Q` and `E`; primary attack holds the left mouse button.
- Coordinates must be normalized from automatically detected window and HUD geometry, never hard-coded screen pixels.
- Calibration or focus loss must release all input and pause.
- `F12` must always release all input and stop active automation.
- Anti-cheat bypass and concealment are out of scope.

---

### Task 1: Project foundation and domain contracts

**Files:**
- Create: `pyproject.toml`
- Create: `src/hero_siege_bot/__init__.py`
- Create: `src/hero_siege_bot/domain.py`
- Create: `src/hero_siege_bot/config.py`
- Create: `config/default.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `BotConfig`, `Rect`, `Point`, `Detection`, `Observation`, `Action`, `BotState`, and `load_config(path: Path) -> BotConfig`.
- Consumes: none.

- [ ] **Step 1: Add packaging and test configuration**

Create `pyproject.toml` with Python `>=3.12`, runtime dependencies `numpy`, `opencv-python`, `PyYAML`, Windows-only dependencies `dxcam; sys_platform == 'win32'` and `pywin32; sys_platform == 'win32'`, plus dev dependencies `pytest`, `pytest-cov`, `ruff`, and `mypy`. Configure a `src` package layout, Ruff line length `100`, and pytest test path `tests`.

- [ ] **Step 2: Write failing configuration tests**

```python
from pathlib import Path

import pytest

from hero_siege_bot.config import load_config


def test_load_config_rejects_out_of_range_threshold(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("window:\n  title: Hero Siege\nsurvival:\n  health_threshold: 1.2\n")
    with pytest.raises(ValueError, match="health_threshold"):
        load_config(path)


def test_default_config_uses_expected_controls() -> None:
    config = load_config(Path("config/default.yaml"))
    assert config.controls.movement == {"up": "W", "left": "A", "down": "S", "right": "D"}
    assert config.controls.skills == ("Q", "E")
    assert config.controls.potions == ("1", "2")
    assert config.controls.emergency_stop == "F12"
```

- [ ] **Step 3: Run tests and verify import failure**

Run: `pytest tests/test_config.py -v`

Expected: FAIL because `hero_siege_bot.config` does not exist.

- [ ] **Step 4: Implement immutable domain models and validated config**

In `domain.py`, define frozen dataclasses and enums with these exact public fields:

```python
class BotState(StrEnum):
    CALIBRATING = auto()
    EXPLORING = auto()
    COMBAT = auto()
    LOOTING = auto()
    RECOVERING = auto()
    DEAD = auto()
    RESTARTING = auto()
    PAUSED = auto()


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class Detection:
    kind: str
    center: Point
    confidence: float


@dataclass(frozen=True)
class Observation:
    timestamp: float
    calibrated: bool
    calibration_confidence: float
    focused: bool
    health_ratio: float | None
    resource_ratio: float | None
    player_map_position: Point | None
    enemies: tuple[Detection, ...]
    loot: tuple[Detection, ...]
    dead: bool
    restart_visible: bool
    movement_progress: float


@dataclass(frozen=True)
class Action:
    kind: str
    key: str | None = None
    target: Point | None = None
    duration_s: float = 0.0
```

In `config.py`, use frozen nested dataclasses for window, controls, calibration, survival, combat, exploration, recording, and root configuration. Validate every ratio as `0.0 <= value <= 1.0` and every duration as positive. Write the agreed controls and conservative initial timing values to `config/default.yaml`.

- [ ] **Step 5: Run quality checks**

Run: `pytest tests/test_config.py -v && ruff check src tests && mypy src`

Expected: all commands succeed.

- [ ] **Step 6: Commit the foundation**

```bash
git add pyproject.toml src/hero_siege_bot config/default.yaml tests/test_config.py
git commit -m "build: establish bot domain and configuration"
```

---

### Task 2: Deterministic state machine

**Files:**
- Create: `src/hero_siege_bot/state_machine.py`
- Test: `tests/test_state_machine.py`

**Interfaces:**
- Consumes: `BotState` and `Observation` from `hero_siege_bot.domain`.
- Produces: `BotStateMachine(initial: BotState = BotState.CALIBRATING)` with `state` and `update(observation: Observation) -> BotState`.

- [ ] **Step 1: Write transition tests**

Create an `observation(**overrides)` fixture factory with safe defaults, then test:

```python
def test_calibration_enters_exploration() -> None:
    machine = BotStateMachine()
    assert machine.update(observation(calibrated=True, calibration_confidence=0.95)) is BotState.EXPLORING


def test_enemy_interrupts_exploration() -> None:
    machine = BotStateMachine(BotState.EXPLORING)
    enemy = Detection("enemy", Point(0.5, 0.5), 0.9)
    assert machine.update(observation(enemies=(enemy,))) is BotState.COMBAT


def test_focus_loss_always_pauses() -> None:
    for state in BotState:
        machine = BotStateMachine(state)
        assert machine.update(observation(focused=False)) is BotState.PAUSED


def test_death_preempts_active_states() -> None:
    machine = BotStateMachine(BotState.COMBAT)
    assert machine.update(observation(dead=True)) is BotState.DEAD
```

Also cover combat-to-looting, looting-to-exploring, no-progress-to-recovering, dead-to-restarting, and restarting-to-calibrating.

- [ ] **Step 2: Verify tests fail**

Run: `pytest tests/test_state_machine.py -v`

Expected: FAIL because `state_machine.py` does not exist.

- [ ] **Step 3: Implement ordered transition rules**

Implement one pure transition method. Apply global safety transitions first (`focused`, `calibrated`, `dead`), then state-specific transitions. Require configurable confidence for enemies and loot; do not embed input behavior in this module.

- [ ] **Step 4: Verify transitions and quality**

Run: `pytest tests/test_state_machine.py -v && ruff check src tests && mypy src`

Expected: all commands succeed.

- [ ] **Step 5: Commit the state machine**

```bash
git add src/hero_siege_bot/state_machine.py tests/test_state_machine.py
git commit -m "feat: add deterministic farming state machine"
```

---

### Task 3: Window capture and automatic calibration

**Files:**
- Create: `src/hero_siege_bot/capture.py`
- Create: `src/hero_siege_bot/calibration.py`
- Create: `src/hero_siege_bot/assets/anchors/.gitkeep`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Produces: `CapturedFrame(image: NDArray[np.uint8], client_rect: Rect, focused: bool, timestamp: float)`.
- Produces: `WindowCapture.find() -> Rect | None`, `WindowCapture.grab() -> CapturedFrame | None`.
- Produces: `Calibration(regions: Mapping[str, Rect], scale: float, confidence: float)` and `AutoCalibrator.calibrate(frames: Sequence[CapturedFrame]) -> Calibration | None`.
- Consumes: `Rect` and calibration configuration.

- [ ] **Step 1: Write synthetic calibration tests**

Generate a 1920×1080 black image containing distinctive synthetic HUD and mini-map templates at known normalized positions. Resize the complete image to 1280×720 and assert:

```python
result = calibrator.calibrate(frames)
assert result is not None
assert result.confidence >= 0.9
assert rect_iou(result.regions["minimap"], expected_minimap) >= 0.9
assert rect_iou(result.regions["health"], expected_health) >= 0.9
```

Add tests that reject one frame with no anchors and reject anchors that disagree across the frame sequence.

- [ ] **Step 2: Verify calibration tests fail**

Run: `pytest tests/test_calibration.py -v`

Expected: FAIL because calibration classes are missing.

- [ ] **Step 3: Implement multi-scale anchor calibration**

Use grayscale `cv2.matchTemplate(..., cv2.TM_CCOEFF_NORMED)` over a configured scale range. Require the HUD and mini-map anchors in at least three consecutive frames. Compute named regions from normalized offsets relative to matched anchors and set confidence to the minimum stable anchor score.

- [ ] **Step 4: Implement the Windows capture adapter**

Use `win32gui.EnumWindows` to locate a visible title match and obtain client coordinates with `ClientToScreen`. Use one `dxcam.create(output_color="BGR")` camera and call `grab(region=(left, top, right, bottom))`. Keep all pywin32/dxcam imports inside the Windows adapter so offline tests import on macOS or Linux.

- [ ] **Step 5: Run offline tests and static checks**

Run: `pytest tests/test_calibration.py -v && ruff check src tests && mypy src`

Expected: all commands succeed without requiring Windows.

- [ ] **Step 6: Commit capture and calibration**

```bash
git add src/hero_siege_bot/capture.py src/hero_siege_bot/calibration.py src/hero_siege_bot/assets tests/test_calibration.py
git commit -m "feat: add window capture and automatic calibration"
```

---

### Task 4: Mini-map segmentation and frontier exploration

**Files:**
- Create: `src/hero_siege_bot/exploration.py`
- Test: `tests/test_exploration.py`
- Create: `tests/fixtures/minimap/README.md`

**Interfaces:**
- Produces: `MapMasks(explored: NDArray[np.bool_], fog: NDArray[np.bool_], walkable: NDArray[np.bool_])`.
- Produces: `segment_minimap(image: NDArray[np.uint8], config: ExplorationConfig) -> MapMasks`.
- Produces: `FrontierExplorer.choose_target(masks: MapMasks, player: Point) -> Point | None`.
- Produces: `FrontierExplorer.movement_action(player: Point, target: Point) -> Action`.

- [ ] **Step 1: Write segmentation and frontier tests**

Build synthetic mini-maps with a central explored room, dark fog, and one corridor. Assert exact fog/explored masks after morphology, that frontier pixels lie on the explored side of the boundary, and that the selected target is reachable by flood-fill from the player.

Add a test where the closest Euclidean frontier is separated by a wall; the explorer must select the farther reachable frontier. Add a no-progress test that excludes the current frontier after three failed movement samples.

- [ ] **Step 2: Verify tests fail**

Run: `pytest tests/test_exploration.py -v`

Expected: FAIL because `exploration.py` does not exist.

- [ ] **Step 3: Implement mini-map segmentation**

Convert BGR to HSV, classify fog and explored pixels from configurable ranges, apply open/close morphology, and derive walkable space. Keep all thresholds in `ExplorationConfig`.

- [ ] **Step 4: Implement reachable frontier selection**

Define a frontier as an explored/walkable pixel adjacent to fog. Flood-fill reachable pixels from the normalized player position, cluster frontier pixels, and score each cluster by path length, newly revealable fog area, and recent-failure penalty.

- [ ] **Step 5: Implement bounded WASD output**

Convert the target vector into one or two `Action(kind="key_hold", key=..., duration_s=pulse)` values, allowing diagonal movement but clamping every pulse to the configured maximum duration.

- [ ] **Step 6: Verify exploration behavior**

Run: `pytest tests/test_exploration.py -v && ruff check src tests && mypy src`

Expected: all commands succeed.

- [ ] **Step 7: Commit exploration**

```bash
git add src/hero_siege_bot/exploration.py tests/test_exploration.py tests/fixtures/minimap
git commit -m "feat: explore procedural maps from minimap frontiers"
```

---

### Task 5: Visual perception pipeline

**Files:**
- Create: `src/hero_siege_bot/perception.py`
- Create: `src/hero_siege_bot/detectors.py`
- Create: `tests/fixtures/frames/README.md`
- Test: `tests/test_detectors.py`
- Test: `tests/test_perception.py`

**Interfaces:**
- Produces: `Detector` protocol with `detect(image: NDArray[np.uint8]) -> tuple[Detection, ...]`.
- Produces: `BarReader.read_ratio(image: NDArray[np.uint8]) -> tuple[float | None, float]`.
- Produces: `TemplateDetector`, `MotionColorDetector`, and `ScreenStateDetector`.
- Produces: `Perception.observe(frame: CapturedFrame, calibration: Calibration) -> Observation`.
- Consumes: calibrated named regions, `segment_minimap`, detector configuration, and prior frame.

- [ ] **Step 1: Write detector tests with generated images**

Test bar ratios at 0%, 25%, 50%, and 100%, rejecting images with insufficient border confidence. Test template detection with two non-overlapping matches and non-maximum suppression. Test motion/color detection using two sequential frames where one colored sprite moves. Test death/restart state detection from synthetic templates.

- [ ] **Step 2: Verify detector tests fail**

Run: `pytest tests/test_detectors.py tests/test_perception.py -v`

Expected: FAIL because detector modules are missing.

- [ ] **Step 3: Implement focused detectors**

Implement each detector as a small class. Return normalized centers in `[0, 1]`, confidence in `[0, 1]`, and immutable tuples. Use color masks plus temporal motion for initial enemy candidates and template/color filtering for loot candidates.

- [ ] **Step 4: Compose observations**

Crop only named calibrated regions, read bars, segment the mini-map, locate the player marker, and combine enemy, loot, death, and restart detections into one `Observation`. If a required crop is invalid, return `calibrated=False` instead of raising.

- [ ] **Step 5: Add recorded-frame fixture rules**

In `tests/fixtures/frames/README.md`, specify filenames and expected sidecar YAML fields: frame size, HUD rectangles, player map point, bar ratios, enemy boxes, loot boxes, death, and restart visibility. Tests must skip recorded cases only when no private fixtures have been added.

- [ ] **Step 6: Verify perception**

Run: `pytest tests/test_detectors.py tests/test_perception.py -v && ruff check src tests && mypy src`

Expected: generated-image tests pass.

- [ ] **Step 7: Commit perception**

```bash
git add src/hero_siege_bot/perception.py src/hero_siege_bot/detectors.py tests/test_detectors.py tests/test_perception.py tests/fixtures/frames
git commit -m "feat: derive game observations from captured frames"
```

---

### Task 6: Safe input and action controllers

**Files:**
- Create: `src/hero_siege_bot/input.py`
- Create: `src/hero_siege_bot/controllers.py`
- Test: `tests/test_input.py`
- Test: `tests/test_controllers.py`

**Interfaces:**
- Produces: `InputBackend` protocol with `key_down`, `key_up`, `mouse_move`, `mouse_down`, `mouse_up`.
- Produces: `DryRunInputBackend.events: list[tuple[str, object]]` and Windows `SendInputBackend`.
- Produces: `SafeInput.execute(actions: Sequence[Action]) -> None`, `SafeInput.release_all() -> None`, and `SafeInput.emergency_stop() -> None`.
- Produces: `CombatController.actions(observation: Observation, now: float) -> tuple[Action, ...]`.
- Produces: `SurvivalController.actions(observation: Observation, now: float) -> tuple[Action, ...]`.
- Produces: `LootController.actions(observation: Observation, now: float) -> tuple[Action, ...]`.

- [ ] **Step 1: Write safety tests**

Using `DryRunInputBackend`, assert that a bounded key hold emits down then up, mouse holds never exceed configured maximums, `release_all()` releases every tracked key/button, and `emergency_stop()` rejects later actions until explicitly reset.

- [ ] **Step 2: Write controller tests**

Assert that combat targets the highest-confidence nearby enemy, emits a bounded left-button hold, and emits `Q/E` only after their cooldowns. Assert potion `1` is used below the health threshold and not repeated during cooldown. Assert loot times out and returns no action after its configured deadline.

- [ ] **Step 3: Verify tests fail**

Run: `pytest tests/test_input.py tests/test_controllers.py -v`

Expected: FAIL because input and controller modules are missing.

- [ ] **Step 4: Implement safe input orchestration**

Track every pressed key and mouse button in `SafeInput`. Wrap execution in `try/finally` so bounded holds always release. Register a Windows `F12` hotkey that calls `emergency_stop`; keep registration behind a platform adapter.

- [ ] **Step 5: Implement `SendInputBackend`**

Use pywin32 structures or `ctypes.windll.user32.SendInput` with scan codes for WASD, `1`, `2`, `Q`, `E`, and `F12`. Convert normalized target coordinates to calibrated client coordinates before mouse movement. Do not use window messages or process hooks.

- [ ] **Step 6: Implement combat, survival, and loot policies**

Keep cooldown state inside controller instances. Emit only domain `Action` values; controllers must never call an input backend directly.

- [ ] **Step 7: Verify input and controllers**

Run: `pytest tests/test_input.py tests/test_controllers.py -v && ruff check src tests && mypy src`

Expected: all offline tests succeed.

- [ ] **Step 8: Commit input and controllers**

```bash
git add src/hero_siege_bot/input.py src/hero_siege_bot/controllers.py tests/test_input.py tests/test_controllers.py
git commit -m "feat: add safe controls for combat and survival"
```

---

### Task 7: Runtime loop, diagnostics, and recovery

**Files:**
- Create: `src/hero_siege_bot/runtime.py`
- Create: `src/hero_siege_bot/diagnostics.py`
- Create: `src/hero_siege_bot/cli.py`
- Test: `tests/test_runtime.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `BotRuntime.step() -> BotState` and `BotRuntime.run(stop: threading.Event) -> None`.
- Produces: `JsonlRecorder.record(observation: Observation, state: BotState, actions: Sequence[Action]) -> None`.
- Produces: CLI commands `hero-siege-bot dry-run --config PATH` and `hero-siege-bot run --config PATH`.
- Consumes: all previously defined modules.

- [ ] **Step 1: Write runtime tests with fakes**

Create fake capture, calibrator, perception, explorer, controllers, recorder, and input. Verify:

```python
def test_focus_loss_releases_input_and_pauses(runtime, input_spy) -> None:
    runtime.capture.next_frame = frame(focused=False)
    assert runtime.step() is BotState.PAUSED
    assert input_spy.release_all_calls == 1


def test_no_progress_enters_recovery_and_changes_direction(runtime) -> None:
    runtime.perception.observations = [obs(movement_progress=0.0)] * 3
    states = [runtime.step() for _ in range(3)]
    assert states[-1] is BotState.RECOVERING
```

Also verify calibration before input, combat preemption, loot timeout, death/restart, recorder output, and release-all after an exception.

- [ ] **Step 2: Verify runtime tests fail**

Run: `pytest tests/test_runtime.py -v`

Expected: FAIL because runtime modules are missing.

- [ ] **Step 3: Implement one-step orchestration**

`step()` must capture, calibrate/recalibrate, observe, update state, select state-specific actions, record the decision, render diagnostics when enabled, then execute actions. Any low-confidence safety condition calls `release_all()` and returns `PAUSED`.

- [ ] **Step 4: Implement recovery policy**

After three no-progress samples, emit a short release-all, one orthogonal pulse, and one reverse pulse on successive steps. If progress remains zero, blacklist the current frontier and return to exploration.

- [ ] **Step 5: Implement JSONL recording and overlay**

Write timestamped session directories containing `events.jsonl` and selected PNG evidence frames. The overlay must draw calibrated regions, enemy/loot markers, mini-map frontier and target, confidence values, proposed actions, and current state; it must never modify images passed into perception.

- [ ] **Step 6: Implement CLI and dry-run**

Register `[project.scripts] hero-siege-bot = "hero_siege_bot.cli:main"`. `dry-run` must instantiate `DryRunInputBackend`; `run` must refuse non-Windows platforms and require an explicit `--enable-input` flag.

- [ ] **Step 7: Verify the complete offline suite**

Run: `pytest --cov=hero_siege_bot --cov-report=term-missing && ruff check src tests && mypy src`

Expected: all tests and checks pass; safety and state modules have branch coverage.

- [ ] **Step 8: Commit runtime**

```bash
git add pyproject.toml src/hero_siege_bot/runtime.py src/hero_siege_bot/diagnostics.py src/hero_siege_bot/cli.py tests/test_runtime.py
git commit -m "feat: orchestrate safe farming runtime"
```

---

### Task 8: Windows calibration and MVP acceptance

**Files:**
- Create: `docs/calibration.md`
- Create: `docs/windows-smoke-test.md`
- Create: `scripts/collect_frames.py`
- Modify: `config/default.yaml`
- Add: `src/hero_siege_bot/assets/anchors/*`
- Add: `tests/fixtures/frames/*`

**Interfaces:**
- Consumes: the completed CLI, recorder, fixture format, and selected Hero Siege location/build.
- Produces: real HUD anchors, measured detector thresholds, reproducible smoke-test instructions, and acceptance evidence.

- [ ] **Step 1: Install and verify on Windows**

Run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest -v
hero-siege-bot --help
```

Expected: tests pass and both `dry-run` and `run` are listed.

- [ ] **Step 2: Collect calibration frames**

Implement `scripts/collect_frames.py` to locate the window and save lossless PNG frames without emitting input. Collect frames at 1280×720, 1920×1080, and one moved/resized borderless window position.

- [ ] **Step 3: Create real anchors and fixture annotations**

Crop stable mini-map and HUD anchors that do not contain changing numbers or fill levels. Add sidecar YAML annotations for health/resource bars, player mini-map position, representative enemies, loot, death, and restart screens.

- [ ] **Step 4: Tune only configuration values**

Adjust HSV ranges, template thresholds, morphology kernels, cooldowns, and pulse durations in `config/default.yaml`. Do not introduce resolution-specific coordinates; failed resolutions require calibration improvements.

- [ ] **Step 5: Run live dry-run validation**

Run: `hero-siege-bot dry-run --config config/default.yaml`

Expected: overlay tracks window movement, HUD regions, mini-map player marker, fog/frontiers, enemies, loot, bars, and proposed actions without sending input.

- [ ] **Step 6: Run staged input smoke tests**

Follow `docs/windows-smoke-test.md` to enable, in order: emergency stop, movement pulses, exploration, attack hold, skills, potions, loot, recovery, death, and restart. Confirm `F12`, focus loss, and closing the game each release all input.

- [ ] **Step 7: Run the 30-minute acceptance session**

Run: `hero-siege-bot run --config config/default.yaml --enable-input`

Expected: one procedural location runs for 30 minutes without manual intervention, demonstrates combat, potions, loot, a blocked-movement recovery, and at least one death/restart cycle. Save `events.jsonl`, evidence frames, and the final configuration.

- [ ] **Step 8: Document measured results**

In `docs/calibration.md`, record supported resolutions, UI scale, selected location/build, detector confidence statistics, observed failure modes, and the acceptance-session result.

- [ ] **Step 9: Run final checks**

Run: `pytest --cov=hero_siege_bot && ruff check src tests scripts && mypy src`

Expected: all checks succeed.

- [ ] **Step 10: Commit Windows calibration**

```bash
git add config/default.yaml src/hero_siege_bot/assets tests/fixtures docs/calibration.md docs/windows-smoke-test.md scripts/collect_frames.py
git commit -m "test: calibrate vision bot on Hero Siege"
```
