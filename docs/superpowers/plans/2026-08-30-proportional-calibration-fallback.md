# Proportional Calibration Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Calibrate the supported borderless Hero Siege HUD from stable window geometry when dynamic visual anchors remain below 0.9.

**Architecture:** Preserve strict template profiles as the primary path in `AutoCalibrator`. Add a normalized, frame-relative fallback that becomes eligible only after three focused frames with identical image and client geometry; expose the selected method and bounded rejection diagnostics without changing perception or input contracts.

**Tech Stack:** Python 3.12+, OpenCV, NumPy, immutable dataclasses, pytest, Ruff, mypy, setuptools.

## Global Constraints

- Keep visual-template confidence at `0.9`; do not solve the defect by lowering it.
- Template calibration wins whenever a strict profile succeeds.
- Proportional fallback requires at least three focused frames with identical image dimensions and client geometry.
- Focus loss or geometry change continues to invalidate runtime calibration and release all input.
- Computed rectangles must remain within the captured frame.
- Package version and release tag are `0.1.0a5`.
- The supplied 1600×1024 image is evidence for calibration only; user-created enemy, loot, death, and restart templates are not redistributed.

---

### Task 1: Proportional Calibration Fallback

**Files:**
- Modify: `src/hero_siege_bot/calibration.py`
- Modify: `src/hero_siege_bot/cli.py`
- Modify: `tests/test_calibration.py`
- Create: `tests/fixtures/frames/boreal_island_1600x1024.png`
- Create: `tests/fixtures/frames/boreal_island_1600x1024.yaml`

**Interfaces:**
- Consumes: `CapturedFrame`, `CalibrationConfig`, strict `CalibrationProfile` definitions.
- Produces: `NormalizedRegion(x, y, width, height)`, optional fallback regions on `AutoCalibrator`, and `Calibration.method: str`.

- [ ] **Step 1: Add the supplied real frame and truthful sidecar**

Copy `/tmp/hsb-frame-a5.STjVQW/frame_0001_1600x1024.png` to
`tests/fixtures/frames/boreal_island_1600x1024.png`. Record only verified frame
geometry and region measurements in the YAML sidecar:

```yaml
frame_size: {width: 1600, height: 1024}
hud_rectangles:
  health: {x: 87, y: 26, width: 163, height: 19}
  resource: {x: 87, y: 53, width: 163, height: 16}
  minimap: {x: 1403, y: 0, width: 197, height: 226}
  gameplay: {x: 0, y: 0, width: 1600, height: 1024}
  screen_state: {x: 0, y: 0, width: 1600, height: 1024}
```

- [ ] **Step 2: Write failing real-frame fallback tests**

Add tests proving the current strict profiles return no candidate for the
supplied image, while the wished-for fallback returns in-bounds regions:

```python
def test_proportional_fallback_calibrates_real_1600x1024_frame() -> None:
    frames = _captured_frames(BOREAL_IMAGE, count=3)
    calibrator = _calibrator_with_proportional_fallback()

    result = calibrator.calibrate(frames)

    assert result is not None
    assert result.method == "proportional"
    assert result.regions == BOREAL_REGIONS
    assert result.confidence >= 0.9
```

Add a 1024×655 fixture test allowing at most two pixels of rounding difference
for health/resource and requiring the minimap to meet the annotated edge.

- [ ] **Step 3: Run focused tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_calibration.py \
  -k "proportional_fallback" -v
```

Expected: FAIL because `NormalizedRegion`, fallback construction, and
`Calibration.method` do not exist.

- [ ] **Step 4: Implement the minimal normalized fallback**

Add immutable normalized geometry:

```python
@dataclass(frozen=True)
class NormalizedRegion:
    x: float
    y: float
    width: float
    height: float
```

Extend `Calibration` backward-compatibly:

```python
@dataclass(frozen=True)
class Calibration:
    regions: Mapping[str, Rect]
    scale: float
    confidence: float
    method: str = "template"
```

Give `AutoCalibrator` optional `fallback_regions` and
`fallback_confidence=0.9`. After all strict profiles fail, use only the last
required frame window, require every frame to be focused, and compare both
`image.shape[:2]` and `client_rect`. Convert normalized values with `round`,
then clip with `_clip_to_frame`.

Configure `_load_calibrator` with normalized measurements from the 1600×1024
evidence:

```python
{
    "health": NormalizedRegion(87 / 1600, 26 / 1024, 163 / 1600, 19 / 1024),
    "resource": NormalizedRegion(87 / 1600, 53 / 1024, 163 / 1600, 16 / 1024),
    "minimap": NormalizedRegion(1403 / 1600, 0.0, 197 / 1600, 226 / 1024),
    "gameplay": NormalizedRegion(0.0, 0.0, 1.0, 1.0),
    "screen_state": NormalizedRegion(0.0, 0.0, 1.0, 1.0),
}
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_calibration.py -v
```

Expected: all calibration tests PASS.

- [ ] **Step 6: Add and verify safety boundary tests**

Add parameterized tests for one/two frames, an unfocused frame, changed image
shape, changed client position, and changed client size. Add a strict-profile
test asserting `method == "template"` even when fallback is configured.

Run:

```bash
.venv/bin/python -m pytest tests/test_calibration.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

```bash
git add src/hero_siege_bot/calibration.py src/hero_siege_bot/cli.py \
  tests/test_calibration.py tests/fixtures/frames/boreal_island_1600x1024.*
git commit -m "fix: add proportional calibration fallback"
```

---

### Task 2: Runtime Calibration Diagnostics

**Files:**
- Modify: `src/hero_siege_bot/calibration.py`
- Modify: `src/hero_siege_bot/runtime.py`
- Modify: `src/hero_siege_bot/cli.py`
- Modify: `tests/test_calibration.py`
- Modify: `tests/test_runtime.py`

**Interfaces:**
- Consumes: `AutoCalibrator.last_diagnostic`, accepted `Calibration.method`.
- Produces: optional `calibration_reporter: Callable[[str], None]` on `BotRuntime`, reporting only changed diagnostic messages.

- [ ] **Step 1: Write failing diagnostics tests**

Add tests for exact diagnostics:

```python
assert calibrator.last_diagnostic == "waiting for 3 stable frames (2/3)"
assert result.method == "proportional"
assert calibrator.last_diagnostic == "calibrated with proportional geometry"
```

In `tests/test_runtime.py`, assert a reporter receives a changed diagnostic
once rather than on every loop.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_calibration.py tests/test_runtime.py \
  -k "diagnostic or reporter" -v
```

Expected: FAIL because diagnostic state and reporter wiring do not exist.

- [ ] **Step 3: Implement bounded diagnostics**

Store `last_diagnostic` on `AutoCalibrator`; set it for insufficient frames,
unstable focus/geometry, strict-template success, and proportional success.
Add an optional protocol/property access in runtime without requiring custom
test calibrators to implement it. Cache the last emitted message and invoke
`calibration_reporter` only when it changes.

Wire CLI output as:

```text
CALIBRATING
calibration: waiting for 3 stable frames (2/3)
calibration: calibrated with proportional geometry
EXPLORING
```

- [ ] **Step 4: Run focused and full tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_calibration.py tests/test_runtime.py -v
.venv/bin/python -m pytest -q
```

Expected: all tests PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/hero_siege_bot/calibration.py src/hero_siege_bot/runtime.py \
  src/hero_siege_bot/cli.py tests/test_calibration.py tests/test_runtime.py
git commit -m "feat: report calibration diagnostics"
```

---

### Task 3: v0.1.0a5 Documentation, Review, and Release

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `docs/calibration.md`
- Create: `docs/release-v0.1.0a5.md`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: finished fallback and diagnostic behavior.
- Produces: versioned wheel, source distribution, Windows ZIP, checksums, tag, and GitHub prerelease.

- [ ] **Step 1: Add a failing version assertion**

```python
def test_package_version_is_a5() -> None:
    assert _project_version(Path("pyproject.toml")) == "0.1.0a5"
```

- [ ] **Step 2: Run version test and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_config.py::test_package_version_is_a5 -v
```

Expected: FAIL with actual version `0.1.0a4`.

- [ ] **Step 3: Update version and documentation**

Set `version = "0.1.0a5"`. Document that strict anchors remain primary,
proportional fallback requires three stable focused frames, and externally
created detector templates must remain in
`src/hero_siege_bot/assets/templates` for editable Windows installs.

- [ ] **Step 4: Verify code and package**

Run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy src
.venv/bin/python -m build
.venv/bin/python -m twine check \
  dist/hero_siege_bot-0.1.0a5-py3-none-any.whl \
  dist/hero_siege_bot-0.1.0a5.tar.gz
```

Expected: every command exits 0.

- [ ] **Step 5: Request code review and fix blocking findings**

Review the complete range from `origin/main` to `HEAD`, specifically checking
that template profiles win, fallback cannot calibrate unstable geometry,
normalized crops are valid at both fixture sizes, and detector templates are
not accidentally redistributed.

- [ ] **Step 6: Commit Task 3**

```bash
git add pyproject.toml README.md docs/calibration.md \
  docs/release-v0.1.0a5.md tests/test_config.py
git commit -m "chore: prepare v0.1.0a5"
```

- [ ] **Step 7: Merge, rebuild, and publish**

After user-authorized integration, fast-forward `main`, rerun the complete
verification, create the Windows ZIP with the a5 wheel, generate SHA-256
checksums, push `main` and tag `v0.1.0a5`, then publish a GitHub prerelease.

The release notes must tell this Windows tester to preserve their existing
`assets/templates/*.png` files when replacing source files.
