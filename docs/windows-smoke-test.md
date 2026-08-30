# Windows smoke test and MVP acceptance

Run this checklist only on a Windows test machine with Hero Siege. Use a
non-critical character/location and keep physical access to the keyboard.
Record every command, result, failure, `diagnostics/events.jsonl`, and evidence
frame. A checkbox means observed on that machine, not inferred from unit tests.

## 1. Install and fail-closed checks

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest -v
hero-siege-bot --help
```

Confirm `dry-run` and `run` are listed. Before live validation, add manually
validated `enemy.png`, `loot.png`, `death.png`, and `restart.png` under
`src/hero_siege_bot/assets/templates/`. The supplied still frame does not
justify those templates. The CLI intentionally fails closed while any are
missing.

With Hero Siege closed:

```powershell
python scripts\collect_frames.py --count 1
hero-siege-bot run --config config\default.yaml --enable-input
```

Both commands must report that the window was not found, and no key or mouse
button may remain pressed.

## 2. Collect and validate calibration

Follow `docs/calibration.md` to collect 1280×720, 1920×1080, and moved/resized
borderless frame sets. Add verified sidecars/templates, then run:

```powershell
pytest -v
hero-siege-bot dry-run --config config\default.yaml
```

During dry-run:

- move and resize the borderless window;
- switch between the two target resolutions;
- observe HUD, mini-map marker, fog/frontiers, enemies, loot, bar ratios, and
  proposed actions in diagnostics;
- confirm no game movement, attack, skill, potion, loot, or restart input occurs.

Stop with `Ctrl+C`. Save diagnostics and record detection counts/confidences.
Do not continue to input tests if calibration drops, regions leave the client
area, or dry-run sends input.

## 3. Staged input checks

Start each stage from a safe, focused game state:

```powershell
hero-siege-bot run --config config\default.yaml --enable-input
```

The current MVP has no per-action feature flags. Therefore isolate stages by
choosing game states that trigger only the behavior under test, stopping the
process between stages, and reviewing `events.jsonl` before proceeding.

1. **Emergency stop:** during a movement pulse, press `Ctrl+Shift+F10`. Verify
   movement immediately stops, the process fails closed on further actions,
   and Windows key-state inspection or a text field shows no stuck `W/A/S/D`.
2. **Focus loss:** restart, allow one movement pulse, then Alt+Tab. Verify all
   input releases and no input reaches the unfocused application.
3. **Game close:** restart, allow one movement pulse, then close Hero Siege.
   Verify all input releases. Stop the bot with `Ctrl+C`.
4. **Movement/exploration:** in a cleared area, observe bounded WASD pulses,
   frontier changes, and window-relative targeting.
5. **Attack hold:** enter one controlled enemy encounter. Confirm the mouse
   hold never exceeds `combat.attack_hold_s`.
6. **Skills:** observe Q and E once each, then confirm their configured cooldown
   intervals from event timestamps.
7. **Potions:** safely lower health, then resource, and confirm keys 1 and 2
   trigger only below thresholds and respect `potion_cooldown_s`.
8. **Loot:** defeat one enemy with known loot and confirm the looting transition
   and bounded input.
9. **Recovery:** create a harmless obstruction. Confirm, in order,
   release-all, orthogonal pulse, reverse pulse, frontier blacklist, and resumed
   exploration.
10. **Death/restart:** permit one controlled death. Confirm all input releases
    in `dead`, restart is clicked only when the verified restart control is
    visible, and recalibration occurs before further input.

At every stage, press `Ctrl+Shift+F10` if behavior differs from diagnostics. A
failed emergency hotkey, focus-loss, or game-close release is a stop-ship
failure.

## 4. Thirty-minute acceptance

Only after all stages pass:

```powershell
hero-siege-bot run --config config\default.yaml --enable-input
```

Run one procedural location for 30 uninterrupted minutes. The evidence must
show combat, potion use, loot, blocked-movement recovery, and at least one
death/restart cycle. Save:

- `diagnostics/events.jsonl`;
- all evidence PNGs;
- the exact `config/default.yaml`;
- game build, location, UI scale, Windows display scaling, resolution, start
  time, end time, and any intervention.

Any manual gameplay intervention invalidates the acceptance run. Copy measured
statistics and the result into `docs/calibration.md`; do not mark a resolution
supported from an incomplete run.
