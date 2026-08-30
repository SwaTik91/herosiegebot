# Hero Siege calibration

## Evidence currently available

The repository contains one user-supplied real frame:
`tests/fixtures/frames/highland_graveyard_1024x655.png`. Although the supplied
description called it 1024×687, the decoded PNG is 1024×655; annotations use the
decoded dimensions. The frame visibly shows Highland Graveyard / Woodhill
Plains, Hell, zone level 290. The Hero Siege build and UI scale are not visible
and remain unknown.

Its YAML sidecar records only manually verifiable facts:

- health and resource bars are full, consistent with the visible `2414 / 2414`
  and `1081 / 1081` labels;
- normal gameplay is visible, so `death` and `restart_visible` are false;
- player-map position, enemy boxes, and loot boxes are explicitly `unknown`
  because a single still does not independently establish those semantic
  identities. Unknown is not treated as verified absence.

Two stable, number-free crops were taken from this frame:

- `hud_status_right_cap.png`: frame pixels `(160, 12, 20, 32)`;
- `minimap_top_left_corner.png`: frame pixels `(898, 0, 24, 24)`.

No player-marker, enemy, loot, bar-border, fog, cooldown, or pulse setting was
changed from this single frame: there is not enough semantic or temporal
evidence to tune those values honestly.

The exact fixture matches both anchors at the configured threshold and
reconstructs the annotated HUD regions. Deterministic offline tests also apply
0.8× and 1.2× resizing, translation, and positive/negative brightness offsets;
each calibration sequence uses three distinct frames with per-frame
brightness/contrast variation and seeded non-anchor background perturbations.
Both anchors remain above 0.9 confidence, region origins remain within two
pixels, and region overlap remains at least 0.8 IoU. These are synthetic
robustness checks on one source image, not claims of live multi-frame,
multi-resolution, UI-scale, or Windows stability.

## Windows frame collection

Use Python 3.12 and run from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python scripts\collect_frames.py --output-dir calibration-frames\1280x720 --count 30
```

`collect_frames.py` only constructs `WindowCapture`; it does not import or
construct an input backend. It writes lossless, numbered PNGs of the client
area and exits with an error if the window cannot be found or captured.

Repeat after setting the Hero Siege client area to each target:

```powershell
python scripts\collect_frames.py --output-dir calibration-frames\1920x1080 --count 30
python scripts\collect_frames.py --output-dir calibration-frames\borderless-moved --count 30
```

For the third set, use borderless mode, move the window away from `(0, 0)`, and
resize it once. Record Windows display scaling, in-game UI scale, game build,
location, and the client size printed in each filename.

## Annotation and tuning procedure

1. Pair every retained `<case>.png` with `<case>.yaml` using the schema in
   `tests/fixtures/frames/README.md`.
2. Verify enemy and loot boxes across adjacent frames or in-game behavior; do
   not infer them from appearance in one still.
3. Capture death and restart as separate cases with their state flags set.
4. Crop anchors only from fixed chrome. Exclude text, changing numbers, fill
   levels, map contents, animation, and character-specific art.
5. Change only configuration values. Run the fixture suite after every change.
6. For each detector and each resolution, record true positives, false
   positives, false negatives, sample count, and confidence min/median/max.
7. Reject a resolution if calibration needs absolute coordinates. Improve
   anchors or relative calibration instead.

## Results and known failures

- Offline fixture geometry: passed at decoded size 1024×655.
- Offline player-map semantic identity: unknown; no detector claim recorded.
- Supported Windows resolutions: not established.
- Measured detector confidence statistics: not established beyond exact anchor
  matching and deterministic transformations of one source frame.
- Known evidence gaps: no validated enemy, loot, death, or restart examples;
  no UI-scale/build metadata; no temporal sequence for motion or bar changes.
- Live dry-run, SendInput smoke tests, and 30-minute acceptance: blocked until
  run on Windows with Hero Siege. Follow `docs/windows-smoke-test.md`.
