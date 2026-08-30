# Hero Siege Bot v0.1.0a5

## Changes

- Keeps strict template-anchor calibration as the primary method. A valid
  configured profile always wins.
- Adds a proportional geometry fallback only after template profiles are
  rejected. The fallback requires three consecutive focused frames with
  identical image dimensions and client rectangles.
- Requires independent health/resource-bar and minimap visual evidence before
  proportional geometry is eligible. Stable dimensions alone never identify
  the supported HUD; contentless and unsupported captures are rejected.
- Enforces the strict calibration confidence range from `0.9` through `1.0`;
  lower and non-finite configuration values are invalid.
- Keeps calibration pending when focus or capture geometry is unstable, so an
  unsafe fallback cannot enable input.
- Invalidates cached calibration and partial frame history whenever capture is
  unavailable.
- Prints changed calibration diagnostics for frame progress, unstable focus or
  geometry, capture/focus/geometry invalidation, and the selected calibration
  method. Repeated identical messages are suppressed.
- Validates normalized fallback crops at the original 1024×655 fixture size and
  the 1600×1024 scaled fixture size.

The exact reachable console order for a proportional calibration is:

```text
CALIBRATING
calibration: waiting for 3 stable frames (1/3)
calibration: waiting for 3 stable frames (2/3)
calibration: calibrated with proportional geometry
EXPLORING
```

## Windows upgrade note

Detector templates created by the tester are external user data. They are not
included in the wheel, source distribution, or Windows test archive.

Before extracting a fresh release, preserve every existing
`src\hero_siege_bot\assets\templates\*.png` file outside the installation,
including additional files such as `portal.png`. Extract the fresh release into
a new folder, re-add all preserved detector templates under the same
`assets\templates` path, and rerun the editable installation. Do not remove the
previous installation until the templates have been preserved and restored.

Do not restore customized files from `assets\anchors` into `0.1.0a5`. This
release ships its own calibration anchors and falls back to proportional
geometry when strict profiles do not match. Restoring stale or false anchors
could make template-first calibration select incorrect geometry.

## Validation scope

Offline tests cover strict-template precedence, the three-frame fallback gate,
real-fixture HUD eligibility, contentless/unsupported rejection, focus,
geometry and outage invalidation, independently annotated crop validity at
both fixture sizes, and changed-only diagnostic output. Live Windows capture,
input smoke tests, and the 30-minute acceptance run remain required before
broader use.

No tag, GitHub release, push, merge, or publication is performed by this
release-preparation change.
