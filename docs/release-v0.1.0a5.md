# Hero Siege Bot v0.1.0a5

## Changes

- Keeps strict template-anchor calibration as the primary method. A valid
  configured profile always wins.
- Adds a proportional geometry fallback only after template profiles are
  rejected. The fallback requires three consecutive focused frames with
  identical image dimensions and client rectangles.
- Keeps calibration pending when focus or capture geometry is unstable, so an
  unsafe fallback cannot enable input.
- Prints changed calibration diagnostics for frame progress, template
  rejection, unstable focus or geometry, and the selected calibration method.
  Repeated identical messages are suppressed.
- Validates normalized fallback crops at the original 1024×655 fixture size and
  the 1600×1024 scaled fixture size.

## Windows upgrade note

Detector templates created by the tester are external user data. They are not
included in the wheel, source distribution, or Windows test archive.

Before extracting a fresh release, preserve the existing
`src\hero_siege_bot\assets\templates\*.png` files outside the installation.
Extract the fresh release into a new folder, re-add those PNG files under the
same path, and rerun the editable installation. Do not remove the previous
installation until the templates have been preserved and restored.

## Validation scope

Offline tests cover strict-template precedence, the three-frame fallback gate,
focus and geometry instability, normalized crop validity at both fixture
sizes, and changed-only diagnostic output. Live Windows capture, input smoke
tests, and the 30-minute acceptance run remain required before broader use.

No tag, GitHub release, push, merge, or publication is performed by this
release-preparation change.
