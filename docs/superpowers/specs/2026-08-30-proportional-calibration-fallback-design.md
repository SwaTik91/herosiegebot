# Proportional Calibration Fallback Design

## Problem

The current HUD and minimap templates include dynamic game pixels. On the
supplied focused 1600×1024 frame, the current HUD profile scores 0.525 and the
minimap scores 0.864 against a required confidence of 0.9. The legacy profile
also produces a higher-scoring false match. Capture and focus are healthy, but
calibration therefore returns `None` and runtime alternates between `PAUSED`
and `CALIBRATING`.

## Design

Keep strict template profiles as the primary calibration method. Add a
proportional fallback profile for the supported borderless Hero Siege HUD.
The fallback becomes eligible only after at least three focused frames have
identical image dimensions and client geometry.

Fallback regions are derived from normalized measurements shared by the
verified 1024×655 fixture and the supplied 1600×1024 frame:

- health and resource bars at the upper left;
- minimap against the upper-right edge;
- gameplay and screen-state regions covering the full captured frame.

All computed rectangles are clipped to the captured frame. The fallback does
not lower template thresholds and cannot select the false legacy anchor.
Focus loss or any geometry change continues to invalidate calibration.

## Data Flow

`BotRuntime` continues collecting focused frames. `AutoCalibrator` first
evaluates existing visual profiles. If none succeeds, it asks the proportional
profile to validate frame stability and derive regions. The returned
`Calibration` records its method so console diagnostics can report whether
templates or proportional geometry were used.

Perception and controllers receive the same region mapping as before; no input
behavior changes.

## Safety and Diagnostics

- Fewer than three stable frames never calibrate.
- Mixed frame sizes or client geometries reject the fallback.
- Existing focus and geometry invalidation remains mandatory.
- Full-frame and edge clipping prevents invalid crops.
- Console output reports calibration method and concise rejection reasons.
- The configured template confidence remains 0.9.

## Testing

- Reproduce template failure on the supplied 1600×1024 frame and verify the
  fallback produces valid regions and calibrated perception.
- Verify proportional regions against the existing 1024×655 annotated fixture.
- Reject fewer than three frames, mixed dimensions, and changed client geometry.
- Verify strict template profiles still win when their anchors are valid.
- Run the full test suite, Ruff, mypy, package build, and artifact checks.

## Release

Bump the package to `0.1.0a5`, document the fallback and diagnostics, build a
Windows test bundle, and publish a prerelease after review and verification.
