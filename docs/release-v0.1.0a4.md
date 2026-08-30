# Hero Siege Bot v0.1.0a4

## Changes

- Marks only the intentional full-frame `gameplay` and `screen_state` profile
  regions as frame-clipped.
- Clips those derived regions to the captured frame after anchor scaling, which
  removes harmless rounding overscan such as `Rect(-6, -1, 1638, 1045)` on a
  1600×1024 frame.
- Keeps HUD and minimap regions strict. Invalid or out-of-frame anchored crops
  are still rejected by perception instead of being silently corrected.
- Adds a 1600×1024 regression fixture built from the installed anchors and an
  integration test from calibration through perception.

## Validation scope

Offline tests cover the supplied scaled-profile geometry, high-confidence
matching, strict non-clippable regions, and valid perception crops. Live Windows
capture and the staged/30-minute smoke test are still required before broader
use.

No package or release was published as part of this change.
