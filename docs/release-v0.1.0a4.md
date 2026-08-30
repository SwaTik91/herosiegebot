# Hero Siege Bot v0.1.0a4

## Changes

- Marks only the intentional full-frame `gameplay` and `screen_state` profile
  regions as fully frame-clipped.
- Clips those derived regions to the captured frame after anchor scaling, which
  removes harmless rounding overscan such as `Rect(-6, -1, 1638, 1045)` on a
  1600×1024 frame.
- Clips minimap edges only when each crossed edge exceeds the frame by at most
  5% of the region dimension. The reproduced `Rect(1403, 0, 200, 226)` becomes
  `Rect(1403, 0, 197, 226)`; larger overscan remains invalid.
- Keeps health/resource regions strict and preserves invalid-region rejection
  for minimap overscan beyond the explicit tolerance.
- Raises the shipped YAML and fallback calibration `max_scale` from 1.5 to 2.0,
  so the 1.6-scale Windows reproduction works without test-only configuration.
- Adds a 1600×1024 regression fixture built from the installed anchors and an
  integration test from calibration through perception.

## Validation scope

Offline tests cover the supplied scaled-profile geometry, high-confidence
matching, bounded minimap edge clipping, rejection beyond tolerance, strict
non-clippable regions, and valid perception crops. Live Windows capture and the
staged/30-minute smoke test are still required before broader use.

No package or release was published as part of this change.
