# Hero Siege Bot v0.1.0a3

## Changes

- Added a HUD-v2 calibration anchor from the supplied current Hero Siege frame.
- Added independent old-HUD and HUD-v2 calibration profiles that share the
  minimap anchor. One complete profile must remain stable for at least three
  frames; incompatible HUD anchors are no longer required simultaneously.
- Selects the highest-confidence profile when multiple complete profiles pass.
- Prints the initial runtime state and later state changes without per-frame log
  flooding.
- Preserves normalized, scale-aware region construction, calibration safety
  thresholds, focus fail-safes, and input-free dry-run behavior.

## Validation scope

Offline fixtures cover three visually varied frames for both known 1024×655 UI
profiles. Enemy, loot, player, death, and restart labels were not inferred from
the supplied screenshot.

Windows validation is still required for live capture, display scaling, other
resolutions and UI scales, state transitions, and the full staged/30-minute
smoke test.
