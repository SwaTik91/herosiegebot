# Recorded frame fixtures

Task 5 uses generated images so detector expectations remain deterministic. Private
recorded fixtures may be added as paired `<case>.png` and `<case>.yaml` files. Recorded
tests skip only when this directory contains no such pairs.

Each sidecar YAML file must contain:

- `frame_size`: `{width, height}`
- `hud_rectangles`: named `{x, y, width, height}` entries for `health`, `resource`,
  `minimap`, `gameplay`, and `screen_state`
- `player_map_point`: normalized `{x, y}`, or `null` when not visible
- `bar_ratios`: `health` and `resource`, each a ratio in `[0, 1]` or `null`
- `enemy_boxes`: a list of `{x, y, width, height}` frame-pixel rectangles
- `loot_boxes`: a list of `{x, y, width, height}` frame-pixel rectangles
- `death`: boolean
- `restart_visible`: boolean

Do not infer labels from a single screenshot. Add a case only after its boxes and state
flags have been manually verified, and keep production thresholds in configuration.
