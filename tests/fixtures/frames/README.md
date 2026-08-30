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

`highland_graveyard_1024x655` is the first user-supplied real fixture. Its decoded
dimensions are 1024×655 (despite an initial 1024×687 description). Empty enemy and loot
lists are intentional: those classes cannot be verified from the single still. The full
bars, mini-map player point, and absence of death/restart UI are directly visible.
