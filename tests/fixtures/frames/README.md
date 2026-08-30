# Recorded frame fixtures

Task 5 uses generated images so detector expectations remain deterministic. Private
recorded fixtures may be added as paired `<case>.png` and `<case>.yaml` files. Recorded
tests skip only when this directory contains no such pairs.

Each sidecar YAML file must contain:

- `frame_size`: `{width, height}`
- `hud_rectangles`: named `{x, y, width, height}` entries for `health`, `resource`,
  `minimap`, `gameplay`, and `screen_state`
- `player_map_point`, `bar_ratios`, `enemy_boxes`, `loot_boxes`, `death`, and
  `restart_visible`: each is an annotation object with exactly:
  - `status`: `verified` or `unknown`
  - `value`: the field value when verified, otherwise `null`

Verified values use these shapes:

- `player_map_point`: normalized `{x, y}`, or `null` only when absence was verified
- `bar_ratios`: `health` and `resource`, each a ratio in `[0, 1]` or `null`
- `enemy_boxes` and `loot_boxes`: lists of `{x, y, width, height}` frame-pixel
  rectangles; an empty list means verified absence, not unknown
- `death` and `restart_visible`: boolean

Do not infer labels from a single screenshot. Add a case only after its boxes and state
flags have been manually verified, and keep production thresholds in configuration.

`highland_graveyard_1024x655` is the first user-supplied real fixture. Its decoded
dimensions are 1024×655 (despite an initial 1024×687 description). Enemy, loot, and
player-map annotations are `unknown`: one still does not establish those semantic
identities. The full bars and absence of death/restart UI are directly visible.
