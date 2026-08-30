# Hero Siege Vision Farm Bot — Design

## Goal

Build a Windows Python bot for Hero Siege that completes a repeatable farming loop on one procedurally generated location using only screen capture and normal keyboard/mouse input.

The MVP must run autonomously for 30 minutes and handle exploration, combat, loot collection, recovery from getting stuck, death, and restarting a run.

## Constraints

- Target platform: Windows.
- Game mode: borderless window.
- No process-memory access, DLL injection, hooks, or code injection.
- Input: WASD movement, potions on `1` and `2`, skills on `Q` and `E`, and the primary attack by holding the left mouse button.
- Loot is collected by approaching it and clicking or holding the left mouse button.
- The location layout changes every run.
- The mini-map is available, but fog of war hides unexplored areas.
- The exact farming location and character build will be configured later.
- Anti-cheat bypass and concealment are outside the project scope.

## Recommended Approach

Use a hybrid of classical computer vision and an explicit state machine:

- capture frames with `dxcam`;
- identify interface anchors, fog of war, enemies, loot, and game states with OpenCV;
- send input through the Windows `SendInput` API;
- explore the map with a frontier-based algorithm;
- keep optional detector interfaces so a YOLO model can replace individual classical detectors later.

This approach is easier to develop, inspect, and tune than end-to-end machine learning, while avoiding the initial dataset requirements of a neural detector.

## Architecture

### WindowCapture

Finds the Hero Siege window by process or title, tracks its client bounds, and captures only the game image. It reports loss of focus, window movement, and capture failure.

### AutoCalibrator

Finds stable HUD and mini-map anchors over several frames, estimates UI scale, and produces normalized regions of interest. It recalibrates after a resolution, window position, or UI-layout change.

Calibration carries a confidence score. No input is emitted until the score passes its configured threshold.

### Perception

Converts each captured frame into a structured observation:

- player health and secondary resource;
- mini-map player position;
- explored, walkable, and fog-of-war regions;
- likely enemies and loot;
- death and restart screens;
- confidence values for every detection.

Detection should initially use template matching, color masks, temporal differencing, and contour filtering. Detector interfaces must allow later replacement by trained models.

### MapExplorer

Maintains a local occupancy representation derived from the visible mini-map. It detects frontiers between explored space and fog of war, scores reachable frontiers, and selects a movement target.

Progress is measured by changes in mini-map position and newly revealed area. If movement produces no progress, the explorer changes direction and eventually abandons the current frontier.

### CombatController

Suspends exploration when an enemy is detected. It aims the cursor at a selected enemy, holds the left mouse button, and uses `Q` and `E` according to configurable cooldown timers.

The controller does not assume exact cooldown data from memory. It relies on timers and, where reliable, visible skill-state cues.

### SurvivalController

Estimates resource ratios from the HUD and presses `1` or `2` below configurable thresholds. Per-key cooldown guards prevent repeated consumption.

### LootController

After combat, selects nearby visible loot, approaches it, and clicks or holds the left mouse button. It abandons a target after a timeout to avoid blocking the farming loop.

### InputController

Emits keyboard and mouse events through Windows `SendInput`. It supports short movement pulses, bounded mouse-button holds, release-all, dry-run mode, and an emergency stop on `F12`.

### BotStateMachine

Coordinates the system through explicit states:

1. `CALIBRATING`
2. `EXPLORING`
3. `COMBAT`
4. `LOOTING`
5. `RECOVERING`
6. `DEAD`
7. `RESTARTING`
8. `PAUSED`

State transitions depend on observations and confidence thresholds. Losing the window, focus, or required HUD anchors always enters `PAUSED` and releases all input.

### Recorder and Overlay

Records selected frames, observations, state transitions, actions, and errors. A diagnostic overlay displays calibrated regions, detections, fog/frontiers, the movement target, and the current state.

## Runtime Flow

1. Locate the borderless game window.
2. Capture several input-free frames.
3. Detect HUD and mini-map anchors and calculate normalized coordinates.
4. Pause and save a diagnostic frame if calibration confidence is insufficient.
5. Segment explored space and fog of war on the mini-map.
6. Select a reachable frontier and move with bounded WASD pulses.
7. Interrupt exploration when an enemy is detected.
8. Aim, hold the primary attack, and apply configured skills.
9. Use potions when visible resource thresholds are crossed.
10. Collect nearby loot after combat.
11. Detect lack of movement or map-reveal progress and run recovery behavior.
12. On death, release all input, identify the restart control, and begin a new run.

## Configuration

Configuration is stored in a human-editable profile and includes:

- process/window identification;
- expected HUD anchor templates;
- confidence thresholds;
- movement pulse durations;
- health/resource potion thresholds;
- skill timers;
- combat and loot timeouts;
- emergency-stop key;
- recording and overlay options.

Coordinates are not stored as fixed screen pixels. They are normalized against automatically detected window and HUD geometry.

## Failure Handling

- Window, focus, capture, or calibration loss: release all input and pause.
- Stuck movement: try a short orthogonal movement, reverse, then select another frontier.
- Uncertain detection: prefer no action and record evidence.
- Lost enemy or loot target: timeout and return to exploration.
- Unexpected screen: release input, save a frame, and attempt recalibration.
- Shutdown or exception: execute release-all before process exit.

## Verification

### Offline tests

- HUD anchor detection at several resolutions and window positions.
- Health/resource ratio estimation from recorded frames.
- Fog, explored-area, and frontier segmentation.
- Enemy, loot, death, and restart-screen detection.
- State-machine transitions using scripted observations.
- Input safety rules with a fake input backend.

### Dry-run test

Run live capture and the diagnostic overlay without emitting input. Verify calibrated regions, detections, selected frontiers, and proposed actions.

### Controlled integration tests

Enable one subsystem at a time: movement, exploration, combat, potions, loot, recovery, then death/restart.

### MVP acceptance

On one selected procedurally generated location, the bot runs for 30 minutes without manual intervention and demonstrates:

- automatic calibration;
- fog-of-war exploration;
- combat using primary attack and skills;
- potion use;
- loot collection;
- recovery from at least one blocked movement case;
- completion of at least one death-and-restart cycle.

## Deferred Work

- Supporting multiple locations or character profiles.
- Training and integrating a YOLO detector.
- Inventory management, selling, storage, and town navigation.
- Long-duration unattended operation beyond the 30-minute MVP target.
