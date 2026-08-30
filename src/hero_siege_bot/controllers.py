import math

from hero_siege_bot.config import CombatConfig, SurvivalConfig
from hero_siege_bot.domain import Action, Detection, Observation

_KEY_TAP_DURATION_S = 0.05
_COMBAT_SKILL_KEYS = ("Q", "E")
COMBAT_TARGET_MATCH_RADIUS_NORMALIZED = 0.08


def _best_detection(
    detections: tuple[Detection, ...],
    confidence_threshold: float,
) -> Detection | None:
    candidates = (
        detection
        for detection in detections
        if detection.confidence >= confidence_threshold
    )
    return max(candidates, key=lambda detection: detection.confidence, default=None)


class CombatController:
    def __init__(self, config: CombatConfig) -> None:
        if set(config.skill_cooldowns_s) != set(_COMBAT_SKILL_KEYS):
            raise ValueError("combat skills must contain exactly Q/E")
        self._config = config
        self._last_skill_use: dict[str, float] = {}
        self._started_at: float | None = None
        self._active_target: Detection | None = None
        self.abandoned = False
        self.abandoned_target: Detection | None = None

    def reset_abandonment(self) -> None:
        """Start a fresh encounter after runtime observes an enemy-free frame."""
        self._started_at = None
        self._active_target = None
        self.abandoned = False
        self.abandoned_target = None

    def actions(self, observation: Observation, now: float) -> tuple[Action, ...]:
        target = _best_detection(
            observation.enemies,
            self._config.detection_confidence,
        )
        if target is None:
            self.reset_abandonment()
            return ()
        if (
            self._started_at is None
            or self._active_target is None
            or self._target_distance(target, self._active_target)
            > COMBAT_TARGET_MATCH_RADIUS_NORMALIZED
        ):
            self._started_at = now
            self.abandoned = False
            self.abandoned_target = None
        self._active_target = target
        if now - self._started_at >= self._config.combat_timeout_s:
            self.abandoned = True
            self.abandoned_target = target
            return ()
        self.abandoned = False
        self.abandoned_target = None

        actions = [
            Action(kind="mouse_move", target=target.center),
            Action(
                kind="mouse_hold",
                key="left",
                duration_s=self._config.attack_hold_s,
            ),
        ]
        for key in _COMBAT_SKILL_KEYS:
            cooldown_s = self._config.skill_cooldowns_s[key]
            last_used = self._last_skill_use.get(key)
            if last_used is None or now - last_used >= cooldown_s:
                actions.append(
                    Action(
                        kind="key_hold",
                        key=key,
                        duration_s=_KEY_TAP_DURATION_S,
                    )
                )
                self._last_skill_use[key] = now
        return tuple(actions)

    @staticmethod
    def _target_distance(first: Detection, second: Detection) -> float:
        return math.hypot(
            first.center.x - second.center.x,
            first.center.y - second.center.y,
        )


class SurvivalController:
    def __init__(self, config: SurvivalConfig) -> None:
        self._config = config
        self._last_potion_use: dict[str, float] = {}

    def actions(self, observation: Observation, now: float) -> tuple[Action, ...]:
        actions: list[Action] = []
        if (
            observation.health_ratio is not None
            and observation.health_ratio < self._config.health_threshold
            and self._is_ready("1", now)
        ):
            actions.append(self._potion_action("1", now))
        if (
            observation.resource_ratio is not None
            and observation.resource_ratio < self._config.resource_threshold
            and self._is_ready("2", now)
        ):
            actions.append(self._potion_action("2", now))
        return tuple(actions)

    def _is_ready(self, key: str, now: float) -> bool:
        last_used = self._last_potion_use.get(key)
        return (
            last_used is None
            or now - last_used >= self._config.potion_cooldown_s
        )

    def _potion_action(self, key: str, now: float) -> Action:
        self._last_potion_use[key] = now
        return Action(kind="key_hold", key=key, duration_s=_KEY_TAP_DURATION_S)


class LootController:
    def __init__(self, config: CombatConfig) -> None:
        self._config = config
        self._started_at: float | None = None
        self.abandoned = False

    def actions(self, observation: Observation, now: float) -> tuple[Action, ...]:
        target = _best_detection(
            observation.loot,
            self._config.detection_confidence,
        )
        if target is None:
            self._started_at = None
            self.abandoned = False
            return ()
        if self._started_at is None:
            self._started_at = now
        if now - self._started_at >= self._config.loot_timeout_s:
            self.abandoned = True
            return ()
        self.abandoned = False
        return (
            Action(kind="mouse_move", target=target.center),
            Action(
                kind="mouse_hold",
                key="left",
                duration_s=self._config.attack_hold_s,
            ),
        )
