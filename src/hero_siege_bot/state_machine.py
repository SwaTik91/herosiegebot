from hero_siege_bot.domain import BotState, Detection, Observation


class BotStateMachine:
    def __init__(
        self,
        initial: BotState = BotState.CALIBRATING,
        *,
        calibration_confidence: float = 0.9,
        detection_confidence: float = 0.7,
    ) -> None:
        self.state = initial
        self._calibration_confidence = calibration_confidence
        self._detection_confidence = detection_confidence

    def update(self, observation: Observation) -> BotState:
        self.state = self._transition(observation)
        return self.state

    def _transition(self, observation: Observation) -> BotState:
        if not observation.focused:
            return BotState.PAUSED
        if not observation.calibrated:
            return BotState.CALIBRATING
        if observation.dead:
            return BotState.DEAD

        enemies_visible = self._has_confident_detection(observation.enemies)
        loot_visible = self._has_confident_detection(observation.loot)

        if self.state is BotState.CALIBRATING:
            if observation.calibration_confidence >= self._calibration_confidence:
                return BotState.EXPLORING
            return BotState.CALIBRATING

        if self.state is BotState.EXPLORING:
            if enemies_visible:
                return BotState.COMBAT
            if observation.movement_progress <= 0.0:
                return BotState.RECOVERING
            return BotState.EXPLORING

        if self.state is BotState.COMBAT:
            if enemies_visible:
                return BotState.COMBAT
            return BotState.LOOTING

        if self.state is BotState.LOOTING:
            if enemies_visible:
                return BotState.COMBAT
            if loot_visible:
                return BotState.LOOTING
            return BotState.EXPLORING

        if self.state is BotState.RECOVERING:
            if enemies_visible:
                return BotState.COMBAT
            if observation.movement_progress > 0.0:
                return BotState.EXPLORING
            return BotState.RECOVERING

        if self.state is BotState.DEAD:
            if observation.restart_target is not None:
                return BotState.RESTARTING
            return BotState.DEAD

        if self.state is BotState.RESTARTING:
            return BotState.CALIBRATING

        return BotState.CALIBRATING

    def _has_confident_detection(self, detections: tuple[Detection, ...]) -> bool:
        return any(
            detection.confidence >= self._detection_confidence
            for detection in detections
        )
