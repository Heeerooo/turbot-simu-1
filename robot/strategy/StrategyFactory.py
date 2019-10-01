from robot.strategy.CapOffsetStrategy import CapOffsetStrategy
from robot.strategy.CapStandardStrategy import CapStandardStrategy
from robot.strategy.CircleStrategy import CircleStrategy
from robot.strategy.ImageStraitLineStrategy import ImageStraitLineStrategy
from robot.strategy.LineAngleOffset import LineAngleOffset
from robot.strategy.TurnOffsetStrategy import TurnOffsetStrategy


class StrategyFactory:

    def __init__(self, car, image_analyzer, logger):
        self.car = car
        self.image_analyzer = image_analyzer
        self.logger = logger

    def create_lao(self, additional_offset=0, angle_coef=None, offset_coef=None):
        if angle_coef is not None and offset_coef is not None:
            return LineAngleOffset(self.image_analyzer, additional_offset, angle_coef, offset_coef)
        else:
            return LineAngleOffset(self.image_analyzer, additional_offset)

    def create_image_straight_line(self, cap_target, integral_enable=False):
        return ImageStraitLineStrategy(self.image_analyzer, cap_target, integral_enable)

    def create_cap_standard(self, cap_target, vitesse):
        return CapStandardStrategy(self.car, cap_target, vitesse)

    def create_circle(self, p_coef, i_coef, d_coef, nominal_speed, avoidance_speed, obstacle_offset=0):
        strategy = CircleStrategy(self.image_analyzer, p_coef, i_coef, d_coef, avoidance_speed, nominal_speed, obstacle_offset)
        self.logger.strategy = strategy
        return strategy

    def create_cap_offset(self, cap_target, vitesse, p_correction_coef, i_correction_coef):
        return CapOffsetStrategy(self.car, self.image_analyzer, cap_target, vitesse, p_correction_coef,
                                 i_correction_coef)

    def create_turn_offset(self, steering_target, p_coef, i_coef):
        return TurnOffsetStrategy(self.image_analyzer, steering_target, p_coef, i_coef)
