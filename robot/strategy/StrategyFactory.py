from robot.strategy.CapOffsetStrategy import CapOffsetStrategy
from robot.strategy.CapStandardStrategy import CapStandardStrategy
from robot.strategy.CircleStrategy import CircleStrategy
from robot.strategy.ImageStraitLineStrategy import ImageStraitLineStrategy
from robot.strategy.LineAngleOffset import LineAngleOffset


class StrategyFactory:

    def __init__(self, car, image_analyzer):
        self.car = car
        self.image_analyzer = image_analyzer

    def create_lao(self, additional_offset=0, angle_coef=None, offset_coef=None):
        if angle_coef is not None and offset_coef is not None:
            return LineAngleOffset(self.image_analyzer, additional_offset, angle_coef, offset_coef)
        else:
            return LineAngleOffset(self.image_analyzer, additional_offset)

    def create_image_straight_line(self, cap_target, integral_enable=False):
        return ImageStraitLineStrategy(self.image_analyzer, cap_target, integral_enable)

    def create_cap_standard(self, cap_target, vitesse):
        return CapStandardStrategy(self.car, cap_target, vitesse)

    def create_circle(self, p_coef, i_coef, nominal_speed, avoidance_speed, obstacle_offset=0):
        return CircleStrategy(self.image_analyzer, p_coef, i_coef, avoidance_speed, nominal_speed, obstacle_offset)

    def create_cap_offset(self, cap_target, vitesse, p_correction_coef, i_correction_coef):
        return CapOffsetStrategy(self.car, self.image_analyzer, cap_target, vitesse, p_correction_coef,
                                 i_correction_coef)
