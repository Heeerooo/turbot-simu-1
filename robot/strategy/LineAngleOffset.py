import numpy as np

from robot.strategy.Strategy import Strategy

COEF_P_LINE_ANGLE = 50
COEF_P_LINE_OFFSET = 0.2
RATION_ANGLE_OFFSET = 300
GAIN = 60
# Width in pixels in which we avoid obstacles (if obstacle is not in corridor, we do not avoid it)
WIDTH_HALF_CORRIDOR = 50
# Offset to take into account the width of the robot when avoiding obstacles
ROBOT_WIDTH_AVOIDANCE = 40
# Quand l'obstacle est sur la ligne, on s'ecarte un peu plus, avec une marge supplémentaire
COEFF_AVOIDANCE_SAME_SIDE = 1.5
# Quand l'obstacle n'est pas sur la ligne, on s'ecarte un peu moins que la largeur qui separe l'obstacle de la ligne
COEFF_AVOIDANCE_OTHER_SIDE = 0.5


def should_compute_obstacle_offset(distance_obstacle_line):
    return distance_obstacle_line is None or abs(distance_obstacle_line) > WIDTH_HALF_CORRIDOR


class LineAngleOffset(Strategy):

    def __init__(self, image_analyzer, additional_offset):
        self.additional_offset = additional_offset
        self.image_analyzer = image_analyzer

    def compute_steering(self):
        self.image_analyzer.analyze()
        coeff_poly_1_line = self.image_analyzer.poly_coeff_1
        distance_obstacle_line = self.image_analyzer.distance_obstacle_line

        obstacle_avoidance_additional_offset = 0 \
            if should_compute_obstacle_offset(distance_obstacle_line) \
            else self.compute_obstacle_offset(distance_obstacle_line)

        line_offset = self.image_analyzer.pixel_offset_line

        if coeff_poly_1_line is not None and line_offset is not None:
            angle_line = -np.arctan(coeff_poly_1_line[0])
            return GAIN * angle_line + GAIN / RATION_ANGLE_OFFSET * (
                    line_offset + self.additional_offset + obstacle_avoidance_additional_offset)
        else:
            return None

    def compute_obstacle_offset(self, distance_obstacle_line):
        side_avoidance = self.image_analyzer.side_avoidance

        coeff_avoidance = COEFF_AVOIDANCE_SAME_SIDE \
            if (np.sign(distance_obstacle_line) == np.sign(side_avoidance)) \
            else COEFF_AVOIDANCE_OTHER_SIDE

        return (coeff_avoidance * distance_obstacle_line) + (ROBOT_WIDTH_AVOIDANCE * side_avoidance)