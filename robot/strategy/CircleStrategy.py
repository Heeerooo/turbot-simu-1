import numpy as np

from circle import angle_intersection
from robot.strategy.Strategy import Strategy


class CircleStrategy(Strategy):

    def __init__(self,
                 image_analyzer,
                 p_coef,
                 i_coef,
                 d_coef,
                 avoidance_speed,
                 nominal_speed,
                 obstacle_offset):
        self.i_coef = i_coef
        self.nominal_speed = nominal_speed
        self.avoidance_speed = avoidance_speed
        self.obstacle_offset = obstacle_offset
        self.p_coef = p_coef
        self.image_analyzer = image_analyzer
        self.d_coef = d_coef

    cumul_error = 0
    current_obstacle_offset = 0
    previous_error_angle = 0
    previous_raw_error = 0
    side_avoidance = None
    MAX_CUMUL_ERROR = 1
    AVOID_OFFSET_INCREASE_STEP = 0.05
    log_error_angle = 0
    log_error_angle_2 = 0
    log_error_angle_3 = 0
    log_error_angle_4 = 0

    def compute_steering(self):
        self.image_analyzer.analyze()

        # poly_2_coefs = self.image_analyzer.poly_2_coefs
        # if poly_2_coefs is None:
        #     return None
        #
        # error_angle = angle_intersection(poly_2_coefs[0],poly_2_coefs[1],poly_2_coefs[2],
        #                                  self.image_analyzer.circle_poly2_intersect_radius,
        #                                  self.image_analyzer.final_image_height,
        #                                  self.image_analyzer.final_image_width)

        # FIXME test purposes . poly1 instead of poly2
        print("WARNING test purposes poly1 instead of poly2")
        poly_2_coefs = self.image_analyzer.poly_1_coefs
        if poly_2_coefs is None:
            error_angle = None
        else:
            error_angle = angle_intersection(0, poly_2_coefs[0], poly_2_coefs[1],
                                             self.image_analyzer.circle_poly2_intersect_radius,
                                             self.image_analyzer.final_image_height,
                                             self.image_analyzer.final_image_width)

        self.log_error_angle = error_angle

        if error_angle is None:
            error_angle = np.sign(self.previous_raw_error) * 0.5
        else:

            self.previous_raw_error = error_angle

        error_angle += self.compute_obstacle_error()

        self.log_error_angle_2 = error_angle
        error_angle = (np.sign(error_angle) * abs(error_angle) ** 1.2) * 1.3
        self.log_error_angle_3 = error_angle

        self.cumul_error = self.cumul_error * 0.9
        self.cumul_error += error_angle
        self.cumul_error = np.clip(self.cumul_error, -self.MAX_CUMUL_ERROR, self.MAX_CUMUL_ERROR)

        ecart_error = error_angle - self.previous_error_angle
        self.log_error_angle_4 = error_angle

        self.previous_error_angle = error_angle
        return self.p_coef * error_angle + self.cumul_error * self.i_coef + ecart_error * self.d_coef

    def compute_obstacle_error(self):
        # Compute side avoidance if obstacle not in lock zone or no side computed yet
        if not self.image_analyzer.obstacle_in_lock_zone or self.side_avoidance is None:
            self.side_avoidance = self.image_analyzer.obstacle_poly2_side

        if self.side_avoidance is not None and self.image_analyzer.obstacle_in_avoidance_zone \
                and obstacle_line_distance_small_enough(self.image_analyzer.distance_obstacle_line,
                                                        self.image_analyzer.side_avoidance):

            # obstacle detected, increase progressively offset
            self.current_obstacle_offset += self.AVOID_OFFSET_INCREASE_STEP * self.side_avoidance
            self.current_obstacle_offset = np.clip(self.current_obstacle_offset, - self.obstacle_offset,
                                                   self.obstacle_offset)
        else:
            # No obstacle, obstacle offset returns to zero
            if self.current_obstacle_offset > 0:
                self.current_obstacle_offset -= self.AVOID_OFFSET_INCREASE_STEP
                self.current_obstacle_offset = max(self.current_obstacle_offset, 0)
            else:
                self.current_obstacle_offset += self.AVOID_OFFSET_INCREASE_STEP
                self.current_obstacle_offset = min(self.current_obstacle_offset, 0)
        print(self.current_obstacle_offset)
        return self.current_obstacle_offset

    def compute_speed(self):
        if self.image_analyzer.obstacle_in_slow_zone \
                and obstacle_line_distance_small_enough(self.image_analyzer.distance_obstacle_line,
                                                        self.image_analyzer.obstacle_poly2_side):
            return self.avoidance_speed
        else:
            return self.nominal_speed


WIDTH_HALF_CORRIDOR = 50


def obstacle_line_distance_small_enough(distance_obstacle_line, side_avoidance):
    return distance_obstacle_line is not None \
           and ((distance_obstacle_line < WIDTH_HALF_CORRIDOR and side_avoidance < 0)
                or (distance_obstacle_line > -WIDTH_HALF_CORRIDOR and side_avoidance > 0))
