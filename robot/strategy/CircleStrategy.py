from circle import angle_intersection

from robot.strategy.Strategy import Strategy


class CircleStrategy(Strategy):

    def __init__(self, image_analyzer, p_coef, circle_radius=150, obstacle_offset=0.2):
        self.obstacle_offset = obstacle_offset
        self.circle_radius = circle_radius
        self.p_coef = p_coef
        self.image_analyzer = image_analyzer

    previous_error_angle = 0
    side_avoidance = None

    def compute_steering(self):
        self.image_analyzer.circle_poly2_intersect_radius = self.circle_radius
        self.image_analyzer.analyze()

        poly_2_coefs = self.image_analyzer.poly_2_coefs
        if poly_2_coefs is None:
            return None

        error_angle = angle_intersection(*poly_2_coefs,
                                         self.circle_radius,
                                         self.image_analyzer.final_image_height,
                                         self.image_analyzer.final_image_width)

        if error_angle is None:
            return self.p_coef * self.previous_error_angle
        else:

            if not self.image_analyzer.obstacle_in_lock_zone:
                self.side_avoidance = self.image_analyzer.side_avoidance

            if self.side_avoidance is not None\
                    and should_compute_obstacle_offset(self.image_analyzer.distance_obstacle_line):
                error_angle += self.obstacle_offset * self.side_avoidance

            self.previous_error_angle = error_angle
            return self.p_coef * error_angle


WIDTH_HALF_CORRIDOR = 50


def should_compute_obstacle_offset(distance_obstacle_line):
    return distance_obstacle_line is not None or abs(distance_obstacle_line) > WIDTH_HALF_CORRIDOR
