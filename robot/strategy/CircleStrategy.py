from circle import angle_intersection

from robot.strategy.Strategy import Strategy


class CircleStrategy(Strategy):

    def __init__(self, image_analyzer, p_coef, circle_radius):
        self.circle_radius = circle_radius
        self.p_coef = p_coef
        self.image_analyzer = image_analyzer

    previous_error_angle = 0

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
            self.previous_error_angle = error_angle
            return self.p_coef * error_angle
