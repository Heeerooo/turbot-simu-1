from robot.strategy.Strategy import Strategy


class TurnOffsetStrategy(Strategy):

    def __init__(self, image_analyzer, target_steering, p_coef):
        self.p_coef = p_coef
        self.target_steering = target_steering
        self.image_analyzer = image_analyzer

    def compute_speed(self):
        return None

    def compute_steering(self):
        self.image_analyzer.analyze()
        line_offset = self.image_analyzer.pixel_offset_poly1

        steering_command = self.target_steering
        if line_offset is not None:
            steering_command += line_offset * self.p_coef

        print("steering command", steering_command)
        return steering_command
