import numpy as np

from robot.strategy.Strategy import Strategy

MAX_CUMUL = 1000


class TurnOffsetStrategy(Strategy):

    def __init__(self, image_analyzer, target_steering, p_coef, i_coef):
        self.cumul_error_offset = 0
        self.i_coef = i_coef
        self.p_coef = p_coef
        self.target_steering = target_steering
        self.image_analyzer = image_analyzer

    def compute_speed(self):
        return None

    def compute_steering(self):
        steering_command = self.target_steering

        self.image_analyzer.analyze()
        error_offset = self.image_analyzer.pixel_offset_poly1

        if error_offset is not None:
            steering_command += error_offset * self.p_coef
            self.cumul_error_offset = self.cumul_error_offset * 0.9
            self.cumul_error_offset += error_offset
            self.cumul_error_offset = np.clip(self.cumul_error_offset, -MAX_CUMUL, MAX_CUMUL)

        steering_command += self.cumul_error_offset * self.i_coef

        return steering_command
