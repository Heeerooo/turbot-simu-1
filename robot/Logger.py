import os
import time

import numpy as np

from robot.Component import Component

LOG_FORMAT = {
    0: "time",
    1: "final image",
    2: "poly coef 1",
    3: "pixel line offset",
    4: "distance obstacle line",
    5: "steering",
    6: "actives rotations",
    7: "actives translations",
    8: "perspective image",
    9: "translated image",
    10: "rotated image"
}

class Logger(Component):

    def __init__(self, simulator, image_analyzer,
                 car, sequencer, log_dir,
                 time, steering_controller, image_warper, size_log_stack=10):
        self.image_warper = image_warper
        self.size_log_stack = size_log_stack
        self.steering_controller = steering_controller
        self.time = time
        self.log_dir = log_dir
        self.image_analyzer = image_analyzer
        self.sequencer = sequencer
        self.car = car
        self.simulator = simulator
        self.previous_joint_pos = 0
        self.first_pos = None
        self.log_array = []
        self.run_session = "run_" + str(time.time())
        self.increment_session = 1
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

    def execute(self):
        self.log()

    def log(self):

        self.log_array.append([time.time(),
                               self.image_analyzer.final_mask_for_display,
                               self.image_analyzer.poly_coeff_1,
                               self.image_analyzer.pixel_offset_line,
                               self.image_analyzer.distance_obstacle_line,
                               self.steering_controller.steering,
                               self.image_warper.actives_rotations,
                               self.image_warper.actives_translations,
                               self.image_warper.perspective,
                               self.image_warper.translated,
                               self.image_warper.rotated
                               ])

        if len(self.log_array) >= self.size_log_stack:
            np.savez(self.log_dir + "/" + self.run_session + "_" + "%03d" % self.increment_session, data=self.log_array)
            self.increment_session += 1
            self.log_array.clear()

        print("tacho : %s" % self.car.get_tacho())
        print("time : %fs " % self.car.get_time())

