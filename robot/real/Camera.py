import string

import numpy as np

from robot.Component import Component


class Camera(Component):

    def __init__(self, mask_line_file_path: string, mask_obstacle_file_path: string):
        self.mask_obstacle_file_path = mask_obstacle_file_path
        self.mask_line_file_path = mask_line_file_path

    mask_line = None

    mask_obstacles = None

    fresh_image = False

    def execute(self):
        self.fresh_image = True
        # Consume masks produced by inference process and shape them in open cv format
        self.mask_line = (np.load(self.mask_line_file_path) * 255).astype('uint8')
        self.mask_obstacles = (np.load(self.mask_obstacle_file_path) * 255).astype('uint8')

    def get_images(self):
        self.fresh_image = False
        return self.mask_line, self.mask_obstacles

    def has_new_image(self):
        return self.fresh_image
