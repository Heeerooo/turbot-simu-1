# coding=utf-8
import cv2
import numpy as np


class ImageAnalyzer:
    # Constants for cleaning inference results. IMPORTANT to recalibrate this on real conditions track.
    MIN_AREA_RATIO = 0.35  # if area / area_of_biggest_contour is less than this ratio, contour is bad
    MIN_AREA_TO_KEEP = 100.  # if max_area if less than this, reject all image
    MIN_THRESHOLD_CONTOUR = 10
    MAX_VALUE_CONTOUR = 255

    LINE_THRESHOLD = 0.10
    BOTTOM_OBSTACLE_WINDOW_HEIGHT = 5
    LINE_WINDOW_HEIGHT_AT_OBSTACLE = 5

    poly_1_coefs = None
    poly_2_coefs = None
    pixel_offset_line = None
    distance_obstacle_line = None
    side_avoidance = None
    offset_baseline_height = None
    lock_zone_radius = None
    avoidance_zone_radius = None
    obstacle_in_lock_zone = False
    obstacle_in_avoidance_zone = False
    final_mask_for_display = None
    circle_poly2_intersect_radius = None
    process_obstacles = True

    def __init__(self, car, image_warper, show_and_wait=False, log=True):
        self.log = log
        self.car = car
        self.image_warper = image_warper
        self.show_and_wait = show_and_wait
        self.clip_length = 0
        self.final_image_height = image_warper.warped_height
        self.final_image_width = image_warper.warped_width

    def reset(self):
        self.poly_2_coefs = None
        self.poly_1_coefs = None
        self.circle_poly2_intersect_radius = None
        self.obstacle_in_lock_zone = False
        self.obstacle_in_avoidance_zone = False
        self.lock_zone_radius = None
        self.avoidance_zone_radius = None
        self.offset_baseline_height = None

    def analyze(self):
        if not self.car.has_new_image():
            return

        mask_line, mask_obstacles = self.car.get_images()
        if mask_line is not None and mask_obstacles is not None:
            mask_line = self.clean_mask_line(mask_line)
            mask_obstacles = clean_mask_obstacle(mask_obstacles)
            mask_line = self.image_warper.warp(mask_line, "line")
            self.compute_lines(mask_line)

            if self.process_obstacles:
                mask_obstacles = self.image_warper.warp(mask_obstacles, "obstacle")
                mask_line = self.clip_image(mask_line)
                self.compute_obstacles(mask_line, mask_obstacles)

            if self.show_and_wait or self.log:
                self.draw_log_image(mask_line, mask_obstacles)

                if self.show_and_wait:
                    cv2.imshow('merged final', self.final_mask_for_display)
                    cv2.waitKey(0)

    def compute_lines(self, mask_line):
        self.poly_1_interpol(mask_line)
        self.poly_2_interpol(mask_line)
        self.compute_robot_horizontal_offset_from_poly1()

    def compute_obstacles(self, mask_line, mask_obstacles):
        self.compute_obstacle_line_position(mask_line, mask_obstacles)
        self.compute_obstacle_lock_zone(mask_obstacles)
        self.compute_obstacle_avoidance_zone(mask_obstacles)

    def draw_log_image(self, mask_line, mask_obstacles):
        # Display final mask for debug
        self.final_mask_for_display = np.zeros((mask_line.shape[0], mask_line.shape[1], 3))
        self.final_mask_for_display[..., 2] = mask_line
        if self.process_obstacles:
            self.final_mask_for_display[..., 1] = mask_obstacles
        if self.offset_baseline_height is not None:
            self.draw_line_offset_line()
        if self.poly_1_coefs is not None:
            draw_interpol_poly1(self.final_mask_for_display, self.poly_1_coefs)
        if self.poly_2_coefs is not None:
            draw_interpol_poly2(self.final_mask_for_display, self.poly_2_coefs)
        if self.circle_poly2_intersect_radius is not None:
            draw_circle(self.final_mask_for_display, self.circle_poly2_intersect_radius, "white")
        if self.lock_zone_radius is not None:
            draw_circle(self.final_mask_for_display, self.lock_zone_radius, "green")
        if self.avoidance_zone_radius is not None:
            draw_circle(self.final_mask_for_display, self.avoidance_zone_radius, "red")

    def clip_image(self, image):
        image[:self.clip_length, :] = 0
        return image

    def clean_mask_line(self, image):

        # Get contours
        _, thresh = cv2.threshold(image, self.MIN_THRESHOLD_CONTOUR, self.MAX_VALUE_CONTOUR, 0)
        result = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Open cv version compatibility issue
        if len(result) == 2:
            contours = result[0]
        else:
            contours = result[1]
        if len(contours) == 0:
            # No contours, no need to remove anything
            return image
        else:
            # Compute areas of contours
            areas = []
            for cnt in contours:
                areas.append(cv2.contourArea(cnt))

            # Find bad contours
            bad_indexes = []
            max_area = np.max(areas)
            for i, cnt in enumerate(contours):
                # Check if area if too small, in this case, reject contour
                area = areas[i]
                if (area / max_area) < self.MIN_AREA_RATIO:
                    bad_indexes.append(i)
                else:
                    # Compute sum of pixels on area
                    mask = np.ones(image.shape[:2], dtype="uint8") * 255
                    cv2.drawContours(mask, [cnt], -1, 0, -1)
                    # extractMat = np.multiply(image, mask > 0)
                    # sumContour = np.multiply(extractMat, mask > 0).sum()
                    # print("Sum contour {:d}: {:.1f}".format(i, sumContour))

            bad_contours = [contours[i] for i in bad_indexes]

            # Create mask with contours to remove
            mask = np.ones(image.shape[:2], dtype="uint8") * 255
            cv2.drawContours(mask, bad_contours, -1, 0, -1)

            # Apply mask on matrix
            result = np.multiply(image, mask > 0)

            return result

    def poly_1_interpol(self, image):
        self.poly_1_coefs = self.poly_interpol(image, 1)

    def poly_2_interpol(self, image):
        self.poly_2_coefs = self.poly_interpol(image, 2)

    def poly_interpol(self, image, degree):
        nonzeros_indexes = np.nonzero(image > self.LINE_THRESHOLD)
        y = nonzeros_indexes[0]
        x = nonzeros_indexes[1]
        if len(x) < 2:
            return None
        else:
            return np.polyfit(y, x, degree)

    def draw_line_offset_line(self):
        lineY = (self.final_image_height - self.offset_baseline_height)
        shape = self.final_mask_for_display.shape
        lineX = np.arange(0, shape[1] - 1)
        self.final_mask_for_display[lineY, lineX, :] = 1

    def compute_robot_horizontal_offset_from_poly1(self):
        if self.offset_baseline_height is None:
            return

        if self.poly_1_coefs is None:
            self.pixel_offset_line = None
        else:
            self.pixel_offset_line = (self.poly_1_coefs[0] * (
                    self.final_image_height - self.offset_baseline_height)
                                      + self.poly_1_coefs[1]) - (self.final_image_width / 2)

    def compute_obstacle_line_position(self, mask_line, mask_obstacles):

        obstacle_pixels_y, obstacle_pixels_x, = np.nonzero(mask_obstacles)
        line_pixels_y, line_pixels_x = np.nonzero(mask_line)

        if len(obstacle_pixels_y) == 0 or len(obstacle_pixels_x) == 0 \
                or len(line_pixels_y) == 0 or len(line_pixels_x) == 0:
            self.distance_obstacle_line = None
            self.side_avoidance = None
            return

        lowest_obstacle_y = np.max(obstacle_pixels_y)
        lowest_obstacle_pixels_x = obstacle_pixels_x[
            np.where(obstacle_pixels_y >= lowest_obstacle_y - self.BOTTOM_OBSTACLE_WINDOW_HEIGHT)]

        if len(line_pixels_x) == 0:
            self.distance_obstacle_line = None
            self.side_avoidance = None
            return

        low_left_obstacle_x = np.min(lowest_obstacle_pixels_x)
        low_left_obstacle_y = lowest_obstacle_y
        low_right_obstacle_x = np.max(lowest_obstacle_pixels_x)
        low_right_obstacle_y = lowest_obstacle_y

        # Find points on the line that are the closest to the bottom left and bottom right of the obstacle
        idx_line_closest_to_left_obstacle = np.argmin(
            np.square(low_left_obstacle_x - line_pixels_x) + np.square(low_left_obstacle_y - line_pixels_y))
        idx_line_closest_to_right_obstacle = np.argmin(
            np.square(low_right_obstacle_x - line_pixels_x) + np.square(low_right_obstacle_y - line_pixels_y))
        x_line_closest_left = line_pixels_x[idx_line_closest_to_left_obstacle]
        y_line_closest_left = line_pixels_y[idx_line_closest_to_left_obstacle]
        x_line_closest_right = line_pixels_x[idx_line_closest_to_right_obstacle]
        y_line_closest_right = line_pixels_y[idx_line_closest_to_right_obstacle]

        # Compute distances
        distance_left_obstacle = np.sqrt(
            (x_line_closest_left - low_left_obstacle_x) ** 2 + (y_line_closest_left - low_left_obstacle_y) ** 2)
        distance_right_obstacle = np.sqrt(
            (x_line_closest_right - low_right_obstacle_x) ** 2 + (y_line_closest_right - low_right_obstacle_y) ** 2)

        # Find position according to line
        position_left_obstacle = np.sign(low_left_obstacle_x - x_line_closest_left)
        position_right_obstacle = np.sign(low_right_obstacle_x - x_line_closest_right)

        # Transform distance to signed distance
        distance_left_obstacle *= position_left_obstacle
        distance_right_obstacle *= position_right_obstacle

        self.side_avoidance = 1 if (abs(distance_left_obstacle) > abs(distance_right_obstacle)) else -1
        self.distance_obstacle_line = min(distance_left_obstacle, distance_right_obstacle, key=abs)

    def set_clip_length(self, clip_length):
        if clip_length < 0 or clip_length > self.final_image_height:
            raise Exception("Clip lenght out of final image bounds")
        self.clip_length = clip_length

    def set_offset_baseline_height(self, offset_line_height):
        if offset_line_height < 0 or offset_line_height > self.final_image_height:
            raise Exception("offset line height out of final image bounds")
        self.offset_baseline_height = offset_line_height

    def set_lock_zone_radius(self, lock_zone_radius):
        if lock_zone_radius < 0 or lock_zone_radius > self.final_image_height:
            raise Exception("lock zone lenght out of final image bounds")
        self.lock_zone_radius = lock_zone_radius

    def set_process_obstacle(self, process_obstacles):
        if not isinstance(process_obstacles, bool):
            raise Exception("process obstacle value must be bool")
        self.process_obstacles = process_obstacles

    def set_avoidance_zone_radius(self, avoidance_zone_radius):
        if avoidance_zone_radius < 0 or avoidance_zone_radius > self.final_image_height:
            raise Exception("avoidance zone lenght out of final image bounds")
        self.avoidance_zone_radius = avoidance_zone_radius

    def compute_obstacle_avoidance_zone(self, mask_obstacles):
        self.obstacle_in_avoidance_zone = compute_bottom_center_circle_zone_presence(mask_obstacles,
                                                                                     self.avoidance_zone_radius)

    def compute_obstacle_lock_zone(self, mask_obstacles):
        self.obstacle_in_lock_zone = compute_bottom_center_circle_zone_presence(mask_obstacles,
                                                                                self.lock_zone_radius)


def compute_bottom_center_circle_zone_presence(image, radius):
    if radius is None or image is None:
        return False

    shape = image.shape
    circle_zone_image = np.zeros(image.shape)
    cv2.circle(circle_zone_image, (round(shape[1] / 2), shape[0]), radius, 255, -1)
    return True in np.logical_and(image, circle_zone_image)


def draw_interpol_poly1(image, poly_coefs):
    def poly1(x):
        return poly_coefs[0] * x + poly_coefs[1]

    return draw_interpol(image, poly1)


def draw_interpol_poly2(image, poly_coefs):
    def poly2(x):
        return poly_coefs[0] * x * x + poly_coefs[1] * x + poly_coefs[2]

    return draw_interpol(image, poly2)


def draw_interpol(image, interpol_function):
    shape = image.shape
    xall = np.arange(0, shape[0] - 1)
    ypoly = interpol_function(xall).astype(int)
    ypoly = np.clip(ypoly, 0, shape[1] - 2)
    image[xall, ypoly, :] = 0
    image[xall, ypoly + 1, :] = 1
    return image


def draw_circle(image, r, color):
    shape = image.shape
    xmax = shape[0]
    ymax = shape[1]
    # Plot circle
    for y in range(ymax):
        t = r ** 2 - (y - ymax / 2) ** 2
        if t > 0:
            x = xmax - int(np.sqrt(t))
            if color is "green":
                image[x, y, 1] = 1
                image[x - 1, y, 1] = 1
            elif color is "red":
                image[x, y, 2] = 1
                image[x - 1, y, 2] = 1
            else:
                image[x, y, :] = 1
                image[x - 1, y, :] = 1
    return image


def clean_mask_obstacle(mask_obstacles):
    return np.clip(mask_obstacles, 0, 1) * 255
