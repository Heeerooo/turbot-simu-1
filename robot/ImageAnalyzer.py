# coding=utf-8
import cv2
import numpy as np

from robot.Config import CAMERA_DELAY


class ImageAnalyzer:
    # Constantes
    MODEL_FILENAME = 'deep_learning_models/craie_quarter_filters_6.h5'
    DELAY_EXECUTION = 0.07
    LINE_THRESHOLD = 0.10
    EVITEMENT_OFFSET = 0.30
    X_INFERENCE_POINT_1 = 100  # Point depuis le haut de l'image pris pour calculer l'ecart par rapport a la ligne
    X_INFERENCE_POINT_2 = 150  # Point depuis le haut de l'image pris pour calculer l'ecart par rapport a la ligne
    WIDTH = 320
    HEIGHT = 240
    SAVE_TO_FILENAME = "/tmp_ram/imageAnalysisResult.json"
    LOG_EVERY_N_IMAGES = 20  # Loggue les images toutes les N
    LOG_BUFFER_SIZE = 10  # Taille du buffer (nombre d'images enregistrees dans un fichier)

    # Constants for cleaning inference results. IMPORTANT to recalibrate this on real conditions track.
    MIN_AREA_RATIO = 0.35  # if area / area_of_biggest_contour is less than this ratio, contour is bad
    MIN_AREA_TO_KEEP = 100.  # if max_area if less than this, reject all image
    MIN_THRESHOLD_CONTOUR = 10
    MAX_VALUE_CONTOUR = 255

    # Param�tres de classe
    position_consigne = 0.0
    logTimestampMemory = None
    logImageMemory = None
    logMask0Memory = None
    logMask1Memory = None
    log_counter = 0
    new_image_arrived = True

    # Param�tres pour mesurer les temps d'ex�cution
    time_start = 0.
    max_time = 0.
    first_time = True
    last_execution_time = 0

    def __init__(self, simulator, cam_handle):
        self.cam_handle = cam_handle
        self.simulator = simulator

        # Initialize image_ligne with empty image
        self.image_ligne = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.uint8)

    def execute(self):
        resolution, byte_array_image_string = self.simulator.get_gray_image(self.cam_handle, CAMERA_DELAY)
        if resolution is None and byte_array_image_string is None:
            return
        mask0 = self.convert_image_to_numpy(byte_array_image_string, resolution)
        mask0 = self.clean_mask(mask0)
        self.image_ligne = mask0

    def get_image_ligne(self):
        return self.image_ligne

    def convert_image_to_numpy(self, byte_array_image_string, resolution):
        return np.flipud(np.fromstring(byte_array_image_string, dtype=np.uint8).reshape(resolution[::-1]))

    def clean_mask(self, image):

        # Scale to openCV format: transform [0.,1.] to [0,255]
        int_mat = (image * 255).astype(np.uint8)

        # Get contours
        _, thresh = cv2.threshold(int_mat, self.MIN_THRESHOLD_CONTOUR, self.MAX_VALUE_CONTOUR, 0)
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
                    mask = np.ones(int_mat.shape[:2], dtype="uint8") * 255
                    cv2.drawContours(mask, [cnt], -1, 0, -1)
                    # extractMat = np.multiply(image, mask > 0)
                    # sumContour = np.multiply(extractMat, mask > 0).sum()
                    # print("Sum contour {:d}: {:.1f}".format(i, sumContour))

            bad_contours = [contours[i] for i in bad_indexes]

            # Create mask with contours to remove
            mask = np.ones(int_mat.shape[:2], dtype="uint8") * 255
            cv2.drawContours(mask, bad_contours, -1, 0, -1)

            # Apply mask on matrix
            result = np.multiply(int_mat, mask > 0)

            # Rescale to [0,1]
            result = result / 255.

            return result