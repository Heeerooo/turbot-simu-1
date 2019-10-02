class StartLightDetector:

    def __init__(self, detect_zone_center, detect_zone_shape, time, usb_cam, thresh, delay_seconds):
        self.time = time
        self.thresh = thresh
        self.delay_seconds = delay_seconds
        self.usb_cam = usb_cam
        self.detect_zone_center = detect_zone_center
        self.detect_zone_shape = detect_zone_shape

    previous_average_light_intensity = None
    previous_detection_time = 0

    def start(self):
        self.usb_cam.open()
        previous_average_light_intensity = None

    def detect_start_light(self):
        if self.time.time() - self.previous_detection_time < self.delay_seconds:
            return False

        current_image = self.usb_cam.read()
        start_x = round(self.detect_zone_center[0] - self.detect_zone_shape[0] / 2)
        end_x = round(self.detect_zone_center[0] + self.detect_zone_shape[0] / 2)
        start_y = round(self.detect_zone_center[1] - self.detect_zone_shape[1] / 2)
        end_y = round(self.detect_zone_center[1] + self.detect_zone_shape[1] / 2)

        for point in (start_x, end_x):
            if not 0 < point < current_image.shape[0]:
                raise IndexError(" detect zone out of image")
        for point in (start_y, end_y):
            if not 0 < point < current_image.shape[1]:
                raise IndexError(" detect zone out of image")

        detect_zone = current_image[start_y:end_y, start_x:end_x, :]
        detect_zone_area = detect_zone.shape[0] * detect_zone.shape[1]
        average_light_intensity = detect_zone.sum() / (detect_zone_area * 3)

        print("average %d" % average_light_intensity)

        if self.previous_average_light_intensity is None:
            self.previous_average_light_intensity = average_light_intensity
            return False

        delta_intensity = average_light_intensity - self.previous_average_light_intensity
        self.previous_average_light_intensity = average_light_intensity

        print("delta %d" % delta_intensity)

        self.previous_detection_time = self.time.time()

        return delta_intensity > self.thresh

    def stop(self):
        self.usb_cam.release()
