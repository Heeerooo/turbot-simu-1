import cv2

from robot.real.Config import CAM_HEIGHT, CAM_WIDTH, CAM_HANDLE


class UsbCam:

    def __init__(self):
        self.stream = cv2.VideoCapture()

    def read(self):
        return self.stream.read()[1]

    def release(self):
        self.stream.release()

    def open(self):
        self.stream.open(CAM_HANDLE)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
