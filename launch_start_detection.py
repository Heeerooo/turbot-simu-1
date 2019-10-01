import os
from pathlib import Path

from robot.real.StartLightDetector import StartLightDetector
from robot.real.Time import Time
from robot.real.UsbCam import UsbCam

INFERENCE_DISABLE_FILE = "inference.disable"

current_dir = os.path.dirname(os.path.realpath(__file__))

usb_cam = UsbCam()

time = Time()

start_light_detector = StartLightDetector(detect_zone_center=(100,100),
                                          detect_zone_shape=(100,100),
                                          time=time,
                                          usb_cam=usb_cam,
                                          delay_seconds=0.1,
                                          thresh=30)


print("disabling inference")
open(current_dir + "/" + INFERENCE_DISABLE_FILE, 'a').close()

time.sleep(1)
while True:

    if start_light_detector.detect_start_light():
        print("Go")
        break
    time.sleep(0.01)

usb_cam.release()
print("enabling inference")
if Path(INFERENCE_DISABLE_FILE).is_file():
    os.remove(INFERENCE_DISABLE_FILE)
