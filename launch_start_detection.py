from robot.real.StartLightDetector import StartLightDetector
from robot.real.Time import Time
from robot.real.UsbCam import UsbCam

usb_cam = UsbCam()

time = Time()

start_light_detector = StartLightDetector(detect_zone_center=(100,100),
                                          detect_zone_shape=(100,100),
                                          time=time,
                                          usb_cam=usb_cam,
                                          delay_seconds=0.1,
                                          thresh=30)
while True:
    if start_light_detector.detect_start_light():
        print("Go")
        break
    time.sleep(0.01)
