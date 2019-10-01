from robot.real.StartLightDetector import StartLightDetector
from robot.real.Time import Time
from robot.real.UsbCam import UsbCam

usb_cam = UsbCam()

time = Time()

start_light_detector = StartLightDetector(detect_zone_center=(0,0),
                                          detect_zone_shape=(100,100),
                                          time=time,
                                          usb_cam=usb_cam,
                                          delay_seconds=0.1,
                                          thresh=10000)
while True:
    print(start_light_detector.detect_start_light())
    time.sleep(0.01)
