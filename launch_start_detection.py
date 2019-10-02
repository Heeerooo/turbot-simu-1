from InferenceEnable import disable_inference, enable_inference
from robot.real.StartLightDetector import StartLightDetector
from robot.real.Time import Time
from robot.real.UsbCam import UsbCam

time = Time()

print("disabling inference")
disable_inference()

time.sleep(0.5)

usb_cam = UsbCam()

start_light_detector = StartLightDetector(detect_zone_center=(100, 100),
                                          detect_zone_shape=(100, 100),
                                          time=time,
                                          usb_cam=usb_cam,
                                          delay_seconds=0.1,
                                          thresh=30)

start_light_detector.start()
while True:
    if start_light_detector.detect_start_light():
        start_light_detector.stop()
        print("Go")
        break
    time.sleep(0.01)

print("enabling inference")
enable_inference()
