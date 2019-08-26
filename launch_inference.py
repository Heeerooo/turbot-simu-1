#!/usr/bin/env python3
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.models import load_model

from robot.real.Camera import Camera
from robot.real.UsbCam import UsbCam

MODEL_FILENAME = 'deep_learning/models/final_race_model_5_3.h5'

RAM_DISK_DIR = "/tmp_ram"

INFERENCE_DISABLE_FILE = "inference.disable"

MASK_LINE_FILE = RAM_DISK_DIR + "/mask_line.npy"

MASK_OBSTACLE_FILE = RAM_DISK_DIR + "/mask_obstacle.npy"

MASK_OBSTACLE_FILE_TMP = RAM_DISK_DIR + "/mask_obstacle.tmp.npy"

MASK_LINE_FILE_TMP = RAM_DISK_DIR + "/mask_line.tmp.npy"

# Bug fix for tensorflow on TX2
# See here: https://devtalk.nvidia.com/default/topic/1030875/jetson-tx2/gpu-sync-failed-in-tx2-when-running-tensorflow/
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# Load model
seq = load_model(MODEL_FILENAME)

usbCam = UsbCam()

cam = Camera(MASK_LINE_FILE, MASK_OBSTACLE_FILE)

while True:
    begin_time = time.time()

    # Check if inference is enabled
    if Path(INFERENCE_DISABLE_FILE).is_file():
        time.sleep(0.1)
        continue

    frame = usbCam.read()

    # Process inference
    predicted_masks = seq.predict(frame[np.newaxis, :, :, :])[0, ...]
    mask_line = predicted_masks[:, :, 0]
    mask_obstacle = predicted_masks[:, :, 1]

    prediction_time = time.time()

    mask_line = mask_line > 0.1
    mask_obstacle = mask_obstacle > 0.1

    # Save mask in ram disk files
    np.save(MASK_LINE_FILE_TMP, mask_line)
    np.save(MASK_OBSTACLE_FILE_TMP, mask_obstacle)

    os.rename(MASK_LINE_FILE_TMP, MASK_LINE_FILE)
    os.rename(MASK_OBSTACLE_FILE_TMP, MASK_OBSTACLE_FILE)
