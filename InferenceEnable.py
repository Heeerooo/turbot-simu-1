import os
from pathlib import Path

RAM_DISK_DIR = "/tmp_ram"

INFERENCE_DISABLE_FILE = RAM_DISK_DIR + "/inference.disable"


def disable_inference():
    open(INFERENCE_DISABLE_FILE, 'a').close()


def is_inference_enabled():
    return Path(INFERENCE_DISABLE_FILE).is_file()


def enable_inference():
    if Path(INFERENCE_DISABLE_FILE).is_file():
        os.remove(INFERENCE_DISABLE_FILE)
