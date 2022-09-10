import os

import torch
from PIL import Image
import io
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import importlib.util
import sys

def lazy_import(name):
    spec = importlib.util.find_spec(name)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module

models = {}
def get_models():
    models['yolo5'] = get_yolov5()
    models['scaled_yolo4'] = get_scaled_yolo4()
    models['ssd'] = get_ssd()
    return models

def get_yolov5():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./model/yolov5.pt')
    model.conf = 0.5
    return model

def get_scaled_yolo4():
    from ScaledYOLOv4.models import experimental
    from ScaledYOLOv4.utils import torch_utils
    from ScaledYOLOv4.models import common, yolo

    common_orig = sys.modules["models.common"]
    yolo_orig = sys.modules["models.yolo"]

    # needed for unpickling
    sys.modules["models.common"] = common
    sys.modules["models.yolo"] = yolo
    device = torch_utils.select_device("cpu")
    model = experimental.attempt_load("./model/scaled_yolov4.pt", device, True)

    sys.modules["models.common"] = common_orig
    sys.modules["models.yolo"] = yolo_orig

    return model

def get_ssd():
    # return ssd model here
    return models['yolo5']


def get_image_from_bytes(binary_image, max_size=1024):
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    width, height = input_image.size
    resize_factor = min(max_size / width, max_size / height)
    resized_image = input_image.resize(
        (
            int(input_image.width * resize_factor),
            int(input_image.height * resize_factor),
        )
    )
    return resized_image
