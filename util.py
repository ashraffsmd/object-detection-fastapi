import torch
from PIL import Image
import io
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def get_models():
    models = {}
    models['yolo5'] = get_yolov5()
    models['scaled_yolo4'] = get_scaled_yolo4()
    models['ssd'] = get_ssd()
    return models

def get_yolov5():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./model/yolov5.pt')
    model.conf = 0.5
    return model

def get_scaled_yolo4():
    # return scaled yolo 4 model here
    return get_yolov5()

def get_ssd():
    # return ssd model here
    return get_yolov5()


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
