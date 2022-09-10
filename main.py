import os

from fastapi import FastAPI, File, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from util import *
from starlette.responses import Response
import io
from PIL import Image
import json
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

app = FastAPI(
    title="Custom YOLOV5 Machine Learning API",
    description="""Obtain object value out of image
                    and return image and json result""",
    version="0.0.1",
)

models = get_models()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_PATH = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_PATH / "templates"))
STATIC_PATH = str(BASE_PATH / "static")

app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")

@app.get("/", status_code=200)
def root(request: Request) -> dict:
    """
    Root GET
    """
    return TEMPLATES.TemplateResponse(
        "index.html",
        {"request": request},
    )

@app.post("/get-objects")
async def detect_food_return_json_result(file: bytes = File(...), model_type: str = Form(None)):
    input_image = get_image_from_bytes(file)
    model = models[model_type]
    if model_type == 'yolo5':
        results = model(input_image)
        detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
        detect_res = json.loads(detect_res)
        return {"result": detect_res}
    elif model_type == 'scaled_yolo4':
        from ScaledYOLOv4 import detect
        from ScaledYOLOv4.utils import torch_utils
        device = torch_utils.select_device("cpu")
        detect_res = detect.getPredictions(model, input_image, device)
        detect_res = json.loads(json.dumps(detect_res))
        return {"result": detect_res}
    elif model_type == 'ssd':
        results = model(input_image)
        detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
        detect_res = json.loads(detect_res)
        return {"result": detect_res}


@app.post("/get-detection")
async def detect_food_return_base64_img(file: bytes = File(...), model_type: str = Form(None)):
    input_image = get_image_from_bytes(file)
    model = models[model_type]
    if model_type == 'yolo5':
        results = model(input_image)
        ims = results.render()  # Return images with boxes and labels
        for img in ims:
            bytes_io = io.BytesIO()
            img_base64 = Image.fromarray(img)
            img_base64.save(bytes_io, format="jpeg")
        return Response(content=bytes_io.getvalue(), media_type="image/jpeg")
    elif model_type == 'scaled_yolo4':
        from ScaledYOLOv4 import detect
        from ScaledYOLOv4.utils import torch_utils
        device = torch_utils.select_device("cpu")

        ims = detect.detectImage(model, input_image, device)

        for img in ims:
            bytes_io = io.BytesIO()
            img_base64 = Image.fromarray(img)
            img_base64.save(bytes_io, format="jpeg")
        return Response(content=bytes_io.getvalue(), media_type="image/jpeg")
    elif model_type == 'ssd':
        results = model(input_image)
        ims = results.render()  # Return images with boxes and labels
        for img in ims:
            bytes_io = io.BytesIO()
            img_base64 = Image.fromarray(img)
            img_base64.save(bytes_io, format="jpeg")
        return Response(content=bytes_io.getvalue(), media_type="image/jpeg")
