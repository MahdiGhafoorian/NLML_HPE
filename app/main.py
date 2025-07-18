# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 17:14:15 2025

@author: Mahdi Ghafourian
"""

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import cv2
import io

from app.pose_estimator import HeadPoseEstimator

app = FastAPI()

# Serve HTML template
templates = Jinja2Templates(directory="app/templates")

# Load model at startup
pose_model = HeadPoseEstimator()

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict_frame")
async def predict_frame(file: UploadFile = File(...)):
    # Read image file from client
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Run head pose estimation
    annotated_frame = pose_model.draw_pose(frame)

    # Encode result to JPEG
    _, buffer = cv2.imencode(".jpg", annotated_frame)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")