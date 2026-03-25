# -*- coding: utf-8 -*-
"""
Document Layout Detection API
==============================
POST /detect -> Upload an image, get JSON of detected layout boxes
"""
import io
import numpy as np
import cv2
import onnxruntime as ort
import urllib.request
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List

# ======================================
# CONFIG
# ======================================
HF_ONNX_URL = "https://huggingface.co/Achouche/detectron2/resolve/main/detectron2.onnx"
INPUT_H = 800
INPUT_W = 1333
SCORE_THRESH = 0.5
DISTANCE_THRESH = 20

CLASS_NAMES = {
    0: "text",
    1: "title",
    2: "table",
    3: "list",
    4: "figure",
}

# ======================================
# LOAD MODEL FROM HUGGING FACE (IN MEMORY)
# ======================================
print("Loading ONNX model from Hugging Face...")
with urllib.request.urlopen(HF_ONNX_URL) as f:
    model_bytes = f.read()

session = ort.InferenceSession(io.BytesIO(model_bytes).read(), providers=["CPUExecutionProvider"])
print("ONNX model loaded successfully!")

# ======================================
# RESPONSE SCHEMA
# ======================================
class BoundingBox(BaseModel):
    type: str
    bbox: List[int]      # [x1, y1, x2, y2]
    score: float

class DetectionResponse(BaseModel):
    count: int
    detections: List[BoundingBox]

# ======================================
# HELPERS
# ======================================
def merge_boxes(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, distance_threshold: int = 20):
    """
    Merge boxes of the same class if they are close or overlapping.
    """
    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=classes.dtype)

    merged_boxes, merged_scores, merged_classes = [], [], []

    for cls in np.unique(classes):
        mask = classes == cls
        cls_boxes = boxes[mask].tolist()
        cls_scores = scores[mask].tolist()

        while cls_boxes:
            base = cls_boxes.pop(0)
            best_score = cls_scores.pop(0)
            to_merge = []

            for i, other in enumerate(cls_boxes):
                dx = max(0, max(base[0], other[0]) - min(base[2], other[2]))
                dy = max(0, max(base[1], other[1]) - min(base[3], other[3]))
                if dx <= distance_threshold and dy <= distance_threshold:
                    to_merge.append(i)

            for idx in reversed(to_merge):
                other = cls_boxes.pop(idx)
                other_score = cls_scores.pop(idx)
                base = [
                    min(base[0], other[0]),
                    min(base[1], other[1]),
                    max(base[2], other[2]),
                    max(base[3], other[3]),
                ]
                best_score = max(best_score, other_score)

            merged_boxes.append(base)
            merged_scores.append(best_score)
            merged_classes.append(cls)

    return np.array(merged_boxes), np.array(merged_scores), np.array(merged_classes)


def run_inference(image: np.ndarray) -> List[dict]:
    """
    Preprocess image, run ONNX inference, rescale, filter, merge, and format results.
    """
    orig_h, orig_w = image.shape[:2]

    # 1. Preprocess
    img = cv2.resize(image, (INPUT_W, INPUT_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = img.transpose(2, 0, 1)  # HWC -> CHW

    # 2. Build inputs
    inputs = {}
    for inp in session.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        if shape[-2:] == [INPUT_H, INPUT_W]:
            inputs[inp.name] = img
        else:
            inputs[inp.name] = np.zeros(shape, dtype=np.float32)

    # 3. Inference
    raw_boxes, raw_scores, raw_classes = session.run(None, inputs)

    # 4. Rescale & filter
    scale_x, scale_y = orig_w / INPUT_W, orig_h / INPUT_H
    boxes, scores, classes = [], [], []
    for box, score, cls in zip(raw_boxes, raw_scores, raw_classes):
        if score < SCORE_THRESH:
            continue
        x1, y1, x2, y2 = box
        boxes.append([int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)])
        scores.append(float(score))
        classes.append(int(cls))

    if not boxes:
        return []

    # 5. Merge boxes by class
    boxes_arr, scores_arr, classes_arr = merge_boxes(np.array(boxes), np.array(scores), np.array(classes), distance_threshold=DISTANCE_THRESH)

    # 6. Format output
    results = [
        {"type": CLASS_NAMES.get(int(cls), str(int(cls))), "bbox": [int(b) for b in box], "score": round(float(sc), 4)}
        for box, sc, cls in zip(boxes_arr, scores_arr, classes_arr)
    ]
    results.sort(key=lambda r: r["bbox"][1])
    return results


def decode_image(raw: bytes) -> np.ndarray:
    """Convert raw bytes to OpenCV BGR image."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the uploaded image.")
    return img


# ======================================
# FASTAPI APP
# ======================================
from fastapi import FastAPI

app = FastAPI(title="Document Layout Detection API", version="1.0.0")


@app.get("/", tags=["Health"])
def health():
    return {"status": "ok", "model": "Detectron2 ONNX from Hugging Face"}


@app.post("/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG image.")
    raw = await file.read()
    try:
        image = decode_image(raw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    try:
        detections = run_inference(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    return DetectionResponse(count=len(detections), detections=detections)
