import os
import cv2
import numpy as np
from flask import Flask, request, jsonify

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

app = Flask(__name__)

# =========================
# 🔧 CONFIGURATION
# =========================
MODEL_DIR = "model"
CONFIG_PATH = os.path.join(MODEL_DIR, "config.yaml")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "model_final.pth")

# =========================
# 🧠 LOAD MODEL (once)
# =========================
cfg = get_cfg()
cfg.merge_from_file(CONFIG_PATH)

cfg.MODEL.WEIGHTS = WEIGHTS_PATH
cfg.MODEL.DEVICE = "cpu"  # Render = CPU

# Optional: adjust threshold
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

# =========================
# 🏠 HEALTH CHECK
# =========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "Detectron2 API is running"
    })

# =========================
# 🔍 PREDICT ENDPOINT
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]

    # Read image
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    # =========================
    # ⚡ Resize (important for Render RAM)
    # =========================
    max_size = 800
    h, w = image.shape[:2]

    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    # =========================
    # 🧠 INFERENCE
    # =========================
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")

    # =========================
    # 📊 FORMAT RESULTS
    # =========================
    boxes = instances.pred_boxes.tensor.numpy().tolist() if instances.has("pred_boxes") else []
    scores = instances.scores.numpy().tolist() if instances.has("scores") else []
    classes = instances.pred_classes.numpy().tolist() if instances.has("pred_classes") else []

    results = []
    for i in range(len(scores)):
        results.append({
            "box": boxes[i],
            "score": scores[i],
            "class": int(classes[i])
        })

    return jsonify({
        "num_detections": len(results),
        "detections": results
    })

# =========================
# 🚀 RUN SERVER
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
