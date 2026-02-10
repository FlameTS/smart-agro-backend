# ==============================
# Smart Agro Backend (FastAPI)
# ==============================

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware


import torch
from torchvision import transforms
from PIL import Image
import sqlite3
import io

from model.model_def import CropDiseaseCNN

import numpy as np

def is_likely_leaf(pil_image):
    img = np.array(pil_image)

    # Convert to float
    img = img.astype("float")

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    # Green dominance heuristic
    green_ratio = np.mean(g > r) + np.mean(g > b)
    green_ratio /= 2

    return green_ratio > 0.45

# ------------------------------
# App initialization
# ------------------------------
app = FastAPI(title="Smart Agro Crop Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow React frontend (dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Device
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------
# Load model ONCE
# ------------------------------
checkpoint = torch.load(
    "model/crop_disease_model.pth",
    map_location=device
)

class_names = checkpoint["class_names"]

model = CropDiseaseCNN(num_classes=len(class_names))
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

print("Model loaded successfully")

# ------------------------------
# Image transform (same as validation)
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------
# Database connection
# ------------------------------
def get_connection():
    return sqlite3.connect("DB/smart_agro.db")

# ------------------------------
# Health check
# ------------------------------
@app.get("/")
def health_check():
    return {"status": "Smart Agro API is running"}

# ------------------------------
# Prediction endpoint
# ------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image bytes
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 🔴 Leaf validity check (BEFORE inference)
        if not is_likely_leaf(pil_image):
            return {
                "status": "uncertain",
                "confidence": 0.0,
                "message": "Image does not appear to be a crop leaf."
            }

        # Preprocess for model
        image = transform(pil_image).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(image)
            probs = torch.softmax(outputs, dim=1)

        max_prob, max_index = torch.max(probs, dim=1)
        confidence = max_prob.item()
        predicted_label = class_names[max_index.item()]

        # 🔴 Confidence threshold check
        if confidence < 0.6:
            return {
                "status": "uncertain",
                "confidence": confidence,
                "message": "Unable to confidently identify the crop or disease."
            }

        crop, disease = predicted_label.split("___")

        # --------------------------
        # Database lookup
        # --------------------------
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT info FROM crops WHERE crop_name = ?",
            (crop,)
        )
        crop_info = cursor.fetchone()

        disease_data = None
        if "healthy" not in disease.lower():
            cursor.execute(
                "SELECT description, cure FROM diseases WHERE disease_name = ?",
                (disease,)
            )
            disease_data = cursor.fetchone()

        conn.close()

        # --------------------------
        # Response
        # --------------------------
        return {
            "status": "success",
            "crop": crop,
            "disease": disease.replace("_", " "),
            "confidence": confidence,   # 0–1 ONLY
            "crop_info": crop_info[0] if crop_info else "Information not available",
            "disease_info": disease_data[0] if disease_data else None,
            "cure": disease_data[1] if disease_data else None
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
