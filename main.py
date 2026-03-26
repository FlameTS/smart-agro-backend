# ==============================
# Smart Agro Backend (FastAPI)
# Supabase Edition
# ==============================

import os
import uuid
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import base64
from deep_translator import GoogleTranslator

import torch
from torchvision import transforms
from PIL import Image
from supabase import create_client, Client
import io

from model.model_def import CropDiseaseCNN

import numpy as np

from huggingface_hub import hf_hub_download

# ------------------------------
# Supabase client
# ------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
print("Supabase client initialized")


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

#-----------------------
# Translation function
#-----------------------
def translate_to_english(text):
    return GoogleTranslator(source='auto', target='en').translate(text)

def translate_from_english(text, target_lang):
    return GoogleTranslator(source='en', target=target_lang).translate(text)

# ------------------------------
# Device
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------
# Load model ONCE
# ------------------------------
model_path = hf_hub_download(
    repo_id="Tarman21/smart-agro-model-v1",
    filename="crop_disease_model.pth"
)

checkpoint = torch.load(model_path, map_location=device)

class_names = checkpoint["class_names"]

model = CropDiseaseCNN(num_classes=len(class_names))
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

target_layer = model.features[6]
cam = GradCAM(model=model, target_layers=[target_layer])

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
# Health check
# ------------------------------
@app.get("/")
def health_check():
    return {"status": "Smart Agro API is running", "database": "Supabase"}

# ------------------------------
# Prediction endpoint
# ------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), lang: str = "en"):
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
        # Grad-CAM Generation
        # --------------------------

        targets = [ClassifierOutputTarget(max_index.item())]

        model.zero_grad()
        grayscale_cam = cam(input_tensor=image, targets=targets)
        heatmap = grayscale_cam[0]

        from io import BytesIO

        original_image = pil_image.resize((224, 224))
        rgb_img = np.array(original_image).astype(np.float32) / 255.0
        
        heatmap = np.resize(heatmap, (224, 224))
        
        visualization = show_cam_on_image(rgb_img, heatmap, use_rgb=True)
        
        # Convert numpy image to PIL
        visualization_img = Image.fromarray(visualization)
        
        buffer = BytesIO()
        visualization_img.save(buffer, format="JPEG")
        gradcam_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # --------------------------
        # Supabase database lookup
        # --------------------------

        # Fetch crop info
        crop_result = (
            supabase.table("crops")
            .select("info")
            .eq("crop_name", crop)
            .execute()
        )
        crop_info = crop_result.data[0]["info"] if crop_result.data else "Information not available"

        # Fetch disease info (only if not healthy)
        disease_description = None
        disease_cure = None
        if "healthy" not in disease.lower():
            disease_result = (
                supabase.table("diseases")
                .select("description, cure")
                .eq("disease_name", disease)
                .execute()
            )
            if disease_result.data:
                disease_description = disease_result.data[0]["description"]
                disease_cure = disease_result.data[0]["cure"]

        # --------------------------
        # Upload image to Supabase Storage
        # --------------------------
        image_url = None
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:8]
            file_ext = file.filename.rsplit(".", 1)[-1] if "." in file.filename else "jpg"
            storage_path = f"{crop}/{timestamp}_{unique_id}.{file_ext}"

            supabase.storage.from_("crop-images").upload(
                path=storage_path,
                file=image_bytes,
                file_options={"content-type": file.content_type or "image/jpeg"}
            )

            image_url = f"{SUPABASE_URL}/storage/v1/object/public/crop-images/{storage_path}"
        except Exception:
            pass  # Don't fail the response if upload fails

        # --------------------------
        # Log prediction to history
        # --------------------------
        try:
            supabase.table("prediction_history").insert({
                "crop_name": crop,
                "disease_name": disease,
                "confidence": round(confidence, 4),
                "image_uploaded": image_url is not None,
                "image_url": image_url
            }).execute()
        except Exception:
            pass  # Don't fail the response if logging fails
        
        #--------------------------
        # Translate to target language
        #--------------------------
        if lang != "en":
            crop_info = translate_from_english(crop_info, lang)
            if disease_description:
                disease_description = translate_from_english(disease_description, lang)
            if disease_cure:
                disease_cure = translate_from_english(disease_cure, lang)

        # --------------------------
        # Response
        # --------------------------
        return {
            "status": "success",
            "crop": crop,
            "disease": disease.replace("_", " "),
            "confidence": confidence,   # 0–1 ONLY
            "crop_info": crop_info,
            "disease_info": disease_description,
            "cure": disease_cure,
            "gradcam_image": gradcam_base64,
            "image_url": image_url
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
