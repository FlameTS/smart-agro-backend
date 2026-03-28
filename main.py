# ==============================
# Smart Agro Backend (FastAPI)
# Supabase Edition
# Lang Implemented
# Groq API Implemented Chatbot
# ==============================

import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, Request
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

#----------------------------------
#Groq Chatbot Implementation
#------------------------------
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are an expert agricultural assistant helping farmers in India.
You help with crop diseases, treatments, farming tips, and plant health.
Keep responses short, simple, and farmer-friendly.
If asked about a specific disease, give practical treatment steps.
Respond in the same language the user writes in (English, Hindi, Punjabi, or Tamil)."""

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()

    user_message = body.get("message", "")

    if not user_message:
        return {"reply": "Please ask a question."}

    language = body.get("language", "en")
    disease_context = body.get("disease_context", None)
    session_id = body.get("session_id", "default")

    lang_map = {
        "en": "English",
        "hi": "Hindi",
        "pa": "Punjabi",
        "ta": "Tamil"
    }

    lang_name = lang_map.get(language, "English")
    
    # Build messages
    messages = [ 
        {"role": "system", "content": SYSTEM_PROMPT + f" Answer in {lang_name}."}
    ]

    # If disease context exists (result page), inject it
    if disease_context:
        disease = disease_context.get("disease", "")
        crop = disease_context.get("crop", "")
        confidence = disease_context.get("confidence", "")

        context_msg = f"The farmer just got a prediction result: Disease = {disease}, Crop = {crop}, Confidence = {confidence}%. Answer accordingly."
        messages.append({"role": "system", "content": context_msg})

    messages.append({"role": "user", "content": user_message})

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        max_tokens=300,
        temperature=0.7,
    )

    reply = response.choices[0].message.content

    #Save Chat History
    try:
        supabase.table("chat_history").insert([
            {"session_id": session_id, "role": "user", "message": user_message, "language": language},
            {"session_id": session_id, "role": "bot", "message": reply, "language": language}
        ]).execute()
    except Exception as e:
        print(f"Error saving chat history: {e}")
    
    return {"reply": reply}

# Clear Chat History
@app.delete("/chat/history/{session_id}")
async def clear_chat_history(session_id: str):
    try:
        supabase.table("chat_history").delete().eq("session_id", session_id).execute()
    except Exception as e:
        print(f"Error clearing chat history: {e}")
        pass
    return {"status":"cleared"}

@app.get("/chat/sessions")
async def get_sessions():
    try:
        result = supabase.table("chat_history") \
            .select("session_id, message, created_at") \
            .eq("role", "user") \
            .order("created_at", desc=True) \
            .execute()

        seen = {}
        for row in result.data:
            sid = row["session_id"]
            if sid not in seen:
                seen[sid] = {
                    "session_id": sid,
                    "preview": row["message"][:60] + "..." if len(row["message"]) > 60 else row["message"],
                    "date": row["created_at"][:10]
                }
        return {"sessions": list(seen.values())}
    except Exception as e:
        return {"sessions": [], "error": str(e)}


@app.get("/chat/sessions/{session_id}")
async def get_session_messages(session_id: str):
    try:
        result = supabase.table("chat_history") \
            .select("role, message, created_at") \
            .eq("session_id", session_id) \
            .order("created_at", desc=False) \
            .execute()
        return {"messages": result.data}
    except Exception as e:
        return {"messages": [], "error": str(e)}

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
# model_path = hf_hub_download(
#     repo_id="Tarman21/smart-agro-model-v1",
#     filename="crop_disease_model.pth"
# )

# checkpoint = torch.load(model_path, map_location=device)

# class_names = checkpoint["class_names"]

# model = CropDiseaseCNN(num_classes=len(class_names))
# model.load_state_dict(checkpoint["model_state"])
# model.to(device)
# model.eval()

# target_layer = model.features[6]
# cam = GradCAM(model=model, target_layers=[target_layer])

# print("Model loaded successfully")

# ------------------------------
# Lazy model loader (cached)
# ------------------------------
_model = None
_cam = None
_class_names = None

def load_model():
    global _model, _cam, _class_names

    if _model is None:
        print("🔄 Loading model...")

        model_path = hf_hub_download(
            repo_id="Tarman21/smart-agro-model-v1",
            filename="crop_disease_model.pth"
        )

        checkpoint = torch.load(model_path, map_location=device)
        _class_names = checkpoint["class_names"]

        _model = CropDiseaseCNN(num_classes=len(_class_names))
        _model.load_state_dict(checkpoint["model_state"])
        _model.to(device)
        _model.eval()

        target_layer = _model.features[6]
        _cam = GradCAM(model=_model, target_layers=[target_layer])

        print("✅ Model loaded successfully")

    return _model, _cam, _class_names

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
async def predict(
    file: UploadFile = File(...), 
    lang: str = Form("en")
    ):
    try:
        model, cam, class_names = load_model()
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
            crop = translate_from_english(crop, lang)
            disease = translate_from_english(disease.replace("_", " "), lang)

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
            "disease": disease,
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
