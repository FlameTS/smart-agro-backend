import torch
import os

BASE = r"D:\BCY 3rd Year\EPICS\EP1\my-crop-backend"
model_path = os.path.join(BASE, "weights", "crop_disease_model.pth")

try:
    checkpoint = torch.load(model_path, map_location="cpu")
    print("Class names:", checkpoint.get("class_names", "Not found"))
except Exception as e:
    print(f"Error reading checkpoint: {e}")
