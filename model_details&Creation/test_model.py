import torch
from torchvision import transforms
from PIL import Image

from model.model_def import CropDiseaseCNN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

#Load model checkpoint

checkpoint = torch.load("model/crop_disease_model.pth", map_location=device)

class_names = checkpoint["class_names"]

model = CropDiseaseCNN(num_classes=len(class_names))
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

print("Model loaded successfully")
print("Number of classes:", len(class_names))

#image transform it must match validation

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#prediction function

def predict_image(image_path, threshold=0.6):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)

        confidence, predicted_idx = torch.max(probs, dim=1)

    predicted_class = class_names[predicted_idx.item()]
    confidence = confidence.item() * 100

    return predicted_class, confidence


#Test image
TEST_IMAGE = r"AppleCedarRust1.JPG"

predicted_class, confidence = predict_image(TEST_IMAGE)

print("\n Prediction Result")
print()
print("Predicted Class:", predicted_class)
print("Confidence:{:.2f}%".format(confidence))