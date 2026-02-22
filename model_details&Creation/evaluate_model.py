import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Import model architecture
from model_def import CropDiseaseCNN


# ----------------------------
# 1. CONFIG
# ----------------------------
MODEL_PATH = r"D:/BCY 3rd Year/EPICS/P1_dependances/model/crop_disease_model.pth"
DATA_DIR = r"D:/BCY 3rd Year/EPICS/P1_dependances/model/valid"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# 2. CHECK PATH
# ----------------------------
print("Checking dataset path...")
print("Path:", DATA_DIR)
print("Exists:", os.path.exists(DATA_DIR))

if not os.path.exists(DATA_DIR):
    raise ValueError("Dataset path does NOT exist. Fix DATA_DIR.")


# ----------------------------
# 3. LOAD CHECKPOINT
# ----------------------------
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

class_names = checkpoint["class_names"]
num_classes = len(class_names)

model = CropDiseaseCNN(num_classes)
model.load_state_dict(checkpoint["model_state"])
model.to(DEVICE)
model.eval()

print("Model loaded successfully")
print("Number of classes:", num_classes)


# ----------------------------
# 4. VALIDATION TRANSFORM
# (Same normalization as training)
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

print("Classes found in folder:", dataset.classes)
print("Total images:", len(dataset))

# Force correct class order
dataset.classes = class_names
dataset.class_to_idx = {cls: i for i, cls in enumerate(class_names)}

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


# ----------------------------
# 5. EVALUATION
# ----------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# ----------------------------
# 6. METRICS
# ----------------------------
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

print("\nOverall Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 (Weighted):", f1_weighted)
print("F1 (Macro):", f1_macro)


# ----------------------------
# 7. CONFUSION MATRIX
# ----------------------------
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(12, 10))
sns.heatmap(cm,
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()


# ----------------------------
# 8. PER-CLASS ACCURACY
# ----------------------------
class_accuracy = cm.diagonal() / cm.sum(axis=1)

print("\nPer-Class Accuracy:")
for idx, class_name in enumerate(class_names):
    print(f"{class_name}: {class_accuracy[idx]:.4f}")


# ----------------------------
# 9. SAVE DIGITAL REPORT
# ----------------------------
report = classification_report(
    all_labels,
    all_preds,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv("classification_report.csv")

with open("metrics_summary.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 (Weighted): {f1_weighted}\n")
    f.write(f"F1 (Macro): {f1_macro}\n")

print("\nEvaluation completed successfully!")
print("Saved files:")
print(" - confusion_matrix.png")
print(" - classification_report.csv")
print(" - metrics_summary.txt")
