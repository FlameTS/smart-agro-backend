import os
import numpy as np
from PIL import Image
import torch
import cv2

# SAM2 is installed as a package via `pip install -e sam2_repo` in the Dockerfile.
# Do NOT manually insert sys.path — it causes import conflicts.
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO


class YOLOSam2Pipeline:

    COLORS = [
        (220, 60,  60),  (60, 180,  60),  (60, 100, 220),
        (200,140,  30),  (160,  60, 200),  (30, 180, 180),
        (220,120,  60),  (60, 220, 120),  (120,  60, 220),
        (180, 30, 120),  (30, 120, 180),  (220, 200,  60),
        (60, 220, 200),  (200,  60, 220),  (120, 220,  60),
        (60,  60, 220),  (220,  60, 160),  (160, 220,  60),
        (60, 160, 220),  (220, 160,  60),  (100, 220, 160),
        (160, 100, 220),  (220, 100, 160),  (100, 160, 220),
        (180, 220, 100),  (220, 180, 100),
    ]

    def __init__(self, yolo_weights: str, sam2_weights: str, sam2_cfg: str, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Pipeline] Device: {self.device}")

        print("[Pipeline] Loading YOLO...")
        self.yolo = YOLO(yolo_weights)
        print("[Pipeline] YOLO ready.")

        print("[Pipeline] Loading SAM2...")
        # build_sam2 resolves sam2_cfg relative to the sam2 package's own configs/ directory.
        # For sam2_hiera_small the correct key is "sam2_hiera_s.yaml".
        # If you get a FileNotFoundError here, check:
        #   python -c "import sam2; import os; print(os.path.dirname(sam2.__file__))"
        # and list the configs/ subfolder to find the exact filename.
        sam2_model = build_sam2(sam2_cfg, sam2_weights, device=self.device)
        self.sam2 = SAM2ImagePredictor(sam2_model)
        print("[Pipeline] SAM2 ready.")

    def run(self, image_input) -> dict:
        if isinstance(image_input, str):
            image = np.array(Image.open(image_input).convert("RGB"))
        else:
            image = image_input

        h, w = image.shape[:2]

        yolo_results = self.yolo(image, verbose=False)[0]
        boxes     = yolo_results.boxes.xyxy.cpu().numpy()
        confs     = yolo_results.boxes.conf.cpu().numpy()
        class_ids = yolo_results.boxes.cls.cpu().numpy()

        if len(boxes) == 0:
            return {
                "detections": [],
                "annotated_image": image,
                "detection_count": 0,
            }

        self.sam2.set_image(image)
        detections = []
        annotated  = image.copy()

        for i, box in enumerate(boxes):
            class_id   = int(class_ids[i])
            class_name = self.yolo.names[class_id]
            confidence = float(confs[i])
            color      = self.COLORS[class_id % len(self.COLORS)]
            # cv2 uses BGR
            color_bgr  = (color[2], color[1], color[0])
            # numpy overlay uses RGB
            color_rgb  = color

            masks, scores, _ = self.sam2.predict(
                box=box,
                multimask_output=True,
            )
            best_mask = masks[np.argmax(scores)].astype(bool)

            annotated[best_mask] = (
                annotated[best_mask] * 0.45 +
                np.array(color_rgb) * 0.55
            ).astype(np.uint8)

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)

            label = f"{class_name.split('___')[-1].replace('_',' ')} {confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color_bgr, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            detections.append({
                "class_id":      class_id,
                "class_name":    class_name,
                "confidence":    round(confidence, 4),
                "box":           [x1, y1, x2, y2],
                "mask_coverage": round(float(best_mask.sum()) / (h * w), 4),
            })

        return {
            "detections":      detections,
            "annotated_image": annotated,
            "detection_count": len(detections),
        }