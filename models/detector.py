"""Object Detection Module

Detects vehicles, pedestrians, animals, and other hazards using YOLOv8
fine-tuned on the Indian Driving Dataset (IDD).

Supports:
- Standard COCO classes (person, car, truck, etc.)
- IDD-specific classes (autorickshaw, rider, motorcycle, animal)
- Configurable confidence thresholds
- Multiple detection profiles (fast, balanced, safety-first)
"""

from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="models/weights/idd_best.pt", conf_thresh=0.2, profile="balanced"):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.profile = profile
        self.class_risk = {
            "person": 1.0,
            "pedestrian": 1.0,
            "bicycle": 0.9,
            "motorcycle": 1.0,
            "rider": 1.0,
            "car": 0.85,
            "bus": 0.95,
            "truck": 0.95,
        }

    def detect(self, image):
        iou = 0.55
        max_det = 24

        if self.profile == "fast":
            iou = 0.6
            max_det = 18
        elif self.profile == "safety-first":
            iou = 0.5
            max_det = 32

        results = self.model(
            image,
            conf=self.conf_thresh,
            iou=iou,
            max_det=max_det,
            verbose=False
        )[0]

        detections = []
        img_h, img_w = image.shape[:2]
        min_area = 0.0005 * (img_h * img_w)  # dynamic threshold

        for box in results.boxes:
            conf = float(box.conf[0])

            # (optional safety check)
            if conf < self.conf_thresh:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id].lower()

            area = (x2 - x1) * (y2 - y1)

            if area < min_area:
                continue

            if label == "rider":
                continue

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "confidence": conf,
                "class_risk": self.class_risk.get(label, 0.5),
            })

        return detections