from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt", conf_thresh=0.4):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def detect(self, image):
        results = self.model(image, verbose=False)[0]
        detections = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.conf_thresh:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "confidence": conf
            })

        return detections