from ultralytics import YOLO

class ObjectDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")  # lightweight version

    def detect(self, image):
        results = self.model(image)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = self.model.names[cls_id]

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "confidence": conf
            })

        return detections