from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="models/weights/idd_best.pt", conf_thresh=0.2):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def detect(self, image):
        results = self.model(image, conf=self.conf_thresh, iou=0.6,max_det=30, verbose=False)[0]
        detections = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.conf_thresh:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            area = (x2-x1)*(y2-y1)

            if area < 800:
                continue
            if label == "rider":
                continue

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "confidence": conf
            })

        print("Detected objects:")
        for d in detections:
            print(d["label"], d["confidence"])

        return detections