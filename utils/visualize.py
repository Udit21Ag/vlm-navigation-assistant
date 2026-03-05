import cv2

def visualize_depth(depth_map):
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype("uint8")
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    return depth_color

def draw_boxes(image, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = det["label"]
        conf = det.get("confidence", 0)
        direction = det.get("direction", "")
        distance = det.get("distance", "")
        risk = det.get("risk_score", 0)

        text = f"{label} {conf:.2f} | {direction} | {distance}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)

        cv2.putText(
            image,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            2
        )

        # Draw footpoint (important for perspective debugging)
        cv2.circle(image, (int((x1+x2)/2), y2), 4, (0,0,255), -1)

    return image