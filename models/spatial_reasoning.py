import numpy as np
class SpatialReasoner:
    def __init__(self, image_width, image_height, depth_map=None):
        self.w = image_width
        self.h = image_height
        self.depth_map = depth_map

    def compute_position(self, detection):
        x1, y1, x2, y2 = map(int, detection["bbox"])

        center_x = max(0, min((x1 + x2) / 2, self.w - 1))
        bottom_y = max(0, min(y2, self.h - 1))

        if center_x < self.w * 0.2:
            direction = "far left"
        elif center_x < self.w * 0.4:
            direction = "left"
        elif center_x < self.w * 0.6:
            direction = "center"
        elif center_x < self.w * 0.8:
            direction = "right"
        else:
            direction = "far right"

        if self.depth_map is not None:

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.w - 1, x2)
            y2 = min(self.h - 1, y2)

            region = self.depth_map[y1:y2, x1:x2]

            if region.size == 0:
                depth_value = self.depth_map[int(bottom_y), int(center_x)]
            else:
                depth_value = np.median(region)

            # MiDaS produces inverse depth
            depth_score = 1.0 / (depth_value + 1e-6)

            if depth_score > 0.015:
                distance = "very close"
            elif depth_score > 0.010:
                distance = "near"
            elif depth_score > 0.005:
                distance = "moderate distance"
            else:
                distance = "far"

        else:
            normalized_depth = 1 - (bottom_y / self.h)

            if normalized_depth < 0.15:
                distance = "very close"
            elif normalized_depth < 0.30:
                distance = "near"
            elif normalized_depth < 0.50:
                distance = "moderate distance"
            else:
                distance = "far"

        return direction, distance

    def assign_risk(self, detection):
        x1, y1, x2, y2 = map(int, detection["bbox"])

        obj_x = max(0, min((x1 + x2) / 2, self.w - 1))
        obj_y = max(0, min(y2, self.h - 1))

        fallback_depth_score = obj_y / self.h
        fallback_center_score = 1 - abs(obj_x - self.w/2) / (self.w/2)
        risk_score = 0.7 * fallback_depth_score + 0.3 * fallback_center_score

        if self.depth_map is not None:

            obj_x_int = int(np.clip(obj_x, 0, self.w - 1))
            obj_y_int = int(np.clip(obj_y, 0, self.h - 1))

            y1r = max(0, obj_y_int - 3)
            y2r = min(self.h, obj_y_int + 3)
            x1r = max(0, obj_x_int - 3)
            x2r = min(self.w, obj_x_int + 3)

            region = self.depth_map[y1r:y2r, x1r:x2r]

            if region.size == 0:
                depth_value = self.depth_map[obj_y_int, obj_x_int]
            else:
                depth_value = np.mean(region)

            depth_score = 1.0 / (depth_value + 1e-6)
            depth_score = depth_score / (depth_score + 1)

            lateral_score = 1 - abs(obj_x - self.w/2) / (self.w/2)
            lateral_score = max(0, lateral_score)

            risk_score = 0.8 * depth_score + 0.2 * lateral_score

        direction, distance = self.compute_position(detection)

        detection["direction"] = direction
        detection["distance"] = distance
        detection["risk_score"] = float(risk_score)
        # Raw MiDaS depth value — used by temporal reasoner for velocity
        detection["raw_depth_value"] = float(depth_value) if self.depth_map is not None else 0.0

        return detection

    def prioritize_hazards(self, detections):
        if not detections:
            return []

        enriched = [self.assign_risk(det) for det in detections]
        enriched.sort(key=lambda x: x["risk_score"], reverse=True)

        # Only consider objects reasonably close
        relevant = [
            det for det in enriched
            if det["distance"] in ["very close", "near", "moderate distance"]
        ]

        if not relevant:
            return []

        # Remove duplicates by label + direction
        unique = relevant

        return unique