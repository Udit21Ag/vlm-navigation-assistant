import numpy as np
class SpatialReasoner:
    def __init__(self, image_width, image_height, depth_map=None):
        self.w = image_width
        self.h = image_height
        self.depth_map = depth_map

    def compute_position(self, detection):
        x1, y1, x2, y2 = map(int, detection["bbox"])

        center_x = (x1 + x2) / 2
        bottom_y = y2

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

            obj_x = int(np.clip(center_x, 0, self.w - 1))
            obj_y = int(np.clip(bottom_y, 0, self.h - 1))

            depth_value = self.depth_map[obj_y, obj_x]

            min_depth = np.min(self.depth_map)
            max_depth = np.max(self.depth_map)

            normalized_depth = (depth_value - min_depth) / (max_depth - min_depth + 1e-6)

            if normalized_depth < 0.15:
                distance = "very close"
            elif normalized_depth < 0.30:
                distance = "near"
            elif normalized_depth < 0.50:
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

        obj_x = (x1 + x2) / 2
        obj_y = y2

        # -------------------------------
        # Fallback 2D geometric risk
        # -------------------------------
        fallback_depth_score = obj_y / self.h
        fallback_center_score = 1 - abs(obj_x - self.w/2) / (self.w/2)
        risk_score = 0.7 * fallback_depth_score + 0.3 * fallback_center_score

        # -------------------------------
        # Depth-aware 3D refinement
        # -------------------------------
        if self.depth_map is not None:

            # Clamp coordinates safely
            obj_x_int = int(np.clip(obj_x, 0, self.w - 1))
            obj_y_int = int(np.clip(obj_y, 0, self.h - 1))

            region = self.depth_map[obj_y_int-3:obj_y_int+3, obj_x_int-3:obj_x_int+3]
            depth_value = np.mean(region)

            # Normalize full depth map once
            min_depth = np.min(self.depth_map)
            max_depth = np.max(self.depth_map)

            normalized_depth = (depth_value - min_depth) / (max_depth - min_depth + 1e-6)

            # Closer objects → higher score
            depth_score = 1 - normalized_depth

            # ---------------------------------
            # Ground-plane lateral projection
            # ---------------------------------
            pixel_offset = obj_x - (self.w / 2)

            # approximate lateral displacement in 3D
            ground_x = pixel_offset * depth_value

            # normalize lateral displacement
            lateral_score = 1 / (abs(ground_x) + 1e-6)

            # normalize lateral score
            lateral_score = lateral_score / (lateral_score + 1)

            # Final combined geometric risk
            risk_score = 0.8 * depth_score + 0.2 * lateral_score

        direction, distance = self.compute_position(detection)

        detection["direction"] = direction
        detection["distance"] = distance
        detection["risk_score"] = float(risk_score)

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
        seen = set()
        unique = []

        for det in relevant:
            key = (det["label"], det["direction"])
            if key in seen:
                continue
            seen.add(key)
            unique.append(det)

        return unique[:2]