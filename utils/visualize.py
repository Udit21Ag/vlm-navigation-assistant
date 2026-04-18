"""Visualization Utilities

Draws object detections and spatial information on video frames.
Includes bounding boxes, class labels, and spatial tags (direction + distance).
Optionally overlays walkable-surface mask from the RoadDetector.
"""

import cv2
import numpy as np

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


def draw_road_overlay(image, road_state, alpha=0.25):
    """Draw a semi-transparent green overlay on walkable pixels and
    corridor boundary markers.

    Args:
        image:      BGR frame to draw on (will be modified in-place).
        road_state: Dict returned by RoadDetector.detect().
        alpha:      Opacity of the green overlay (0 = invisible, 1 = solid).

    Returns:
        image with overlay applied.
    """
    if road_state is None:
        return image

    walkable_mask = road_state.get("walkable_mask")
    if walkable_mask is None:
        return image

    h, w = image.shape[:2]
    mh, mw = walkable_mask.shape[:2]

    # Resize mask to match frame if dimensions differ (e.g. depth was on
    # a smaller resolution than the display frame).
    if mh != h or mw != w:
        walkable_mask = cv2.resize(walkable_mask, (w, h),
                                   interpolation=cv2.INTER_NEAREST)

    # Create green overlay where walkable
    overlay = image.copy()
    green = (0, 200, 80)  # BGR green
    overlay[walkable_mask > 0] = green
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Draw corridor boundaries
    corridor_width = road_state.get("corridor_width", 0)
    corridor_center = road_state.get("corridor_center", w // 2)
    proc_shape = road_state.get("proc_shape", (mh, mw))

    if corridor_width > 10:
        # Scale corridor coordinates to display frame
        scale_x = w / max(proc_shape[1], 1)
        cx = int(corridor_center * scale_x)
        cw_half = int((corridor_width / 2) * scale_x)

        # Draw dashed corridor lines
        left_x = max(0, cx - cw_half)
        right_x = min(w - 1, cx + cw_half)
        dash_len = 15
        gap_len = 10

        for y in range(h // 3, h, dash_len + gap_len):
            y_end = min(y + dash_len, h)
            cv2.line(image, (left_x, y), (left_x, y_end),
                     (0, 255, 255), 2)   # cyan left
            cv2.line(image, (right_x, y), (right_x, y_end),
                     (0, 255, 255), 2)   # cyan right

        # Label
        cv2.putText(
            image, "CORRIDOR",
            (cx - 35, h // 3 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
        )

    # Zone drivable scores overlay (top-right corner)
    zone_drivable = road_state.get("zone_drivable", {})
    if zone_drivable:
        y_start = 75
        for i, (zone, score) in enumerate(zone_drivable.items()):
            short_name = zone.replace("far ", "F").upper()[:3]
            bar_len = int(score * 60)
            x_base = w - 120
            y_pos = y_start + i * 18

            # Draw bar background
            cv2.rectangle(image, (x_base, y_pos - 10),
                         (x_base + 60, y_pos + 4), (40, 40, 40), -1)
            # Draw bar fill (green=walkable, red=blocked)
            bar_color = (0, int(200 * score), int(200 * (1 - score)))
            cv2.rectangle(image, (x_base, y_pos - 10),
                         (x_base + bar_len, y_pos + 4), bar_color, -1)
            # Label
            cv2.putText(image, f"{short_name} {score:.0%}",
                       (x_base - 55, y_pos + 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    return image