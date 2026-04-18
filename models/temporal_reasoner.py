"""Temporal Reasoning Module

Analyzes object motion and state across multiple frames to detect hazardous
motion patterns (approaching, crossing) and estimate time-to-collision.

Features:
- Motion vector computation (frame-to-frame displacement)
- Approaching vehicle detection
- Crossing pedestrian detection
- Time-to-collision (TTC) estimation
- Multi-frame state tracking
"""

import time
from collections import defaultdict, deque

_HISTORY_LEN = 10

class TemporalReasoner:
    """Builds a Temporal Scene Graph from tracked detections."""

    def __init__(self, history_length=_HISTORY_LEN):
        self._history = defaultdict(lambda: deque(maxlen=history_length))

    # --------------------------------------------------------------
    def update(self, tracked_detections, timestamp=None):

        ts = timestamp or time.monotonic()
        active_ids = set()

        for det in tracked_detections:
            tid = det["track_id"]
            active_ids.add(tid)

            self._history[tid].append({
                "zone": det.get("direction", "center"),
                "distance": det.get("distance", "far"),
                "depth": det.get("raw_depth_value", 0.0),
                "timestamp": ts,
                "bbox": det.get("bbox", [0, 0, 0, 0]),
                "label": det.get("label", "unknown"),
                "risk": det.get("risk_score", 0.0),
                "proximity": det.get("proximity_score", det.get("risk_score", 0.0)),
            })

        # Remove stale tracks
        stale = [
            tid for tid in self._history
            if tid not in active_ids
            and len(self._history[tid]) > 0
            and (ts - self._history[tid][-1]["timestamp"]) > 3.0
        ]
        for tid in stale:
            del self._history[tid]

        temporal_objects = []
        for tid in active_ids:
            hist = self._history[tid]
            if len(hist) < 1:
                continue

            temporal_objects.append(
                self._build_temporal_object(tid, hist)
            )

        temporal_objects.sort(key=lambda o: o["risk"], reverse=True)
        return temporal_objects

    # --------------------------------------------------------------
    def _build_temporal_object(self, track_id, history):

        latest = history[-1]

        zones = [s["zone"] for s in history]
        distances = [s["distance"] for s in history]

        return {
            "track_id": track_id,
            "label": latest["label"],
            "zone": latest["zone"],
            "zone_trajectory": self._trajectory_string(zones),
            "distance": latest["distance"],
            "distance_trajectory": self._trajectory_string(distances),
            "motion": self._compute_motion(history),
            "velocity": self._compute_velocity(history),
            "risk": latest["risk"],
            "proximity": latest["proximity"],
            "ttc": self._compute_ttc(history),
            "frames_tracked": len(history),
        }

    # --------------------------------------------------------------
    @staticmethod
    def _trajectory_string(values):
        if not values:
            return ""
        collapsed = [values[0]]
        for v in values[1:]:
            if v != collapsed[-1]:
                collapsed.append(v)
        return " → ".join(collapsed)

    # --------------------------------------------------------------
    @staticmethod
    def _compute_motion(history):

        if len(history) < 3:
            return "stationary"

        hist = list(history)

        depths = [s["depth"] for s in hist]

        mid = len(depths) // 2
        first_half = depths[:mid]
        second_half = depths[mid:]

        if not first_half or not second_half:
            return "stationary"

        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)

        diff = avg_second - avg_first

        # Higher threshold to aggressively filter ego-motion
        # (walking camera makes everything appear approaching at ~0.04-0.08 per step)
        threshold = 0.10

        # ----------------------------------------------------------
        # PRIORITY 1: depth-based motion WITH bbox cross-check
        # True approaching objects get BOTH closer (depth increase) AND
        # larger in the frame (bbox area grows).  Ego-motion increases
        # depth for everything but bbox size stays roughly constant.
        # ----------------------------------------------------------
        if diff > threshold:
            # Cross-check: did the bounding box actually grow?
            first_bboxes = [s["bbox"] for s in hist[:mid]]
            second_bboxes = [s["bbox"] for s in hist[mid:]]

            def avg_area(bboxes):
                areas = []
                for b in bboxes:
                    if len(b) == 4:
                        areas.append((b[2] - b[0]) * (b[3] - b[1]))
                return sum(areas) / max(len(areas), 1)

            area_first = avg_area(first_bboxes)
            area_second = avg_area(second_bboxes)

            # Require at least 8% bbox growth to confirm approach
            # (rules out ego-motion which doesn't change object size)
            if area_first > 0 and (area_second - area_first) / area_first > 0.08:
                return "approaching"
            # Depth says approaching but bbox didn't grow → likely ego-motion
            # Still mark as approaching if depth change is very large (>0.18)
            elif diff > 0.18:
                return "approaching"
            # Otherwise treat as stationary (ego-motion artefact)
        elif diff < -threshold:
            return "receding"

        # ----------------------------------------------------------
        # PRIORITY 2: lateral crossing
        # ----------------------------------------------------------
        zone_order = {
            "far left": 0,
            "left": 1,
            "center": 2,
            "right": 3,
            "far right": 4
        }

        z_vals = [zone_order.get(z, 2) for z in [s["zone"] for s in hist]]

        mid = len(z_vals) // 2
        z1 = z_vals[:mid]
        z2 = z_vals[mid:]

        if z1 and z2:
            shift = abs((sum(z2)/len(z2)) - (sum(z1)/len(z1)))

            # 🔥 FIXED threshold
            if shift > 1.2:
                return "crossing"

        return "stationary"

    @staticmethod
    def _compute_velocity(history):

        if len(history) < 2:
            return 0.0

        last = history[-1]
        prev = history[-2]

        dt = last["timestamp"] - prev["timestamp"]
        if dt < 1e-6:
            return 0.0

        return (last["depth"] - prev["depth"]) / dt

    @staticmethod
    def _compute_ttc(history):
        if len(history) < 3:
            return float("inf")

        # Use average of last 3 frames to smooth out noise
        recent = list(history)[-3:]
        first = recent[0]
        last = recent[-1]

        dt = last["timestamp"] - first["timestamp"]
        if dt < 0.05:  # need at least 50ms span for reliable estimate
            return float("inf")

        depth_delta = last["depth"] - first["depth"]

        # Only compute TTC for objects getting closer (positive delta)
        if depth_delta <= 0.01:
            return float("inf")

        relative_speed = depth_delta / dt
        if relative_speed < 0.02:  # ignore very slow changes (likely noise)
            return float("inf")

        remaining = max(0.0, 1.0 - last["depth"])
        return remaining / relative_speed