"""
Layer 3B — Temporal Reasoning

Maintains a per-track history and computes zone trajectories,
distance trajectories, motion states, and velocity estimates
from the sliding window of tracked detections.
"""

import time
from collections import defaultdict, deque


# Maximum history entries kept per track
_HISTORY_LEN = 10


class TemporalReasoner:
    """Builds a Temporal Scene Graph from tracked, spatially-enriched detections."""

    def __init__(self, history_length=_HISTORY_LEN):
        # track_id → deque of snapshots
        # Each snapshot: {zone, distance, depth, timestamp, bbox}
        self._history = defaultdict(lambda: deque(maxlen=history_length))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, tracked_detections, timestamp=None):
        """
        Feed one frame's worth of tracked + spatially-enriched detections
        and return the temporal scene graph.

        Args:
            tracked_detections: list of dicts with keys
                track_id, label, bbox, direction, distance,
                raw_depth_value, risk_score
            timestamp: float (time.monotonic) — auto-set if None.

        Returns:
            List of TemporalObject dicts.
        """
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
            })

        # Prune tracks that haven't been seen for a while
        stale = [
            tid for tid in self._history
            if tid not in active_ids
            and len(self._history[tid]) > 0
            and (ts - self._history[tid][-1]["timestamp"]) > 3.0
        ]
        for tid in stale:
            del self._history[tid]

        # Build the temporal scene graph for currently active tracks
        temporal_objects = []
        for tid in active_ids:
            hist = self._history[tid]
            if len(hist) == 0:
                continue
            tobj = self._build_temporal_object(tid, hist)
            temporal_objects.append(tobj)

        # Sort by risk descending
        temporal_objects.sort(key=lambda o: o["risk"], reverse=True)
        return temporal_objects

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_temporal_object(self, track_id, history):
        latest = history[-1]
        label = latest["label"]

        # --- Zone trajectory ---
        zones = [s["zone"] for s in history]
        zone_traj = self._trajectory_string(zones)

        # --- Distance trajectory ---
        distances = [s["distance"] for s in history]
        dist_traj = self._trajectory_string(distances)

        # --- Motion state ---
        motion = self._compute_motion(history)

        # --- Velocity (depth change per second) ---
        velocity = self._compute_velocity(history)

        return {
            "track_id": track_id,
            "label": label,
            "zone": latest["zone"],
            "zone_trajectory": zone_traj,
            "distance": latest["distance"],
            "distance_trajectory": dist_traj,
            "motion": motion,
            "velocity": velocity,
            "risk": latest["risk"],
            "frames_tracked": len(history),
        }

    @staticmethod
    def _trajectory_string(values):
        """Collapse consecutive duplicates into a readable trajectory."""
        if not values:
            return ""
        collapsed = [values[0]]
        for v in values[1:]:
            if v != collapsed[-1]:
                collapsed.append(v)
        return " → ".join(collapsed)

    @staticmethod
    def _compute_motion(history):
        """Determine motion state from depth history."""
        if len(history) < 2:
            return "stationary"

        # Convert deque to list so slicing works
        hist_list = list(history)

        depths = [s["depth"] for s in hist_list]
        mid = len(depths) // 2
        first_half = depths[:mid]
        second_half = depths[mid:]

        if not first_half or not second_half:
            return "stationary"

        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)

        # MiDaS: higher depth value = closer
        diff = avg_second - avg_first
        threshold = 3.0  # tunable

        # Check for lateral crossing
        mid_h = len(hist_list) // 2
        zones_first = [s["zone"] for s in hist_list[:mid_h]]
        zones_second = [s["zone"] for s in hist_list[mid_h:]]

        zone_order = {"far left": 0, "left": 1, "center": 2, "right": 3, "far right": 4}
        z_first = [zone_order.get(z, 2) for z in zones_first]
        z_second = [zone_order.get(z, 2) for z in zones_second]

        lateral_shift = abs(
            (sum(z_second) / len(z_second)) - (sum(z_first) / len(z_first))
        )
        if lateral_shift > 0.8:
            return "crossing"

        if diff > threshold:
            return "approaching"
        elif diff < -threshold:
            return "receding"
        else:
            return "stationary"

    @staticmethod
    def _compute_velocity(history):
        """Depth change per second (positive = approaching)."""
        if len(history) < 2:
            return 0.0

        first = history[0]
        last = history[-1]
        dt = last["timestamp"] - first["timestamp"]
        if dt < 1e-6:
            return 0.0

        # MiDaS: higher value = closer → positive diff = approaching
        return (last["depth"] - first["depth"]) / dt
