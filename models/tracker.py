"""
Layer 3A — Object Tracking (DeepSORT wrapper)

Wraps the `deep-sort-realtime` package to assign persistent track IDs
to detected objects across consecutive video frames.
"""

from deep_sort_realtime.deepsort_tracker import DeepSort


class ObjectTracker:
    """Lightweight DeepSORT wrapper for multi-object tracking."""

    def __init__(self, max_age=15, n_init=1, max_iou_distance=0.7):
        """
        Args:
            max_age: Max frames a track is kept without a matching detection.
            n_init: Number of consecutive detections before a track is confirmed.
            max_iou_distance: Maximum IoU distance for data association.
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
        )

    def update(self, detections, frame):
        """
        Update tracker with new per-frame detections.

        Args:
            detections: list of dicts with keys
                        {bbox: [x1,y1,x2,y2], label, confidence, ...}
            frame: the BGR image (used by DeepSORT for Re-ID features).

        Returns:
            List of tracked detection dicts, each enriched with:
                track_id  (int)   — persistent ID across frames
                age       (int)   — number of frames this track has existed
            Only confirmed (active) tracks are returned.
        """
        if not detections:
            # Still need to advance tracker state even with no detections
            self.tracker.update_tracks([], frame=frame)
            return []

        # deep-sort-realtime expects detections as list of
        # ([x1, y1, w, h], confidence, class_name)
        raw_dets = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1
            raw_dets.append(
                ([x1, y1, w, h], det["confidence"], det["label"])
            )

        tracks = self.tracker.update_tracks(raw_dets, frame=frame)

        tracked = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            det_class = track.get_det_class()

            tracked.append({
                "track_id": track.track_id,
                "bbox": list(ltrb),
                "label": det_class if det_class else "unknown",
                "confidence": track.get_det_conf() or 0.0,
                "age": track.age,
            })

        return tracked
