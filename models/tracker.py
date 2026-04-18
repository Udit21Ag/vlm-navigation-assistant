"""Object Tracking Module

Tracks detected objects across multiple frames using DeepSORT algorithm.
Maintains consistent track IDs and motion vectors.

Features:
- Frame-to-frame object matching
- Track ID persistence
- Motion vector computation
- Handles track creation/deletion with configurable timeouts
"""

from dataclasses import dataclass


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


@dataclass
class _Track:
    track_id: int
    bbox: list
    label: str
    confidence: float
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    last_center: tuple = None
    velocity: tuple = (0.0, 0.0)
    extra_fields: dict = None  # Preserve spatial reasoning fields (direction, distance, etc.)

    def __post_init__(self):
        if self.extra_fields is None:
            self.extra_fields = {}


class ObjectTracker:
    def __init__(self, iou_threshold=0.25, max_age=6, min_hits=1):
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self._tracks = []

    def update(self, detections, frame=None):
        tracked = []

        for track in self._tracks:
            track.age += 1
            track.time_since_update += 1

        unmatched_detections = set(range(len(detections)))
        matched_tracks = set()

        for track in self._tracks:
            best_det_idx = None
            best_iou = 0.0

            for det_idx in list(unmatched_detections):
                det = detections[det_idx]
                iou = _bbox_iou(track.bbox, det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx

            if best_det_idx is not None and best_iou >= self.iou_threshold:
                det = detections[best_det_idx]
                prev_center = track.last_center or _bbox_center(track.bbox)
                new_center = _bbox_center(det["bbox"])

                dt = 1.0
                track.velocity = (
                    new_center[0] - prev_center[0],
                    new_center[1] - prev_center[1],
                )
                track.bbox = det["bbox"]
                track.label = det.get("label", track.label)
                track.confidence = float(det.get("confidence", track.confidence))
                track.last_center = new_center
                track.hits += 1
                track.time_since_update = 0
                # Preserve spatial reasoning fields
                track.extra_fields = {
                    k: v for k, v in det.items()
                    if k not in ("bbox", "label", "confidence")
                }
                matched_tracks.add(track.track_id)
                unmatched_detections.remove(best_det_idx)

        for det_idx in unmatched_detections:
            det = detections[det_idx]
            center = _bbox_center(det["bbox"])
            extra = {k: v for k, v in det.items() if k not in ("bbox", "label", "confidence")}
            track = _Track(
                track_id=self.next_id,
                bbox=det["bbox"],
                label=det.get("label", "unknown"),
                confidence=float(det.get("confidence", 0.0)),
                hits=1,
                age=1,
                time_since_update=0,
                last_center=center,
                velocity=(0.0, 0.0),
                extra_fields=extra,
            )
            self.next_id += 1
            self._tracks.append(track)

        self._tracks = [
            track for track in self._tracks
            if track.time_since_update <= self.max_age
        ]

        for track in self._tracks:
            if track.hits < self.min_hits and track.time_since_update > 0:
                continue

            obj = {
                "track_id": track.track_id,
                "bbox": [float(v) for v in track.bbox],
                "label": track.label,
                "confidence": track.confidence,
                "track_age": track.age,
                "track_hits": track.hits,
                "track_lost": track.time_since_update,
                "track_velocity": track.velocity,
            }
            # Pass through spatial reasoning fields (direction, distance, risk_score, etc.)
            obj.update(track.extra_fields)
            tracked.append(obj)

        return tracked