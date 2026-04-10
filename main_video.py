"""
Layer 8 — Real-Time Video Navigation Entry Point

Usage:
    python main_video.py --source 0                   # webcam
    python main_video.py --source path/to/video.mp4   # video file
    python main_video.py --source 0 --no-tts          # silent mode

Press 'q' to quit the live window.
"""

import argparse
import os
import time
import cv2

from models.frame_sampler import FrameSampler
from models.depth_estimator import DepthEstimator
from models.detector import ObjectDetector
from models.spatial_reasoning import SpatialReasoner
from models.tracker import ObjectTracker
from models.temporal_reasoner import TemporalReasoner
from models.scene_memory import SceneMemory
from models.navigation_planner import NavigationPlanner
from caption.temporal_caption import TemporalCaptionGenerator
from tts.event_speaker import EventSpeaker
from utils.visualize import draw_boxes


# ── HUD overlay ──────────────────────────────────────────────────────
def _overlay_instruction(frame, caption, urgency):
    """Draw the current navigation instruction on the frame."""
    h, w = frame.shape[:2]

    # Background bar
    bar_h = 50
    overlay = frame.copy()
    color = {
        "critical": (0, 0, 200),
        "warning":  (0, 140, 255),
        "info":     (80, 80, 80),
    }.get(urgency, (80, 80, 80))

    cv2.rectangle(overlay, (0, h - bar_h), (w, h), color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Text
    cv2.putText(
        frame, caption,
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )
    return frame


def _overlay_fps(frame, fps):
    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
    )
    return frame


# ── Main pipeline ────────────────────────────────────────────────────
def run(source, use_tts=True, sample_interval_ms=300):

    os.makedirs("outputs", exist_ok=True)

    # ── Initialise all modules once ──────────────────────────────────
    print("[INIT] Loading depth estimator (MiDaS) …")
    depth_estimator = DepthEstimator()

    print("[INIT] Loading object detector (YOLOv8) …")
    detector = ObjectDetector()

    print("[INIT] Initialising tracker (DeepSORT) …")
    tracker = ObjectTracker()

    temporal  = TemporalReasoner()
    memory    = SceneMemory()
    planner   = NavigationPlanner()
    captioner = TemporalCaptionGenerator()
    speaker   = EventSpeaker() if use_tts else None

    print("[INIT] All modules ready.  Starting video loop …\n")

    # ── Video loop ───────────────────────────────────────────────────
    frame_count = 0
    t_loop_start = time.monotonic()

    with FrameSampler(source, sample_interval_ms=sample_interval_ms) as sampler:
        for frame, timestamp in sampler:

            t0 = time.monotonic()
            h, w = frame.shape[:2]
            frame_count += 1

            # 1. Depth estimation
            depth_map = depth_estimator.estimate_depth(frame)

            # 2. Object detection
            detections = detector.detect(frame)

            # 3. Spatial reasoning (enriches detections in-place)
            reasoner = SpatialReasoner(w, h, depth_map=depth_map)
            enriched = reasoner.prioritize_hazards(detections)

            # 4. Object tracking (assign persistent IDs)
            tracked = tracker.update(enriched, frame)

            # Merge spatial info back onto tracked detections
            #   tracked dicts have track_id + bbox but lack
            #   direction/distance — re-enrich from spatial reasoner
            for t_det in tracked:
                t_det_enriched = reasoner.assign_risk(t_det)
                t_det.update(t_det_enriched)

            # 5. Temporal reasoning
            temporal_objects = temporal.update(tracked, timestamp)

            # 6. Scene memory
            memory.update(temporal_objects)
            cost_map = memory.get_cost_map()
            safest   = memory.get_safest_direction()

            # 7. Navigation decision
            instruction, urgency = planner.decide(
                temporal_objects, cost_map, safest
            )

            # 8. Caption generation (smoothed)
            smoothed, full_caption = captioner.generate(
                temporal_objects, instruction, urgency
            )

            # 9. TTS (event-driven, non-blocking)
            if speaker:
                speaker.speak(smoothed, urgency)

            # 10. Visualize
            vis = frame.copy()
            vis = draw_boxes(vis, enriched)
            vis = _overlay_instruction(vis, full_caption, urgency)

            elapsed = time.monotonic() - t0
            fps = 1.0 / max(elapsed, 1e-6)
            vis = _overlay_fps(vis, fps)

            cv2.imshow("Navigation Assistant", vis)

            # Log every frame to console
            n_obj = len(temporal_objects)
            print(f"[F{frame_count:04d}] ({n_obj} objects) {full_caption}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[EXIT] User pressed 'q'.  Shutting down …")
                break

    # ── Cleanup ──────────────────────────────────────────────────────
    cv2.destroyAllWindows()
    if speaker:
        speaker.shutdown()

    total = time.monotonic() - t_loop_start
    print(f"[DONE] Processed {frame_count} frames in {total:.1f}s "
          f"({frame_count / max(total, 1e-6):.1f} sampled FPS)")


# ── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time video navigation assistant"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: webcam index (0, 1, …) or path to video file",
    )
    parser.add_argument(
        "--no-tts", action="store_true",
        help="Disable text-to-speech output",
    )
    parser.add_argument(
        "--interval", type=int, default=300,
        help="Frame sampling interval in milliseconds (default: 300)",
    )
    args = parser.parse_args()

    # If the source looks like a plain integer, treat it as a webcam index
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    run(source, use_tts=not args.no_tts, sample_interval_ms=args.interval)
