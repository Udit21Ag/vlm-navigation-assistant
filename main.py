import argparse
import os
import time
import cv2
import datetime
from concurrent.futures import ThreadPoolExecutor
from models.frame_sampler import FrameSampler
from models.depth_estimator import DepthEstimator
from models.detector import ObjectDetector
from models.spatial_reasoning import SpatialReasoner
from models.tracker import ObjectTracker
from models.temporal_reasoner import TemporalReasoner
from models.scene_memory import SceneMemory
from models.navigation_planner import NavigationPlanner
from models.metrics import RuntimeMetrics
from caption.temporal_caption import TemporalCaptionGenerator
from tts.event_speaker import EventSpeaker
from utils.visualize import draw_boxes
# from models.vlm_reasoner import VLMReasoner

# ─────────────────────────────────────────────────────────────
# Overlay functions
# ─────────────────────────────────────────────────────────────
def _overlay_instruction(frame, caption, urgency):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    color = {
        "critical": (0, 0, 200),
        "warning":  (0, 140, 255),
        "info":     (80, 80, 80),
    }.get(urgency, (80, 80, 80))

    cv2.rectangle(overlay, (0, h - 50), (w, h), color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(
        frame, caption,
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )
    return frame


def _overlay_fps(frame, fps):
    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
    )
    return frame


def _overlay_metrics(frame, metrics):
    h, w = frame.shape[:2]
    text = f"LAT {metrics['latency']*1000:.0f}ms  FLIP {metrics['flip_rate']:.2f}"
    cv2.putText(
        frame, text,
        (10, h - 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
    )
    return frame


# ─────────────────────────────────────────────────────────────
# IMAGE MODE
# ─────────────────────────────────────────────────────────────
def run_image(image_path, use_tts=True):

    print("[INIT] Loading models...")

    depth_estimator = DepthEstimator()
    detector = ObjectDetector()
    memory = SceneMemory()
    planner = NavigationPlanner()
    captioner = TemporalCaptionGenerator()
    speaker = EventSpeaker() if use_tts else None
    # VLM disabled: keep code commented for future use.
    # vlm = VLMReasoner()

    frame = cv2.imread(image_path)
    if frame is None:
        raise RuntimeError("Image not found")

    h, w = frame.shape[:2]

    depth_map = depth_estimator.estimate_depth(frame)
    reasoner = SpatialReasoner(w, h, depth_map)

    detections = detector.detect(frame)

    # Enrich all detections so output image always shows object boxes + spatial tags.
    enriched_all = [reasoner.assign_risk(d.copy()) for d in detections]
    enriched = reasoner.prioritize_hazards(detections)

    temporal_objects = [{
        "track_id": 0,
        "label": d["label"],
        "zone": d["direction"],
        "distance": d["distance"],
        "motion": "stationary",
        "risk": d["risk_score"],
    } for d in enriched]

    # Simple cost logic
    cost_map = {
        "far left": 0.1,
        "left": 0.2,
        "center": 0.9,
        "right": 0.3,
        "far right": 0.2,
    }

    safest = min(cost_map, key=cost_map.get)

    instruction, urgency = planner.decide(
        temporal_objects, cost_map, safest, memory.get_best_corridor()
    )

    # VLM disabled: use temporal caption pipeline directly.
    _, full_caption = captioner.generate(
        temporal_objects, instruction, urgency
    )

    if speaker:
        speaker.speak(instruction, urgency)

    vis = draw_boxes(frame.copy(), enriched_all)
    vis = _overlay_instruction(vis, full_caption, urgency)

    os.makedirs("outputs", exist_ok=True)

    filename = os.path.basename(image_path)
    name = os.path.splitext(filename)[0]
    timestamp = datetime.datetime.now().strftime("%H%M%S")

    out_path = f"outputs/{name}_{timestamp}.jpg"
    cv2.imwrite(out_path, vis)

    print("[SAVED]", out_path)

    cv2.imshow("Image Result", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if speaker:
        speaker.shutdown()


# ─────────────────────────────────────────────────────────────
# VIDEO MODE
# ─────────────────────────────────────────────────────────────
def run(source, use_tts=True, sample_interval_ms=300):

    os.makedirs("outputs", exist_ok=True)

    print("[INIT] Loading models...")
    depth_estimator = DepthEstimator()
    detector = ObjectDetector()
    tracker = ObjectTracker()

    temporal = TemporalReasoner()
    memory = SceneMemory()
    planner = NavigationPlanner()
    # VLM disabled: keep code commented for future use.
    # vlm = VLMReasoner()
    captioner = TemporalCaptionGenerator()
    speaker = EventSpeaker() if use_tts else None
    metrics = RuntimeMetrics()

    print("[INIT] Ready.\n")

    # SETTINGS
    DEPTH_INTERVAL = 6
    DETECTION_INTERVAL = 2
    SEGMENT_INTERVAL = 3
    RESIZE_SCALE = 0.6

    executor = ThreadPoolExecutor(max_workers=3)

    # VIDEO WRITER INIT
    # ───────────────── VIDEO WRITER INIT (FIXED) ─────────────────

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    out_path = "outputs/output.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # mac best codec

    writer = cv2.VideoWriter(
        out_path,
        fourcc,
        10,
        (w, h),
        True
    )

    # fallback if codec fails
    if not writer.isOpened():
        print("⚠️ MP4 failed. Switching to AVI...")

        out_path = "outputs/output.avi"

        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*'XVID'),
            10,
            (w, h),
            True
        )

    t_loop_start=time.monotonic()

    frame_count = 0
    depth_map = None
    prev_detections = []
    reasoner = None
    detect_future = None
    depth_future = None
    adaptive_interval = sample_interval_ms

    with FrameSampler(source, sample_interval_ms=sample_interval_ms) as sampler:

        for frame, timestamp in sampler:

            t0 = time.monotonic()
            frame_count += 1

            # Resize
            small = cv2.resize(frame, None, fx=RESIZE_SCALE, fy=RESIZE_SCALE)
            h_s, w_s = small.shape[:2]

            # Depth & detections run asynchronously but persist across frames.
            if (frame_count % DEPTH_INTERVAL == 0 or depth_map is None) and depth_future is None:
                depth_future = executor.submit(depth_estimator.estimate_depth, small)

            if (frame_count % DETECTION_INTERVAL == 0 or not prev_detections) and detect_future is None:
                detect_future = executor.submit(detector.detect, small)

            if depth_future is not None and depth_future.done():
                depth_map = depth_future.result()
                depth_future = None

            if detect_future is not None and detect_future.done():
                prev_detections = detect_future.result()
                detect_future = None

            detections = prev_detections

            if not detections:
                continue

            if depth_map is None:
                depth_map = depth_estimator.estimate_depth(small)  # fallback

            # Spatial
            if reasoner is None:
                reasoner = SpatialReasoner(w_s, h_s, depth_map)
            else:
                reasoner.depth_map = depth_map

            

            enriched = reasoner.prioritize_hazards(detections)

            tracked = tracker.update(enriched, small)

            # Scale boxes back
            for d in tracked:
                d["bbox"] = [int(x / RESIZE_SCALE) for x in d["bbox"]]

            # Temporal
            temporal_objects = temporal.update(tracked, timestamp)

            # Memory
            memory.update(temporal_objects)
            cost_map = memory.get_cost_map()
            safest = memory.get_safest_direction()
            corridor = memory.get_best_corridor()

            # Planning
            instruction, urgency = planner.decide(
                temporal_objects, cost_map, safest, corridor
            )

            # Caption
            # VLM disabled: always generate from temporal captioner.
            _, full_caption = captioner.generate(
                temporal_objects, instruction, urgency
            )

            # TTS
            if speaker:
                speaker.speak(instruction, urgency)

            # Visual
            vis = draw_boxes(frame.copy(), tracked)
            vis = _overlay_instruction(vis, full_caption, urgency)

            fps = 1.0 / max(time.monotonic() - t0, 1e-6)
            metrics.add_frame_time(time.monotonic() - t0)
            metrics.add_latency(time.monotonic() - timestamp)
            metrics.add_instruction(instruction)
            vis = _overlay_fps(vis, fps)
            vis = _overlay_metrics(vis, metrics.snapshot())

            if urgency == "critical":
                adaptive_interval = max(60, sample_interval_ms // 2)
            elif urgency == "warning":
                adaptive_interval = max(90, int(sample_interval_ms * 0.75))
            else:
                adaptive_interval = min(250, int(sample_interval_ms * 1.1))

            sampler.set_interval(adaptive_interval)

            # SAVE VIDEO
            # Ensure correct size (VERY IMPORTANT)
            if vis.shape[1] != w or vis.shape[0] != h:
                vis = cv2.resize(vis, (w, h))

            writer.write(vis)

            # SAVE FRAMES (optional)
            if frame_count % 5 == 0:
                cv2.imwrite(f"outputs/frame_{frame_count:04d}.jpg", vis)

            cv2.imshow("Navigation Assistant", vis)

            if frame_count % 5 == 0:
                print(f"[F{frame_count:04d}] {full_caption}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    writer.release()
    cv2.destroyAllWindows()

    if speaker:
        speaker.shutdown()

    total = time.monotonic() - t_loop_start
    print(f"[DONE] {frame_count} frames in {total:.1f}s "
          f"({frame_count / max(total, 1e-6):.1f} FPS)")

    print(f"[SAVED] Video → {out_path}")
    executor.shutdown(wait=True)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--interval", type=int, default=300)

    args = parser.parse_args()

    if args.image:
        run_image(args.image, use_tts=not args.no_tts)

    elif args.source:
        try:
            source = int(args.source)
        except:
            source = args.source

        run(source, use_tts=not args.no_tts, sample_interval_ms=args.interval)

    else:
        print("Provide --source or --image")