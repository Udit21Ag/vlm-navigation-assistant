"""
Vision-Based Navigation Assistant - Main Entry Point

Processes images or video streams to generate navigation guidance for visually
impaired users. Combines object detection, depth estimation, spatial reasoning,
and rule-based navigation planning.

Usage:
    python main.py --image samples/road.jpg
    python main.py --source samples/vid.mp4
    python main.py --source samples/vid.mp4 --no-tts
"""

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
from utils.visualize import draw_boxes, draw_road_overlay
from models.vlm_reasoner import VLMReasoner
from models.road_detector import RoadDetector

# ─────────────────────────────────────────────────────────────
# Overlay functions
# ─────────────────────────────────────────────────────────────
def _wrap_text(text, max_width, font, font_scale, thickness):
    """Wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = " ".join(current_line + [word])
        (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        
        if text_width > max_width and current_line:
            lines.append(" ".join(current_line))
            current_line = [word]
        else:
            current_line.append(word)
    
    if current_line:
        lines.append(" ".join(current_line))
    
    return lines


def _overlay_instruction(frame, caption, urgency):
    h, w = frame.shape[:2]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    line_height = 25
    padding = 10
    max_text_width = w - 20  # Leave 10px margin on each side
    
    # Wrap text
    lines = _wrap_text(caption, max_text_width, font, font_scale, thickness)
    
    # Calculate rectangle height based on number of lines
    rect_height = len(lines) * line_height + 2 * padding
    
    # Draw semi-transparent overlay rectangle
    overlay = frame.copy()
    color = {
        "critical": (0, 0, 200),
        "warning":  (0, 140, 255),
        "info":     (80, 80, 80),
    }.get(urgency, (80, 80, 80))
    
    cv2.rectangle(overlay, (0, h - rect_height), (w, h), color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw wrapped text lines
    y_pos = h - rect_height + padding + 15  # Vertical position of first line
    for line in lines:
        cv2.putText(
            frame, line,
            (10, y_pos),
            font, font_scale, (255, 255, 255), thickness
        )
        y_pos += line_height
    
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
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
    )
    return frame


# ─────────────────────────────────────────────────────────────
# IMAGE MODE
# ─────────────────────────────────────────────────────────────
def run_image(image_path, use_tts=True):

    # Validate model weights before loading
    for path, desc in [("models/weights/idd_best.pt", "YOLOv8-IDD"),
                       ("MiDaS/weights/dpt_levit_224.pt", "MiDaS depth")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required model weight not found: {path} ({desc})\n"
                f"See README.md for download instructions."
            )

    print("[INIT] Loading models...")

    depth_estimator = DepthEstimator()
    detector = ObjectDetector()
    memory = SceneMemory()
    planner = NavigationPlanner()
    captioner = TemporalCaptionGenerator()
    vlm = VLMReasoner()
    road_detector = RoadDetector()
    speaker = EventSpeaker() if use_tts else None

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

    # Road / walkable-space detection (uses the same depth map — no extra model)
    road_state = road_detector.detect(depth_map, frame, enriched_all)

    temporal_objects = [{
        "track_id": 0,
        "label": d["label"],
        "zone": d["direction"],
        "distance": d["distance"],
        "motion": "stationary",
        "risk": d["risk_score"],
    } for d in enriched]

    # Update memory with temporal objects to compute actual cost map
    memory.update(temporal_objects, road_state=road_state)
    cost_map = memory.get_cost_map()
    safest = memory.get_safest_direction()

    instruction, urgency = planner.decide(
        temporal_objects, cost_map, safest, memory.get_best_corridor(),
        road_state=road_state
    )

    # Generate temporal caption first (good grouping, prioritization, conciseness)
    _, temporal_caption = captioner.generate(
        temporal_objects, instruction, urgency, road_state=road_state
    )

    # Refine temporal caption with VLM visual grounding
    full_caption = vlm.generate(frame, temporal_caption, instruction)

    print(f"output -> {full_caption}")

    if speaker:
        speaker.speak(full_caption, urgency)

    vis = draw_road_overlay(frame.copy(), road_state)
    vis = draw_boxes(vis, enriched_all)
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
def run(source, use_tts=True, sample_interval_ms=300, save_frames=False):

    os.makedirs("outputs", exist_ok=True)

    # Validate model weights before loading
    for path, desc in [("models/weights/idd_best.pt", "YOLOv8-IDD"),
                       ("MiDaS/weights/dpt_levit_224.pt", "MiDaS depth")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required model weight not found: {path} ({desc})\n"
                f"See README.md for download instructions."
            )

    print("[INIT] Loading models...")
    depth_estimator = DepthEstimator()
    detector = ObjectDetector()
    tracker = ObjectTracker()

    temporal = TemporalReasoner()
    memory = SceneMemory()
    planner = NavigationPlanner()
    captioner = TemporalCaptionGenerator()
    road_detector = RoadDetector()
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
    has_seen_hazard = False
    last_smoothed_instruction = ""
    instruction_streak = 0
    road_state = None          # latest RoadDetector output
    ROAD_INTERVAL = 3          # run road detection every N frames

    with FrameSampler(source, sample_interval_ms=sample_interval_ms) as sampler:

      try:
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

            # Initialize depth and reasoner on first frame (don't wait for detections)
            if reasoner is None:
                if depth_map is None:
                    depth_map = depth_estimator.estimate_depth(small)
                reasoner = SpatialReasoner(w_s, h_s, depth_map)
            
            # Update depth map for reasoner if available
            if depth_map is not None:
                reasoner.depth_map = depth_map

            # Process detections (can be empty initially, tracker carries forward)
            enriched = reasoner.prioritize_hazards(detections) if detections else []

            tracked = tracker.update(enriched, small)

            # Scale boxes back
            for d in tracked:
                d["bbox"] = [int(x / RESIZE_SCALE) for x in d["bbox"]]

            # Temporal
            temporal_objects = temporal.update(tracked, timestamp)
            if temporal_objects:
                has_seen_hazard = True

            # Road / walkable-space detection (runs on the existing depth map)
            if depth_map is not None and (frame_count % ROAD_INTERVAL == 0 or road_state is None):
                road_state = road_detector.detect(
                    depth_map, small, prev_detections
                )

            # Memory
            memory.update(temporal_objects, road_state=road_state)
            cost_map = memory.get_cost_map()
            safest = memory.get_safest_direction()
            corridor = memory.get_best_corridor()

            # Planning
            instruction, urgency = planner.decide(
                temporal_objects, cost_map, safest, corridor,
                road_state=road_state
            )

            # Caption
            # VLM disabled: always generate from temporal captioner.
            smoothed_instruction, full_caption = captioner.generate(
                temporal_objects, instruction, urgency,
                road_state=road_state
            )

            if smoothed_instruction == last_smoothed_instruction:
                instruction_streak += 1
            else:
                last_smoothed_instruction = smoothed_instruction
                instruction_streak = 1

            # TTS (use smoothed instruction, same as visual caption).
            # Passive prompts are allowed, but only when stable; this avoids
            # one-frame "Continue forward" blips on short clips.
            if speaker:
                is_passive = "continue" in smoothed_instruction.lower()
                if not is_passive:
                    speaker.speak(smoothed_instruction, urgency)
                elif has_seen_hazard and instruction_streak >= 3:
                    speaker.speak(smoothed_instruction, urgency)

            # Visual — road overlay + boxes + instruction
            vis = draw_road_overlay(frame.copy(), road_state)
            vis = draw_boxes(vis, tracked)
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

            # SAVE FRAMES (only when flag is set)
            if save_frames and frame_count % 5 == 0:
                cv2.imwrite(f"outputs/frame_{frame_count:04d}.jpg", vis)

            cv2.imshow("Navigation Assistant", vis)

            if frame_count % 5 == 0:
                print(f"[F{frame_count:04d}] {full_caption}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

      finally:
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
    parser.add_argument("--save-frames", action="store_true",
                        help="Save individual frames to outputs/ (can fill disk on long videos)")

    args = parser.parse_args()

    if args.image:
        run_image(args.image, use_tts=not args.no_tts)

    elif args.source:
        try:
            source = int(args.source)
        except ValueError:
            source = args.source

        run(source, use_tts=not args.no_tts, sample_interval_ms=args.interval,
            save_frames=args.save_frames)

    else:
        print("Provide --source or --image")