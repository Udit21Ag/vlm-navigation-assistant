import cv2
import argparse
import os
import time

from caption.rule_based_caption import CaptionGenerator
from models.detector import ObjectDetector
from models.spatial_reasoning import SpatialReasoner
from models.depth_estimator import DepthEstimator
from models.scene_graph import SceneGraphBuilder
from models.navigation_agent import NavigationAgent
from tts.speak import Speaker
from utils.visualize import draw_boxes, visualize_depth


SAMPLE_INTERVAL_MS = 350
BUFFER_SIZE = 5


def run_pipeline(frame, detector, depth_estimator, speaker=None):
    h, w, _ = frame.shape

    depth_map = depth_estimator.estimate_depth(frame)

    detections = detector.detect(frame)

    reasoner = SpatialReasoner(w, h, depth_map=depth_map)
    prioritized = reasoner.prioritize_hazards(detections)

    builder = SceneGraphBuilder()
    scene_graph = builder.build(prioritized)

    agent = NavigationAgent()
    decision = agent.decide(scene_graph)

    caption_gen = CaptionGenerator()
    message = caption_gen.generate(scene_graph, decision)

    print("Generated:", message)

    if speaker:
        speaker.speak(message)

    boxed_image = draw_boxes(frame.copy(), prioritized)

    return boxed_image, depth_map, message


def process_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load image.")
        return

    os.makedirs("outputs", exist_ok=True)

    detector = ObjectDetector()
    depth_estimator = DepthEstimator()
    speaker = Speaker()

    boxed_image, depth_map, _ = run_pipeline(
        image, detector, depth_estimator, speaker
    )

    depth_vis = visualize_depth(depth_map)

    cv2.imwrite("outputs/depth_map.jpg", depth_vis)

    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)

    cv2.imwrite(f"outputs/{name}_boxed{ext}", boxed_image)


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    detector = ObjectDetector()
    depth_estimator = DepthEstimator()
    speaker = Speaker()

    frame_buffer = []
    last_sample_time = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        current_time = time.time() * 1000

        if current_time - last_sample_time >= SAMPLE_INTERVAL_MS:
            frame_buffer.append(frame)

            if len(frame_buffer) > BUFFER_SIZE:
                frame_buffer.pop(0)

            boxed_image, _, _ = run_pipeline(
                frame, detector, depth_estimator, speaker
            )

            cv2.imshow("Navigation Output", boxed_image)

            last_sample_time = current_time

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--video", type=str, default=None)

    args = parser.parse_args()

    if args.image:
        process_image(args.image)

    elif args.video:
        process_video(args.video)

    else:
        print("Provide either --image or --video")