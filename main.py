import cv2
import argparse
import os

from caption.rule_based_caption import CaptionGenerator
from models.detector import ObjectDetector
from models.spatial_reasoning import SpatialReasoner
from models.depth_estimator import DepthEstimator
from models.scene_graph import SceneGraphBuilder
from models.navigation_agent import NavigationAgent
from tts.speak import Speaker
from utils.visualize import draw_boxes, visualize_depth


def main(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load image.")
        return

    h, w, _ = image.shape

    os.makedirs("outputs", exist_ok=True)

    # 1️⃣ Depth estimation
    depth_estimator = DepthEstimator()
    depth_map = depth_estimator.estimate_depth(image)

    depth_vis = visualize_depth(depth_map)
    cv2.imwrite("outputs/depth_map.jpg", depth_vis)

    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype("uint8")
    cv2.imwrite("outputs/depth_debug.jpg", depth_norm)

    # 2️⃣ Object detection
    detector = ObjectDetector()
    detections = detector.detect(image)

    # 3️⃣ Spatial reasoning
    reasoner = SpatialReasoner(w, h, depth_map=depth_map)
    prioritized = reasoner.prioritize_hazards(detections)

    # 5️⃣ Scene graph creation
    builder = SceneGraphBuilder()
    scene_graph = builder.build(prioritized)

    print("Scene Graph:", scene_graph)

    # 6️⃣ Navigation reasoning
    agent = NavigationAgent()
    decision = agent.decide(scene_graph)

    # 7️⃣ Caption generation
    caption_gen = CaptionGenerator()
    message = caption_gen.generate(scene_graph, decision)

    print("Generated:", message)

    # 8️⃣ Text to speech
    speaker = Speaker()
    speaker.speak(message)

    # 9️⃣ Visualization
    boxed_image = draw_boxes(image.copy(), prioritized)

    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)

    output_path = os.path.join("outputs", f"{name}_boxed{ext}")
    cv2.imwrite(output_path, boxed_image)

    print(f"Boxed image saved at: {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)

    args = parser.parse_args()

    main(args.image)