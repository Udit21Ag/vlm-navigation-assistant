import cv2
import argparse
from models.detector import ObjectDetector
from models.spatial_reasoning import SpatialReasoner
from caption.rule_based_caption import CaptionGenerator
from tts.speak import Speaker
from utils.visualize import draw_boxes

def main(image_path):
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    detector = ObjectDetector()
    detections = detector.detect(image)

    reasoner = SpatialReasoner(w, h)
    caption_data = []

    for det in detections:
        direction, distance = reasoner.compute_position(det)
        det["direction"] = direction
        det["distance"] = distance
        caption_data.append(det)

    caption_gen = CaptionGenerator()
    message = caption_gen.generate(caption_data)

    print("Generated:", message)

    speaker = Speaker()
    speaker.speak(message)
    image = draw_boxes(image, caption_data)
    cv2.imshow("Detection", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    main(args.image)