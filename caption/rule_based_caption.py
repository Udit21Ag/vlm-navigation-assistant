class CaptionGenerator:
    def generate(self, detections):
        hazards = ["car", "bus", "truck", "motorcycle"]

        hazard_objects = [
            obj for obj in detections if obj["label"] in hazards
        ]

        if not hazard_objects:
            return "Path appears clear."

        # Sort by proximity (very close > near > far)
        priority = {"very close": 3, "near": 2, "far": 1}
        hazard_objects.sort(
            key=lambda x: priority[x["distance"]], reverse=True
        )

        # Remove duplicates by label + direction
        seen = set()
        messages = []

        for obj in hazard_objects:
            key = (obj["label"], obj["direction"])
            if key in seen:
                continue
            seen.add(key)

            label = obj["label"]
            direction = obj["direction"]
            distance = obj["distance"]

            messages.append(
                f"{label} {distance} on your {direction}"
            )

        # Only mention top 2 hazards to avoid clutter
        messages = messages[:2]

        return ". ".join(messages) + "."