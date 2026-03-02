class SpatialReasoner:
    def __init__(self, image_width, image_height):
        self.w = image_width
        self.h = image_height

    def compute_position(self, detection):
        x1, y1, x2, y2 = detection["bbox"]
        center_x = (x1 + x2) / 2
        area = (x2 - x1) * (y2 - y1)

        # Direction
        if center_x < self.w / 3:
            direction = "left"
        elif center_x > 2 * self.w / 3:
            direction = "right"
        else:
            direction = "center"

        # Distance proxy
        area_ratio = area / (self.w * self.h)
        if area_ratio > 0.15:
            distance = "very close"
        elif area_ratio > 0.05:
            distance = "near"
        else:
            distance = "far"

        return direction, distance