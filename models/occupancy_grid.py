from collections import defaultdict

import numpy as np


class OccupancyGrid:
    def __init__(self, lateral_bins=5, forward_bins=8):
        self.lateral_bins = lateral_bins
        self.forward_bins = forward_bins
        self.grid = np.zeros((forward_bins, lateral_bins), dtype=np.float32)

    def update_from_object(self, obj):
        zone = obj.get("zone", "center")
        distance = obj.get("distance", "far")
        risk = float(obj.get("risk", 0.0))

        col = {
            "far left": 0,
            "left": 1,
            "center": 2,
            "right": 3,
            "far right": 4,
        }.get(zone, 2)

        row = {
            "very close": 0,
            "near": 1,
            "moderate distance": 3,
            "far": 5,
        }.get(distance, 5)

        row = min(self.forward_bins - 1, max(0, row))
        col = min(self.lateral_bins - 1, max(0, col))

        self.grid[row, col] = min(1.0, self.grid[row, col] * 0.7 + risk)

    def decay(self, factor=0.92):
        self.grid *= factor

    def get_cost_grid(self):
        return self.grid.copy()