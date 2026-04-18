"""Scene Memory & Memory Management Module

Maintains temporal state of hazards across frames using exponential decay.
Generates cost maps for navigation planning.

Features:
- Zone-based cost accumulation (5 zones: far-left, left, center, right, far-right)
- Exponential decay of stale hazards (prevents persistent false alarms)
- Dynamic cost computation (combines static baseline + dynamic hazards)
- Occupancy grid for spatial probability mapping
- Corridor detection for safe navigation paths
"""

import time
import math

from models.occupancy_grid import OccupancyGrid
from models.corridor_estimator import CorridorEstimator

ZONES = ["far left", "left", "center", "right", "far right"]


class SceneMemory:

    def __init__(self):
        self._static_cost = {
            "far left":  0.15,
            "left":      0.25,
            "center":    0.6,   # adjusted
            "right":     0.25,
            "far right": 0.15,
        }

        self._dynamic = {z: (0.0, 0.0) for z in ZONES}
        self._decay_half_life = 2.0
        self._grid = OccupancyGrid()
        self._corridor = CorridorEstimator()
        self._road_state = None  # populated by RoadDetector each frame

    def update(self, temporal_objects, road_state=None):
        # Store latest road state for corridor estimator + cost map
        if road_state is not None:
            self._road_state = road_state

        now = time.monotonic()
        self._grid.decay()  # Forget stale occupancy data each frame
        new_dynamic = {z: 0.0 for z in ZONES}

        for obj in temporal_objects:
            zone = obj.get("zone", "center")
            if zone not in new_dynamic:
                zone = "center"

            motion = obj.get("motion", "stationary")
            distance = obj.get("distance", "far")

            dist_cost = {
                "very close": 1.0,
                "near": 0.7,
                "moderate distance": 0.4,
                "far": 0.1,
            }.get(distance, 0.2)

            motion_mult = {
                "approaching": 1.5,
                "crossing": 1.3,
                "stationary": 1.0,
                "receding": 0.4,
            }.get(motion, 1.0)

            cost = dist_cost * motion_mult
            proximity = float(obj.get("proximity", obj.get("risk", 0.0)))
            ttc = obj.get("ttc", float("inf"))

            if ttc != float("inf"):
                cost += max(0.0, 0.4 / max(ttc, 0.1))
            cost += 0.6 * proximity

            # 🔥 Lateral spread
            spread = {zone: cost}

            if zone == "center":
                spread["left"] = cost * 0.4
                spread["right"] = cost * 0.4
            elif zone == "left":
                spread["center"] = cost * 0.3
            elif zone == "right":
                spread["center"] = cost * 0.3

            for z2, c in spread.items():
                new_dynamic[z2] += c
                new_dynamic[z2] = min(new_dynamic[z2], 2.0)

            self._grid.update_from_object(obj)

        # Merge with decay
        for z in ZONES:
            prev_cost, prev_time = self._dynamic[z]

            if prev_cost > 0 and prev_time > 0:
                elapsed = now - prev_time
                decay = math.exp(-0.693 * elapsed / self._decay_half_life)
                decayed = prev_cost * decay
            else:
                decayed = 0.0

            if new_dynamic[z] > decayed:
                self._dynamic[z] = (new_dynamic[z], now)
            else:
                self._dynamic[z] = (decayed, prev_time)

    def get_cost_map(self):
        now = time.monotonic()
        costs = {}

        for z in ZONES:
            static = self._static_cost.get(z, 0.5)

            dyn_cost, dyn_time = self._dynamic[z]

            if dyn_cost > 0 and dyn_time > 0:
                elapsed = now - dyn_time
                decay = math.exp(-0.693 * elapsed / self._decay_half_life)
                dyn_cost = dyn_cost * decay
            else:
                dyn_cost = 0.0

            costs[z] = static + dyn_cost

        # Apply road-drivability penalty: non-walkable zones get higher cost
        if self._road_state is not None:
            zone_drivable = self._road_state.get("zone_drivable", {})
            edge_rel = float(self._road_state.get("edge_reliability", 0.0))
            for z in ZONES:
                drivable = float(zone_drivable.get(z, 1.0))
                # Penalty scales with confidence (edge_reliability)
                penalty = (1.0 - drivable) * (0.5 + 0.4 * edge_rel)
                costs[z] += penalty

        return costs

    def get_cost_grid(self):
        return self._grid.get_cost_grid()

    def get_best_corridor(self):
        return self._corridor.select_best_corridor(
            self.get_cost_grid(),
            self.get_cost_map(),
            road_state=self._road_state,
        )

    def get_safest_direction(self):
        corridor = self.get_best_corridor()
        if corridor:
            return corridor.get("direction", self._min_cost_zone())
        return self._min_cost_zone()

    def _min_cost_zone(self):
        """Return lowest-cost zone, preferring center over edges on ties."""
        cost_map = self.get_cost_map()
        _ZONE_PRIORITY = ["center", "right", "left", "far right", "far left"]
        return min(_ZONE_PRIORITY, key=lambda z: cost_map.get(z, 999.0))

    def is_road_zone(self, zone):
        return zone == "center"