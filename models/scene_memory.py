"""
Layer 4 — Scene Memory

Two-layer cost-map system that prevents unsafe decisions:

A. Static Prior  — road zones are always penalized.
B. Dynamic Hazard — populated from tracked objects and decays over time.

The combined cost map drives the NavigationPlanner's direction selection.
"""

import time
import math


# The five horizontal zones used throughout the pipeline
ZONES = ["far left", "left", "center", "right", "far right"]


class SceneMemory:
    """Persistent + short-term scene memory expressed as a 5-zone cost map."""

    def __init__(self):
        # -----------------------------------------------------------------
        # A. Static Prior — road/center zones are inherently riskier
        # -----------------------------------------------------------------
        self._static_cost = {
            "far left":  0.15,
            "left":      0.25,
            "center":    0.80,
            "right":     0.25,
            "far right": 0.15,
        }

        # -----------------------------------------------------------------
        # B. Dynamic Hazard Layer — {zone: (cost, last_update_time)}
        # -----------------------------------------------------------------
        self._dynamic = {z: (0.0, 0.0) for z in ZONES}

        # Decay half-life in seconds — cost halves every N seconds
        self._decay_half_life = 2.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, temporal_objects):
        """
        Refresh the dynamic hazard layer from the current temporal scene
        graph produced by `TemporalReasoner.update()`.

        Args:
            temporal_objects: list of TemporalObject dicts with keys
                zone, motion, distance, risk, label …
        """
        now = time.monotonic()

        # Reset dynamic costs for this tick — we'll re-accumulate
        new_dynamic = {z: 0.0 for z in ZONES}

        for obj in temporal_objects:
            zone = obj.get("zone", "center")
            if zone not in new_dynamic:
                zone = "center"

            motion = obj.get("motion", "stationary")
            distance = obj.get("distance", "far")

            # Base cost from distance
            dist_cost = {
                "very close": 1.0,
                "near":       0.7,
                "moderate distance": 0.4,
                "far":        0.1,
            }.get(distance, 0.2)

            # Motion multiplier
            motion_mult = {
                "approaching": 1.5,
                "crossing":    1.3,
                "stationary":  1.0,
                "receding":    0.4,
            }.get(motion, 1.0)

            cost = dist_cost * motion_mult
            new_dynamic[zone] = max(new_dynamic[zone], cost)

        # Merge with decayed previous dynamic state
        for z in ZONES:
            prev_cost, prev_time = self._dynamic[z]
            if prev_cost > 0 and prev_time > 0:
                elapsed = now - prev_time
                decay = math.exp(-0.693 * elapsed / self._decay_half_life)
                decayed = prev_cost * decay
            else:
                decayed = 0.0

            # Take the max of freshly-computed and decayed-old
            final = max(new_dynamic[z], decayed)
            self._dynamic[z] = (final, now)

    def get_cost_map(self):
        """
        Return the combined cost per zone.

        Returns:
            dict  {zone_name: float}   — higher = more dangerous
        """
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

        return costs

    def get_safest_direction(self):
        """Return the zone with the lowest combined cost."""
        costs = self.get_cost_map()
        return min(costs, key=costs.get)

    def is_road_zone(self, zone):
        """Heuristic: center zone is the road."""
        return zone == "center"
