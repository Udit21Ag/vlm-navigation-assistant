"""Navigation Planning Module

Rule-based multi-layer planner that converts spatial hazards into navigation
commands (STOP, AVOID, MOVE, EDGE, FORWARD).

Implements 8 priority rules:
1. Time-to-collision check (STOP)
2. Very close hazards (AVOID)
3. Center-blocked with close hazards (MOVE)
4. Approaching motion (AVOID)
5. Both sides blocked (SUGGEST)
5.7. Road penalty enforcement (EDGE)
5.5. Distant hazards with balance check (FORWARD)
6. Corridor-based routing

Rules are evaluated in priority order; first match wins.
"""

# Distance priority helper
_DIST_ORDER = {
    "very close": 0,
    "near": 1,
    "moderate distance": 2,
    "far": 3,
}

class NavigationPlanner:

    STOP       = "Stop immediately."
    MOVE_LEFT  = "Move left."
    MOVE_RIGHT = "Move right."
    EDGE_LEFT  = "Stay on the left edge."
    EDGE_RIGHT = "Stay on the right edge."
    GAP_LEFT   = "Gap on the left, move left."
    GAP_RIGHT  = "Gap on the right, move right."
    FORWARD    = "Continue forward."

    ROAD_THRESHOLD = 1.2

    def __init__(self, state=None):
        self.state = state
    # Vehicle labels that warrant STOP even in crowd mode
    _VEHICLE_LABELS = {"car", "bus", "truck", "autorickshaw"}

    def decide(self, temporal_objects, cost_map, safest_zone, corridor=None):

        if not temporal_objects:
            return self.FORWARD, "info"

        temporal_objects = sorted(
            temporal_objects,
            key=lambda o: (
                _DIST_ORDER.get(o.get("distance", "far"), 3),
                0 if o.get("motion", "stationary") == "approaching" else 1
            )
        )

        # --------------------------------------------------------------
        # Crowd detection — walking through a crowd is normal, not an emergency
        # When many objects are close (pedestrians), navigate through gaps
        # instead of spamming STOP.
        # --------------------------------------------------------------
        total = len(temporal_objects)
        close_count = sum(
            1 for o in temporal_objects
            if o["distance"] in ("very close", "near")
        )
        is_crowd = total >= 8 and close_count >= total * 0.35

        if is_crowd:
            # In crowd mode: only STOP for vehicles, not pedestrians
            for obj in temporal_objects:
                if obj["label"] in self._VEHICLE_LABELS:
                    ttc = obj.get("ttc", float("inf"))
                    if obj["distance"] == "very close" and obj["motion"] == "approaching":
                        return self.STOP, "critical"
                    if ttc != float("inf") and ttc < 0.8:
                        return self.STOP, "critical"
                    if obj["distance"] == "very close":
                        return self._avoid(obj, cost_map, safest_zone), "warning"

            # Skip to gap-finding for pedestrians in crowd
            return self._find_crowd_gap(temporal_objects, cost_map, safest_zone)

        # --------------------------------------------------------------
        # Rule 1 — Immediate hazard (non-crowd mode)
        # --------------------------------------------------------------
        for obj in temporal_objects:
            ttc = obj.get("ttc", float("inf"))

            if ttc != float("inf") and ttc < 1.0:
                return self.STOP, "critical"

            if obj["distance"] == "very close" and obj["motion"] == "approaching":
                return self.STOP, "critical"

        # --------------------------------------------------------------
        # Rule 2 — Very close object
        # --------------------------------------------------------------
        for obj in temporal_objects:
            if obj["distance"] == "very close":
                return self._avoid(obj, cost_map, safest_zone), "critical"

        # --------------------------------------------------------------
        # Rule 3 — Center blocked
        # --------------------------------------------------------------
        center_objects = [
            o for o in temporal_objects
            if o["zone"] == "center"
            and o["distance"] in ("very close", "near", "moderate distance")
        ]
        if center_objects:
            # Bias against traffic classes
            def traffic_penalty(obj):
                return 2.0 if obj["label"] in ["car", "motorcycle", "bus", "truck"] else 1.0

            left_risk = sum(
                o["risk"] * traffic_penalty(o)
                for o in temporal_objects
                if o["zone"] in ("left", "far left")
            )

            right_risk = sum(
                o["risk"] * traffic_penalty(o)
                for o in temporal_objects
                if o["zone"] in ("right", "far right")
            )

            if left_risk > right_risk:
                return self.MOVE_RIGHT, "warning"
            elif right_risk > left_risk:
                return self.MOVE_LEFT, "warning"
            else:
                # Tie-break using cost map (avoids systematic left-bias)
                return self._suggest_direction(cost_map, safest_zone), "warning"

        # --------------------------------------------------------------
        # Rule 4 — Approaching objects
        # --------------------------------------------------------------
        for obj in temporal_objects:
            if obj["motion"] == "approaching" and obj["distance"] in ("near", "moderate distance"):
                return self._avoid(obj, cost_map, safest_zone), "warning"

        # --------------------------------------------------------------
        # Rule 5 — Both sides blocked
        # --------------------------------------------------------------
        left_zones  = {"left", "far left"}
        right_zones = {"right", "far right"}

        active_zones = {
            o["zone"] for o in temporal_objects
            if o["distance"] in ("very close", "near", "moderate distance")
        }

        if (active_zones & left_zones) and (active_zones & right_zones):
            return self._suggest_direction(cost_map, safest_zone), "warning"

        # --------------------------------------------------------------
        # Rule 5.7 — Road penalty (PRIORITY: center is blocked)
        # Only fires when there are actual close/near/moderate hazards.
        # Far-only objects should NOT trigger edge navigation.
        # --------------------------------------------------------------
        has_proximate_hazards = any(
            o["distance"] in ("very close", "near", "moderate distance")
            for o in temporal_objects
        )

        if has_proximate_hazards and cost_map.get("center", 0) > self.ROAD_THRESHOLD:
            # Use cost_map comparison directly to avoid safest_zone left-bias
            left_cost = cost_map.get("left", 0.25) + cost_map.get("far left", 0.15)
            right_cost = cost_map.get("right", 0.25) + cost_map.get("far right", 0.15)

            if left_cost < right_cost - 0.1:
                return self.EDGE_LEFT, "info"
            elif right_cost < left_cost - 0.1:
                return self.EDGE_RIGHT, "info"
            else:
                # Both sides roughly equal — pick the side with fewer hazards
                return self._suggest_direction(cost_map, safest_zone), "info"

        # --------------------------------------------------------------
        # Rule 5.5 — Distant/far hazards → find gaps and direct user
        # Analyzes zone distribution to find the least occupied side
        # and gives specific directional guidance.
        # --------------------------------------------------------------
        far_objects = [o for o in temporal_objects if o["distance"] == "far"]
        moderate_objects = [o for o in temporal_objects if o["distance"] == "moderate distance"]
        very_close_near = [o for o in temporal_objects if o["distance"] in ("very close", "near")]

        if not very_close_near and (far_objects or moderate_objects):
            all_distant = far_objects + moderate_objects

            # Count objects per zone (weighted by risk)
            zone_load = {"far left": 0.0, "left": 0.0, "center": 0.0,
                         "right": 0.0, "far right": 0.0}
            for o in all_distant:
                z = o.get("zone", "center")
                if z in zone_load:
                    zone_load[z] += max(o.get("risk", 0.1), 0.1)

            # Aggregate left vs right load
            left_load = zone_load["far left"] + zone_load["left"]
            right_load = zone_load["far right"] + zone_load["right"]
            center_load = zone_load["center"]
            total = left_load + right_load + center_load

            # If very few objects overall, just go forward
            if total < 0.5:
                return self.FORWARD, "info"

            # Find the gap — the side with significantly less load
            load_diff = left_load - right_load

            if load_diff > 0.3:
                # Left is more crowded → gap is on the right
                return self.GAP_RIGHT, "info"
            elif load_diff < -0.3:
                # Right is more crowded → gap is on the left
                return self.GAP_LEFT, "info"
            elif center_load > left_load and center_load > right_load:
                # Center is most crowded, pick least crowded side
                if left_load <= right_load:
                    return self.GAP_LEFT, "info"
                else:
                    return self.GAP_RIGHT, "info"
            else:
                # Roughly balanced — forward is fine if center isn't too crowded
                if center_load < 1.0:
                    return self.FORWARD, "info"
                # Center crowded but sides balanced — pick right (pedestrian side)
                return self.GAP_RIGHT, "info"

        # --------------------------------------------------------------
        # Rule 6 — (moved to before Rule 5.5 as Rule 5.7)
        # --------------------------------------------------------------

        if corridor:
            direction = corridor.get("direction", "center")

            if direction == "far left":
                instruction = self.EDGE_LEFT
            elif direction == "left":
                instruction = self.MOVE_LEFT
            elif direction == "center":
                instruction = self.FORWARD
            elif direction == "right":
                instruction = self.MOVE_RIGHT
            elif direction == "far right":
                instruction = self.EDGE_RIGHT
            else:
                instruction = self.FORWARD

            return instruction, "info"

        return self.FORWARD, "info"

    # --------------------------------------------------------------
    def _avoid(self, obj, cost_map, safest_zone):
        zone = obj.get("zone", "center")

        if zone in ("left", "far left"):
            return self.MOVE_RIGHT
        elif zone in ("right", "far right"):
            return self.MOVE_LEFT
        else:
            return self._suggest_direction(cost_map, safest_zone)

    # --------------------------------------------------------------
    @staticmethod
    def _suggest_direction(cost_map, safest_zone):
        """Pick left or right using cost_map comparison (avoids safest_zone left-bias)."""
        left_cost = cost_map.get("left", 1) + cost_map.get("far left", 1)
        right_cost = cost_map.get("right", 1) + cost_map.get("far right", 1)

        if left_cost < right_cost - 0.05:
            return NavigationPlanner.MOVE_LEFT
        elif right_cost < left_cost - 0.05:
            return NavigationPlanner.MOVE_RIGHT
        else:
            # Costs nearly equal — use safest_zone as tiebreaker
            if safest_zone in ("left", "far left"):
                return NavigationPlanner.MOVE_LEFT
            elif safest_zone in ("right", "far right"):
                return NavigationPlanner.MOVE_RIGHT
            else:
                # True tie — default to right (safer pedestrian side in left-hand-traffic countries)
                return NavigationPlanner.MOVE_RIGHT

    # --------------------------------------------------------------
    @staticmethod
    def _find_crowd_gap(temporal_objects, cost_map, safest_zone):
        """Find the least occupied zone in a crowd and direct user there."""

        # Count objects per zone, weighted by proximity (closer = heavier)
        zone_load = {"far left": 0.0, "left": 0.0, "center": 0.0,
                     "right": 0.0, "far right": 0.0}
        _DIST_WEIGHT = {"very close": 1.0, "near": 0.7, "moderate distance": 0.4, "far": 0.15}

        for o in temporal_objects:
            z = o.get("zone", "center")
            if z in zone_load:
                w = _DIST_WEIGHT.get(o.get("distance", "far"), 0.15)
                zone_load[z] += w

        left_load = zone_load["far left"] + zone_load["left"]
        right_load = zone_load["far right"] + zone_load["right"]
        center_load = zone_load["center"]

        # Find least crowded side
        load_diff = left_load - right_load

        if load_diff > 0.5:
            return NavigationPlanner.GAP_RIGHT, "warning"
        elif load_diff < -0.5:
            return NavigationPlanner.GAP_LEFT, "warning"
        elif center_load > left_load and center_load > right_load:
            # Center most crowded — pick the emptier side
            if left_load <= right_load:
                return NavigationPlanner.GAP_LEFT, "warning"
            else:
                return NavigationPlanner.GAP_RIGHT, "warning"
        else:
            # All zones roughly equal — suggest the side with lower cost_map
            left_cost = cost_map.get("left", 1) + cost_map.get("far left", 1)
            right_cost = cost_map.get("right", 1) + cost_map.get("far right", 1)
            if left_cost < right_cost:
                return NavigationPlanner.GAP_LEFT, "warning"
            return NavigationPlanner.GAP_RIGHT, "warning"