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
    CAUTION    = "Proceed with caution."
    FORWARD    = "Continue forward."

    ROAD_THRESHOLD = 1.2

    def __init__(self, state=None):
        self.state = state

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
        # Rule 1 — Immediate hazard
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
            # Count hazards on each side
            left_risk = sum(
                o["risk"] for o in temporal_objects
                if o["zone"] in ("left", "far left")
            )

            right_risk = sum(
                o["risk"] for o in temporal_objects
                if o["zone"] in ("right", "far right")
            )

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
            else:
                return self.MOVE_LEFT, "warning"

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
        # Check this BEFORE far hazards rule because center blockage takes
        # precedence over individual distant objects
        # --------------------------------------------------------------
        if cost_map.get("center", 0) > self.ROAD_THRESHOLD:
            if safest_zone in ("left", "far left"):
                return self.EDGE_LEFT, "info"
            elif safest_zone in ("right", "far right"):
                return self.EDGE_RIGHT, "info"
            else:
                return self._suggest_direction(cost_map, safest_zone), "info"

        # --------------------------------------------------------------
        # Rule 5.5 — Distant hazards (far + single-side moderate distance)
        # For hazards that are either far OR only on one side at moderate distance,
        # prefer forward movement. Only suggest lateral movement if one side is
        # significantly safer.
        # (This rule only triggers if center penalty is not active)
        # --------------------------------------------------------------
        
        # Check for far hazards
        far_objects = [
            o for o in temporal_objects
            if o["distance"] == "far"
        ]
        
        # Check for single-side moderate hazards (only left OR only right blocked)
        moderate_objects = [
            o for o in temporal_objects
            if o["distance"] == "moderate distance"
        ]
        
        very_close_near_objects = [
            o for o in temporal_objects
            if o["distance"] in ("very close", "near")
        ]
        
        # Determine if we should apply distant-hazard preference
        left_moderate = [o for o in moderate_objects if o["zone"] in ("left", "far left")]
        right_moderate = [o for o in moderate_objects if o["zone"] in ("right", "far right")]
        left_far = [o for o in far_objects if o["zone"] in ("left", "far left")]
        right_far = [o for o in far_objects if o["zone"] in ("right", "far right")]
        
        # Apply rule if:
        # 1. Only far hazards exist, OR
        # 2. No very-close/near hazards AND only one side has moderate hazards (not both)
        should_apply_distant_rule = (
            (far_objects and not very_close_near_objects and not moderate_objects) or
            (moderate_objects and not very_close_near_objects and not (left_moderate and right_moderate))
        )
        
        if should_apply_distant_rule:
            # Combine far + moderate hazards for decision
            distant_objects = far_objects + moderate_objects
            
            left_cost = cost_map.get("left", 0.25) + cost_map.get("far left", 0.15)
            right_cost = cost_map.get("right", 0.25) + cost_map.get("far right", 0.15)
            
            # If cost difference is small, continue forward (less disruptive)
            if abs(left_cost - right_cost) < 0.35:
                return self.FORWARD, "info"
            
            # Both sides have hazards and cost difference is significant
            distant_objects = far_objects + moderate_objects
            left_risk = sum(
                o["risk"] for o in distant_objects
                if o["zone"] in ("left", "far left")
            )
            right_risk = sum(
                o["risk"] for o in distant_objects
                if o["zone"] in ("right", "far right")
            )
            
            if left_risk > right_risk:
                return self.MOVE_RIGHT, "info"
            elif right_risk > left_risk:
                return self.MOVE_LEFT, "info"
            # If equal, fall through to corridor

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

        if safest_zone in ("left", "far left"):
            return NavigationPlanner.MOVE_LEFT
        elif safest_zone in ("right", "far right"):
            return NavigationPlanner.MOVE_RIGHT
        else:
            left_cost = cost_map.get("left", 1) + cost_map.get("far left", 1)
            right_cost = cost_map.get("right", 1) + cost_map.get("far right", 1)

            return NavigationPlanner.MOVE_LEFT if left_cost < right_cost else NavigationPlanner.MOVE_RIGHT