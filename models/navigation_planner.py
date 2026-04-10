"""
Layer 5 — Navigation Decision Engine

Rule-based planner that consumes the temporal scene graph and
the cost map from SceneMemory to output a single navigation
instruction per frame cycle.
"""


class NavigationPlanner:
    """Produces one navigation instruction from temporal objects + cost map."""

    # Instruction constants — the smoothing layer uses these for comparison
    STOP       = "Stop immediately."
    MOVE_LEFT  = "Move left."
    MOVE_RIGHT = "Move right."
    EDGE_LEFT  = "Stay on the left edge."
    EDGE_RIGHT = "Stay on the right edge."
    CAUTION    = "Proceed with caution. Obstacles on both sides."
    FORWARD    = "Continue forward."

    def decide(self, temporal_objects, cost_map, safest_zone):
        """
        Args:
            temporal_objects: list from TemporalReasoner.update()
            cost_map: dict {zone: cost} from SceneMemory.get_cost_map()
            safest_zone: str from SceneMemory.get_safest_direction()

        Returns:
            (instruction_text: str, urgency: str)
            urgency is one of "critical", "warning", "info"
        """
        if not temporal_objects:
            return self.FORWARD, "info"

        # ------------------------------------------------------------------
        # Rule 1 — Immediate hazard: very close + approaching
        # ------------------------------------------------------------------
        for obj in temporal_objects:
            if (obj["distance"] == "very close"
                    and obj["motion"] == "approaching"):
                return self.STOP, "critical"

        # ------------------------------------------------------------------
        # Rule 2 — Very close object (any motion)
        # ------------------------------------------------------------------
        for obj in temporal_objects:
            if obj["distance"] == "very close":
                return self._avoid(obj, cost_map, safest_zone), "critical"

        # ------------------------------------------------------------------
        # Rule 3 — Center zone is blocked → suggest alternative
        # ------------------------------------------------------------------
        center_objects = [
            o for o in temporal_objects
            if o["zone"] == "center"
            and o["distance"] in ("very close", "near", "moderate distance")
        ]
        if center_objects:
            return self._suggest_direction(cost_map, safest_zone), "warning"

        # ------------------------------------------------------------------
        # Rule 4 — Approaching objects from the side
        # ------------------------------------------------------------------
        approaching = [
            o for o in temporal_objects
            if o["motion"] == "approaching"
            and o["distance"] in ("near", "moderate distance")
        ]
        if approaching:
            obj = approaching[0]
            return self._avoid(obj, cost_map, safest_zone), "warning"

        # ------------------------------------------------------------------
        # Rule 5 — Objects on both sides → caution
        # ------------------------------------------------------------------
        left_zones  = {"left", "far left"}
        right_zones = {"right", "far right"}
        active_zones = {o["zone"] for o in temporal_objects
                        if o["distance"] in ("very close", "near", "moderate distance")}
        has_left  = bool(active_zones & left_zones)
        has_right = bool(active_zones & right_zones)
        if has_left and has_right:
            return self.CAUTION, "warning"

        # ------------------------------------------------------------------
        # Rule 6 — Road zone penalty (static prior)
        # ------------------------------------------------------------------
        if cost_map.get("center", 0) > 1.2:
            return self._suggest_direction(cost_map, safest_zone), "info"

        # ------------------------------------------------------------------
        # Default — path appears clear
        # ------------------------------------------------------------------
        return self.FORWARD, "info"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _avoid(self, obj, cost_map, safest_zone):
        """Generate an avoidance instruction away from the object's zone."""
        zone = obj.get("zone", "center")

        if zone in ("left", "far left"):
            return self.MOVE_RIGHT
        elif zone in ("right", "far right"):
            return self.MOVE_LEFT
        else:
            # Object in center — pick the cheapest side
            return self._suggest_direction(cost_map, safest_zone)

    @staticmethod
    def _suggest_direction(cost_map, safest_zone):
        """Pick a direction instruction based on the safest zone."""
        if safest_zone in ("far left", "left"):
            return NavigationPlanner.MOVE_LEFT
        elif safest_zone in ("far right", "right"):
            return NavigationPlanner.MOVE_RIGHT
        else:
            # Both sides roughly equal — default to right (Indian roads:
            # pedestrians usually walk on the left-facing-traffic side,
            # but we keep it generic)
            left_cost = cost_map.get("left", 1) + cost_map.get("far left", 1)
            right_cost = cost_map.get("right", 1) + cost_map.get("far right", 1)
            if left_cost <= right_cost:
                return NavigationPlanner.MOVE_LEFT
            return NavigationPlanner.MOVE_RIGHT
