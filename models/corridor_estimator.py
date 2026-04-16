class CorridorEstimator:
    def select_best_corridor(self, cost_grid, cost_map, road_state=None):
        if cost_grid is None and cost_map is None:
            return None

        zone_scores = {
            "far left": float(cost_map.get("far left", 1.0)),
            "left": float(cost_map.get("left", 1.0)),
            "center": float(cost_map.get("center", 1.0)),
            "right": float(cost_map.get("right", 1.0)),
            "far right": float(cost_map.get("far right", 1.0)),
        }

        if cost_grid is not None:
            # Map occupancy-grid columns into navigation zones.
            col_map = {
                "far left": [0],
                "left": [1],
                "center": [2],
                "right": [3],
                "far right": [4],
            }
            for zone, cols in col_map.items():
                vals = []
                for c in cols:
                    if 0 <= c < cost_grid.shape[1]:
                        vals.append(float(cost_grid[:, c].mean()))
                if vals:
                    zone_scores[zone] += 0.6 * (sum(vals) / len(vals))

        if road_state:
            zone_drivable = road_state.get("zone_drivable", {})
            edge_reliability = float(road_state.get("edge_reliability", 0.0))
            corridor_width = float(road_state.get("corridor_width", 0.0))
            proc_h, proc_w = road_state.get("proc_shape", (1, 1))
            width_ratio = corridor_width / max(float(proc_w), 1.0)

            for zone in zone_scores:
                drivable = float(zone_drivable.get(zone, 0.0))
                # Penalize non-drivable zones. Higher penalty when edge extraction is confident.
                zone_scores[zone] += (1.0 - drivable) * (0.7 + 0.5 * edge_reliability)

            # If road is narrow, avoid lateral extremes unless their score is clearly better.
            if width_ratio < 0.20:
                zone_scores["far left"] += 0.25
                zone_scores["far right"] += 0.25

        best_zone = min(zone_scores, key=zone_scores.get)
        return {
            "direction": best_zone,
            "score": zone_scores[best_zone],
            "zone_scores": zone_scores,
        }