"""Road / Walkable-Space Detection Module

Estimates drivable (walkable) road surface from the MiDaS depth map using
geometric heuristics — no additional neural-network inference required.

Algorithm:
1. Ground-plane reference — median depth of the bottom 15% of each column.
2. Inlier detection — pixels in the lower 60% whose depth is within
   ±tolerance of the ground reference are marked "walkable."
3. Object exclusion — detected object bounding boxes are masked out so
   obstacles standing on the road aren't counted as walkable.
4. Morphological cleanup — open + close to reduce noise.
5. Zone-level drivable fractions — 5-zone breakdown (far left → far right).
6. Free-corridor detection — widest contiguous walkable band.
7. Edge reliability — Canny + Hough on the lower frame for curb/boundary
   confidence.

Typical overhead: ~2-4 ms per frame on CPU.
"""

import cv2
import numpy as np


# Zone boundaries (same ratios as SpatialReasoner)
_ZONE_BOUNDS = {
    "far left":  (0.00, 0.15),
    "left":      (0.15, 0.35),
    "center":    (0.35, 0.65),
    "right":     (0.65, 0.85),
    "far right": (0.85, 1.00),
}


class RoadDetector:
    """Depth-based walkable-area estimator."""

    def __init__(self, ground_ratio=0.15, analysis_ratio=0.60,
                 depth_tolerance=0.12, min_walkable_fraction=0.30):
        """
        Args:
            ground_ratio:  Fraction of frame height (from bottom) used to
                           estimate the ground-plane reference depth.
            analysis_ratio: Fraction of frame height (from bottom) analysed
                           for walkable surface. The top 40% is sky/buildings
                           and is ignored.
            depth_tolerance: Max allowed deviation from the ground-plane
                           reference to still count as "walkable."
            min_walkable_fraction: Below this threshold a zone is considered
                           non-walkable for navigation purposes.
        """
        self._ground_ratio = ground_ratio
        self._analysis_ratio = analysis_ratio
        self._tol = depth_tolerance
        self._min_walkable = min_walkable_fraction

        # Morphological kernel (shared across calls)
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (7, 7)
        )

        # Temporal smoothing of zone drivable scores
        self._prev_zone_drivable = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, depth_map, frame=None, detections=None):
        """Analyse the depth map and return a road-state dict.

        Args:
            depth_map: 2-D ndarray (H, W) with values in [0, 1].
                       Higher = closer (MiDaS convention after normalisation).
            frame:     Original BGR frame (used only for edge analysis).
            detections: List of detection dicts with "bbox" keys. Bounding
                       boxes of detected objects are masked out so they don't
                       contaminate the walkable surface estimate.

        Returns:
            dict with keys:
                walkable_mask   – uint8 (H, W) binary mask, 255 = walkable
                zone_drivable   – {"far left": 0.0–1.0, …}
                corridor_width  – int, widest free-corridor in pixels
                corridor_center – int, x-centre of that corridor
                edge_reliability– float 0–1
                proc_shape      – (H, W) of the processed depth map
        """
        h, w = depth_map.shape[:2]

        # 1. Ground-plane reference (median depth per column, bottom slice)
        ground_row = max(1, int(h * (1 - self._ground_ratio)))
        ground_strip = depth_map[ground_row:, :]          # bottom 15%
        ground_ref = np.median(ground_strip, axis=0)      # (W,) vector

        # 2. Analysis region (bottom 60% of frame)
        analysis_row = max(0, int(h * (1 - self._analysis_ratio)))
        analysis_region = depth_map[analysis_row:, :]     # (H', W)

        # Broadcast ground_ref across rows of analysis region
        diff = np.abs(analysis_region - ground_ref[np.newaxis, :])

        # Walkable = depth close to ground reference AND reasonably close
        # to the camera (high depth value means close in normalised MiDaS).
        # Exclude sky/far background by requiring depth > 0.08
        walkable_region = (
            (diff < self._tol) & (analysis_region > 0.08)
        ).astype(np.uint8) * 255

        # 3. Mask out detected objects so boxes on the road are excluded
        if detections:
            for det in detections:
                bx1, by1, bx2, by2 = [int(v) for v in det["bbox"]]
                # Convert to analysis-region coordinates
                by1_local = max(0, by1 - analysis_row)
                by2_local = max(0, by2 - analysis_row)
                bx1 = max(0, min(bx1, w - 1))
                bx2 = max(0, min(bx2, w))
                if by2_local > by1_local:
                    walkable_region[by1_local:by2_local, bx1:bx2] = 0

        # 4. Morphological cleanup
        walkable_region = cv2.morphologyEx(
            walkable_region, cv2.MORPH_OPEN, self._morph_kernel
        )
        walkable_region = cv2.morphologyEx(
            walkable_region, cv2.MORPH_CLOSE, self._morph_kernel
        )

        # Build full-size mask (top portion = 0)
        walkable_mask = np.zeros((h, w), dtype=np.uint8)
        walkable_mask[analysis_row:, :] = walkable_region

        # 5. Zone-level drivable fractions
        zone_drivable = {}
        for zone, (x_lo, x_hi) in _ZONE_BOUNDS.items():
            col_lo = int(x_lo * w)
            col_hi = int(x_hi * w)
            strip = walkable_mask[analysis_row:, col_lo:col_hi]
            total_pixels = max(strip.size, 1)
            walkable_pixels = np.count_nonzero(strip)
            zone_drivable[zone] = walkable_pixels / total_pixels

        # Temporal smoothing (EMA α = 0.6 for responsiveness)
        if self._prev_zone_drivable is not None:
            for z in zone_drivable:
                zone_drivable[z] = (
                    0.6 * zone_drivable[z]
                    + 0.4 * self._prev_zone_drivable.get(z, zone_drivable[z])
                )
        self._prev_zone_drivable = dict(zone_drivable)

        # 6. Free-corridor detection (widest contiguous walkable band)
        corridor_width, corridor_center = self._find_corridor(
            walkable_mask, analysis_row
        )

        # 7. Edge reliability (Canny on lower frame)
        edge_reliability = 0.0
        if frame is not None:
            edge_reliability = self._compute_edge_reliability(
                frame, analysis_row
            )

        return {
            "walkable_mask": walkable_mask,
            "zone_drivable": zone_drivable,
            "corridor_width": corridor_width,
            "corridor_center": corridor_center,
            "edge_reliability": edge_reliability,
            "proc_shape": (h, w),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _find_corridor(walkable_mask, analysis_row):
        """Find the widest contiguous horizontal band of walkable pixels.

        We scan a representative row near the bottom third of the analysis
        region (where the road surface is most reliable).
        """
        h, w = walkable_mask.shape
        # Pick a representative row: 70% down in the analysis region
        target_row = min(
            h - 1,
            analysis_row + int((h - analysis_row) * 0.70)
        )

        row = walkable_mask[target_row, :]
        best_width = 0
        best_center = w // 2
        start = None

        for i in range(w):
            if row[i] > 0:
                if start is None:
                    start = i
            else:
                if start is not None:
                    run_len = i - start
                    if run_len > best_width:
                        best_width = run_len
                        best_center = start + run_len // 2
                    start = None

        # Handle run that reaches the right edge
        if start is not None:
            run_len = w - start
            if run_len > best_width:
                best_width = run_len
                best_center = start + run_len // 2

        return best_width, best_center

    @staticmethod
    def _compute_edge_reliability(frame, analysis_row):
        """Canny edge density in the lower frame as a proxy for road-boundary
        confidence.  More edges near the analysis-region border → more likely
        there are real curbs / walls / lane lines.

        Returns a float in [0, 1].
        """
        h, w = frame.shape[:2]
        lower = frame[analysis_row:, :]

        gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Focus on left and right 20% strips (where curbs typically sit)
        strip_w = max(1, int(w * 0.20))
        left_strip = edges[:, :strip_w]
        right_strip = edges[:, w - strip_w:]

        edge_count = np.count_nonzero(left_strip) + np.count_nonzero(right_strip)
        total_pixels = max(left_strip.size + right_strip.size, 1)

        density = edge_count / total_pixels
        # Map density [0, 0.15] → reliability [0, 1]
        reliability = min(density / 0.15, 1.0)
        return float(reliability)
