"""
Anatomical Eye Tracker — Pure CV, No AI
Detects pupil, iris circle, sclera extent, eyelid bounds, and gaze direction.

Pipeline:
  1. Pupil detection (dark blob + relative area filter)
  2. Iris circle fitting (radial ray casting + RANSAC)
  3. Sclera extent (horizontal gradient scan)
  4. Eyelid bounds (vertical gradient scan)
  5. Gaze estimation (relative positions)
"""
import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional

# ─── Configuration ──────────────────────────────────────────────
DEFAULT_TRACKER_CONFIG = {
    # Processing
    "processing_width": 400,

    # Pupil detection
    "dark_percentile": 10,
    "min_area": 5,
    "max_area": 1200,
    "max_area_fraction": 0.02,       # 2% of frame = definitely not a pupil
    "min_circularity": 0.15,
    "min_contrast": 20,
    "contrast_ring": 2.5,

    # Spatial filters
    "center_weight": 0.3,
    "pair_distance_min": 15,
    "pair_distance_max": 160,
    "pair_boost": 1.5,
    "size_consistency_penalty": 0.5,

    # Iris fitting
    "iris_num_rays": 36,             # Rays cast outward from pupil
    "iris_max_search": 30,           # Max pixels to search along each ray
    "iris_min_radius": 4,            # Min plausible iris radius (at proc res)
    "iris_max_radius": 22,           # Max plausible iris radius (at proc res)
    "iris_ransac_iters": 50,         # RANSAC iterations for circle fitting
    "iris_ransac_thresh": 3.0,       # RANSAC inlier distance threshold (px)
    "iris_min_inlier_ratio": 0.35,   # Min fraction of rays that must agree
    "iris_gradient_min": 8,          # Min gradient to count as iris-sclera boundary

    # Sclera detection
    "sclera_max_search": 120,        # Max pixels to search for eye corners
    "sclera_strip_height": 7,        # Vertical strip height for averaging
    "sclera_gradient_min": 10,       # Min gradient magnitude for corner detection
    "sclera_min_extent": 3,          # Min sclera extent beyond iris (px)

    # Eyelid detection
    "eyelid_search_mult": 1.5,       # Search ± this × iris_radius vertically
    "eyelid_gradient_min": 10,       # Min gradient for eyelid edge

    # Sclera validation (nose-lock rejection)
    "sclera_validation": True,       # Enable sclera brightness check
    "sclera_bright_min": 0.25,       # Min fraction of surround pixels that must be "bright"
    "sclera_contrast_ratio": 1.25,   # Min contrast ratio (surround/iris)

    # Position smoothing (EMA)
    "position_ema_alpha": 0.35,      # Lower = smoother but laggier (0.0=frozen, 1.0=raw)
    "radius_ema_alpha": 0.25,        # Radius smoothing (even more stable)

    # Kalman filter
    "kalman_process_noise": 1e-2,
    "kalman_measurement_noise": 1.0,

    # Gradient ring (scoring)
    "gradient_weight": 0.2,
}


# ─── Data Classes ───────────────────────────────────────────────
@dataclass
class EyeModel:
    """Complete anatomical model of one eye."""
    # Pupil
    pupil_center: tuple  # (x, y) in processing coords
    pupil_radius: int

    # Iris (fitted circle)
    iris_center: Optional[tuple] = None  # (cx, cy)
    iris_radius: Optional[int] = None
    iris_confidence: float = 0.0         # inlier ratio from RANSAC

    # Sclera extent (eye corners)
    left_corner: Optional[tuple] = None   # (x, y)
    right_corner: Optional[tuple] = None  # (x, y)

    # Eyelid bounds
    top_eyelid: Optional[int] = None      # y coordinate
    bottom_eyelid: Optional[int] = None   # y coordinate

    # Gaze
    gaze_x: float = 0.0    # -1.0 (left) to +1.0 (right)
    gaze_y: float = 0.0    # -1.0 (up) to +1.0 (down)

    # Score from pupil detection
    score: float = 0.0


# ─── Kalman Tracker ─────────────────────────────────────────────
class KalmanTracker:
    """Lightweight 2D Kalman filter tracking position + velocity + radius."""

    def __init__(self, x, y, r, process_noise=1e-2, meas_noise=1.0):
        self.state = np.array([x, y, 0.0, 0.0, r], dtype=np.float64)
        self.F = np.eye(5)
        self.F[0, 2] = 1.0  # x += vx
        self.F[1, 3] = 1.0  # y += vy
        self.H = np.zeros((3, 5))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 4] = 1.0
        self.P = np.eye(5) * 10.0
        self.Q = np.eye(5) * process_noise
        self.R = np.eye(3) * meas_noise
        self.age = 0
        self.missed = 0

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.missed += 1
        return self.state[:2].astype(int), self.state[4]

    def update(self, x, y, r):
        z = np.array([x, y, r], dtype=np.float64)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ (z - self.H @ self.state)
        self.P = (np.eye(5) - K @ self.H) @ self.P
        self.missed = 0

    @property
    def pos(self):
        return (int(self.state[0]), int(self.state[1]))

    @property
    def radius(self):
        return max(3, self.state[4])


# ─── Main Tracker ───────────────────────────────────────────────
class EyeTracker:
    """
    Anatomical eye tracker: finds pupil → fits iris circle → detects sclera
    extent → finds eyelid bounds → estimates gaze direction.
    """

    def __init__(self, config=None):
        self.config = {**DEFAULT_TRACKER_CONFIG, **(config or {})}
        self.proc_w = self.config["processing_width"]
        self.trackers = []
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

        # Performance
        self.fps = 0
        self.last_time = time.time()
        self.frame_count = 0

        # EMA smoothed positions per tracker (keyed by tracker id)
        # Each entry: {"iris_center", "iris_radius", "left_corner", "right_corner",
        #              "top_eyelid", "bottom_eyelid", "pupil_center"}
        self._smooth = {}  # tracker_id -> dict of smoothed values
        self._tracker_ids = {}  # tracker obj id -> stable int id
        self._next_id = 0

        # Last results
        self.last_models = []  # List[EyeModel]

    # ═══════════════════════════════════════════════════════════
    # Stage 1: Pupil Detection
    # ═══════════════════════════════════════════════════════════
    def _find_pupil_candidates(self, gray, roi_mask=None):
        """Find dark blob candidates that might be pupils."""
        cfg = self.config
        search = cv2.bitwise_and(gray, roi_mask) if roi_mask is not None else gray
        blurred = cv2.GaussianBlur(search, (9, 9), 0)

        # Adaptive percentile threshold
        if roi_mask is not None:
            valid_px = blurred[roi_mask > 0]
            lim = np.percentile(valid_px, cfg["dark_percentile"]) if valid_px.size > 0 else 0
        else:
            lim = np.percentile(blurred, cfg["dark_percentile"])

        _, thresh = cv2.threshold(blurred, lim, 255, cv2.THRESH_BINARY_INV)
        if roi_mask is not None:
            thresh = cv2.bitwise_and(thresh, roi_mask)

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = gray.shape
        max_area_abs = cfg["max_area"]
        max_area_rel = cfg["max_area_fraction"] * h * w
        max_area = min(max_area_abs, max_area_rel)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < cfg["min_area"] or area > max_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > cfg["min_circularity"]:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                candidates.append({
                    "center": (int(x), int(y)),
                    "radius": int(max(3, radius)),
                    "circularity": circularity,
                })
        return candidates

    def _score_candidates(self, gray, candidates):
        """Score by contrast + center proximity + gradient ring."""
        h, w = gray.shape
        cx_frame, cy_frame = w / 2, h / 2
        cfg = self.config
        ring = cfg["contrast_ring"]
        center_w = cfg["center_weight"]
        grad_w = cfg["gradient_weight"]

        scored = []
        for c in candidates:
            cx, cy = c["center"]
            r = c["radius"]
            outer_r = int(r * ring)

            if cx - outer_r < 0 or cx + outer_r >= w or cy - outer_r < 0 or cy + outer_r >= h:
                continue

            roi = gray[cy - outer_r:cy + outer_r, cx - outer_r:cx + outer_r]
            if roi.size == 0:
                continue
            Y, X = np.ogrid[:roi.shape[0], :roi.shape[1]]
            dist = np.sqrt((X - outer_r) ** 2 + (Y - outer_r) ** 2)

            inner_mask = dist <= r
            outer_mask = (dist > r) & (dist <= outer_r)

            inner_mean = np.mean(roi[inner_mask]) if np.any(inner_mask) else 255
            outer_mean = np.mean(roi[outer_mask]) if np.any(outer_mask) else 0

            contrast_score = outer_mean - inner_mean
            if contrast_score <= cfg["min_contrast"]:
                continue

            # Gradient ring bonus
            mid_r = int(r * (1 + ring) / 2)
            mid_mask = (dist > r) & (dist <= mid_r)
            mid_mean = np.mean(roi[mid_mask]) if np.any(mid_mask) else inner_mean
            gradient_ok = inner_mean < mid_mean < outer_mean
            gradient_bonus = grad_w * contrast_score if gradient_ok else 0

            # Center proximity
            dist_to_center = np.sqrt((cx - cx_frame) ** 2 + (cy - cy_frame) ** 2)
            max_dist = np.sqrt(cx_frame ** 2 + cy_frame ** 2)
            proximity_score = 1.0 - (dist_to_center / max_dist)
            center_bonus = center_w * contrast_score * proximity_score

            total = contrast_score + gradient_bonus + center_bonus
            c["score"] = total
            c["contrast"] = contrast_score
            scored.append(c)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    def _pick_best_pair(self, scored):
        """Find the best eye pair from scored candidates."""
        if len(scored) < 2:
            return scored

        cfg = self.config
        best_pair = None
        best_pair_score = -1

        for i in range(min(len(scored), 6)):
            for j in range(i + 1, min(len(scored), 6)):
                a, b = scored[i], scored[j]
                ax, ay = a["center"]
                bx, by = b["center"]
                dist = np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

                if dist < cfg["pair_distance_min"] or dist > cfg["pair_distance_max"]:
                    continue

                ra, rb = a["radius"], b["radius"]
                size_ratio = min(ra, rb) / max(ra, rb) if max(ra, rb) > 0 else 1.0
                size_penalty = 1.0 - cfg["size_consistency_penalty"] * (1.0 - size_ratio)
                pair_score = (a["score"] + b["score"]) * cfg["pair_boost"] * size_penalty

                if pair_score > best_pair_score:
                    best_pair_score = pair_score
                    best_pair = (i, j)

        if best_pair:
            i, j = best_pair
            return [scored[i], scored[j]]
        return scored[:2]

    # ═══════════════════════════════════════════════════════════
    # Stage 2: Iris Circle Fitting
    # ═══════════════════════════════════════════════════════════
    def _fit_iris_circle(self, gray, pupil_cx, pupil_cy, pupil_r):
        """
        Cast rays outward from pupil center, find iris-sclera boundary
        via gradient peaks, then RANSAC-fit a circle.
        """
        cfg = self.config
        h, w = gray.shape
        num_rays = cfg["iris_num_rays"]
        max_search = cfg["iris_max_search"]
        min_r = cfg["iris_min_radius"]
        max_r = cfg["iris_max_radius"]

        # Cast rays at evenly spaced angles
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        boundary_points = []

        for angle in angles:
            dx = np.cos(angle)
            dy = np.sin(angle)

            # Sample brightness along the ray starting just outside pupil
            start = max(pupil_r - 1, 2)
            samples = []
            coords = []
            for dist in range(start, start + max_search):
                sx = int(pupil_cx + dx * dist)
                sy = int(pupil_cy + dy * dist)
                if 0 <= sx < w and 0 <= sy < h:
                    samples.append(float(gray[sy, sx]))
                    coords.append((sx, sy, dist))
                else:
                    break

            if len(samples) < 5:
                continue

            # Compute gradient (brightness increase = iris→sclera boundary)
            samples_arr = np.array(samples, dtype=np.float64)
            # Smooth a bit to avoid noise spikes
            if len(samples_arr) >= 5:
                kernel_size = 3
                samples_smooth = np.convolve(samples_arr, np.ones(kernel_size) / kernel_size, mode='valid')
                grad = np.diff(samples_smooth)
                offset = kernel_size // 2  # offset from smoothing
            else:
                grad = np.diff(samples_arr)
                offset = 0

            if len(grad) == 0:
                continue

            # Find the FIRST strong positive gradient (iris→sclera boundary)
            # Using first-strong-peak instead of global-max prevents overshooting
            # into the brow/eyelid where there might be an even bigger gradient.
            grad_thresh = cfg["iris_gradient_min"]
            peak_idx = -1
            peak_val = 0
            for gi, gv in enumerate(grad):
                if gv >= grad_thresh:
                    peak_idx = gi
                    peak_val = gv
                    break

            # Fallback: if no peak above threshold, use global max
            if peak_idx < 0:
                peak_idx = np.argmax(grad)
                peak_val = grad[peak_idx]

            # Must be a meaningful gradient
            if peak_val < 5:
                continue

            # Map back to coordinates
            coord_idx = peak_idx + offset
            if coord_idx < len(coords):
                sx, sy, dist_from_center = coords[coord_idx]
                # Check distance is plausible for an iris
                if min_r <= dist_from_center <= max_r:
                    boundary_points.append((sx, sy))

        if len(boundary_points) < 6:
            return None, None, 0.0

        # RANSAC circle fit
        points = np.array(boundary_points, dtype=np.float64)
        best_circle = None
        best_inliers = 0
        n_pts = len(points)
        thresh = cfg["iris_ransac_thresh"]

        for _ in range(cfg["iris_ransac_iters"]):
            # Sample 3 random points
            idx = np.random.choice(n_pts, 3, replace=False)
            p1, p2, p3 = points[idx]

            # Fit circle through 3 points
            circle = self._circle_from_3_points(p1, p2, p3)
            if circle is None:
                continue

            cx, cy, r = circle
            if r < min_r or r > max_r:
                continue

            # Count inliers
            dists = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
            inliers = np.sum(np.abs(dists - r) < thresh)

            if inliers > best_inliers:
                best_inliers = inliers
                best_circle = circle

        if best_circle is None:
            return None, None, 0.0

        inlier_ratio = best_inliers / n_pts
        if inlier_ratio < cfg["iris_min_inlier_ratio"]:
            return None, None, inlier_ratio

        cx, cy, r = best_circle

        # Refine: refit using all inliers
        dists = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
        inlier_mask = np.abs(dists - r) < thresh
        inlier_pts = points[inlier_mask]
        if len(inlier_pts) >= 3:
            refined = self._least_squares_circle(inlier_pts)
            if refined is not None:
                cx, cy, r = refined

        return (int(cx), int(cy)), int(max(min_r, r)), inlier_ratio

    @staticmethod
    def _circle_from_3_points(p1, p2, p3):
        """Compute circle passing through 3 points. Returns (cx, cy, r) or None."""
        ax, ay = p1
        bx, by = p2
        cx, cy = p3

        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-10:
            return None

        ux = ((ax * ax + ay * ay) * (by - cy) +
              (bx * bx + by * by) * (cy - ay) +
              (cx * cx + cy * cy) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx - bx) +
              (bx * bx + by * by) * (ax - cx) +
              (cx * cx + cy * cy) * (bx - ax)) / d

        r = np.sqrt((ax - ux) ** 2 + (ay - uy) ** 2)
        return (ux, uy, r)

    @staticmethod
    def _least_squares_circle(points):
        """Algebraic least-squares circle fit. Returns (cx, cy, r) or None."""
        x = points[:, 0]
        y = points[:, 1]
        # Build system: x^2 + y^2 + Dx + Ey + F = 0
        A = np.column_stack([x, y, np.ones(len(x))])
        b = -(x ** 2 + y ** 2)
        try:
            result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None

        D, E, F = result
        cx = -D / 2
        cy = -E / 2
        r_sq = cx ** 2 + cy ** 2 - F
        if r_sq <= 0:
            return None
        return (cx, cy, np.sqrt(r_sq))

    # ═══════════════════════════════════════════════════════════
    # Sclera Validation (nose-lock rejection)
    # ═══════════════════════════════════════════════════════════
    def _validate_sclera(self, gray, iris_cx, iris_cy, iris_r):
        """
        Check that the area just outside the iris is bright (sclera-like).
        Nostrils are dark circles surrounded by darker skin — no sclera.
        Uses contrast ratio: surround must be significantly brighter than iris center.
        Returns True if this looks like a real eye.
        """
        if not self.config["sclera_validation"]:
            return True

        h, w = gray.shape
        bright_min_frac = self.config["sclera_bright_min"]

        # Sample an annular ring from iris_r to iris_r * 1.8
        outer_r = int(iris_r * 1.8)
        inner_r = iris_r

        # Build a bounding box around the annulus
        x1 = max(0, iris_cx - outer_r)
        y1 = max(0, iris_cy - outer_r)
        x2 = min(w, iris_cx + outer_r)
        y2 = min(h, iris_cy + outer_r)

        if x2 - x1 < 4 or y2 - y1 < 4:
            return False

        roi = gray[y1:y2, x1:x2].astype(np.float64)
        cy_local = iris_cy - y1
        cx_local = iris_cx - x1

        Y, X = np.ogrid[:roi.shape[0], :roi.shape[1]]
        dist = np.sqrt((X - cx_local) ** 2 + (Y - cy_local) ** 2)

        # Get iris interior brightness (the dark part)
        iris_mask = dist <= inner_r
        iris_pixels = roi[iris_mask]
        if iris_pixels.size < 5:
            return False
        iris_mean = np.mean(iris_pixels)

        # Annular mask: between iris edge and 1.8× iris radius
        annular_mask = (dist > inner_r) & (dist <= outer_r)
        annular_pixels = roi[annular_mask]

        if annular_pixels.size < 10:
            return False

        # The surround must be significantly brighter than the iris interior.
        # For real eyes: sclera (~180+) vs iris (~40-100) → ratio ~2-4×
        # For nostrils: skin (~80-140) vs nostril (~20-60) → ratio can be 2-3× too
        # So we also check absolute brightness: sclera is genuinely BRIGHT,
        # not just brighter-than-iris.
        surround_mean = np.mean(annular_pixels)

        # Require surround to be significantly brighter than iris
        if iris_mean < 1:
            iris_mean = 1  # avoid division by zero
        contrast_ratio = surround_mean / iris_mean
        min_ratio = self.config["sclera_contrast_ratio"]

        # Check that a good fraction of surround pixels are brighter than iris
        bright_count = np.sum(annular_pixels > iris_mean * 1.15)
        bright_fraction = bright_count / annular_pixels.size

        return contrast_ratio >= min_ratio and bright_fraction >= bright_min_frac

    # ═══════════════════════════════════════════════════════════
    # Stage 3: Sclera Extent
    # ═══════════════════════════════════════════════════════════
    def _find_sclera_extent(self, gray, iris_cx, iris_cy, iris_r):
        """
        Scan horizontally from iris edges to find eye corners.
        Uses gradient-based detection (skin-color invariant).
        """
        cfg = self.config
        h, w = gray.shape
        max_search = cfg["sclera_max_search"]
        strip_h = cfg["sclera_strip_height"]
        grad_min = cfg["sclera_gradient_min"]
        min_ext = cfg["sclera_min_extent"]

        # Sample a horizontal strip at iris center height
        y_top = max(0, iris_cy - strip_h // 2)
        y_bot = min(h, iris_cy + strip_h // 2 + 1)
        if y_top >= y_bot:
            return None, None

        strip = gray[y_top:y_bot, :].astype(np.float64)
        row_avg = np.mean(strip, axis=0)  # Average vertically → 1D profile

        # --- Left corner ---
        left_corner = None
        search_start = iris_cx - iris_r
        search_end = max(0, iris_cx - iris_r - max_search)

        if search_start > search_end + min_ext:
            segment = row_avg[search_end:search_start]
            if len(segment) > 3:
                # Smooth
                k = 3
                seg_smooth = np.convolve(segment, np.ones(k) / k, mode='valid')
                grad = np.diff(seg_smooth)
                # We're scanning right-to-left conceptually; in array, rightmost = iris edge
                # Largest positive gradient = brightness increasing toward iris = sclera→skin boundary
                # Actually we want the point where brightness drops going away from iris
                # = negative gradient going left from iris = positive gradient in array (left to right)
                if len(grad) > 0:
                    # Find peak NEGATIVE gradient (bright sclera → darker skin going left)
                    # In array order (left→right), sclera is on the right, skin on left
                    # So skin→sclera = positive gradient going left→right
                    # The corner = largest positive gradient peak
                    peak_idx = np.argmax(np.abs(grad))
                    if np.abs(grad[peak_idx]) >= grad_min:
                        left_x = search_end + peak_idx + k // 2
                        left_corner = (int(left_x), iris_cy)

        # --- Right corner ---
        right_corner = None
        search_start_r = iris_cx + iris_r
        search_end_r = min(w, iris_cx + iris_r + max_search)

        if search_end_r > search_start_r + min_ext:
            segment = row_avg[search_start_r:search_end_r]
            if len(segment) > 3:
                k = 3
                seg_smooth = np.convolve(segment, np.ones(k) / k, mode='valid')
                grad = np.diff(seg_smooth)
                if len(grad) > 0:
                    # Sclera→skin going right = negative gradient peak
                    peak_idx = np.argmax(np.abs(grad))
                    if np.abs(grad[peak_idx]) >= grad_min:
                        right_x = search_start_r + peak_idx + k // 2
                        right_corner = (int(right_x), iris_cy)

        return left_corner, right_corner

    # ═══════════════════════════════════════════════════════════
    # Stage 4: Eyelid Bounds
    # ═══════════════════════════════════════════════════════════
    def _find_eyelid_bounds(self, gray, iris_cx, iris_cy, iris_r):
        """
        Scan vertically from iris center to find top and bottom eyelid bounds.
        Uses gradient-based detection.
        """
        cfg = self.config
        h, w = gray.shape
        search_extent = int(iris_r * cfg["eyelid_search_mult"])
        grad_min = cfg["eyelid_gradient_min"]

        # Sample a vertical strip at iris center, a few pixels wide for noise reduction
        strip_w = 5
        x_left = max(0, iris_cx - strip_w // 2)
        x_right = min(w, iris_cx + strip_w // 2 + 1)
        if x_left >= x_right:
            return None, None

        strip = gray[:, x_left:x_right].astype(np.float64)
        col_avg = np.mean(strip, axis=1)  # Average horizontally → 1D vertical profile

        # --- Top eyelid ---
        top_eyelid = None
        search_top = max(0, iris_cy - iris_r - search_extent)
        search_bot_top = iris_cy - iris_r  # just above iris

        if search_bot_top > search_top + 2:
            segment = col_avg[search_top:search_bot_top]
            if len(segment) > 3:
                k = 3
                seg_smooth = np.convolve(segment, np.ones(k) / k, mode='valid')
                grad = np.diff(seg_smooth)
                if len(grad) > 0:
                    # Going top→bottom: skin → sclera = big positive gradient
                    # Find the largest positive gradient (skin→eye transition)
                    peak_idx = np.argmax(grad)
                    if grad[peak_idx] >= grad_min:
                        top_y = search_top + peak_idx + k // 2
                        top_eyelid = int(top_y)

        # --- Bottom eyelid ---
        bottom_eyelid = None
        search_top_bot = iris_cy + iris_r   # just below iris
        search_bot = min(h, iris_cy + iris_r + search_extent)

        if search_bot > search_top_bot + 2:
            segment = col_avg[search_top_bot:search_bot]
            if len(segment) > 3:
                k = 3
                seg_smooth = np.convolve(segment, np.ones(k) / k, mode='valid')
                grad = np.diff(seg_smooth)
                if len(grad) > 0:
                    # Going top→bottom: sclera → skin = big negative gradient
                    peak_idx = np.argmin(grad)
                    if grad[peak_idx] <= -grad_min:
                        bot_y = search_top_bot + peak_idx + k // 2
                        bottom_eyelid = int(bot_y)

        return top_eyelid, bottom_eyelid

    # ═══════════════════════════════════════════════════════════
    # Stage 5: Gaze Estimation
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def _compute_gaze(eye):
        """Compute gaze direction from the anatomical eye model."""
        # Horizontal gaze from sclera
        if eye.left_corner and eye.right_corner and eye.iris_center:
            lx = eye.left_corner[0]
            rx = eye.right_corner[0]
            eye_width = rx - lx
            if eye_width > 0:
                eye.gaze_x = ((eye.iris_center[0] - lx) / eye_width - 0.5) * 2.0

        # Vertical gaze from eyelid bounds
        if eye.top_eyelid is not None and eye.bottom_eyelid is not None and eye.iris_center:
            eye_height = eye.bottom_eyelid - eye.top_eyelid
            if eye_height > 0:
                eye.gaze_y = ((eye.iris_center[1] - eye.top_eyelid) / eye_height - 0.5) * 2.0

        # Finer gaze from pupil-within-iris offset
        if eye.iris_center and eye.iris_radius and eye.iris_radius > 0:
            pupil_offset_x = (eye.pupil_center[0] - eye.iris_center[0]) / eye.iris_radius
            pupil_offset_y = (eye.pupil_center[1] - eye.iris_center[1]) / eye.iris_radius
            # Blend sclera-based gaze with pupil-offset gaze
            if eye.left_corner and eye.right_corner:
                eye.gaze_x = 0.6 * eye.gaze_x + 0.4 * np.clip(pupil_offset_x * 2.0, -1, 1)
            else:
                eye.gaze_x = np.clip(pupil_offset_x * 2.0, -1, 1)
            if eye.top_eyelid is not None and eye.bottom_eyelid is not None:
                eye.gaze_y = 0.6 * eye.gaze_y + 0.4 * np.clip(pupil_offset_y * 2.0, -1, 1)
            else:
                eye.gaze_y = np.clip(pupil_offset_y * 2.0, -1, 1)

    # ═══════════════════════════════════════════════════════════
    # EMA Position Smoothing
    # ═══════════════════════════════════════════════════════════
    def _get_tracker_id(self, tracker):
        """Get or assign a stable ID to a tracker object."""
        obj_id = id(tracker)
        if obj_id not in self._tracker_ids:
            self._tracker_ids[obj_id] = self._next_id
            self._next_id += 1
        return self._tracker_ids[obj_id]

    def _smooth_eye_model(self, eye, tracker):
        """
        Apply EMA smoothing to the drawn position of eye components.
        Gaze values are NOT smoothed — they stay raw and responsive.
        """
        tid = self._get_tracker_id(tracker)
        alpha = self.config["position_ema_alpha"]
        r_alpha = self.config["radius_ema_alpha"]

        if tid not in self._smooth:
            # First frame for this tracker — initialize with raw values
            self._smooth[tid] = {
                "pupil_center": eye.pupil_center,
                "iris_center": eye.iris_center,
                "iris_radius": float(eye.iris_radius) if eye.iris_radius else None,
                "left_corner": eye.left_corner,
                "right_corner": eye.right_corner,
                "top_eyelid": float(eye.top_eyelid) if eye.top_eyelid is not None else None,
                "bottom_eyelid": float(eye.bottom_eyelid) if eye.bottom_eyelid is not None else None,
            }
            return eye

        s = self._smooth[tid]

        def ema_pt(old, new, a):
            if old is None:
                return new
            if new is None:
                return old
            return (old[0] * (1 - a) + new[0] * a, old[1] * (1 - a) + new[1] * a)

        def ema_val(old, new, a):
            if old is None:
                return new
            if new is None:
                return old
            return old * (1 - a) + new * a

        # Smooth positions
        s["pupil_center"] = ema_pt(s["pupil_center"], eye.pupil_center, alpha)
        s["iris_center"] = ema_pt(s["iris_center"], eye.iris_center, alpha)
        s["iris_radius"] = ema_val(s["iris_radius"],
                                    float(eye.iris_radius) if eye.iris_radius else None, r_alpha)
        s["left_corner"] = ema_pt(s["left_corner"], eye.left_corner, alpha)
        s["right_corner"] = ema_pt(s["right_corner"], eye.right_corner, alpha)
        s["top_eyelid"] = ema_val(s["top_eyelid"],
                                   float(eye.top_eyelid) if eye.top_eyelid is not None else None, alpha)
        s["bottom_eyelid"] = ema_val(s["bottom_eyelid"],
                                      float(eye.bottom_eyelid) if eye.bottom_eyelid is not None else None, alpha)

        # Write smoothed values back to eye model (gaze stays raw!)
        if s["pupil_center"]:
            eye.pupil_center = (int(s["pupil_center"][0]), int(s["pupil_center"][1]))
        if s["iris_center"]:
            eye.iris_center = (int(s["iris_center"][0]), int(s["iris_center"][1]))
        if s["iris_radius"] is not None:
            eye.iris_radius = int(s["iris_radius"])
        if s["left_corner"]:
            eye.left_corner = (int(s["left_corner"][0]), int(s["left_corner"][1]))
        if s["right_corner"]:
            eye.right_corner = (int(s["right_corner"][0]), int(s["right_corner"][1]))
        if s["top_eyelid"] is not None:
            eye.top_eyelid = int(s["top_eyelid"])
        if s["bottom_eyelid"] is not None:
            eye.bottom_eyelid = int(s["bottom_eyelid"])

        return eye

    # ═══════════════════════════════════════════════════════════
    # Kalman Tracking
    # ═══════════════════════════════════════════════════════════
    def _update_trackers(self, best_pupils):
        """Match detections to Kalman trackers, create/destroy as needed."""
        used_trackers = set()

        for pupil in best_pupils:
            px, py = pupil["center"]
            pr = pupil["radius"]

            best_t = None
            min_dist = float('inf')
            for idx, t in enumerate(self.trackers):
                if idx in used_trackers:
                    continue
                d = np.sqrt((px - t.state[0]) ** 2 + (py - t.state[1]) ** 2)
                if d < 50 and d < min_dist:
                    min_dist = d
                    best_t = idx

            if best_t is not None:
                self.trackers[best_t].update(px, py, pr)
                used_trackers.add(best_t)
            else:
                t = KalmanTracker(px, py, pr,
                                  self.config["kalman_process_noise"],
                                  self.config["kalman_measurement_noise"])
                self.trackers.append(t)
                used_trackers.add(len(self.trackers) - 1)

        # Remove stale trackers — also clean up smooth state
        surviving = [t for t in self.trackers if t.missed < 10]
        dead_ids = set()
        for t in self.trackers:
            if t not in surviving:
                obj_id = id(t)
                if obj_id in self._tracker_ids:
                    dead_ids.add(self._tracker_ids[obj_id])
                    del self._tracker_ids[obj_id]
        for tid in dead_ids:
            self._smooth.pop(tid, None)

        self.trackers = surviving
        if len(self.trackers) > 2:
            self.trackers.sort(key=lambda t: t.missed)
            self.trackers = self.trackers[:2]

    # ═══════════════════════════════════════════════════════════
    # Drawing / Visualization
    # ═══════════════════════════════════════════════════════════
    def _draw_eye_model(self, frame, eye, scale):
        """Draw the anatomical eye model on the frame."""
        # Scale from processing coords to original frame coords
        def s(pt):
            return (int(pt[0] / scale), int(pt[1] / scale))

        def sr(r):
            return int(r / scale)

        # Iris circle (green)
        if eye.iris_center and eye.iris_radius:
            cv2.circle(frame, s(eye.iris_center), sr(eye.iris_radius), (0, 255, 0), 2, cv2.LINE_AA)

        # Pupil dot (red, small fixed size)
        cv2.circle(frame, s(eye.pupil_center), 3, (0, 0, 255), -1)

        # Sclera lines (yellow)
        if eye.iris_center and eye.iris_radius:
            iris_left_edge = (eye.iris_center[0] - eye.iris_radius, eye.iris_center[1])
            iris_right_edge = (eye.iris_center[0] + eye.iris_radius, eye.iris_center[1])

            if eye.left_corner:
                cv2.line(frame, s(eye.left_corner), s(iris_left_edge), (0, 255, 255), 1, cv2.LINE_AA)
            if eye.right_corner:
                cv2.line(frame, s(iris_right_edge), s(eye.right_corner), (0, 255, 255), 1, cv2.LINE_AA)

        # Eyelid lines (magenta)
        if eye.iris_center and eye.iris_radius:
            iris_top = (eye.iris_center[0], eye.iris_center[1] - eye.iris_radius)
            iris_bottom = (eye.iris_center[0], eye.iris_center[1] + eye.iris_radius)

            if eye.top_eyelid is not None:
                top_pt = (eye.iris_center[0], eye.top_eyelid)
                cv2.line(frame, s(iris_top), s(top_pt), (255, 0, 255), 1, cv2.LINE_AA)
            if eye.bottom_eyelid is not None:
                bot_pt = (eye.iris_center[0], eye.bottom_eyelid)
                cv2.line(frame, s(iris_bottom), s(bot_pt), (255, 0, 255), 1, cv2.LINE_AA)

        # Gaze indicator (cyan crosshair near the eye)
        if eye.iris_center:
            gc = s(eye.iris_center)
            gx_off = int(eye.gaze_x * 20)
            gy_off = int(eye.gaze_y * 20)
            gaze_pt = (gc[0] + gx_off, gc[1] + gy_off)
            cv2.circle(frame, gaze_pt, 3, (255, 255, 0), -1)
            cv2.line(frame, gc, gaze_pt, (255, 255, 0), 1, cv2.LINE_AA)

    # ═══════════════════════════════════════════════════════════
    # Main Processing
    # ═══════════════════════════════════════════════════════════
    def process_frame(self, frame, debug_mode=False):
        """
        Process a single frame:  pupil → iris → sclera → eyelid → gaze.
        Returns the annotated frame.
        """
        # FPS tracking
        current_time = time.time()
        self.frame_count += 1
        elapsed = current_time - self.last_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = current_time

        orig_h, orig_w = frame.shape[:2]
        scale = self.proc_w / orig_w
        proc_h = int(orig_h * scale)

        small = cv2.resize(frame, (self.proc_w, proc_h))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)

        # ROI from existing trackers
        roi_mask = None
        if self.trackers:
            roi_mask = np.zeros_like(gray)
            for t in self.trackers:
                pos, r = t.predict()
                cv2.circle(roi_mask, tuple(pos), int(r * 5), 255, -1)

        # Stage 1: Find pupils
        candidates = self._find_pupil_candidates(gray, roi_mask)
        if not candidates and roi_mask is not None:
            candidates = self._find_pupil_candidates(gray, None)
        elif not self.trackers:
            candidates = self._find_pupil_candidates(gray, None)

        scored = self._score_candidates(gray, candidates)
        best_pupils = self._pick_best_pair(scored)

        # Update Kalman trackers
        self._update_trackers(best_pupils)

        # Build eye models from tracked positions
        eye_models = []
        for t in self.trackers:
            px, py = t.pos
            pr = int(t.radius)

            eye = EyeModel(
                pupil_center=(px, py),
                pupil_radius=pr,
            )

            # Stage 2: Fit iris circle
            iris_center, iris_radius, confidence = self._fit_iris_circle(gray, px, py, pr)
            if iris_center and iris_radius:
                # Sclera validation: reject if no bright surround (nose-lock prevention)
                if not self._validate_sclera(gray, iris_center[0], iris_center[1], iris_radius):
                    # This dark blob is NOT an eye — skip full analysis
                    continue

                eye.iris_center = iris_center
                eye.iris_radius = iris_radius
                eye.iris_confidence = confidence

                # Stage 3: Find sclera extent
                left_corner, right_corner = self._find_sclera_extent(
                    gray, iris_center[0], iris_center[1], iris_radius
                )
                eye.left_corner = left_corner
                eye.right_corner = right_corner

                # Stage 4: Find eyelid bounds
                top_lid, bottom_lid = self._find_eyelid_bounds(
                    gray, iris_center[0], iris_center[1], iris_radius
                )
                eye.top_eyelid = top_lid
                eye.bottom_eyelid = bottom_lid

                # Stage 5: Gaze (computed on RAW positions — stays responsive)
                self._compute_gaze(eye)

                # Apply EMA smoothing to DRAWN positions only
                eye = self._smooth_eye_model(eye, t)

            eye_models.append(eye)

        self.last_models = eye_models

        # Build output frame
        if debug_mode:
            output = frame.copy()
            # Draw all raw candidates as small red dots
            for c in candidates:
                ccx, ccy = int(c["center"][0] / scale), int(c["center"][1] / scale)
                cv2.circle(output, (ccx, ccy), 2, (0, 0, 128), -1)
        else:
            output = frame.copy()

        # Draw eye models
        for eye in eye_models:
            self._draw_eye_model(output, eye, scale)

        # HUD
        gaze_str = ""
        if eye_models:
            avg_gx = np.mean([e.gaze_x for e in eye_models])
            avg_gy = np.mean([e.gaze_y for e in eye_models])
            gaze_str = f" | Gaze: ({avg_gx:+.2f}, {avg_gy:+.2f})"

        iris_info = ""
        for i, e in enumerate(eye_models):
            if e.iris_center and e.iris_radius:
                iris_info += f" | Eye{i}: r={e.iris_radius} conf={e.iris_confidence:.0%}"

        info = f"FPS: {self.fps:.0f} | Eyes: {len(eye_models)}{gaze_str}{iris_info}"
        cv2.putText(output, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

        return output
