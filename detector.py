import cv2
import numpy as np
import time

# ─── Tunable Hyperparameters ────────────────────────────────────
DEFAULT_CONFIG = {
    # Detection
    "dark_percentile": 10,
    "min_area": 5,
    "max_area": 1200,
    "max_area_fraction": 0.02,    # 2% of frame area — anything bigger is not a pupil
    "min_circularity": 0.15,      # Lenient — let spatial filters do the work
    "min_contrast": 23,
    "contrast_ring": 2.5,
    
    # Spatial filters
    "center_weight": 0.3,         # How much center proximity boosts score (0=off, 1=strong)
    "pair_distance_min": 20,      # Min px distance between eye pair (at proc resolution)
    "pair_distance_max": 150,     # Max px distance between eye pair
    "pair_boost": 1.5,            # Score multiplier for paired candidates
    "size_consistency_penalty": 0.5,  # Penalty applied when eye sizes differ (0=off, 1=harsh)
    
    # Gradient ring
    "gradient_weight": 0.2,       # How much gradient ring score contributes
    
    # Cropping
    "crop_v_mult": 3.5,
    "crop_h_mult": 6.5,
    
    # Kalman filter
    "kalman_process_noise": 1e-2,
    "kalman_measurement_noise": 1.0,
    
    # Processing
    "processing_width": 400,
}


class KalmanTracker:
    """Lightweight 2D Kalman filter tracking position + velocity."""
    
    def __init__(self, x, y, r, process_noise=1e-2, meas_noise=1.0):
        # State: [x, y, vx, vy, r]
        self.state = np.array([x, y, 0.0, 0.0, r], dtype=np.float64)
        
        # State transition: position += velocity
        self.F = np.eye(5)
        self.F[0, 2] = 1.0  # x += vx
        self.F[1, 3] = 1.0  # y += vy
        
        # Measurement matrix: we observe [x, y, r]
        self.H = np.zeros((3, 5))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 4] = 1.0
        
        # Covariance
        self.P = np.eye(5) * 10.0
        self.Q = np.eye(5) * process_noise  # Process noise
        self.R = np.eye(3) * meas_noise      # Measurement noise
        
        self.age = 0          # Frames since creation
        self.missed = 0       # Consecutive frames without measurement

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


class EyeDetector:
    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.proc_w = self.config["processing_width"]
        self.trackers = []  # List of KalmanTracker
        
        # CLAHE
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        
        # Performance metrics
        self.fps = 0
        self.last_time = time.time()
        self.frame_count = 0
        self.jitter = 0
        self.last_raw_centers = []

    def find_pupil_candidates(self, gray_frame, roi_mask=None):
        search_gray = gray_frame
        if roi_mask is not None:
            search_gray = cv2.bitwise_and(gray_frame, roi_mask)

        blurred = cv2.GaussianBlur(search_gray, (9, 9), 0)

        if roi_mask is not None:
            vp = blurred[roi_mask > 0]
            lim = np.percentile(vp, self.config["dark_percentile"]) if vp.size > 0 else 0
        else:
            lim = np.percentile(blurred, self.config["dark_percentile"])

        _, thresh = cv2.threshold(blurred, lim, 255, cv2.THRESH_BINARY_INV)
        if roi_mask is not None:
            thresh = cv2.bitwise_and(thresh, roi_mask)

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = gray_frame.shape
        max_area_abs = self.config["max_area"]
        max_area_rel = self.config["max_area_fraction"] * h * w
        max_area = min(max_area_abs, max_area_rel)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.config["min_area"] or area > max_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > self.config["min_circularity"]:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                candidates.append({
                    "center": (int(x), int(y)),
                    "radius": int(max(3, radius)),
                    "circularity": circularity,
                })
        return candidates

    def score_candidates(self, gray_frame, candidates):
        """Score by contrast + proximity to center + gradient ring."""
        h, w = gray_frame.shape
        cx_frame, cy_frame = w / 2, h / 2
        ring = self.config["contrast_ring"]
        center_w = self.config["center_weight"]
        grad_w = self.config["gradient_weight"]

        scored = []
        for c in candidates:
            cx, cy = c["center"]
            r = c["radius"]
            outer_r = int(r * ring)

            if cx - outer_r < 0 or cx + outer_r >= w or cy - outer_r < 0 or cy + outer_r >= h:
                continue

            roi = gray_frame[cy - outer_r:cy + outer_r, cx - outer_r:cx + outer_r]
            if roi.size == 0:
                continue
            Y, X = np.ogrid[:roi.shape[0], :roi.shape[1]]
            dist = np.sqrt((X - outer_r) ** 2 + (Y - outer_r) ** 2)

            inner_mask = dist <= r
            outer_mask = (dist > r) & (dist <= outer_r)

            inner_mean = np.mean(roi[inner_mask]) if np.any(inner_mask) else 255
            outer_mean = np.mean(roi[outer_mask]) if np.any(outer_mask) else 0

            contrast_score = outer_mean - inner_mean
            if contrast_score <= self.config["min_contrast"]:
                continue

            # Gradient ring: check mid-ring brightness is between inner and outer
            mid_r = int(r * (1 + ring) / 2)
            mid_mask = (dist > r) & (dist <= mid_r)
            mid_mean = np.mean(roi[mid_mask]) if np.any(mid_mask) else inner_mean
            gradient_ok = inner_mean < mid_mean < outer_mean
            gradient_bonus = grad_w * contrast_score if gradient_ok else 0

            # Proximity weighting: distance from frame center
            dist_to_center = np.sqrt((cx - cx_frame) ** 2 + (cy - cy_frame) ** 2)
            max_dist = np.sqrt(cx_frame ** 2 + cy_frame ** 2)
            proximity_score = 1.0 - (dist_to_center / max_dist)  # 1.0 at center, 0 at corner
            center_bonus = center_w * contrast_score * proximity_score

            total = contrast_score + gradient_bonus + center_bonus
            c["score"] = total
            c["contrast"] = contrast_score
            scored.append(c)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    def apply_pair_filters(self, scored):
        """Pair proximity + size consistency scoring."""
        if len(scored) < 2:
            return scored

        cfg = self.config
        best_pair = None
        best_pair_score = -1

        # Try all pairs to find the best eye pair
        for i in range(min(len(scored), 6)):
            for j in range(i + 1, min(len(scored), 6)):
                a, b = scored[i], scored[j]
                ax, ay = a["center"]
                bx, by = b["center"]
                dist = np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

                # Check distance is plausible for an eye pair
                if dist < cfg["pair_distance_min"] or dist > cfg["pair_distance_max"]:
                    continue

                # Size consistency: soft penalty based on radius difference
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
        
        # Fallback: return top 2
        return scored[:2]

    def get_crop_box(self, cx, cy, r, scale, orig_w, orig_h):
        """Returns (x1, y1, x2, y2) crop box in original resolution."""
        fx = int(cx / scale)
        fy = int(cy / scale)
        hv = int(r * self.config["crop_v_mult"] / scale)
        hh = int(r * self.config["crop_h_mult"] / scale)
        return (max(0, fx - hh), max(0, fy - hv), min(orig_w, fx + hh), min(orig_h, fy + hv))

    def process_frame(self, frame, debug_mode=False):
        current_time = time.time()
        self.frame_count += 1
        elapsed = current_time - self.last_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = current_time

        orig_h, orig_w, _ = frame.shape
        scale = self.proc_w / orig_w
        proc_h = int(orig_h * scale)

        small_frame = cv2.resize(frame, (self.proc_w, proc_h))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # CLAHE: normalize contrast across skin tones and lighting
        gray = self.clahe.apply(gray)

        # Adaptive ROI from trackers
        roi_mask = None
        if self.trackers:
            roi_mask = np.zeros_like(gray)
            for t in self.trackers:
                pos, r = t.predict()
                cv2.circle(roi_mask, pos, int(r * 4), 255, -1)

        # Find & score candidates
        candidates = self.find_pupil_candidates(gray, roi_mask)
        if not candidates and roi_mask is not None:
            candidates = self.find_pupil_candidates(gray, None)  # Fallback: full frame
        elif not self.trackers:
            candidates = self.find_pupil_candidates(gray, None)

        scored = self.score_candidates(gray, candidates)
        best_eyes = self.apply_pair_filters(scored)

        # Kalman update: match detections to trackers
        current_raw_centers = []
        used_trackers = set()
        
        for eye in best_eyes:
            ecx, ecy = eye["center"]
            er = eye["radius"]
            current_raw_centers.append((ecx, ecy))

            # Find closest tracker
            best_t = None
            min_dist = float('inf')
            for idx, t in enumerate(self.trackers):
                if idx in used_trackers:
                    continue
                d = np.sqrt((ecx - t.state[0]) ** 2 + (ecy - t.state[1]) ** 2)
                if d < 50 and d < min_dist:
                    min_dist = d
                    best_t = idx

            if best_t is not None:
                self.trackers[best_t].update(ecx, ecy, er)
                used_trackers.add(best_t)
            else:
                # New tracker
                t = KalmanTracker(ecx, ecy, er,
                                  self.config["kalman_process_noise"],
                                  self.config["kalman_measurement_noise"])
                self.trackers.append(t)
                used_trackers.add(len(self.trackers) - 1)

        # Remove stale trackers (missed > 10 frames)
        self.trackers = [t for t in self.trackers if t.missed < 10]
        # Keep at most 2 trackers (2 eyes)
        if len(self.trackers) > 2:
            self.trackers.sort(key=lambda t: t.missed)
            self.trackers = self.trackers[:2]

        # Jitter
        if self.last_raw_centers and current_raw_centers:
            dists = [min(np.sqrt((c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2) 
                        for p in self.last_raw_centers) for c in current_raw_centers]
            self.jitter = np.mean(dists)
        self.last_raw_centers = current_raw_centers

        # Build crop boxes from Kalman-smoothed positions
        crop_boxes = []
        for t in self.trackers:
            box = self.get_crop_box(t.state[0], t.state[1], t.radius, scale, orig_w, orig_h)
            crop_boxes.append(box)
        self.last_crop_boxes = crop_boxes

        # Build output
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for (x1, y1, x2, y2) in crop_boxes:
            mask[y1:y2, x1:x2] = 255

        output = np.zeros_like(frame)
        if debug_mode:
            output = cv2.addWeighted(frame, 0.2, output, 0.8, 0)
            for c in candidates:
                ccx, ccy = int(c["center"][0] / scale), int(c["center"][1] / scale)
                cv2.circle(output, (ccx, ccy), 3, (0, 0, 255), 1)

        output[mask == 255] = frame[mask == 255]

        info = f"FPS: {self.fps:.1f} | Jitter: {self.jitter:.2f} | Trackers: {len(self.trackers)}"
        cv2.putText(output, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return output
