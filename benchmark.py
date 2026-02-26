"""
Eye Cropper Benchmark
Compares our pure-heuristic eye detector against dlib's 68-point face landmark model.

Two key metrics:
  1. Capture Rate: % of frames where the AI eye center falls inside our crop box
  2. Area Reduction: % of total screen area eliminated by our crops

Controls:
  q / ESC  = Quit
  UP/DOWN  = Adjust crop_v_mult (vertical tightness)
  LEFT/RIGHT = Adjust crop_h_mult (horizontal tightness)
  [/]      = Adjust dark_percentile
  ,/.      = Adjust min_contrast

Run: uv run benchmark.py
"""
import cv2
import numpy as np
import dlib
import sys
import time
from detector import EyeDetector, DEFAULT_CONFIG

class AIGroundTruth:
    """Uses dlib's 68-point landmark model as the gold-standard eye detector."""
    
    def __init__(self, model_path="shape_predictor_68_face_landmarks.dat"):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)
        self.RIGHT_EYE = list(range(36, 42))
        self.LEFT_EYE = list(range(42, 48))
    
    def get_eye_centers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        centers = []
        if faces:
            shape = self.predictor(gray, faces[0])
            for eye_indices in [self.RIGHT_EYE, self.LEFT_EYE]:
                pts = [(shape.part(i).x, shape.part(i).y) for i in eye_indices]
                cx = int(np.mean([p[0] for p in pts]))
                cy = int(np.mean([p[1] for p in pts]))
                centers.append((cx, cy))
        return centers


class BenchmarkMetrics:
    def __init__(self, window=90):
        self.window = window
        self.capture_history = []
        self.area_history = []
    
    def update(self, ai_centers, crop_boxes, frame_shape):
        h, w = frame_shape[:2]
        total_area = h * w
        
        crop_area = sum((x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in crop_boxes)
        reduction = (1.0 - crop_area / total_area) * 100.0 if total_area > 0 else 0
        self.area_history.append(reduction)
        if len(self.area_history) > self.window:
            self.area_history.pop(0)
        
        if ai_centers:
            for (cx, cy) in ai_centers:
                captured = any(x1 <= cx <= x2 and y1 <= cy <= y2 for (x1, y1, x2, y2) in crop_boxes)
                self.capture_history.append(captured)
            if len(self.capture_history) > self.window * 2:
                self.capture_history = self.capture_history[-self.window * 2:]
    
    @property
    def capture_rate(self):
        return sum(self.capture_history) / len(self.capture_history) * 100.0 if self.capture_history else 0.0
    
    @property
    def area_reduction(self):
        return np.mean(self.area_history) if self.area_history else 0.0


def main():
    print("=== Eye Cropper Benchmark ===")
    
    config = {**DEFAULT_CONFIG}
    heuristic = EyeDetector(config)
    
    print("Loading dlib face landmark model...")
    try:
        ai = AIGroundTruth()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    metrics = BenchmarkMetrics(window=90)
    
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            sys.exit(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Benchmark running. Press 'q' to quit.")
    print("UP/DOWN = crop_v_mult | LEFT/RIGHT = crop_h_mult | [/] = dark_pct | ,/. = contrast")
    
    fps_time = time.time()
    fps_count = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run heuristic
        output = heuristic.process_frame(frame, debug_mode=True)
        crop_boxes = getattr(heuristic, 'last_crop_boxes', [])
        
        # Run AI ground truth
        ai_centers = ai.get_eye_centers(frame)
        
        # Update metrics
        metrics.update(ai_centers, crop_boxes, frame.shape)
        
        # Draw crop box outlines (cyan)
        for (x1, y1, x2, y2) in crop_boxes:
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        # Draw AI eye centers (bright green)
        for (cx, cy) in ai_centers:
            cv2.circle(output, (cx, cy), 8, (0, 255, 0), -1)
            cv2.circle(output, (cx, cy), 12, (0, 255, 0), 2)
            cv2.line(output, (cx - 15, cy), (cx + 15, cy), (0, 255, 0), 1)
            cv2.line(output, (cx, cy - 15), (cx, cy + 15), (0, 255, 0), 1)
        
        # FPS
        fps_count += 1
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            fps = fps_count / elapsed
            fps_count = 0
            fps_time = time.time()
        
        # --- Metrics Panel ---
        orig_h, orig_w = frame.shape[:2]
        panel_h = 100
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (orig_w, panel_h), (0, 0, 0), -1)
        output = cv2.addWeighted(overlay, 0.7, output, 0.3, 0)
        
        cr = metrics.capture_rate
        cr_color = (0, 255, 0) if cr > 80 else (0, 255, 255) if cr > 50 else (0, 0, 255)
        cv2.putText(output, f"Capture Rate: {cr:.1f}%", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cr_color, 2)
        bar_w = int(cr / 100 * 200)
        cv2.rectangle(output, (280, 10), (280 + bar_w, 30), cr_color, -1)
        cv2.rectangle(output, (280, 10), (480, 30), (100, 100, 100), 1)
        
        ar = metrics.area_reduction
        ar_color = (0, 255, 0) if ar > 80 else (0, 255, 255) if ar > 50 else (0, 0, 255)
        cv2.putText(output, f"Area Reduction: {ar:.1f}%", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ar_color, 2)
        bar_w2 = int(ar / 100 * 200)
        cv2.rectangle(output, (280, 40), (280 + bar_w2, 60), ar_color, -1)
        cv2.rectangle(output, (280, 40), (480, 60), (100, 100, 100), 1)
        
        cv2.putText(output, f"FPS: {fps:.0f}", (500, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output, f"AI: {len(ai_centers)} eyes", (500, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Current hyperparameter values
        hp_text = (f"v={config['crop_v_mult']:.1f}  h={config['crop_h_mult']:.1f}  "
                   f"dark={config['dark_percentile']}  contrast={config['min_contrast']}")
        cv2.putText(output, hp_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        
        cv2.imshow('Eye Cropper Benchmark', output)
        
        # --- Keyboard Controls ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == 0:  # UP arrow
            config["crop_v_mult"] = max(1.0, config["crop_v_mult"] - 0.5)
            heuristic.config["crop_v_mult"] = config["crop_v_mult"]
            print(f"crop_v_mult: {config['crop_v_mult']:.1f}")
        elif key == 1:  # DOWN arrow
            config["crop_v_mult"] += 0.5
            heuristic.config["crop_v_mult"] = config["crop_v_mult"]
            print(f"crop_v_mult: {config['crop_v_mult']:.1f}")
        elif key == 2:  # LEFT arrow
            config["crop_h_mult"] = max(1.0, config["crop_h_mult"] - 0.5)
            heuristic.config["crop_h_mult"] = config["crop_h_mult"]
            print(f"crop_h_mult: {config['crop_h_mult']:.1f}")
        elif key == 3:  # RIGHT arrow
            config["crop_h_mult"] += 0.5
            heuristic.config["crop_h_mult"] = config["crop_h_mult"]
            print(f"crop_h_mult: {config['crop_h_mult']:.1f}")
        elif key == ord('['):
            config["dark_percentile"] = max(1, config["dark_percentile"] - 1)
            heuristic.config["dark_percentile"] = config["dark_percentile"]
            print(f"dark_percentile: {config['dark_percentile']}")
        elif key == ord(']'):
            config["dark_percentile"] = min(50, config["dark_percentile"] + 1)
            heuristic.config["dark_percentile"] = config["dark_percentile"]
            print(f"dark_percentile: {config['dark_percentile']}")
        elif key == ord(','):
            config["min_contrast"] = max(1, config["min_contrast"] - 1)
            heuristic.config["min_contrast"] = config["min_contrast"]
            print(f"min_contrast: {config['min_contrast']}")
        elif key == ord('.'):
            config["min_contrast"] += 1
            heuristic.config["min_contrast"] = config["min_contrast"]
            print(f"min_contrast: {config['min_contrast']}")
    
    # Final report
    print(f"\n{'='*40}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'='*40}")
    print(f"  Capture Rate:   {metrics.capture_rate:.1f}%")
    print(f"  Area Reduction: {metrics.area_reduction:.1f}%")
    print(f"{'='*40}")
    print(f"  Final Config:")
    print(f"    crop_v_mult:     {config['crop_v_mult']:.1f}")
    print(f"    crop_h_mult:     {config['crop_h_mult']:.1f}")
    print(f"    dark_percentile: {config['dark_percentile']}")
    print(f"    min_contrast:    {config['min_contrast']}")
    print(f"{'='*40}")
    
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == "__main__":
    main()
