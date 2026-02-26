"""
Eye Crop — Production (v2)
Pure-heuristic eye cropper. No AI, no models, just math.
Advanced: CLAHE + proximity weighting + pair scoring + gradient ring + Kalman filter.

Usage:
    python eye_crop.py          # Live camera feed with crop boxes
    python eye_crop.py --fps    # Show FPS counter

Press 'q' to quit.
"""
import cv2
import sys
import time
from detector import EyeDetector

def main():
    show_fps = '--fps' in sys.argv
    detector = EyeDetector()

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print("Error: No camera found.")
            sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    t0 = time.time()
    fc = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detector (debug_mode=False for clean output)
        _ = detector.process_frame(frame, debug_mode=False)
        
        # Draw crop boxes on full camera feed
        out = frame.copy()
        for (x1, y1, x2, y2) in detector.last_crop_boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)

        if show_fps:
            fc += 1
            el = time.time() - t0
            if el >= 1.0:
                fps = fc / el
                fc = 0
                t0 = time.time()
            cv2.putText(out, f"{fps:.0f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Eye Crop', out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
