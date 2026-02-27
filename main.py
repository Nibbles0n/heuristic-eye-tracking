"""
Eye Cropper / Eye Tracker — Main entry point.

Modes:
  - Crop mode (default): Dark blob detection + crop boxes
  - Track mode (press 't'): Full anatomical tracking (iris circle, sclera, gaze)

Controls:
  q / ESC  = Quit
  t        = Toggle between Crop and Track mode
  d        = Toggle debug mode (shows raw candidates)
"""
import cv2
import sys
from detector import EyeDetector
from eye_tracker import EyeTracker


def main():
    print("Initializing detectors...")
    detector = EyeDetector()
    tracker = EyeTracker()

    # On macOS, index 0 is usually the FaceTime camera.
    print("Opening camera (index 0)...")
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Error: Could not open camera 0. Trying index 1...")
        cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print("Error: Could not open any camera.")
            sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Eye Cropper Started.")
    print("  Press 't' to toggle Track mode (anatomical eye tracking)")
    print("  Press 'd' to toggle debug mode")
    print("  Press 'q' or ESC to quit")

    debug_mode = False
    use_tracker = True  # Start in tracker mode to show off the new feature

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        if use_tracker:
            output = tracker.process_frame(frame, debug_mode=debug_mode)
            mode_label = "TRACK"
        else:
            output = detector.process_frame(frame, debug_mode=debug_mode)
            mode_label = "CROP"

        # Mode label
        cv2.putText(output, f"[{mode_label}]", (output.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

        cv2.imshow('Eye Tracker', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("Quitting...")
            break
        elif key == ord('t'):
            use_tracker = not use_tracker
            print(f"Mode: {'TRACK' if use_tracker else 'CROP'}")
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug: {'ON' if debug_mode else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Extra pump for macOS cleanup
    print("Camera released.")


if __name__ == "__main__":
    main()
