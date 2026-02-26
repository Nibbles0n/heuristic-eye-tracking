import cv2
import sys
from detector import EyeDetector

def main():
    print("Initializing detector...")
    detector = EyeDetector()
    
    # On macOS, index 0 is usually the FaceTime camera.
    # We specify CAP_AVFOUNDATION for better reliability on Mac.
    print("Opening camera (index 0)...")
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    
    if not cap.isOpened():
        print("Error: Could not open camera 0. Trying index 1...")
        cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print("Error: Could not open any camera.")
            sys.exit(1)

    # Set some properties for better responsiveness
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Eye Cropper Started. Press 'q' to quit.")
    print("Note: If no window appears, check if another app is using the camera.")
    
    frame_count = 0
    debug_mode = True # Start with debug mode on so user sees what's happening
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
            
        frame_count += 1
        output = detector.process_frame(frame, debug_mode=debug_mode)
        
        # Show output
        cv2.imshow('Eye Cropper Output', output)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
        elif key == 27: # ESC
            break
            
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1) # Extra pump for macOS cleanup
    print("Camera released.")

if __name__ == "__main__":
    main()
