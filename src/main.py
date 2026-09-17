"""
Air Canvas AI - Main Application Entry Point.
Runs the webcam loop, tracks hand gestures, handles UI interaction, and renders the canvas.
"""

import argparse
import sys
import time
import cv2
from src.hand_tracker import HandTracker
from src.canvas import Canvas
from src.ui_overlay import UIOverlay


def parse_args():
    parser = argparse.ArgumentParser(description="Air Canvas AI - Gesture Drawing Application")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Window width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Window height (default: 720)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("   AIR CANVAS AI - Computer Vision Hand Drawing")
    print("=" * 60)
    print("Initializing camera...")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"[ERROR] Could not access webcam at index {args.camera}.")
        print("Please check your camera permissions or specify another index via --camera <id>.")
        sys.exit(1)

    tracker = HandTracker(max_hands=1, detection_confidence=0.75, tracking_confidence=0.65)
    canvas = Canvas(width=args.width, height=args.height)
    ui = UIOverlay(width=args.width)

    blackboard_mode = False
    prev_time = time.time()

    print("\n[INFO] Controls & Gestures:")
    print(" - Index Finger ONLY: Draw on canvas")
    print(" - Index + Middle Fingers: Selection mode (choose colors/tools on top bar)")
    print(" - 'c': Clear canvas")
    print(" - 's': Save artwork to saved_drawings/")
    print(" - 'b': Toggle AR view / Blackboard view")
    print(" - '+/-': Increase / Decrease brush thickness")
    print(" - 'q': Quit application\n")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("[WARNING] Frame capture failed. Retrying...")
                continue

            # Mirror the frame horizontally for natural user experience
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Sync canvas dimensions if webcam resolution differs from requested
            if canvas.width != w or canvas.height != h:
                canvas.resize(w, h)
                ui.update_dimensions(w)

            # Process hand landmarks
            frame = tracker.find_hands(frame, draw=False)
            lm_list = tracker.find_positions(frame)

            mode_str = "STANDBY"

            if len(lm_list) >= 21:
                # Landmark 8: Index fingertip, Landmark 12: Middle fingertip
                x1, y1 = lm_list[8][1], lm_list[8][2]
                x2, y2 = lm_list[12][1], lm_list[12][2]

                fingers = tracker.fingers_up()

                # GESTURE 1: SELECTION MODE (Index & Middle fingers are both UP)
                if fingers[1] == 1 and fingers[2] == 1:
                    mode_str = "SELECTION / HOVER"
                    canvas.reset_point()

                    # Draw selection cursor indicator between the two fingertips
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.circle(frame, (cx, cy), 15, (255, 255, 0), cv2.FILLED)
                    cv2.circle(frame, (cx, cy), 20, (255, 255, 255), 2)

                    # Check if user clicked/hovered over the top toolbar buttons
                    ui.check_interaction(cx, cy, canvas)

                # GESTURE 2: DRAWING MODE (ONLY Index finger is UP)
                elif fingers[1] == 1 and fingers[2] == 0:
                    mode_str = "DRAWING"
                    # Visual feedback indicator at fingertip
                    tip_color = (200, 200, 200) if canvas.is_eraser_active() else canvas.current_color
                    brush_rad = canvas.eraser_thickness // 2 if canvas.is_eraser_active() else canvas.brush_thickness // 2
                    cv2.circle(frame, (x1, y1), max(brush_rad, 6), tip_color, cv2.FILLED)
                    cv2.circle(frame, (x1, y1), max(brush_rad, 6) + 2, (255, 255, 255), 1)

                    # Only draw if below the header toolbar
                    if y1 > ui.header_height:
                        canvas.draw((x1, y1))
                    else:
                        canvas.reset_point()

                # GESTURE 3: STANDBY (Other finger combinations or fist)
                else:
                    mode_str = "STANDBY"
                    canvas.reset_point()
            else:
                mode_str = "NO HAND DETECTED"
                canvas.reset_point()

            # Render output frame (either AR camera blend or pure blackboard)
            if blackboard_mode:
                display_frame = canvas.canvas.copy()
            else:
                display_frame = canvas.blend(frame)

            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # Render UI overlays (buttons, current tool, status hints, notifications)
            current_thickness = canvas.eraser_thickness if canvas.is_eraser_active() else canvas.brush_thickness
            ui.draw(
                frame=display_frame,
                current_mode=mode_str,
                active_color=canvas.current_color,
                brush_size=current_thickness,
                fps=fps,
            )

            cv2.imshow("Air Canvas AI", display_frame)

            # Keyboard shortcuts handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Quitting application.")
                break
            elif key == ord("c"):
                canvas.clear()
                ui.show_toast("Canvas Cleared!")
            elif key == ord("s"):
                saved_path = canvas.save_snapshot()
                ui.show_toast("Snapshot Saved!")
                print(f"[INFO] Artwork saved to: {saved_path}")
            elif key == ord("b"):
                blackboard_mode = not blackboard_mode
                ui.show_toast("Blackboard View" if blackboard_mode else "AR View")
            elif key in (ord("+"), ord("=")):
                if canvas.is_eraser_active():
                    canvas.set_eraser_thickness(canvas.eraser_thickness + 5)
                else:
                    canvas.set_brush_thickness(canvas.brush_thickness + 2)
            elif key in (ord("-"), ord("_")):
                if canvas.is_eraser_active():
                    canvas.set_eraser_thickness(canvas.eraser_thickness - 5)
                else:
                    canvas.set_brush_thickness(canvas.brush_thickness - 2)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Cleaned up resources and exited.")


if __name__ == "__main__":
    main()
