"""
Air Canvas AI - Main Application Loop.
Features adaptive jitter smoothing, pinch-to-draw precision,
smart geometric shape auto-detection with anchor snapping, and open-palm clearing.
"""

import argparse
import sys
import time
import cv2
from src.hand_tracker import HandTracker
from src.canvas import Canvas
from src.ui_overlay import UIOverlay
from src.shape_recognizer import ShapeRecognizer


def parse_args():
    parser = argparse.ArgumentParser(description="Air Canvas AI - Gesture Drawing Application")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Window width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Window height (default: 720)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print("   AIR CANVAS AI - High-Precision Hand Drawing & Smart Shapes")
    print("=" * 65)
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
    recognizer = ShapeRecognizer(snap_threshold=28.0)

    blackboard_mode = False
    prev_time = time.time()
    palm_hold_start: float = 0.0

    # Shape grab & move state
    grabbed_shape_idx: Optional[int] = None
    last_grab_pos: Optional[Tuple[int, int]] = None
    hover_shape_idx: Optional[int] = None

    print("\n[INFO] Controls & Gestures:")
    print(" - PINCH (Thumb + Index together):")
    print("     * On an existing shape -> GRAB and MOVE the shape to another place!")
    print("     * On empty canvas -> DRAW a new stroke / shape")
    print("     * On top toolbar -> SELECT color or tool")
    print(" - RELEASE PINCH:")
    print("     * If moving shape -> PLACES shape at new position (auto-snaps to nearby anchors)")
    print("     * If drawing -> AUTO-RECOGNIZES shape (Line, Box, Circle, Triangle, Curve)")
    print(" - CONNECT SHAPES: Start or move near a corner/anchor to snap & connect")
    print(" - OPEN PALM (hold 1 sec): Wipe/clear canvas")
    print(" - 'z': Undo last shape")
    print(" - 'c': Clear canvas")
    print(" - 's': Save artwork to saved_drawings/")
    print(" - 'b': Toggle AR view / Blackboard view")
    print(" - '+/-': Increase / Decrease brush thickness")
    print(" - 'q': Quit application\n")

    try:
        while True:
            curr_time = time.time()
            dt = curr_time - prev_time
            fps = 1.0 / dt if dt > 0 else 0
            prev_time = curr_time

            success, frame = cap.read()
            if not success:
                print("[WARNING] Frame capture failed. Retrying...")
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            if canvas.width != w or canvas.height != h:
                canvas.resize(w, h)
                ui.update_dimensions(w)

            # Process hand landmarks and draw skeleton for clear visual feedback
            frame = tracker.find_hands(frame, draw=True)
            lm_list = tracker.find_positions(frame)

            mode_str = "STANDBY"
            is_pinch = False
            is_snapped = False
            target_pt = (0, 0)
            palm_clear_progress = 0.0
            hover_shape_idx = None

            if len(lm_list) >= 21:
                is_pinch, norm_dist, midpoint = tracker.is_pinching()
                is_palm = tracker.is_open_palm()

                # GESTURE: OPEN PALM CLEAR COUNTDOWN
                if is_palm and not is_pinch:
                    mode_str = "OPEN PALM (CLEARING...)"
                    if palm_hold_start == 0.0:
                        palm_hold_start = curr_time

                    elapsed = curr_time - palm_hold_start
                    palm_clear_progress = min(1.0, elapsed / 0.9)

                    if palm_clear_progress >= 1.0:
                        canvas.clear()
                        grabbed_shape_idx = None
                        last_grab_pos = None
                        ui.show_toast("Canvas Cleared via Open Palm!")
                        palm_hold_start = 0.0
                        palm_clear_progress = 0.0
                else:
                    palm_hold_start = 0.0
                    palm_clear_progress = 0.0

                    # Check magnetic snap to existing shape anchors
                    snapped_pt, is_snapped = recognizer.snap_point(midpoint, canvas.anchors)
                    target_pt = snapped_pt if is_snapped else midpoint

                    # 1. PINCH ACTIVE
                    if is_pinch:
                        # Top toolbar interaction
                        if midpoint[1] <= ui.header_height:
                            mode_str = "TOOLBAR CLICK"
                            ui.check_interaction(midpoint[0], midpoint[1], canvas)
                            canvas.current_stroke = []
                            grabbed_shape_idx = None
                            last_grab_pos = None
                        else:
                            # If we are already dragging a grabbed shape:
                            if grabbed_shape_idx is not None:
                                mode_str = f"MOVING {canvas.shapes[grabbed_shape_idx].shape_type.upper()}"
                                if last_grab_pos is not None:
                                    dx = target_pt[0] - last_grab_pos[0]
                                    dy = target_pt[1] - last_grab_pos[1]
                                    canvas.move_shape(grabbed_shape_idx, dx, dy)
                                last_grab_pos = target_pt

                            # If not dragging yet, check if pinch started ON an existing shape:
                            elif last_grab_pos is None and not canvas.is_eraser_active():
                                candidate_idx = canvas.find_shape_at(target_pt, margin=26.0)
                                if candidate_idx is not None:
                                    grabbed_shape_idx = candidate_idx
                                    last_grab_pos = target_pt
                                    stype = canvas.shapes[grabbed_shape_idx].shape_type.capitalize()
                                    mode_str = f"GRABBED {stype.upper()}"
                                    ui.show_toast(f"Grabbed {stype}! Drag to move")
                                else:
                                    # Regular pinch drawing
                                    mode_str = "PINCH DRAWING"
                                    canvas.add_stroke_point(target_pt)
                            else:
                                # Normal drawing/erasing
                                if canvas.is_eraser_active():
                                    mode_str = "ERASING"
                                    canvas.draw_direct(target_pt)
                                else:
                                    mode_str = "PINCH DRAWING"
                                    canvas.add_stroke_point(target_pt)

                    # 2. PINCH RELEASED
                    else:
                        # If we just dropped a grabbed shape:
                        if grabbed_shape_idx is not None:
                            snapped = canvas.snap_shape_anchors(grabbed_shape_idx, snap_threshold=recognizer.snap_threshold)
                            if snapped:
                                ui.show_toast("Shape Placed & Connected!")
                            else:
                                ui.show_toast("Shape Placed!")
                            grabbed_shape_idx = None
                            last_grab_pos = None
                            mode_str = "HOVER / NAVIGATION"

                        # If we were drawing a stroke and released pinch:
                        elif len(canvas.current_stroke) >= 2:
                            recognized = canvas.finish_stroke(recognizer)
                            if recognized:
                                shape_name = recognized.shape_type.capitalize()
                                ui.show_toast(f"Smart Shape: {shape_name}")
                            mode_str = "HOVER / NAVIGATION"
                        else:
                            canvas.current_stroke = []
                            # Check if hovering over any existing shape:
                            hover_shape_idx = canvas.find_shape_at(target_pt, margin=24.0)
                            if hover_shape_idx is not None:
                                s_type = canvas.shapes[hover_shape_idx].shape_type.upper()
                                mode_str = f"HOVER {s_type} (PINCH TO GRAB)"
                            else:
                                mode_str = "HOVER / NAVIGATION"

                        canvas.reset_point()
            else:
                # No hand in frame
                palm_hold_start = 0.0
                palm_clear_progress = 0.0
                if grabbed_shape_idx is not None:
                    grabbed_shape_idx = None
                    last_grab_pos = None
                if len(canvas.current_stroke) >= 2:
                    canvas.finish_stroke(recognizer)
                canvas.current_stroke = []
                canvas.reset_point()
                mode_str = "NO HAND DETECTED"


            # 1. Base display frame
            if blackboard_mode:
                display_frame = canvas.canvas.copy()
            else:
                display_frame = canvas.blend(frame)

            # 2. Render shape bounding box (when dragging or hovering over a shape)
            if grabbed_shape_idx is not None:
                canvas.render_shape_box(display_frame, grabbed_shape_idx, is_grabbed=True)
            elif hover_shape_idx is not None:
                canvas.render_shape_box(display_frame, hover_shape_idx, is_grabbed=False)

            # 3. Render shape connection anchors & live drawing stroke
            if len(lm_list) >= 21:
                canvas.render_anchors(display_frame, hover_pt=target_pt, snap_threshold=recognizer.snap_threshold)
                canvas.render_live_preview(display_frame)
                ui.draw_cursor(
                    frame=display_frame,
                    point=target_pt,
                    is_pinching=is_pinch,
                    is_snapped=is_snapped,
                    active_color=canvas.current_color,
                    brush_thickness=canvas.brush_thickness,
                )


            # 3. Render open-palm radial meter if active
            if palm_clear_progress > 0.0:
                ui.draw_palm_clear_progress(display_frame, palm_clear_progress)

            # 4. Render header toolbar and HUD
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
            elif key == ord("z"):
                if canvas.undo():
                    ui.show_toast("Undo Last Shape")
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
