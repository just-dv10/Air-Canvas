"""
Air Canvas AI - Main Application Loop (2-Hand Edition).
Simultaneous two-hand tracking: draw, grab, scale, and connect shapes
with both hands independently in real time.
"""

import argparse
import sys
import time
import math
from typing import Optional, Tuple, Dict
import cv2
from src.hand_tracker import HandTracker
from src.canvas import Canvas
from src.ui_overlay import UIOverlay
from src.shape_recognizer import ShapeRecognizer


def parse_args():
    p = argparse.ArgumentParser(description="Air Canvas AI - 2-Hand Gesture Drawing")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width",  type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print("   AIR CANVAS AI  •  Two-Hand Edition")
    print("=" * 65)
    print("Initializing camera...")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at index {args.camera}.")
        sys.exit(1)

    tracker   = HandTracker(max_hands=2, detection_confidence=0.5, tracking_confidence=0.5)
    canvas    = Canvas(width=args.width, height=args.height)
    ui        = UIOverlay(width=args.width)
    recognizer = ShapeRecognizer(snap_threshold=28.0)

    blackboard_mode = False
    prev_time       = time.time()
    palm_hold_start: float = 0.0
    palm_clear_progress: float = 0.0

    # Per-hand drag state  {hand_label: grabbed_shape_idx or None}
    grabbed: Dict[str, Optional[int]] = {"Right": None, "Left": None}
    last_grab_pos: Dict[str, Optional[Tuple[int, int]]] = {"Right": None, "Left": None}

    # Two-hand pinch-to-scale state
    prev_two_hand_dist: Optional[float] = None
    scale_shape_idx:   Optional[int]    = None

    print("\n[INFO] Gesture Guide:")
    print("  🤏 Pinch on shape     → Grab & drag it anywhere!")
    print("  🤏 Pinch in air       → Draw a new stroke (auto-shapes on release)")
    print("  🤏 Both hands pinch   → Scale the grabbed shape (spread=bigger)")
    print("  🖐️  Open palm ~1s     → Clear canvas")
    print("  Z Undo | C Clear | S Save | B Blackboard | +/- Brush | Q Quit\n")

    try:
        while True:
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            if canvas.width != w or canvas.height != h:
                canvas.resize(w, h)
                ui.update_dimensions(w)

            # ── TRACKING ──────────────────────────────────────────────────
            frame = tracker.find_hands(frame, draw=True)
            hands = tracker.hands_data          # list of HandData objects
            num_hands = len(hands)

            mode_str = "STANDBY" if num_hands else "NO HAND DETECTED"
            palm_detected = any(hd.is_open_palm for hd in hands) and num_hands >= 1

            # ── OPEN PALM CLEAR ───────────────────────────────────────────
            if palm_detected and not any(hd.is_pinch for hd in hands):
                mode_str = "OPEN PALM (CLEARING...)"
                if palm_hold_start == 0.0:
                    palm_hold_start = now
                elapsed = now - palm_hold_start
                palm_clear_progress = min(1.0, elapsed / 0.9)
                if palm_clear_progress >= 1.0:
                    canvas.clear()
                    grabbed = {"Right": None, "Left": None}
                    last_grab_pos = {"Right": None, "Left": None}
                    prev_two_hand_dist = None
                    scale_shape_idx = None
                    ui.show_toast("Canvas Cleared!")
                    palm_hold_start = 0.0
                    palm_clear_progress = 0.0
            else:
                if not palm_detected:
                    palm_hold_start = 0.0
                    palm_clear_progress = 0.0

                # ── TWO-HAND PINCH-SCALE ──────────────────────────────────
                if num_hands == 2 and all(hd.is_pinch for hd in hands):
                    p0 = hands[0].pinch_midpoint
                    p1 = hands[1].pinch_midpoint
                    two_dist = math.hypot(p0[0] - p1[0], p0[1] - p1[1])
                    origin = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)

                    # Find any grabbed shape from either hand
                    si = grabbed.get("Right") if grabbed.get("Right") is not None else grabbed.get("Left")
                    if si is None:
                        si = canvas.find_shape_at(origin, margin=60.0)

                    if si is not None:
                        scale_shape_idx = si
                        if prev_two_hand_dist is not None and prev_two_hand_dist > 1.0:
                            factor = two_dist / prev_two_hand_dist
                            factor = max(0.92, min(factor, 1.08))   # clamp jitter
                            canvas.scale_shape(si, factor, origin)
                        prev_two_hand_dist = two_dist
                        mode_str = "TWO-HAND SCALING"
                    else:
                        prev_two_hand_dist = None
                else:
                    prev_two_hand_dist = None
                    if scale_shape_idx is not None:
                        scale_shape_idx = None

                # ── PER-HAND ACTIONS ──────────────────────────────────────
                active_hand_labels = [hd.label for hd in hands]

                # Release state for hands not currently tracked
                for label in list(grabbed.keys()):
                    if label not in active_hand_labels:
                        if grabbed[label] is not None:
                            canvas.snap_shape_anchors(grabbed[label], snap_threshold=recognizer.snap_threshold)
                            ui.show_toast("Shape Placed!")
                            grabbed[label] = None
                            last_grab_pos[label] = None
                        stroke = canvas.current_strokes.get(label, [])
                        if len(stroke) >= 2:
                            recognized = canvas.finish_stroke(recognizer, hand=label)
                            if recognized:
                                ui.show_toast(f"Shape: {recognized.shape_type.capitalize()}")
                        else:
                            canvas.current_strokes[label] = []
                        canvas.reset_point(label)

                for hd in hands:
                    label = hd.label
                    midpoint = hd.pinch_midpoint
                    target_pt, is_snapped = recognizer.snap_point(midpoint, canvas.anchors)

                    if hd.is_pinch:
                        # Toolbar click?
                        if midpoint[1] <= ui.header_height:
                            ui.check_interaction(midpoint[0], midpoint[1], canvas)
                            canvas.current_strokes[label] = []
                            grabbed[label] = None
                            last_grab_pos[label] = None
                            mode_str = "TOOLBAR"
                            continue

                        # Already dragging?
                        if grabbed[label] is not None:
                            if last_grab_pos[label] is not None:
                                dx = target_pt[0] - last_grab_pos[label][0]
                                dy = target_pt[1] - last_grab_pos[label][1]
                                canvas.move_shape(grabbed[label], dx, dy)
                            last_grab_pos[label] = target_pt
                            stype = canvas.shapes[grabbed[label]].shape_type.upper()
                            mode_str = f"MOVING {stype} [{label.upper()}]"

                        # Try to grab a shape?
                        elif last_grab_pos[label] is None and not canvas.is_eraser_active():
                            ci = canvas.find_shape_at(target_pt, margin=26.0)
                            if ci is not None:
                                grabbed[label] = ci
                                last_grab_pos[label] = target_pt
                                stype = canvas.shapes[ci].shape_type.capitalize()
                                mode_str = f"GRABBED {stype.upper()} [{label.upper()}]"
                                ui.show_toast(f"{label} hand grabbed {stype}!")
                            else:
                                # Draw
                                if canvas.is_eraser_active():
                                    canvas.draw_direct(target_pt, hand=label)
                                    mode_str = f"ERASING [{label.upper()}]"
                                else:
                                    canvas.add_stroke_point(target_pt, hand=label)
                                    mode_str = f"DRAWING [{label.upper()}]"
                        else:
                            if canvas.is_eraser_active():
                                canvas.draw_direct(target_pt, hand=label)
                                mode_str = f"ERASING [{label.upper()}]"
                            else:
                                canvas.add_stroke_point(target_pt, hand=label)
                                mode_str = f"DRAWING [{label.upper()}]"

                    else:
                        # Pinch released
                        if grabbed[label] is not None:
                            snapped = canvas.snap_shape_anchors(grabbed[label], snap_threshold=recognizer.snap_threshold)
                            ui.show_toast("Placed & Connected!" if snapped else "Shape Placed!")
                            grabbed[label] = None
                            last_grab_pos[label] = None
                        elif len(canvas.current_strokes.get(label, [])) >= 2:
                            recognized = canvas.finish_stroke(recognizer, hand=label)
                            if recognized:
                                ui.show_toast(f"Shape: {recognized.shape_type.capitalize()}")
                        else:
                            canvas.current_strokes[label] = []
                        canvas.reset_point(label)

                        if mode_str == "STANDBY":
                            mode_str = "HOVER / NAVIGATION"

            # ── RENDER ────────────────────────────────────────────────────
            if blackboard_mode:
                display = canvas.canvas.copy()
            else:
                display = canvas.blend(frame)

            # Bounding boxes for grabbed / hovered shapes
            shown_idxs = set()
            for label, idx in grabbed.items():
                if idx is not None and idx not in shown_idxs:
                    canvas.render_shape_box(display, idx, is_grabbed=True)
                    shown_idxs.add(idx)

            # Hover box when no hands are grabbing
            if not shown_idxs and hands:
                best_pt = hands[0].pinch_midpoint
                hi = canvas.find_shape_at(best_pt, margin=24.0)
                if hi is not None:
                    canvas.render_shape_box(display, hi, is_grabbed=False)

            # Anchors and live stroke previews
            all_hover_pts = [hd.pinch_midpoint for hd in hands] if hands else []
            hover_for_snap = all_hover_pts[0] if all_hover_pts else None
            canvas.render_anchors(display, hover_pt=hover_for_snap, snap_threshold=recognizer.snap_threshold)
            canvas.render_live_preview(display)

            # Per-hand cursors
            for hd in hands:
                snap_pt, is_snapped = recognizer.snap_point(hd.pinch_midpoint, canvas.anchors)
                ui.draw_cursor(
                    frame=display,
                    point=snap_pt if is_snapped else hd.pinch_midpoint,
                    is_pinching=hd.is_pinch,
                    is_snapped=is_snapped,
                    hand_label=hd.label,
                    brush_size=canvas.brush_thickness,
                    active_color=canvas.current_color,
                )

            # Palm clear radial meter
            if palm_clear_progress > 0.0:
                ui.draw_palm_clear_progress(display, palm_clear_progress)

            # Master UI overlay
            current_thickness = canvas.eraser_thickness if canvas.is_eraser_active() else canvas.brush_thickness
            ui.draw(
                frame=display,
                current_mode=mode_str,
                active_color=canvas.current_color,
                brush_size=current_thickness,
                fps=fps,
                num_hands=num_hands,
            )

            cv2.imshow("Air Canvas AI", display)

            # ── KEYBOARD SHORTCUTS ────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Quit.")
                break
            elif key == ord("z"):
                if canvas.undo():
                    ui.show_toast("Undo!")
            elif key == ord("c"):
                canvas.clear()
                grabbed = {"Right": None, "Left": None}
                ui.show_toast("Canvas Cleared!")
            elif key == ord("s"):
                path = canvas.save_snapshot()
                ui.show_toast("Snapshot Saved!")
                print(f"[INFO] Saved: {path}")
            elif key == ord("b"):
                blackboard_mode = not blackboard_mode
                ui.show_toast("Blackboard" if blackboard_mode else "AR View")
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
        print("\n[INFO] Interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Cleaned up.")


if __name__ == "__main__":
    main()
