"""
Air Canvas AI - Main Application Loop (Two-Hand Edition).
Independent simultaneous two-hand tracking: draw, grab, scale, and connect shapes
with both hands independently in real time without shape clashing.
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

    tracker    = HandTracker(max_hands=2, detection_confidence=0.5, tracking_confidence=0.5)
    canvas     = Canvas(width=args.width, height=args.height)
    ui         = UIOverlay(width=args.width)
    recognizer = ShapeRecognizer(snap_threshold=28.0)

    blackboard_mode = False
    prev_time       = time.time()
    palm_hold_start: float = 0.0
    palm_clear_progress: float = 0.0

    # Per-hand drag state: { "Right": shape_idx or None, "Left": shape_idx or None }
    grabbed: Dict[str, Optional[int]] = {"Right": None, "Left": None}
    last_grab_pos: Dict[str, Optional[Tuple[int, int]]] = {"Right": None, "Left": None}

    # Two-hand pinch-to-scale state (ONLY active when both hands grab the SAME shape)
    prev_two_hand_dist: Optional[float] = None
    prev_two_hand_mid:  Optional[Tuple[int, int]] = None

    print("\n[INFO] Gesture Guide:")
    print("  🤏 Pinch on shape       → Grab & drag it (Left & Right hands work independently!)")
    print("  🤏 Pinch in empty space → Draw a stroke (auto-shapes on pinch release)")
    print("  🤏 Both hands on 1 shape→ Scale & rotate the shape (spread=bigger, pinch=smaller)")
    print("  🖐️  Open palm ~1s       → Clear canvas")
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
                    prev_two_hand_mid = None
                    ui.show_toast("Canvas Cleared!")
                    palm_hold_start = 0.0
                    palm_clear_progress = 0.0
            else:
                if not palm_detected:
                    palm_hold_start = 0.0
                    palm_clear_progress = 0.0

                active_hand_labels = [hd.label for hd in hands]

                # ── RELEASE HANDS THAT DISAPPEARED ────────────────────────
                for label in list(grabbed.keys()):
                    if label not in active_hand_labels:
                        if grabbed[label] is not None:
                            released_idx = grabbed[label]
                            grabbed[label] = None
                            last_grab_pos[label] = None
                            other_hand = "Left" if label == "Right" else "Right"
                            # If other hand is NOT holding this exact same shape, snap and drop
                            if grabbed.get(other_hand) != released_idx:
                                other_held = [v for k, v in grabbed.items() if v is not None]
                                canvas.snap_shape_anchors(released_idx, snap_threshold=recognizer.snap_threshold, exclude_indices=other_held)
                                ui.show_toast("Shape Placed!")
                        stroke = canvas.current_strokes.get(label, [])
                        if len(stroke) >= 2:
                            recognized = canvas.finish_stroke(recognizer, hand=label)
                            if recognized:
                                ui.show_toast(f"Shape: {recognized.shape_type.capitalize()}")
                        else:
                            canvas.current_strokes[label] = []
                        canvas.reset_point(label)

                # ── PROCESS PER-HAND INPUT INTENT ─────────────────────────
                for hd in hands:
                    label = hd.label
                    midpoint = hd.pinch_midpoint
                    target_pt, is_snapped = recognizer.snap_point(midpoint, canvas.anchors)

                    if hd.is_pinch:
                        # 1. Toolbar click
                        if midpoint[1] <= ui.header_height:
                            ui.check_interaction(midpoint[0], midpoint[1], canvas)
                            canvas.current_strokes[label] = []
                            grabbed[label] = None
                            last_grab_pos[label] = None
                            continue

                        # 2. Already grabbing a shape: maintain grasp
                        if grabbed[label] is not None:
                            pass

                        # 3. Not grabbing yet: try to grab or draw
                        elif last_grab_pos[label] is None and not canvas.is_eraser_active():
                            other_hand = "Left" if label == "Right" else "Right"
                            other_held = grabbed.get(other_hand)

                            # First prioritize an unheld shape
                            ci = canvas.find_shape_at(
                                target_pt,
                                margin=26.0,
                                exclude_indices=[other_held] if other_held is not None else None,
                            )
                            # If no unheld shape was found, check if pinching the same shape as other hand (2-hand scale)
                            if ci is None and other_held is not None:
                                ci = canvas.find_shape_at(target_pt, margin=36.0)

                            if ci is not None:
                                grabbed[label] = ci
                                last_grab_pos[label] = target_pt
                                stype = canvas.shapes[ci].shape_type.capitalize()
                                if ci == other_held:
                                    ui.show_toast(f"Both hands scaling {stype}!")
                                else:
                                    ui.show_toast(f"{label} grabbed {stype}!")
                            else:
                                # Start drawing stroke
                                if canvas.is_eraser_active():
                                    canvas.draw_direct(target_pt, hand=label)
                                else:
                                    canvas.add_stroke_point(target_pt, hand=label)
                        else:
                            # Continue drawing stroke
                            if canvas.is_eraser_active():
                                canvas.draw_direct(target_pt, hand=label)
                            else:
                                canvas.add_stroke_point(target_pt, hand=label)

                    else:
                        # Pinch released for this hand
                        if grabbed[label] is not None:
                            released_idx = grabbed[label]
                            grabbed[label] = None
                            last_grab_pos[label] = None
                            other_hand = "Left" if label == "Right" else "Right"
                            # If other hand is NOT holding this exact same shape, snap anchors and drop!
                            if grabbed.get(other_hand) != released_idx:
                                other_held = [v for k, v in grabbed.items() if v is not None]
                                snapped = canvas.snap_shape_anchors(
                                    released_idx,
                                    snap_threshold=recognizer.snap_threshold,
                                    exclude_indices=other_held,
                                )
                                ui.show_toast("Placed & Connected!" if snapped else "Shape Placed!")
                        elif len(canvas.current_strokes.get(label, [])) >= 2:
                            recognized = canvas.finish_stroke(recognizer, hand=label)
                            if recognized:
                                ui.show_toast(f"Shape: {recognized.shape_type.capitalize()}")
                        else:
                            canvas.current_strokes[label] = []
                        canvas.reset_point(label)

                # ── EXECUTE MOVEMENT & SCALING ────────────────────────────
                r_idx = grabbed.get("Right")
                l_idx = grabbed.get("Left")

                # Scenario 1: Both hands holding the EXACT SAME shape (Two-Hand Scale)
                if r_idx is not None and l_idx is not None and r_idx == l_idx:
                    si = r_idx
                    hd_r = next((h for h in hands if h.label == "Right"), None)
                    hd_l = next((h for h in hands if h.label == "Left"), None)
                    if hd_r and hd_l:
                        p_r = hd_r.pinch_midpoint
                        p_l = hd_l.pinch_midpoint
                        two_dist = math.hypot(p_r[0] - p_l[0], p_r[1] - p_l[1])
                        two_mid  = ((p_r[0] + p_l[0]) // 2, (p_r[1] + p_l[1]) // 2)

                        if prev_two_hand_dist is not None and prev_two_hand_dist > 5.0:
                            factor = two_dist / prev_two_hand_dist
                            factor = max(0.92, min(factor, 1.08))
                            canvas.scale_shape(si, factor, two_mid)

                        if prev_two_hand_mid is not None:
                            dx = two_mid[0] - prev_two_hand_mid[0]
                            dy = two_mid[1] - prev_two_hand_mid[1]
                            canvas.move_shape(si, dx, dy)

                        prev_two_hand_dist = two_dist
                        prev_two_hand_mid  = two_mid
                        last_grab_pos["Right"] = p_r
                        last_grab_pos["Left"]  = p_l
                        stype = canvas.shapes[si].shape_type.upper()
                        mode_str = f"TWO-HAND SCALE & MOVE [{stype}]"

                # Scenario 2: Two DIFFERENT shapes, or 1 shape, or drawing
                else:
                    prev_two_hand_dist = None
                    prev_two_hand_mid  = None

                    move_descs = []
                    draw_descs = []

                    for hd in hands:
                        lbl = hd.label
                        s_idx = grabbed.get(lbl)
                        if s_idx is not None and hd.is_pinch:
                            tgt_pt, _ = recognizer.snap_point(hd.pinch_midpoint, canvas.anchors)
                            if last_grab_pos.get(lbl) is not None:
                                dx = tgt_pt[0] - last_grab_pos[lbl][0]
                                dy = tgt_pt[1] - last_grab_pos[lbl][1]
                                canvas.move_shape(s_idx, dx, dy)
                            last_grab_pos[lbl] = tgt_pt
                            move_descs.append(f"{canvas.shapes[s_idx].shape_type[:4].upper()} ({lbl[0]})")
                        elif hd.is_pinch and hd.pinch_midpoint[1] > ui.header_height:
                            if canvas.is_eraser_active():
                                draw_descs.append(f"ERASE ({lbl[0]})")
                            else:
                                draw_descs.append(f"DRAW ({lbl[0]})")

                    if len(move_descs) == 2:
                        mode_str = f"DUAL DRAG: {move_descs[0]} & {move_descs[1]}"
                    elif len(move_descs) == 1 and len(draw_descs) == 1:
                        mode_str = f"DRAG {move_descs[0]} | {draw_descs[0]}"
                    elif len(move_descs) == 1:
                        mode_str = f"MOVING {move_descs[0]}"
                    elif len(draw_descs) == 2:
                        mode_str = f"DUAL DRAW: {draw_descs[0]} & {draw_descs[1]}"
                    elif len(draw_descs) == 1:
                        mode_str = f"{draw_descs[0]}ING"
                    elif num_hands > 0:
                        mode_str = "HOVER / NAVIGATION"

            # ── RENDER ────────────────────────────────────────────────────
            if blackboard_mode:
                display = canvas.canvas.copy()
            else:
                display = canvas.blend(frame)

            # Bounding boxes for grabbed shapes (with hand-specific colors)
            shown_idxs = set()
            r_idx = grabbed.get("Right")
            l_idx = grabbed.get("Left")

            if r_idx is not None and l_idx is not None and r_idx == l_idx:
                canvas.render_shape_box(
                    display, r_idx, is_grabbed=True,
                    custom_color=(0, 215, 255),
                    custom_label="TWO-HAND SCALE",
                )
                shown_idxs.add(r_idx)
            else:
                if r_idx is not None:
                    canvas.render_shape_box(
                        display, r_idx, is_grabbed=True,
                        custom_color=(255, 230, 0),
                        custom_label="RIGHT HAND",
                    )
                    shown_idxs.add(r_idx)
                if l_idx is not None:
                    canvas.render_shape_box(
                        display, l_idx, is_grabbed=True,
                        custom_color=(220, 20, 255),
                        custom_label="LEFT HAND",
                    )
                    shown_idxs.add(l_idx)

            # Hover box when no hands are grabbing that shape
            if not shown_idxs and hands:
                for hd in hands:
                    hi = canvas.find_shape_at(hd.pinch_midpoint, margin=24.0)
                    if hi is not None and hi not in shown_idxs:
                        canvas.render_shape_box(display, hi, is_grabbed=False)
                        shown_idxs.add(hi)
                        break

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
