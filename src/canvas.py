"""
Canvas Module for managing shapes, anchors, live stroke previews,
color palettes, blending, and snapshot exports.
"""

import os
import time
from typing import Tuple, List, Optional, Dict
import cv2
import numpy as np
from src.shape_recognizer import RecognizedShape, ShapeRecognizer


class Canvas:
    """
    Manages vector shapes, anchor snapping points, live stroke buffers,
    and alpha-blending onto webcam frames.
    """

    PALETTE = {
        "Cyan": (255, 200, 0),
        "Neon Pink": (180, 0, 255),
        "Emerald": (0, 230, 115),
        "Amber": (0, 165, 255),
        "White": (255, 255, 255),
        "Eraser": (0, 0, 0),
    }

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

        self.current_color = self.PALETTE["Cyan"]
        self.brush_thickness = 6
        self.eraser_thickness = 45

        # Shape repository and anchor connectivity
        self.shapes: List[RecognizedShape] = []
        self.anchors: List[Tuple[int, int]] = []
        self.current_strokes: Dict[str, List[Tuple[int, int]]] = {"Right": [], "Left": []}
        self.prev_points: Dict[str, Optional[Tuple[int, int]]] = {"Right": None, "Left": None}

    @property
    def current_stroke(self) -> List[Tuple[int, int]]:
        """Backwards compatibility for primary hand stroke."""
        return self.current_strokes["Right"]

    @current_stroke.setter
    def current_stroke(self, val: List[Tuple[int, int]]) -> None:
        self.current_strokes["Right"] = val

    @property
    def prev_point(self) -> Optional[Tuple[int, int]]:
        return self.prev_points["Right"]

    @prev_point.setter
    def prev_point(self, val: Optional[Tuple[int, int]]) -> None:
        self.prev_points["Right"] = val

    def reset_point(self, hand: str = "Right") -> None:
        """Resets the previous drawing point for the given hand."""
        self.prev_points[hand] = None

    def set_color(self, color_bgr: Tuple[int, int, int]) -> None:
        """Sets the active drawing color."""
        self.current_color = color_bgr

    def is_eraser_active(self) -> bool:
        """Returns True if the current tool is the eraser."""
        return self.current_color == self.PALETTE["Eraser"]

    def set_brush_thickness(self, size: int) -> None:
        """Clamps brush thickness within safe bounds [2, 50]."""
        self.brush_thickness = max(2, min(size, 50))

    def set_eraser_thickness(self, size: int) -> None:
        """Clamps eraser thickness within safe bounds [15, 120]."""
        self.eraser_thickness = max(15, min(size, 120))

    def add_stroke_point(self, point: Tuple[int, int], hand: str = "Right") -> None:
        """Adds a point to the stroke for the specified hand."""
        stroke = self.current_strokes.setdefault(hand, [])
        if not stroke or stroke[-1] != point:
            stroke.append(point)

    def draw_direct(self, point: Tuple[int, int], hand: str = "Right") -> None:
        """Draws directly onto canvas buffer (used for eraser strokes)."""
        prev = self.prev_points.get(hand)
        if prev is None:
            self.prev_points[hand] = point
            return

        thickness = self.eraser_thickness if self.is_eraser_active() else self.brush_thickness
        cv2.line(self.canvas, prev, point, self.current_color, thickness, lineType=cv2.LINE_AA)
        cv2.circle(self.canvas, point, thickness // 2, self.current_color, cv2.FILLED)
        self.prev_points[hand] = point

    # Backwards-compatible alias
    draw = draw_direct

    def finish_stroke(self, recognizer: ShapeRecognizer, hand: str = "Right") -> Optional[RecognizedShape]:
        """
        Processes stroke for specified hand, recognizes shape, and commits to canvas.
        """
        stroke = self.current_strokes.get(hand, [])
        if len(stroke) < 2:
            self.current_strokes[hand] = []
            return None

        if self.is_eraser_active():
            self.current_strokes[hand] = []
            return None

        shape = recognizer.recognize(
            points=stroke,
            color=self.current_color,
            thickness=self.brush_thickness,
            existing_anchors=self.anchors,
        )

        self.shapes.append(shape)
        for anchor in shape.anchors:
            if anchor not in self.anchors:
                self.anchors.append(anchor)

        self._render_all_shapes()
        self.current_strokes[hand] = []
        return shape

    def scale_shape(self, shape_idx: int, factor: float, origin: Optional[Tuple[int, int]] = None) -> None:
        """Scales shape by factor relative to origin and refreshes canvas."""
        if 0 <= shape_idx < len(self.shapes):
            self.shapes[shape_idx].scale(factor, origin)
            self._rebuild_anchors()
            self._render_all_shapes()


    def _render_all_shapes(self) -> None:
        """Re-draws all committed shapes onto the canvas."""
        self.canvas[:] = 0
        for s in self.shapes:
            if s.shape_type == "line":
                if len(s.points) >= 2:
                    cv2.line(self.canvas, s.points[0], s.points[1], s.color, s.thickness, cv2.LINE_AA)
            elif s.shape_type == "circle":
                center = s.metadata.get("center", s.points[0])
                radius = s.metadata.get("radius", 20)
                cv2.circle(self.canvas, center, radius, s.color, s.thickness, cv2.LINE_AA)
            elif s.shape_type in ("rectangle", "triangle"):
                pts = np.array(s.points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(self.canvas, [pts], isClosed=True, color=s.color, thickness=s.thickness, lineType=cv2.LINE_AA)
            elif s.shape_type == "curve":
                if len(s.points) >= 2:
                    pts = np.array(s.points, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(self.canvas, [pts], isClosed=False, color=s.color, thickness=s.thickness, lineType=cv2.LINE_AA)

    def _rebuild_anchors(self) -> None:
        """Reconstructs unique anchor points from all current shapes."""
        self.anchors = []
        for s in self.shapes:
            for a in s.anchors:
                if a not in self.anchors:
                    self.anchors.append(a)

    def find_shape_at(self, pt: Tuple[int, int], margin: float = 24.0) -> Optional[int]:
        """Returns the index of the topmost shape containing or near pt."""
        for i in reversed(range(len(self.shapes))):
            if self.shapes[i].contains_point(pt, margin=margin):
                return i
        return None

    def move_shape(self, shape_idx: int, dx: int, dy: int) -> None:
        """Translates shape points and refreshes canvas."""
        if 0 <= shape_idx < len(self.shapes):
            self.shapes[shape_idx].translate(dx, dy)
            self._rebuild_anchors()
            self._render_all_shapes()

    def snap_shape_anchors(self, shape_idx: int, snap_threshold: float = 28.0) -> bool:
        """
        After moving a shape, snaps it if any of its anchors are close to another shape's anchors.
        """
        if not (0 <= shape_idx < len(self.shapes)):
            return False

        moved_shape = self.shapes[shape_idx]
        other_anchors = []
        for i, s in enumerate(self.shapes):
            if i != shape_idx:
                other_anchors.extend(s.anchors)

        if not other_anchors:
            return False

        best_delta = None
        min_dist = float("inf")
        for a in moved_shape.anchors:
            for target in other_anchors:
                d = np.hypot(a[0] - target[0], a[1] - target[1])
                if d < min_dist and d <= snap_threshold:
                    min_dist = d
                    best_delta = (target[0] - a[0], target[1] - a[1])

        if best_delta:
            self.move_shape(shape_idx, int(best_delta[0]), int(best_delta[1]))
            return True
        return False

    def render_shape_box(self, frame: cv2.Mat, shape_idx: int, is_grabbed: bool = False) -> None:
        """Draws bounding box feedback when hovering or grabbing a shape."""
        if not (0 <= shape_idx < len(self.shapes)):
            return

        shape = self.shapes[shape_idx]
        x1, y1, x2, y2 = shape.get_bounds()
        pad = 10
        bx1, by1 = max(0, x1 - pad), max(0, y1 - pad)
        bx2, by2 = min(self.width, x2 + pad), min(self.height, y2 + pad)

        border_color = (0, 255, 0) if is_grabbed else (0, 220, 255)
        thickness = 2 if is_grabbed else 1

        # Bounding box
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), border_color, thickness, cv2.LINE_AA)

        # Corner handles
        h_size = 6
        for cx, cy in [(bx1, by1), (bx2, by1), (bx2, by2), (bx1, by2)]:
            cv2.rectangle(frame, (cx - h_size // 2, cy - h_size // 2), (cx + h_size // 2, cy + h_size // 2), border_color, cv2.FILLED)

        # Tag label
        label = "DRAGGING" if is_grabbed else "PINCH TO GRAB"
        tag_bg = (0, 180, 0) if is_grabbed else (25, 25, 30)
        text_color = (0, 0, 0) if is_grabbed else (255, 255, 255)
        cv2.rectangle(frame, (bx1, max(0, by1 - 22)), (bx1 + 115, by1), tag_bg, cv2.FILLED)
        cv2.putText(frame, label, (bx1 + 6, by1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1, cv2.LINE_AA)

    def undo(self) -> bool:
        """Removes the last drawn shape."""
        if not self.shapes:
            return False
        self.shapes.pop()
        self._rebuild_anchors()
        self._render_all_shapes()
        return True


    def clear(self) -> None:
        """Clears all strokes, shapes, and connection anchors."""
        self.canvas[:] = 0
        self.shapes = []
        self.anchors = []
        self.current_strokes = {"Right": [], "Left": []}
        self.prev_points = {"Right": None, "Left": None}

    def render_live_preview(self, frame: cv2.Mat) -> None:
        """Renders both hands' active strokes as live previews in real time."""
        for hand, stroke in self.current_strokes.items():
            if len(stroke) >= 2:
                pts = np.array(stroke, dtype=np.int32).reshape((-1, 1, 2))
                thickness = self.eraser_thickness if self.is_eraser_active() else self.brush_thickness
                color = self.current_color if not self.is_eraser_active() else (0, 0, 0)
                cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    def render_anchors(
        self,
        frame: cv2.Mat,
        hover_pt: Optional[Tuple[int, int]] = None,
        snap_threshold: float = 28.0,
    ) -> None:
        """
        Renders connection anchor dots at shape vertices.
        Highlights magnetic snap targets when hover_pt is within snapping range.
        """
        for a in self.anchors:
            is_hovered = False
            if hover_pt is not None:
                dist = np.hypot(hover_pt[0] - a[0], hover_pt[1] - a[1])
                is_hovered = dist <= snap_threshold

            if is_hovered:
                # Magnetic snap visual: double glowing ring
                cv2.circle(frame, a, 9, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, a, 4, (0, 255, 0), cv2.FILLED)
            else:
                # Idle anchor: subtle dot
                cv2.circle(frame, a, 3, (180, 180, 180), cv2.FILLED)

    def blend(self, frame: cv2.Mat) -> cv2.Mat:
        """Alpha-blends the drawing canvas over the input frame."""
        if frame.shape[:2] != self.canvas.shape[:2]:
            self.resize(frame.shape[1], frame.shape[0])

        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask_inv = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)

        frame_bg = cv2.bitwise_and(frame, mask_inv)
        blended = cv2.bitwise_or(frame_bg, self.canvas)
        return blended

    def resize(self, new_width: int, new_height: int) -> None:
        """Resizes canvas dimensions."""
        self.width = new_width
        self.height = new_height
        self.canvas = cv2.resize(self.canvas, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    def save_snapshot(self, output_dir: str = "saved_drawings") -> str:
        """Saves current canvas drawing to disk with timestamp."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"air_canvas_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, self.canvas)
        return os.path.abspath(filepath)
