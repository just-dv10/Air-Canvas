"""
Canvas Module for managing shapes, anchors, live stroke previews,
color palettes, blending, and snapshot exports.
"""

import os
import time
from typing import Tuple, List, Optional
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
        self.current_stroke: List[Tuple[int, int]] = []
        self.prev_point: Optional[Tuple[int, int]] = None

    def reset_point(self) -> None:
        """Resets the previous drawing point."""
        self.prev_point = None

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

    def add_stroke_point(self, point: Tuple[int, int]) -> None:
        """Adds a point to the currently active stroke."""
        # Avoid recording duplicate consecutive points
        if not self.current_stroke or self.current_stroke[-1] != point:
            self.current_stroke.append(point)

    def draw_direct(self, point: Tuple[int, int]) -> None:
        """Draws directly onto the canvas buffer (used for eraser strokes)."""
        if self.prev_point is None:
            self.prev_point = point
            return

        thickness = self.eraser_thickness if self.is_eraser_active() else self.brush_thickness
        cv2.line(self.canvas, self.prev_point, point, self.current_color, thickness, lineType=cv2.LINE_AA)
        cv2.circle(self.canvas, point, thickness // 2, self.current_color, cv2.FILLED)
        self.prev_point = point

    # Backwards-compatible alias
    draw = draw_direct


    def finish_stroke(self, recognizer: ShapeRecognizer) -> Optional[RecognizedShape]:
        """
        Processes the active stroke: recognizes geometric shape or curve,
        commits it to canvas, registers anchors, and clears the stroke buffer.
        """
        if len(self.current_stroke) < 2:
            self.current_stroke = []
            return None

        # Eraser stroke is already applied or can wipe
        if self.is_eraser_active():
            self.current_stroke = []
            return None

        shape = recognizer.recognize(
            points=self.current_stroke,
            color=self.current_color,
            thickness=self.brush_thickness,
            existing_anchors=self.anchors,
        )

        self.shapes.append(shape)
        # Register new anchors for connecting future shapes
        for anchor in shape.anchors:
            if anchor not in self.anchors:
                self.anchors.append(anchor)

        self._render_all_shapes()
        self.current_stroke = []
        return shape

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

    def undo(self) -> bool:
        """Removes the last drawn shape."""
        if not self.shapes:
            return False
        self.shapes.pop()
        # Rebuild anchors list
        self.anchors = []
        for s in self.shapes:
            for a in s.anchors:
                if a not in self.anchors:
                    self.anchors.append(a)
        self._render_all_shapes()
        return True

    def clear(self) -> None:
        """Clears all strokes, shapes, and connection anchors."""
        self.canvas[:] = 0
        self.shapes = []
        self.anchors = []
        self.current_stroke = []
        self.prev_point = None

    def render_live_preview(self, frame: cv2.Mat) -> None:
        """Renders the stroke currently being drawn by the user in real time."""
        if len(self.current_stroke) >= 2:
            pts = np.array(self.current_stroke, dtype=np.int32).reshape((-1, 1, 2))
            thickness = self.eraser_thickness if self.is_eraser_active() else self.brush_thickness
            cv2.polylines(frame, [pts], isClosed=False, color=self.current_color, thickness=thickness, lineType=cv2.LINE_AA)

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
