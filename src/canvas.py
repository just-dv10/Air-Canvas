"""
Canvas Module for managing the virtual drawing surface, strokes, colors, and blending.
"""

import os
import time
from typing import Tuple, Optional
import cv2
import numpy as np


class Canvas:
    """
    Manages the off-screen drawing buffer, stroke smoothing,
    color state, eraser mechanics, and snapshot export.
    """

    # Predefined color palette (BGR)
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
        self.brush_thickness = 8
        self.eraser_thickness = 50

        # State tracking for continuous line drawing
        self.prev_point: Optional[Tuple[int, int]] = None

    def reset_point(self) -> None:
        """Resets the previous drawing point to break continuous line drawing."""
        self.prev_point = None

    def set_color(self, color_bgr: Tuple[int, int, int]) -> None:
        """Sets the active drawing color."""
        self.current_color = color_bgr

    def is_eraser_active(self) -> bool:
        """Returns True if the current color is the eraser (0, 0, 0)."""
        return self.current_color == self.PALETTE["Eraser"]

    def set_brush_thickness(self, size: int) -> None:
        """Sets brush thickness within safe bounds [2, 50]."""
        self.brush_thickness = max(2, min(size, 50))

    def set_eraser_thickness(self, size: int) -> None:
        """Sets eraser thickness within safe bounds [15, 120]."""
        self.eraser_thickness = max(15, min(size, 120))

    def draw(self, point: Tuple[int, int]) -> None:
        """
        Draws from prev_point to point.
        If prev_point is None, initializes it.
        """
        if self.prev_point is None:
            self.prev_point = point
            return

        thickness = self.eraser_thickness if self.is_eraser_active() else self.brush_thickness

        # Draw smooth line between consecutive points
        cv2.line(
            self.canvas,
            self.prev_point,
            point,
            self.current_color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        # Draw circle cap at destination point for rounded look
        cv2.circle(
            self.canvas,
            point,
            thickness // 2,
            self.current_color,
            cv2.FILLED,
        )

        self.prev_point = point

    def clear(self) -> None:
        """Wipes the canvas completely clean."""
        self.canvas[:] = 0
        self.prev_point = None

    def blend(self, frame: cv2.Mat) -> cv2.Mat:
        """
        Alpha-blends the drawing canvas over the input camera frame.
        Where the canvas has strokes, they replace the background frame.
        """
        # Ensure canvas dimensions match camera frame
        if frame.shape[:2] != self.canvas.shape[:2]:
            self.resize(frame.shape[1], frame.shape[0])

        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask_inv = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)

        # Clear space on frame and add color strokes
        frame_bg = cv2.bitwise_and(frame, mask_inv)
        blended = cv2.bitwise_or(frame_bg, self.canvas)
        return blended

    def resize(self, new_width: int, new_height: int) -> None:
        """Resizes canvas when camera resolution changes."""
        self.width = new_width
        self.height = new_height
        self.canvas = cv2.resize(self.canvas, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    def save_snapshot(self, output_dir: str = "saved_drawings") -> str:
        """
        Saves current canvas drawing to disk with timestamp.
        Returns the absolute filepath of the saved image.
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"air_canvas_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, self.canvas)
        return os.path.abspath(filepath)
