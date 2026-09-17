"""
UI Overlay Module for rendering interactive buttons, color palettes,
pinch cursors, snapping indicators, open-palm radial countdown, and HUD hints.
"""

from typing import Tuple, List, Optional
import time
import cv2
import numpy as np
from src.canvas import Canvas


class Button:
    """Represents an interactive clickable button on the toolbar."""

    def __init__(
        self,
        name: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color_bgr: Tuple[int, int, int],
        action_type: str = "color",
    ):
        self.name = name
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.color_bgr = color_bgr
        self.action_type = action_type  # 'color', 'eraser', 'undo', 'clear'

    def is_inside(self, x: int, y: int) -> bool:
        """Checks if coordinate falls inside button boundary."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class UIOverlay:
    """
    Renders the modern toolbar, active tool feedback, magnetic snap indicator,
    palm clear radial meter, and HUD notifications.
    """

    def __init__(self, width: int = 1280, header_height: int = 80):
        self.width = width
        self.header_height = header_height
        self.buttons: List[Button] = []
        self._build_buttons()

        self.toast_message: Optional[str] = None
        self.toast_expire_time: float = 0.0

    def _build_buttons(self) -> None:
        """Constructs layout of interactive buttons in the top toolbar."""
        self.buttons = []
        items = [
            ("Cyan", Canvas.PALETTE["Cyan"], "color"),
            ("Pink", Canvas.PALETTE["Neon Pink"], "color"),
            ("Green", Canvas.PALETTE["Emerald"], "color"),
            ("Amber", Canvas.PALETTE["Amber"], "color"),
            ("White", Canvas.PALETTE["White"], "color"),
            ("Eraser", (60, 60, 60), "eraser"),
            ("Undo", (100, 70, 30), "undo"),
            ("CLEAR", (30, 30, 180), "clear"),
        ]

        num_items = len(items)
        margin = 12
        gap = 10
        btn_width = (self.width - (2 * margin) - ((num_items - 1) * gap)) // num_items
        btn_y1 = 12
        btn_y2 = self.header_height - 12

        for i, (name, bgr, action) in enumerate(items):
            x1 = margin + i * (btn_width + gap)
            x2 = x1 + btn_width
            self.buttons.append(Button(name, x1, btn_y1, x2, btn_y2, bgr, action))

    def update_dimensions(self, width: int) -> None:
        """Re-computes button layout if frame width changes."""
        if width != self.width:
            self.width = width
            self._build_buttons()

    def show_toast(self, message: str, duration: float = 2.2) -> None:
        """Displays a temporary popup message."""
        self.toast_message = message
        self.toast_expire_time = time.time() + duration

    def check_interaction(
        self, x: int, y: int, canvas: Canvas
    ) -> Optional[str]:
        """
        Tests if coordinate (x, y) hit any button.
        Updates canvas state accordingly.
        """
        if y > self.header_height:
            return None

        for btn in self.buttons:
            if btn.is_inside(x, y):
                if btn.action_type == "color":
                    canvas.set_color(btn.color_bgr)
                    self.show_toast(f"Switched to {btn.name}")
                    return f"Selected {btn.name}"
                elif btn.action_type == "eraser":
                    canvas.set_color(Canvas.PALETTE["Eraser"])
                    self.show_toast("Eraser Activated")
                    return "Eraser"
                elif btn.action_type == "undo":
                    if canvas.undo():
                        self.show_toast("Undo Last Shape")
                    return "Undo"
                elif btn.action_type == "clear":
                    canvas.clear()
                    self.show_toast("Canvas Cleared!")
                    return "Cleared"
        return None

    def draw_palm_clear_progress(
        self, frame: cv2.Mat, progress: float
    ) -> None:
        """
        Renders a radial progress circle when user shows open palm to clear.
        progress ranges from 0.0 to 1.0.
        """
        h, w, _ = frame.shape
        center = (w // 2, h // 2)
        radius = 55

        # Background disc
        overlay = frame.copy()
        cv2.circle(overlay, center, radius + 15, (20, 20, 30), cv2.FILLED)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Base circle
        cv2.circle(frame, center, radius, (80, 80, 80), 4, cv2.LINE_AA)

        # Radial arc for progress
        angle = int(360 * min(1.0, progress))
        cv2.ellipse(
            frame,
            center,
            (radius, radius),
            -90,
            0,
            angle,
            (0, 0, 255),
            6,
            cv2.LINE_AA,
        )

        # Text prompt
        pct = int(progress * 100)
        cv2.putText(
            frame,
            f"CLEARING",
            (center[0] - 42, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"{pct}%",
            (center[0] - 18, center[1] + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def draw_cursor(
        self,
        frame: cv2.Mat,
        point: Tuple[int, int],
        is_pinching: bool,
        is_snapped: bool,
        active_color: Tuple[int, int, int],
        brush_thickness: int,
    ) -> None:
        """Renders precision cursor feedback at fingertip coordinates."""
        cx, cy = point

        if is_snapped:
            # Magnetic Snap Indicator: bright yellow reticle
            cv2.circle(frame, (cx, cy), 14, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
        elif is_pinching:
            # Pinch Active (Drawing): solid colored dot matching stroke color
            rad = max(brush_thickness // 2, 6)
            cv2.circle(frame, (cx, cy), rad, active_color, cv2.FILLED)
            cv2.circle(frame, (cx, cy), rad + 3, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            # Hover / Navigation: crosshair reticle
            cv2.circle(frame, (cx, cy), 8, (255, 200, 0), 2, cv2.LINE_AA)
            cv2.line(frame, (cx - 12, cy), (cx + 12, cy), (255, 255, 255), 1)
            cv2.line(frame, (cx, cy - 12), (cx, cy + 12), (255, 255, 255), 1)

    def draw(
        self,
        frame: cv2.Mat,
        current_mode: str,
        active_color: Tuple[int, int, int],
        brush_size: int,
        fps: float = 0.0,
    ) -> None:
        """Renders the top header toolbar and bottom status bar."""
        h, w, _ = frame.shape
        self.update_dimensions(w)

        # 1. Semi-transparent header background
        header_overlay = frame.copy()
        cv2.rectangle(
            header_overlay,
            (0, 0),
            (w, self.header_height),
            (18, 18, 22),
            cv2.FILLED,
        )
        cv2.addWeighted(header_overlay, 0.8, frame, 0.2, 0, frame)

        # 2. Draw Toolbar Buttons
        for btn in self.buttons:
            is_active = False
            if btn.action_type == "color" and btn.color_bgr == active_color:
                is_active = True
            elif btn.action_type == "eraser" and active_color == Canvas.PALETTE["Eraser"]:
                is_active = True

            cv2.rectangle(frame, (btn.x1, btn.y1), (btn.x2, btn.y2), btn.color_bgr, cv2.FILLED)

            border_color = (255, 255, 255) if is_active else (60, 60, 65)
            border_thickness = 3 if is_active else 1
            cv2.rectangle(frame, (btn.x1, btn.y1), (btn.x2, btn.y2), border_color, border_thickness)

            text_color = (0, 0, 0) if (btn.action_type == "color" and btn.name in ["Cyan", "Amber", "White", "Green"]) else (255, 255, 255)
            font_scale = 0.52
            text_size = cv2.getTextSize(btn.name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = btn.x1 + (btn.x2 - btn.x1 - text_size[0]) // 2
            text_y = btn.y1 + (btn.y2 - btn.y1 + text_size[1]) // 2
            cv2.putText(frame, btn.name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)

        # 3. Mode & Status Bar at Bottom
        status_bar_y = h - 34
        cv2.rectangle(frame, (0, status_bar_y), (w, h), (14, 14, 18), cv2.FILLED)

        # Mode indicator
        mode_color = (0, 255, 120) if "PINCH" in current_mode or "DRAW" in current_mode else (255, 180, 0)
        cv2.putText(frame, f"MODE: {current_mode}", (16, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.52, mode_color, 2, cv2.LINE_AA)

        # Hotkeys info
        info_str = f"Brush: {brush_size}px | FPS: {int(fps)} | Controls: Pinch=Draw | Open Palm=Clear | [Z] Undo [C] Clear [S] Save [B] Board [Q] Quit"
        cv2.putText(frame, info_str, (260, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        # 4. Toast notification popup
        if self.toast_message and time.time() < self.toast_expire_time:
            tw, th = cv2.getTextSize(self.toast_message, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
            tx = (w - tw) // 2
            ty = self.header_height + 40
            cv2.rectangle(frame, (tx - 15, ty - th - 10), (tx + tw + 15, ty + 10), (22, 22, 28), cv2.FILLED)
            cv2.rectangle(frame, (tx - 15, ty - th - 10), (tx + tw + 15, ty + 10), (0, 230, 115), 2)
            cv2.putText(frame, self.toast_message, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        elif self.toast_message and time.time() >= self.toast_expire_time:
            self.toast_message = None
