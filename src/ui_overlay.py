"""
UI Overlay Module for rendering interactive buttons, color palettes,
status banners, and HUD hints on the camera frame.
"""

from typing import Tuple, List, Dict, Optional
import time
import cv2
from src.canvas import Canvas


class Button:
    """Represents an interactive clickable zone on the header toolbar."""

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
        self.action_type = action_type  # 'color', 'eraser', 'clear'

    def is_inside(self, x: int, y: int) -> bool:
        """Checks if a coordinate falls inside this button boundary."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class UIOverlay:
    """
    Renders the modern floating header toolbar, active tool feedback,
    and HUD notifications.
    """

    def __init__(self, width: int = 1280, header_height: int = 85):
        self.width = width
        self.header_height = header_height
        self.buttons: List[Button] = []
        self._build_buttons()

        # Temporary banner / message system
        self.toast_message: Optional[str] = None
        self.toast_expire_time: float = 0.0

    def _build_buttons(self) -> None:
        """Constructs layout of interactive buttons in the top toolbar."""
        self.buttons = []
        # Toolbar layout items
        items = [
            ("Cyan", Canvas.PALETTE["Cyan"], "color"),
            ("Pink", Canvas.PALETTE["Neon Pink"], "color"),
            ("Green", Canvas.PALETTE["Emerald"], "color"),
            ("Amber", Canvas.PALETTE["Amber"], "color"),
            ("White", Canvas.PALETTE["White"], "color"),
            ("Eraser", (70, 70, 70), "eraser"),
            ("CLEAR", (30, 30, 180), "clear"),
        ]

        num_items = len(items)
        margin = 15
        gap = 12
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

    def show_toast(self, message: str, duration: float = 2.5) -> None:
        """Displays a temporary popup message on the screen."""
        self.toast_message = message
        self.toast_expire_time = time.time() + duration

    def check_interaction(
        self, x: int, y: int, canvas: Canvas
    ) -> Optional[str]:
        """
        Tests if coordinate (x, y) hit any button.
        Updates canvas state accordingly.
        Returns the action description or None.
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
                elif btn.action_type == "clear":
                    canvas.clear()
                    self.show_toast("Canvas Cleared!")
                    return "Cleared"
        return None

    def draw(
        self,
        frame: cv2.Mat,
        current_mode: str,
        active_color: Tuple[int, int, int],
        brush_size: int,
        fps: float = 0.0,
    ) -> None:
        """Renders the complete UI overlay onto the frame."""
        h, w, _ = frame.shape
        self.update_dimensions(w)

        # 1. Semi-transparent header background
        header_overlay = frame.copy()
        cv2.rectangle(
            header_overlay,
            (0, 0),
            (w, self.header_height),
            (20, 20, 24),
            cv2.FILLED,
        )
        cv2.addWeighted(header_overlay, 0.75, frame, 0.25, 0, frame)

        # 2. Draw Toolbar Buttons
        for btn in self.buttons:
            # Check if this button represents the currently active tool
            is_active = False
            if btn.action_type == "color" and btn.color_bgr == active_color:
                is_active = True
            elif btn.action_type == "eraser" and active_color == Canvas.PALETTE["Eraser"]:
                is_active = True

            # Button background
            cv2.rectangle(
                frame,
                (btn.x1, btn.y1),
                (btn.x2, btn.y2),
                btn.color_bgr,
                cv2.FILLED,
            )

            # Border (thicker if active)
            border_color = (255, 255, 255) if is_active else (50, 50, 50)
            border_thickness = 3 if is_active else 1
            cv2.rectangle(
                frame,
                (btn.x1, btn.y1),
                (btn.x2, btn.y2),
                border_color,
                border_thickness,
            )

            # Button Text
            text_color = (0, 0, 0) if (btn.action_type == "color" and btn.name in ["Cyan", "Amber", "White", "Green"]) else (255, 255, 255)
            font_scale = 0.55
            text_size = cv2.getTextSize(btn.name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = btn.x1 + (btn.x2 - btn.x1 - text_size[0]) // 2
            text_y = btn.y1 + (btn.y2 - btn.y1 + text_size[1]) // 2
            cv2.putText(
                frame,
                btn.name,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                2,
                cv2.LINE_AA,
            )

        # 3. Mode & Status Bar at Bottom
        status_bar_y = h - 35
        cv2.rectangle(frame, (0, status_bar_y), (w, h), (15, 15, 18), cv2.FILLED)

        # Mode indicator text
        mode_color = (0, 255, 128) if "DRAW" in current_mode else (255, 180, 0)
        cv2.putText(
            frame,
            f"MODE: {current_mode}",
            (20, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            mode_color,
            2,
            cv2.LINE_AA,
        )

        # Brush size and FPS
        info_str = f"Brush: {brush_size}px | FPS: {int(fps)} | Hotkeys: [C] Clear [S] Save [B] Board [+/-] Size [Q] Quit"
        cv2.putText(
            frame,
            info_str,
            (240, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # 4. Toast notification if active
        if self.toast_message and time.time() < self.toast_expire_time:
            tw, th = cv2.getTextSize(self.toast_message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            tx = (w - tw) // 2
            ty = self.header_height + 40
            # Background pill
            cv2.rectangle(frame, (tx - 15, ty - th - 10), (tx + tw + 15, ty + 10), (25, 25, 30), cv2.FILLED)
            cv2.rectangle(frame, (tx - 15, ty - th - 10), (tx + tw + 15, ty + 10), (0, 230, 115), 2)
            cv2.putText(
                frame,
                self.toast_message,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        elif self.toast_message and time.time() >= self.toast_expire_time:
            self.toast_message = None
