"""
Modern Glassmorphism UI Overlay for Air Canvas AI.
Renders a sleek floating toolbar with pill buttons, glowing neon active rings,
dual-hand status badges, FPS counter, brush preview, and radial palm-clear meter.
"""

from typing import Tuple, List, Optional, Dict
import time
import math
import cv2
import numpy as np
from src.canvas import Canvas


# === DESIGN TOKENS ===
BG_GLASS  = (12, 12, 16)          # Near-black glass base
ACCENT_R  = (255, 230, 0)          # Right hand / Cyan accent (BGR)
ACCENT_L  = (220, 20, 255)         # Left hand / Magenta accent (BGR)
ACCENT_G  = (0, 230, 120)          # Success / confirm green
WHITE     = (255, 255, 255)
DIM       = (140, 140, 150)
TOOLBAR_H = 84


class Button:
    """Rounded-rect interactive toolbar button."""

    def __init__(
        self,
        name: str,
        x1: int, y1: int, x2: int, y2: int,
        color_bgr: Tuple[int, int, int],
        action_type: str = "color",
        icon: str = "",
    ):
        self.name = name
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.color_bgr = color_bgr
        self.action_type = action_type
        self.icon = icon

    def is_inside(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


def _draw_rounded_rect(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],
    radius: int = 10,
    thickness: int = -1,
    alpha: float = 1.0,
) -> None:
    """Draws a filled or outlined rounded rectangle with optional alpha blend."""
    if thickness == -1:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
            cv2.circle(overlay, (cx, cy), radius, color, -1)
        if alpha < 1.0:
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        else:
            frame[:] = overlay[:]
    else:
        cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
            cv2.circle(frame, (cx, cy), radius, color, thickness)


def _pill_text(
    frame: np.ndarray,
    text: str,
    cx: int,
    cy: int,
    bg: Tuple[int, int, int],
    fg: Tuple[int, int, int] = WHITE,
    font_scale: float = 0.45,
    pad_x: int = 12,
    pad_y: int = 7,
    radius: int = 10,
    alpha: float = 0.82,
) -> None:
    """Draws a floating pill badge with centered text."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    bx1 = cx - tw // 2 - pad_x
    by1 = cy - th // 2 - pad_y
    bx2 = cx + tw // 2 + pad_x
    by2 = cy + th // 2 + pad_y
    _draw_rounded_rect(frame, bx1, by1, bx2, by2, bg, radius=radius, alpha=alpha)
    cv2.putText(frame, text, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, fg, 1, cv2.LINE_AA)


class UIOverlay:
    """
    Glassmorphism UI system. Renders:
    - Frosted floating toolbar with glowing neon pill buttons.
    - Dual-hand status badge (App title + hand count).
    - FPS counter badge (top-right).
    - Mode status badge (bottom-left).
    - Live brush circle preview (bottom-center).
    - Keyboard hint badge (bottom-right).
    - Toast notifications.
    - Radial palm-clear meter.
    - Dual cursor reticles for each hand.
    """

    def __init__(self, width: int = 1280, header_height: int = TOOLBAR_H):
        self.width = width
        self.header_height = header_height
        self.buttons: List[Button] = []
        self._build_buttons()
        self.toast_message: Optional[str] = None
        self.toast_expire_time: float = 0.0

    def _build_buttons(self) -> None:
        """Constructs layout of interactive rounded-pill buttons in the toolbar."""
        self.buttons = []
        items = [
            ("Cyan",  Canvas.PALETTE["Cyan"],     "color"),
            ("Pink",  Canvas.PALETTE["Neon Pink"], "color"),
            ("Green", Canvas.PALETTE["Emerald"],   "color"),
            ("Amber", Canvas.PALETTE["Amber"],     "color"),
            ("White", Canvas.PALETTE["White"],     "color"),
            ("Erase", (60, 60, 60),                "eraser"),
            ("Undo",  (80, 55, 20),                "undo"),
            ("CLEAR", (25, 25, 160),               "clear"),
        ]

        n = len(items)
        margin = 14
        gap = 8
        btn_w = (self.width - 2 * margin - (n - 1) * gap) // n
        btn_y1 = 12
        btn_y2 = self.header_height - 12

        for i, (name, bgr, action) in enumerate(items):
            x1 = margin + i * (btn_w + gap)
            self.buttons.append(Button(name, x1, btn_y1, x1 + btn_w, btn_y2, bgr, action))

    def update_dimensions(self, width: int) -> None:
        if width != self.width:
            self.width = width
            self._build_buttons()

    def show_toast(self, message: str, duration: float = 2.2) -> None:
        self.toast_message = message
        self.toast_expire_time = time.time() + duration

    def check_interaction(
        self, x: int, y: int, canvas: Canvas
    ) -> Optional[str]:
        if y > self.header_height:
            return None
        for btn in self.buttons:
            if btn.is_inside(x, y):
                if btn.action_type == "color":
                    canvas.set_color(btn.color_bgr)
                    self.show_toast(f"Color: {btn.name}")
                    return f"Selected {btn.name}"
                elif btn.action_type == "eraser":
                    canvas.set_color(Canvas.PALETTE["Eraser"])
                    self.show_toast("Eraser Activated")
                    return "Eraser"
                elif btn.action_type == "undo":
                    if canvas.undo():
                        self.show_toast("Undo!")
                    return "Undo"
                elif btn.action_type == "clear":
                    canvas.clear()
                    self.show_toast("Canvas Cleared!")
                    return "Cleared"
        return None

    # ─── DRAW TOOLBAR ────────────────────────────────────────────────────────
    def _draw_toolbar(
        self,
        frame: np.ndarray,
        active_color: Tuple[int, int, int],
    ) -> None:
        h, w = frame.shape[:2]

        # Frosted glass background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, self.header_height), BG_GLASS, -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        # Thin separator line
        cv2.line(frame, (0, self.header_height), (w, self.header_height), (50, 50, 60), 1)

        for btn in self.buttons:
            is_active = (
                (btn.action_type == "color" and btn.color_bgr == active_color)
                or (btn.action_type == "eraser" and active_color == Canvas.PALETTE["Eraser"])
            )

            # Button body (pill-shaped)
            r = (btn.y2 - btn.y1) // 2
            _draw_rounded_rect(frame, btn.x1, btn.y1, btn.x2, btn.y2, btn.color_bgr, radius=r)

            # Active neon glow ring
            if is_active:
                _draw_rounded_rect(frame, btn.x1 - 2, btn.y1 - 2, btn.x2 + 2, btn.y2 + 2,
                                   ACCENT_R, radius=r + 2, thickness=2)
                # Inner white highlight
                _draw_rounded_rect(frame, btn.x1 + 1, btn.y1 + 1, btn.x2 - 1, btn.y2 - 1,
                                   WHITE, radius=r - 1, thickness=1)
            else:
                # Subtle border for inactive
                _draw_rounded_rect(frame, btn.x1, btn.y1, btn.x2, btn.y2,
                                   (55, 55, 65), radius=r, thickness=1)

            # Button label
            dark_labels = {"Cyan", "Amber", "White", "Green"}
            fg = (20, 20, 20) if btn.name in dark_labels else WHITE
            fs = 0.44
            (tw, th), _ = cv2.getTextSize(btn.name, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            tx = btn.x1 + (btn.x2 - btn.x1 - tw) // 2
            ty = btn.y1 + (btn.y2 - btn.y1 + th) // 2
            cv2.putText(frame, btn.name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, fg, 1, cv2.LINE_AA)

    # ─── DRAW HAND STATUS BADGE ────────────────────────────────────────────
    def _draw_hand_badge(
        self, frame: np.ndarray, num_hands: int, mode: str
    ) -> None:
        h, w = frame.shape[:2]
        # Dot color based on detection
        dot_color = ACCENT_G if num_hands > 0 else (80, 80, 80)
        hand_text = f"  Air Canvas AI  •  {num_hands} Hand{'s' if num_hands != 1 else ''} Active"

        # Draw pill
        (tw, th), _ = cv2.getTextSize(hand_text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        px, py = 14, self.header_height + 16
        pill_x1, pill_y1 = px, py
        pill_x2 = px + tw + 36
        pill_y2 = py + th + 14

        _draw_rounded_rect(frame, pill_x1, pill_y1, pill_x2, pill_y2, BG_GLASS, radius=12, alpha=0.85)
        cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), (40, 40, 50), 1)

        # Animated status dot
        dot_cx = pill_x1 + 18
        dot_cy = (pill_y1 + pill_y2) // 2
        cv2.circle(frame, (dot_cx, dot_cy), 5, dot_color, -1)
        cv2.circle(frame, (dot_cx, dot_cy), 5, (255, 255, 255), 1)

        cv2.putText(frame, hand_text, (pill_x1 + 28, pill_y1 + th + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, WHITE, 1, cv2.LINE_AA)

    # ─── DRAW FPS BADGE (top-right) ────────────────────────────────────────
    def _draw_fps_badge(self, frame: np.ndarray, fps: float) -> None:
        h, w = frame.shape[:2]
        text = f"FPS  {int(fps)}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        px = w - tw - 40
        py = self.header_height + 14
        _draw_rounded_rect(frame, px - 12, py - 2, px + tw + 14, py + th + 14, BG_GLASS, radius=10, alpha=0.82)
        fps_color = ACCENT_G if fps >= 25 else ((0, 170, 255) if fps >= 15 else (0, 60, 255))
        cv2.putText(frame, text, (px, py + th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.46, fps_color, 1, cv2.LINE_AA)

    # ─── DRAW MODE BADGE (bottom-left) ────────────────────────────────────
    def _draw_mode_badge(self, frame: np.ndarray, mode: str) -> None:
        h, w = frame.shape[:2]
        # Color-coded badge
        if any(k in mode for k in ("DRAW", "PINCH")):
            bg = (10, 100, 10); fg = ACCENT_G
        elif "MOVING" in mode or "GRABBED" in mode:
            bg = (10, 70, 120); fg = ACCENT_R
        elif "PALM" in mode:
            bg = (100, 10, 10); fg = (80, 100, 255)
        elif "HOVER" in mode:
            bg = (25, 25, 40); fg = DIM
        else:
            bg = BG_GLASS; fg = DIM

        text = f"  {mode}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        px = 14
        py = h - 40
        _draw_rounded_rect(frame, px, py, px + tw + 20, py + th + 16, bg, radius=10, alpha=0.88)
        cv2.putText(frame, text, (px + 10, py + th + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, fg, 1, cv2.LINE_AA)

    # ─── DRAW BRUSH PREVIEW (bottom-center) ───────────────────────────────
    def _draw_brush_preview(
        self, frame: np.ndarray, brush_size: int, color: Tuple[int, int, int]
    ) -> None:
        h, w = frame.shape[:2]
        cx, cy = w // 2, h - 24
        # Glass pill
        _draw_rounded_rect(frame, cx - 55, cy - 20, cx + 55, cy + 18, BG_GLASS, radius=12, alpha=0.80)
        # Brush circle preview
        radius = max(4, min(brush_size // 2, 18))
        draw_color = (120, 120, 120) if color == Canvas.PALETTE["Eraser"] else color
        cv2.circle(frame, (cx - 20, cy), radius, draw_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx - 20, cy), radius, WHITE, 1, cv2.LINE_AA)
        size_label = f"{brush_size}px"
        cv2.putText(frame, size_label, (cx + 4, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, DIM, 1, cv2.LINE_AA)

    # ─── DRAW HOTKEYS HINT (bottom-right) ─────────────────────────────────
    def _draw_hint_badge(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        text = "Z Undo  C Clear  S Save  B Board  Q Quit"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        px = w - tw - 26
        py = h - 38
        _draw_rounded_rect(frame, px - 10, py - 4, px + tw + 14, py + th + 14, BG_GLASS, radius=8, alpha=0.72)
        cv2.putText(frame, text, (px, py + th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, DIM, 1, cv2.LINE_AA)

    # ─── DRAW TOAST NOTIFICATION ──────────────────────────────────────────
    def _draw_toast(self, frame: np.ndarray) -> None:
        if not self.toast_message:
            return
        now = time.time()
        if now >= self.toast_expire_time:
            self.toast_message = None
            return

        h, w = frame.shape[:2]
        (tw, th), _ = cv2.getTextSize(self.toast_message, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)
        cx = w // 2
        cy = self.header_height + 58
        bx1 = cx - tw // 2 - 20
        by1 = cy - th // 2 - 12
        bx2 = cx + tw // 2 + 20
        by2 = cy + th // 2 + 12

        _draw_rounded_rect(frame, bx1, by1, bx2, by2, (18, 18, 24), radius=12, alpha=0.88)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), ACCENT_G, 1)
        cv2.putText(frame, self.toast_message, (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, WHITE, 1, cv2.LINE_AA)

    # ─── DRAW PALM CLEAR RADIAL METER ─────────────────────────────────────
    def draw_palm_clear_progress(self, frame: np.ndarray, progress: float) -> None:
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        radius = 58

        # Frosted disc backing
        overlay = frame.copy()
        cv2.circle(overlay, center, radius + 18, (14, 14, 20), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        # Track
        cv2.circle(frame, center, radius, (50, 50, 60), 4, cv2.LINE_AA)

        # Arc (sweeps clockwise from top)
        angle = int(360 * min(1.0, progress))
        cv2.ellipse(frame, center, (radius, radius), -90, 0, angle, (0, 80, 255), 6, cv2.LINE_AA)

        # Glowing dot at arc tip
        tip_rad = math.radians(-90 + angle)
        tip_x = int(center[0] + radius * math.cos(tip_rad))
        tip_y = int(center[1] + radius * math.sin(tip_rad))
        cv2.circle(frame, (tip_x, tip_y), 8, (0, 140, 255), -1, cv2.LINE_AA)

        pct = int(progress * 100)
        cv2.putText(frame, "CLEARING", (center[0] - 38, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{pct}%", (center[0] - 16, center[1] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 180, 255), 2, cv2.LINE_AA)

    # ─── DRAW HAND CURSORS ────────────────────────────────────────────────
    def draw_cursor(
        self,
        frame: np.ndarray,
        point: Tuple[int, int],
        is_pinching: bool,
        is_snapped: bool,
        hand_label: str = "Right",
        brush_size: int = 6,
        active_color: Tuple[int, int, int] = (255, 230, 0),
    ) -> None:
        cx, cy = point
        accent = ACCENT_R if hand_label == "Right" else ACCENT_L

        if is_snapped:
            # Magnetic snap: double glowing ring in accent color
            cv2.circle(frame, (cx, cy), 16, accent, 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 8, accent, 1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 4, ACCENT_G, -1, cv2.LINE_AA)
        elif is_pinching:
            # Pinching / drawing: solid dot in active color with accent ring
            rad = max(brush_size // 2, 6)
            draw_color = (120, 120, 120) if active_color == Canvas.PALETTE["Eraser"] else active_color
            cv2.circle(frame, (cx, cy), rad, draw_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), rad + 4, accent, 2, cv2.LINE_AA)
        else:
            # Hover: crosshair reticle in accent color
            cv2.circle(frame, (cx, cy), 10, accent, 2, cv2.LINE_AA)
            cv2.line(frame, (cx - 16, cy), (cx - 12, cy), accent, 2)
            cv2.line(frame, (cx + 12, cy), (cx + 16, cy), accent, 2)
            cv2.line(frame, (cx, cy - 16), (cx, cy - 12), accent, 2)
            cv2.line(frame, (cx, cy + 12), (cx, cy + 16), accent, 2)
            cv2.circle(frame, (cx, cy), 2, accent, -1, cv2.LINE_AA)

    # ─── MASTER DRAW METHOD ───────────────────────────────────────────────
    def draw(
        self,
        frame: np.ndarray,
        current_mode: str,
        active_color: Tuple[int, int, int],
        brush_size: int,
        fps: float = 0.0,
        num_hands: int = 0,
    ) -> None:
        """Renders the full UI over the frame."""
        self._draw_toolbar(frame, active_color)
        self._draw_hand_badge(frame, num_hands, current_mode)
        self._draw_fps_badge(frame, fps)
        self._draw_mode_badge(frame, current_mode)
        self._draw_brush_preview(frame, brush_size, active_color)
        self._draw_hint_badge(frame)
        self._draw_toast(frame)
