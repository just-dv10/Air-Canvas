"""
Unit tests for Canvas and UIOverlay logic in Air Canvas AI.
These tests verify core rendering, drawing state machines, and button interactions
without requiring a live physical webcam.
"""

import os
import unittest
import numpy as np
from src.canvas import Canvas
from src.ui_overlay import UIOverlay


class TestCanvas(unittest.TestCase):
    def setUp(self):
        self.canvas = Canvas(width=640, height=480)

    def test_canvas_initialization(self):
        self.assertEqual(self.canvas.width, 640)
        self.assertEqual(self.canvas.height, 480)
        self.assertEqual(self.canvas.canvas.shape, (480, 640, 3))
        # Initially empty (all zeros)
        self.assertEqual(np.count_nonzero(self.canvas.canvas), 0)

    def test_color_switching(self):
        self.canvas.set_color(Canvas.PALETTE["Neon Pink"])
        self.assertEqual(self.canvas.current_color, Canvas.PALETTE["Neon Pink"])
        self.assertFalse(self.canvas.is_eraser_active())

        # Switch to Eraser
        self.canvas.set_color(Canvas.PALETTE["Eraser"])
        self.assertTrue(self.canvas.is_eraser_active())

    def test_drawing_strokes(self):
        # Draw from (100, 100) to (150, 150)
        self.canvas.set_color(Canvas.PALETTE["Cyan"])
        self.canvas.draw((100, 100))
        self.canvas.draw((150, 150))

        # Canvas should now have non-zero pixels
        self.assertGreater(np.count_nonzero(self.canvas.canvas), 0)

    def test_clear_canvas(self):
        self.canvas.draw((50, 50))
        self.canvas.draw((100, 100))
        self.assertGreater(np.count_nonzero(self.canvas.canvas), 0)

        self.canvas.clear()
        self.assertEqual(np.count_nonzero(self.canvas.canvas), 0)
        self.assertIsNone(self.canvas.prev_point)

    def test_brush_thickness_bounds(self):
        self.canvas.set_brush_thickness(1)  # Below min 2
        self.assertEqual(self.canvas.brush_thickness, 2)

        self.canvas.set_brush_thickness(100)  # Above max 50
        self.assertEqual(self.canvas.brush_thickness, 50)

    def test_blend_with_frame(self):
        # Create a dummy frame (e.g. solid grey)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        self.canvas.set_color(Canvas.PALETTE["Cyan"])
        self.canvas.draw((200, 200))
        self.canvas.draw((250, 250))

        blended = self.canvas.blend(frame)
        self.assertEqual(blended.shape, (480, 640, 3))
        # Ensure blended frame contains changes
        self.assertFalse(np.array_equal(blended, frame))

    def test_save_snapshot(self):
        self.canvas.draw((10, 10))
        self.canvas.draw((30, 30))
        test_dir = "test_output_saved"
        saved_file = self.canvas.save_snapshot(output_dir=test_dir)
        try:
            self.assertTrue(os.path.exists(saved_file))
            self.assertTrue(saved_file.endswith(".png"))
        finally:
            if os.path.exists(saved_file):
                os.remove(saved_file)
            if os.path.exists(test_dir):
                os.rmdir(test_dir)


class TestUIOverlay(unittest.TestCase):
    def setUp(self):
        self.ui = UIOverlay(width=640, header_height=80)
        self.canvas = Canvas(width=640, height=480)

    def test_buttons_created(self):
        self.assertGreaterEqual(len(self.ui.buttons), 6)

    def test_button_interaction(self):
        # Click the first button (Cyan)
        btn1 = self.ui.buttons[0]
        mid_x = (btn1.x1 + btn1.x2) // 2
        mid_y = (btn1.y1 + btn1.y2) // 2

        action = self.ui.check_interaction(mid_x, mid_y, self.canvas)
        self.assertIsNotNone(action)
        self.assertEqual(self.canvas.current_color, btn1.color_bgr)

    def test_clear_button_interaction(self):
        # Draw something first
        self.canvas.draw((10, 10))
        self.canvas.draw((20, 20))
        self.assertGreater(np.count_nonzero(self.canvas.canvas), 0)

        # Find Clear button
        clear_btn = next(b for b in self.ui.buttons if b.name == "CLEAR")
        cx = (clear_btn.x1 + clear_btn.x2) // 2
        cy = (clear_btn.y1 + clear_btn.y2) // 2

        action = self.ui.check_interaction(cx, cy, self.canvas)
        self.assertEqual(action, "Cleared")
        self.assertEqual(np.count_nonzero(self.canvas.canvas), 0)


if __name__ == "__main__":
    unittest.main()
