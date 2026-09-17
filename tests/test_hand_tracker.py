"""
Unit tests for HandTracker (two-hand edition).
"""

import unittest
import numpy as np
from src.hand_tracker import HandTracker


class TestHandTracker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tracker = HandTracker(max_hands=2)

    def test_instantiation(self):
        self.assertIsNotNone(self.tracker)
        self.assertEqual(self.tracker.max_hands, 2)
        self.assertIn("Right", self.tracker.stabilizers)
        self.assertIn("Left", self.tracker.stabilizers)

    def test_process_empty_frame_returns_no_hands(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        processed = self.tracker.find_hands(frame, draw=False)
        self.assertEqual(processed.shape, (720, 1280, 3))
        self.assertEqual(len(self.tracker.hands_data), 0)
        self.assertEqual(self.tracker.find_positions(frame, 0), [])

    def test_backwards_compat_methods_return_defaults(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.tracker.find_hands(frame, draw=False)
        is_pinch, dist, mid = self.tracker.is_pinching()
        self.assertFalse(is_pinch)
        self.assertFalse(self.tracker.is_open_palm())
        self.assertEqual(self.tracker.fingers_up(), [0, 0, 0, 0, 0])

    def test_find_distance_no_hands(self):
        dist, p1, p2 = self.tracker.find_distance(4, 8)
        self.assertEqual(dist, 0.0)


if __name__ == "__main__":
    unittest.main()
