"""
Unit tests for HandTracker module.
"""

import unittest
import numpy as np
from src.hand_tracker import HandTracker


class TestHandTracker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tracker = HandTracker()

    def test_instantiation(self):
        self.assertIn(self.tracker.backend, ["solutions", "tasks"])

    def test_process_empty_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        processed = self.tracker.find_hands(frame)
        self.assertEqual(processed.shape, (480, 640, 3))
        positions = self.tracker.find_positions(frame)
        self.assertEqual(len(positions), 0)
        fingers = self.tracker.fingers_up()
        self.assertEqual(fingers, [0, 0, 0, 0, 0])

    def test_find_distance_no_hands(self):
        dist, p1, p2 = self.tracker.find_distance(4, 8)
        self.assertEqual(dist, 0.0)


if __name__ == "__main__":
    unittest.main()
