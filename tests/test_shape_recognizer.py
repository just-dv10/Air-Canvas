"""
Unit tests for ShapeRecognizer and geometric shape fitting.
"""

import math
import unittest
from src.shape_recognizer import ShapeRecognizer


class TestShapeRecognizer(unittest.TestCase):
    def setUp(self):
        self.recognizer = ShapeRecognizer(snap_threshold=25.0)

    def test_recognize_straight_line(self):
        # Generate 20 collinear points along a line
        line_pts = [(i * 10, i * 10) for i in range(20)]
        shape = self.recognizer.recognize(line_pts, color=(255, 0, 0), thickness=5)
        self.assertEqual(shape.shape_type, "line")
        self.assertEqual(shape.points[0], (0, 0))
        self.assertEqual(shape.points[-1], (190, 190))
        self.assertIn((0, 0), shape.anchors)
        self.assertIn((190, 190), shape.anchors)

    def test_recognize_rectangle(self):
        # Generate points forming a rough box
        box_pts = []
        # Top edge
        for x in range(100, 250, 10):
            box_pts.append((x, 100))
        # Right edge
        for y in range(100, 200, 10):
            box_pts.append((250, y))
        # Bottom edge
        for x in range(250, 100, -10):
            box_pts.append((x, 200))
        # Left edge
        for y in range(200, 100, -10):
            box_pts.append((100, y))
        box_pts.append((100, 100))

        shape = self.recognizer.recognize(box_pts, color=(0, 255, 0), thickness=4)
        self.assertEqual(shape.shape_type, "rectangle")
        self.assertEqual(len(shape.points), 4)
        self.assertGreaterEqual(len(shape.anchors), 4)

    def test_recognize_circle(self):
        # Generate points along a circle
        cx, cy, r = 200, 200, 60
        circle_pts = []
        for deg in range(0, 360, 10):
            rad = math.radians(deg)
            circle_pts.append((int(cx + r * math.cos(rad)), int(cy + r * math.sin(rad))))
        circle_pts.append(circle_pts[0])

        shape = self.recognizer.recognize(circle_pts, color=(0, 0, 255), thickness=3)
        self.assertEqual(shape.shape_type, "circle")
        self.assertIn("center", shape.metadata)
        self.assertAlmostEqual(shape.metadata["radius"], 60, delta=10)

    def test_recognize_triangle(self):
        # 3 corners
        p1, p2, p3 = (100, 300), (250, 100), (400, 300)
        tri_pts = []
        # p1 to p2
        for t in range(10):
            tri_pts.append((int(p1[0] + t/10 * (p2[0] - p1[0])), int(p1[1] + t/10 * (p2[1] - p1[1]))))
        # p2 to p3
        for t in range(10):
            tri_pts.append((int(p2[0] + t/10 * (p3[0] - p2[0])), int(p2[1] + t/10 * (p3[1] - p2[1]))))
        # p3 to p1
        for t in range(10):
            tri_pts.append((int(p3[0] + t/10 * (p1[0] - p3[0])), int(p3[1] + t/10 * (p1[1] - p3[1]))))
        tri_pts.append(p1)

        shape = self.recognizer.recognize(tri_pts, color=(255, 255, 0), thickness=4)
        self.assertEqual(shape.shape_type, "triangle")
        self.assertEqual(len(shape.points), 3)

    def test_snap_to_anchor(self):
        anchors = [(100, 100), (300, 300)]
        # Nearby point (105, 98) -> distance is ~5.4 < 25
        nearby = (105, 98)
        snapped, did_snap = self.recognizer.snap_point(nearby, anchors)
        self.assertTrue(did_snap)
        self.assertEqual(snapped, (100, 100))

        # Far point (150, 150) -> distance is ~70 > 25
        far = (150, 150)
        snapped_far, did_snap_far = self.recognizer.snap_point(far, anchors)
        self.assertFalse(did_snap_far)
        self.assertEqual(snapped_far, far)


if __name__ == "__main__":
    unittest.main()
