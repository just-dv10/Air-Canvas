"""
Smart Shape Recognition and Snapping Engine.
Analyzes hand-drawn strokes to automatically recognize and fit geometric shapes
(Straight Line, Rectangle, Circle, Triangle, or Freehand Curve) with anchor snapping.
"""

import math
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np


class RecognizedShape:
    """Represents a committed recognized geometric shape."""

    def __init__(
        self,
        shape_type: str,
        points: List[Tuple[int, int]],
        anchors: List[Tuple[int, int]],
        color: Tuple[int, int, int],
        thickness: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.shape_type = shape_type  # 'line', 'rectangle', 'circle', 'triangle', 'curve'
        self.points = points          # Path points or corners to draw
        self.anchors = anchors        # Connectable vertices / snapping points
        self.color = color
        self.thickness = thickness
        self.metadata = metadata or {}

    def translate(self, dx: int, dy: int) -> None:
        """Moves all shape points and anchors by (dx, dy)."""
        self.points = [(x + dx, y + dy) for (x, y) in self.points]
        self.anchors = [(x + dx, y + dy) for (x, y) in self.anchors]
        if "center" in self.metadata:
            cx, cy = self.metadata["center"]
            self.metadata["center"] = (cx + dx, cy + dy)
        if "x" in self.metadata and "y" in self.metadata:
            self.metadata["x"] += dx
            self.metadata["y"] += dy

    def get_bounds(self) -> Tuple[int, int, int, int]:
        """Returns (min_x, min_y, max_x, max_y) bounding box."""
        if self.shape_type == "circle":
            cx, cy = self.metadata.get("center", self.points[0] if self.points else (0, 0))
            r = self.metadata.get("radius", 20)
            return (cx - r, cy - r, cx + r, cy + r)

        if not self.points:
            return (0, 0, 0, 0)

        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))

    def contains_point(self, pt: Tuple[int, int], margin: float = 22.0) -> bool:
        """Checks if a point (x, y) is inside or near this shape."""
        px, py = pt

        if self.shape_type == "circle":
            cx, cy = self.metadata.get("center", self.points[0] if self.points else (0, 0))
            r = self.metadata.get("radius", 20)
            return math.hypot(px - cx, py - cy) <= (r + margin)

        elif self.shape_type in ("rectangle", "triangle"):
            if not self.points:
                return False
            contour = np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))
            dist = cv2.pointPolygonTest(contour, (float(px), float(py)), True)
            return dist >= -margin

        elif self.shape_type in ("line", "curve"):
            if len(self.points) < 2:
                return False
            for i in range(len(self.points) - 1):
                p1 = self.points[i]
                p2 = self.points[i + 1]
                # Distance from point to line segment
                l2 = (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2
                if l2 == 0:
                    d = math.hypot(px - p1[0], py - p1[1])
                else:
                    t = max(0.0, min(1.0, ((px - p1[0]) * (p2[0] - p1[0]) + (py - p1[1]) * (p2[1] - p1[1])) / l2))
                    proj_x = p1[0] + t * (p2[0] - p1[0])
                    proj_y = p1[1] + t * (p2[1] - p1[1])
                    d = math.hypot(px - proj_x, py - proj_y)
                if d <= margin:
                    return True
            return False

        return False



class ShapeRecognizer:
    """
    Classifies raw stroke trajectories into clean geometric shapes
    and provides vertex snapping to easily connect multi-shape artwork.
    """

    def __init__(self, snap_threshold: float = 28.0):
        self.snap_threshold = snap_threshold

    @staticmethod
    def calculate_path_length(points: List[Tuple[int, int]]) -> float:
        """Computes cumulative Euclidean distance along the stroke path."""
        if len(points) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(points)):
            total += math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
        return total

    def snap_point(
        self,
        point: Tuple[int, int],
        anchors: List[Tuple[int, int]],
    ) -> Tuple[Tuple[int, int], bool]:
        """
        If point is within snap_threshold of any anchor, snaps to the closest anchor.
        Returns (snapped_point, did_snap).
        """
        if not anchors:
            return point, False

        best_anchor = None
        min_dist = float("inf")
        for a in anchors:
            d = math.hypot(point[0] - a[0], point[1] - a[1])
            if d < min_dist:
                min_dist = d
                best_anchor = a

        if best_anchor and min_dist <= self.snap_threshold:
            return best_anchor, True
        return point, False

    def recognize(
        self,
        points: List[Tuple[int, int]],
        color: Tuple[int, int, int],
        thickness: int,
        existing_anchors: Optional[List[Tuple[int, int]]] = None,
    ) -> RecognizedShape:
        """
        Recognizes shape from raw stroke points, applies snapping to endpoints,
        and returns a clean RecognizedShape object.
        """
        if not points:
            return RecognizedShape("empty", [], [], color, thickness)

        existing_anchors = existing_anchors or []

        # Snap start and end points to existing anchors if near
        snapped_start, _ = self.snap_point(points[0], existing_anchors)
        snapped_end, _ = self.snap_point(points[-1], existing_anchors)
        points[0] = snapped_start
        points[-1] = snapped_end

        if len(points) < 4:
            # Very short tap/stroke -> treat as a line or dot
            anchors = [points[0], points[-1]]
            return RecognizedShape("line", [points[0], points[-1]], anchors, color, thickness)

        path_length = self.calculate_path_length(points)
        endpoint_dist = math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])

        # Check if shape is closed (endpoints close relative to total length)
        is_closed = (endpoint_dist < max(35.0, 0.22 * path_length)) and len(points) >= 5


        contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

        if is_closed:
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)

            if perimeter > 0:
                circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
            else:
                circularity = 0.0

            # Polygon Approximation (Triangle / Rectangle / Polygon)
            epsilon = 0.045 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            num_vertices = len(approx)

            # TRIANGLE (3 vertices)
            if num_vertices == 3:
                vertices = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
                return RecognizedShape(
                    shape_type="triangle",
                    points=vertices,
                    anchors=vertices,
                    color=color,
                    thickness=thickness,
                )

            # RECTANGLE / QUADRILATERAL (4 vertices)
            elif num_vertices == 4:
                # Use bounding box for crisp right angles
                x, y, w, h = cv2.boundingRect(contour)
                corners = [
                    (x, y),
                    (x + w, y),
                    (x + w, y + h),
                    (x, y + h),
                ]
                anchors = corners + [
                    (x + w // 2, y),
                    (x + w // 2, y + h),
                    (x, y + h // 2),
                    (x + w, y + h // 2),
                ]
                return RecognizedShape(
                    shape_type="rectangle",
                    points=corners,
                    anchors=anchors,
                    color=color,
                    thickness=thickness,
                    metadata={"x": x, "y": y, "w": w, "h": h},
                )

            # CIRCLE / OVAL (5+ vertices with high circularity)
            elif circularity > 0.65 or num_vertices >= 6:
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                center = (int(cx), int(cy))
                r = int(radius)
                anchors = [
                    (center[0], center[1] - r),  # Top
                    (center[0], center[1] + r),  # Bottom
                    (center[0] - r, center[1]),  # Left
                    (center[0] + r, center[1]),  # Right
                    center,                      # Center
                ]
                return RecognizedShape(
                    shape_type="circle",
                    points=[center],
                    anchors=anchors,
                    color=color,
                    thickness=thickness,
                    metadata={"center": center, "radius": r},
                )


        # OPEN SHAPES: Straight Line vs Freehand Curve
        straightness = endpoint_dist / max(1.0, path_length)

        # 3. STRAIGHT LINE
        if straightness > 0.85:
            start_pt = points[0]
            end_pt = points[-1]
            mid_pt = ((start_pt[0] + end_pt[0]) // 2, (start_pt[1] + end_pt[1]) // 2)
            anchors = [start_pt, end_pt, mid_pt]
            return RecognizedShape(
                shape_type="line",
                points=[start_pt, end_pt],
                anchors=anchors,
                color=color,
                thickness=thickness,
            )

        # 4. FREEHAND CURVE
        # Downsample points for smooth curve representation
        downsampled = points[:: max(1, len(points) // 40)]
        if downsampled[-1] != points[-1]:
            downsampled.append(points[-1])
        anchors = [downsampled[0], downsampled[-1]]
        return RecognizedShape(
            shape_type="curve",
            points=downsampled,
            anchors=anchors,
            color=color,
            thickness=thickness,
        )
