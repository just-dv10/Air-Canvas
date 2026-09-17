"""
Adaptive Coordinate Stabilizer Module.
Applies velocity-dependent Exponential Smoothing (EMA) to eliminate micro-jitters
when drawing or hovering, while preserving instantaneous responsiveness during fast motions.
"""

from typing import Tuple, Optional
import math


class AdaptiveStabilizer:
    """
    Dynamically adjusts smoothing weight based on velocity.
    - Low velocity (subtle hand tremors / holding still): high smoothing (low alpha) for rock-solid stability.
    - High velocity (sweeping strokes / fast hand motion): low smoothing (high alpha) for zero lag.
    """

    def __init__(
        self,
        min_alpha: float = 0.25,
        max_alpha: float = 0.85,
        velocity_threshold: float = 30.0,
    ):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.velocity_threshold = velocity_threshold

        self.filtered_x: Optional[float] = None
        self.filtered_y: Optional[float] = None

    def reset(self) -> None:
        """Clears the internal history."""
        self.filtered_x = None
        self.filtered_y = None

    def update(self, raw_x: int, raw_y: int) -> Tuple[int, int]:
        """
        Updates filter with a new raw (x, y) coordinate.
        Returns the stabilized integer coordinate (x, y).
        """
        if self.filtered_x is None or self.filtered_y is None:
            self.filtered_x = float(raw_x)
            self.filtered_y = float(raw_y)
            return raw_x, raw_y

        # Compute Euclidean distance from previous filtered coordinate (velocity proxy)
        dx = raw_x - self.filtered_x
        dy = raw_y - self.filtered_y
        dist = math.hypot(dx, dy)

        # Compute dynamic alpha between min_alpha and max_alpha
        ratio = min(1.0, dist / max(1.0, self.velocity_threshold))
        alpha = self.min_alpha + ratio * (self.max_alpha - self.min_alpha)

        # Apply exponential moving average
        self.filtered_x = alpha * raw_x + (1.0 - alpha) * self.filtered_x
        self.filtered_y = alpha * raw_y + (1.0 - alpha) * self.filtered_y

        return int(round(self.filtered_x)), int(round(self.filtered_y))
