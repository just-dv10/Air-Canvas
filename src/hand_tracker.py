"""
Universal Hand Tracker Module supporting simultaneous 2-Hand Detection.
Features independent landmark stabilization, per-hand pinch tracking,
open-palm recognition, and distinctive neon skeleton visualization.
"""

import math
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
import mediapipe as mp
from src.stabilizer import AdaptiveStabilizer


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]


class HandData:
    """Encapsulates tracking states for a single detected hand."""

    def __init__(
        self,
        label: str,  # "Left" or "Right"
        lm_list: List[Tuple[int, int, int]],
        is_pinch: bool,
        pinch_dist: float,
        pinch_midpoint: Tuple[int, int],
        is_open_palm: bool,
        fingers: List[int],
    ):
        self.label = label
        self.lm_list = lm_list
        self.is_pinch = is_pinch
        self.pinch_dist = pinch_dist
        self.pinch_midpoint = pinch_midpoint
        self.is_open_palm = is_open_palm
        self.fingers = fingers


class HandTracker:
    """
    Tracks up to 2 hands simultaneously with dedicated coordinate stabilizers
    and distinct visual styling for Left and Right hands.
    """

    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
    ):
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.tip_ids = [4, 8, 12, 16, 20]
        self.lm_list: List[Tuple[int, int, int]] = []
        self.hands_data: List[HandData] = []

        # Independent stabilizers for Left and Right hands
        self.stabilizers = {
            "Right": {"index": AdaptiveStabilizer(), "thumb": AdaptiveStabilizer()},
            "Left": {"index": AdaptiveStabilizer(), "thumb": AdaptiveStabilizer()},
        }

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )

    def reset_stabilizers(self) -> None:
        """Resets tracking stabilizers."""
        for side in ("Right", "Left"):
            self.stabilizers[side]["index"].reset()
            self.stabilizers[side]["thumb"].reset()

    def find_hands(self, frame: cv2.Mat, draw: bool = True) -> cv2.Mat:
        """
        Detects both hands in the frame, updates stabilized landmarks,
        and renders neon skeletons.
        """
        self.lm_list = []
        self.hands_data = []

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            num_detected = len(results.multi_hand_landmarks)

            for i in range(num_detected):
                raw_landmarks = results.multi_hand_landmarks[i]

                # Determine handedness (in mirrored display view)
                label = "Right"
                if results.multi_handedness and i < len(results.multi_handedness):
                    # MediaPipe classifies from user perspective; in mirrored frame, flip label
                    raw_label = results.multi_handedness[i].classification[0].label
                    label = "Right" if raw_label == "Left" else "Left"

                # Extract and stabilize landmarks
                hand_lms: List[Tuple[int, int, int]] = []
                idx_stab = self.stabilizers[label]["index"]
                thb_stab = self.stabilizers[label]["thumb"]

                for idx, lm in enumerate(raw_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if idx == 8:
                        cx, cy = idx_stab.update(cx, cy)
                    elif idx == 4:
                        cx, cy = thb_stab.update(cx, cy)
                    hand_lms.append((idx, cx, cy))

                # Analyze gestures for this hand
                is_pinch, norm_dist, midpoint = self._calc_pinch(hand_lms)
                fingers = self._calc_fingers_up(hand_lms, label)
                is_palm = self._calc_open_palm(hand_lms, fingers)

                hand_obj = HandData(
                    label=label,
                    lm_list=hand_lms,
                    is_pinch=is_pinch,
                    pinch_dist=norm_dist,
                    pinch_midpoint=midpoint,
                    is_open_palm=is_palm,
                    fingers=fingers,
                )
                self.hands_data.append(hand_obj)

                # Draw skeleton
                if draw:
                    self._draw_hand_skeleton(frame, hand_lms, label, is_pinch, midpoint)

            # Keep primary hand landmarks for backwards-compatibility
            if self.hands_data:
                self.lm_list = self.hands_data[0].lm_list
        else:
            self.reset_stabilizers()

        return frame

    def _draw_hand_skeleton(
        self,
        frame: cv2.Mat,
        lms: List[Tuple[int, int, int]],
        label: str,
        is_pinch: bool,
        midpoint: Tuple[int, int],
    ) -> None:
        """Renders distinctive neon skeleton for Right (Cyan) or Left (Magenta) hand."""
        if len(lms) < 21:
            return

        # Color theme: Right Hand = Cyan (255, 230, 0), Left Hand = Magenta (220, 20, 255)
        bone_color = (255, 230, 0) if label == "Right" else (220, 20, 255)
        joint_color = (0, 255, 255) if label == "Right" else (255, 120, 255)

        # Draw bones
        for p1, p2 in HAND_CONNECTIONS:
            pt1 = (lms[p1][1], lms[p1][2])
            pt2 = (lms[p2][1], lms[p2][2])
            cv2.line(frame, pt1, pt2, bone_color, 2, cv2.LINE_AA)

        # Draw joints
        for idx, x, y in lms:
            rad = 5 if idx in self.tip_ids else 3
            cv2.circle(frame, (x, y), rad, joint_color, cv2.FILLED)

        # Draw label tag near wrist (landmark 0)
        wx, wy = lms[0][1], lms[0][2]
        badge_text = f"{label} Hand"
        cv2.putText(
            frame,
            badge_text,
            (wx - 30, wy + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            bone_color,
            1,
            cv2.LINE_AA,
        )

    def _calc_pinch(
        self, lms: List[Tuple[int, int, int]], threshold_ratio: float = 0.35, min_pixel_dist: float = 42.0
    ) -> Tuple[bool, float, Tuple[int, int]]:
        """Calculates pinch between Index (8) and Thumb (4)."""
        if len(lms) < 21:
            return False, 1.0, (0, 0)

        ix, iy = lms[8][1], lms[8][2]
        tx, ty = lms[4][1], lms[4][2]

        raw_dist = math.hypot(ix - tx, iy - ty)
        # Reference scale: distance between wrist (0) and middle MCP (9)
        scale = max(30.0, math.hypot(lms[9][1] - lms[0][1], lms[9][2] - lms[0][2]))
        norm_dist = raw_dist / scale

        midpoint = ((ix + tx) // 2, (iy + ty) // 2)
        is_pinch = (norm_dist < threshold_ratio) or (raw_dist < min_pixel_dist)
        return is_pinch, norm_dist, midpoint

    def _calc_fingers_up(self, lms: List[Tuple[int, int, int]], label: str) -> List[int]:
        """Returns [thumb, index, middle, ring, pinky] (1 = extended)."""
        if len(lms) < 21:
            return [0, 0, 0, 0, 0]

        fingers = []
        # Thumb: compare x-coordinate of tip (4) with joint (3)
        if label == "Right":
            fingers.append(1 if lms[4][1] > lms[3][1] else 0)
        else:
            fingers.append(1 if lms[4][1] < lms[3][1] else 0)

        # 4 fingers: tip y must be above PIP joint (id - 2)
        for i in range(1, 5):
            tip = self.tip_ids[i]
            pip = tip - 2
            fingers.append(1 if lms[tip][2] < lms[pip][2] else 0)

        return fingers

    def _calc_open_palm(self, lms: List[Tuple[int, int, int]], fingers: List[int]) -> bool:
        """Checks if hand is spread open (all 5 fingers extended)."""
        if sum(fingers) != 5 or len(lms) < 21:
            return False
        scale = max(30.0, math.hypot(lms[9][1] - lms[0][1], lms[9][2] - lms[0][2]))
        spread = abs(lms[20][1] - lms[8][1])
        return spread > (0.6 * scale)

    # Backwards-compatible utility methods
    def is_pinching(self) -> Tuple[bool, float, Tuple[int, int]]:
        if self.hands_data:
            return self.hands_data[0].is_pinch, self.hands_data[0].pinch_dist, self.hands_data[0].pinch_midpoint
        return False, 1.0, (0, 0)

    def is_open_palm(self) -> bool:
        if self.hands_data:
            return self.hands_data[0].is_open_palm
        return False

    def fingers_up(self) -> List[int]:
        if self.hands_data:
            return self.hands_data[0].fingers
        return [0, 0, 0, 0, 0]

    def find_positions(self, frame: cv2.Mat, hand_index: int = 0) -> List[Tuple[int, int, int]]:
        if self.hands_data and hand_index < len(self.hands_data):
            return self.hands_data[hand_index].lm_list
        return []

    def find_distance(self, p1_id: int, p2_id: int) -> Tuple[float, Tuple[int, int], Tuple[int, int]]:
        if not self.lm_list:
            return 0.0, (0, 0), (0, 0)
        x1, y1 = self.lm_list[p1_id][1], self.lm_list[p1_id][2]
        x2, y2 = self.lm_list[p2_id][1], self.lm_list[p2_id][2]
        return math.hypot(x2 - x1, y2 - y1), (x1, y1), (x2, y2)
