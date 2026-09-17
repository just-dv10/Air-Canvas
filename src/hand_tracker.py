"""
Hand Tracker Module supporting both MediaPipe Tasks API and Legacy Solutions API.
Provides coordinate stabilization, pinch detection, and palm gesture recognition.
"""

import os
import math
import urllib.request
from typing import List, Tuple, Optional
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


class HandTracker:
    """
    Universal hand landmark detector with coordinate stabilization,
    pinch-to-draw gesture recognition, and open-palm detection.
    """

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.6,
        model_dir: str = "models",
    ):
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.model_dir = model_dir

        self.tip_ids = [4, 8, 12, 16, 20]
        self.lm_list: List[Tuple[int, int, int]] = []
        self.is_right_hand: bool = True

        # Adaptive stabilizers for primary fingertips to eliminate jitter
        self.index_stabilizer = AdaptiveStabilizer()
        self.thumb_stabilizer = AdaptiveStabilizer()

        # Check if legacy mp.solutions exists
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.backend = "solutions"
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_hands,
                min_detection_confidence=self.detection_confidence,
                min_tracking_confidence=self.tracking_confidence,
            )
            self.mp_draw = mp.solutions.drawing_utils
        else:
            self.backend = "tasks"
            self._init_tasks_backend()

    def _init_tasks_backend(self) -> None:
        """Initializes MediaPipe Tasks HandLandmarker, downloading weights if needed."""
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, "hand_landmarker.task")

        if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
            print("[INFO] Downloading MediaPipe hand landmark model (~7.8MB)...")
            urllib.request.urlretrieve(self.MODEL_URL, model_path)
            print("[INFO] Model download complete.")

        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options

        options = vision.HandLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def reset_stabilizers(self) -> None:
        """Resets tracking stabilizers when hand leaves the camera frame."""
        self.index_stabilizer.reset()
        self.thumb_stabilizer.reset()

    def find_hands(self, frame: cv2.Mat, draw: bool = True) -> cv2.Mat:
        """
        Detects hand landmarks in the frame, applies stabilization,
        and optionally renders the skeleton.
        """
        self.lm_list = []
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        raw_landmarks = []
        if self.backend == "solutions":
            results = self.hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                primary_hand = results.multi_hand_landmarks[0]
                if results.multi_handedness:
                    self.is_right_hand = results.multi_handedness[0].classification[0].label == "Right"
                raw_landmarks = primary_hand.landmark
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.landmarker.detect(mp_image)
            if results.hand_landmarks and len(results.hand_landmarks) > 0:
                primary_hand = results.hand_landmarks[0]
                if results.handedness and len(results.handedness) > 0:
                    self.is_right_hand = results.handedness[0][0].category_name == "Right"
                raw_landmarks = primary_hand

        if raw_landmarks:
            for idx, lm in enumerate(raw_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Apply adaptive stabilization to Index tip (8) and Thumb tip (4)
                if idx == 8:
                    cx, cy = self.index_stabilizer.update(cx, cy)
                elif idx == 4:
                    cx, cy = self.thumb_stabilizer.update(cx, cy)
                self.lm_list.append((idx, cx, cy))

            if draw:
                self._draw_skeleton(frame)
        else:
            self.reset_stabilizers()

        return frame

    def _draw_skeleton(self, frame: cv2.Mat) -> None:
        """Renders clean skeleton connecting the 21 landmarks."""
        if len(self.lm_list) < 21:
            return

        for p1, p2 in HAND_CONNECTIONS:
            pt1 = (self.lm_list[p1][1], self.lm_list[p1][2])
            pt2 = (self.lm_list[p2][1], self.lm_list[p2][2])
            cv2.line(frame, pt1, pt2, (0, 200, 255), 2, cv2.LINE_AA)

        for idx, x, y in self.lm_list:
            radius = 6 if idx in self.tip_ids else 4
            color = (255, 0, 180) if idx == 8 else ((0, 255, 255) if idx == 4 else (0, 255, 0))
            cv2.circle(frame, (x, y), radius, color, cv2.FILLED)

    def find_positions(
        self, frame: cv2.Mat, hand_index: int = 0
    ) -> List[Tuple[int, int, int]]:
        """Returns the list of 21 landmark positions (id, x, y)."""
        return self.lm_list

    def get_hand_scale(self) -> float:
        """
        Returns reference distance between wrist (0) and middle MCP (9)
        to normalize gesture thresholds across varying camera distances.
        """
        if len(self.lm_list) < 21:
            return 100.0
        w_x, w_y = self.lm_list[0][1], self.lm_list[0][2]
        m_x, m_y = self.lm_list[9][1], self.lm_list[9][2]
        scale = math.hypot(m_x - w_x, m_y - w_y)
        return max(30.0, scale)

    def is_pinching(
        self, threshold_ratio: float = 0.28
    ) -> Tuple[bool, float, Tuple[int, int]]:
        """
        Checks if index finger (8) and thumb (4) are pinching together.
        Returns:
            (is_pinch, normalized_distance, pinch_midpoint)
        """
        if len(self.lm_list) < 21:
            return False, 1.0, (0, 0)

        # Index tip & Thumb tip coordinates
        ix, iy = self.lm_list[8][1], self.lm_list[8][2]
        tx, ty = self.lm_list[4][1], self.lm_list[4][2]

        raw_dist = math.hypot(ix - tx, iy - ty)
        hand_scale = self.get_hand_scale()
        norm_dist = raw_dist / hand_scale

        midpoint = ((ix + tx) // 2, (iy + ty) // 2)
        is_pinch = norm_dist < threshold_ratio
        return is_pinch, norm_dist, midpoint

    def fingers_up(self) -> List[int]:
        """
        Determines which fingers are raised.
        Returns [thumb, index, middle, ring, pinky] (1 = extended, 0 = folded).
        """
        fingers: List[int] = []
        if len(self.lm_list) < 21:
            return [0, 0, 0, 0, 0]

        # Thumb: compare x-coordinate of tip (4) with joint (3)
        if self.is_right_hand:
            fingers.append(1 if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1] else 0)
        else:
            fingers.append(1 if self.lm_list[self.tip_ids[0]][1] < self.lm_list[self.tip_ids[0] - 1][1] else 0)

        # Other 4 fingers: tip y-coordinate must be above PIP joint (id - 2)
        for i in range(1, 5):
            tip_id = self.tip_ids[i]
            pip_id = tip_id - 2
            if self.lm_list[tip_id][2] < self.lm_list[pip_id][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def is_open_palm(self) -> bool:
        """
        Checks if the hand is an open palm (all 5 fingers up & spread).
        """
        if len(self.lm_list) < 21:
            return False

        fingers = self.fingers_up()
        if sum(fingers) != 5:
            return False

        # Additional check: Index tip, Middle tip, Ring tip, Pinky tip well separated
        ix = self.lm_list[8][1]
        px = self.lm_list[20][1]
        spread = abs(px - ix)
        return spread > (0.6 * self.get_hand_scale())

    def find_distance(
        self, p1_id: int, p2_id: int
    ) -> Tuple[float, Tuple[int, int], Tuple[int, int]]:
        """Calculates euclidean distance between two landmark points."""
        if len(self.lm_list) < 21:
            return 0.0, (0, 0), (0, 0)
        x1, y1 = self.lm_list[p1_id][1], self.lm_list[p1_id][2]
        x2, y2 = self.lm_list[p2_id][1], self.lm_list[p2_id][2]
        length = math.hypot(x2 - x1, y2 - y1)
        return length, (x1, y1), (x2, y2)

