"""
Hand Tracker Module supporting both MediaPipe Tasks API and Legacy Solutions API.
Provides universal compatibility across Python 3.8 - 3.12+ and all OS platforms.
"""

import os
import math
import urllib.request
from typing import List, Tuple, Optional
import cv2
import numpy as np
import mediapipe as mp


# Standard hand connections between 21 landmarks
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
    Universal hand landmark detector with support for modern MediaPipe Tasks
    and legacy MediaPipe Solutions.
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

    def find_hands(self, frame: cv2.Mat, draw: bool = True) -> cv2.Mat:
        """
        Detects hand landmarks in the frame and optionally renders the skeleton.
        """
        self.lm_list = []
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.backend == "solutions":
            results = self.hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                # Primary hand
                primary_hand = results.multi_hand_landmarks[0]
                if results.multi_handedness:
                    self.is_right_hand = results.multi_handedness[0].classification[0].label == "Right"

                for idx, lm in enumerate(primary_hand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lm_list.append((idx, cx, cy))

                if draw:
                    self.mp_draw.draw_landmarks(
                        frame,
                        primary_hand,
                        self.mp_hands.HAND_CONNECTIONS,
                    )
        else:
            # Tasks API backend
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.landmarker.detect(mp_image)

            if results.hand_landmarks and len(results.hand_landmarks) > 0:
                primary_hand = results.hand_landmarks[0]
                if results.handedness and len(results.handedness) > 0:
                    self.is_right_hand = results.handedness[0][0].category_name == "Right"

                for idx, lm in enumerate(primary_hand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lm_list.append((idx, cx, cy))

                if draw:
                    self._draw_skeleton(frame)

        return frame

    def _draw_skeleton(self, frame: cv2.Mat) -> None:
        """Custom clean drawing for hand skeleton using OpenCV."""
        if len(self.lm_list) < 21:
            return

        # Draw bones / connection lines
        for p1, p2 in HAND_CONNECTIONS:
            pt1 = (self.lm_list[p1][1], self.lm_list[p1][2])
            pt2 = (self.lm_list[p2][1], self.lm_list[p2][2])
            cv2.line(frame, pt1, pt2, (0, 215, 255), 2, cv2.LINE_AA)

        # Draw joint nodes
        for idx, x, y in self.lm_list:
            radius = 5 if idx not in self.tip_ids else 7
            color = (255, 0, 128) if idx in self.tip_ids else (0, 255, 0)
            cv2.circle(frame, (x, y), radius, color, cv2.FILLED)

    def find_positions(
        self, frame: cv2.Mat, hand_index: int = 0
    ) -> List[Tuple[int, int, int]]:
        """Returns the list of 21 landmark positions (id, x, y)."""
        return self.lm_list

    def fingers_up(self) -> List[int]:
        """
        Determines which fingers are raised.
        Returns [thumb, index, middle, ring, pinky] (1 = extended, 0 = folded).
        """
        fingers: List[int] = []
        if len(self.lm_list) < 21:
            return [0, 0, 0, 0, 0]

        # Thumb: In mirrored view, compare x-coordinate of tip (4) with joint (3)
        if self.is_right_hand:
            fingers.append(1 if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1] else 0)
        else:
            fingers.append(1 if self.lm_list[self.tip_ids[0]][1] < self.lm_list[self.tip_ids[0] - 1][1] else 0)

        # 4 fingers: tip (id) y-coordinate must be above PIP joint (id - 2)
        for i in range(1, 5):
            tip_id = self.tip_ids[i]
            pip_id = tip_id - 2
            if self.lm_list[tip_id][2] < self.lm_list[pip_id][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def find_distance(
        self, p1_id: int, p2_id: int
    ) -> Tuple[float, Tuple[int, int], Tuple[int, int]]:
        """Calculates euclidean distance between two landmarks."""
        if len(self.lm_list) < 21:
            return 0.0, (0, 0), (0, 0)

        x1, y1 = self.lm_list[p1_id][1], self.lm_list[p1_id][2]
        x2, y2 = self.lm_list[p2_id][1], self.lm_list[p2_id][2]
        length = math.hypot(x2 - x1, y2 - y1)
        return length, (x1, y1), (x2, y2)
