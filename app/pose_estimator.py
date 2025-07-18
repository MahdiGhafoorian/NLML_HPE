# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 17:09:02 2025

@author: Mahdi Ghafourian
"""

import math
import cv2
import numpy as np
import torch
import mediapipe as mp
from math import cos, sin, radians
from helpers import FeatureExtractor as FE

class HeadPoseEstimator:
    def __init__(self, model_path="models/combined_model_scripted.pth", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.alpha = 0.4
        self.prev_tdx, self.prev_tdy = None, None
        self.MAX_CENTER_JUMP = 100
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.yaw_smoothed = None
        self.pitch_smoothed = None
        self.roll_smoothed = None

    def draw_pose(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return frame  # No face found

        for landmarks in results.multi_face_landmarks:
            input_landmarks = FE.get_feature_vector_from_image(
                self.face_mesh, frame, normalize=True, isPil=False)

            if (input_landmarks == 0).all():
                return frame  # Invalid

            input_landmarks = input_landmarks.unsqueeze(dim=0).to(self.device)

            with torch.no_grad():
                predictions = self.model(input_landmarks)
                yaw, pitch, roll = map(lambda x: round(np.degrees(x.item()), 2), predictions)

            # Exponential smoothing
            if self.yaw_smoothed is None:
                self.yaw_smoothed, self.pitch_smoothed, self.roll_smoothed = yaw, pitch, roll
            else:
                self.yaw_smoothed = self.alpha * yaw + (1 - self.alpha) * self.yaw_smoothed
                self.pitch_smoothed = self.alpha * pitch + (1 - self.alpha) * self.pitch_smoothed
                self.roll_smoothed = self.alpha * roll + (1 - self.alpha) * self.roll_smoothed

            yaw, pitch, roll = self.yaw_smoothed, self.pitch_smoothed, self.roll_smoothed

            frame, self.prev_tdx, self.prev_tdy = visualize_axes_on_face(
                self.prev_tdx, self.prev_tdy, self.MAX_CENTER_JUMP, frame, landmarks.landmark, yaw, pitch, roll)

            # Add text
            cv2.putText(frame, f"Yaw: {yaw:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Roll: {roll:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame
    
def visualize_axes_on_face(prev_tdx, prev_tdy, MAX_CENTER_JUMP, frame, landmarks, yaw, pitch, roll, size=80):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    # Calculate the center of the face by averaging the coordinates of key landmarks
    # We can take landmarks around the eyes and nose as they roughly form the center
    nose_idx = 1  # Typically, the nose is at index 1 in MediaPipe
    left_eye_idx = 33  # Left eye can be a good reference
    right_eye_idx = 263  # Right eye can be a good reference

    # Get the 2D coordinates of the landmarks
    nose = landmarks[nose_idx]
    left_eye = landmarks[left_eye_idx]
    right_eye = landmarks[right_eye_idx]

    # Calculate the center of the face
    new_tdx = (nose.x + left_eye.x + right_eye.x) * frame.shape[1] / 3
    new_tdy = (nose.y + left_eye.y + right_eye.y) * frame.shape[0] / 3
    
    # If first frame, just use the new values
    if prev_tdx is None or prev_tdy is None:
        tdx, tdy = new_tdx, new_tdy
    else:
        # Compute Euclidean distance between previous and new center
        dist = math.sqrt((new_tdx - prev_tdx)**2 + (new_tdy - prev_tdy)**2)        
    
        if dist > MAX_CENTER_JUMP:
            # Use previous stable values
            tdx, tdy = prev_tdx, prev_tdy
        else:
            # Accept new values
            tdx, tdy = new_tdx, new_tdy 

    # X-Axis pointing to right, drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis, drawn in green
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen), drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy
    
    # Draw the axes
    cv2.line(frame, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)  # X-Axis (Red)
    cv2.line(frame, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)  # Y-Axis (Green)
    cv2.line(frame, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)  # Z-Axis (Blue)    

    return frame, tdx, tdy