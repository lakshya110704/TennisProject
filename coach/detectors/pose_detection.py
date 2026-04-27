import cv2
import mediapipe as mp
import numpy as np

class PoseDetector:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(static_image_mode=False,
                                           model_complexity=1,
                                           enable_segmentation=False,
                                           min_detection_confidence=0.5)
    
    def detect(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks)
        else:
            return None