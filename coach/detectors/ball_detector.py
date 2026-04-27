import cv2
import numpy as np


class ColorBallDetector:
    """
    Fast tennis ball detector using HSV color filtering + contour analysis.

    Replaces TrackNet for webcam use: ~1ms per frame vs 100-500ms,
    no model file needed, and works better when the ball is large and
    the lighting is reasonably controlled.

    Tune HSV bounds if detection is poor in your lighting:
        lower = [25, 80, 80]  →  hue 25-65 covers yellow-green
        upper = [65, 255, 255]
    """

    _LOWER = np.array([25,  80,  80], dtype=np.uint8)
    _UPPER = np.array([65, 255, 255], dtype=np.uint8)

    def __init__(self, max_dist: int = 120):
        """
        max_dist: maximum pixel jump allowed between frames.
                  Detections farther than this from prev_pos are discarded as outliers.
        """
        self.max_dist = max_dist
        self._kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray,
               prev_pos: tuple = (None, None)) -> tuple:
        """
        Returns (x_px, y_px) of the ball centre, or (None, None) if not found.
        prev_pos: last known (x, y) in pixels — used to reject far-away blobs.
        """
        h, w = frame.shape[:2]

        # --- colour mask ---
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._LOWER, self._UPPER)
        # Close small gaps, remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (None, None)

        # --- size filter: ball radius should be 0.3%–6% of frame width ---
        min_area = np.pi * (w * 0.003) ** 2
        max_area = np.pi * (w * 0.06)  ** 2

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (min_area <= area <= max_area):
                continue
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            # Circularity: 1.0 = perfect circle; filter out court lines / streaks
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.35:
                continue
            M = cv2.moments(c)
            if M['m00'] == 0:
                continue
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            candidates.append((float(cx), float(cy), circularity))

        if not candidates:
            return (None, None)

        # --- pick best candidate ---
        if prev_pos[0] is not None:
            # Prefer closest to last known position
            best = min(candidates,
                       key=lambda c: (c[0] - prev_pos[0])**2 + (c[1] - prev_pos[1])**2)
            dist = np.sqrt((best[0] - prev_pos[0])**2 + (best[1] - prev_pos[1])**2)
            if dist > self.max_dist:
                return (None, None)   # too far — likely a false positive
        else:
            # No history: pick most circular blob
            best = max(candidates, key=lambda c: c[2])

        return (best[0], best[1])
