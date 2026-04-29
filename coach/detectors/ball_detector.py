import cv2
import numpy as np


class ColorBallDetector:
    """
    Tennis ball detector: HSV color filter + MOG2 background subtraction,
    with a Kalman filter that smooths detected positions and predicts the
    ball's location for up to _MAX_MISS consecutive frames when detection fails.
    """

    _LOWER = np.array([25,  80,  80], dtype=np.uint8)
    _UPPER = np.array([65, 255, 255], dtype=np.uint8)

    _BG_WARMUP = 30   # frames before motion gate is active
    _MAX_MISS  = 8    # frames to keep predicting when raw detection is lost

    def __init__(self, max_dist: int = 120):
        self.max_dist     = max_dist
        self._kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._bg_sub      = cv2.createBackgroundSubtractorMOG2(
            history=60, varThreshold=20, detectShadows=False
        )
        self._frame_count = 0
        self._kf          = self._init_kalman()
        self._kf_active   = False
        self._miss_count  = 0

    # ----------------------------------------------------------------- kalman

    def _init_kalman(self) -> cv2.KalmanFilter:
        """Constant-velocity Kalman filter: state [x, y, vx, vy]."""
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 3.0
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0
        kf.errorCovPost        = np.eye(4, dtype=np.float32) * 10.0
        return kf

    # ----------------------------------------------------------------- public

    def detect(self, frame: np.ndarray,
               prev_pos: tuple = (None, None)) -> tuple:
        """
        Returns (x_px, y_px) of the ball centre, or (None, None) if not found.

        When the color/motion detector loses the ball briefly, the Kalman filter
        predicts its position for up to _MAX_MISS consecutive frames.
        prev_pos is still accepted for API compatibility but the Kalman state
        is now the primary continuity mechanism.
        """
        h, w = frame.shape[:2]
        self._frame_count += 1

        raw = self._raw_detect(frame, w, prev_pos)

        # First-ever detection — initialise Kalman state
        if not self._kf_active:
            if raw is None:
                return (None, None)
            self._kf.statePost = np.array(
                [[raw[0]], [raw[1]], [0.0], [0.0]], dtype=np.float32
            )
            self._kf_active = True

        # Predict (always advances the internal state forward one step)
        predicted = self._kf.predict()

        if raw is not None:
            # Good detection — correct the filter and return smoothed position
            self._miss_count = 0
            meas = np.array([[np.float32(raw[0])], [np.float32(raw[1])]])
            corrected = self._kf.correct(meas)
            return (float(corrected[0]), float(corrected[1]))

        # No raw detection — use prediction for a limited number of frames
        self._miss_count += 1
        if self._miss_count <= self._MAX_MISS:
            px, py = float(predicted[0]), float(predicted[1])
            if 0 <= px < w and 0 <= py < h:
                return (px, py)

        # Too many misses — reset so we don't hallucinate a ball that's gone
        self._kf_active  = False
        self._miss_count = 0
        return (None, None)

    # ----------------------------------------------------------------- internal

    def _raw_detect(self, frame: np.ndarray, w: int,
                    prev_pos: tuple) -> tuple | None:
        """Color + motion detection. Returns (x, y) pixels or None."""
        fg = self._bg_sub.apply(frame)
        fg = cv2.dilate(fg, self._kernel, iterations=2)

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._LOWER, self._UPPER)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel)

        if self._frame_count > self._BG_WARMUP:
            mask = cv2.bitwise_and(mask, fg)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

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
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.35:
                continue
            M = cv2.moments(c)
            if M['m00'] == 0:
                continue
            candidates.append((M['m10'] / M['m00'], M['m01'] / M['m00'], circularity))

        if not candidates:
            return None

        if prev_pos[0] is not None:
            best = min(candidates,
                       key=lambda c: (c[0] - prev_pos[0])**2 + (c[1] - prev_pos[1])**2)
            if np.sqrt((best[0] - prev_pos[0])**2 + (best[1] - prev_pos[1])**2) > self.max_dist:
                return None
        else:
            best = max(candidates, key=lambda c: c[2])

        return (float(best[0]), float(best[1]))
