import numpy as np
from collections import deque


_LM = {
    'l_hip': 23, 'r_hip': 24,
    'l_ankle': 27, 'r_ankle': 28,
}

# Thresholds in normalized coords (MediaPipe returns 0-1)
_IDLE_SPEED      = 0.003   # below this per-frame speed = not moving
_STANCE_MIN      = 0.12    # ankle separation below this = feet too close
_STANCE_MAX      = 0.45    # above this = too wide to move quickly
_EDGE_THRESHOLD  = 0.14    # distance from home position to count as "off center"


class FootworkAnalyzer:
    """
    Tracks player movement over time from MediaPipe hip + ankle landmarks.
    All positions are in normalized (0-1) frame coords.
    """

    def __init__(self, history_len: int = 90):
        self._history: deque = deque(maxlen=history_len)   # (cx, cy, frame_idx) or None
        self._home: tuple | None = None   # baseline center position
        self._frame_idx: int = 0
        self._home_sample_frames = 20     # frames before we lock in home position

    # ------------------------------------------------------------------ update

    def update(self, landmarks: np.ndarray | None):
        """Call once per frame. Landmarks may be None if pose not detected."""
        self._frame_idx += 1

        if landmarks is None or len(landmarks) < 29:
            self._history.append(None)
            return

        l_hip = landmarks[_LM['l_hip']][:2]
        r_hip = landmarks[_LM['r_hip']][:2]
        cx, cy = float((l_hip[0] + r_hip[0]) / 2), float((l_hip[1] + r_hip[1]) / 2)
        self._history.append((cx, cy, self._frame_idx))

        # Lock in home position once we have enough stable frames
        if self._home is None:
            valid = [h for h in self._history if h is not None]
            if len(valid) >= self._home_sample_frames:
                self._home = (
                    float(np.mean([h[0] for h in valid])),
                    float(np.mean([h[1] for h in valid])),
                )

    # ----------------------------------------------------------------- metrics

    def _recent_speeds(self, n: int = 15) -> list[float]:
        valid = [h for h in self._history if h is not None]
        if len(valid) < 2:
            return []
        recent = valid[-min(n, len(valid)):]
        speeds = []
        for i in range(1, len(recent)):
            dx = recent[i][0] - recent[i - 1][0]
            dy = recent[i][1] - recent[i - 1][1]
            dt = max(recent[i][2] - recent[i - 1][2], 1)
            speeds.append(np.sqrt(dx ** 2 + dy ** 2) / dt)
        return speeds

    def get_metrics(self, landmarks: np.ndarray | None) -> dict:
        """
        Compute footwork metrics for the current frame.

        Returns a dict with:
            stance_width   – ankle separation (normalized)
            speed          – avg movement speed over last ~15 frames
            is_idle        – True if barely moving
            idle_ratio     – fraction of history that was idle
            off_center     – True if player is far from their home position
            dist_from_home – float distance from home position
        """
        if landmarks is None or len(landmarks) < 29:
            return {}

        l_ankle = landmarks[_LM['l_ankle']][:2]
        r_ankle = landmarks[_LM['r_ankle']][:2]
        l_hip   = landmarks[_LM['l_hip']][:2]
        r_hip   = landmarks[_LM['r_hip']][:2]
        hip_c   = np.array([(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2])

        speeds = self._recent_speeds()
        avg_speed = float(np.mean(speeds)) if speeds else 0.0

        all_valid = [h for h in self._history if h is not None]
        all_speeds: list[float] = []
        for i in range(1, len(all_valid)):
            dx = all_valid[i][0] - all_valid[i - 1][0]
            dy = all_valid[i][1] - all_valid[i - 1][1]
            dt = max(all_valid[i][2] - all_valid[i - 1][2], 1)
            all_speeds.append(np.sqrt(dx ** 2 + dy ** 2) / dt)
        idle_ratio = (
            sum(1 for s in all_speeds if s < _IDLE_SPEED) / len(all_speeds)
            if all_speeds else 0.0
        )

        dist_from_home = 0.0
        off_center = False
        if self._home is not None:
            dist_from_home = float(np.linalg.norm(hip_c - np.array(self._home)))
            off_center = dist_from_home > _EDGE_THRESHOLD

        return {
            'stance_width':   float(np.linalg.norm(l_ankle - r_ankle)),
            'speed':          avg_speed,
            'is_idle':        avg_speed < _IDLE_SPEED,
            'idle_ratio':     float(idle_ratio),
            'off_center':     off_center,
            'dist_from_home': dist_from_home,
        }

    # ---------------------------------------------------------------- feedback

    def get_feedback(self, metrics: dict) -> list[tuple[str, str]]:
        """Return list of (priority, message). Priority: 'high' | 'medium' | 'low'."""
        if not metrics:
            return []

        tips = []

        stance = metrics.get('stance_width')
        if stance is not None:
            if stance < _STANCE_MIN:
                tips.append(('high',   "Widen your stance — feet should be shoulder-width apart"))
            elif stance > _STANCE_MAX:
                tips.append(('medium', "Narrow your stance — you're too wide to push off quickly"))

        if metrics.get('idle_ratio', 0.0) > 0.65:
            tips.append(('medium', "Stay light on your feet — keep bouncing between shots"))

        if metrics.get('off_center'):
            tips.append(('high', "Return to center — you're too far out of position"))

        return tips

    # ------------------------------------------------------------- session stats

    def get_session_stats(self) -> dict:
        """Aggregate stats for the post-session report."""
        all_valid = [h for h in self._history if h is not None]
        if len(all_valid) < 2:
            return {}

        speeds: list[float] = []
        for i in range(1, len(all_valid)):
            dx = all_valid[i][0] - all_valid[i - 1][0]
            dy = all_valid[i][1] - all_valid[i - 1][1]
            dt = max(all_valid[i][2] - all_valid[i - 1][2], 1)
            speeds.append(np.sqrt(dx ** 2 + dy ** 2) / dt)

        idle_pct = round(100 * sum(1 for s in speeds if s < _IDLE_SPEED) / len(speeds), 1)

        return {
            'avg_speed_normalized': round(float(np.mean(speeds)), 5),
            'max_speed_normalized': round(float(np.max(speeds)), 5),
            'idle_pct':             idle_pct,
        }
