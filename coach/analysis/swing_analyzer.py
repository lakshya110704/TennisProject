import numpy as np


# MediaPipe pose landmark indices
_LM = {
    'nose': 0,
    'l_shoulder': 11, 'r_shoulder': 12,
    'l_elbow': 13,    'r_elbow': 14,
    'l_wrist': 15,    'r_wrist': 16,
    'l_hip': 23,      'r_hip': 24,
    'l_knee': 25,     'r_knee': 26,
    'l_ankle': 27,    'r_ankle': 28,
}


def _angle(a, b, c):
    """Angle at point b formed by a-b-c, in degrees."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


class SwingAnalyzer:
    """
    Analyzes player swing mechanics from MediaPipe pose landmarks.
    All landmark coordinates are normalized (0-1) as returned by MediaPipe.
    """

    # Thresholds (degrees) based on tennis biomechanics
    ELBOW_MIN = 95      # below this = arm too cramped
    ELBOW_MAX = 168     # above this = arm locked straight
    KNEE_STRAIGHT = 165 # above this = not bending knees
    KNEE_OVERLOW = 105  # below this = crouching too much
    SHOULDER_TURN_MIN = 18  # below this = poor shoulder rotation

    def __init__(self, dominant_hand: str = 'right'):
        if dominant_hand not in ('right', 'left'):
            raise ValueError("dominant_hand must be 'right' or 'left'")
        self.dominant_hand = dominant_hand
        self._prev_wrist_y = None   # for follow-through detection
        self._wrist_peak_y = None   # highest (lowest y-value) wrist reached

    def _arm_landmarks(self, lm: np.ndarray):
        """Return (shoulder, elbow, wrist, hip, knee, ankle) for dominant arm."""
        if self.dominant_hand == 'right':
            return (
                lm[_LM['r_shoulder']][:2],
                lm[_LM['r_elbow']][:2],
                lm[_LM['r_wrist']][:2],
                lm[_LM['r_hip']][:2],
                lm[_LM['r_knee']][:2],
                lm[_LM['r_ankle']][:2],
            )
        return (
            lm[_LM['l_shoulder']][:2],
            lm[_LM['l_elbow']][:2],
            lm[_LM['l_wrist']][:2],
            lm[_LM['l_hip']][:2],
            lm[_LM['l_knee']][:2],
            lm[_LM['l_ankle']][:2],
        )

    def analyze(self, landmarks: np.ndarray) -> dict:
        """
        Compute swing metrics from a single frame's landmarks.

        Returns a dict with keys:
            elbow_angle       – shoulder-elbow-wrist angle (degrees)
            knee_angle        – hip-knee-ankle angle (degrees)
            shoulder_rotation – how much shoulders are rotated vs hips (degrees)
            wrist_above_shoulder – bool, True when wrist is above shoulder (follow-through)
            contact_in_front  – bool, True when wrist is in front of hip line
            arm_extension     – 0-1 ratio of how extended the arm is
        """
        if landmarks is None or len(landmarks) < 29:
            return {}

        shoulder, elbow, wrist, hip, knee, ankle = self._arm_landmarks(landmarks)

        l_shoulder = landmarks[_LM['l_shoulder']][:2]
        r_shoulder = landmarks[_LM['r_shoulder']][:2]
        l_hip = landmarks[_LM['l_hip']][:2]
        r_hip = landmarks[_LM['r_hip']][:2]

        # Elbow angle
        elbow_angle = _angle(shoulder, elbow, wrist)

        # Knee bend
        knee_angle = _angle(hip, knee, ankle)

        # Shoulder rotation relative to hip line
        # Both are horizontal vectors; we compare their angles
        sh_vec = r_shoulder - l_shoulder
        hi_vec = r_hip - l_hip
        sh_deg = float(np.degrees(np.arctan2(sh_vec[1], sh_vec[0])))
        hi_deg = float(np.degrees(np.arctan2(hi_vec[1], hi_vec[0])))
        shoulder_rotation = abs(sh_deg - hi_deg)
        # Clamp wrap-around artifacts
        if shoulder_rotation > 90:
            shoulder_rotation = 180 - shoulder_rotation

        # Follow-through: wrist y < shoulder y (image coords — y=0 is top)
        wrist_above_shoulder = bool(wrist[1] < shoulder[1])

        # Contact point: wrist x should be in front of hip
        # For right-handed players hitting forehand, wrist x > hip x (further right)
        # We just check wrist is not behind the opposite hip
        opp_hip = l_hip if self.dominant_hand == 'right' else r_hip
        contact_in_front = bool(wrist[0] > opp_hip[0]) if self.dominant_hand == 'right' \
            else bool(wrist[0] < opp_hip[0])

        # Arm extension ratio (0 = fully bent, 1 = fully extended)
        upper_arm = np.linalg.norm(elbow - shoulder) + 1e-9
        forearm = np.linalg.norm(wrist - elbow)
        full_arm = np.linalg.norm(wrist - shoulder)
        arm_extension = float(full_arm / (upper_arm + forearm))

        return {
            'elbow_angle': elbow_angle,
            'knee_angle': knee_angle,
            'shoulder_rotation': shoulder_rotation,
            'wrist_above_shoulder': wrist_above_shoulder,
            'contact_in_front': contact_in_front,
            'arm_extension': arm_extension,
        }

    def get_feedback(self, metrics: dict) -> list[tuple[str, str]]:
        """
        Return a list of (priority, message) tuples.
        Priority is 'high', 'medium', or 'low'.
        """
        if not metrics:
            return []

        tips = []

        elbow = metrics.get('elbow_angle')
        if elbow is not None:
            if elbow < self.ELBOW_MIN:
                tips.append(('high', "Extend your arm — elbow is too cramped at contact"))
            elif elbow > self.ELBOW_MAX:
                tips.append(('medium', "Soften your elbow — avoid locking your arm straight"))

        knee = metrics.get('knee_angle')
        if knee is not None:
            if knee > self.KNEE_STRAIGHT:
                tips.append(('high', "Bend your knees — stay low for balance and power"))
            elif knee < self.KNEE_OVERLOW:
                tips.append(('low', "You can rise a little — you're crouching too much"))

        rot = metrics.get('shoulder_rotation')
        if rot is not None and rot < self.SHOULDER_TURN_MIN:
            tips.append(('medium', "Rotate your shoulders on the backswing for more power"))

        if metrics.get('contact_in_front') is False:
            tips.append(('high', "Hit the ball in front of your body — you're hitting late"))

        if metrics.get('wrist_above_shoulder') is False and elbow is not None and elbow < 150:
            # Only flag missing follow-through when arm was actually swinging (not idle)
            tips.append(('medium', "Follow through high — finish with your wrist above your shoulder"))

        return tips
