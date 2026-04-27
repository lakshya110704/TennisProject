import numpy as np
from collections import deque
from dataclasses import dataclass, field


@dataclass
class ShotEvent:
    frame_idx:  int
    shot_type:  str          # 'Forehand' | 'Backhand' | 'Serve / Overhead' | 'Unknown'
    ball_pos:   tuple        # (x, y) in pixels at moment of contact
    wrist_speed: float = 0.0 # normalized wrist speed at contact


# MediaPipe indices we care about
_R_SHOULDER, _L_SHOULDER = 12, 11
_R_WRIST,    _L_WRIST    = 16, 15
_R_HIP,      _L_HIP      = 24, 23


class ShotDetector:
    """
    Detects shot events by watching wrist speed + ball proximity.

    Logic:
      A shot is triggered when:
        1. Dominant wrist is moving fast  (swing in progress)
        2. Ball is within proximity of the wrist
        3. Minimum cooldown since last shot has passed

    Shot type is classified from ball position relative to body center.
    """

    WRIST_SPEED_THRESHOLD = 0.018   # normalized coords / frame
    BALL_PROXIMITY_RATIO  = 0.18    # fraction of frame width
    SHOT_COOLDOWN_FRAMES  = 20      # min frames between consecutive shots
    HISTORY_LEN           = 8       # frames kept for wrist speed calculation

    def __init__(self, dominant_hand: str = 'right',
                 frame_w: int = 1280, frame_h: int = 720):
        self.dominant_hand = dominant_hand
        self.frame_w = frame_w
        self.frame_h = frame_h

        self._wrist_history: deque = deque(maxlen=self.HISTORY_LEN)  # (x, y) normalized
        self._last_shot_frame = -self.SHOT_COOLDOWN_FRAMES
        self._phase = 'idle'      # 'idle' | 'backswing' | 'swing' | 'follow_through'

        self.shot_count = 0
        self.shot_log: list[ShotEvent] = []

    # ----------------------------------------------------------------- public

    def update(self, frame_idx: int,
               ball_pos: tuple,
               landmarks: np.ndarray | None) -> ShotEvent | None:
        """
        Call once per frame. Returns a ShotEvent if a shot was just detected,
        otherwise None.
        """
        wrist_xy = self._get_wrist(landmarks)
        self._wrist_history.append(wrist_xy)

        wrist_speed = self._compute_wrist_speed()
        self._update_phase(wrist_speed)

        if not self._can_trigger(frame_idx):
            return None

        if wrist_speed < self.WRIST_SPEED_THRESHOLD:
            return None

        if not self._ball_near_wrist(ball_pos, wrist_xy):
            return None

        # --- shot confirmed ---
        shot_type = self._classify(ball_pos, landmarks)
        event = ShotEvent(
            frame_idx=frame_idx,
            shot_type=shot_type,
            ball_pos=ball_pos,
            wrist_speed=round(wrist_speed, 5),
        )
        self._last_shot_frame = frame_idx
        self.shot_count += 1
        self.shot_log.append(event)
        return event

    @property
    def phase(self) -> str:
        """Current swing phase: 'idle' | 'backswing' | 'swing' | 'follow_through'."""
        return self._phase

    def session_summary(self) -> dict:
        """Shot type breakdown for the session report."""
        if not self.shot_log:
            return {'total': 0}
        counts: dict[str, int] = {}
        for ev in self.shot_log:
            counts[ev.shot_type] = counts.get(ev.shot_type, 0) + 1
        return {'total': self.shot_count, 'breakdown': counts}

    # ----------------------------------------------------------------- helpers

    def _get_wrist(self, landmarks: np.ndarray | None) -> tuple | None:
        if landmarks is None or len(landmarks) < 17:
            return None
        idx = _R_WRIST if self.dominant_hand == 'right' else _L_WRIST
        return (float(landmarks[idx][0]), float(landmarks[idx][1]))

    def _compute_wrist_speed(self) -> float:
        """Average normalized wrist speed over the last few valid frames."""
        valid = [p for p in self._wrist_history if p is not None]
        if len(valid) < 2:
            return 0.0
        speeds = []
        for i in range(1, len(valid)):
            dx = valid[i][0] - valid[i - 1][0]
            dy = valid[i][1] - valid[i - 1][1]
            speeds.append(np.sqrt(dx ** 2 + dy ** 2))
        return float(np.mean(speeds))

    def _update_phase(self, wrist_speed: float):
        """Track which phase of the swing we're in."""
        if wrist_speed < 0.004:
            self._phase = 'idle'
        elif wrist_speed < self.WRIST_SPEED_THRESHOLD * 0.6:
            self._phase = 'backswing'
        elif wrist_speed >= self.WRIST_SPEED_THRESHOLD:
            self._phase = 'swing'
        else:
            if self._phase == 'swing':
                self._phase = 'follow_through'

    def _can_trigger(self, frame_idx: int) -> bool:
        return (frame_idx - self._last_shot_frame) >= self.SHOT_COOLDOWN_FRAMES

    def _ball_near_wrist(self, ball_pos: tuple, wrist_xy: tuple | None) -> bool:
        if ball_pos[0] is None or wrist_xy is None:
            return False
        # convert wrist from normalized to pixel coords
        wx = wrist_xy[0] * self.frame_w
        wy = wrist_xy[1] * self.frame_h
        dist = np.sqrt((ball_pos[0] - wx) ** 2 + (ball_pos[1] - wy) ** 2)
        return dist < self.frame_w * self.BALL_PROXIMITY_RATIO

    def _classify(self, ball_pos: tuple, landmarks: np.ndarray | None) -> str:
        """
        Classify shot type using three signals:
          1. Ball height      → Serve / Overhead
          2. Wrist x-range    → Volley (compact swing)
          3. Wrist swing dir  → Forehand vs Backhand (tiebreak: ball side)
        """
        if landmarks is None or ball_pos[0] is None:
            return 'Unknown'

        r_shoulder = landmarks[_R_SHOULDER][:2]
        l_shoulder = landmarks[_L_SHOULDER][:2]
        r_hip      = landmarks[_R_HIP][:2]
        l_hip      = landmarks[_L_HIP][:2]

        shoulder_y  = float((r_shoulder[1] + l_shoulder[1]) / 2)
        ball_norm_x = ball_pos[0] / self.frame_w
        ball_norm_y = ball_pos[1] / self.frame_h
        body_cx     = float((r_hip[0] + l_hip[0]) / 2)

        # 1. Serve / Overhead — ball well above shoulder line
        if ball_norm_y < shoulder_y - 0.10:
            return 'Serve / Overhead'

        # Wrist travel over recent history
        valid    = [p for p in self._wrist_history if p is not None]
        swing_dx = 0.0
        x_range  = 0.0
        if len(valid) >= 2:
            swing_dx = valid[-1][0] - valid[0][0]   # positive = moving right
            x_range  = max(p[0] for p in valid) - min(p[0] for p in valid)

        # 2. Volley — wrist barely moved laterally (compact punch)
        if len(valid) >= 4 and x_range < 0.07:
            return 'Volley'

        # 3. Forehand vs Backhand
        #    Primary: swing direction (where the wrist is heading at contact)
        #    Tiebreak: which side of the body the ball is on
        if self.dominant_hand == 'right':
            wrist_fh = swing_dx < -0.01   # wrist sweeping left = forehand follow-through
            ball_fh  = ball_norm_x >= body_cx
            return 'Forehand' if (wrist_fh or ball_fh) else 'Backhand'
        else:
            wrist_fh = swing_dx > 0.01
            ball_fh  = ball_norm_x <= body_cx
            return 'Forehand' if (wrist_fh or ball_fh) else 'Backhand'
