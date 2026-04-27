import cv2
import numpy as np
from collections import deque
from coach.coaching.coaching_engine import Tip


# MediaPipe pose connections (pairs of landmark indices)
_POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    (11, 23), (12, 24), (23, 24),                        # torso
    (23, 25), (25, 27), (24, 26), (26, 28),              # legs
    (27, 29), (28, 30), (29, 31), (30, 32),              # feet
]

# Tip priority → BGR color
_TIP_COLOR = {
    'high':   (0,   60,  220),   # red
    'ai':     (200, 130,  0),    # blue-ish teal
    'medium': (0,  165,  255),   # orange
    'low':    (0,  200,  100),   # green
}

# Ball trail: most-recent dot is brightest yellow, oldest is dark
_TRAIL_LEN = 12


class Overlay:
    """
    Renders coaching visuals onto a video frame (in-place).

    Usage:
        ov = Overlay(frame_w=1280, frame_h=720)
        annotated = ov.draw(frame, active_tip, metrics, phase, shot_count)
        # separately, before calling draw():
        ov.update_ball(ball_pos)   # call every frame with the latest ball position
    """

    def __init__(self, frame_w: int = 1280, frame_h: int = 720):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self._ball_trail: deque = deque(maxlen=_TRAIL_LEN)

    # ------------------------------------------------------------------ public

    def update_ball(self, ball_pos: tuple):
        """Call every frame with the latest (x_px, y_px) or (None, None)."""
        if ball_pos[0] is not None:
            self._ball_trail.append(ball_pos)

    def draw(
        self,
        frame: np.ndarray,
        active_tip: Tip | None,
        metrics: dict,
        phase: str,
        shot_count: int,
        landmarks: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Annotate frame with skeleton, ball trail, tip panel, and stats HUD.
        Returns the annotated frame (same array, modified in-place).
        """
        if landmarks is not None:
            self._draw_skeleton(frame, landmarks)

        self._draw_ball_trail(frame)

        self._draw_stats_hud(frame, metrics, phase, shot_count)

        if active_tip is not None:
            self._draw_tip_panel(frame, active_tip)

        return frame

    # --------------------------------------------------------------- skeleton

    def _draw_skeleton(self, frame: np.ndarray, landmarks: np.ndarray):
        h, w = frame.shape[:2]

        # Convert normalized → pixel coords once
        pts: dict[int, tuple[int, int]] = {}
        for idx in range(min(len(landmarks), 33)):
            x = int(landmarks[idx][0] * w)
            y = int(landmarks[idx][1] * h)
            pts[idx] = (x, y)

        # Connections
        for a, b in _POSE_CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (180, 180, 180), 2, cv2.LINE_AA)

        # Joint dots
        for idx, (x, y) in pts.items():
            cv2.circle(frame, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 4, (80, 80, 80), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ trail

    def _draw_ball_trail(self, frame: np.ndarray):
        trail = list(self._ball_trail)
        n = len(trail)
        for i, (x, y) in enumerate(trail):
            # Fade from dim to bright yellow as i approaches n-1
            alpha = (i + 1) / n
            radius = max(2, int(6 * alpha))
            brightness = int(255 * alpha)
            color = (0, brightness, brightness)   # yellow fades in
            cv2.circle(frame, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)

    # ----------------------------------------------------------------- tip panel

    def _draw_tip_panel(self, frame: np.ndarray, tip: Tip):
        h, w = frame.shape[:2]
        color = _TIP_COLOR.get(tip.priority, (200, 200, 200))

        # Semi-transparent background bar at bottom
        bar_h = 56
        bar_y = h - bar_h
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bar_y), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Priority badge
        label = tip.priority.upper()
        badge_w = 80
        cv2.rectangle(frame, (0, bar_y), (badge_w, h), color, -1)
        cv2.putText(frame, label, (8, bar_y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # Tip text — wrap long lines
        words = tip.message.split()
        lines: list[str] = []
        current = ""
        for word in words:
            test = (current + " " + word).strip()
            if cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0] < w - badge_w - 20:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)

        # Draw up to 2 lines
        for i, line in enumerate(lines[:2]):
            y = bar_y + 24 + i * 24
            cv2.putText(frame, line, (badge_w + 12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA)

    # ----------------------------------------------------------------- stats HUD

    def _draw_stats_hud(self, frame: np.ndarray, metrics: dict, phase: str, shot_count: int):
        """Top-right corner: swing phase, shot counter, and key metric values."""
        fps = metrics.get('fps')
        lines = [
            f"FPS   : {fps:.0f}" if fps is not None else "FPS   : --",
            f"Phase : {phase}",
            f"Shots : {shot_count}",
        ]

        elbow = metrics.get('elbow_angle')
        if elbow is not None:
            lines.append(f"Elbow : {elbow:.0f}")

        knee = metrics.get('knee_angle')
        if knee is not None:
            lines.append(f"Knee  : {knee:.0f}")

        rot = metrics.get('shoulder_rotation')
        if rot is not None:
            lines.append(f"Rot   : {rot:.0f}")

        speed = metrics.get('speed')
        if speed is not None:
            lines.append(f"Speed : {speed:.4f}")

        # Compute panel width from longest line
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.52
        thickness = 1
        padding = 8
        line_h = 20

        max_w = max(cv2.getTextSize(l, font, font_scale, thickness)[0][0] for l in lines)
        panel_w = max_w + padding * 2
        panel_h = len(lines) * line_h + padding * 2

        x0 = self.frame_w - panel_w - 8
        y0 = 8

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        for i, line in enumerate(lines):
            y = y0 + padding + (i + 1) * line_h - 4
            cv2.putText(frame, line, (x0 + padding, y),
                        font, font_scale, (220, 220, 220), thickness, cv2.LINE_AA)
