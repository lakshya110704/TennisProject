import time
from dataclasses import dataclass
from coach.analysis.swing_analyzer import SwingAnalyzer
from coach.analysis.footwork_analyzer import FootworkAnalyzer
from coach.analysis.shot_detector import ShotDetector, ShotEvent


@dataclass
class Tip:
    message:  str
    priority: str   # 'high' | 'medium' | 'low' | 'ai'
    category: str   # 'swing' | 'footwork' | 'ai'


# Lower number = shown first
_PRIORITY_ORDER = {'high': 0, 'ai': 1, 'medium': 2, 'low': 3}

# How long each priority level stays on screen (seconds)
_DISPLAY_DURATION = {'high': 5.0, 'ai': 7.0, 'medium': 4.0, 'low': 3.0}

# How long before the same tip can appear again (seconds)
_COOLDOWN = {'high': 30.0, 'ai': 90.0, 'medium': 25.0, 'low': 20.0}


class CoachingEngine:
    """
    Central coordinator that:
      - runs all three analyzers every frame
      - maintains a priority-sorted tip queue
      - surfaces one tip at a time with per-tip cooldowns
      - accepts AI tips injected asynchronously from ai_coach.py
      - exposes session summary data for the post-session report
    """

    def __init__(self, dominant_hand: str = 'right',
                 frame_w: int = 1280, frame_h: int = 720):
        self.swing    = SwingAnalyzer(dominant_hand)
        self.footwork = FootworkAnalyzer()
        self.shots    = ShotDetector(dominant_hand, frame_w, frame_h)

        self._active_tip: Tip | None = None
        self._tip_expiry: float = 0.0

        # message → time.time() after which it can appear again
        self._tip_cooldowns: dict[str, float] = {}

        # sorted list of pending tips (sorted by priority number)
        self._pending: list[Tip] = []

        # AI tip set by inject_ai_tip(), consumed next frame
        self._ai_tip_queued: str | None = None

        # session-level tracking
        self.bounce_count: int = 0
        self._tip_counts: dict[str, int] = {}
        self._swing_metrics_log: list[dict] = []   # one entry per shot, for report

    # ------------------------------------------------------------------ main

    def process_frame(
        self,
        frame_idx: int,
        landmarks,          # np.ndarray | None  (33 × 3, MediaPipe)
        ball_pos: tuple,    # (x_px, y_px) or (None, None)
        is_bounce: bool,
    ) -> tuple[Tip | None, dict, ShotEvent | None]:
        """
        Call once per frame.

        Returns:
            active_tip  – the Tip currently displayed (or None)
            metrics     – combined dict from all analyzers
            shot_event  – ShotEvent if a shot was just detected, else None
        """
        now = time.time()

        # ---------- analyzers ----------
        swing_metrics = self.swing.analyze(landmarks)
        for priority, msg in self.swing.get_feedback(swing_metrics):
            self._enqueue(Tip(msg, priority, 'swing'), now)

        self.footwork.update(landmarks)
        foot_metrics = self.footwork.get_metrics(landmarks)
        for priority, msg in self.footwork.get_feedback(foot_metrics):
            self._enqueue(Tip(msg, priority, 'footwork'), now)

        shot_event = self.shots.update(frame_idx, ball_pos, landmarks)

        if is_bounce:
            self.bounce_count += 1

        # log swing metrics + inject stroke-specific tips at moment of contact
        if shot_event and swing_metrics:
            self._swing_metrics_log.append({
                'frame':      shot_event.frame_idx,
                'shot_type':  shot_event.shot_type,
                **swing_metrics,
            })
            for priority, msg in self.swing.get_shot_feedback(swing_metrics, shot_event.shot_type):
                self._enqueue(Tip(msg, priority, 'swing'), now)

        # ---------- inject AI tip if one arrived ----------
        if self._ai_tip_queued:
            self._enqueue(Tip(self._ai_tip_queued, 'ai', 'ai'), now)
            self._ai_tip_queued = None

        # ---------- advance active tip if expired ----------
        if self._active_tip is None or now >= self._tip_expiry:
            self._advance(now)

        metrics = {**swing_metrics, **foot_metrics, 'bounces': self.bounce_count}
        return self._active_tip, metrics, shot_event

    # ----------------------------------------------------------------- AI hook

    def inject_ai_tip(self, message: str):
        """
        Thread-safe enough for our use case.
        Called by ai_coach.py when Claude's response is ready.
        """
        self._ai_tip_queued = message

    # --------------------------------------------------------------- internals

    def _enqueue(self, tip: Tip, now: float):
        if now < self._tip_cooldowns.get(tip.message, 0):
            return
        if any(t.message == tip.message for t in self._pending):
            return
        self._pending.append(tip)
        self._pending.sort(key=lambda t: _PRIORITY_ORDER.get(t.priority, 99))

    def _advance(self, now: float):
        if not self._pending:
            self._active_tip = None
            return
        tip = self._pending.pop(0)
        self._active_tip = tip
        self._tip_expiry = now + _DISPLAY_DURATION.get(tip.priority, 4.0)
        self._tip_cooldowns[tip.message] = now + _COOLDOWN.get(tip.priority, 25.0)
        self._tip_counts[tip.message] = self._tip_counts.get(tip.message, 0) + 1

    # ----------------------------------------------------------- session report

    def get_session_summary(self) -> dict:
        """Aggregate data for session_report.py."""
        avg_metrics: dict = {}
        if self._swing_metrics_log:
            keys = [k for k in self._swing_metrics_log[0]
                    if isinstance(self._swing_metrics_log[0][k], float)]
            for k in keys:
                vals = [e[k] for e in self._swing_metrics_log if k in e]
                avg_metrics[f'avg_{k}'] = round(sum(vals) / len(vals), 2)

        top_tips = sorted(self._tip_counts.items(), key=lambda x: -x[1])[:5]

        return {
            'shots':        self.shots.session_summary(),
            'bounces':      self.bounce_count,
            'footwork':     self.footwork.get_session_stats(),
            'avg_swing':    avg_metrics,
            'top_tips':     top_tips,
        }
