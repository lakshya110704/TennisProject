#!/usr/bin/env python3
"""
live_coach.py — entry point for the Tennis AI Coach.

Usage:
    python live_coach.py                          # webcam, right-handed
    python live_coach.py --source video.mp4       # video file
    python live_coach.py --hand left              # left-handed player
    python live_coach.py --no-audio --no-ai       # rule-based only, silent
    python live_coach.py --save out.mp4           # save annotated video
"""
import os
import warnings
import logging

# Must be set before any mediapipe / TF / absl import
os.environ["GLOG_minloglevel"]    = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
logging.getLogger("absl").setLevel(logging.ERROR)

import argparse
import sys
import queue
import threading
import time
import cv2
import numpy as np
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))

from coach.detectors.ball_detector import ColorBallDetector
from coach.detectors.bounce_detector import BounceDetector
from coach.detectors.pose_detection import PoseDetector
from coach.coaching.coaching_engine import CoachingEngine
from coach.coaching.ai_coach import AICoach
from coach.output.audio_feedback import AudioFeedback
from coach.output.overlay import Overlay
from coach.coaching import session_report


# ---------------------------------------------------------------- background threads

class _CaptureThread:
    """
    Continuously drains the camera buffer and keeps only the latest frame.
    Without this, slow processing causes cap.read() to return stale frames.
    """
    def __init__(self, cap: cv2.VideoCapture):
        self._cap    = cap
        self._frame  = None
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._ended  = False          # set when source runs out of frames
        self._t      = threading.Thread(target=self._run, daemon=True, name="capture")
        self._t.start()

    def _run(self):
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                self._ended = True    # video finished or cap released
                break
            with self._lock:
                self._frame = frame

    @property
    def ended(self) -> bool:
        return self._ended

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self):
        self._stop.set()
        self._cap.release()   # unblocks cap.read() immediately so the thread exits
        self._t.join(timeout=2.0)


class _PoseThread:
    """Runs MediaPipe pose detection in a background thread."""
    def __init__(self, detector: PoseDetector):
        self._detector  = detector
        self._q         = queue.Queue(maxsize=1)
        self._landmarks = None
        self._lock      = threading.Lock()
        self._stop      = threading.Event()
        self._t         = threading.Thread(target=self._run, daemon=True, name="pose-det")
        self._t.start()

    def push(self, frame: np.ndarray):
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            pass

    def get_landmarks(self):
        with self._lock:
            return self._landmarks

    def _run(self):
        while not self._stop.is_set():
            try:
                frame = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            lm = self._detector.detect(frame)
            with self._lock:
                self._landmarks = lm

    def stop(self):
        self._stop.set()
        self._t.join(timeout=2.0)


class _BallThread:
    """
    Runs ColorBallDetector in a background thread.
    ~1ms per frame — far cheaper than TrackNet.
    Stale frames are dropped (queue depth = 1).
    """
    def __init__(self, detector: ColorBallDetector):
        self._detector = detector
        self._q        = queue.Queue(maxsize=1)
        self._ball_pos = (None, None)
        self._lock     = threading.Lock()
        self._stop     = threading.Event()
        self._t        = threading.Thread(target=self._run, daemon=True, name="ball-det")
        self._t.start()

    def push(self, frame: np.ndarray):
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            pass

    def get_pos(self) -> tuple:
        with self._lock:
            return self._ball_pos

    def _run(self):
        while not self._stop.is_set():
            try:
                frame = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._lock:
                prev = self._ball_pos
            pos = self._detector.detect(frame, prev)
            with self._lock:
                self._ball_pos = pos

    def stop(self):
        self._stop.set()
        self._t.join(timeout=2.0)


# ---------------------------------------------------------------- argument parsing

def _parse_args():
    p = argparse.ArgumentParser(description="Tennis AI Coach — real-time coaching overlay")
    p.add_argument('--source', default='webcam',
                   help="'webcam' or path to a video file (default: webcam)")
    p.add_argument('--hand', choices=['right', 'left'], default='right',
                   help="Dominant hand of the player (default: right)")
    p.add_argument('--no-audio', action='store_true',
                   help="Disable text-to-speech output")
    p.add_argument('--no-ai', action='store_true',
                   help="Disable Claude AI coaching tips")
    p.add_argument('--model-dir', default=os.path.join(_HERE, 'models'),
                   help="Directory containing model files (default: <script-dir>/models/)")
    p.add_argument('--save', metavar='OUT.mp4',
                   help="Also write annotated video to this file")
    p.add_argument('--max-frames', type=int, default=0,
                   help="Stop after this many frames (0 = process entire video)")
    return p.parse_args()


def _open_source(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0 if source == 'webcam' else source)
    if not cap.isOpened():
        print(f"[Error] Cannot open source: {source}", file=sys.stderr)
        sys.exit(1)
    return cap


# ---------------------------------------------------------------- main loop

def main():
    args = _parse_args()

    cap     = _open_source(args.source)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"[Coach] Source : {args.source}  {frame_w}×{frame_h} @ {fps:.0f} fps")
    print(f"[Coach] Hand   : {args.hand}   Audio: {not args.no_audio}   AI: {not args.no_ai}")

    # ---- detectors ----
    print("[Coach] Initialising detectors…")
    ball_det   = ColorBallDetector()
    bounce_det = BounceDetector(f'{args.model_dir}/bounce_model.cbm')
    pose_det   = PoseDetector()
    print("[Coach] Ready.")

    # ---- coaching layer ----
    engine   = CoachingEngine(dominant_hand=args.hand, frame_w=frame_w, frame_h=frame_h)
    ai_coach = None if args.no_ai else AICoach()
    audio    = None if args.no_audio else AudioFeedback()
    ov       = Overlay(frame_w, frame_h)

    if audio:
        audio.start()

    # For video files, read frames sequentially in the main loop.
    # _CaptureThread (drains buffer continuously) is only useful for webcam,
    # where it prevents stale frames from accumulating.  For a file it reads
    # frames faster than real-time, causing every main-loop iteration to see
    # a frame many positions ahead — making the video play at several × speed.
    is_file   = (args.source != 'webcam')
    capture_t = None if is_file else _CaptureThread(cap)
    pose_t    = _PoseThread(pose_det)
    ball_t    = _BallThread(ball_det)

    # ---- optional video writer ----
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save, fourcc, fps, (frame_w, frame_h))
        print(f"[Coach] Saving annotated video → {args.save}")

    # ---- rolling state ----
    ball_win_x: deque = deque(maxlen=90)
    ball_win_y: deque = deque(maxlen=90)
    ball_win_f: deque = deque(maxlen=90)

    bounce_frame_set: set[int] = set()
    BOUNCE_INTERVAL  = 30
    MIN_BALL_DETECTS = 10

    recent_tips: list[str] = []
    last_active_msg: str | None = None
    frame_idx = 0

    # Audio: only speak high/ai tips, at most once every N seconds
    AUDIO_COOLDOWN  = 2.0
    last_audio_time = 0.0

    # Frame rate limiter — keeps video playing at its native speed
    frame_delay = 1.0 / fps

    # FPS tracking
    fps_time  = time.time()
    fps_count = 0
    fps_display = 0.0

    # Wait for first frame (webcam only — file reads are immediate)
    if capture_t is not None:
        while True:
            ret, _ = capture_t.read()
            if ret:
                break
            time.sleep(0.01)

    print("[Coach] Running — press Q to quit.\n")

    try:
        while True:
            frame_start = time.time()   # used for per-frame throttle below

            if is_file:
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                if capture_t.ended:
                    break
                ret, frame = capture_t.read()
                if not ret:
                    time.sleep(0.005)
                    continue

            # ---- push to background threads ----
            pose_t.push(frame)
            landmarks = pose_t.get_landmarks()

            if landmarks is not None:
                ball_t.push(frame)
            ball_pos = ball_t.get_pos()

            ball_win_x.append(ball_pos[0])
            ball_win_y.append(ball_pos[1])
            ball_win_f.append(frame_idx)

            # ---- bounce (periodic) ----
            is_bounce = False
            if (frame_idx > 0
                    and frame_idx % BOUNCE_INTERVAL == 0
                    and sum(1 for v in ball_win_x if v is not None) >= MIN_BALL_DETECTS):
                try:
                    xs      = list(ball_win_x)
                    ys      = list(ball_win_y)
                    start_f = ball_win_f[0]
                    rel     = bounce_det.predict(xs, ys)
                    bounce_frame_set = {start_f + r for r in rel}
                except Exception as exc:
                    print(f"[Coach] Bounce error: {exc}")
                    bounce_frame_set = set()

            if frame_idx in bounce_frame_set:
                is_bounce = True

            # ---- coaching engine ----
            active_tip, metrics, shot_event = engine.process_frame(
                frame_idx, landmarks, ball_pos, is_bounce
            )

            # ---- AI tip on shot events ----
            if shot_event and ai_coach:
                swing_m = engine.swing.analyze(landmarks) if landmarks is not None else {}
                foot_m  = engine.footwork.get_metrics(landmarks)
                ai_coach.request_tip(engine, shot_event.shot_type,
                                     swing_m, foot_m, recent_tips[-8:])

            # ---- audio ----
            if active_tip and active_tip.message != last_active_msg:
                recent_tips.append(active_tip.message)
                last_active_msg = active_tip.message
                now = time.time()
                if (audio
                        and active_tip.priority in ('high', 'ai')
                        and now - last_audio_time >= AUDIO_COOLDOWN):
                    audio.speak(active_tip.message, priority=active_tip.priority)
                    last_audio_time = now

            # ---- FPS counter ----
            fps_count += 1
            now = time.time()
            if now - fps_time >= 1.0:
                fps_display = fps_count / (now - fps_time)
                fps_count = 0
                fps_time  = now
            metrics['fps'] = fps_display

            # ---- render ----
            ov.update_ball(ball_pos)
            annotated = ov.draw(frame, active_tip, metrics,
                                engine.shots.phase, engine.shots.shot_count, landmarks)

            cv2.imshow("Tennis AI Coach  [Q = quit]", annotated)
            if writer:
                writer.write(annotated)

            # Throttle to source FPS so video doesn't play faster than real-time
            elapsed = time.time() - frame_start
            wait_ms = max(1, int((frame_delay - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                print(f"[Coach] Reached --max-frames {args.max_frames}, stopping.")
                break

    finally:
        if capture_t is not None:
            capture_t.stop()   # releases cap and joins
        else:
            cap.release()
        pose_t.stop()
        ball_t.stop()
        cv2.destroyAllWindows()
        if writer:
            writer.release()

        # Speak the single most-repeated coaching cue so the player knows
        # what to focus on next time (tips that fired during play but were
        # silenced by the cooldown still count in _tip_counts).
        if audio and frame_idx > 0:
            top = engine.get_session_summary().get('top_tips', [])
            if top:
                audio.speak(f"Session complete. Key focus: {top[0][0]}", priority='high')
                time.sleep(6)   # give say time to finish before we kill the thread

        if audio:
            audio.stop()

    # ---- post-session report ----
    if frame_idx > 0:
        summary = engine.get_session_summary()
        session_report.print_summary(summary)
        session_report.generate(summary, output_dir=os.path.join(_HERE, 'sessions'))
    else:
        print("[Coach] No frames processed.")


if __name__ == '__main__':
    main()
