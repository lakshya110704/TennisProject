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
import argparse
import sys
import os
import queue
import threading
import time
import cv2
import numpy as np
import torch
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))

from coach.detectors.ball_detector import BallDetector
from coach.detectors.bounce_detector import BounceDetector
from coach.detectors.pose_detection import PoseDetector
from coach.coaching.coaching_engine import CoachingEngine
from coach.coaching.ai_coach import AICoach
from coach.output.audio_feedback import AudioFeedback
from coach.output.overlay import Overlay
from coach.coaching import session_report


# ---------------------------------------------------------------- ball inference

def _infer_ball_frame(
    detector: BallDetector,
    frames: list,           # already resized to (detector.width, detector.height)
    prev_pred: list,
) -> tuple[tuple, list]:
    imgs = np.concatenate((frames[2], frames[1], frames[0]), axis=2).astype(np.float32) / 255.0
    imgs = np.rollaxis(imgs, 2, 0)
    inp  = np.expand_dims(imgs, axis=0)

    with torch.no_grad():
        out = detector.model(torch.from_numpy(inp).float().to(detector.device))

    output = out.argmax(dim=1).detach().cpu().numpy()
    x, y = detector.postprocess(output, prev_pred)
    return (x, y), ([x, y] if x is not None else prev_pred)


# ---------------------------------------------------------------- background threads

class _CaptureThread:
    """
    Continuously drains the camera buffer and keeps only the latest frame.
    Without this, slow processing causes cap.read() to return frames that
    are several seconds old.
    """
    def __init__(self, cap: cv2.VideoCapture):
        self._cap   = cap
        self._frame = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._t     = threading.Thread(target=self._run, daemon=True, name="capture")
        self._t.start()

    def _run(self):
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self):
        self._stop.set()


class _PoseThread:
    """
    Runs MediaPipe pose detection in a background thread.
    Push a frame in; read back the latest landmarks without blocking.
    """
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


class _BallThread:
    """
    Runs TrackNet inference in a background thread.
    Accepts pre-resized frames (640×360) to reduce copy cost.
    Stale frames are dropped (queue depth = 1) so inference always works on
    the most recent frame available.
    """
    def __init__(self, detector: BallDetector):
        self._detector  = detector
        self._q         = queue.Queue(maxsize=1)
        self._ball_pos  = (None, None)
        self._lock      = threading.Lock()
        self._stop      = threading.Event()
        self._buf       = deque(maxlen=3)
        self._prev_pred = [None, None]
        self._t         = threading.Thread(target=self._run, daemon=True, name="ball-det")
        self._t.start()

    def push(self, frame: np.ndarray):
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            pass   # drop stale frame — the next one will be fresher

    def get_pos(self) -> tuple:
        with self._lock:
            return self._ball_pos

    def _run(self):
        while not self._stop.is_set():
            try:
                frame = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            self._buf.append(frame)
            if len(self._buf) == 3:
                pos, self._prev_pred = _infer_ball_frame(
                    self._detector, list(self._buf), self._prev_pred
                )
                with self._lock:
                    self._ball_pos = pos

    def stop(self):
        self._stop.set()


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

    # ---- models ----
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    if device == 'cpu':
        print("[Coach] Warning: no GPU found — TrackNet will run slowly on CPU")
    print(f"[Coach] Loading models ({device})…")
    ball_det   = BallDetector(f'{args.model_dir}/ball_model.pt', device=device)
    bounce_det = BounceDetector(f'{args.model_dir}/bounce_model.cbm')
    pose_det   = PoseDetector()
    print("[Coach] Models ready.")

    # ---- coaching layer ----
    engine   = CoachingEngine(dominant_hand=args.hand, frame_w=frame_w, frame_h=frame_h)
    ai_coach = None if args.no_ai else AICoach()
    audio    = None if args.no_audio else AudioFeedback()
    ov       = Overlay(frame_w, frame_h)

    if audio:
        audio.start()

    # ---- background threads ----
    capture_t = _CaptureThread(cap)
    pose_t    = _PoseThread(pose_det)
    ball_t    = _BallThread(ball_det)
    _BALL_W, _BALL_H = ball_det.width, ball_det.height   # 640×360

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

    # Wait for first frame before entering loop
    while True:
        ret, _ = capture_t.read()
        if ret:
            break
        time.sleep(0.01)

    print("[Coach] Running — press Q to quit.\n")

    try:
        while True:
            ret, frame = capture_t.read()
            if not ret:
                time.sleep(0.005)
                continue

            # ---- push full frame to pose thread ----
            pose_t.push(frame)

            # ---- read latest landmarks from pose thread ----
            landmarks = pose_t.get_landmarks()

            # ---- push pre-resized frame to ball thread (only when player visible) ----
            if landmarks is not None:
                small = cv2.resize(frame, (_BALL_W, _BALL_H))
                ball_t.push(small)

            # ---- read latest ball position from ball thread ----
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
                if audio:
                    audio.speak(active_tip.message, priority=active_tip.priority)

            # ---- render ----
            ov.update_ball(ball_pos)
            annotated = ov.draw(frame, active_tip, metrics,
                                engine.shots.phase, engine.shots.shot_count, landmarks)

            cv2.imshow("Tennis AI Coach  [Q = quit]", annotated)
            if writer:
                writer.write(annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                print(f"[Coach] Reached --max-frames {args.max_frames}, stopping.")
                break

    finally:
        capture_t.stop()
        pose_t.stop()
        ball_t.stop()
        cap.release()
        cv2.destroyAllWindows()
        if writer:
            writer.release()
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
