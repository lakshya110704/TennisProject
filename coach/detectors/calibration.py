import time
import cv2
import numpy as np
from dataclasses import dataclass

from coach.analysis.swing_analyzer import _angle, _LM


@dataclass
class CalibrationData:
    """Body-proportion measurements taken at session start."""
    natural_knee_angle:  float = 170.0   # avg knee angle while standing straight (degrees)
    shoulder_width_norm: float = 0.20    # left-to-right shoulder in normalised coords
    arm_length_norm:     float = 0.35    # dominant shoulder-to-wrist distance (normalised)
    valid: bool = False                  # False → not enough data, keep defaults


class CalibrationMode:
    """
    Displays a 5-second countdown and collects standing body-proportion
    measurements from the live camera.  Call run() before starting the
    main coaching loop; pass the returned CalibrationData to each analyzer.

    Only meaningful for webcam input — video files skip calibration.
    """

    DURATION   = 5.0   # seconds of data collection
    MIN_FRAMES = 30    # minimum valid-pose frames to accept the result

    def run(self, cap: cv2.VideoCapture,
            pose_det,
            frame_w: int, frame_h: int) -> CalibrationData:
        """
        Blocks for ~DURATION seconds then returns CalibrationData.
        cap must already be open; it is not released here.
        """
        knees, shoulders, arms = [], [], []
        start = time.time()

        while True:
            elapsed   = time.time() - start
            remaining = max(0.0, self.DURATION - elapsed)

            ret, frame = cap.read()
            if not ret:
                break

            lm = pose_det.detect(frame)
            if lm is not None and len(lm) >= 29:
                self._collect(lm, knees, shoulders, arms)

            self._draw_ui(frame, remaining, len(knees), lm is not None, frame_w, frame_h)
            cv2.imshow("Tennis AI Coach  [Q = quit]", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or elapsed >= self.DURATION:
                break

        # --- brief "complete" flash ---
        if ret:
            ok = len(knees) >= self.MIN_FRAMES
            self._draw_complete(frame, ok)
            cv2.imshow("Tennis AI Coach  [Q = quit]", frame)
            cv2.waitKey(1200)

        # --- aggregate ---
        data = CalibrationData()
        if len(knees) >= self.MIN_FRAMES:
            data.natural_knee_angle  = float(np.mean(knees))
            data.shoulder_width_norm = float(np.mean(shoulders))
            data.arm_length_norm     = float(np.mean(arms))
            data.valid               = True
            print(
                f"[Calibration] knee={data.natural_knee_angle:.1f}°  "
                f"shoulder_w={data.shoulder_width_norm:.3f}  "
                f"arm_len={data.arm_length_norm:.3f}  "
                f"(n={len(knees)} frames)"
            )
        else:
            print(f"[Calibration] Only {len(knees)} valid frames — using default thresholds.")

        return data

    # ----------------------------------------------------------------- internal

    def _collect(self, lm: np.ndarray,
                 knees: list, shoulders: list, arms: list):
        """Extract one frame's measurements and append to running lists."""
        try:
            # Average both legs for a more stable knee-angle reading
            for hip_k, knee_k, ank_k in [
                ('r_hip', 'r_knee', 'r_ankle'),
                ('l_hip', 'l_knee', 'l_ankle'),
            ]:
                a = _angle(lm[_LM[hip_k]][:2], lm[_LM[knee_k]][:2], lm[_LM[ank_k]][:2])
                knees.append(a)
        except Exception:
            pass

        try:
            sw = float(abs(lm[_LM['r_shoulder']][0] - lm[_LM['l_shoulder']][0]))
            shoulders.append(sw)
        except Exception:
            pass

        try:
            al = float(np.linalg.norm(
                lm[_LM['r_wrist']][:2] - lm[_LM['r_shoulder']][:2]
            ))
            arms.append(al)
        except Exception:
            pass

    def _draw_ui(self, frame: np.ndarray, remaining: float,
                 frame_count: int, pose_found: bool,
                 frame_w: int, frame_h: int):
        h, w = frame.shape[:2]

        # --- top instruction banner ---
        banner_h = 90
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, banner_h), (10, 10, 10), -1)
        cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)

        cv2.putText(frame,
                    "CALIBRATION — Stand straight, face the camera",
                    (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                    (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame,
                    "Lift your racket to ready position and hold still",
                    (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 200, 255), 1, cv2.LINE_AA)

        status_text  = "Pose detected" if pose_found else "No pose — step into frame"
        status_color = (0, 210, 90)    if pose_found else (0, 100, 220)
        cv2.putText(frame, status_text,
                    (16, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    status_color, 1, cv2.LINE_AA)

        frames_txt = f"{frame_count} frames collected"
        cv2.putText(frame, frames_txt,
                    (w - 200, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (140, 140, 140), 1, cv2.LINE_AA)

        # --- bottom progress bar ---
        bar_y  = h - 28
        bx0    = 16
        bx1    = w - 16
        fill_x = int(bx0 + (bx1 - bx0) * (1.0 - remaining / self.DURATION))

        cv2.rectangle(frame, (bx0, bar_y), (bx1, bar_y + 16), (30, 30, 30), -1)
        cv2.rectangle(frame, (bx0, bar_y), (fill_x, bar_y + 16), (0, 200, 90), -1)
        cv2.rectangle(frame, (bx0, bar_y), (bx1, bar_y + 16), (70, 70, 70), 1)

        cv2.putText(frame, f"{remaining:.1f}s",
                    (bx1 + 6, bar_y + 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (190, 190, 190), 1, cv2.LINE_AA)

    def _draw_complete(self, frame: np.ndarray, success: bool):
        h, w = frame.shape[:2]
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)

        msg   = "Calibration complete — coaching starts now!" if success \
                else "Not enough data — using default thresholds"
        color = (0, 220, 100) if success else (0, 120, 220)
        tw    = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0][0]
        cv2.putText(frame, msg,
                    ((w - tw) // 2, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
