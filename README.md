# Tennis AI Coach

A real-time AI coaching system that watches you play tennis via webcam or video file and gives live feedback on your swing mechanics, footwork, and positioning — both as a visual overlay and spoken audio cues. After each session it generates an HTML summary report.

---

## How it works

Each frame is processed by a pipeline of three background threads that run in parallel so the display is never blocked:

- **Camera thread** — continuously drains the capture buffer so the main loop always gets the freshest frame
- **Pose thread** — MediaPipe detects 33 body landmarks every frame
- **Ball thread** — TrackNet locates the ball across 3 consecutive frames (only runs when a player is visible)

The coaching engine receives pose + ball data every frame, runs biomechanical analysis, maintains a priority tip queue, and fires Claude Haiku in a separate daemon thread after each detected shot for richer AI-generated feedback.

---

## Features

- Live skeleton overlay, ball trail, and stats HUD on the video feed
- Rule-based tips for swing mechanics (elbow angle, knee bend, shoulder rotation, contact point, follow-through) and footwork (stance width, idle ratio, off-center position)
- AI coaching tips via Claude Haiku — triggered after each detected shot, injected asynchronously so they never delay the display
- Text-to-speech audio feedback (offline, via pyttsx3) with priority queue
- Shot detection and classification (Forehand / Backhand / Serve) using wrist speed + ball proximity
- Post-session HTML report with shot breakdown, average swing metrics, footwork stats, and most frequent coaching cues

---

## Project structure

```
TennisProject/
├── live_coach.py              # entry point
├── requirements.txt
└── coach/
    ├── detectors/             # ball_detector, bounce_detector, pose_detection, tracknet
    ├── court/                 # court_detection_net, homography, postprocess
    ├── analysis/              # swing_analyzer, footwork_analyzer, shot_detector
    ├── coaching/              # coaching_engine, ai_coach, session_report
    └── output/                # overlay, audio_feedback
```

---

## Setup

```bash
git clone https://github.com/lakshya110704/TennisProject.git
cd TennisProject
pip install -r requirements.txt
```

Download model weights and place them in a `models/` folder:
- `models/ball_model.pt` — TrackNet (ball tracking)
- `models/bounce_model.cbm` — CatBoost (bounce detection)
- `models/court_model.pt` — Court keypoint detector

---

## Usage

```bash
# Webcam (run from your own terminal — macOS needs camera permission)
python live_coach.py --source webcam

# Video file
python live_coach.py --source path/to/video.mp4

# Left-handed player
python live_coach.py --source webcam --hand left

# Save annotated output video
python live_coach.py --source webcam --save output.mp4

# Disable AI tips or audio
python live_coach.py --source webcam --no-ai --no-audio
```

Set `ANTHROPIC_API_KEY` in your environment to enable Claude coaching tips.

Press **Q** to quit. A session report is saved to `sessions/` on exit.

---

## Models used

| Model | Purpose |
|---|---|
| [TrackNet](https://github.com/yastrebksv/TrackNet) | Ball detection across 3 consecutive frames |
| CatBoostRegressor | Bounce detection from ball trajectory |
| Court keypoint net | 14-point court homography |
| MediaPipe Pose | 33-landmark body pose (built-in, no download needed) |
