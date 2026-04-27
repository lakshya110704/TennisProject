import threading
import time
import anthropic


# Haiku is intentional here — tips must arrive within a few seconds of a shot event.
# Opus/Sonnet latency would make the feedback arrive too late to be useful mid-rally.
_MODEL = "claude-haiku-4-5"

_SYSTEM_PROMPT = """You are an elite tennis coach watching a live training session.
After each shot you observe, give ONE short coaching tip.

Rules:
- Exactly 1-2 sentences. No more.
- Be direct and specific ("Finish your swing higher" not "Consider following through more")
- Focus only on the single most impactful thing to fix right now
- Never repeat advice that was recently given
- No filler: no "Great job!", no "Keep it up!", no preamble
- Output the tip and nothing else"""


class AICoach:
    """
    Calls Claude in a background thread after shot events.
    The response is injected into CoachingEngine via inject_ai_tip().

    Usage:
        coach = AICoach()                          # reads ANTHROPIC_API_KEY from env
        coach.request_tip(engine, shot_type, swing_metrics, foot_metrics, recent_tips)
    """

    MIN_CALL_INTERVAL = 30.0   # seconds — never hammer the API between points

    def __init__(self, api_key: str | None = None):
        # api_key=None → SDK reads ANTHROPIC_API_KEY environment variable
        self._client = anthropic.Anthropic(api_key=api_key)
        self._last_call_time: float = 0.0
        self._lock = threading.Lock()
        self.call_count: int = 0

    # ----------------------------------------------------------------- public

    def request_tip(
        self,
        coaching_engine,        # CoachingEngine — has inject_ai_tip()
        shot_type: str,
        swing_metrics: dict,
        foot_metrics: dict,
        recent_tips: list[str],
    ) -> bool:
        """
        Non-blocking. Fires a background thread if the cooldown has passed.
        Returns True if a request was dispatched, False if still cooling down.
        """
        with self._lock:
            now = time.time()
            if now - self._last_call_time < self.MIN_CALL_INTERVAL:
                return False
            self._last_call_time = now
            self.call_count += 1

        thread = threading.Thread(
            target=self._call_claude,
            args=(coaching_engine, shot_type, swing_metrics, foot_metrics, recent_tips),
            daemon=True,
            name=f"ai-coach-{self.call_count}",
        )
        thread.start()
        return True

    # ---------------------------------------------------------------- private

    def _call_claude(
        self,
        coaching_engine,
        shot_type: str,
        swing_metrics: dict,
        foot_metrics: dict,
        recent_tips: list[str],
    ):
        try:
            user_msg = self._build_prompt(shot_type, swing_metrics, foot_metrics, recent_tips)

            response = self._client.messages.create(
                model=_MODEL,
                max_tokens=80,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},   # cached — never changes
                    }
                ],
                messages=[{"role": "user", "content": user_msg}],
            )

            tip = response.content[0].text.strip()
            if tip:
                coaching_engine.inject_ai_tip(tip)

        except anthropic.RateLimitError:
            print("[AICoach] Rate limited — skipping this tip")
        except anthropic.APIError as e:
            print(f"[AICoach] API error ({e.status_code}): {e.message}")
        except Exception as e:
            print(f"[AICoach] Unexpected error: {e}")

    def _build_prompt(
        self,
        shot_type: str,
        swing_metrics: dict,
        foot_metrics: dict,
        recent_tips: list[str],
    ) -> str:
        lines = [f"Shot just detected: {shot_type}"]

        # --- swing ---
        elbow = swing_metrics.get("elbow_angle")
        knee  = swing_metrics.get("knee_angle")
        rot   = swing_metrics.get("shoulder_rotation")

        if elbow is not None:
            note = ""
            if elbow < 95:   note = " ← too cramped"
            elif elbow > 168: note = " ← arm locked"
            lines.append(f"Elbow angle: {elbow:.0f}°{note}  (ideal 120-160°)")

        if knee is not None:
            note = " ← knees too straight" if knee > 165 else ""
            lines.append(f"Knee bend angle: {knee:.0f}°{note}  (ideal 130-160°)")

        if rot is not None:
            note = " ← little body rotation" if rot < 18 else ""
            lines.append(f"Shoulder rotation: {rot:.0f}°{note}  (ideal >20°)")

        if swing_metrics.get("contact_in_front") is False:
            lines.append("Contact point: hitting late — ball behind body")

        if swing_metrics.get("wrist_above_shoulder") is False and elbow and elbow < 150:
            lines.append("Follow-through: wrist not finishing above shoulder")

        # --- footwork ---
        stance = foot_metrics.get("stance_width")
        if stance is not None:
            if stance < 0.12:
                lines.append("Stance: feet too close together")
            elif stance > 0.45:
                lines.append("Stance: feet too wide apart")

        idle = foot_metrics.get("idle_ratio", 0.0)
        if idle > 0.65:
            lines.append(f"Movement: player stationary {idle*100:.0f}% of the time — not staying on toes")

        if foot_metrics.get("off_center"):
            lines.append("Positioning: player out of position, too far from center")

        # --- avoid repeating recent advice ---
        if recent_tips:
            last = "; ".join(recent_tips[-4:])
            lines.append(f"\nTips already given (do NOT repeat): {last}")

        lines.append("\nWhat is the single most important coaching point right now?")
        return "\n".join(lines)
