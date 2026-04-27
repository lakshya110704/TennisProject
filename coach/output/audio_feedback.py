import queue
import subprocess
import threading
import sys


class AudioFeedback:
    """
    Speaks coaching tips using the OS text-to-speech engine.

    macOS: uses the built-in `say` command — no dependencies, no threading quirks.
    Other platforms: falls back to pyttsx3 if available, otherwise silent.
    """

    def __init__(self, rate: int = 200):
        self._rate     = rate   # words per minute (macOS `say` flag)
        self._queue:   queue.Queue = queue.Queue()
        self._stop     = threading.Event()
        self._thread:  threading.Thread | None = None
        self._is_mac   = sys.platform == "darwin"

    # ------------------------------------------------------------------ public

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="audio-feedback"
        )
        self._thread.start()

    def stop(self):
        # Drain queue then signal thread to exit
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(None)   # sentinel to unblock get()
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def speak(self, message: str, priority: str = 'medium'):
        """
        Speak message. Drains any waiting tip first so audio never lags.
        """
        # Drop stale queued tips — only keep the latest
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(message)

    # ----------------------------------------------------------------- private

    def _run(self):
        if self._is_mac:
            self._run_mac()
        else:
            self._run_pyttsx3()

    def _run_mac(self):
        proc = None
        while not self._stop.is_set():
            try:
                msg = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if msg is None:
                break
            # Kill any currently-running say so the new tip is heard immediately
            if proc and proc.poll() is None:
                proc.kill()
                proc.wait()
            try:
                proc = subprocess.Popen(["say", "-r", str(self._rate), msg])
            except Exception as e:
                print(f"[AudioFeedback] say error: {e}")
        # Clean up any lingering say process on exit
        if proc and proc.poll() is None:
            proc.kill()
            proc.wait()

    def _run_pyttsx3(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', self._rate)
            engine.setProperty('volume', 0.9)
        except Exception as e:
            print(f"[AudioFeedback] TTS unavailable: {e}")
            return

        while not self._stop.is_set():
            try:
                msg = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if msg is None:
                break
            try:
                engine.say(msg)
                engine.runAndWait()
            except Exception as e:
                print(f"[AudioFeedback] TTS error: {e}")

        try:
            engine.stop()
        except Exception:
            pass
