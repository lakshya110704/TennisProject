import threading
import queue
import pyttsx3


class AudioFeedback:
    """
    Speaks coaching tips aloud using pyttsx3 (offline TTS).

    Tips are queued and spoken one at a time in a background thread.
    A high-priority tip can interrupt a low-priority one that is still waiting.

    Usage:
        audio = AudioFeedback()
        audio.start()
        audio.speak("Bend your knees", priority='high')
        audio.stop()   # call on exit — waits for current speech to finish
    """

    # How many waiting tips can accumulate before we drop old low-priority ones
    _QUEUE_MAX = 3

    def __init__(self, rate: int = 165, volume: float = 0.9):
        self._rate = rate       # words per minute
        self._volume = volume

        # Each item: (priority_value, message)
        # Lower value = higher priority (mirrors CoachingEngine ordering)
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = 0           # tie-breaker so same-priority items stay FIFO
        self._seq_lock = threading.Lock()

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # pyttsx3 engine lives on the worker thread (not thread-safe across threads)
        self._engine: pyttsx3.Engine | None = None

    # ------------------------------------------------------------------ public

    def start(self):
        """Start the background speaker thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="audio-feedback",
        )
        self._thread.start()

    def stop(self):
        """Signal the thread to finish after current utterance and wait."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def speak(self, message: str, priority: str = 'medium'):
        """
        Enqueue a tip to be spoken.

        priority: 'high' | 'ai' | 'medium' | 'low'
        High-priority tips jump to the front; excess low-priority tips are
        dropped if the queue is already full.
        """
        pval = {'high': 0, 'ai': 1, 'medium': 2, 'low': 3}.get(priority, 2)

        with self._seq_lock:
            seq = self._seq
            self._seq += 1

        # Drop old low-priority tips when the queue is getting long
        if self._queue.qsize() >= self._QUEUE_MAX and pval >= 2:
            return

        self._queue.put((pval, seq, message))

    @property
    def is_speaking(self) -> bool:
        """True if the worker thread exists and is alive."""
        return self._thread is not None and self._thread.is_alive()

    # ----------------------------------------------------------------- private

    def _run(self):
        self._engine = pyttsx3.init()
        self._engine.setProperty('rate', self._rate)
        self._engine.setProperty('volume', self._volume)

        while not self._stop_event.is_set():
            try:
                pval, _seq, message = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._engine.say(message)
                self._engine.runAndWait()
            except Exception as e:
                print(f"[AudioFeedback] TTS error: {e}")

        # Cleanup
        try:
            self._engine.stop()
        except Exception:
            pass
