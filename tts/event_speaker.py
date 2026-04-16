import platform
import subprocess
import threading
import time
import queue


class EventSpeaker:
    """Non-blocking, event-driven TTS engine."""

    def __init__(self, cooldown_seconds=2.0, rate=170):
        self._cooldown = cooldown_seconds
        self._rate = rate
        self._is_mac = platform.system() == "Darwin"

        self._last_spoken_text = ""
        self._last_spoken_time = 0.0

        self._queue = queue.Queue()   # FIXED
        self._current_proc = None

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # -------------------------------
    # Public API
    # -------------------------------
    def speak(self, text, urgency="info"):
        now = time.monotonic()

        changed = (text != self._last_spoken_text)
        elapsed = now - self._last_spoken_time
        cooldown_ok = elapsed >= self._cooldown
        critical_cooldown = elapsed >= max(0.4, self._cooldown * 0.25)

        is_passive = "continue" in text.lower()  # FIXED

        should_speak = False

        if urgency == "critical":
            should_speak = critical_cooldown
        elif changed and cooldown_ok:
            should_speak = True
        elif cooldown_ok and not is_passive:
            should_speak = True

        if not should_speak:
            return

        # Clear old messages
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        self._queue.put(text)

        self._last_spoken_text = text
        self._last_spoken_time = now

    # -------------------------------
    # Worker thread
    # -------------------------------
    def _worker(self):
        while True:
            text = self._queue.get()

            if text is None:
                break

            self._kill_current()

            try:
                if self._is_mac:
                    self._speak_mac(text)
                else:
                    self._speak_pyttsx3(text)
            except Exception as e:
                print(f"[TTS] Error: {e}")

    def _speak_mac(self, text):
        """macOS native TTS"""
        self._current_proc = subprocess.Popen(
            ["say", text],   # FIXED (no escaping)
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._current_proc.wait()

    def _speak_pyttsx3(self, text):
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    def _kill_current(self):
        if self._current_proc is not None:
            try:
                if self._current_proc.poll() is None:
                    self._current_proc.terminate()
                    self._current_proc.wait(timeout=1)
            except Exception:
                pass
            self._current_proc = None

    # -------------------------------
    # Cleanup
    # -------------------------------
    def shutdown(self):
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass

        self._queue.put(None)
        self._thread.join(timeout=5)
        self._kill_current()