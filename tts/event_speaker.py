"""Text-to-Speech Module - Non-Blocking Audio Output

Speaks navigation instructions asynchronously using platform-native TTS:
- macOS: native `say` command
- Linux/Windows: pyttsx3 library

Features:
- Non-blocking async processing (uses worker thread)
- Deduplication (same instruction won't repeat within cooldown)
- Instruction-change detection (smart TTS gating)
- Passive instruction filtering (\"Continue forward\" gated in normal mode)
- Queue management (prevents audio pile-up)

TTS Gating Rules:
- Critical urgency: Always speak (0.4s cooldown)
- Non-passive + changed: Speak if > 0.6s since last
- Other: Speak if > 2s since last and changed
- Passive in info mode: Never speaks (suppressed)
"""

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
        quick_change_ok = elapsed >= 0.6

        is_passive = "continue" in text.lower()  # FIXED

        should_speak = False

        if urgency == "critical":
            should_speak = critical_cooldown
        elif changed and (cooldown_ok or (not is_passive and quick_change_ok)):
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
        if not hasattr(self, '_pyttsx3_engine'):
            import pyttsx3
            self._pyttsx3_engine = pyttsx3.init()
            self._pyttsx3_engine.setProperty("rate", self._rate)
        self._pyttsx3_engine.say(text)
        self._pyttsx3_engine.runAndWait()

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
        # Drain any pending utterances so stale guidance is not spoken after loop end.
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        # Allow in-flight speech to finish naturally when possible.
        self._queue.put(None)
        self._thread.join(timeout=5)
        self._kill_current()