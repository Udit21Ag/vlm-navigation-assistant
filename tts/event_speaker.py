"""
Layer 7 — Event-Driven Text-to-Speech

Runs TTS in a background process so it never blocks the video loop.
Only speaks when:
  1. The instruction has CHANGED, or
  2. A cooldown period (default 2 s) has elapsed, or
  3. An URGENT hazard overrides the cooldown.

On macOS uses the native `say` command (avoids pyttsx3 NSRunLoop
threading issues).  On other platforms falls back to pyttsx3 in a
subprocess.
"""

import platform
import subprocess
import threading
import time
import queue


class EventSpeaker:
    """Non-blocking, event-driven TTS engine."""

    def __init__(self, cooldown_seconds=2.0, rate=170):
        """
        Args:
            cooldown_seconds: Minimum gap between two speech events.
            rate: Speech rate (words per minute) — used by pyttsx3 fallback.
        """
        self._cooldown = cooldown_seconds
        self._rate = rate
        self._is_mac = platform.system() == "Darwin"

        self._last_spoken_text = ""
        self._last_spoken_time = 0.0

        # Single-slot queue — only the latest message matters
        self._queue = queue.Queue(maxsize=1)
        self._current_proc = None  # track running speech subprocess

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API (called from the main video loop — non-blocking)
    # ------------------------------------------------------------------
    def speak(self, text, urgency="info"):
        """
        Enqueue a message for speech if the trigger conditions are met.

        Args:
            text: The instruction text to speak.
            urgency: "critical" | "warning" | "info"
        """
        now = time.monotonic()
        changed = (text != self._last_spoken_text)
        elapsed = now - self._last_spoken_time
        cooldown_ok = elapsed >= self._cooldown

        should_speak = False

        # Define passive instructions that shouldn't be repeated
        is_passive = text in (
            "Continue forward.",
            "Continue forward",
        )

        if urgency == "critical":
            should_speak = True            # always speak urgent hazards
        elif changed and cooldown_ok:
            should_speak = True            # instruction changed + cooldown ok
        elif cooldown_ok and text and not is_passive:
            should_speak = True            # periodic refresh (not passive)

        if not should_speak:
            return

        # Drop stale messages — only keep the latest
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass

        self._queue.put(text)
        self._last_spoken_text = text
        self._last_spoken_time = now

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------
    def _worker(self):
        """Daemon thread that pulls messages and speaks them."""
        while True:
            text = self._queue.get()  # blocks until a message arrives
            if text is None:
                break  # poison pill → shut down

            # Kill any in-progress speech so we always speak the LATEST
            self._kill_current()

            try:
                if self._is_mac:
                    self._speak_mac(text)
                else:
                    self._speak_pyttsx3(text)
            except Exception as e:
                print(f"[TTS] Error: {e}")

    def _speak_mac(self, text):
        """Use macOS native `say` command — runs in its own process."""
        # Sanitise text: remove characters that could break the shell
        safe = text.replace('"', '\\"')
        self._current_proc = subprocess.Popen(
            ["say", safe],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._current_proc.wait()

    def _speak_pyttsx3(self, text):
        """Fallback for non-macOS: pyttsx3 in the worker thread."""
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    def _kill_current(self):
        """Terminate any currently-running speech subprocess."""
        if self._current_proc is not None:
            try:
                if self._current_proc.poll() is None:
                    self._current_proc.terminate()
                    self._current_proc.wait(timeout=1)
            except Exception:
                pass
            self._current_proc = None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def shutdown(self):
        """Gracefully stop the background thread."""
        # Drain the queue
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass

        # Wait for current speech to finish (up to 5s)
        self._queue.put(None)
        self._thread.join(timeout=5)
        self._kill_current()
