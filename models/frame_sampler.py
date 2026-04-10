"""
Layer 1 — Video Input & Frame Sampling

Captures frames from a video source (webcam or file) and samples
one frame every `sample_interval_ms` milliseconds.  Maintains a
sliding buffer of the last N sampled frames for temporal modules.

For VIDEO FILES  → uses video-time–based seeking (CAP_PROP_POS_MSEC)
For WEBCAM/LIVE  → uses wall-clock timing
"""

import cv2
import time
from collections import deque


class FrameSampler:
    """Iterator that yields sampled frames from a video source."""

    def __init__(self, source=0, sample_interval_ms=300, buffer_size=5):
        """
        Args:
            source: Webcam index (int) or path to video file (str).
            sample_interval_ms: Minimum gap between sampled frames in ms.
            buffer_size: Number of recent frames kept in the sliding buffer.
        """
        self.source = source
        self.sample_interval_ms = sample_interval_ms
        self.buffer_size = buffer_size

        # The sliding window — most-recent frame is at the right end
        self.buffer = deque(maxlen=buffer_size)

        self._cap = None
        self._last_sample_time = 0.0
        self._is_file = isinstance(source, str)

    # ------------------------------------------------------------------
    # Context-manager support so callers can use `with FrameSampler(…):`
    # ------------------------------------------------------------------
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def open(self):
        """Open the video source."""
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open video source: {self.source}"
            )
        self._last_sample_time = 0.0

        if self._is_file:
            fps = self._cap.get(cv2.CAP_PROP_FPS) or 30
            total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = total / fps if fps else 0
            print(f"[VIDEO] {self.source}")
            print(f"        {total} frames, {fps:.1f} FPS, "
                  f"{duration:.1f}s duration")
            print(f"        Sampling 1 frame every {self.sample_interval_ms}ms "
                  f"→ ~{int(duration * 1000 / self.sample_interval_ms)} "
                  f"frames to process")

    def release(self):
        """Release the video source."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self):
        """Frames-per-second reported by the capture device."""
        if self._cap is None:
            return 0
        return self._cap.get(cv2.CAP_PROP_FPS) or 30

    def __iter__(self):
        """Yield (frame, timestamp) tuples at the configured sample rate."""
        if self._cap is None:
            self.open()

        if self._is_file:
            yield from self._iter_file()
        else:
            yield from self._iter_realtime()

    # ------------------------------------------------------------------
    # Video file: seek by video time (not wall clock)
    # ------------------------------------------------------------------
    def _iter_file(self):
        """
        For video files, use the video's own timeline to decide which
        frames to sample.  This correctly skips frames so a 60-second
        video at 300ms sampling yields ~200 frames regardless of how
        long each frame takes to process.
        """
        next_sample_ms = 0.0

        while True:
            # Seek to the next target position in the video
            self._cap.set(cv2.CAP_PROP_POS_MSEC, next_sample_ms)
            ret, frame = self._cap.read()
            if not ret:
                break

            # Record the actual position we landed on
            actual_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            next_sample_ms = actual_ms + self.sample_interval_ms

            now = time.monotonic()
            self.buffer.append((frame, now))

            yield frame, now

    # ------------------------------------------------------------------
    # Webcam / live stream: use wall-clock timing
    # ------------------------------------------------------------------
    def _iter_realtime(self):
        """
        For live sources, read continuously and yield frames only
        when enough wall-clock time has elapsed since the last sample.
        """
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            now = time.monotonic()
            elapsed_ms = (now - self._last_sample_time) * 1000

            if elapsed_ms < self.sample_interval_ms:
                continue  # skip — too soon

            self._last_sample_time = now
            self.buffer.append((frame, now))

            yield frame, now
