import time
from collections import deque


class RuntimeMetrics:
    def __init__(self, window=60):
        self._frame_times = deque(maxlen=window)
        self._latencies = deque(maxlen=window)
        self._instructions = deque(maxlen=window)

    def add_frame_time(self, seconds):
        self._frame_times.append(float(seconds))

    def add_latency(self, seconds):
        self._latencies.append(float(seconds))

    def add_instruction(self, instruction):
        self._instructions.append(instruction)

    def snapshot(self):
        fps = 0.0
        if self._frame_times:
            avg = sum(self._frame_times) / len(self._frame_times)
            fps = 1.0 / avg if avg > 1e-6 else 0.0

        latency = 0.0
        if self._latencies:
            latency = sum(self._latencies) / len(self._latencies)

        flip_rate = 0.0
        if len(self._instructions) >= 2:
            flips = sum(1 for i in range(1, len(self._instructions)) if self._instructions[i] != self._instructions[i - 1])
            flip_rate = flips / max(1, len(self._instructions) - 1)

        return {
            "fps": fps,
            "latency": latency,
            "flip_rate": flip_rate,
        }