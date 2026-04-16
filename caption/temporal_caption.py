from collections import deque, Counter, defaultdict


class TemporalCaptionGenerator:
    """Generates smoothed, temporal-aware captions."""

    def __init__(self, smoothing_window=3, max_descriptions=2):
        """
        Args:
            smoothing_window: Number of recent instructions used for smoothing.
            max_descriptions: Maximum number of object descriptions.
        """
        self._recent_instructions = deque(maxlen=smoothing_window)
        self._max_descriptions = max_descriptions
        self._recent_phrases = deque(maxlen=4)

    # Public API
    def generate(self, temporal_objects, instruction, urgency="info"):
        """
        Build caption from temporal scene + navigation instruction.

        Returns:
            (smoothed_instruction: str, full_caption: str)
        """

        # --- Smooth instruction first ---
        smoothed = self._smooth(instruction, urgency)

        # --- Skip verbose description for passive navigation ---
        if "continue" in smoothed.lower():
            return smoothed, smoothed

        # --- Build scene description ---
        scene_parts = self._describe_scene(temporal_objects)

        # --- Limit verbosity ---
        parts = scene_parts[:self._max_descriptions]
        parts.append(smoothed)

        full_caption = " ".join(parts)
        self._recent_phrases.append(full_caption)

        return smoothed, full_caption

    # Scene description (grouped + prioritized)
    def _describe_scene(self, temporal_objects):
        if not temporal_objects:
            return []

        groups = defaultdict(list)

        for obj in temporal_objects:
            key = (obj["label"], obj.get("zone", "center"))
            groups[key].append(obj)

        descriptions = []

        for (label, zone), members in groups.items():

            # Prioritize:
            # 1. Distance (closer first)
            # 2. Approaching motion
            best = min(
                members,
                key=lambda o: (
                    _DIST_ORDER.get(o.get("distance", "far"), 3),
                    0 if o.get("motion") == "approaching" else 1
                )
            )

            motion = best.get("motion", "stationary")
            distance = best.get("distance", "far")
            count = len(members)
            ttc = best.get("ttc", float("inf"))

            phrase = self._build_phrase(label, motion, zone, distance, count, ttc)

            if phrase:
                descriptions.append(phrase)

        # Sort by urgency (closest first)
        descriptions.sort(
            key=lambda p: next(
                (i for i, d in enumerate(_DIST_NAMES) if d in p.lower()),
                99,
            )
        )

        return descriptions

    # Phrase builder (short, TTS-friendly)
    @staticmethod
    def _build_phrase(label, motion, zone, distance, count=1, ttc=None):

        # Direction
        if zone == "center":
            direction = "ahead"
        elif zone in ("left", "far left"):
            direction = "left"
        else:
            direction = "right"

        # Naming
        if count > 1:
            name = f"{count} {label}s"
        else:
            name = label

        # Motion-aware phrasing (shortened)
        if motion == "approaching":
            return f"{name} approaching {direction}."
        elif motion == "crossing":
            return f"{name} crossing {direction}."
        elif motion == "receding":
            return f"{name} moving away {direction}."
        else:
            return f"{name} {distance} {direction}."

    # ------------------------------------------------------------------
    # Anti-flicker smoothing
    # ------------------------------------------------------------------
    def _smooth(self, instruction, urgency):

        if urgency == "critical":
            self._recent_instructions.clear()
            self._recent_instructions.append(instruction)
            return instruction

        self._recent_instructions.append(instruction)

        counts = Counter(self._recent_instructions)
        winner, _ = counts.most_common(1)[0]

        # Normalize passive instructions
        if "continue" in winner.lower():
            return "Continue forward."

        return winner

# Distance ordering (lower = more urgent)
_DIST_ORDER = {
    "very close": 0,
    "near": 1,
    "moderate distance": 2,
    "far": 3,
}

_DIST_NAMES = ["very close", "near", "moderate distance", "far"]