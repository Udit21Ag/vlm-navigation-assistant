"""
Layer 6 — Temporal Caption Generation with Anti-Flicker

Produces descriptive, motion-aware scene descriptions AND navigation
instructions.  Groups objects by label to avoid repetition.
Smooths the navigation component over a short window to prevent
rapid flickering.
"""

from collections import deque, Counter, defaultdict


class TemporalCaptionGenerator:
    """Generates smoothed, temporal-aware captions."""

    def __init__(self, smoothing_window=3, max_descriptions=3):
        """
        Args:
            smoothing_window: Number of recent instructions used for
                              majority-vote anti-flicker.
            max_descriptions: Maximum number of object descriptions
                              included in the caption.
        """
        self._recent_instructions = deque(maxlen=smoothing_window)
        self._max_descriptions = max_descriptions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, temporal_objects, instruction, urgency="info"):
        """
        Build a full caption from the temporal scene graph and navigation
        instruction, then smooth the instruction part.

        Args:
            temporal_objects: list of TemporalObject dicts from
                              TemporalReasoner.update().
            instruction: raw instruction string from NavigationPlanner.
            urgency: "critical" | "warning" | "info"

        Returns:
            (smoothed_instruction: str, full_caption: str)
        """
        # --- Scene descriptions (grouped by label + zone) ---
        scene_parts = self._describe_scene(temporal_objects)

        # --- Anti-flicker smoothing on the navigation instruction ---
        smoothed = self._smooth(instruction, urgency)

        # --- Combine: scene description + navigation instruction ---
        parts = scene_parts[:self._max_descriptions]
        parts.append(smoothed)
        full_caption = " ".join(parts)

        return smoothed, full_caption

    # ------------------------------------------------------------------
    # Scene description — groups multiple same-label objects
    # ------------------------------------------------------------------
    def _describe_scene(self, temporal_objects):
        """
        Return a list of descriptive phrases for detected objects.
        Groups objects of the same label in the same zone to avoid
        repetition like 'Car ahead. Car ahead. Car ahead.'
        """
        if not temporal_objects:
            return []

        # Group by (label, zone) → pick the most relevant motion/distance
        groups = defaultdict(list)
        for obj in temporal_objects:
            key = (obj["label"], obj.get("zone", "center"))
            groups[key].append(obj)

        descriptions = []
        for (label, zone), members in groups.items():
            # Pick the closest / most urgent member for description
            best = min(
                members,
                key=lambda o: _DIST_ORDER.get(o.get("distance", "far"), 3),
            )
            motion = best.get("motion", "stationary")
            distance = best.get("distance", "far")
            count = len(members)

            phrase = self._build_phrase(label, motion, zone, distance, count)
            if phrase:
                descriptions.append(phrase)

        # Sort by distance urgency (closest first)
        descriptions.sort(
            key=lambda p: next(
                (i for i, d in enumerate(_DIST_NAMES) if d in p.lower()),
                99,
            )
        )

        return descriptions

    @staticmethod
    def _build_phrase(label, motion, zone, distance, count=1):
        """Build a descriptive, motion-aware phrase for detected objects."""
        # Direction word
        if zone == "center":
            dir_word = "ahead"
        elif zone in ("left", "far left"):
            dir_word = "on the left"
        else:
            dir_word = "on the right"

        # Pluralise if needed
        if count > 1:
            name = f"{count} {label}s"
        else:
            name = label.capitalize()

        if motion == "approaching":
            return f"{name} approaching {distance} {dir_word}."
        elif motion == "crossing":
            return f"{name} crossing {dir_word}."
        elif motion == "receding":
            return f"{name} moving away {dir_word}."
        else:
            return f"{name} {distance} {dir_word}."

    # ------------------------------------------------------------------
    # Anti-flicker smoothing (majority vote on navigation instruction)
    # ------------------------------------------------------------------
    def _smooth(self, instruction, urgency):
        """
        Push the latest instruction into the window and return the
        majority-voted instruction.  Critical instructions bypass
        the smoothing entirely.
        """
        if urgency == "critical":
            self._recent_instructions.clear()
            self._recent_instructions.append(instruction)
            return instruction

        self._recent_instructions.append(instruction)

        counts = Counter(self._recent_instructions)
        winner, _ = counts.most_common(1)[0]
        return winner


# Distance ordering for sorting (closer = lower number = first)
_DIST_ORDER = {
    "very close": 0,
    "near": 1,
    "moderate distance": 2,
    "far": 3,
}
_DIST_NAMES = ["very close", "near", "moderate distance", "far"]
