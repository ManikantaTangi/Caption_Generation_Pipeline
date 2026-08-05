"""
Stage 9 - Caption Refiner
============================
Purpose
    Final surface-level cleanup pass: enforce `max_caption_length` (word
    budget), normalise whitespace/punctuation left by template slot
    filling, and guarantee the caption ends with terminal punctuation --
    the last checkpoint before text is handed to the user.

Input
    Hedged caption string (this stage, from LanguageController).

Algorithm
    Deterministic string post-processing: collapse whitespace, truncate
    to `max_caption_length` words (append ellipsis if truncated),
    capitalise the first character, ensure trailing '.'. O(len(caption)).
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


class CaptionRefiner:
    """Final cleanup and length-enforcement pass on a generated caption."""

    def __init__(self, max_caption_length: int) -> None:
        self.max_caption_length = max_caption_length

    def refine(self, caption: str) -> str:
        caption = re.sub(r"\s+", " ", caption).strip()
        words = caption.split(" ")
        truncated = len(words) > self.max_caption_length
        if truncated:
            caption = " ".join(words[: self.max_caption_length]).rstrip(",.;") + "..."
        if caption:
            caption = caption[0].upper() + caption[1:]
        if not truncated and not caption.endswith((".", "...", "!", "?")):
            caption += "."
        return caption
