"""
Stage 6 - Triplet Generator
==============================
Purpose
    Formalise each `CandidateFact` into the pipeline-wide `Triplet`
    contract (`subject, predicate, object, score`) used uniformly by
    Stage 8 (verification) and Stage 9 (caption generation). Separated
    from FactExtraction so the extraction *logic* (what counts as a
    fact) and the *representation* (how it's typed/scored) can evolve
    independently -- e.g. swapping in an LLM-based extractor later only
    requires it to emit `CandidateFact`s.

Input
    `List[CandidateFact]` from FactExtractor (this stage).

Algorithm
    Direct 1:1 mapping, `score` initialised to `raw_score` (later
    adjusted by FactRanking / RelationshipGenerator). O(F).
"""
from __future__ import annotations

import logging
from typing import List

from hsi_caption.datatypes import Triplet
from hsi_caption.stage6_facts.fact_extraction import CandidateFact

logger = logging.getLogger(__name__)


class TripletGenerator:
    """Converts candidate facts into formal Triplet objects."""

    def generate(self, candidates: List[CandidateFact]) -> List[Triplet]:
        return [Triplet(subject=c.subject, predicate=c.predicate, object=c.object, score=c.raw_score)
                for c in candidates]
