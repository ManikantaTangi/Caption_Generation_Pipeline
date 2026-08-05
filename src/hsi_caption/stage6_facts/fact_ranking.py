"""
Stage 6 - Fact Ranking
=========================
Purpose
    Deduplicate, score-filter, and rank all candidate triples (from
    TripletGenerator + RelationshipGenerator) into the final
    `StructuredFacts` -- Stage 6's contractual output, consumed
    directly by Stage 8 (verification operates per-triplet) and Stage 9
    (caption generation reads the top-ranked facts first).

Input
    `List[Triplet]` from TripletGenerator and RelationshipGenerator
    (this stage).

Algorithm
    1. Deduplicate on (subject, predicate, object), keeping the max score.
    2. Filter out anything below `min_fact_score`.
    3. Sort descending by score, truncate to `max_facts_per_patch`.
    A simple score-sort is preferred over a learned ranker here because
    every score already carries stage-specific calibrated meaning
    (a fraction, a rule confidence, an ontology coherence score) --
    learning a re-ranker would need labelled caption-quality data we
    don't have. Complexity: O(F log F).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from hsi_caption.datatypes import StructuredFacts, Triplet

logger = logging.getLogger(__name__)


class FactRanking:
    """Deduplicates, filters, and ranks candidate triples."""

    def __init__(self, min_fact_score: float, max_facts_per_patch: int) -> None:
        self.min_fact_score = min_fact_score
        self.max_facts_per_patch = max_facts_per_patch

    def rank(self, patch_id: str, triplets: List[Triplet]) -> StructuredFacts:
        best: Dict[Tuple[str, str, str], Triplet] = {}
        for t in triplets:
            key = (t.subject, t.predicate, t.object)
            if key not in best or t.score > best[key].score:
                best[key] = t

        filtered = [t for t in best.values() if t.score >= self.min_fact_score]
        ranked = sorted(filtered, key=lambda t: t.score, reverse=True)[: self.max_facts_per_patch]

        logger.debug("Ranked facts for %s: %d candidates -> %d kept", patch_id, len(triplets), len(ranked))
        return StructuredFacts(patch_id=patch_id, triplets=list(best.values()), ranked_facts=ranked)
