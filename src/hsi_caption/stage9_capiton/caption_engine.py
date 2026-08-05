"""
Stage 9 - Confidence-Aware Caption Generation (facade)
==========================================================
Wires TemplateSelector + CaptionGenerator + LanguageController +
CaptionRefiner + ExplanationGenerator into one callable consuming
Stage 3/4/7/8's outputs and producing the pipeline's final
`CaptionResult` -- the terminal artefact of the entire 9-stage system.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from hsi_caption.config import CaptionConfig
from hsi_caption.datatypes import CaptionResult, KnowledgeEmbedding, ReasonedKnowledge, UncertaintyEstimate, \
    VerifiedFacts
from hsi_caption.stage9_caption.caption_generator import CaptionGenerator
from hsi_caption.stage9_caption.caption_refiner import CaptionRefiner
from hsi_caption.stage9_caption.explanation_generator import ExplanationGenerator
from hsi_caption.stage9_caption.language_controller import LanguageController
from hsi_caption.stage9_caption.template_selector import TemplateSelector

logger = logging.getLogger(__name__)


class CaptionEngine:
    """Facade combining all five Stage 9 modules."""

    def __init__(self, cfg: CaptionConfig, class_names: Optional[List[str]] = None) -> None:
        self.selector = TemplateSelector(cfg.template_bank_path, cfg.confidence_bands)
        self.generator = CaptionGenerator()
        self.controller = LanguageController()
        self.refiner = CaptionRefiner(cfg.max_caption_length)
        self.explainer = ExplanationGenerator(class_names)

    def process(self, patch_id: str, verified_facts: VerifiedFacts, uncertainty: UncertaintyEstimate,
                reasoned: ReasonedKnowledge, knowledge_embedding: Optional[KnowledgeEmbedding] = None) -> CaptionResult:
        band, template = self.selector.select(uncertainty.confidence_score)
        raw_caption = self.generator.generate(template, verified_facts.verified, uncertainty.confidence_score)
        hedged_caption = self.controller.control(raw_caption, band)
        final_caption = self.refiner.refine(hedged_caption)
        explanation = self.explainer.generate(verified_facts, uncertainty, reasoned, knowledge_embedding)

        return CaptionResult(
            patch_id=patch_id, caption=final_caption, confidence_score=uncertainty.confidence_score,
            confidence_band=band, explanation=explanation,
        )
