"""
Pipeline Orchestrator
========================
Wires all 9 stages into a single end-to-end callable. This module is
the concrete proof that "output of stage N == input of stage N+1":
every stage facade below consumes only objects returned by the
previous stage (plus the shared, validated `PipelineConfig`).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from hsi_caption.config import PipelineConfig
from hsi_caption.datatypes import CaptionResult, HSICube, Patch
from hsi_caption.stage1_preprocessing.config_manager import ConfigManager
from hsi_caption.stage1_preprocessing.dataset_discovery import DatasetDiscovery
from hsi_caption.stage1_preprocessing.dataset_inspector import DatasetInspector, InspectionSummary
from hsi_caption.stage1_preprocessing.dataset_loader import DatasetLoader, SyntheticWHUHiGenerator
from hsi_caption.stage1_preprocessing.dataset_normalization import DatasetNormalizer
from hsi_caption.stage1_preprocessing.dataset_statistics import DatasetStatisticsComputer
from hsi_caption.stage1_preprocessing.dataset_validator import DatasetValidator
from hsi_caption.stage1_preprocessing.official_split_loader import OfficialSplitLoader
from hsi_caption.stage1_preprocessing.patch_generator import PatchGenerator
from hsi_caption.stage2_multi_encoder.multi_encoder import MultiEncoder
from hsi_caption.stage2_multi_encoder.torch_multi_encoder import TorchMultiEncoder
from hsi_caption.stage3_knowledge_engine.knowledge_engine import SpectralKnowledgeEngine
from hsi_caption.stage4_reasoning.reasoning_engine import ReasoningEngine
from hsi_caption.stage5_ontology.ontology_engine import OntologyEngine
from hsi_caption.stage6_facts.structured_fact_generator import StructuredFactGenerator
from hsi_caption.stage7_uncertainty.uncertainty_engine import UncertaintyEngine
from hsi_caption.stage7_uncertainty.torch_uncertainty_engine import TorchUncertaintyEngine
from hsi_caption.stage8_verification.fact_verification_engine import FactVerificationEngine
from hsi_caption.stage9_caption.caption_engine import CaptionEngine

logger = logging.getLogger(__name__)


@dataclass
class PatchPipelineResult:
    """Everything the pipeline produced for one patch, across all 9 stages."""
    patch_id: str
    true_class: str
    caption_result: CaptionResult
    verification_score: float
    material_top1: str
    predicted_material: str


class HSICaptionPipeline:
    """End-to-end orchestrator for the 9-stage captioning pipeline."""

    def __init__(self, config_path: Optional[str] = "configs/config.yaml") -> None:
        self.config: PipelineConfig = ConfigManager(config_path).load()
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Stage 1 sub-modules that are reusable across cubes
        self.dataset_discovery = DatasetDiscovery(self.config.dataset.root_dir)
        self.dataset_loader = DatasetLoader(
            self.config.dataset.class_names, self.config.dataset.num_bands,
            tuple(self.config.dataset.wavelength_range_nm),
        )
        self.synthetic_generator = SyntheticWHUHiGenerator(
            self.config.knowledge.spectral_library_path, self.config.dataset.class_names,
            self.config.dataset.num_bands, tuple(self.config.dataset.wavelength_range_nm),
            seed=self.config.patch.random_seed,
        )
        self.validator = DatasetValidator(self.config.dataset.num_bands, self.config.dataset.num_classes)
        self.inspector = DatasetInspector()
        self.stats_computer = DatasetStatisticsComputer()
        self.normalizer = DatasetNormalizer(self.config.patch.normalize)
        self.patch_generator = PatchGenerator(
            self.config.patch.patch_size, self.config.patch.stride,
            self.config.patch.train_ratio, self.config.patch.val_ratio, self.config.patch.test_ratio,
            self.config.patch.random_seed,
        )

        # Stage 2-9 engines are built lazily in `prepare()` once we know band count / semantic dim
        self.multi_encoder: Optional[MultiEncoder] = None
        self.knowledge_engine: Optional[SpectralKnowledgeEngine] = None
        self.reasoning_engine: Optional[ReasoningEngine] = None
        self.ontology_engine: Optional[OntologyEngine] = None
        self.fact_generator: Optional[StructuredFactGenerator] = None
        self.uncertainty_engine: Optional[UncertaintyEngine] = None
        self.verification_engine: Optional[FactVerificationEngine] = None
        self.caption_engine: Optional[CaptionEngine] = None
        self._last_patch_dataset = None

    # ------------------------------------------------------------------ #
    # Stage 1
    # ------------------------------------------------------------------ #
    def load_cube(self) -> HSICube:
        discovered = self.dataset_discovery.discover()
        if discovered.use_synthetic:
            cube = self.synthetic_generator.generate()
        else:
            label_path = discovered.label_files[0] if discovered.label_files else None
            cube = self.dataset_loader.load(discovered.image_files[0], label_path)
        self.validator.validate_or_raise(cube)
        return cube

    def preprocess(self, cube: HSICube):
        """Runs the remaining Stage-1 modules: inspect -> stats -> normalize -> patch/split.

        Uses WHU-Hi's official Train<N>/Test<N> mask-based split when
        `patch.use_official_split` is enabled *and* the matching mask
        files actually exist on disk (e.g. running against the synthetic
        fallback cube has no such files) -- otherwise falls back to
        PatchGenerator's own stratified random split so the pipeline
        remains runnable in every mode.
        """
        summary: InspectionSummary = self.inspector.inspect(cube)
        stats = self.stats_computer.compute(cube)
        norm_cube, norm_params = self.normalizer.fit_transform(cube, stats)

        train_path, test_path = self._official_split_paths()
        if self.config.patch.use_official_split and train_path and os.path.exists(train_path) \
                and os.path.exists(test_path):
            logger.info("Using official WHU-Hi split: %s / %s", train_path, test_path)
            loader = OfficialSplitLoader(self.patch_generator, self.config.patch.official_val_fraction)
            patch_dataset = loader.load(norm_cube, train_path, test_path, normalization_stats=norm_params)
        else:
            if self.config.patch.use_official_split:
                logger.warning("use_official_split=True but mask files not found at %s / %s; "
                                "falling back to random stratified split.", train_path, test_path)
            patch_dataset = self.patch_generator.generate(norm_cube, normalization_stats=norm_params)

        self._last_patch_dataset = patch_dataset
        return summary, stats, norm_cube, norm_params, patch_dataset

    def _official_split_paths(self) -> tuple:
        cfg = self.config.patch
        n = cfg.official_split_size
        prefix = cfg.official_split_prefix
        train_path = os.path.join(cfg.official_split_dir, f"Train{n}.mat")
        test_path = os.path.join(cfg.official_split_dir, f"Test{n}.mat")
        return train_path, test_path

    # ------------------------------------------------------------------ #
    # Stage 2-9 engine construction (depends on Stage-1 outputs)
    # ------------------------------------------------------------------ #
    def build_engines(self, wavelengths_nm: np.ndarray, norm_params: dict,
                       encoder_weights_path: Optional[str] = None,
                       uncertainty_weights_path: Optional[str] = None,
                       device: str = "cpu") -> None:
        """Builds Stage 2-9 engines.

        If `encoder_weights_path` / `uncertainty_weights_path` are given
        (produced by `demos/train_encoder.py` / `demos/train_uncertainty_head.py`),
        the pipeline uses the *trained* PyTorch backends
        (`TorchMultiEncoder` / `TorchUncertaintyEngine`) for those stages
        instead of the architecturally-identical-but-untrained numpy
        reference implementations. Everything else (Stages 3-6, 8, 9) is
        unchanged either way -- only the learned components differ.
        """
        cfg = self.config
        if encoder_weights_path:
            self.multi_encoder = TorchMultiEncoder(encoder_weights_path, device=device)
        else:
            self.multi_encoder = MultiEncoder(cfg.dataset.num_bands, cfg.encoder, cfg.patch.random_seed)

        fit_patches = self._last_patch_dataset.train if self._last_patch_dataset is not None else None
        self.knowledge_engine = SpectralKnowledgeEngine(
            wavelengths_nm, cfg.knowledge, normalization_strategy=cfg.patch.normalize,
            normalization_params=norm_params, fit_patches=fit_patches, fit_class_names=cfg.dataset.class_names,
        )
        self.reasoning_engine = ReasoningEngine(cfg.reasoning.rules_path, cfg.reasoning.contradiction_penalty)
        self.ontology_engine = OntologyEngine(cfg.ontology)
        self.fact_generator = StructuredFactGenerator(cfg.fact)

        # semantic_vector length = fused_embed_dim + top_k + n_categories + n_materials
        n_categories = len({e.category for e in self.knowledge_engine.library.entries.values()})
        n_materials = len(self.knowledge_engine.library.names())
        semantic_dim = cfg.encoder.fused_embed_dim + cfg.knowledge.top_k_materials + n_categories + n_materials

        if uncertainty_weights_path:
            self.uncertainty_engine = TorchUncertaintyEngine(
                uncertainty_weights_path, cfg.uncertainty, device=device,
            )
        else:
            self.uncertainty_engine = UncertaintyEngine(
                semantic_dim, cfg.dataset.num_classes, cfg.uncertainty, cfg.encoder.dropout, cfg.patch.random_seed,
            )
        self.verification_engine = FactVerificationEngine(
            self.knowledge_engine.kg, self.ontology_engine.ontology, cfg.verification, cfg.dataset.class_names,
        )
        self.caption_engine = CaptionEngine(cfg.caption, cfg.dataset.class_names)

    # ------------------------------------------------------------------ #
    # Per-patch Stage 2-9 execution
    # ------------------------------------------------------------------ #
    def run_patch(self, patch: Patch) -> PatchPipelineResult:
        assert self.multi_encoder is not None, "Call build_engines() before run_patch()."

        fused_embedding = self.multi_encoder.encode(patch)                                  # Stage 2
        knowledge_embedding = self.knowledge_engine.process(patch, fused_embedding)          # Stage 3
        reasoned = self.reasoning_engine.reason(
            patch.patch_id, knowledge_embedding.material_fractions, patch.label_patch)       # Stage 4
        ontology_rep = self.ontology_engine.process(
            patch.patch_id, knowledge_embedding.material_fractions, reasoned)                # Stage 5
        structured_facts = self.fact_generator.process(
            patch.patch_id, knowledge_embedding.material_fractions, reasoned, ontology_rep)  # Stage 6
        uncertainty = self.uncertainty_engine.process(
            patch.patch_id, knowledge_embedding.semantic_vector)                             # Stage 7
        verified_facts = self.verification_engine.process(
            structured_facts, knowledge_embedding, reasoned, ontology_rep, uncertainty)               # Stage 8
        caption_result = self.caption_engine.process(
            patch.patch_id, verified_facts, uncertainty, reasoned, knowledge_embedding)               # Stage 9

        true_class = self.config.dataset.class_names[patch.center_label]
        material_top1 = knowledge_embedding.material_matches[0].material_name if knowledge_embedding.material_matches else "n/a"

        material_triplets = sorted(
            [t for t in verified_facts.verified if t.predicate == "hasMaterial"], key=lambda t: t.score, reverse=True
        )
        if not material_triplets:
            material_triplets = sorted(verified_facts.verified, key=lambda t: t.score, reverse=True)
        predicted_material = material_triplets[0].object if material_triplets else "n/a"

        return PatchPipelineResult(
            patch_id=patch.patch_id, true_class=true_class, caption_result=caption_result,
            verification_score=verified_facts.verification_score, material_top1=material_top1,
            predicted_material=predicted_material,
        )

    def run(self, max_patches: Optional[int] = None, encoder_weights_path: Optional[str] = None,
            uncertainty_weights_path: Optional[str] = None, device: str = "cpu") -> List[PatchPipelineResult]:
        """Full end-to-end run: Stage 1 -> Stage 9 for every test patch.

        Pass `encoder_weights_path`/`uncertainty_weights_path` (from the
        training scripts in `demos/`) to run with trained models instead
        of the untrained numpy reference implementations.
        """
        cube = self.load_cube()
        summary, stats, norm_cube, norm_params, patch_dataset = self.preprocess(cube)
        self.build_engines(cube.wavelengths_nm, norm_params, encoder_weights_path,
                            uncertainty_weights_path, device)

        patches = patch_dataset.test if patch_dataset.test else patch_dataset.train
        if max_patches is not None:
            patches = patches[:max_patches]

        results = [self.run_patch(p) for p in patches]
        logger.info("Pipeline run complete: %d patches captioned.", len(results))
        return results
