"""
Global configuration management for the Knowledge-Guided Hyperspectral
Image Caption Generation Pipeline.

Design rationale
-----------------
A single, strongly-typed `PipelineConfig` dataclass (composed of one
sub-dataclass per stage) is loaded once from YAML and threaded through
every stage via dependency injection. This gives us:

* Single source of truth (no magic numbers scattered across modules).
* Fail-fast validation (bad config raises at startup, not mid-run).
* Reproducibility (the exact config used for a run can be dumped
  alongside results for IEEE experiment logging).
"""
from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when configuration is missing, malformed, or inconsistent."""


@dataclass
class DatasetConfig:
    name: str = "WHU-Hi-LongKou"
    root_dir: str = "data/synthetic"
    image_key: str = "cube"
    label_key: str = "labels"
    num_bands: int = 270
    num_classes: int = 10
    class_names: List[str] = field(default_factory=lambda: [
        "Background", "Corn", "Cotton", "Sesame", "Broad-leaf soybean",
        "Narrow-leaf soybean", "Rice", "Water", "Roads and houses", "Mixed weed",
    ])
    wavelength_range_nm: List[float] = field(default_factory=lambda: [400.0, 1000.0])


@dataclass
class PatchConfig:
    patch_size: int = 15
    stride: int = 5
    normalize: str = "minmax"  # {"minmax", "zscore"}
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    use_official_split: bool = False
    official_split_dir: str = "data/WHU-Hi-LongKou/Training samples and test samples"
    official_split_size: int = 200  # one of {25,50,100,150,200,250,300} per WHU-Hi's file naming
    official_split_prefix: str = "LK"  # "LK" for LongKou, "HC" for HanChuan, "HH" for HongHu
    official_val_fraction: float = 0.1


@dataclass
class EncoderConfig:
    spectral_embed_dim: int = 64
    spatial_embed_dim: int = 64
    metadata_embed_dim: int = 16
    fused_embed_dim: int = 128
    spectral_conv_kernels: List[int] = field(default_factory=lambda: [7, 5, 3])
    spatial_patch_tokens: int = 9
    fusion_strategy: str = "attention"  # {"concat", "sum", "attention"}
    dropout: float = 0.1
    mc_dropout_passes: int = 30


@dataclass
class KnowledgeConfig:
    spectral_library_path: str = "configs/spectral_library.yaml"
    similarity_metric: str = "spectral_angle_mapper"  # {"sam", "cosine", "euclidean"}
    top_k_materials: int = 3
    kg_similarity_threshold: float = 0.75


@dataclass
class ReasoningConfig:
    rules_path: str = "configs/rules.yaml"
    max_inference_hops: int = 3
    contradiction_penalty: float = 0.5


@dataclass
class OntologyConfig:
    ontology_path: str = "configs/ontology.yaml"
    match_threshold: float = 0.6
    refine_iterations: int = 2


@dataclass
class FactConfig:
    max_facts_per_patch: int = 8
    min_fact_score: float = 0.3
    relation_confidence_floor: float = 0.2


@dataclass
class UncertaintyConfig:
    mc_dropout_passes: int = 30
    temperature: float = 1.5
    calibration_bins: int = 10
    entropy_normalize: bool = True


@dataclass
class VerificationConfig:
    verification_threshold: float = 0.55
    semantic_similarity_weight: float = 0.20
    kg_weight: float = 0.20
    ontology_weight: float = 0.15
    rule_weight: float = 0.15
    classifier_weight: float = 0.30


@dataclass
class CaptionConfig:
    template_bank_path: str = "configs/caption_templates.yaml"
    confidence_bands: Dict[str, List[float]] = field(default_factory=lambda: {
        "high": [0.75, 1.01],
        "medium": [0.45, 0.75],
        "low": [0.0, 0.45],
    })
    max_caption_length: int = 60


@dataclass
class PipelineConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    patch: PatchConfig = field(default_factory=PatchConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    ontology: OntologyConfig = field(default_factory=OntologyConfig)
    fact: FactConfig = field(default_factory=FactConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    caption: CaptionConfig = field(default_factory=CaptionConfig)
    log_level: str = "INFO"
    output_dir: str = "outputs"

    @staticmethod
    def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = PipelineConfig._merge(base[key], value)
            else:
                base[key] = value
        return base

    @classmethod
    def from_yaml(cls, path: Optional[str]) -> "PipelineConfig":
        """Load configuration from YAML, falling back to defaults for
        anything unspecified. Raises ConfigError on malformed YAML."""
        default = dataclasses.asdict(cls())
        if path is None:
            merged = default
        else:
            if not os.path.exists(path):
                logger.warning("Config file %s not found, using defaults.", path)
                merged = default
            else:
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        user_cfg = yaml.safe_load(fh) or {}
                except yaml.YAMLError as exc:
                    raise ConfigError(f"Malformed YAML in {path}: {exc}") from exc
                merged = cls._merge(default, user_cfg)
        return cls._from_dict(merged)

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        kwargs = {}
        for f in dataclasses.fields(cls):
            sub = d.get(f.name)
            if dataclasses.is_dataclass(f.default_factory() if callable(f.default_factory) else None):
                sub_cls = type(f.default_factory())
                kwargs[f.name] = sub_cls(**sub) if isinstance(sub, dict) else f.default_factory()
            else:
                kwargs[f.name] = sub if sub is not None else f.default
        return cls(**kwargs)

    def validate(self) -> None:
        """Fail-fast sanity checks across stage configs."""
        if self.patch.patch_size <= 0:
            raise ConfigError("patch.patch_size must be positive.")
        ratios = (self.patch.train_ratio, self.patch.val_ratio, self.patch.test_ratio)
        if not abs(sum(ratios) - 1.0) < 1e-6:
            raise ConfigError(f"train/val/test ratios must sum to 1.0, got {sum(ratios)}")
        if self.dataset.num_classes != len(self.dataset.class_names):
            raise ConfigError("num_classes must match length of class_names.")
        weights = (
            self.verification.semantic_similarity_weight
            + self.verification.kg_weight
            + self.verification.ontology_weight
            + self.verification.rule_weight
            + self.verification.classifier_weight
        )
        if not abs(weights - 1.0) < 1e-6:
            raise ConfigError(f"verification weights must sum to 1.0, got {weights}")
        logger.info("Configuration validated successfully.")


def configure_logging(level: str = "INFO") -> None:
    """Central logging configuration used by every stage/module."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
