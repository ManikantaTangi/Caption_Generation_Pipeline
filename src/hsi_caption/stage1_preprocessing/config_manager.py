"""
Stage 1 - Configuration Manager
================================
Purpose
    Single entry point that loads, validates, and exposes the pipeline's
    typed configuration to every downstream module. No other module reads
    YAML directly; they all receive a validated `PipelineConfig` through
    the constructor (dependency injection), which keeps the system
    testable and swap-in-configurable.

Role in pipeline
    Runs before anything else. Its output (a validated PipelineConfig)
    is threaded through all 9 stages.
"""
from __future__ import annotations

import logging
from typing import Optional

from hsi_caption.config import ConfigError, PipelineConfig, configure_logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Loads and validates the full pipeline configuration.

    Complexity: O(1) w.r.t. dataset size (config files are small/fixed).
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._config_path = config_path
        self._config: Optional[PipelineConfig] = None

    def load(self) -> PipelineConfig:
        """Load config from YAML (or defaults) and validate it.

        Returns
        -------
        PipelineConfig
            Fully populated, validated configuration object.

        Raises
        ------
        ConfigError
            If the YAML is malformed or fails cross-field validation.
        """
        try:
            cfg = PipelineConfig.from_yaml(self._config_path)
            cfg.validate()
        except ConfigError:
            logger.exception("Configuration failed validation.")
            raise
        configure_logging(cfg.log_level)
        logger.info("Loaded configuration from %s", self._config_path or "<defaults>")
        self._config = cfg
        return cfg

    @property
    def config(self) -> PipelineConfig:
        if self._config is None:
            raise RuntimeError("ConfigManager.load() must be called before accessing config.")
        return self._config
