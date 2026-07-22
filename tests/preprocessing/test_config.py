"""
Unit Tests: Configuration Manager
"""

"""
Unit Tests for Configuration Loader
"""

import unittest

from src.utils.config import config_loader


class TestConfigurationManager(unittest.TestCase):

    def test_load_paths(self):
        config = config_loader.load("paths.yaml")
        self.assertIsInstance(config, dict)

    def test_load_dataset(self):
        config = config_loader.load("dataset.yaml")
        self.assertIsInstance(config, dict)

    def test_load_model(self):
        config = config_loader.load("model.yaml")
        self.assertIsInstance(config, dict)


if __name__ == "__main__":
    unittest.main()