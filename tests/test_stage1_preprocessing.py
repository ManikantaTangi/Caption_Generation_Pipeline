import numpy as np
import pytest

from hsi_caption.datatypes import HSICube
from hsi_caption.stage1_preprocessing.dataset_discovery import DatasetDiscovery
from hsi_caption.stage1_preprocessing.dataset_normalization import DatasetNormalizer, NormalizationError
from hsi_caption.stage1_preprocessing.dataset_statistics import DatasetStatisticsComputer
from hsi_caption.stage1_preprocessing.dataset_validator import DatasetValidationError, DatasetValidator
from hsi_caption.stage1_preprocessing.dataset_inspector import DatasetInspector
from hsi_caption.stage1_preprocessing.patch_generator import PatchGenerationError, PatchGenerator


class TestDatasetDiscovery:
    def test_missing_dir_falls_back_to_synthetic(self):
        result = DatasetDiscovery("/nonexistent/path/xyz").discover()
        assert result.use_synthetic is True

    def test_empty_dir_falls_back_to_synthetic(self, tmp_path):
        result = DatasetDiscovery(str(tmp_path)).discover()
        assert result.use_synthetic is True

    def test_finds_image_and_label_files(self, tmp_path):
        (tmp_path / "whu_hi_image.npy").touch()
        (tmp_path / "whu_hi_gt.npy").touch()
        result = DatasetDiscovery(str(tmp_path)).discover()
        assert result.use_synthetic is False
        assert len(result.image_files) == 1
        assert len(result.label_files) == 1


class TestDatasetValidator:
    def test_valid_cube_passes(self, small_cube, cfg):
        validator = DatasetValidator(cfg.dataset.num_bands, cfg.dataset.num_classes)
        report = validator.validate(small_cube)
        assert report.is_valid

    def test_band_mismatch_fails(self, cfg):
        bad_cube = HSICube(
            data=np.random.rand(10, 10, 5).astype(np.float32), labels=np.zeros((10, 10), dtype=np.int32),
            wavelengths_nm=np.linspace(400, 1000, 5), class_names=cfg.dataset.class_names,
        )
        validator = DatasetValidator(cfg.dataset.num_bands, cfg.dataset.num_classes)
        report = validator.validate(bad_cube)
        assert not report.is_valid
        assert any("Band count" in issue for issue in report.issues)

    def test_nan_detected(self, cfg):
        data = np.random.rand(5, 5, cfg.dataset.num_bands).astype(np.float32)
        data[0, 0, 0] = np.nan
        bad_cube = HSICube(
            data=data, labels=np.zeros((5, 5), dtype=np.int32),
            wavelengths_nm=np.linspace(400, 1000, cfg.dataset.num_bands), class_names=cfg.dataset.class_names,
        )
        validator = DatasetValidator(cfg.dataset.num_bands, cfg.dataset.num_classes)
        with pytest.raises(DatasetValidationError):
            validator.validate_or_raise(bad_cube)

    def test_out_of_range_label_detected(self, cfg):
        data = np.random.rand(5, 5, cfg.dataset.num_bands).astype(np.float32)
        labels = np.full((5, 5), cfg.dataset.num_classes + 1, dtype=np.int32)
        bad_cube = HSICube(
            data=data, labels=labels,
            wavelengths_nm=np.linspace(400, 1000, cfg.dataset.num_bands), class_names=cfg.dataset.class_names,
        )
        validator = DatasetValidator(cfg.dataset.num_bands, cfg.dataset.num_classes)
        report = validator.validate(bad_cube)
        assert not report.is_valid


class TestDatasetInspector:
    def test_inspect_returns_shape_and_distribution(self, small_cube):
        summary = DatasetInspector().inspect(small_cube)
        assert summary.height == small_cube.height
        assert summary.width == small_cube.width
        assert summary.num_bands == small_cube.num_bands
        assert abs(sum(summary.class_pixel_fraction.values()) - 1.0) < 1e-6

    def test_as_text_is_nonempty_string(self, small_cube):
        text = DatasetInspector().inspect(small_cube).as_text()
        assert isinstance(text, str) and len(text) > 0


class TestDatasetStatistics:
    def test_band_stats_shapes(self, small_cube):
        stats = DatasetStatisticsComputer().compute(small_cube)
        assert stats.band_mean.shape == (small_cube.num_bands,)
        assert stats.band_std.shape == (small_cube.num_bands,)
        assert np.all(stats.band_std > 0)

    def test_imbalance_ratio_is_positive(self, small_cube):
        stats = DatasetStatisticsComputer().compute(small_cube)
        assert stats.class_imbalance_ratio >= 1.0


class TestDatasetNormalizer:
    def test_minmax_produces_unit_range(self, small_cube):
        stats = DatasetStatisticsComputer().compute(small_cube)
        norm_cube, params = DatasetNormalizer("minmax").fit_transform(small_cube, stats)
        assert norm_cube.data.min() >= -1e-6
        assert norm_cube.data.max() <= 1.0 + 1e-6

    def test_zscore_produces_near_zero_mean(self, small_cube):
        stats = DatasetStatisticsComputer().compute(small_cube)
        norm_cube, params = DatasetNormalizer("zscore").fit_transform(small_cube, stats)
        assert abs(float(norm_cube.data.mean())) < 0.5

    def test_unsupported_strategy_raises(self):
        with pytest.raises(NormalizationError):
            DatasetNormalizer("unsupported_strategy")


class TestPatchGenerator:
    def test_even_patch_size_rejected(self):
        with pytest.raises(PatchGenerationError):
            PatchGenerator(patch_size=14, stride=1, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    def test_generate_produces_correct_patch_shape(self, patch_dataset, cfg):
        patch = patch_dataset.train[0]
        expected = (cfg.patch.patch_size, cfg.patch.patch_size, cfg.dataset.num_bands)
        assert patch.cube_data.shape == expected

    def test_splits_are_disjoint(self, patch_dataset):
        train_ids = {p.patch_id for p in patch_dataset.train}
        val_ids = {p.patch_id for p in patch_dataset.val}
        test_ids = {p.patch_id for p in patch_dataset.test}
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_metadata_has_expected_keys(self, patch_dataset):
        patch = patch_dataset.train[0]
        assert set(patch.metadata.keys()) == {"row_norm", "col_norm", "local_class_purity"}
