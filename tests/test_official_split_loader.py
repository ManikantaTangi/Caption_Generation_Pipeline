import numpy as np
import pytest
from scipy.io import savemat

from hsi_caption.stage1_preprocessing.official_split_loader import OfficialSplitLoadError, OfficialSplitLoader
from hsi_caption.stage1_preprocessing.patch_generator import PatchGenerator


def _write_masks(tmp_path, labels, num_classes, n_per_class=4, seed=0):
    train_mask = np.zeros(labels.shape, dtype=np.uint8)
    test_mask = np.zeros(labels.shape, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for cls in range(1, num_classes):
        rows, cols = np.where(labels == cls)
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        n_train = min(n_per_class, len(idx) // 2)
        n_test = min(n_per_class, len(idx) - n_train)
        for i in idx[:n_train]:
            train_mask[rows[i], cols[i]] = cls
        for i in idx[n_train:n_train + n_test]:
            test_mask[rows[i], cols[i]] = cls
    train_path = str(tmp_path / "Train_fake.mat")
    test_path = str(tmp_path / "Test_fake.mat")
    savemat(train_path, {"FAKEtrain": train_mask})
    savemat(test_path, {"FAKEtest": test_mask})
    return train_path, test_path


class TestOfficialSplitLoader:
    def test_missing_file_raises(self, cfg, normalized_cube_and_params):
        norm_cube, _ = normalized_cube_and_params
        pg = PatchGenerator(cfg.patch.patch_size, stride=1, train_ratio=0.7, val_ratio=0.15,
                             test_ratio=0.15, random_seed=1)
        loader = OfficialSplitLoader(pg)
        with pytest.raises(OfficialSplitLoadError):
            loader.load(norm_cube, "/nonexistent/Train.mat", "/nonexistent/Test.mat")

    def test_shape_mismatch_raises(self, tmp_path, cfg, normalized_cube_and_params):
        norm_cube, _ = normalized_cube_and_params
        bad_shape_mask = np.zeros((3, 3), dtype=np.uint8)
        train_path = str(tmp_path / "Train.mat")
        test_path = str(tmp_path / "Test.mat")
        savemat(train_path, {"m": bad_shape_mask})
        savemat(test_path, {"m": bad_shape_mask})
        pg = PatchGenerator(cfg.patch.patch_size, stride=1, train_ratio=0.7, val_ratio=0.15,
                             test_ratio=0.15, random_seed=1)
        loader = OfficialSplitLoader(pg)
        with pytest.raises(OfficialSplitLoadError):
            loader.load(norm_cube, train_path, test_path)

    def test_load_produces_disjoint_nonoverlapping_splits(self, tmp_path, cfg, normalized_cube_and_params):
        norm_cube, params = normalized_cube_and_params
        train_path, test_path = _write_masks(tmp_path, norm_cube.labels, cfg.dataset.num_classes)
        pg = PatchGenerator(cfg.patch.patch_size, stride=1, train_ratio=0.7, val_ratio=0.15,
                             test_ratio=0.15, random_seed=1)
        loader = OfficialSplitLoader(pg, val_fraction=0.25)
        ds = loader.load(norm_cube, train_path, test_path, normalization_stats=params)

        assert len(ds.train) > 0 and len(ds.test) > 0
        train_val_coords = {(p.center_row, p.center_col) for p in ds.train + ds.val}
        test_coords = {(p.center_row, p.center_col) for p in ds.test}
        assert train_val_coords.isdisjoint(test_coords)

    def test_val_carved_out_of_train_only(self, tmp_path, cfg, normalized_cube_and_params):
        norm_cube, params = normalized_cube_and_params
        train_path, test_path = _write_masks(tmp_path, norm_cube.labels, cfg.dataset.num_classes, n_per_class=8)
        pg = PatchGenerator(cfg.patch.patch_size, stride=1, train_ratio=0.7, val_ratio=0.15,
                             test_ratio=0.15, random_seed=1)
        loader = OfficialSplitLoader(pg, val_fraction=0.5)
        ds = loader.load(norm_cube, train_path, test_path, normalization_stats=params)
        assert len(ds.val) > 0
