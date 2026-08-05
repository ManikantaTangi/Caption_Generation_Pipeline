import numpy as np
import pytest

from hsi_caption.stage3_knowledge_engine.knowledge_engine import SpectralKnowledgeEngine
from hsi_caption.stage3_knowledge_engine.knowledge_graph import KnowledgeGraphBuilder
from hsi_caption.stage3_knowledge_engine.knowledge_retrieval import KnowledgeRetrieval, KnowledgeRetrievalError
from hsi_caption.stage3_knowledge_engine.material_identification import MaterialIdentifier
from hsi_caption.stage3_knowledge_engine.spectral_library import SpectralLibrary
from hsi_caption.stage2_multi_encoder.multi_encoder import MultiEncoder


class TestSpectralLibrary:
    def test_loads_expected_material_count(self, cfg, small_cube):
        lib = SpectralLibrary(cfg.knowledge.spectral_library_path, small_cube.wavelengths_nm)
        assert len(lib.names()) == cfg.dataset.num_classes

    def test_matrix_shape(self, cfg, small_cube):
        lib = SpectralLibrary(cfg.knowledge.spectral_library_path, small_cube.wavelengths_nm)
        mat, names = lib.matrix()
        assert mat.shape == (len(names), small_cube.num_bands)

    def test_normalization_changes_spectrum(self, cfg, small_cube):
        lib = SpectralLibrary(cfg.knowledge.spectral_library_path, small_cube.wavelengths_nm)
        before = lib.entries["Rice"].spectrum.copy()
        lib.apply_normalization("minmax", {"band_min": np.zeros_like(before), "band_max": np.ones_like(before) * 2})
        after = lib.entries["Rice"].spectrum
        assert not np.allclose(before, after)

    def test_refit_from_patches_uses_real_labelled_spectra(self, cfg, small_cube, patch_dataset):
        """Regression test for the real-data bug: rescaling the illustrative
        hand-authored library via the cube's min-max stats can collapse
        every material to near-indistinguishable shapes when the cube's raw
        dynamic range differs wildly from the library's assumed units.
        Fitting the library empirically from real labelled patches must
        avoid this and produce clearly distinguishable per-class spectra."""
        lib = SpectralLibrary(cfg.knowledge.spectral_library_path, small_cube.wavelengths_nm)
        lib.refit_from_patches(patch_dataset.train, cfg.dataset.class_names)

        present_classes = {cfg.dataset.class_names[p.center_label] for p in patch_dataset.train}
        for name in present_classes:
            if name in lib.entries:
                assert lib.entries[name].spectrum.shape == (small_cube.num_bands,)

        # spectra for two different, well-represented classes should not be identical
        names = list(present_classes)
        if len(names) >= 2:
            s0, s1 = lib.entries[names[0]].spectrum, lib.entries[names[1]].spectrum
            assert not np.allclose(s0, s1)

    def test_refit_keeps_fallback_for_absent_classes(self, cfg, small_cube, patch_dataset):
        lib = SpectralLibrary(cfg.knowledge.spectral_library_path, small_cube.wavelengths_nm)
        original_background = lib.entries["Background"].spectrum.copy()
        # Background is never a training-patch label under WHU-Hi's official split convention
        train_no_background = [p for p in patch_dataset.train
                                if cfg.dataset.class_names[p.center_label] != "Background"]
        lib.refit_from_patches(train_no_background, cfg.dataset.class_names)
        assert np.allclose(lib.entries["Background"].spectrum, original_background)


class TestKnowledgeRetrieval:
    def test_unsupported_metric_raises(self, cfg, small_cube):
        lib = SpectralLibrary(cfg.knowledge.spectral_library_path, small_cube.wavelengths_nm)
        with pytest.raises(KnowledgeRetrievalError):
            KnowledgeRetrieval(lib, "unsupported", top_k=3)

    def test_retrieve_returns_top_k_sorted(self, cfg, small_cube):
        lib = SpectralLibrary(cfg.knowledge.spectral_library_path, small_cube.wavelengths_nm)
        retrieval = KnowledgeRetrieval(lib, "spectral_angle_mapper", top_k=3)
        matches = retrieval.retrieve(lib.entries["Water"].spectrum)
        assert len(matches) == 3
        assert matches[0].similarity >= matches[1].similarity >= matches[2].similarity
        assert matches[0].material_name == "Water"  # exact match to itself should rank first

    @pytest.mark.parametrize("metric", ["spectral_angle_mapper", "cosine", "euclidean"])
    def test_all_metrics_run(self, cfg, small_cube, metric):
        lib = SpectralLibrary(cfg.knowledge.spectral_library_path, small_cube.wavelengths_nm)
        retrieval = KnowledgeRetrieval(lib, metric, top_k=2)
        matches = retrieval.retrieve(lib.entries["Corn"].spectrum)
        assert len(matches) == 2


class TestMaterialIdentifier:
    def test_fractions_sum_to_one(self, cfg, small_cube, sample_patch):
        lib = SpectralLibrary(cfg.knowledge.spectral_library_path, small_cube.wavelengths_nm)
        retrieval = KnowledgeRetrieval(lib, "spectral_angle_mapper", top_k=1)
        identifier = MaterialIdentifier(retrieval)
        fractions = identifier.identify(sample_patch.cube_data)
        assert abs(sum(fractions.values()) - 1.0) < 1e-6

    def test_all_materials_present_as_keys(self, cfg, small_cube, sample_patch):
        lib = SpectralLibrary(cfg.knowledge.spectral_library_path, small_cube.wavelengths_nm)
        retrieval = KnowledgeRetrieval(lib, "spectral_angle_mapper", top_k=1)
        identifier = MaterialIdentifier(retrieval)
        fractions = identifier.identify(sample_patch.cube_data)
        assert set(fractions.keys()) == set(lib.names())


class TestKnowledgeGraphBuilder:
    def test_graph_has_material_and_category_nodes(self, cfg, small_cube):
        lib = SpectralLibrary(cfg.knowledge.spectral_library_path, small_cube.wavelengths_nm)
        kg = KnowledgeGraphBuilder(lib, cfg.knowledge.kg_similarity_threshold)
        assert any(n.startswith("material::") for n in kg.graph.nodes)
        assert any(n.startswith("category::") for n in kg.graph.nodes)

    def test_related_materials_returns_list(self, cfg, small_cube):
        lib = SpectralLibrary(cfg.knowledge.spectral_library_path, small_cube.wavelengths_nm)
        kg = KnowledgeGraphBuilder(lib, cfg.knowledge.kg_similarity_threshold)
        related = kg.related_materials("Rice")
        assert isinstance(related, list)


class TestSpectralKnowledgeEngineFacade:
    def test_process_returns_valid_knowledge_embedding(self, cfg, small_cube, sample_patch,
                                                          normalized_cube_and_params):
        _, norm_params = normalized_cube_and_params
        ske = SpectralKnowledgeEngine(small_cube.wavelengths_nm, cfg.knowledge,
                                       normalization_strategy=cfg.patch.normalize,
                                       normalization_params=norm_params)
        encoder = MultiEncoder(cfg.dataset.num_bands, cfg.encoder, cfg.patch.random_seed)
        fused = encoder.encode(sample_patch)
        ke = ske.process(sample_patch, fused)
        assert len(ke.material_matches) == cfg.knowledge.top_k_materials
        assert ke.semantic_vector.ndim == 1
