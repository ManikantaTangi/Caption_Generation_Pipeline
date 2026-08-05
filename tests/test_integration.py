import os

from hsi_caption.pipeline import HSICaptionPipeline

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.test.yaml")


class TestFullPipelineIntegration:
    def test_end_to_end_run_produces_captions(self):
        pipeline = HSICaptionPipeline(CONFIG_PATH)
        results = pipeline.run(max_patches=5)
        assert len(results) == 5
        for r in results:
            assert isinstance(r.caption_result.caption, str) and len(r.caption_result.caption) > 0
            assert 0.0 <= r.caption_result.confidence_score <= 1.0
            assert r.caption_result.confidence_band in ("high", "medium", "low")
            assert len(r.caption_result.explanation) > 0

    def test_stage_output_types_match_next_stage_input(self):
        """Explicitly checks the 'output of stage N == input of stage N+1' contract."""
        pipeline = HSICaptionPipeline(CONFIG_PATH)
        cube = pipeline.load_cube()
        summary, stats, norm_cube, norm_params, patch_dataset = pipeline.preprocess(cube)
        pipeline.build_engines(cube.wavelengths_nm, norm_params)

        patch = patch_dataset.test[0] if patch_dataset.test else patch_dataset.train[0]

        fused = pipeline.multi_encoder.encode(patch)
        assert fused.fused_vector.shape[0] > 0

        ke = pipeline.knowledge_engine.process(patch, fused)
        assert ke.semantic_vector.shape[0] > fused.fused_vector.shape[0]  # semantic vector extends fused vector

        rk = pipeline.reasoning_engine.reason(patch.patch_id, ke.material_fractions, patch.label_patch)
        assert rk.patch_id == patch.patch_id

        osr = pipeline.ontology_engine.process(patch.patch_id, ke.material_fractions, rk)
        assert osr.patch_id == patch.patch_id

        sf = pipeline.fact_generator.process(patch.patch_id, ke.material_fractions, rk, osr)
        assert sf.patch_id == patch.patch_id

        ue = pipeline.uncertainty_engine.process(patch.patch_id, ke.semantic_vector)
        assert ue.patch_id == patch.patch_id

        vf = pipeline.verification_engine.process(sf, ke, rk, osr, ue)
        assert vf.patch_id == patch.patch_id

        cr = pipeline.caption_engine.process(patch.patch_id, vf, ue, rk, ke)
        assert cr.patch_id == patch.patch_id
        assert cr.caption
