"""Tests for E2E manifest schema validation.

Trace: ARCH-E2E-001, UD-E2E-MANIFEST
Intent: Validate E2E manifest schema enforcement for required fields, type checks, and skip semantics
Preconditions: e2e_harness manifest_loader is importable
Postconditions: Invalid manifests raise appropriate ValueError/TypeError; valid manifests pass validation
"""
import json
import os
import pytest
import warnings

# Try to import manifest_loader
try:
    from tests.e2e_harness.manifest_loader import load_manifest, _validate_manifest
except ImportError:
    pytest.skip("e2e_harness not available", allow_module_level=True)


class TestManifestValidation:
    """Test manifest schema validation."""

    def _write_manifest(self, tmp_path, data):
        path = os.path.join(str(tmp_path), "test.json")
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def test_missing_name_raises(self, tmp_path):
        """Manifest without 'name' should raise ValueError."""
        path = self._write_manifest(tmp_path, {"hf_id": "org/model", "family": "qwen"})
        with pytest.raises(ValueError, match="name"):
            _validate_manifest(json.load(open(path)), path)

    def test_missing_hf_id_raises_when_not_skipped(self, tmp_path):
        """Manifest without 'hf_id' (and no skip) should raise ValueError."""
        path = self._write_manifest(tmp_path, {"name": "test-model", "family": "qwen"})
        with pytest.raises(ValueError, match="hf_id"):
            _validate_manifest(json.load(open(path)), path)

    def test_missing_family_raises_when_not_skipped(self, tmp_path):
        """Manifest without 'family' (and no skip) should raise ValueError."""
        path = self._write_manifest(tmp_path, {"name": "test-model", "hf_id": "org/model"})
        with pytest.raises(ValueError, match="family"):
            _validate_manifest(json.load(open(path)), path)

    def test_skipped_manifest_allows_missing_hf_id(self, tmp_path):
        """Skipped manifests don't need hf_id or family."""
        data = {"name": "test-model", "skip": "not available"}
        _validate_manifest(data, "test.json")  # Should not raise

    def test_wrong_type_max_new_tokens_string(self, tmp_path):
        """max_new_tokens must be int, not string."""
        data = {"name": "test", "hf_id": "org/m", "family": "qwen", "max_new_tokens": "20"}
        with pytest.raises(TypeError, match="max_new_tokens"):
            _validate_manifest(data, "test.json")

    def test_wrong_type_max_new_tokens_float(self, tmp_path):
        """max_new_tokens must be int, not float."""
        data = {"name": "test", "hf_id": "org/m", "family": "qwen", "max_new_tokens": 20.5}
        with pytest.raises(TypeError, match="max_new_tokens"):
            _validate_manifest(data, "test.json")

    def test_wrong_type_max_cache_length(self, tmp_path):
        """max_cache_length must be int, not float."""
        data = {"name": "test", "hf_id": "org/m", "family": "qwen", "max_cache_length": 256.5}
        with pytest.raises(TypeError, match="max_cache_length"):
            _validate_manifest(data, "test.json")

    def test_unknown_runtime_strategy_warns(self, tmp_path):
        """Unknown runtime_strategy should emit a warning."""
        data = {
            "name": "test",
            "hf_id": "org/m",
            "family": "qwen",
            "runtime_strategy": "bogus_strategy",
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_manifest(data, "test.json")
            assert any("bogus_strategy" in str(warning.message) for warning in w)

    def test_known_runtime_strategy_no_warning(self, tmp_path):
        """Known runtime_strategy should not emit a warning."""
        data = {
            "name": "test",
            "hf_id": "org/m",
            "family": "qwen",
            "runtime_strategy": "decoder_kv_cache",
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_manifest(data, "test.json")
            strategy_warnings = [
                x for x in w if "runtime_strategy" in str(x.message)
            ]
            assert len(strategy_warnings) == 0

    def test_valid_manifest_passes(self, tmp_path):
        """A fully valid manifest should pass without errors."""
        data = {
            "name": "qwen3-test",
            "hf_id": "Qwen/Qwen3-0.6B",
            "family": "qwen",
            "bundle": "qwen3-test.trtfb",
            "runtime_strategy": "decoder_kv_cache",
            "max_cache_length": 256,
            "max_new_tokens": 20,
            "prompt": "Hello",
        }
        _validate_manifest(data, "test.json")  # Should not raise

    def test_model_id_accepted_as_hf_id_alternative(self, tmp_path):
        """model_id is accepted as an alternative to hf_id."""
        data = {
            "name": "test-model",
            "model_id": "org/model",
            "family": "qwen",
        }
        _validate_manifest(data, "test.json")  # Should not raise

    def test_load_manifest_calls_validation(self, tmp_path):
        """load_manifest should call _validate_manifest and raise on bad input."""
        path = self._write_manifest(tmp_path, {"hf_id": "org/model", "family": "qwen"})
        with pytest.raises(ValueError, match="name"):
            load_manifest(path)

    def test_gated_manifest_requires_auth_preflight(self, tmp_path):
        path = self._write_manifest(tmp_path, {
            "name": "gated-test",
            "hf_id": "org/gated-model",
            "family": "qwen",
            "runtime_strategy": "decoder_kv_cache",
            "gated": True,
        })
        case = load_manifest(path)
        matches = [
            req for req in case.preflight
            if req.kind == "hf_auth_token_present"
        ]
        assert len(matches) == 1
        assert matches[0].gating is True

    def test_remote_code_manifest_auth_preflight_is_diagnostic(self, tmp_path):
        path = self._write_manifest(tmp_path, {
            "name": "remote-code-test",
            "hf_id": "org/remote-code-model",
            "family": "eagle_vlm",
            "runtime_strategy": "embedding",
            "trust_remote_code": True,
        })
        case = load_manifest(path)
        matches = [
            req for req in case.preflight
            if req.kind == "hf_auth_token_present"
        ]
        assert len(matches) == 1
        assert matches[0].gating is False

    def test_bool_not_accepted_as_int(self, tmp_path):
        """Boolean values should not pass the int type check."""
        data = {
            "name": "test",
            "hf_id": "org/m",
            "family": "qwen",
            "max_new_tokens": True,
        }
        with pytest.raises(TypeError, match="max_new_tokens"):
            _validate_manifest(data, "test.json")

    def test_execution_profiles_must_be_object(self, tmp_path):
        data = {
            "name": "test",
            "hf_id": "org/m",
            "family": "qwen",
            "execution_profiles": "chronos",
        }
        with pytest.raises(TypeError, match="execution_profiles"):
            _validate_manifest(data, "test.json")

    def test_execution_profiles_reject_unknown_phase(self, tmp_path):
        data = {
            "name": "test",
            "hf_id": "org/m",
            "family": "qwen",
            "execution_profiles": {"build": "chronos", "verify": "chronos"},
        }
        with pytest.raises(ValueError, match="unsupported phase"):
            _validate_manifest(data, "test.json")

    def test_load_manifest_applies_family_default_execution_profiles(self, tmp_path):
        path = self._write_manifest(
            tmp_path,
            {
                "name": "chronos-case",
                "hf_id": "amazon/chronos-bolt-tiny",
                "family": "chronos_bolt",
                "runtime_strategy": "chronos_bolt_torchtrt",
                "reference_backend": "torch_reference",
            },
        )
        case = load_manifest(path)
        assert case.execution_profiles["build"] == "chronos"
        assert case.execution_profiles["runtime"] == "base"
        assert case.execution_profiles["reference"] == "chronos"

    def test_load_manifest_preserves_execution_profile_overrides(self, tmp_path):
        path = self._write_manifest(
            tmp_path,
            {
                "name": "chronos-case",
                "hf_id": "amazon/chronos-bolt-tiny",
                "family": "chronos_bolt",
                "runtime_strategy": "chronos_bolt_torchtrt",
                "reference_backend": "torch_reference",
                "execution_profiles": {"runtime": "custom-runtime"},
            },
        )
        case = load_manifest(path)
        assert case.execution_profiles["build"] == "chronos"
        assert case.execution_profiles["runtime"] == "custom-runtime"
        assert case.execution_profiles["reference"] == "chronos"

    def test_quantization_block_propagates_to_metadata(self, tmp_path):
        """Quantization manifests should preserve the generic quant block."""
        path = self._write_manifest(tmp_path, {
            "name": "qwen3-test-fp8",
            "hf_id": "Qwen/Qwen3-0.6B",
            "family": "qwen",
            "runtime_strategy": "decoder_kv_cache",
            "precision": "bf16",
            "quantization": {
                "format": "fp8",
                "scale_source": "precomputed",
                "scale_artifact": "scales/qwen3-fp8.json",
                "calibration_samples": 16,
            },
        })
        case = load_manifest(path)
        assert case.metadata["precision"] == "bf16"
        assert case.metadata["quantization"]["format"] == "fp8"
        assert case.metadata["quantization"]["scale_artifact"] == "scales/qwen3-fp8.json"

    def test_skip_comparison_populates_metadata(self, tmp_path):
        """skip_comparison should set skip_comparison_reason without setting skip_reason."""
        path = self._write_manifest(tmp_path, {
            "name": "rerank-test",
            "hf_id": "org/rerank",
            "family": "eagle_vlm",
            "runtime_strategy": "reranking",
            "skip_comparison": "reference shape mismatch",
        })
        case = load_manifest(path)
        assert case.metadata["skip_comparison_reason"] == "reference shape mismatch"
        # Partial skip must NOT set skip_reason (that would trigger full pytest.skip)
        assert "skip_reason" not in case.metadata

    def test_skip_comparison_does_not_exempt_required_fields(self, tmp_path):
        """skip_comparison still requires hf_id + family (unlike skip)."""
        path = self._write_manifest(tmp_path, {
            "name": "rerank-test",
            "skip_comparison": "reference shape mismatch",
        })
        with pytest.raises(ValueError, match="hf_id"):
            _validate_manifest(json.load(open(path)), path)

    def test_skip_comparison_bool_defaults_reason(self, tmp_path):
        """skip_comparison: true should produce a default reason string."""
        path = self._write_manifest(tmp_path, {
            "name": "rerank-test",
            "hf_id": "org/rerank",
            "family": "eagle_vlm",
            "runtime_strategy": "reranking",
            "skip_comparison": True,
        })
        case = load_manifest(path)
        assert case.metadata["skip_comparison_reason"]

    def test_skip_and_skip_comparison_are_independent(self, tmp_path):
        """`skip` still takes precedence (full skip); `skip_comparison` alone is partial."""
        path_full = self._write_manifest(tmp_path, {
            "name": "full-skip",
            "skip": "broken",
        })
        case_full = load_manifest(path_full)
        assert case_full.metadata["skip_reason"] == "broken"
        assert "skip_comparison_reason" not in case_full.metadata

    def test_flux2_fp8_manifest_uses_end_to_end_image_contract(self):
        """FLUX.2 FP8 should not inherit unrelated optional debug substages."""
        manifest_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "e2e",
            "models",
            "flux-2-dev-fp8.json",
        )
        case = load_manifest(manifest_path)

        assert case.reference_family == "diffusers_image_gen"
        assert case.user_contract == "diffusion_image"
        assert [stage.name for stage in case.stages] == ["end_to_end"]
        assert all(stage.required for stage in case.stages)
        assert "Wan-specific" in case.metadata["notes"]

    def test_sana_wm_manifest_preserves_official_demo_inputs(self):
        """SANA-WM e2e should expose the model-card image/prompt/action contract."""
        manifest_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "e2e",
            "models",
            "sana-wm-bidirectional.json",
        )
        case = load_manifest(manifest_path)

        assert case.task_strategy == "diffusion_media_generation"
        assert case.reference_backend == "hf_diffusers"
        assert case.inputs["image"] == "asset/sana_wm/demo_0.png"
        assert case.inputs["prompt_file"] == "asset/sana_wm/demo_0.txt"
        assert case.inputs["action"] == "w-80,jw-40,w-40,lw-60,w-100"
        assert case.inputs["translation_speed"] == 0.055
        assert case.inputs["rotation_speed_deg"] == 1.2
        assert case.inputs["sana_wm_require_official_script"] is True
        assert case.inputs["video_num_frames"] == 321
        asset_paths = {
            req.args["path"]
            for req in case.preflight
            if req.kind == "asset_exists"
        }
        assert asset_paths == {
            "asset/sana_wm/demo_0.png",
            "asset/sana_wm/demo_0.txt",
        }
        script_reqs = [
            req for req in case.preflight if req.kind == "sana_wm_script_available"
        ]
        assert len(script_reqs) == 1
        assert script_reqs[0].gating is False
        entrypoint_reqs = [
            req
            for req in case.preflight
            if req.kind == "sana_wm_runtime_entrypoint_available"
        ]
        assert len(entrypoint_reqs) == 1
        assert entrypoint_reqs[0].args["hf_id"] == case.hf_id
        assert entrypoint_reqs[0].gating is True
        module_reqs = {
            (req.args["module"], req.args["phase"])
            for req in case.preflight
            if req.kind == "python_module_available"
        }
        assert ("ftfy", "build") not in module_reqs
        assert {
            ("diffusers", "runtime"),
            ("diffusers", "reference"),
            ("PIL", "runtime"),
            ("PIL", "reference"),
        }.issubset(module_reqs)
