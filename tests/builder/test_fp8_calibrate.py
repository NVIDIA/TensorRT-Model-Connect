"""Generic FP8 calibration utility tests."""

from __future__ import annotations

class TestFp8Calibrate:
    """Test fp8_calibrate utility functions (pure Python, no GPU)."""

    def test_import(self):
        """Verify fp8_calibrate can be imported."""
        from tensorrt_model_connect import fp8_calibrate
        assert hasattr(fp8_calibrate, "extract_scales_from_state_dict")

    def test_maxbound_constants(self):
        """Verify maxbound constants are defined correctly."""
        from tensorrt_model_connect.fp8_calibrate import _MAXBOUND, _DEFAULT_MAXBOUND
        assert _MAXBOUND[(4, 3)] == 448.0  # FP8 E4M3
        assert _MAXBOUND[(5, 2)] == 57344.0  # FP8 E5M2
        assert _MAXBOUND[(0, 8)] == 127.0  # INT8
        assert _DEFAULT_MAXBOUND == 448.0

    def test_extract_scales_empty(self):
        """Scale extraction with empty state dict returns empty."""
        from tensorrt_model_connect.fp8_calibrate import extract_scales_from_state_dict
        result = extract_scales_from_state_dict({})
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_extract_scales_with_amax(self):
        """Scale extraction finds and maps _amax tensors."""
        from tensorrt_model_connect.fp8_calibrate import extract_scales_from_state_dict

        state = {
            "model.layers.0.self_attn.q_proj.input_quantizer._amax": 224.0,
            "model.layers.0.self_attn.q_proj.weight_quantizer._amax": 112.0,
            "model.layers.0.mlp.gate_proj.input_quantizer._amax": 100.0,
            # Missing weight_quantizer — should not appear in output
        }
        result = extract_scales_from_state_dict(state)
        assert "model.layers.0.self_attn.q_proj" in result
        entry = result["model.layers.0.self_attn.q_proj"]
        assert "input_scale" in entry
        assert "weight_scale" in entry
        assert abs(entry["input_scale"] - 224.0 / 448.0) < 1e-6
        assert abs(entry["weight_scale"] - 112.0 / 448.0) < 1e-6
        # Incomplete entry should not be present
        assert "model.layers.0.mlp.gate_proj" not in result

    def test_extract_scales_with_attention_bmm_amax(self):
        """Scale extraction preserves ModelOpt MHA quantizer amax tensors."""
        from tensorrt_model_connect.fp8_calibrate import extract_scales_from_state_dict

        state = {
            "transformer_blocks.0.attn.q_bmm_quantizer._amax": 44.8,
            "transformer_blocks.0.attn.k_bmm_quantizer._amax": 89.6,
            "transformer_blocks.0.attn.v_bmm_quantizer._amax": 134.4,
            "transformer_blocks.0.attn.softmax_quantizer._amax": 22.4,
            "transformer_blocks.0.attn.bmm2_output_quantizer._amax": 67.2,
            # Missing softmax_quantizer: ModelOpt SDPA FP8 hard-codes this
            # amax to 1.0, so extraction derives 1.0 / 448.0.
            "transformer_blocks.1.attn.q_bmm_quantizer._amax": 44.8,
            "transformer_blocks.1.attn.k_bmm_quantizer._amax": 89.6,
            "transformer_blocks.1.attn.v_bmm_quantizer._amax": 134.4,
        }

        result = extract_scales_from_state_dict(state)

        entry = result["transformer_blocks.0.attn"]
        assert abs(entry["q_bmm_scale"] - 0.1) < 1e-6
        assert abs(entry["k_bmm_scale"] - 0.2) < 1e-6
        assert abs(entry["v_bmm_scale"] - 0.3) < 1e-6
        assert abs(entry["softmax_scale"] - 0.05) < 1e-6
        assert abs(entry["bmm2_output_scale"] - 0.15) < 1e-6

        default_softmax = result["transformer_blocks.1.attn"]["softmax_scale"]
        assert abs(default_softmax - (1.0 / 448.0)) < 1e-9

    def test_extract_scales_with_exclude(self):
        """Scale extraction respects exclude_pattern."""
        import re
        from tensorrt_model_connect.fp8_calibrate import extract_scales_from_state_dict

        state = {
            "model.layers.0.self_attn.q_proj.input_quantizer._amax": 224.0,
            "model.layers.0.self_attn.q_proj.weight_quantizer._amax": 112.0,
        }
        # Exclude everything matching self_attn
        result = extract_scales_from_state_dict(
            state, exclude_pattern=re.compile(r".*self_attn.*"))
        assert len(result) == 0

    def test_maxbound_from_config(self):
        """Maxbound extraction from ModelOpt config."""
        from tensorrt_model_connect.fp8_calibrate import _maxbound_from_config

        config = {"quant_cfg": {"*weight_quantizer": {"num_bits": (4, 3)}}}
        assert _maxbound_from_config(config) == 448.0

        config = {"quant_cfg": {"*weight_quantizer": {"num_bits": (0, 8)}}}
        assert _maxbound_from_config(config) == 127.0

        # Unknown format falls back to default
        assert _maxbound_from_config({}) == 448.0
