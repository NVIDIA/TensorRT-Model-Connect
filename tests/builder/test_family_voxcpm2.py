"""VoxCPM2 family registration tests."""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest
from safetensors.numpy import save_file

from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.runtime_config import clear_for_testing, lookup
from tensorrt_model_connect.runtime_config.schemas import load_all


def _voxcpm2_config(tmp_path) -> ModelConfig:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architecture": "voxcpm2",
                "sample_rate": 48000,
                "cfg_value": 2.0,
                "inference_timesteps": 10,
            }
        ),
        encoding="utf-8",
    )
    return ModelConfig.from_dir(tmp_path)


def _write_raw_voxcpm2_checkpoint(tmp_path) -> ModelConfig:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architecture": "voxcpm2",
                "lm_config": {
                    "hidden_size": 2048,
                    "num_hidden_layers": 28,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 2,
                    "vocab_size": 73448,
                },
                "patch_size": 4,
                "feat_dim": 64,
                "scalar_quantization_latent_dim": 512,
                "scalar_quantization_scale": 9,
                "residual_lm_num_layers": 8,
                "encoder_config": {
                    "hidden_dim": 1024,
                    "ffn_dim": 4096,
                    "num_heads": 16,
                    "num_layers": 12,
                },
                "dit_config": {
                    "hidden_dim": 1024,
                    "ffn_dim": 4096,
                    "num_heads": 16,
                    "num_layers": 12,
                    "cfm_config": {
                        "solver": "euler",
                        "inference_cfg_rate": 2.0,
                    },
                },
                "audio_vae_config": {
                    "sample_rate": 16000,
                    "out_sample_rate": 48000,
                    "latent_dim": 64,
                },
                "max_length": 8192,
                "sample_rate": 48000,
                "cfg_value": 2.0,
                "inference_timesteps": 10,
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "base_lm.layers.0.weight": np.array([1.0], dtype=np.float32),
            "enc_to_lm_proj.weight": np.array([2.0], dtype=np.float32),
            "feat_decoder.estimator.weight": np.array([3.0], dtype=np.float32),
            "feat_encoder.layers.0.weight": np.array([4.0], dtype=np.float32),
            "fusion_concat_proj.weight": np.array([5.0], dtype=np.float32),
            "lm_to_dit_proj.weight": np.array([6.0], dtype=np.float32),
            "residual_lm.layers.0.weight": np.array([7.0], dtype=np.float32),
            "res_to_dit_proj.weight": np.array([8.0], dtype=np.float32),
            "stop_head.weight": np.array([9.0], dtype=np.float32),
            "unrelated.weight": np.array([10.0], dtype=np.float32),
        },
        tmp_path / "model.safetensors",
    )
    (tmp_path / "audiovae.pth").write_bytes(b"raw-audio-vae")
    for filename in (
        "tokenization_voxcpm2.py",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    ):
        (tmp_path / filename).write_text("{}", encoding="utf-8")
    return ModelConfig.from_dir(tmp_path)


def test_model_config_uses_architecture_fallback_for_voxcpm2(tmp_path):
    cfg = _voxcpm2_config(tmp_path)

    assert cfg.model_type == "voxcpm2"
    assert cfg.architectures == ["voxcpm2"]


def test_voxcpm2_plugin_records_metadata_and_audio_defaults(tmp_path):
    from tensorrt_model_connect.families.voxcpm2.plugin import plugin

    cfg = _voxcpm2_config(tmp_path)

    assert plugin.matches(cfg.model_type)
    assert plugin.runtime_strategy == "text_to_audio_voxcpm2"
    weights = plugin.load_weights(str(tmp_path), cfg)
    assert weights["_architecture"] == "voxcpm2"
    assert weights["_voxcpm2_components"] == (
        "locenc",
        "tslm",
        "ralm",
        "locdit",
        "audiovae",
    )
    assert weights["_voxcpm2_engine_sections"] == (
        "locenc_engine_plan",
        "tslm_engine_plan",
        "ralm_engine_plan",
        "locdit_engine_plan",
        "audiovae_engine_plan",
    )
    assert weights["_voxcpm2_raw_component_sources"] == {}
    assert weights["_sample_rate"] == 48000
    assert weights["_reference_sample_rate"] == 16000
    assert weights["_cfg_value"] == 2.0
    assert weights["_inference_timesteps"] == 10

    audio_cfg = plugin.get_audio_config(cfg)
    assert audio_cfg["sample_rate"] == 48000
    assert audio_cfg["reference_sample_rate"] == 16000
    assert audio_cfg["voxcpm2_cfg_value"] == 2.0
    assert audio_cfg["voxcpm2_inference_timesteps"] == 10
    assert audio_cfg["voxcpm2_patch_size"] == 4
    assert audio_cfg["voxcpm2_feat_dim"] == 64


def test_voxcpm2_audio_defaults_follow_nested_audio_vae_config(tmp_path):
    from tensorrt_model_connect.families.voxcpm2.plugin import plugin

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architecture": "voxcpm2",
                "lm_config": {"hidden_size": 2048},
                "audio_vae_config": {
                    "sample_rate": 16000,
                    "out_sample_rate": 48000,
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = ModelConfig.from_dir(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)
    audio_cfg = plugin.get_audio_config(cfg)

    assert weights["_sample_rate"] == 48000
    assert weights["_reference_sample_rate"] == 16000
    assert audio_cfg["sample_rate"] == 48000
    assert audio_cfg["reference_sample_rate"] == 16000


def test_voxcpm2_top_level_audio_rates_override_nested_defaults(tmp_path):
    from tensorrt_model_connect.families.voxcpm2.plugin import plugin

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architecture": "voxcpm2",
                "sample_rate": 44100,
                "reference_sample_rate": 22050,
                "audio_vae_config": {
                    "sample_rate": 16000,
                    "out_sample_rate": 48000,
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = ModelConfig.from_dir(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)
    audio_cfg = plugin.get_audio_config(cfg)

    assert weights["_sample_rate"] == 44100
    assert weights["_reference_sample_rate"] == 22050
    assert audio_cfg["sample_rate"] == 44100
    assert audio_cfg["reference_sample_rate"] == 22050


def test_voxcpm2_build_boundary_is_explicit(tmp_path):
    from tensorrt_model_connect.families.voxcpm2.plugin import plugin

    cfg = _voxcpm2_config(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)

    with pytest.raises(
        NotImplementedError, match="missing artifacts for locenc, tslm, ralm, locdit, audiovae"
    ):
        plugin.build_engine(cfg, weights, max_cache_length=16)


def test_voxcpm2_raw_checkpoint_sources_are_recorded(tmp_path):
    from tensorrt_model_connect.families.voxcpm2.plugin import plugin

    cfg = _write_raw_voxcpm2_checkpoint(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)
    sources = weights["_voxcpm2_raw_component_sources"]

    assert tuple(sources) == ("locenc", "tslm", "ralm", "locdit", "audiovae")
    assert sources["locenc"].config_keys == (
        "lm_config",
        "encoder_config",
        "patch_size",
        "feat_dim",
    )
    assert sources["locenc"].config_values["lm_config"]["hidden_size"] == 2048
    assert sources["locenc"].config_values["encoder_config"]["hidden_dim"] == 1024
    assert sources["locenc"].config_values["patch_size"] == 4
    assert sources["locenc"].config_values["feat_dim"] == 64
    assert sources["locenc"].weight_files == ("model.safetensors",)
    assert sources["locenc"].state_dict_prefixes == ("feat_encoder.", "enc_to_lm_proj.")
    assert sources["locenc"].asset_files == ()
    assert sources["tslm"].config_values["lm_config"]["hidden_size"] == 2048
    assert sources["tslm"].config_values["max_length"] == 8192
    assert sources["tslm"].config_values["scalar_quantization_latent_dim"] == 512
    assert sources["tslm"].config_values["scalar_quantization_scale"] == 9
    assert sources["tslm"].state_dict_prefixes == (
        "base_lm.",
        "fsq_layer.",
        "stop_proj.",
        "stop_head.",
    )
    assert sources["tslm"].asset_files == (
        "tokenization_voxcpm2.py",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    )
    assert sources["ralm"].config_values["residual_lm_num_layers"] == 8
    assert sources["ralm"].config_values["scalar_quantization_latent_dim"] == 512
    assert sources["ralm"].state_dict_prefixes == ("fusion_concat_proj.", "residual_lm.")
    assert sources["ralm"].asset_files == ()
    assert sources["locdit"].config_keys == (
        "lm_config",
        "dit_config",
        "dit_config.cfm_config",
        "patch_size",
        "feat_dim",
    )
    assert sources["locdit"].config_values["dit_config"]["hidden_dim"] == 1024
    assert sources["locdit"].config_values["dit_config.cfm_config"]["solver"] == "euler"
    assert sources["locdit"].state_dict_prefixes == (
        "lm_to_dit_proj.",
        "res_to_dit_proj.",
        "feat_decoder.",
    )
    assert sources["locdit"].asset_files == ()
    assert sources["audiovae"].config_values["audio_vae_config"]["out_sample_rate"] == 48000
    assert sources["audiovae"].weight_files == ("audiovae.pth",)
    assert sources["audiovae"].state_dict_prefixes == ()
    assert sources["audiovae"].asset_files == ()


def test_voxcpm2_component_specs_include_tensor_contracts():
    from tensorrt_model_connect.families.voxcpm2 import component_builders

    specs = component_builders.VOXCPM2_COMPONENT_SPECS

    assert specs[0].input_tensor.name == "audio_feats"
    assert specs[0].input_tensor.dtype_contract == ("float32", "bfloat16")
    assert specs[0].input_tensor.rank == 3
    assert specs[0].input_tensor.symbolic_shape == ("text_steps", "patch_size", "feat_dim")
    assert specs[0].output_tensor.name == "local_text_features"
    assert specs[0].output_tensor.dtype_contract == ("float32", "bfloat16")
    assert specs[0].output_tensor.rank == 2
    assert specs[0].output_tensor.symbolic_shape == ("text_steps", "lm_hidden_size")
    assert specs[4].input_tensor.name == "audio_vae_latents"
    assert specs[4].input_tensor.rank == 2
    assert specs[4].output_tensor.name == "waveform_f32"
    assert specs[4].output_tensor.dtype_contract == ("float32",)
    assert specs[4].output_tensor.rank == 1
    assert specs[4].output_tensor.symbolic_shape == ("audio_samples",)


def test_voxcpm2_component_specs_include_upstream_handoff_metadata():
    from tensorrt_model_connect.families.voxcpm2 import component_builders

    specs = {spec.name: spec for spec in component_builders.VOXCPM2_COMPONENT_SPECS}

    assert specs["locenc"].upstream_inputs == ("audio_feats",)
    assert specs["locenc"].upstream_outputs == ("feat_embed", "local_text_features")
    assert [
        module.describe() for module in specs["locenc"].upstream_modules
    ] == [
        "voxcpm.modules.locenc.VoxCPMLocEnc(feat_encoder.)",
        "torch.nn.Linear(enc_to_lm_proj.)",
    ]

    assert specs["tslm"].upstream_inputs == (
        "text_tokens",
        "text_mask",
        "local_text_features",
        "audio_mask",
    )
    assert specs["tslm"].upstream_outputs == (
        "semantic_lm_states",
        "lm_hidden",
        "stop_logits",
    )
    assert specs["tslm"].required_side_inputs == (
        "text_tokens",
        "text_mask",
        "audio_mask",
    )
    assert [
        module.describe() for module in specs["tslm"].upstream_modules
    ] == [
        "voxcpm.modules.minicpm4.MiniCPMModel(base_lm.)",
        "voxcpm.modules.layers.ScalarQuantizationLayer(fsq_layer.)",
        "torch.nn.Linear(stop_proj., stop_head.)",
    ]

    assert specs["ralm"].upstream_outputs == ("residual_hidden",)
    assert specs["ralm"].required_side_inputs == ("audio_mask", "local_text_features")
    assert specs["ralm"].required_control_inputs == ()
    assert [
        module.describe() for module in specs["locdit"].upstream_modules
    ] == [
        "voxcpm.modules.locdit.UnifiedCFM(feat_decoder.)",
        "voxcpm.modules.locdit.VoxCPMLocDiTV2(feat_decoder.estimator.)",
        "torch.nn.Linear(lm_to_dit_proj., res_to_dit_proj.)",
    ]
    assert specs["locdit"].required_side_inputs == ("lm_hidden",)
    assert specs["locdit"].required_control_inputs == ("cfg_value", "inference_timesteps")
    assert specs["audiovae"].upstream_inputs == ("audio_vae_latents",)
    assert specs["audiovae"].upstream_outputs == ("waveform_f32",)


def test_voxcpm2_raw_checkpoint_reports_native_builder_gap(tmp_path, monkeypatch):
    from tensorrt_model_connect.families.voxcpm2 import component_builders
    from tensorrt_model_connect.families.voxcpm2.plugin import plugin

    cfg = _write_raw_voxcpm2_checkpoint(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)
    fake_builders = dict(component_builders.DEFAULT_COMPONENT_BUILDERS)
    fake_builders["locenc"] = lambda ctx: b"LOCENC-TRT"
    fake_builders["tslm"] = lambda ctx: b"TSLM-TRT"
    fake_builders["ralm"] = lambda ctx: b"RALM-TRT"
    fake_builders["locdit"] = lambda ctx: component_builders._raise_native_builder_gap(
        ctx, component_builders.prepare_component_inputs(ctx)
    )
    monkeypatch.setattr(plugin, "component_builders", fake_builders)

    with pytest.raises(NotImplementedError) as error:
        plugin.build_engine(cfg, weights, max_cache_length=16)

    message = str(error.value)
    assert "raw checkpoint sources are present for locenc, tslm, ralm, locdit, audiovae" in message
    assert "native TRT builders are incomplete" in message
    assert "assets: tokenization_voxcpm2.py, tokenizer_config.json" in message
    assert "locdit(config: lm_config, dit_config, dit_config.cfm_config, patch_size, feat_dim" in message
    assert "audiovae(config: audio_vae_config" in message
    assert "still requires every component builder to complete" in message
    assert "component 'locdit' is not implemented yet" in message
    assert "input binding 'residual_hidden'" in message
    assert "output binding 'audio_vae_latents'" in message
    assert "residual_hidden:float32|bfloat16" in message
    assert "Prepared safetensors checkpoint inputs with 3 state entries" in message
    assert "native text-to-audio runtime to write the TRT WAV artifact" in message
    assert "Upstream handoff:" in message
    assert "voxcpm.modules.locdit.UnifiedCFM(feat_decoder.)" in message
    assert "torch.nn.Linear(lm_to_dit_proj., res_to_dit_proj.)" in message
    assert "runtime inputs: lm_hidden, residual_hidden, feat_cond" in message
    assert "runtime outputs: audio_vae_latents" in message
    assert "required_side=text_tokens,text_mask,audio_mask" in message
    assert "Runtime binding contract:" in message
    assert "ralm(semantic_lm_states=>residual_hidden" in message
    assert "required_side=audio_mask,local_text_features" in message
    assert "locdit(residual_hidden=>audio_vae_latents" in message
    assert "required_side=lm_hidden" in message
    assert "required_controls=cfg_value,inference_timesteps" in message


def test_voxcpm2_tslm_builder_wraps_upstream_modules_for_export(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from tensorrt_model_connect.families.voxcpm2 import component_builders

    class FakeMiniCPM4Config:
        def __init__(self, **kwargs):
            self.hidden_size = int(kwargs["hidden_size"])
            self.vocab_size = int(kwargs["vocab_size"])

    class FakeMiniCPMModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)

        def load_state_dict(self, state_dict, strict=True):
            assert "dummy" in state_dict
            return torch.nn.modules.module._IncompatibleKeys([], [])

        def forward(self, *, inputs_embeds, is_causal):
            assert is_causal is True
            return inputs_embeds + 1.0, ()

    class FakeScalarQuantizationLayer(torch.nn.Module):
        def __init__(self, *args):
            super().__init__()
            self.args = args

        def load_state_dict(self, state_dict, strict=True):
            assert "dummy" in state_dict
            return torch.nn.modules.module._IncompatibleKeys([], [])

        def forward(self, values):
            return values * 2.0

    voxcpm_mod = types.ModuleType("voxcpm")
    voxcpm_modules_mod = types.ModuleType("voxcpm.modules")
    layers_mod = types.ModuleType("voxcpm.modules.layers")
    minicpm_mod = types.ModuleType("voxcpm.modules.minicpm4")
    layers_mod.ScalarQuantizationLayer = FakeScalarQuantizationLayer
    minicpm_mod.MiniCPM4Config = FakeMiniCPM4Config
    minicpm_mod.MiniCPMModel = FakeMiniCPMModel
    monkeypatch.setitem(sys.modules, "voxcpm", voxcpm_mod)
    monkeypatch.setitem(sys.modules, "voxcpm.modules", voxcpm_modules_mod)
    monkeypatch.setitem(sys.modules, "voxcpm.modules.layers", layers_mod)
    monkeypatch.setitem(sys.modules, "voxcpm.modules.minicpm4", minicpm_mod)

    save_file(
        {
            "base_lm.dummy": np.array([1.0], dtype=np.float32),
            "fsq_layer.dummy": np.array([2.0], dtype=np.float32),
            "stop_proj.weight": np.eye(2, dtype=np.float32),
            "stop_proj.bias": np.zeros(2, dtype=np.float32),
            "stop_head.weight": np.ones((2, 2), dtype=np.float32),
        },
        tmp_path / "model.safetensors",
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architecture": "voxcpm2",
                "lm_config": {"hidden_size": 2, "vocab_size": 128},
                "max_length": 8,
                "scalar_quantization_latent_dim": 1,
                "scalar_quantization_scale": 3,
            }
        ),
        encoding="utf-8",
    )
    cfg = ModelConfig.from_dir(tmp_path)
    source = types.SimpleNamespace(
        config_values={
            "lm_config": {"hidden_size": 2, "vocab_size": 128},
            "max_length": 8,
            "scalar_quantization_latent_dim": 1,
            "scalar_quantization_scale": 3,
        },
        weight_files=("model.safetensors",),
        state_dict_prefixes=("base_lm.", "fsq_layer.", "stop_proj.", "stop_head."),
        asset_files=(),
    )
    captured = {}

    def fake_compile(wrapper, example_args, *, verbose):
        outputs = wrapper(*example_args)
        captured["verbose"] = verbose
        captured["input_shapes"] = tuple(tuple(arg.shape) for arg in example_args)
        captured["output_shapes"] = tuple(tuple(out.shape) for out in outputs)
        captured["output_dtypes"] = tuple(out.dtype for out in outputs)
        return b"TSLM-PLAN"

    monkeypatch.setattr(component_builders, "_compile_voxcpm2_tslm_onnx", fake_compile)

    plan = component_builders.build_tslm_engine(
        component_builders.VoxCPM2ComponentBuildContext(
            spec=component_builders.VOXCPM2_COMPONENT_SPECS[1],
            model_dir=tmp_path,
            config=cfg,
            source=source,
            precision="fp32",
            verbose=True,
            max_cache_length=3,
        )
    )

    assert plan == b"TSLM-PLAN"
    assert captured["verbose"] is True
    assert captured["input_shapes"] == ((3, 2), (3,), (3,), (3,))
    assert captured["output_shapes"] == ((3, 2), (3, 2), (3, 2))
    assert captured["output_dtypes"] == (torch.float32, torch.float32, torch.float32)


def test_voxcpm2_ralm_builder_wraps_upstream_modules_for_export(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from tensorrt_model_connect.families.voxcpm2 import component_builders

    class FakeMiniCPM4Config:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeMiniCPMModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

        def load_state_dict(self, state_dict, strict=True):
            assert "dummy" in state_dict
            return torch.nn.modules.module._IncompatibleKeys([], [])

        def forward(self, *, inputs_embeds, is_causal):
            assert is_causal is True
            return inputs_embeds + 1.0, ()

    voxcpm_mod = types.ModuleType("voxcpm")
    voxcpm_modules_mod = types.ModuleType("voxcpm.modules")
    minicpm_mod = types.ModuleType("voxcpm.modules.minicpm4")
    minicpm_mod.MiniCPM4Config = FakeMiniCPM4Config
    minicpm_mod.MiniCPMModel = FakeMiniCPMModel
    monkeypatch.setitem(sys.modules, "voxcpm", voxcpm_mod)
    monkeypatch.setitem(sys.modules, "voxcpm.modules", voxcpm_modules_mod)
    monkeypatch.setitem(sys.modules, "voxcpm.modules.minicpm4", minicpm_mod)

    save_file(
        {
            "fusion_concat_proj.weight": np.ones((2, 4), dtype=np.float32),
            "fusion_concat_proj.bias": np.zeros(2, dtype=np.float32),
            "residual_lm.dummy": np.array([1.0], dtype=np.float32),
        },
        tmp_path / "model.safetensors",
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architecture": "voxcpm2",
                "lm_config": {"hidden_size": 2, "vocab_size": 128},
                "residual_lm_num_layers": 4,
                "residual_lm_no_rope": True,
                "max_length": 8,
            }
        ),
        encoding="utf-8",
    )
    cfg = ModelConfig.from_dir(tmp_path)
    source = types.SimpleNamespace(
        config_values={
            "lm_config": {"hidden_size": 2, "vocab_size": 128},
            "residual_lm_num_layers": 4,
            "max_length": 8,
        },
        weight_files=("model.safetensors",),
        state_dict_prefixes=("fusion_concat_proj.", "residual_lm."),
        asset_files=(),
    )
    captured = {}

    def fake_compile(wrapper, example_args, *, verbose):
        output = wrapper(*example_args)
        captured["wrapper"] = wrapper
        captured["verbose"] = verbose
        captured["input_shapes"] = tuple(tuple(arg.shape) for arg in example_args)
        captured["output_shape"] = tuple(output.shape)
        captured["output_dtype"] = output.dtype
        return b"RALM-PLAN"

    monkeypatch.setattr(component_builders, "_compile_voxcpm2_ralm_onnx", fake_compile)

    plan = component_builders.build_ralm_engine(
        component_builders.VoxCPM2ComponentBuildContext(
            spec=component_builders.VOXCPM2_COMPONENT_SPECS[2],
            model_dir=tmp_path,
            config=cfg,
            source=source,
            precision="fp32",
            verbose=True,
            max_cache_length=3,
        )
    )

    assert plan == b"RALM-PLAN"
    assert captured["verbose"] is True
    assert captured["input_shapes"] == ((3, 2), (3,), (3, 2))
    assert captured["output_shape"] == (3, 2)
    assert captured["output_dtype"] == torch.float32
    residual_config = captured["wrapper"].residual_lm.config.kwargs
    assert residual_config["num_hidden_layers"] == 4
    assert residual_config["vocab_size"] == 0
    assert residual_config["no_rope"] is True


def test_voxcpm2_locdit_builder_exports_named_trt_engine(tmp_path, monkeypatch):
    from tensorrt_model_connect import checkpoint_mapper
    from tensorrt_model_connect.families.voxcpm2 import component_builders
    from tensorrt_model_connect.families.voxcpm2.plugin import plugin

    class FakeTorchModule:
        def __init__(self):
            self.eval_called = False
            self.dtype = None

        def eval(self):
            self.eval_called = True
            return self

        def to(self, *, dtype):
            self.dtype = dtype
            return self

    class FakeTensor:
        def __init__(self, shape, dtype):
            self.shape = tuple(shape)
            self.dtype = dtype

    class FakeMiniCPM4Config:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.hidden_size = kwargs["hidden_size"]

    class FakeCfmConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeVoxCPMLocDiTV2(FakeTorchModule):
        def __init__(self, config, *, in_channels):
            super().__init__()
            self.config = config
            self.in_channels = in_channels

    class FakeUnifiedCFM(FakeTorchModule):
        def __init__(self, *, in_channels, cfm_params, estimator, mean_mode):
            super().__init__()
            self.in_channels = in_channels
            self.cfm_params = cfm_params
            self.estimator = estimator
            self.mean_mode = mean_mode
            self.loaded = None
            self.strict = None

        def load_state_dict(self, state, *, strict):
            self.loaded = state
            self.strict = strict

    class FakeLinear(FakeTorchModule):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.loaded = None
            self.strict = None

        def load_state_dict(self, state, *, strict):
            self.loaded = state
            self.strict = strict

    fake_torch = types.SimpleNamespace(
        bfloat16="bf16",
        float16="fp16",
        float32="fp32",
        int32="int32",
        as_tensor=lambda value: value,
        nn=types.SimpleNamespace(Module=FakeTorchModule, Linear=FakeLinear),
        zeros=lambda shape, *, dtype: FakeTensor(shape, dtype),
        tensor=lambda value, *, dtype: FakeTensor((len(value),), dtype),
    )
    fake_voxcpm = types.ModuleType("voxcpm")
    fake_voxcpm_modules = types.ModuleType("voxcpm.modules")
    fake_locdit = types.ModuleType("voxcpm.modules.locdit")
    fake_minicpm4 = types.ModuleType("voxcpm.modules.minicpm4")
    fake_locdit.CfmConfig = FakeCfmConfig
    fake_locdit.UnifiedCFM = FakeUnifiedCFM
    fake_locdit.VoxCPMLocDiTV2 = FakeVoxCPMLocDiTV2
    fake_minicpm4.MiniCPM4Config = FakeMiniCPM4Config
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "voxcpm", fake_voxcpm)
    monkeypatch.setitem(sys.modules, "voxcpm.modules", fake_voxcpm_modules)
    monkeypatch.setitem(sys.modules, "voxcpm.modules.locdit", fake_locdit)
    monkeypatch.setitem(sys.modules, "voxcpm.modules.minicpm4", fake_minicpm4)
    monkeypatch.setattr(checkpoint_mapper, "_detect_framework", lambda: "numpy")

    captured = {}

    def _fake_compile(wrapper, example_args, *, verbose):
        captured["wrapper"] = wrapper
        captured["example_args"] = example_args
        captured["verbose"] = verbose
        return b"LOCDIT-TRT"

    monkeypatch.setattr(
        component_builders,
        "_compile_voxcpm2_locdit_onnx",
        _fake_compile,
    )

    cfg = _write_raw_voxcpm2_checkpoint(tmp_path)
    save_file(
        {
            "feat_decoder.estimator.decoder.weight": np.array([1.0], dtype=np.float32),
            "lm_to_dit_proj.bias": np.array([2.0], dtype=np.float32),
            "lm_to_dit_proj.weight": np.array([[3.0]], dtype=np.float32),
            "res_to_dit_proj.bias": np.array([4.0], dtype=np.float32),
            "res_to_dit_proj.weight": np.array([[5.0]], dtype=np.float32),
            "unrelated.weight": np.array([6.0], dtype=np.float32),
        },
        tmp_path / "model.safetensors",
    )
    weights = plugin.load_weights(str(tmp_path), cfg)

    def _fake_component_builder(component_name):
        def _builder(ctx):
            assert ctx.max_cache_length == 5
            return f"{component_name}-plan".encode("ascii")

        return _builder

    fake_builders = {
        spec.name: _fake_component_builder(spec.name)
        for spec in component_builders.VOXCPM2_COMPONENT_SPECS
    }
    fake_builders["locdit"] = component_builders.build_locdit_engine
    monkeypatch.setattr(plugin, "component_builders", fake_builders)

    sections = plugin.build_engine(
        cfg,
        weights,
        max_cache_length=5,
        precision="bf16",
        verbose=True,
    )

    assert sections["locdit_engine_plan"] == b"LOCDIT-TRT"
    assert captured["verbose"] is True
    residual_hidden, lm_hidden, cfg_value, inference_timesteps = captured["example_args"]
    assert residual_hidden.shape == (5, 2048)
    assert residual_hidden.dtype == "bf16"
    assert lm_hidden.shape == (5, 2048)
    assert lm_hidden.dtype == "bf16"
    assert cfg_value.shape == (1,)
    assert cfg_value.dtype == "fp32"
    assert inference_timesteps.shape == (1,)
    assert inference_timesteps.dtype == "int32"

    wrapper = captured["wrapper"]
    feat_decoder = wrapper.feat_decoder
    assert feat_decoder.in_channels == 64
    assert feat_decoder.cfm_params.kwargs["solver"] == "euler"
    assert feat_decoder.estimator.in_channels == 64
    assert feat_decoder.estimator.config.kwargs["hidden_size"] == 1024
    assert feat_decoder.estimator.config.kwargs["num_hidden_layers"] == 12
    assert feat_decoder.estimator.config.kwargs["vocab_size"] == 0
    assert tuple(feat_decoder.loaded) == ("estimator.decoder.weight",)
    assert feat_decoder.strict is True
    assert feat_decoder.dtype == "bf16"
    assert feat_decoder.eval_called is True

    lm_projection = wrapper.lm_to_dit_proj
    assert lm_projection.in_features == 2048
    assert lm_projection.out_features == 1024
    assert tuple(lm_projection.loaded) == ("bias", "weight")
    assert lm_projection.strict is True
    assert lm_projection.dtype == "bf16"
    assert lm_projection.eval_called is True

    residual_projection = wrapper.res_to_dit_proj
    assert residual_projection.in_features == 2048
    assert residual_projection.out_features == 1024
    assert tuple(residual_projection.loaded) == ("bias", "weight")
    assert residual_projection.strict is True
    assert residual_projection.dtype == "bf16"
    assert residual_projection.eval_called is True
    assert wrapper.default_inference_timesteps == 10
    assert wrapper.eval_called is True


def test_voxcpm2_raw_checkpoint_invokes_native_component_builders(tmp_path, monkeypatch):
    from tensorrt_model_connect.families.voxcpm2 import component_builders
    from tensorrt_model_connect.families.voxcpm2.plugin import plugin

    cfg = _write_raw_voxcpm2_checkpoint(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)
    calls = []
    path_calls = []

    def make_builder(component_name):
        def _builder(ctx):
            calls.append(
                (
                    ctx.spec.name,
                    ctx.spec.engine_section,
                    ctx.spec.input_artifact,
                    ctx.spec.output_artifact,
                    ctx.model_dir,
                    ctx.precision,
                    ctx.verbose,
                    ctx.max_cache_length,
                    ctx.source.config_keys,
                    ctx.source.asset_files,
                )
            )
            path_calls.append((ctx.spec.name, ctx.weight_paths, ctx.asset_paths))
            return f"{component_name}-plan".encode("ascii")

        return _builder

    fake_builders = {
        spec.name: make_builder(spec.name)
        for spec in component_builders.VOXCPM2_COMPONENT_SPECS
    }
    monkeypatch.setattr(plugin, "component_builders", fake_builders)

    sections = plugin.build_engine(
        cfg,
        weights,
        max_cache_length=16,
        precision="bf16",
        verbose=True,
    )

    assert sections == {
        spec.engine_section: f"{spec.name}-plan".encode("ascii")
        for spec in component_builders.VOXCPM2_COMPONENT_SPECS
    }
    assert calls == [
        (
            "locenc",
            "locenc_engine_plan",
            "audio_feats",
            "local_text_features",
            tmp_path,
            "bf16",
            True,
            16,
            ("lm_config", "encoder_config", "patch_size", "feat_dim"),
            (),
        ),
        (
            "tslm",
            "tslm_engine_plan",
            "local_text_features",
            "semantic_lm_states",
            tmp_path,
            "bf16",
            True,
            16,
            (
                "lm_config",
                "max_length",
                "scalar_quantization_latent_dim",
                "scalar_quantization_scale",
            ),
            (
                "tokenization_voxcpm2.py",
                "tokenizer_config.json",
                "tokenizer.json",
                "special_tokens_map.json",
            ),
        ),
        (
            "ralm",
            "ralm_engine_plan",
            "semantic_lm_states",
            "residual_hidden",
            tmp_path,
            "bf16",
            True,
            16,
            (
                "lm_config",
                "residual_lm_num_layers",
                "scalar_quantization_latent_dim",
                "scalar_quantization_scale",
            ),
            (),
        ),
        (
            "locdit",
            "locdit_engine_plan",
            "residual_hidden",
            "audio_vae_latents",
            tmp_path,
            "bf16",
            True,
            16,
            ("lm_config", "dit_config", "dit_config.cfm_config", "patch_size", "feat_dim"),
            (),
        ),
        (
            "audiovae",
            "audiovae_engine_plan",
            "audio_vae_latents",
            "waveform_f32",
            tmp_path,
            "bf16",
            True,
            16,
            (
                "audio_vae_config",
                "audio_vae_config.sample_rate",
                "audio_vae_config.out_sample_rate",
            ),
            (),
        ),
    ]
    tokenizer_asset_paths = (
        tmp_path / "tokenization_voxcpm2.py",
        tmp_path / "tokenizer_config.json",
        tmp_path / "tokenizer.json",
        tmp_path / "special_tokens_map.json",
    )
    assert path_calls == [
        ("locenc", (tmp_path / "model.safetensors",), ()),
        ("tslm", (tmp_path / "model.safetensors",), tokenizer_asset_paths),
        ("ralm", (tmp_path / "model.safetensors",), ()),
        ("locdit", (tmp_path / "model.safetensors",), ()),
        ("audiovae", (tmp_path / "audiovae.pth",), ()),
    ]


def test_voxcpm2_raw_checkpoint_reuses_partial_prebuilt_plans(tmp_path, monkeypatch):
    from tensorrt_model_connect.families.voxcpm2 import component_builders
    from tensorrt_model_connect.families.voxcpm2.plugin import plugin

    cfg = _write_raw_voxcpm2_checkpoint(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)
    (tmp_path / "locenc.plan").write_bytes(b"LOCENC-PREBUILT")
    (tmp_path / "tslm.engine").write_bytes(b"TSLM-PREBUILT")
    calls = []

    def _builder(ctx):
        calls.append(ctx.spec.name)
        return f"BUILT-{ctx.spec.name}".encode("ascii")

    fake_builders = {
        spec.name: _builder for spec in component_builders.VOXCPM2_COMPONENT_SPECS
    }
    monkeypatch.setattr(plugin, "component_builders", fake_builders)

    sections = plugin.build_engine(
        cfg,
        weights,
        max_cache_length=32,
        precision="bf16",
        verbose=True,
    )

    assert sections == {
        "locenc_engine_plan": b"LOCENC-PREBUILT",
        "tslm_engine_plan": b"TSLM-PREBUILT",
        "ralm_engine_plan": b"BUILT-ralm",
        "locdit_engine_plan": b"BUILT-locdit",
        "audiovae_engine_plan": b"BUILT-audiovae",
    }
    assert calls == ["ralm", "locdit", "audiovae"]


def test_voxcpm2_locenc_builder_exports_named_trt_engine(tmp_path, monkeypatch):
    from tensorrt_model_connect import checkpoint_mapper
    from tensorrt_model_connect.families.voxcpm2 import component_builders
    from tensorrt_model_connect.families.voxcpm2.plugin import plugin

    class FakeTorchModule:
        def __init__(self):
            self.eval_called = False
            self.dtype = None

        def eval(self):
            self.eval_called = True
            return self

        def to(self, *, dtype):
            self.dtype = dtype
            return self

    class FakeTensor:
        def __init__(self, shape, dtype):
            self.shape = tuple(shape)
            self.dtype = dtype

    class FakeMiniCPM4Config:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.hidden_size = kwargs["hidden_size"]

    class FakeLocEnc(FakeTorchModule):
        def __init__(self, config, *, input_dim):
            super().__init__()
            self.config = config
            self.input_dim = input_dim
            self.loaded = None
            self.strict = None

        def load_state_dict(self, state, *, strict):
            self.loaded = state
            self.strict = strict

    class FakeLinear(FakeTorchModule):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.loaded = None
            self.strict = None

        def load_state_dict(self, state, *, strict):
            self.loaded = state
            self.strict = strict

    fake_torch = types.SimpleNamespace(
        bfloat16="bf16",
        float16="fp16",
        float32="fp32",
        as_tensor=lambda value: value,
        nn=types.SimpleNamespace(Module=FakeTorchModule, Linear=FakeLinear),
        zeros=lambda shape, *, dtype: FakeTensor(shape, dtype),
    )
    fake_voxcpm = types.ModuleType("voxcpm")
    fake_voxcpm_modules = types.ModuleType("voxcpm.modules")
    fake_locenc = types.ModuleType("voxcpm.modules.locenc")
    fake_minicpm4 = types.ModuleType("voxcpm.modules.minicpm4")
    fake_locenc.VoxCPMLocEnc = FakeLocEnc
    fake_minicpm4.MiniCPM4Config = FakeMiniCPM4Config
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "voxcpm", fake_voxcpm)
    monkeypatch.setitem(sys.modules, "voxcpm.modules", fake_voxcpm_modules)
    monkeypatch.setitem(sys.modules, "voxcpm.modules.locenc", fake_locenc)
    monkeypatch.setitem(sys.modules, "voxcpm.modules.minicpm4", fake_minicpm4)
    monkeypatch.setattr(checkpoint_mapper, "_detect_framework", lambda: "numpy")

    captured = {}

    def _fake_compile(wrapper, example_args, *, verbose):
        captured["wrapper"] = wrapper
        captured["example_args"] = example_args
        captured["verbose"] = verbose
        return b"LOCENC-TRT"

    monkeypatch.setattr(
        component_builders,
        "_compile_voxcpm2_locenc_onnx",
        _fake_compile,
    )

    cfg = _write_raw_voxcpm2_checkpoint(tmp_path)
    save_file(
        {
            "enc_to_lm_proj.bias": np.array([1.0], dtype=np.float32),
            "enc_to_lm_proj.weight": np.array([[2.0]], dtype=np.float32),
            "feat_encoder.encoder.layers.0.weight": np.array([3.0], dtype=np.float32),
            "feat_encoder.in_proj.bias": np.array([4.0], dtype=np.float32),
            "feat_encoder.in_proj.weight": np.array([[5.0]], dtype=np.float32),
            "unrelated.weight": np.array([6.0], dtype=np.float32),
        },
        tmp_path / "model.safetensors",
    )
    weights = plugin.load_weights(str(tmp_path), cfg)

    def _fake_component_builder(component_name):
        def _builder(ctx):
            assert ctx.max_cache_length == 9
            return f"{component_name}-plan".encode("ascii")

        return _builder

    fake_builders = {
        spec.name: _fake_component_builder(spec.name)
        for spec in component_builders.VOXCPM2_COMPONENT_SPECS
    }
    fake_builders["locenc"] = component_builders.build_locenc_engine
    monkeypatch.setattr(plugin, "component_builders", fake_builders)

    sections = plugin.build_engine(
        cfg,
        weights,
        max_cache_length=9,
        precision="bf16",
        verbose=True,
    )

    assert sections["locenc_engine_plan"] == b"LOCENC-TRT"
    assert captured["verbose"] is True
    example_audio_feats = captured["example_args"][0]
    assert example_audio_feats.shape == (9, 4, 64)
    assert example_audio_feats.dtype == "bf16"

    wrapper = captured["wrapper"]
    feat_encoder = wrapper.feat_encoder
    assert feat_encoder.config.kwargs["hidden_size"] == 1024
    assert feat_encoder.config.kwargs["num_hidden_layers"] == 12
    assert feat_encoder.config.kwargs["vocab_size"] == 0
    assert feat_encoder.input_dim == 64
    assert tuple(feat_encoder.loaded) == (
        "encoder.layers.0.weight",
        "in_proj.bias",
        "in_proj.weight",
    )
    assert feat_encoder.strict is True
    assert feat_encoder.dtype == "bf16"
    assert feat_encoder.eval_called is True

    projection = wrapper.enc_to_lm_proj
    assert projection.in_features == 1024
    assert projection.out_features == 2048
    assert tuple(projection.loaded) == ("bias", "weight")
    assert projection.strict is True
    assert projection.dtype == "bf16"
    assert projection.eval_called is True
    assert wrapper.eval_called is True


def test_voxcpm2_audiovae_builder_exports_named_trt_decode_engine(tmp_path, monkeypatch):
    from tensorrt_model_connect.families.voxcpm2 import component_builders
    from tensorrt_model_connect.families.voxcpm2.plugin import plugin

    class FakeTorchModule:
        def __init__(self):
            self.eval_called = False

        def eval(self):
            self.eval_called = True
            return self

    class FakeTensor:
        def __init__(self, shape, dtype):
            self.shape = tuple(shape)
            self.dtype = dtype

    class FakeAudioVAEConfigV2:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeAudioVAEV2(FakeTorchModule):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.loaded = None
            self.strict = None
            self.dtype = None

        def load_state_dict(self, state, *, strict):
            self.loaded = state
            self.strict = strict

        def to(self, *, dtype):
            self.dtype = dtype
            return self

    def _fake_torch_load(path, *, map_location, weights_only):
        assert path == tmp_path / "audiovae.pth"
        assert map_location == "cpu"
        assert weights_only is True
        return {"state_dict": {"decoder.conv.weight": np.array([1.0], dtype=np.float32)}}

    fake_torch = types.SimpleNamespace(
        bfloat16="bf16",
        float16="fp16",
        float32="fp32",
        nn=types.SimpleNamespace(Module=FakeTorchModule),
        load=_fake_torch_load,
        zeros=lambda shape, *, dtype: FakeTensor(shape, dtype),
    )
    fake_voxcpm = types.ModuleType("voxcpm")
    fake_voxcpm_modules = types.ModuleType("voxcpm.modules")
    fake_audiovae = types.ModuleType("voxcpm.modules.audiovae")
    fake_audiovae.AudioVAEConfigV2 = FakeAudioVAEConfigV2
    fake_audiovae.AudioVAEV2 = FakeAudioVAEV2
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "voxcpm", fake_voxcpm)
    monkeypatch.setitem(sys.modules, "voxcpm.modules", fake_voxcpm_modules)
    monkeypatch.setitem(sys.modules, "voxcpm.modules.audiovae", fake_audiovae)

    captured = {}

    def _fake_compile(wrapper, example_args, *, verbose):
        captured["wrapper"] = wrapper
        captured["example_args"] = example_args
        captured["verbose"] = verbose
        return b"AUDIOVAE-TRT"

    monkeypatch.setattr(
        component_builders,
        "_compile_voxcpm2_audio_vae_onnx",
        _fake_compile,
    )

    cfg = _write_raw_voxcpm2_checkpoint(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)

    def _fake_component_builder(component_name):
        def _builder(ctx):
            assert ctx.max_cache_length == 7
            return f"{component_name}-plan".encode("ascii")

        return _builder

    fake_builders = {
        spec.name: _fake_component_builder(spec.name)
        for spec in component_builders.VOXCPM2_COMPONENT_SPECS
    }
    fake_builders["audiovae"] = component_builders.build_audiovae_engine
    monkeypatch.setattr(plugin, "component_builders", fake_builders)

    sections = plugin.build_engine(
        cfg,
        weights,
        max_cache_length=7,
        precision="bf16",
        verbose=True,
    )

    assert sections["audiovae_engine_plan"] == b"AUDIOVAE-TRT"
    assert captured["verbose"] is True
    example_latents = captured["example_args"][0]
    assert example_latents.shape == (28, 64)
    assert example_latents.dtype == "bf16"

    wrapper = captured["wrapper"]
    audio_vae = wrapper.module
    assert audio_vae.config.kwargs["out_sample_rate"] == 48000
    assert tuple(audio_vae.loaded) == ("decoder.conv.weight",)
    np.testing.assert_array_equal(
        audio_vae.loaded["decoder.conv.weight"],
        np.array([1.0], dtype=np.float32),
    )
    assert audio_vae.strict is True
    assert audio_vae.dtype == "bf16"
    assert audio_vae.eval_called is True
    assert wrapper.eval_called is True


def test_voxcpm2_component_context_loads_checkpoint_inputs(tmp_path, monkeypatch):
    from tensorrt_model_connect.families.voxcpm2 import component_builders
    from tensorrt_model_connect.families.voxcpm2.plugin import plugin

    cfg = _write_raw_voxcpm2_checkpoint(tmp_path)
    save_file(
        {"locenc.weight": np.array([1.0, 2.0], dtype=np.float32)},
        tmp_path / "model.safetensors",
    )
    weights = plugin.load_weights(str(tmp_path), cfg)
    sources = weights["_voxcpm2_raw_component_sources"]
    specs = {spec.name: spec for spec in component_builders.VOXCPM2_COMPONENT_SPECS}

    locenc_ctx = component_builders.VoxCPM2ComponentBuildContext(
        spec=specs["locenc"],
        model_dir=tmp_path,
        config=cfg,
        source=sources["locenc"],
        precision="bf16",
        verbose=False,
    )
    np.testing.assert_array_equal(
        locenc_ctx.load_safetensor("locenc.weight"),
        np.array([1.0, 2.0], dtype=np.float32),
    )

    def _fake_torch_load(path, *, map_location, weights_only):
        assert path == tmp_path / "audiovae.pth"
        assert map_location == "cpu"
        assert weights_only is True
        return {"state_dict": {"decoder.weight": np.array([3.0], dtype=np.float32)}}

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(load=_fake_torch_load))
    audiovae_ctx = component_builders.VoxCPM2ComponentBuildContext(
        spec=specs["audiovae"],
        model_dir=tmp_path,
        config=cfg,
        source=sources["audiovae"],
        precision="bf16",
        verbose=False,
    )

    state_dict = audiovae_ctx.load_torch_checkpoint()
    np.testing.assert_array_equal(
        state_dict["decoder.weight"],
        np.array([3.0], dtype=np.float32),
    )


def test_voxcpm2_component_context_loads_stage_scoped_safetensors(tmp_path):
    from tensorrt_model_connect.families.voxcpm2 import component_builders
    from tensorrt_model_connect.families.voxcpm2.plugin import plugin

    cfg = _write_raw_voxcpm2_checkpoint(tmp_path)
    save_file(
        {
            "base_lm.layers.0.weight": np.array([1.0], dtype=np.float32),
            "enc_to_lm_proj.weight": np.array([2.0], dtype=np.float32),
            "feat_decoder.estimator.weight": np.array([3.0], dtype=np.float32),
            "feat_encoder.layers.0.weight": np.array([4.0], dtype=np.float32),
            "residual_lm.layers.0.weight": np.array([5.0], dtype=np.float32),
            "unrelated.weight": np.array([6.0], dtype=np.float32),
        },
        tmp_path / "model.safetensors",
    )
    weights = plugin.load_weights(str(tmp_path), cfg)
    sources = weights["_voxcpm2_raw_component_sources"]
    specs = {spec.name: spec for spec in component_builders.VOXCPM2_COMPONENT_SPECS}

    locenc_ctx = component_builders.VoxCPM2ComponentBuildContext(
        spec=specs["locenc"],
        model_dir=tmp_path,
        config=cfg,
        source=sources["locenc"],
        precision="bf16",
        verbose=False,
    )
    locenc_group = locenc_ctx.load_safetensor_group()

    assert tuple(locenc_group) == (
        "enc_to_lm_proj.weight",
        "feat_encoder.layers.0.weight",
    )
    np.testing.assert_array_equal(
        locenc_group["feat_encoder.layers.0.weight"],
        np.array([4.0], dtype=np.float32),
    )

    ralm_ctx = component_builders.VoxCPM2ComponentBuildContext(
        spec=specs["ralm"],
        model_dir=tmp_path,
        config=cfg,
        source=sources["ralm"],
        precision="bf16",
        verbose=False,
    )
    assert tuple(ralm_ctx.load_safetensor_group()) == (
        "residual_lm.layers.0.weight",
    )

    with pytest.raises(KeyError, match="found no safetensors tensors matching"):
        locenc_ctx.load_safetensor_group(("missing_prefix.",))


def test_voxcpm2_component_preflight_resolves_stage_inputs(tmp_path, monkeypatch):
    from tensorrt_model_connect.families.voxcpm2 import component_builders
    from tensorrt_model_connect.families.voxcpm2.plugin import plugin

    cfg = _write_raw_voxcpm2_checkpoint(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)
    sources = weights["_voxcpm2_raw_component_sources"]
    specs = {spec.name: spec for spec in component_builders.VOXCPM2_COMPONENT_SPECS}

    locdit_ctx = component_builders.VoxCPM2ComponentBuildContext(
        spec=specs["locdit"],
        model_dir=tmp_path,
        config=cfg,
        source=sources["locdit"],
        precision="bf16",
        verbose=False,
    )
    locdit_inputs = component_builders.prepare_component_inputs(locdit_ctx)

    assert locdit_inputs.component == "locdit"
    assert locdit_inputs.engine_section == "locdit_engine_plan"
    assert locdit_inputs.input_artifact == "residual_hidden"
    assert locdit_inputs.output_artifact == "audio_vae_latents"
    assert locdit_inputs.input_tensor.name == "residual_hidden"
    assert locdit_inputs.input_tensor.rank == 2
    assert locdit_inputs.output_tensor.name == "audio_vae_latents"
    assert locdit_inputs.output_tensor.dtype_contract == ("float32", "bfloat16")
    assert locdit_inputs.upstream_inputs == (
        "lm_hidden",
        "residual_hidden",
        "feat_cond",
        "cfg_value",
        "inference_timesteps",
    )
    assert locdit_inputs.upstream_outputs == ("audio_vae_latents",)
    assert locdit_inputs.required_side_inputs == ("lm_hidden",)
    assert locdit_inputs.required_control_inputs == (
        "cfg_value",
        "inference_timesteps",
    )
    assert [
        module.describe() for module in locdit_inputs.upstream_modules
    ] == [
        "voxcpm.modules.locdit.UnifiedCFM(feat_decoder.)",
        "voxcpm.modules.locdit.VoxCPMLocDiTV2(feat_decoder.estimator.)",
        "torch.nn.Linear(lm_to_dit_proj., res_to_dit_proj.)",
    ]
    assert locdit_inputs.checkpoint_kind == "safetensors"
    assert locdit_inputs.weight_paths == (tmp_path / "model.safetensors",)
    assert locdit_inputs.asset_paths == ()
    assert locdit_inputs.state_dict_keys == (
        "feat_decoder.estimator.weight",
        "lm_to_dit_proj.weight",
        "res_to_dit_proj.weight",
    )
    assert locdit_inputs.config_values["dit_config"]["hidden_dim"] == 1024

    def _fake_torch_load(path, *, map_location, weights_only):
        assert path == tmp_path / "audiovae.pth"
        assert map_location == "cpu"
        assert weights_only is True
        return {
            "state_dict": {
                "decoder.conv.weight": np.array([1.0], dtype=np.float32),
                "quantizer.embedding.weight": np.array([2.0], dtype=np.float32),
            }
        }

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(load=_fake_torch_load))
    audiovae_ctx = component_builders.VoxCPM2ComponentBuildContext(
        spec=specs["audiovae"],
        model_dir=tmp_path,
        config=cfg,
        source=sources["audiovae"],
        precision="bf16",
        verbose=False,
    )
    audiovae_inputs = component_builders.prepare_component_inputs(audiovae_ctx)

    assert audiovae_inputs.component == "audiovae"
    assert audiovae_inputs.engine_section == "audiovae_engine_plan"
    assert audiovae_inputs.checkpoint_kind == "torch"
    assert audiovae_inputs.weight_paths == (tmp_path / "audiovae.pth",)
    assert audiovae_inputs.state_dict_keys == (
        "decoder.conv.weight",
        "quantizer.embedding.weight",
    )
    assert audiovae_inputs.config_values["audio_vae_config"]["out_sample_rate"] == 48000


def test_voxcpm2_build_engine_packages_prebuilt_component_plans(tmp_path):
    from tensorrt_model_connect.families.voxcpm2.plugin import plugin

    cfg = _voxcpm2_config(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)
    plans = {
        "locenc.plan": b"LOCENC",
        "tslm.engine": b"TSLM",
        "ralm.trtplan": b"RALM",
        "locdit_engine_plan.plan": b"LOCDIT",
        "audiovae_engine_plan": b"AUDIOVAE",
    }
    for filename, data in plans.items():
        (tmp_path / filename).write_bytes(data)

    sections = plugin.build_engine(cfg, weights, max_cache_length=16)

    assert sections == {
        "locenc_engine_plan": b"LOCENC",
        "tslm_engine_plan": b"TSLM",
        "ralm_engine_plan": b"RALM",
        "locdit_engine_plan": b"LOCDIT",
        "audiovae_engine_plan": b"AUDIOVAE",
    }


def test_voxcpm2_build_engine_reports_partial_prebuilt_plan_set(tmp_path):
    from tensorrt_model_connect.families.voxcpm2.plugin import plugin

    cfg = _voxcpm2_config(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)
    (tmp_path / "locenc.plan").write_bytes(b"LOCENC")
    (tmp_path / "tslm.plan").write_bytes(b"TSLM")

    with pytest.raises(NotImplementedError) as error:
        plugin.build_engine(cfg, weights, max_cache_length=16)

    message = str(error.value)
    assert "missing artifacts for ralm, locdit, audiovae" in message
    assert "ralm.plan" in message
    assert "audiovae.trtplan" in message


def test_audio_voxcpm2_schema_defaults_and_validators():
    clear_for_testing()
    try:
        load_all()
        schema = lookup("audio_voxcpm2")
        assert schema is not None
        fields = {field.name: field for field in schema.fields}

        assert fields["cfg_value"].default == 2.0
        assert fields["inference_timesteps"].default == 10
        assert fields["normalize"].default is True
        assert fields["denoise"].default is True
        assert fields["retry_badcase"].default is True
        assert fields["retry_badcase_max_times"].default == 3
        assert fields["retry_badcase_ratio_threshold"].default == 6.0
        assert fields["seed"].default == -1
        assert fields["cfg_value"].validator(0.0)
        assert not fields["cfg_value"].validator(-0.1)
        assert fields["inference_timesteps"].validator(10)
        assert not fields["inference_timesteps"].validator(True)
    finally:
        clear_for_testing()
