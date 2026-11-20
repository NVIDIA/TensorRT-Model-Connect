# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks of the hubert family against Transformers' own HubertForCTC.

The CPU tests run the family's dataflow (``model._forward``) with each
TensorRT helper replaced by the matching torch operation, on small randomly
initialised checkpoints written the way a user's checkpoint is read. They
establish layer order, norm placement and epsilons, the positional trim and the
weight mapping for both encoder layouts, not TensorRT numerics; the GPU tests
at the end build real engines from the same checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from families.hubert import config as config_module, graph as g, model

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

_VOCAB = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "A": 5, "B": 6, "C": 7}
_PRE = {"feature_size": 1, "padding_value": 0.0, "padding_side": "right",
        "sampling_rate": 16000, "do_normalize": True, "return_attention_mask": True}
_TOKENIZER = {"pad_token": "<pad>", "unk_token": "<unk>", "bos_token": "<s>",
              "eos_token": "</s>", "word_delimiter_token": "|", "do_lower_case": False}

# The official facebook/hubert-large-ls960-ft config.json, as published.
_OFFICIAL = {
    "architectures": ["HubertForCTC"], "model_type": "hubert", "conv_bias": True,
    "conv_dim": [512] * 7, "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
    "conv_stride": [5, 2, 2, 2, 2, 2, 2], "do_stable_layer_norm": True,
    "feat_extract_activation": "gelu", "feat_extract_norm": "layer", "hidden_act": "gelu",
    "hidden_size": 1024, "intermediate_size": 4096, "layer_norm_eps": 1e-05,
    "num_attention_heads": 16, "num_conv_pos_embedding_groups": 16,
    "num_conv_pos_embeddings": 128, "num_feat_extract_layers": 7, "num_hidden_layers": 24,
    "pad_token_id": 0, "vocab_size": 32, "mask_time_prob": 0.05,
}
_OFFICIAL_VOCAB = {token: index for index, token in enumerate(
    ["<pad>", "<s>", "</s>", "<unk>", "|", *"ETAONIHSRDLUMWCFGYPBVK'XJQZ"])}


def _reference(*, stable=True, norm="layer", bias=True, proj_norm=True, eps=1e-5):
    """A small random HubertForCTC whose every parameter, norms included, is drawn."""
    torch.manual_seed(23)
    cfg = transformers.HubertConfig(
        hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=24,
        conv_dim=(8, 8, 12), conv_kernel=(5, 3, 2), conv_stride=(3, 2, 2), conv_bias=bias,
        feat_extract_norm=norm, do_stable_layer_norm=stable, feat_proj_layer_norm=proj_norm,
        num_conv_pos_embeddings=4, num_conv_pos_embedding_groups=2, layer_norm_eps=eps,
        hidden_dropout=0.0, attention_dropout=0.0, activation_dropout=0.0, layerdrop=0.0,
        feat_proj_dropout=0.0, final_dropout=0.0, mask_time_prob=0.05, vocab_size=len(_VOCAB),
    )
    cfg.architectures = ["HubertForCTC"]
    reference = transformers.HubertForCTC(cfg).eval()
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.4)
    return cfg, reference


def _checkpoint(directory: Path, cfg, reference, *, tokenizer=_TOKENIZER) -> Path:
    cfg.to_json_file(directory / "config.json")
    (directory / "preprocessor_config.json").write_text(json.dumps(_PRE))
    (directory / "vocab.json").write_text(json.dumps(_VOCAB))
    (directory / "tokenizer_config.json").write_text(json.dumps(tokenizer))
    torch.save(reference.state_dict(), directory / "pytorch_model.bin")
    return directory


def _resolved(directory: Path):
    weights, shapes = model._load(directory)
    return weights, config_module.resolve(directory, shapes)


def _waveform(samples: int):
    time = torch.arange(samples, dtype=torch.float32) / 16000.0
    return (0.5 * torch.sin(2 * torch.pi * 220.0 * time) * torch.cos(2 * torch.pi * 3.0 * time)
            + 0.1 * torch.sin(2 * torch.pi * 1375.0 * time)).reshape(1, -1)


@pytest.fixture
def cpu_graph(monkeypatch):
    """Swap every TensorRT helper for the torch operation it builds."""
    functional = torch.nn.functional

    def tensor(value):
        return value if isinstance(value, torch.Tensor) else torch.from_numpy(
            np.asarray(value, dtype=np.float32))

    def norm(network, inp, gamma, beta, eps, axis, *, param_axis=None):
        out = functional.layer_norm(inp.movedim(axis, -1), (inp.shape[axis],), eps=eps)
        out = out.movedim(-1, axis)
        param_axis = inp.ndim - 1 if param_axis is None else param_axis
        shape = [1] * inp.ndim
        shape[param_axis] = len(gamma)
        return out * tensor(gamma).reshape(shape) + tensor(beta).reshape(shape)

    def conv(network, inp, weight, *, stride=1, padding=0, groups=1, bias=None):
        return functional.conv1d(inp.squeeze(2), tensor(weight),
                                 None if bias is None else tensor(bias),
                                 stride=stride, padding=padding, groups=groups).unsqueeze(2)

    def attention(network, query, key, value, *, num_heads, head_dim):
        def heads(inp):
            return inp.reshape(-1, num_heads, head_dim).transpose(0, 1)
        out = functional.scaled_dot_product_attention(heads(query), heads(key), heads(value))
        return out.transpose(0, 1).reshape(-1, num_heads * head_dim)

    monkeypatch.setattr(model.g, "add_norm", norm)
    monkeypatch.setattr(model.g, "add_conv1d", conv)
    monkeypatch.setattr(model.g, "add_gelu", lambda net, x: functional.gelu(x, approximate="none"))
    monkeypatch.setattr(model.g, "add_sum", lambda net, a, b: a + b)
    monkeypatch.setattr(model.g, "add_linear", lambda net, x, w, b=None: functional.linear(
        x, tensor(w), None if b is None else tensor(b)))
    monkeypatch.setattr(model.g, "add_attention", attention)
    monkeypatch.setattr(model.g, "frames_to_rows",
                        lambda net, x, width: x.squeeze(2).squeeze(0).transpose(0, 1))
    monkeypatch.setattr(model.g, "rows_to_frames",
                        lambda net, rows, width: rows.transpose(0, 1)[None, :, None, :])
    monkeypatch.setattr(model.g, "trim_like",
                        lambda net, frames, like: frames[..., : like.shape[-1]])


_LAYOUTS = [
    # (stable, feat_extract_norm, conv_bias, feat_proj_layer_norm, layer_norm_eps)
    pytest.param(True, "layer", True, True, 1e-5, id="large-layout"),
    pytest.param(False, "group", False, True, 1e-5, id="base-layout"),
    pytest.param(False, "group", False, False, 1e-5, id="base-without-projection-norm"),
    pytest.param(True, "layer", False, False, 1e-5, id="stable-without-bias"),
    pytest.param(True, "layer", True, True, 0.1, id="large-layout-wide-epsilon"),
    pytest.param(False, "group", True, True, 0.1, id="base-layout-wide-epsilon"),
]


@pytest.mark.parametrize("stable, norm, bias, proj_norm, eps", _LAYOUTS)
def test_logits_match_the_reference_on_cpu(tmp_path, cpu_graph, stable, norm, bias, proj_norm,
                                            eps):
    cfg, reference = _reference(stable=stable, norm=norm, bias=bias, proj_norm=proj_norm, eps=eps)
    weights, resolved = _resolved(_checkpoint(tmp_path, cfg, reference))
    waveform = _waveform(1600)
    with torch.no_grad():
        expected = reference(waveform).logits[0]
        actual = model._forward(None, waveform[:, None, None, :], weights, resolved)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_the_trim_side_is_seen_by_the_cpu_check(tmp_path, cpu_graph, monkeypatch):
    # Control: dropping the first positional frame instead of the last is the
    # error the comment in _positional warns about; the parity check sees it.
    cfg, reference = _reference()
    weights, resolved = _resolved(_checkpoint(tmp_path, cfg, reference))
    monkeypatch.setattr(model.g, "trim_like",
                        lambda net, frames, like: frames[..., 1: like.shape[-1] + 1])
    waveform = _waveform(1600)
    with torch.no_grad():
        expected = reference(waveform).logits[0]
        actual = model._forward(None, waveform[:, None, None, :], weights, resolved)
    assert (actual - expected).abs().max() > 1e-2


def test_the_projection_norm_is_seen_by_the_cpu_check(tmp_path, cpu_graph):
    # Control: building the large layout without its feature-projection norm
    # (the wav2vec2-style assumption that it is always there, inverted) moves
    # the logits far outside the parity tolerance.
    cfg, reference = _reference()
    weights, resolved = _resolved(_checkpoint(tmp_path, cfg, reference))
    waveform = _waveform(1600)
    with torch.no_grad():
        expected = reference(waveform).logits[0]
        actual = model._forward(None, waveform[:, None, None, :], weights,
                                {**resolved, "feat_proj_layer_norm": False})
    assert (actual - expected).abs().max() > 1e-2


def test_weight_normed_kernel_reproduces_the_reference_formula():
    rng = np.random.default_rng(0)
    weight_v = rng.normal(size=(8, 4, 6)).astype(np.float32)
    weight_g = rng.normal(size=(1, 1, 6)).astype(np.float32)
    kernel = g.weight_normed_kernel(weight_g, weight_v)
    assert kernel.shape == weight_v.shape
    # Each kernel tap ends up with the length weight_g states for it.
    lengths = np.sqrt((kernel.astype(np.float64) ** 2).sum(axis=(0, 1)))
    assert np.abs(lengths - np.abs(weight_g).reshape(-1)).max() < 1e-5
    # Controls: weight_v as is, or the norm over the other axes, give other taps.
    raw = np.sqrt((weight_v.astype(np.float64) ** 2).sum(axis=(0, 1)))
    assert np.abs(raw - np.abs(weight_g).reshape(-1)).max() > 1e-2
    wrong_axis = weight_v * (weight_g.reshape(1, 1, -1) / np.sqrt(
        (weight_v.astype(np.float64) ** 2).sum(axis=(1, 2), keepdims=True)))
    assert np.abs(np.asarray(wrong_axis) - kernel).max() > 1e-2


def test_both_spellings_of_the_positional_kernel_load_alike(tmp_path):
    cfg, reference = _reference()
    current = _checkpoint(tmp_path, cfg, reference)
    weights, _ = model._load(current)
    state = reference.state_dict()
    assert any(name.endswith("parametrizations.weight.original0") for name in state)
    legacy = {name.replace("parametrizations.weight.original0", "weight_g")
                  .replace("parametrizations.weight.original1", "weight_v"): value
              for name, value in state.items()}
    older = tmp_path / "legacy"
    older.mkdir()
    torch.save(legacy, older / "pytorch_model.bin")
    legacy_weights, _ = model._load(older)
    assert legacy_weights.keys() == weights.keys()
    for name, value in weights.items():
        np.testing.assert_array_equal(legacy_weights[name], value)


def test_a_kernel_named_both_ways_is_refused(tmp_path):
    torch.save({"hubert.encoder.pos_conv_embed.conv.weight_g": torch.ones(1, 1, 4),
                "hubert.encoder.pos_conv_embed.conv.parametrizations.weight.original0":
                    torch.ones(1, 1, 4)}, tmp_path / "pytorch_model.bin")
    with pytest.raises(ValueError, match="twice"):
        model._load(tmp_path)


def test_bin_loader_uses_restricted_cpu_loading(tmp_path, monkeypatch):
    (tmp_path / "pytorch_model.bin").touch()
    observed = {}

    def load(path, **kwargs):
        observed.update(kwargs)
        return {"lm_head.bias": torch.ones(1)}

    monkeypatch.setattr(torch, "load", load)
    model._load(tmp_path)
    assert observed == {"map_location": "cpu", "weights_only": True}


@pytest.mark.parametrize("state", [{}, {"lm_head.weight": "invalid"}, [torch.ones(1)]])
def test_bin_loader_refuses_anything_but_a_tensor_state(tmp_path, state):
    torch.save(state, tmp_path / "pytorch_model.bin")
    with pytest.raises(ValueError, match="nonempty tensor state"):
        model._load(tmp_path)


def test_a_missing_checkpoint_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="pytorch_model.bin"):
        model._load(tmp_path)


def _official(tmp_path: Path, **overrides) -> Path:
    cfg = {**_OFFICIAL, **overrides}
    for key in [key for key, value in overrides.items() if value is None]:
        del cfg[key]
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(_PRE))
    (tmp_path / "vocab.json").write_text(json.dumps(_OFFICIAL_VOCAB))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(_TOKENIZER))
    return tmp_path


def _official_shapes(tmp_path: Path, **overrides):
    """The tensor inventory a config resolves to, plus the training-only mask vector."""
    directory = _official(tmp_path, **overrides)
    cfg = json.loads((directory / "config.json").read_text())
    resolved = {
        "hidden_size": cfg["hidden_size"], "intermediate_size": cfg["intermediate_size"],
        "conv_dim": cfg["conv_dim"], "conv_kernel": cfg["conv_kernel"],
        "conv_bias": cfg.get("conv_bias", False),
        "feat_extract_norm": cfg.get("feat_extract_norm", "group"),
        "feat_proj_layer_norm": cfg.get("feat_proj_layer_norm", True),
        "pos_kernel": cfg["num_conv_pos_embeddings"],
        "pos_groups": cfg["num_conv_pos_embedding_groups"],
        "num_hidden_layers": cfg["num_hidden_layers"], "vocab_size": cfg["vocab_size"],
    }
    shapes = dict(config_module.expected_tensors(resolved))
    shapes["hubert.masked_spec_embed"] = (cfg["hidden_size"],)
    return directory, shapes


def test_the_official_large_checkpoint_resolves(tmp_path):
    directory, shapes = _official_shapes(tmp_path)
    # Measured on the published checkpoint (safetensors header of its Hub
    # conversion): 424 tensors and 315,471,520 parameters, all F32.
    assert len(shapes) == 424
    assert sum(int(np.prod(shape)) for shape in shapes.values()) == 315_471_520
    cfg = config_module.resolve(directory, shapes)
    assert cfg["do_stable_layer_norm"] is True
    assert cfg["feat_extract_norm"] == "layer"
    assert cfg["conv_bias"] is True
    # Omitted in the official config; HubertConfig's default applies.
    assert cfg["feat_proj_layer_norm"] is True
    assert cfg["tokens"][:5] == ["<pad>", "<s>", "</s>", "<unk>", "|"]
    assert cfg["blank_token_id"] == 0
    assert cfg["word_delimiter_token"] == "|"
    assert cfg["sample_rate"] == 16000 and cfg["do_normalize"] is True


@pytest.mark.parametrize("overrides, fragment", [
    ({"conv_pos_batch_norm": True}, "conv_pos_batch_norm"),
    ({"add_adapter": True}, "adapters"),
    ({"adapter_attn_dim": 16}, "adapters"),
    ({"hidden_act": "gelu_new"}, "exact GELU"),
    ({"feat_extract_activation": "relu"}, "exact GELU"),
    ({"num_conv_pos_embeddings": 127}, "even num_conv_pos_embeddings"),
    ({"architectures": ["HubertForSequenceClassification"]}, "HubertForCTC"),
    ({"model_type": "wav2vec2"}, "model_type"),
    ({"feat_extract_norm": "batch"}, "feat_extract_norm"),
    ({"do_stable_layer_norm": "true"}, "boolean do_stable_layer_norm"),
    ({"conv_bias": "true"}, "boolean conv_bias"),
    ({"feat_proj_layer_norm": 1}, "boolean feat_proj_layer_norm"),
    ({"pad_token_id": 1}, "CTC blank"),
    ({"vocab_size": 33}, "vocab.json holds 32"),
])
def test_resolve_refuses_the_variants_it_does_not_build(tmp_path, overrides, fragment):
    # Each of these loads in Transformers and produces plausible output, so the
    # refusal is what stands between them and a silently different graph.
    directory, shapes = _official_shapes(tmp_path)
    _official(tmp_path, **overrides)
    with pytest.raises(NotImplementedError, match=fragment):
        config_module.resolve(directory, shapes)


def test_resolve_refuses_a_checkpoint_missing_a_tensor(tmp_path):
    directory, shapes = _official_shapes(tmp_path)
    del shapes["hubert.feature_projection.layer_norm.weight"]
    with pytest.raises(NotImplementedError, match="missing hubert.feature_projection"):
        config_module.resolve(directory, shapes)


def test_resolve_refuses_a_projection_norm_the_config_turns_off(tmp_path):
    # The weights carry the norm but the config says it is not applied; the
    # reference would ignore the tensors, so the mismatch is refused, not guessed.
    directory, shapes = _official_shapes(tmp_path)
    _official(tmp_path, feat_proj_layer_norm=False)
    with pytest.raises(NotImplementedError, match="does not read"):
        config_module.resolve(directory, shapes)


def test_resolve_refuses_a_tensor_of_the_wrong_shape(tmp_path):
    directory, shapes = _official_shapes(tmp_path)
    shapes["lm_head.weight"] = (32, 768)
    with pytest.raises(NotImplementedError, match="lm_head.weight"):
        config_module.resolve(directory, shapes)


def test_resolve_refuses_an_unread_layer(tmp_path):
    directory, shapes = _official_shapes(tmp_path)
    shapes["hubert.encoder.pos_conv_embed.batch_norm.weight"] = (1024,)
    with pytest.raises(NotImplementedError, match="batch_norm"):
        config_module.resolve(directory, shapes)


def test_resolve_reads_the_pad_token_as_an_added_token(tmp_path):
    directory, shapes = _official_shapes(tmp_path)
    (directory / "tokenizer_config.json").write_text(json.dumps(
        {**_TOKENIZER, "pad_token": {"content": "<pad>", "special": True}}))
    assert config_module.resolve(directory, shapes)["blank_token_id"] == 0


def test_resolve_reads_the_pad_token_as_the_reference_tokenizer_merges_it(tmp_path):
    # Without added_tokens_decoder, Transformers lets special_tokens_map.json
    # override tokenizer_config.json, so the blank follows the map.
    directory, shapes = _official_shapes(tmp_path, pad_token_id=1)
    (directory / "special_tokens_map.json").write_text(json.dumps({"pad_token": "<s>"}))
    assert config_module.resolve(directory, shapes)["blank_token_id"] == 1


def test_resolve_ignores_the_special_tokens_map_when_the_decoder_is_recorded(tmp_path):
    # With added_tokens_decoder, Transformers does not read special_tokens_map.json.
    directory, shapes = _official_shapes(tmp_path)
    (directory / "tokenizer_config.json").write_text(json.dumps(
        {**_TOKENIZER, "added_tokens_decoder": {"0": {"content": "<pad>", "special": True}}}))
    (directory / "special_tokens_map.json").write_text(json.dumps({"pad_token": "<s>"}))
    assert config_module.resolve(directory, shapes)["blank_token_id"] == 0


def test_resolve_refuses_a_special_tokens_map_that_moves_the_blank(tmp_path):
    # Control: the map names another pad token than config.pad_token_id; the
    # reference tokenizer would drop that token, so the checkpoint is refused
    # instead of decoded with tokenizer_config.json's value.
    directory, shapes = _official_shapes(tmp_path)
    (directory / "special_tokens_map.json").write_text(json.dumps({"pad_token": "<s>"}))
    with pytest.raises(NotImplementedError, match="CTC blank"):
        config_module.resolve(directory, shapes)


def test_resolve_defaults_the_layout_as_the_reference_does(tmp_path):
    # Without the keys, HubertConfig gives the base layout: group norm, post-norm
    # blocks, no convolution bias, a projection norm.
    directory, shapes = _official_shapes(
        tmp_path, feat_extract_norm=None, do_stable_layer_norm=None, conv_bias=None)
    cfg = config_module.resolve(directory, shapes)
    assert (cfg["feat_extract_norm"], cfg["do_stable_layer_norm"], cfg["conv_bias"],
            cfg["feat_proj_layer_norm"]) == ("group", False, False, True)


@pytest.mark.parametrize("overrides, fragment", [
    ({"do_lower_case": True}, "do_lower_case"),
    ({"replace_word_delimiter_char": "_"}, "replace_word_delimiter_char"),
    ({"clean_up_tokenization_spaces": True}, "clean_up_tokenization_spaces"),
    ({"tokenizer_class": "Wav2Vec2PhonemeCTCTokenizer"}, "Wav2Vec2CTCTokenizer"),
])
def test_resolve_refuses_decoding_the_runtime_does_not_reproduce(tmp_path, overrides, fragment):
    # Each setting changes the text the reference tokenizer's decode returns.
    directory, shapes = _official_shapes(tmp_path)
    (directory / "tokenizer_config.json").write_text(json.dumps({**_TOKENIZER, **overrides}))
    with pytest.raises(NotImplementedError, match=fragment):
        config_module.resolve(directory, shapes)


def test_resolve_accepts_the_default_ctc_tokenizer_stated_explicitly(tmp_path):
    directory, shapes = _official_shapes(tmp_path)
    (directory / "tokenizer_config.json").write_text(json.dumps(
        {**_TOKENIZER, "tokenizer_class": "Wav2Vec2CTCTokenizer", "replace_word_delimiter_char": " ",
         "clean_up_tokenization_spaces": False}))
    assert config_module.resolve(directory, shapes)["word_delimiter_token"] == "|"


def test_sample_range_is_a_real_range():
    assert config_module.MIN_SAMPLES < config_module.OPT_SAMPLES < config_module.MAX_SAMPLES
    # The profile covers a useful clip: 30 s at 16 kHz.
    assert config_module.MAX_SAMPLES >= 30 * 16000


class _Writer:
    def set_header(self, **kwargs):
        raise AssertionError("the build should have refused before writing")


def _request(**overrides):
    fields = dict(model_dir=Path("/nonexistent"), output_path=Path("/nonexistent/out.bundle"),
                  family="hubert", task="speech_transcription", precision="fp32", backend="trt",
                  max_sequence_length=None, image_height=None, image_width=None,
                  video_num_frames=None, max_batch_size=1, tensor_parallel_size=1,
                  context_parallel_size=1, quantization=None, fp32_layers=(),
                  dynamic_kv_cache=False, verbose=False, mtp_seq_len=None)
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize("overrides, error, fragment", [
    ({"task": "speech_translation"}, ValueError, "task=speech_transcription"),
    ({"backend": "trt_rtx"}, ValueError, "backend=trt"),
    ({"precision": "bf16"}, NotImplementedError, "fp32 or fp16, not bf16"),
    ({"max_batch_size": 2}, NotImplementedError, "max_batch_size"),
    ({"tensor_parallel_size": 2}, NotImplementedError, "tensor parallelism"),
    ({"context_parallel_size": 2}, NotImplementedError, "context parallelism"),
    ({"dynamic_kv_cache": True}, NotImplementedError, "dynamic_kv_cache"),
    ({"video_num_frames": 8}, NotImplementedError, "video_num_frames"),
    ({"image_height": 224}, NotImplementedError, "image size"),
    ({"quantization": "fp8"}, NotImplementedError, "quantization"),
    ({"fp32_layers": (1,)}, NotImplementedError, "mixed-precision"),
    ({"mtp_seq_len": 2}, NotImplementedError, "MTP"),
    ({"max_sequence_length": 100}, ValueError, "at least 4000 samples"),
])
def test_build_refuses_options_before_reading_the_checkpoint(overrides, error, fragment):
    with pytest.raises(error, match=fragment):
        model.build(_request(**overrides), _Writer())


def test_unsupported_precision_is_refused_by_the_engine_builder():
    with pytest.raises(NotImplementedError, match="fp32 or fp16, not bf16"):
        model.build_engine({}, {}, max_samples=config_module.MIN_SAMPLES, precision="bf16")


def _engine_logits(plan, waveform):
    trt = pytest.importorskip("tensorrt")
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    audio = waveform.reshape(1, 1, 1, -1).float().contiguous().cuda()
    context.set_input_shape("audio", tuple(audio.shape))
    context.set_tensor_address("audio", audio.data_ptr())
    logits = torch.empty(tuple(context.get_tensor_shape("logits")), device="cuda")
    context.set_tensor_address("logits", logits.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return engine, logits.cpu()


@pytest.mark.parametrize("stable, norm", [(True, "layer"), (False, "group")])
def test_fp32_engine_matches_the_reference(tmp_path, stable, norm):
    if not torch.cuda.is_available():
        pytest.skip("engine build needs a GPU")
    cfg, reference = _reference(stable=stable, norm=norm, bias=stable)
    weights, resolved = _resolved(_checkpoint(tmp_path, cfg, reference))
    plan = model.build_engine(weights, resolved, max_samples=config_module.MIN_SAMPLES * 4)
    for samples in (config_module.MIN_SAMPLES, config_module.MIN_SAMPLES * 3 + 7):
        waveform = _waveform(samples)
        engine, logits = _engine_logits(plan, waveform)
        with torch.no_grad():
            expected = reference(waveform).logits[0]
        assert engine.get_tensor_dtype("logits") == pytest.importorskip("tensorrt").float32
        assert logits.shape == expected.shape
        torch.testing.assert_close(logits, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("stable, norm", [(True, "layer"), (False, "group")])
def test_fp16_engine_follows_the_reference(tmp_path, stable, norm):
    if not torch.cuda.is_available():
        pytest.skip("engine build needs a GPU")
    trt = pytest.importorskip("tensorrt")
    cfg, reference = _reference(stable=stable, norm=norm, bias=stable)
    weights, resolved = _resolved(_checkpoint(tmp_path, cfg, reference))

    # Every reduction (the norms' mean and variance) runs in fp32 inside the
    # fp16 graph and every matmul in fp16; the engine still reads and writes fp32.
    seen = {}
    original = trt.Builder.build_serialized_network

    def capture(builder, network, config):
        layers = [network.get_layer(index) for index in range(network.num_layers)]
        seen["reduce"] = {layer.get_output(0).dtype for layer in layers
                          if layer.type == trt.LayerType.REDUCE}
        seen["matmul"] = {layer.get_output(0).dtype for layer in layers
                          if layer.type == trt.LayerType.MATRIX_MULTIPLY}
        return original(builder, network, config)

    trt.Builder.build_serialized_network = capture
    try:
        plan = model.build_engine(weights, resolved, max_samples=config_module.MIN_SAMPLES * 2,
                                  precision="fp16")
    finally:
        trt.Builder.build_serialized_network = original
    assert seen["reduce"] == {trt.float32}
    assert seen["matmul"] == {trt.float16}

    waveform = _waveform(config_module.MIN_SAMPLES)
    engine, logits = _engine_logits(plan, waveform)
    assert engine.get_tensor_dtype("audio") == trt.float32
    assert engine.get_tensor_dtype("logits") == trt.float32
    with torch.no_grad():
        expected = reference(waveform).logits[0]
    assert torch.isfinite(logits).all()
    assert torch.nn.functional.cosine_similarity(
        logits.reshape(-1), expected.reshape(-1), dim=0) > 0.999
