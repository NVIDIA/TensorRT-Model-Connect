# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference checks against Transformers' own WavLM.

The CPU checks replace the TensorRT graph operations with torch operations of
the same meaning and compare the family's dataflow - the bucket table, the
folded gate, the norm placement - with randomly initialised Transformers
models. The GPU checks build real engines from the same small models.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from families.wavlm import config as config_module, model

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from families.wavlm.convert_checkpoint import convert  # noqa: E402


def _reference(*, stable=True, norm="layer", bias=False, buckets=8, distance=6):
    torch.manual_seed(29)
    cfg = transformers.WavLMConfig(
        hidden_size=8, num_hidden_layers=2, num_attention_heads=2, intermediate_size=12,
        conv_dim=(4, 8), conv_kernel=(3, 3), conv_stride=(2, 2), conv_bias=bias,
        feat_extract_norm=norm, do_stable_layer_norm=stable,
        num_conv_pos_embeddings=4, num_conv_pos_embedding_groups=2,
        num_buckets=buckets, max_bucket_distance=distance,
        hidden_dropout=0, attention_dropout=0, activation_dropout=0, layerdrop=0,
        feat_proj_dropout=0, final_dropout=0, apply_spec_augment=False, vocab_size=4,
        pad_token_id=3,
    )
    reference = transformers.WavLMForCTC(cfg).eval()
    with torch.no_grad():
        # Random initialisation leaves the gate constant at one, the gate
        # projection near zero and the bucket embedding small; spread all three
        # so that a misplaced gate or a wrong bucket moves the output well
        # beyond float noise.
        for layer in reference.wavlm.encoder.layers:
            layer.attention.gru_rel_pos_const.uniform_(0.5, 2.0)
            layer.attention.gru_rel_pos_linear.weight.normal_(0.0, 1.0)
            layer.attention.gru_rel_pos_linear.bias.normal_(0.0, 1.0)
        reference.wavlm.encoder.layers[0].attention.rel_attn_embed.weight.normal_(0.0, 2.0)
    return cfg, reference


def _resolved(tmp_path, cfg, reference):
    cfg.architectures = ["WavLMForCTC"]
    cfg.to_json_file(tmp_path / "config.json")
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({"sampling_rate": 16000}))
    (tmp_path / "vocab.json").write_text(json.dumps({"|": 0, "A": 1, "B": 2, "[PAD]": 3}))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"pad_token": "[PAD]"}))
    weights = {name: value.detach().numpy() for name, value in reference.state_dict().items()}
    # Current torch stores the weight-normed kernel as parametrizations; the
    # released checkpoints name the same two tensors weight_g and weight_v.
    for old, new in (("parametrizations.weight.original0", "weight_g"),
                     ("parametrizations.weight.original1", "weight_v")):
        for name in [name for name in weights if name.endswith(old)]:
            weights[name[: -len(old)] + new] = weights.pop(name)
    shapes = {name: value.shape for name, value in weights.items()}
    return config_module.resolve(tmp_path, shapes), weights


@pytest.fixture
def cpu_graph(monkeypatch):
    functional = torch.nn.functional

    def tensor(value):
        return value if isinstance(value, torch.Tensor) else torch.from_numpy(np.asarray(value))

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

    def position_bias(network, rows, table, max_distance):
        positions = torch.arange(rows.shape[0])
        offset = (positions[None, :] - positions[:, None]).clamp(-max_distance, max_distance)
        return tensor(table)[offset + max_distance].permute(2, 0, 1)

    def head_gate(network, rows, weight, bias, scale, *, num_heads, head_dim):
        heads = rows.reshape(-1, num_heads, head_dim).transpose(0, 1)
        gates = torch.sigmoid(functional.linear(heads, tensor(weight), tensor(bias)))
        first, second = gates[..., :1], gates[..., 1:]
        return first * (second * tensor(scale).reshape(num_heads, 1, 1) - 1.0) + 2.0

    def attention(network, query, key, value, bias, *, num_heads, head_dim):
        def heads(inp):
            return inp.reshape(-1, num_heads, head_dim).transpose(0, 1)
        scores = heads(query) @ heads(key).transpose(1, 2) * head_dim ** -0.5 + bias
        out = torch.softmax(scores, dim=-1) @ heads(value)
        return out.transpose(0, 1).reshape(-1, num_heads * head_dim)

    monkeypatch.setattr(model.g, "add_norm", norm)
    monkeypatch.setattr(model.g, "add_conv1d", conv)
    monkeypatch.setattr(model.g, "add_gelu", lambda net, x: functional.gelu(x, approximate="none"))
    monkeypatch.setattr(model.g, "add_sum", lambda net, a, b: a + b)
    monkeypatch.setattr(model.g, "add_prod", lambda net, a, b: a * b)
    monkeypatch.setattr(model.g, "add_linear", lambda net, x, w, b=None: functional.linear(
        x, tensor(w), None if b is None else tensor(b)))
    monkeypatch.setattr(model.g, "add_position_bias", position_bias)
    monkeypatch.setattr(model.g, "add_head_gate", head_gate)
    monkeypatch.setattr(model.g, "add_attention", attention)


def test_buckets_match_the_reference_for_the_released_geometry():
    attention = transformers.models.wavlm.modeling_wavlm.WavLMAttention(
        embed_dim=16, num_heads=2, num_buckets=320, max_distance=800)
    offsets = torch.arange(-4000, 4001)
    expected = attention._relative_positions_bucket(offsets).numpy()
    actual = model.relative_buckets(offsets.numpy(), {"num_buckets": 320, "max_distance": 800})
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("frames", [5, 30])
def test_position_bias_table_matches_compute_bias(tmp_path, cpu_graph, frames):
    cfg, reference = _reference()
    resolved, weights = _resolved(tmp_path, cfg, reference)
    attention = reference.wavlm.encoder.layers[0].attention
    with torch.no_grad():
        expected = attention.compute_bias(frames, frames)
    table = model.position_bias_table(weights, resolved)
    actual = model.g.add_position_bias(None, torch.zeros(frames, 8), table,
                                       resolved["max_distance"])
    # 30 frames reach offsets far beyond max_bucket_distance=6, so the clamp is
    # exercised; 5 frames stay inside the exact range.
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("norm", ["group", "layer"])
@pytest.mark.parametrize("bias", [False, True])
def test_feature_encoder_matches_official_cpu(tmp_path, cpu_graph, norm, bias):
    cfg, reference = _reference(norm=norm, bias=bias, stable=norm == "layer")
    resolved, weights = _resolved(tmp_path, cfg, reference)
    waveform = torch.linspace(-0.7, 1.3, 67).reshape(1, -1)
    with torch.no_grad():
        expected = reference.wavlm.feature_extractor(waveform)
        actual = model._feature_encoder(None, waveform[:, None, None, :], weights, resolved)
    torch.testing.assert_close(actual.squeeze(2), expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("stable", [False, True])
@pytest.mark.parametrize("frames", [9, 23])
def test_encoder_matches_official_cpu(tmp_path, cpu_graph, monkeypatch, stable, frames):
    cfg, reference = _reference(stable=stable, norm="layer" if stable else "group")
    resolved, weights = _resolved(tmp_path, cfg, reference)
    encoder = reference.wavlm.encoder
    rows = torch.linspace(-2, 0.7, frames * 8).reshape(frames, 8)
    monkeypatch.setattr(model, "_positional", lambda net, x, w, c:
                        x + encoder.pos_conv_embed(x.unsqueeze(0)).squeeze(0))
    with torch.no_grad():
        expected = encoder(rows.unsqueeze(0)).last_hidden_state.squeeze(0)
        actual = model._encoder(None, rows, weights, resolved)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("fault", ["gate_on_block_input", "ungated_bias", "query_minus_key"])
def test_encoder_check_sees_each_misplaced_term(tmp_path, cpu_graph, monkeypatch, fault):
    # Controls for the check above. Each fault loads and runs: the gate read
    # from the block input instead of its pre-norm output, the bias left
    # ungated, and the offset taken as query minus key. Each must fail it.
    cfg, reference = _reference(stable=True)
    resolved, weights = _resolved(tmp_path, cfg, reference)
    encoder = reference.wavlm.encoder
    rows = torch.linspace(-2, 0.7, 23 * 8).reshape(23, 8)
    monkeypatch.setattr(model, "_positional", lambda net, x, w, c:
                        x + encoder.pos_conv_embed(x.unsqueeze(0)).squeeze(0))
    gate, bias, norm = model.g.add_head_gate, model.g.add_position_bias, model.g.add_norm
    if fault == "gate_on_block_input":
        inputs = []

        def remember(network, inp, *args, **kwargs):
            inputs.append(inp)
            return norm(network, inp, *args, **kwargs)

        monkeypatch.setattr(model.g, "add_norm", remember)
        monkeypatch.setattr(model.g, "add_head_gate", lambda network, rows_, *args, **kwargs:
                            gate(network, inputs[-1], *args, **kwargs))
    elif fault == "ungated_bias":
        monkeypatch.setattr(model.g, "add_head_gate",
                            lambda network, rows_, *args, num_heads, head_dim:
                            torch.ones(num_heads, rows_.shape[0], 1))
    else:
        monkeypatch.setattr(model.g, "add_position_bias", lambda network, rows_, table, distance:
                            bias(network, rows_, table, distance).transpose(1, 2))
    with torch.no_grad():
        expected = encoder(rows.unsqueeze(0)).last_hidden_state.squeeze(0)
        actual = model._encoder(None, rows, weights, resolved)
    assert (actual - expected).abs().max() > 1e-3


def test_bin_checkpoint_loads_weights_on_cpu(tmp_path):
    torch.save({"lm_head.weight": torch.arange(12, dtype=torch.float32).reshape(4, 3)},
               tmp_path / "pytorch_model.bin")
    convert(tmp_path / "pytorch_model.bin", tmp_path / "model.safetensors")
    weights, shapes = model._load(tmp_path)
    np.testing.assert_array_equal(weights["lm_head.weight"], np.arange(12).reshape(4, 3))
    assert shapes["lm_head.weight"] == (4, 3)


def test_bin_checkpoint_rejects_non_tensor_state(tmp_path):
    torch.save({"lm_head.weight": "invalid"}, tmp_path / "pytorch_model.bin")
    with pytest.raises(ValueError, match="tensor"):
        convert(tmp_path / "pytorch_model.bin", tmp_path / "model.safetensors")


def test_bin_loader_uses_restricted_cpu_loading(tmp_path, monkeypatch):
    (tmp_path / "pytorch_model.bin").touch()
    observed = {}

    def load(path, **kwargs):
        observed.update(kwargs)
        return {"weight": torch.ones(1)}

    monkeypatch.setattr(torch, "load", load)
    convert(tmp_path / "pytorch_model.bin", tmp_path / "model.safetensors")
    assert observed == {"map_location": "cpu", "weights_only": True}


def test_conversion_preserves_existing_destination(tmp_path):
    destination = tmp_path / "model.safetensors"
    destination.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        convert(tmp_path / "pytorch_model.bin", destination)
    assert destination.read_bytes() == b"existing"


def test_conversion_rejects_empty_state(tmp_path):
    source = tmp_path / "pytorch_model.bin"
    torch.save({}, source)
    with pytest.raises(ValueError, match="nonempty"):
        convert(source, tmp_path / "model.safetensors")


def test_e2e_preparation_preserves_cached_checkpoint(tmp_path, monkeypatch):
    from families.wavlm.tests import test_e2e

    names = ("config.json", "preprocessor_config.json", "vocab.json", "added_tokens.json",
             "tokenizer_config.json")
    for name in names:
        (tmp_path / name).write_text("{}")
    torch.save({"weight": torch.arange(3, dtype=torch.float16)}, tmp_path / "pytorch_model.bin")
    observed = []

    def build(prepared, bundle, manifest):
        weights, shapes = model._load(prepared)
        np.testing.assert_array_equal(weights["weight"], np.arange(3))
        assert shapes["weight"] == (3,)
        assert all((prepared / name).read_text() == "{}" for name in names)
        observed.append(prepared)

    monkeypatch.setattr(test_e2e, "_build_prepared", build)
    test_e2e._build(tmp_path, tmp_path / "bundle", {})
    assert len(observed) == 1 and not observed[0].exists()
    assert not (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "pytorch_model.bin").is_file()


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


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
@pytest.mark.parametrize("stable, norm", [(False, "group"), (True, "layer")])
def test_engine_follows_the_reference(tmp_path, precision, stable, norm):
    if not torch.cuda.is_available():
        pytest.skip("engine build needs a GPU")
    trt = pytest.importorskip("tensorrt")
    cfg, reference = _reference(stable=stable, norm=norm)
    resolved, weights = _resolved(tmp_path, cfg, reference)
    model._require_weights(weights, resolved)
    # 9000 samples give 2249 frames, far beyond max_bucket_distance=6, so the
    # engine's clamp and its whole bucket table are on the path.
    waveform = torch.sin(torch.linspace(0, 400, 9000)) * 0.5

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
        plan = model.build_engine(weights, resolved, max_samples=16000, precision=precision)
    finally:
        trt.Builder.build_serialized_network = original
    # Every reduction (the norms' mean and variance) runs in fp32; the matmuls
    # run in the requested type; the engine still takes and returns fp32.
    assert seen["reduce"] == {trt.float32}
    assert seen["matmul"] == {trt.float16 if precision == "fp16" else trt.float32}

    engine, logits = _engine_logits(plan, waveform)
    assert engine.get_tensor_dtype("audio") == trt.float32
    assert engine.get_tensor_dtype("logits") == trt.float32
    with torch.no_grad():
        expected = reference(waveform[None]).logits[0]
    assert logits.shape == expected.shape
    assert torch.isfinite(logits).all()
    if precision == "fp32":
        torch.testing.assert_close(logits, expected, atol=2e-4, rtol=2e-4)
    else:
        assert torch.nn.functional.cosine_similarity(
            logits.reshape(-1), expected.reshape(-1), dim=0) > 0.999
