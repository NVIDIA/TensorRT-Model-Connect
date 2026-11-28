# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The engine and its subgraphs against the Transformers modules and the official extractor (GPU).

A small random GraniteSpeech5ForCTC keeps the real front end and shrinks every width. Its attention
context of 8 frames makes a 1.3 s clip span nine blocks at the first layer, the last holding one
real frame, so the block padding and the key mask are exercised. FP32 engines must match the FP32
modules to float accuracy (TF32 off on both sides); BF16 engines are bounded against the same
FP32 modules. The CTC head is scaled so that every frame has a clear first and second logit.
"""

from __future__ import annotations

import json
import math
import weakref
from pathlib import Path

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from families.granite_speech5 import config as config_module  # noqa: E402
from families.granite_speech5.constants import pad_waveform  # noqa: E402
from families.granite_speech5.encoder import (  # noqa: E402
    _block,
    add_front_end,
    attend,
    first_argmax,
    self_condition,
)
from families.granite_speech5.graph import Graph  # noqa: E402
from families.granite_speech5 import model as model_module  # noqa: E402
from families.granite_speech5.model import build_engine  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.trt]

_DTYPES = {"fp32": trt.float32, "bf16": trt.bfloat16}
_PROCESSOR = {"feature_extractor": {"feature_extractor_type": "GraniteSpeech5FeatureExtractor"}}
_TOKENIZER_CONFIG = {"tokenizer_class": "ParakeetTokenizer", "pad_token": "<unk>"}
# Relative L2 error of one block, measured on this model (largest over the lengths below, one B300):
# FP32 9.6e-8, BF16 1.6e-4. Relative positions read with the wrong sign give 7.3e-3.
FP32_BLOCK_ERROR = 1e-5
BF16_BLOCK_ERROR = 1e-3
# The self-conditioning step: FP32 2.8e-8, BF16 7.6e-4 (its posteriors are rounded to BF16 for out_mid);
# without it 0.30, with the softmax over frames 0.37.
BF16_SELF_CONDITIONING_ERROR = 3e-3
# Smallest reference top-2 logit margin at which the engine's frame id must equal the reference's.
CONFIDENT = {"fp32": 1e-3, "bf16": 0.25}
# Shares of all frames of the three clips: the confident ones (measured FP32 1.0, BF16 0.56) and the
# agreeing ones (measured 1.0 for both).
MIN_COVERAGE = {"fp32": 0.9, "bf16": 0.4}
MIN_OVERALL = {"fp32": 1.0, "bf16": 0.95}


class _Weights:
    """Checkpoint stand-in: float32 arrays by name, as Checkpoint.tensor returns them."""

    def __init__(self, module):
        self._state = {name: value.detach().float().cpu().numpy() for name, value in module.state_dict().items()}

    def tensor(self, name: str) -> np.ndarray:
        return self._state[name]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    encoder = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=2, head_dim=32,
                   num_mel_bins=80, max_position_embeddings=16, context_size=8, conv_kernel_size=7,
                   conv_expansion_factor=2, subsample_layers=[0, 1], vocab_size=48)
    config = transformers.GraniteSpeech5CTCConfig(encoder_config=encoder, vocab_size=48, pad_token_id=0)
    torch.manual_seed(0)
    model = transformers.GraniteSpeech5ForCTC(config).eval()
    with torch.no_grad():
        for layer in model.encoder.layers:
            # Batch-norm statistics away from the identity, so their folding is tested; relative positions and
            # queries/keys large enough that attention depends on them (the initializer's 0.02 leaves it uniform).
            layer.conv.norm.running_mean.uniform_(-0.5, 0.5)
            layer.conv.norm.running_var.uniform_(0.5, 2.0)
            layer.self_attn.rel_pos_emb.weight.normal_(0.0, 1.0)
            layer.self_attn.q_proj.weight.mul_(4.0)
            layer.self_attn.k_proj.weight.mul_(4.0)
        model.encoder.out.weight.mul_(32.0)
        model.encoder.out_mid.weight.mul_(20.0)  # the fed-back posteriors move the stream, so their loss shows
    directory = Path(tmp_path_factory.mktemp("tiny-granite-speech5"))
    model.save_pretrained(str(directory))
    (directory / "processor_config.json").write_text(json.dumps(_PROCESSOR))
    (directory / "tokenizer_config.json").write_text(json.dumps(_TOKENIZER_CONFIG))
    cfg = config_module.resolve(directory, {"<unk>": 0})
    saved = torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")  # TF32 off for the reference, restored afterwards
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield model.to("cuda"), cfg
    finally:
        torch.set_float32_matmul_precision(saved[0])
        torch.backends.cudnn.allow_tf32 = saved[1]


def _engine(build, profiles: dict, work):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    graph = Graph(network, work)  # holds the constants' host buffers until the engine is built
    build(graph)
    profile = builder.create_optimization_profile()
    for name, shapes in profiles.items():
        profile.set_shape(name, *shapes)
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    del graph
    assert plan is not None
    return trt.Runtime(logger).deserialize_cuda_engine(plan)


def _run(engine, inputs: dict) -> dict:
    """Run once; outputs are allocated from the shapes the engine infers."""
    context = engine.create_execution_context()
    for name, tensor in inputs.items():
        context.set_input_shape(name, tuple(tensor.shape))
        context.set_tensor_address(name, tensor.data_ptr())
    outputs = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
            continue
        dtype = {trt.float32: torch.float32, trt.int32: torch.int32,
                 trt.bfloat16: torch.bfloat16}[engine.get_tensor_dtype(name)]
        outputs[name] = torch.empty(tuple(context.get_tensor_shape(name)), dtype=dtype, device="cuda")
        context.set_tensor_address(name, outputs[name].data_ptr())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return outputs


def _clip(samples: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    audio = (0.1 * rng.standard_normal(samples)).astype(np.float32)
    audio[: samples // 4] *= 1e-3  # a quiet stretch, so the clip maximum and its floor matter
    return audio


def _relative_error(ours, expected) -> float:
    return float(torch.linalg.vector_norm(ours.float() - expected.float()) / torch.linalg.vector_norm(expected.float()))


@pytest.mark.parametrize("samples", [20800, 21 * 160, 16000 * 3 + 37])
def test_front_end_engine_matches_the_extractor(tiny, samples):
    _, cfg = tiny
    fe = cfg.front_end

    def build(g):
        waveform = g.network.add_input("waveform", trt.float32, (-1,))
        features = add_front_end(g, waveform, fe)
        features.name = "features"
        g.network.mark_output(features)

    engine = _engine(build, {"waveform": ((1632,), (samples + 513,), (16000 * 4 + 513,))}, trt.float32)
    audio = _clip(samples)
    ours = _run(engine, {"waveform": torch.from_numpy(pad_waveform(audio, fe)).cuda()})["features"]
    extractor = transformers.GraniteSpeech5FeatureExtractor()
    expected = extractor(audio, sampling_rate=16000, return_tensors="pt")["input_features"][0].cuda()
    assert ours.shape == expected.shape == (fe.stacked_frames(samples), 320)
    assert bool(torch.isfinite(ours).all())
    assert float((ours - expected).abs().max()) < 1e-3


def _block_engine(cfg, weights, index: int, longest: int, work):
    subsample = index in cfg.encoder.subsample_layers

    def build(g):
        x = g.network.add_input("x", trt.float32, (-1, cfg.encoder.hidden_size))
        out = _block(g, weights, cfg, x, f"encoder.layers.{index}.", g.dim(x, 0), longest, subsample)
        out.name = "y"
        g.network.mark_output(out)

    width = cfg.encoder.hidden_size
    return _engine(build, {"x": ((1, width), (21, width), (longest, width))}, work)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("index", [0, 2])
def test_blocks_match_the_reference(tiny, precision, index):
    """A subsampling block (0) and a plain block (2) over lengths that pad the last attention block."""
    model, cfg = tiny
    engine = _block_engine(cfg, _Weights(model), index, 40, _DTYPES[precision])
    layer = model.encoder.layers[index]
    worst = 0.0
    for frames in (8, 9, 21, 34):
        x = torch.randn(frames, cfg.encoder.hidden_size, generator=torch.Generator().manual_seed(frames)).cuda()
        ours = _run(engine, {"x": x})["y"]
        with torch.no_grad():
            expected = layer(x[None], position_embeddings=model.encoder.attention_dists)[0]
        assert ours.shape == expected.shape, (frames, ours.shape, expected.shape)
        assert bool(torch.isfinite(ours).all()) and bool(torch.isfinite(expected).all()), frames
        error = _relative_error(ours, expected)
        assert math.isfinite(error), (frames, error)  # max() would keep the old worst past a NaN
        worst = max(worst, error)
    print(f"block {index} {precision}: largest relative L2 error {worst:.3e}")
    assert worst < (FP32_BLOCK_ERROR if precision == "fp32" else BF16_BLOCK_ERROR), worst


def test_block_check_detects_mirrored_positions(tiny):
    """Negative control: the reference with i - j mirrored is far outside the FP32 bound."""
    model, cfg = tiny
    engine = _block_engine(cfg, _Weights(model), 2, 40, trt.float32)
    x = torch.randn(21, cfg.encoder.hidden_size, generator=torch.Generator().manual_seed(21)).cuda()
    ours = _run(engine, {"x": x})["y"]
    mirrored = 2 * cfg.encoder.max_position_embeddings - model.encoder.attention_dists
    with torch.no_grad():
        wrong = model.encoder.layers[2](x[None], position_embeddings=mirrored)[0]
    assert bool(torch.isfinite(ours).all()) and bool(torch.isfinite(wrong).all())
    error = _relative_error(ours, wrong)
    print(f"mirrored positions: relative L2 error {error:.3e}")
    assert error > 100 * FP32_BLOCK_ERROR


def test_first_argmax_takes_the_lowest_index():
    logits = np.full((8, 10), -1.0, dtype=np.float32)
    logits[0, [3, 7]] = 2.0          # a tie: the first index wins, as torch.argmax returns it
    logits[1, 9] = 0.5               # the maximum at the last index
    logits[2, :] = 0.0               # every index ties
    logits[3, [0, 1]] = [5.0, 5.0]
    logits[4, :] = np.nan            # no maximum: the sentinel id, outside the vocabulary
    logits[5, 2] = np.inf            # an overflowed logit anywhere in a row: the sentinel too
    logits[6, [1, 4]] = [3.0, -np.inf]
    logits[7, [1, 4]] = [3.0, np.nan]

    def build(g):
        x = g.network.add_input("logits", trt.float32, (-1, 10))
        ids = first_argmax(g, x)
        ids.name = "ids"
        g.network.mark_output(ids)

    engine = _engine(build, {"logits": ((1, 10), (5, 10), (8, 10))}, trt.float32)
    ids = _run(engine, {"logits": torch.from_numpy(logits).cuda()})["ids"].cpu().tolist()
    assert ids == [3, 9, 0, 0, 10, 10, 10, 10]


def _reference_logits(model, audio: np.ndarray):
    features = transformers.GraniteSpeech5FeatureExtractor()(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        return model(input_features=features["input_features"].cuda()).logits[0].float()


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_engine_frame_ids_match_the_model(tiny, precision):
    """The whole engine (front end, four blocks, self-conditioning, CTC argmax) against GraniteSpeech5ForCTC.

    A random model has frames whose two best logits nearly tie: the ids must be equal on every frame whose
    reference margin exceeds the precision's own logit error (``CONFIDENT``), those frames must be a share
    ``MIN_COVERAGE`` of all, and the ids must agree on a share ``MIN_OVERALL`` of all frames.
    """
    model, cfg = tiny
    limits = config_module.sample_limits(cfg, 16000 * 4)
    plan = build_engine(_Weights(model), cfg, limits, precision)
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    confident_frames = agreeing_frames = frames = 0
    for samples in (20800, 21 * 160, 16000 * 3 + 37):
        audio = _clip(samples, seed=samples)
        ours = _run(engine, {"waveform": torch.from_numpy(pad_waveform(audio, cfg.front_end)).cuda()})["frame_ids"]
        logits = _reference_logits(model, audio)
        expected = logits.argmax(-1).int()
        assert ours.shape == expected.shape == (cfg.output_frames(samples),)
        top = torch.topk(logits, 2, dim=-1).values
        confident = (top[:, 0] - top[:, 1]) > CONFIDENT[precision]
        assert bool((ours == expected)[confident].all()), (samples, "a confident frame differs")
        confident_frames += int(confident.sum())
        agreeing_frames += int((ours == expected).sum())
        frames += int(expected.numel())
    coverage, overall = confident_frames / frames, agreeing_frames / frames
    print(f"{precision} engine: {frames} frames, {coverage:.3f} confident (all agree), {overall:.4f} agree overall")
    assert coverage >= MIN_COVERAGE[precision], coverage
    assert overall >= MIN_OVERALL[precision], overall


def test_the_graph_outlives_the_engine_build(tiny, monkeypatch):
    """The constants point into host buffers the graph owns: it must be alive when TensorRT reads them."""
    model, cfg = tiny
    graphs, alive = [], []
    initialise = Graph.__init__

    def tracked(self, *args, **kwargs):
        initialise(self, *args, **kwargs)
        graphs.append(weakref.ref(self))

    class CheckedBuilder(trt.Builder):
        def build_serialized_network(self, network, config):
            alive.append([ref() is not None for ref in graphs])
            return super().build_serialized_network(network, config)

    monkeypatch.setattr(Graph, "__init__", tracked)
    monkeypatch.setattr(model_module.trt, "Builder", CheckedBuilder)
    plan = build_engine(_Weights(model), cfg, config_module.sample_limits(cfg, 16000), "bf16")
    assert plan is not None and plan.nbytes > 0
    assert alive == [[True]]


def test_bf16_constants_refuse_values_beyond_the_bf16_range():
    graph = Graph(None, trt.bfloat16)
    largest = float(np.finfo(np.float32).max)  # finite in FP32, rounds to infinity in BF16
    with pytest.raises(ValueError, match="BF16 cannot hold"):
        graph.weights(np.array([1.0, largest], dtype=np.float32), trt.bfloat16)
    assert graph.weights(np.array([1.0, largest], dtype=np.float32), trt.float32) is not None
    assert graph.weights(np.array([1.0, -3.0e38], dtype=np.float32), trt.bfloat16) is not None


def test_linear_accumulates_the_bias_with_the_product():
    """The bias is accumulated with the BF16 product and rounded once, as the reference's epilogue adds it.

    ``[1, 2**-8] . [1, 1] - 1`` keeps its ``2**-8``; rounding the product to BF16 first gives 0.
    """
    def build(g):
        x = g.network.add_input("x", trt.bfloat16, (-1, 2))
        y = g.linear(x, np.ones((1, 2), dtype=np.float32), np.array([-1.0], dtype=np.float32))
        y.name = "y"
        g.network.mark_output(y)

    engine = _engine(build, {"x": ((1, 2), (1, 2), (4, 2))}, trt.bfloat16)
    x = torch.tensor([[1.0, 2.0 ** -8]], dtype=torch.bfloat16, device="cuda")
    out = _run(engine, {"x": x})["y"]
    assert out.dtype == torch.bfloat16
    assert float(out[0, 0]) == 2.0 ** -8


def _self_conditioning_engine(cfg, weights, work):
    def build(g):
        x = g.network.add_input("x", trt.float32, (-1, cfg.encoder.hidden_size))
        head = g.folded(weights.tensor("encoder.out.weight"), weights.tensor("encoder.out.bias"), trt.float32)
        out = self_condition(g, weights, x, head)
        out.name = "y"
        g.network.mark_output(out)

    width = cfg.encoder.hidden_size
    return _engine(build, {"x": ((1, width), (21, width), (64, width))}, work)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_self_conditioning_matches_the_reference(tiny, precision):
    """The mid-layer CTC posteriors fed back through ``out_mid`` (dropping them or the wrong softmax axis fails)."""
    model, cfg = tiny
    engine = _self_conditioning_engine(cfg, _Weights(model), _DTYPES[precision])
    x = torch.randn(21, cfg.encoder.hidden_size, generator=torch.Generator().manual_seed(5)).cuda()
    ours = _run(engine, {"x": x})["y"]
    encoder = model.encoder
    with torch.no_grad():
        expected = x + encoder.out_mid(torch.softmax(encoder.out(x), dim=-1))
        wrong_axis = x + encoder.out_mid(torch.softmax(encoder.out(x), dim=0))
    assert ours.shape == expected.shape and bool(torch.isfinite(ours).all())
    error = _relative_error(ours, expected)
    print(f"self-conditioning {precision}: relative L2 error {error:.3e}, without it "
          f"{_relative_error(x, expected):.3e}, softmax over frames {_relative_error(wrong_axis, expected):.3e}")
    assert error < (FP32_BLOCK_ERROR if precision == "fp32" else BF16_SELF_CONDITIONING_ERROR), error
    for control in (x, wrong_axis):  # both mistakes lie far outside either bound
        assert _relative_error(control, expected) > 30 * BF16_SELF_CONDITIONING_ERROR


def test_engine_check_detects_missing_self_conditioning(tiny, monkeypatch):
    """Negative control of the whole-engine test: an engine built without self-conditioning disagrees on confident frames."""
    model, cfg = tiny
    monkeypatch.setattr(type(cfg.encoder), "self_conditioning_after", lambda self, layer: False)
    plan = build_engine(_Weights(model), cfg, config_module.sample_limits(cfg, 16000 * 4), "fp32")
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    audio = _clip(20800, seed=20800)
    ours = _run(engine, {"waveform": torch.from_numpy(pad_waveform(audio, cfg.front_end)).cuda()})["frame_ids"]
    logits = _reference_logits(model, audio)
    top = torch.topk(logits, 2, dim=-1).values
    confident = (top[:, 0] - top[:, 1]) > CONFIDENT["fp32"]
    agreement = float((ours == logits.argmax(-1).int())[confident].float().mean())
    print(f"engine without self-conditioning: agreement {agreement:.4f} on confident frames")
    assert agreement < 0.9, agreement


def _valid_keys(g, context: int, valid: int) -> trt.ITensor:
    positions = g.constant(np.arange(context), trt.int32, (1, 1, 1, context))
    return g.binary(positions, g.scalar(valid, trt.int32, 4), trt.ElementWiseOperation.LESS)


def test_attention_scales_the_queries_before_the_products():
    """Scores that fit FP32 although their unscaled sums do not (128 products near 2e18 * 2e18) stay exact."""
    context, width = 8, 128

    def build(g):
        inputs = [g.network.add_input(name, trt.float32, (-1, 1, context, width)) for name in ("q", "k", "v")]
        table = g.constant(np.zeros((width, 2 * context - 1)), trt.float32, (1, 1, width, 2 * context - 1))
        out = attend(g, *inputs, table, _valid_keys(g, context, context))
        out.name = "y"
        g.network.mark_output(out)

    shape = (1, 1, context, width)
    engine = _engine(build, {name: (shape, shape, (2, 1, context, width)) for name in ("q", "k", "v")}, trt.float32)
    rng = np.random.default_rng(3)
    q, k = (np.float32(2e18) * (1 + 0.1 * rng.random(shape)).astype(np.float32) for _ in range(2))
    v = rng.standard_normal(shape).astype(np.float32)
    ours = _run(engine, {name: torch.from_numpy(value).cuda() for name, value in (("q", q), ("k", k), ("v", v))})["y"]
    scores = (q.astype(np.float64) / np.sqrt(width)) @ k.astype(np.float64).transpose(0, 1, 3, 2)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    expected = (weights / weights.sum(-1, keepdims=True)) @ v.astype(np.float64)
    assert np.isfinite((q.astype(np.float64) @ k.astype(np.float64).transpose(0, 1, 3, 2)).max())
    assert float((q.astype(np.float64) ** 2).sum(-1).max()) > float(np.finfo(np.float32).max)  # unscaled sums overflow
    assert bool(torch.isfinite(ours).all())
    assert np.allclose(ours.cpu().numpy(), expected, rtol=1e-5, atol=1e-6)


def test_padded_keys_lose_to_any_valid_score():
    """Padded keys are excluded, not penalised: valid scores of -1e32 still take all the weight."""
    context, width, valid = 4, 4, 2

    def build(g):
        inputs = [g.network.add_input(name, trt.float32, (-1, 1, context, width)) for name in ("q", "k", "v")]
        table = g.constant(np.zeros((width, 2 * context - 1)), trt.float32, (1, 1, width, 2 * context - 1))
        out = attend(g, *inputs, table, _valid_keys(g, context, valid))
        out.name = "y"
        g.network.mark_output(out)

    shape = (1, 1, context, width)
    engine = _engine(build, {name: (shape, shape, shape) for name in ("q", "k", "v")}, trt.float32)
    q, k, v = (np.zeros(shape, np.float32) for _ in range(3))
    q[..., 0] = 2e16  # 1e16 once scaled by 1 / sqrt(4)
    k[..., :valid, 0] = -1e16  # valid keys score -1e32, padded keys 0
    v[..., :valid, :], v[..., valid:, :] = 1.0, 100.0
    ours = _run(engine, {name: torch.from_numpy(value).cuda() for name, value in (("q", q), ("k", k), ("v", v))})["y"]
    assert np.allclose(ours.cpu().numpy(), 1.0, rtol=1e-6, atol=0.0)
