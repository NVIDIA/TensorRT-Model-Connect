# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-side unit tests for ESMC.

Four subjects: ``config.resolve``, which has to refuse the ESMC variants that
load every tensor and still run, the geometry and weight coverage that
``build`` checks before it touches TensorRT, ``graph.rotary_tables``, where a
wrong answer is an array of the right shape full of the wrong numbers, and the
precision policy (fp32 by default, fp16 accepted, everything else refused).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from families.esmc import config as config_module, graph, model as model_module
from tensorrt_model_connect import BuildRequest

_HIDDEN = 64
_INTERMEDIATE = 128
_LAYERS = 2
_HEADS = 4
_HEAD_DIM = _HIDDEN // _HEADS
_VOCAB = 64

# The published vocabulary order of biohub/ESMC-300M and ESMC-600M.
_TOKENIZER = [
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
    ".",
    "-",
    "|",
    "<mask>",
]
_SPECIAL = frozenset({"<cls>", "<pad>", "<eos>", "<unk>", "|", "<mask>"})


def _weights(prefix: str = "esmc.", layers: int = _LAYERS, **overrides) -> dict:
    weights = {
        f"{prefix}embed_tokens.weight": np.zeros((_VOCAB, _HIDDEN), np.float32),
        f"{prefix}norm.weight": np.zeros((_HIDDEN,), np.float32),
        "lm_head.dense.weight": np.zeros((_HIDDEN, _HIDDEN), np.float32),
        "lm_head.dense.bias": np.zeros((_HIDDEN,), np.float32),
        "lm_head.layer_norm.weight": np.zeros((_HIDDEN,), np.float32),
        "lm_head.layer_norm.bias": np.zeros((_HIDDEN,), np.float32),
        "lm_head.decoder.weight": np.zeros((_VOCAB, _HIDDEN), np.float32),
        "lm_head.decoder.bias": np.zeros((_VOCAB,), np.float32),
    }
    for index in range(layers):
        layer = f"{prefix}layers.{index}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            weights[f"{layer}{name}.weight"] = np.zeros((_HIDDEN,), np.float32)
            weights[f"{layer}{name}.bias"] = np.zeros((_HIDDEN,), np.float32)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            weights[f"{layer}self_attn.{name}.weight"] = np.zeros((_HIDDEN, _HIDDEN), np.float32)
        for name in ("q_norm", "k_norm"):
            weights[f"{layer}self_attn.{name}.weight"] = np.zeros((_HIDDEN,), np.float32)
        weights[f"{layer}mlp.gate_proj.weight"] = np.zeros((_INTERMEDIATE, _HIDDEN), np.float32)
        weights[f"{layer}mlp.up_proj.weight"] = np.zeros((_INTERMEDIATE, _HIDDEN), np.float32)
        weights[f"{layer}mlp.down_proj.weight"] = np.zeros((_HIDDEN, _INTERMEDIATE), np.float32)
    weights.update(overrides)
    return weights


def _config(tmp_path: Path, **overrides) -> Path:
    body = {
        "model_type": "esmc",
        "hidden_size": _HIDDEN,
        "intermediate_size": _INTERMEDIATE,
        "num_hidden_layers": _LAYERS,
        "num_attention_heads": _HEADS,
        "num_key_value_heads": _HEADS,
        "head_dim": _HEAD_DIM,
        "vocab_size": _VOCAB,
        "hidden_act": "silu",
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": False,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "mask_token_id": 32,
        "pad_token_id": 1,
        "max_position_embeddings": 2048,
    }
    body.update(overrides)
    (tmp_path / "config.json").write_text(json.dumps(body), encoding="utf-8")
    return tmp_path


def _tokenizer(tmp_path: Path, tokens: list[str] | None = None) -> Path:
    tokens = list(tokens or _TOKENIZER)
    payload = {
        "version": "1.0",
        "added_tokens": [
            {"id": index, "content": token, "special": token in _SPECIAL}
            for index, token in enumerate(tokens)
            if token in _SPECIAL
        ],
        "model": {
            "type": "BPE",
            "unk_token": "<unk>",
            "fuse_unk": False,
            "vocab": {token: index for index, token in enumerate(tokens)},
        },
    }
    (tmp_path / "tokenizer.json").write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def test_geometry_comes_from_the_tensors(tmp_path: Path) -> None:
    cfg = config_module.resolve(_config(tmp_path), _weights(), "esmc.")
    assert cfg["hidden_size"] == _HIDDEN
    assert cfg["intermediate_size"] == _INTERMEDIATE
    assert cfg["num_layers"] == _LAYERS
    assert cfg["num_heads"] == _HEADS
    assert cfg["head_dim"] == _HEAD_DIM
    assert cfg["vocab_size"] == _VOCAB
    assert cfg["eps"] == 1e-5


def test_the_residual_scale_is_the_reference_formula(tmp_path: Path) -> None:
    # The one structural difference that is invisible on 600M: s is exactly 1.0
    # at 36 layers and 0.912871 at 30, so only the 300M build can catch it.
    cfg = config_module.resolve(_config(tmp_path), _weights(), "esmc.")
    assert math.isclose(cfg["residual_scale"], math.sqrt(_LAYERS / 36))
    deep = _weights(layers=36)
    cfg = config_module.resolve(_config(tmp_path, num_hidden_layers=36), deep, "esmc.")
    assert cfg["residual_scale"] == 1.0


def test_a_config_that_contradicts_the_tensors_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        config_module.resolve(_config(tmp_path, hidden_size=_HIDDEN + 8), _weights(), "esmc.")


def test_a_gelu_mlp_is_refused(tmp_path: Path) -> None:
    # ESMC's MLP is SwiGLU on the same three tensors a gelu MLP would read.
    with pytest.raises(NotImplementedError, match="silu"):
        config_module.resolve(_config(tmp_path, hidden_act="gelu"), _weights(), "esmc.")


@pytest.mark.parametrize("flag", ["attention_bias", "mlp_bias"])
def test_biased_projections_are_refused(tmp_path: Path, flag: str) -> None:
    with pytest.raises(NotImplementedError, match="bias-free"):
        config_module.resolve(_config(tmp_path, **{flag: True}), _weights(), "esmc.")


def test_a_tied_lm_head_is_refused(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="tie_word_embeddings"):
        config_module.resolve(_config(tmp_path, tie_word_embeddings=True), _weights(), "esmc.")


@pytest.mark.parametrize(
    "rope",
    [
        {"rope_type": "linear", "rope_theta": 10000.0},
        {"rope_type": "default", "rope_theta": 500000.0},
    ],
)
def test_rotary_variants_are_refused(tmp_path: Path, rope: dict) -> None:
    with pytest.raises(NotImplementedError, match="rotation|theta"):
        config_module.resolve(_config(tmp_path, rope_parameters=rope), _weights(), "esmc.")


def test_grouped_query_attention_is_refused(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="grouped-query"):
        config_module.resolve(_config(tmp_path, num_key_value_heads=2), _weights(), "esmc.")


def test_a_head_width_that_is_not_hidden_over_heads_is_refused(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="head_dim"):
        config_module.resolve(_config(tmp_path, head_dim=_HEAD_DIM // 2), _weights(), "esmc.")


def test_an_odd_head_width_is_refused(tmp_path: Path) -> None:
    # The rotation pairs the two halves of a head, so an odd width has no
    # pairing at all. 64 heads over 64 columns divides evenly and still leaves
    # one column per head.
    with pytest.raises(NotImplementedError, match="even head width"):
        config_module.resolve(
            _config(tmp_path, num_attention_heads=_HIDDEN, num_key_value_heads=_HIDDEN, head_dim=1),
            _weights(),
            "esmc.",
        )


def test_a_mask_token_outside_the_vocabulary_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mask_token_id"):
        config_module.resolve(_config(tmp_path, mask_token_id=_VOCAB + 1), _weights(), "esmc.")


def test_non_contiguous_layer_indices_are_refused(tmp_path: Path) -> None:
    weights = _weights()
    weights["esmc.layers.7.mlp.gate_proj.weight"] = np.zeros((_INTERMEDIATE, _HIDDEN), np.float32)
    with pytest.raises(ValueError, match="contiguous"):
        config_module.resolve(_config(tmp_path), weights, "esmc.")


def test_the_rotary_tables_start_at_the_identity() -> None:
    cos, sin = graph.rotary_tables(8, 16)
    assert cos.shape == (1, 8, 16) and sin.shape == (1, 8, 16)
    # Position 0 rotates by nothing at all: every cosine is one and every
    # sine is zero. A table built from positions 1..n would fail here.
    assert np.allclose(cos[0, 0], np.ones(16))
    assert np.allclose(sin[0, 0], np.zeros(16))


def test_the_rotary_tables_mirror_their_halves() -> None:
    # emb = cat(freqs, freqs): the second half repeats the first, which is
    # what pairs the table with the rotate_half convention. A table built
    # from interleaved frequencies has the same shape and the wrong values.
    cos, sin = graph.rotary_tables(5, 12)
    assert np.allclose(cos[0, :, :6], cos[0, :, 6:])
    assert np.allclose(sin[0, :, :6], sin[0, :, 6:])


def test_the_rotary_schedule_is_the_roformer_one() -> None:
    head_dim = 8
    cos, _ = graph.rotary_tables(3, head_dim)
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2) / head_dim))
    assert np.allclose(cos[0, 2, :4], np.cos(2.0 * inv_freq), atol=1e-6)


def test_the_rotary_schedule_follows_theta() -> None:
    # The schedule is recomputed from the config value, because no checkpoint
    # stores one. A table that ignores theta is right on these two models and
    # wrong on any other rotation.
    head_dim = 8
    cos, _ = graph.rotary_tables(3, head_dim, theta=500000.0)
    inv_freq = 1.0 / (500000.0 ** (np.arange(0, head_dim, 2) / head_dim))
    assert np.allclose(cos[0, 2, :4], np.cos(2.0 * inv_freq), atol=1e-6)


def _request(tmp_path: Path, precision: str = "fp32", **overrides) -> BuildRequest:
    return BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "b.bundle",
        family="esmc",
        task="masked_text_to_token_scores",
        precision=precision,
        max_sequence_length=overrides.pop("max_sequence_length", 32),
        **overrides,
    )


def test_bf16_is_refused(tmp_path: Path) -> None:
    """fp32 and fp16 are the qualified precisions; bf16 is refused at build time."""
    with pytest.raises(NotImplementedError, match="fp32 or fp16"):
        model_module.build(_request(tmp_path, "bf16"), object())


def test_fp16_is_accepted(tmp_path: Path) -> None:
    """fp16 passes the precision check and goes on to read the checkpoint."""
    with pytest.raises(FileNotFoundError, match="missing esmc checkpoint"):
        model_module.build(_request(tmp_path, "fp16"), object())


def test_fp8_is_refused(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="quantization=none"):
        model_module.build(_request(tmp_path, quantization="fp8"), object())


def test_the_default_precision_is_fp32() -> None:
    from families.esmc.support import describe
    from tensorrt_model_connect.model_support import ModelMetadata

    support = describe(ModelMetadata(config={"model_type": "esmc"}, model_index={}))
    assert support.default_precision == "fp32"


def test_a_sequence_length_is_required(tmp_path: Path) -> None:
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "b.bundle",
        family="esmc",
        task="masked_text_to_token_scores",
        precision="fp32",
    )
    with pytest.raises(ValueError, match="max_sequence_length"):
        model_module.build(request, object())


def _tiny_checkpoint(seed: int = 0) -> dict:
    """A small random ESMC checkpoint with every tensor ``build_engine`` reads."""
    rng = np.random.default_rng(seed)

    def mat(*shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    weights = _weights(
        **{
            "esmc.embed_tokens.weight": mat(_VOCAB, _HIDDEN),
            "esmc.norm.weight": np.ones(_HIDDEN, np.float32),
            "lm_head.dense.weight": mat(_HIDDEN, _HIDDEN),
            "lm_head.dense.bias": mat(_HIDDEN),
            "lm_head.layer_norm.weight": np.ones(_HIDDEN, np.float32),
            "lm_head.layer_norm.bias": mat(_HIDDEN),
            "lm_head.decoder.weight": mat(_VOCAB, _HIDDEN),
            "lm_head.decoder.bias": mat(_VOCAB),
        }
    )
    for index in range(_LAYERS):
        layer = f"esmc.layers.{index}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            weights[f"{layer}{name}.weight"] = np.ones(_HIDDEN, np.float32)
            weights[f"{layer}{name}.bias"] = mat(_HIDDEN)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            weights[f"{layer}self_attn.{name}.weight"] = mat(_HIDDEN, _HIDDEN)
        for name in ("q_norm", "k_norm"):
            weights[f"{layer}self_attn.{name}.weight"] = np.ones(_HIDDEN, np.float32)
        weights[f"{layer}mlp.gate_proj.weight"] = mat(_INTERMEDIATE, _HIDDEN)
        weights[f"{layer}mlp.up_proj.weight"] = mat(_INTERMEDIATE, _HIDDEN)
        weights[f"{layer}mlp.down_proj.weight"] = mat(_HIDDEN, _INTERMEDIATE)
    return weights


def test_an_fp16_engine_keeps_the_io_and_the_values(tmp_path: Path) -> None:
    """fp16 changes only the work type: int32 in, float32 logits out, close to fp32."""
    trt = pytest.importorskip("tensorrt")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("engine build needs a GPU")
    weights = _tiny_checkpoint()
    cfg = config_module.resolve(_config(tmp_path), weights, "esmc.")
    window, length = 16, 11
    ids = torch.ones(window, dtype=torch.int32, device="cuda")
    ids[:length] = torch.tensor([0, 5, 6, 32, 7, 8, 9, 32, 10, 11, 2], dtype=torch.int32)
    mask = torch.zeros(window, dtype=torch.int32, device="cuda")
    mask[:length] = 1

    def run(precision):
        plan = model_module.build_engine(weights, cfg, seq_len=window, precision=precision)
        engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
        io = {
            engine.get_tensor_name(i): (
                engine.get_tensor_dtype(engine.get_tensor_name(i)),
                tuple(engine.get_tensor_shape(engine.get_tensor_name(i))),
            )
            for i in range(engine.num_io_tensors)
        }
        logits = torch.empty((window, _VOCAB), dtype=torch.float32, device="cuda")
        context = engine.create_execution_context()
        for name, tensor in (("input_ids", ids), ("attention_mask", mask), ("logits", logits)):
            context.set_tensor_address(name, tensor.data_ptr())
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return io, logits[:length].cpu()

    io32, logits32 = run("fp32")
    io16, logits16 = run("fp16")
    assert (
        io16
        == io32
        == {
            "input_ids": (trt.int32, (window,)),
            "attention_mask": (trt.int32, (window,)),
            "logits": (trt.float32, (window, _VOCAB)),
        }
    )
    assert torch.isfinite(logits16).all()
    assert float((logits16 - logits32).abs().max()) < 1e-2


def test_the_checkpoint_reads_its_own_tensors(tmp_path: Path) -> None:
    """A checkpoint with no extra tensors and the right shapes is accepted."""
    config_dir = _config(tmp_path)
    cfg = config_module.resolve(config_dir, _weights(), "esmc.")
    model_module._require_tensors(_weights(), "esmc.", cfg)


def test_an_unexpected_tensor_is_refused(tmp_path: Path) -> None:
    # A projection bias is the exact tensor a checkpoint of a different ESMC
    # variant would carry, and the graph would drop it without saying so.
    weights = _weights()
    weights["esmc.layers.0.self_attn.q_proj.bias"] = np.zeros((_HIDDEN,), np.float32)
    cfg = config_module.resolve(_config(tmp_path), weights, "esmc.")
    with pytest.raises(ValueError, match="does not read"):
        model_module._require_tensors(weights, "esmc.", cfg)


def test_a_missing_tensor_is_refused(tmp_path: Path) -> None:
    weights = _weights()
    del weights["esmc.layers.1.mlp.down_proj.weight"]
    cfg = config_module.resolve(_config(tmp_path), weights, "esmc.")
    with pytest.raises(KeyError, match="down_proj"):
        model_module._require_tensors(weights, "esmc.", cfg)


def test_a_misshapen_tensor_is_refused(tmp_path: Path) -> None:
    # A shape the geometry checks cannot see: the decoder is the head, not part
    # of the stack the layer count and the widths are measured from.
    weights = _weights()
    weights["lm_head.decoder.weight"] = np.zeros((_VOCAB, _HIDDEN + 8), np.float32)
    cfg = config_module.resolve(_config(tmp_path), weights, "esmc.")
    with pytest.raises(ValueError, match="lm_head.decoder.weight"):
        model_module._require_tensors(weights, "esmc.", cfg)


def test_the_vocabulary_is_the_engine_width(tmp_path: Path) -> None:
    config_dir = _config(tmp_path)
    _tokenizer(config_dir)
    cfg = config_module.resolve(config_dir, _weights(), "esmc.")
    vocabulary, framing = model_module._vocabulary(config_dir, cfg)
    tokens = vocabulary.decode("utf-8").splitlines()
    assert len(tokens) == _VOCAB
    # The real tokens keep their published ids, and the unused columns get one
    # unique placeholder each so the runtime's line count matches the engine.
    assert tokens[: len(_TOKENIZER)] == _TOKENIZER
    assert len(set(tokens)) == _VOCAB
    assert framing == {
        "pad_token_id": 1,
        "cls_token_id": 0,
        "eos_token_id": 2,
        "mask_token_id": 32,
        "unk_token_id": 3,
    }


def test_a_tokenizer_wider_than_the_head_is_refused(tmp_path: Path) -> None:
    config_dir = _config(tmp_path)
    _tokenizer(config_dir, _TOKENIZER + [f"<extra_{index}>" for index in range(_VOCAB)])
    cfg = config_module.resolve(config_dir, _weights(), "esmc.")
    with pytest.raises(ValueError, match="columns"):
        model_module._vocabulary(config_dir, cfg)


@pytest.mark.parametrize("token", ["<cls>", "<pad>", "<eos>", "<mask>", "<unk>"])
@pytest.mark.parametrize("index", [-1, 4, 33, _VOCAB])
def test_a_special_token_must_match_its_vocabulary_row(tmp_path: Path, token, index) -> None:
    config_dir = _config(tmp_path)
    _tokenizer(config_dir)
    path = config_dir / "tokenizer.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    next(entry for entry in payload["added_tokens"] if entry["content"] == token)["id"] = index
    path.write_text(json.dumps(payload), encoding="utf-8")
    cfg = config_module.resolve(config_dir, _weights(), "esmc.")
    with pytest.raises(ValueError, match="model.vocab does not place it at that id"):
        model_module._vocabulary(config_dir, cfg)


def test_a_tokenizer_whose_mask_disagrees_with_the_config_is_refused(tmp_path: Path) -> None:
    config_dir = _config(tmp_path, mask_token_id=31)
    _tokenizer(config_dir)
    cfg = config_module.resolve(config_dir, _weights(), "esmc.")
    with pytest.raises(ValueError, match="mask_token_id"):
        model_module._vocabulary(config_dir, cfg)


def test_open_checkpoint_single_safetensors(tmp_path: Path) -> None:
    """A single model.safetensors checkpoint is loaded as a numpy dictionary."""
    from safetensors.numpy import save_file

    tensors = {
        "esmc.embed_tokens.weight": np.ones((8, 16), dtype=np.float32),
        "lm_head.decoder.bias": np.arange(8, dtype=np.float32),
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))
    loaded = model_module._open_checkpoint(tmp_path)
    assert set(loaded.keys()) == set(tensors.keys())
    assert np.array_equal(loaded["esmc.embed_tokens.weight"], tensors["esmc.embed_tokens.weight"])


def test_open_checkpoint_sharded_safetensors(tmp_path: Path) -> None:
    """Sharded safetensors checkpoints mapped by model.safetensors.index.json are merged."""
    from safetensors.numpy import save_file

    shard1 = {"esmc.embed_tokens.weight": np.ones((8, 16), dtype=np.float32)}
    shard2 = {"lm_head.decoder.bias": np.arange(8, dtype=np.float32)}
    save_file(shard1, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file(shard2, str(tmp_path / "model-00002-of-00002.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "esmc.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "lm_head.decoder.bias": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = model_module._open_checkpoint(tmp_path)
    assert set(loaded.keys()) == {"esmc.embed_tokens.weight", "lm_head.decoder.bias"}
    assert np.array_equal(loaded["esmc.embed_tokens.weight"], shard1["esmc.embed_tokens.weight"])
    assert np.array_equal(loaded["lm_head.decoder.bias"], shard2["lm_head.decoder.bias"])


def test_open_checkpoint_missing_all_raises(tmp_path: Path) -> None:
    """An empty directory raises FileNotFoundError naming the files it looked for."""
    with pytest.raises(FileNotFoundError, match="missing esmc checkpoint"):
        model_module._open_checkpoint(tmp_path)


def test_a_pytorch_binary_is_refused(tmp_path: Path) -> None:
    """This family declares one checkpoint format; a .bin is named, not guessed at."""
    import torch

    torch.save({"esmc.embed_tokens.weight": torch.ones((8, 16))}, tmp_path / "pytorch_model.bin")
    with pytest.raises(ValueError, match="safetensors"):
        model_module._open_checkpoint(tmp_path)


def test_open_checkpoint_missing_shard_raises(tmp_path: Path) -> None:
    """A missing shard referenced by an index file raises FileNotFoundError."""
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight1": "missing_shard.safetensors"}}), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="missing esmc checkpoint shard"):
        model_module._open_checkpoint(tmp_path)


def test_open_checkpoint_invalid_index_raises(tmp_path: Path) -> None:
    """An invalid index JSON or missing weight_map raises ValueError."""
    (tmp_path / "model.safetensors.index.json").write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        model_module._open_checkpoint(tmp_path)

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"no_weight_map": {}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="has no valid weight_map"):
        model_module._open_checkpoint(tmp_path)


def test_open_checkpoint_rejects_path_traversal_and_empty_shards(tmp_path: Path) -> None:
    """Index files specifying paths outside the directory or empty strings raise ValueError."""
    for shard in ("../escaped.safetensors", "   "):
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight1": shard}}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="must be non-empty checkpoint-local filenames"):
            model_module._open_checkpoint(tmp_path)
