# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-side tests for MobileBERT.

CPU: ``config.resolve`` and the builder refuse the layouts and options the
family does not build, ``graph.head_weight`` is the reference's masked-LM
projection, and the tokenizer checks accept the published file.

GPU (``trt`` and ``gpu`` markers): the TensorRT graph of small random
MobileBERT checkpoints against transformers in float32, padded the way the
runtime pads, with random values in every NoNorm and a nonzero padding
embedding so that each rule of the graph moves the result.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from families.mobilebert import config as config_module, graph, model as model_module

_VOCAB, _EMBED, _HIDDEN, _BOTTLENECK, _HEADS, _INTERMEDIATE, _LAYERS = 40, 8, 32, 16, 2, 24, 2


def _config(**overrides) -> dict:
    body = {
        "model_type": "mobilebert",
        "architectures": ["MobileBertForPreTraining"],
        "vocab_size": _VOCAB,
        "embedding_size": _EMBED,
        "hidden_size": _HIDDEN,
        "intra_bottleneck_size": _BOTTLENECK,
        "true_hidden_size": _BOTTLENECK,
        "num_attention_heads": _HEADS,
        "intermediate_size": _INTERMEDIATE,
        "num_hidden_layers": _LAYERS,
        "num_feedforward_networks": 3,
        "max_position_embeddings": 64,
        "type_vocab_size": 2,
        "hidden_act": "relu",
        "normalization_type": "no_norm",
        "trigram_input": True,
        "use_bottleneck": True,
        "key_query_shared_bottleneck": True,
        "use_bottleneck_attention": False,
        "classifier_activation": False,
        "layer_norm_eps": 1e-12,
        "pad_token_id": 0,
    }
    body.update(overrides)
    return body


def _weights(extra_ffns: int = 2) -> dict:
    zeros = np.zeros
    weights = {
        "mobilebert.embeddings.word_embeddings.weight": zeros((_VOCAB, _EMBED), np.float32),
        "mobilebert.embeddings.position_embeddings.weight": zeros((64, _HIDDEN), np.float32),
        "mobilebert.embeddings.token_type_embeddings.weight": zeros((2, _HIDDEN), np.float32),
        "mobilebert.embeddings.embedding_transformation.weight":
            zeros((_HIDDEN, 3 * _EMBED), np.float32),
        "cls.predictions.dense.weight": zeros((_HIDDEN - _EMBED, _VOCAB), np.float32),
    }
    for index in range(_LAYERS):
        layer = f"mobilebert.encoder.layer.{index}."
        weights[f"{layer}attention.self.query.weight"] = zeros((_BOTTLENECK, _BOTTLENECK),
                                                               np.float32)
        weights[f"{layer}intermediate.dense.weight"] = zeros((_INTERMEDIATE, _BOTTLENECK),
                                                             np.float32)
        for ffn in range(extra_ffns):
            weights[f"{layer}ffn.{ffn}.intermediate.dense.weight"] = zeros(
                (_INTERMEDIATE, _BOTTLENECK), np.float32)
    return weights


def _write_config(tmp_path: Path, **overrides) -> Path:
    (tmp_path / "config.json").write_text(json.dumps(_config(**overrides)), encoding="utf-8")
    return tmp_path


def test_resolve_reads_the_geometry_from_the_tensors(tmp_path: Path) -> None:
    cfg = config_module.resolve(_write_config(tmp_path), _weights())
    assert (cfg["vocab_size"], cfg["embedding_size"], cfg["hidden_size"]) == (
        _VOCAB, _EMBED, _HIDDEN)
    assert (cfg["bottleneck_size"], cfg["num_heads"], cfg["head_dim"]) == (
        _BOTTLENECK, _HEADS, _BOTTLENECK // _HEADS)
    assert (cfg["num_layers"], cfg["extra_ffns"], cfg["max_positions"]) == (_LAYERS, 2, 64)
    assert cfg["tie_word_embeddings"] is True


@pytest.mark.parametrize("key, value", [
    ("normalization_type", "layer_norm"),
    ("trigram_input", False),
    ("use_bottleneck", False),
    ("key_query_shared_bottleneck", False),
    ("use_bottleneck_attention", True),
    ("classifier_activation", True),
    ("hidden_act", "gelu"),
])
def test_resolve_refuses_the_layouts_it_does_not_build(tmp_path: Path, key, value) -> None:
    with pytest.raises(NotImplementedError, match=key):
        config_module.resolve(_write_config(tmp_path, **{key: value}), _weights())


def test_a_missing_classifier_activation_means_the_reference_default(tmp_path: Path) -> None:
    """transformers defaults classifier_activation to true, which this family does not build."""
    body = _config()
    del body["classifier_activation"]
    (tmp_path / "config.json").write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(NotImplementedError, match="classifier_activation"):
        config_module.resolve(tmp_path, _weights())


def test_resolve_refuses_other_heads_and_disagreeing_configs(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="MobileBertForPreTraining"):
        config_module.resolve(
            _write_config(tmp_path, architectures=["MobileBertForSequenceClassification"]),
            _weights())
    with pytest.raises(ValueError, match="num_feedforward_networks"):
        config_module.resolve(_write_config(tmp_path, num_feedforward_networks=4), _weights())
    with pytest.raises(ValueError, match="hidden_size"):
        config_module.resolve(_write_config(tmp_path, hidden_size=64), _weights())


def test_head_weight_is_the_reference_projection() -> None:
    rng = np.random.default_rng(0)
    table = rng.standard_normal((_VOCAB, _EMBED)).astype(np.float32)
    extra = rng.standard_normal((_HIDDEN - _EMBED, _VOCAB)).astype(np.float32)
    rows = rng.standard_normal((3, _HIDDEN)).astype(np.float32)
    weights = {"mobilebert.embeddings.word_embeddings.weight": table,
               "cls.predictions.dense.weight": extra}
    combined = graph.head_weight(weights, {"tie_word_embeddings": True})
    # The reference: rows @ cat([decoder.weight.T, dense.weight], 0).
    expected = rows @ np.concatenate((table.T, extra), axis=0)
    np.testing.assert_allclose(rows @ combined.T, expected, rtol=1e-5, atol=1e-5)


def test_ties_that_disagree_are_refused() -> None:
    cfg = {"num_layers": 0, "extra_ffns": 0, "hidden_size": 4, "tie_word_embeddings": True}
    weights = {key: np.zeros((2, 4), np.float32) for key in (
        "mobilebert.embeddings.embedding_transformation.bias",
        "mobilebert.embeddings.LayerNorm.weight", "mobilebert.embeddings.LayerNorm.bias",
        "cls.predictions.transform.dense.weight", "cls.predictions.transform.dense.bias",
        "cls.predictions.transform.LayerNorm.weight", "cls.predictions.transform.LayerNorm.bias",
        "cls.predictions.bias", "cls.seq_relationship.weight", "cls.seq_relationship.bias",
        "mobilebert.embeddings.word_embeddings.weight")}
    model_module._require_weights(weights, cfg)
    weights["cls.predictions.decoder.weight"] = np.ones((2, 4), np.float32)
    with pytest.raises(ValueError, match="decoder weight"):
        model_module._require_weights(weights, cfg)
    del weights["cls.predictions.decoder.weight"]
    weights["cls.predictions.decoder.bias"] = np.ones((2, 4), np.float32)
    with pytest.raises(ValueError, match="decoder bias"):
        model_module._require_weights(weights, cfg)


def _tokenizer_json(**changes) -> dict:
    vocab = {token: index for index, token in enumerate(
        ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"w{i}" for i in range(_VOCAB - 5)])}
    single = [{"SpecialToken": {"id": "[CLS]", "type_id": 0}},
              {"Sequence": {"id": "A", "type_id": 0}},
              {"SpecialToken": {"id": "[SEP]", "type_id": 0}}]
    pair = single + [{"Sequence": {"id": "B", "type_id": 1}},
                     {"SpecialToken": {"id": "[SEP]", "type_id": 1}}]
    body = {
        "added_tokens": [{"id": vocab[token], "content": token, "special": True,
                          "normalized": False, "single_word": False, "lstrip": False,
                          "rstrip": False}
                         for token in ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")],
        "normalizer": {"type": "BertNormalizer", "clean_text": True,
                       "handle_chinese_chars": True, "strip_accents": None, "lowercase": True},
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "post_processor": {"type": "TemplateProcessing", "single": single, "pair": pair},
        # The published file has no model "type"; the fields identify WordPiece.
        "model": {"unk_token": "[UNK]", "continuing_subword_prefix": "##",
                  "max_input_chars_per_word": 100, "vocab": vocab},
    }
    for path, value in changes.items():
        section, key = path.split(".")
        body[section][key] = value
    return body


def _tokenizer(tmp_path: Path, **changes):
    (tmp_path / "tokenizer.json").write_text(json.dumps(_tokenizer_json(**changes)),
                                             encoding="utf-8")
    return model_module._tokenizer(tmp_path, {"vocab_size": _VOCAB, "pad_token_id": 0})


def test_tokenizer_checks_accept_the_published_layout(tmp_path: Path) -> None:
    _, framing = _tokenizer(tmp_path)
    assert framing == {"pad_token_id": 0, "unk_token_id": 1, "cls_token_id": 2,
                       "sep_token_id": 3, "mask_token_id": 4}


@pytest.mark.parametrize("change, error", [
    ({"model.type": "BPE"}, "WordPiece"),
    ({"normalizer.lowercase": False}, "uncased BertNormalizer"),
    ({"normalizer.strip_accents": False}, "uncased BertNormalizer"),
    ({"post_processor.type": "RobertaProcessing"}, "frames"),
])
def test_tokenizer_checks_refuse_what_the_runtime_does_not_implement(tmp_path: Path, change,
                                                                     error) -> None:
    with pytest.raises(NotImplementedError, match=error):
        _tokenizer(tmp_path, **change)


def test_tokenizer_checks_refuse_what_the_runtime_would_refuse_at_load(tmp_path: Path) -> None:
    """Refused before a bundle is written, not when the bundle loads."""
    body = _tokenizer_json()
    body["added_tokens"][4]["normalized"] = True
    (tmp_path / "tokenizer.json").write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(NotImplementedError, match="normalized"):
        model_module._tokenizer(tmp_path, {"vocab_size": _VOCAB, "pad_token_id": 0})

    body = _tokenizer_json()
    body["model"]["vocab"]["w0"] = _VOCAB + 3
    (tmp_path / "tokenizer.json").write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="0..n-1"):
        model_module._tokenizer(tmp_path, {"vocab_size": _VOCAB, "pad_token_id": 0})

    body = _tokenizer_json()
    body["model"]["vocab"]["w\u08d4"] = body["model"]["vocab"].pop("w0")
    (tmp_path / "tokenizer.json").write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(NotImplementedError, match="combining mark"):
        model_module._tokenizer(tmp_path, {"vocab_size": _VOCAB, "pad_token_id": 0})
    assert model_module._reorderable("\u08d4") and not model_module._reorderable("a")


def _request(tmp_path: Path, **overrides):
    fields = {
        "model_dir": tmp_path, "task": "masked_text_to_token_scores", "precision": "fp32",
        "backend": "trt", "max_sequence_length": 16, "image_height": None, "image_width": None,
        "video_num_frames": None, "max_batch_size": 1, "tensor_parallel_size": 1,
        "context_parallel_size": 1, "quantization": None, "fp32_layers": (),
        "dynamic_kv_cache": False, "verbose": False, "mtp_seq_len": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize("overrides, error, message", [
    ({"task": "text_to_token_features"}, ValueError, "supports only task"),
    ({"backend": "trt_rtx"}, ValueError, "backend=trt"),
    ({"precision": "fp16"}, NotImplementedError, "fp32 only"),
    ({"precision": "bf16"}, NotImplementedError, "fp32 only"),
    ({"quantization": "fp8"}, NotImplementedError, "float32 checkpoint"),
    ({"max_batch_size": 2}, NotImplementedError, "max_batch_size"),
    ({"tensor_parallel_size": 2}, NotImplementedError, "tensor parallelism"),
    ({"context_parallel_size": 2}, NotImplementedError, "context parallelism"),
    ({"image_height": 32}, NotImplementedError, "image"),
    ({"video_num_frames": 4}, NotImplementedError, "video"),
    ({"fp32_layers": (1,)}, NotImplementedError, "fp32_layers"),
    ({"mtp_seq_len": 2}, NotImplementedError, "MTP"),
    ({"dynamic_kv_cache": True}, NotImplementedError, "dynamic_kv_cache"),
    ({"max_sequence_length": None}, ValueError, "fixed-length"),
    ({"max_sequence_length": 2}, ValueError, "room"),
])
def test_build_refuses_options_before_reading_the_checkpoint(tmp_path: Path, overrides, error,
                                                             message) -> None:
    with pytest.raises(error, match=message):
        model_module.build(_request(tmp_path, **overrides), writer=None)


# ---------------------------------------------------------------------------
# TensorRT graph against transformers on small random checkpoints (GPU).
# ---------------------------------------------------------------------------


def _random_checkpoint(directory: Path, *, num_feedforward_networks: int, seed: int):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    torch.manual_seed(seed)
    config = transformers.MobileBertConfig(**{
        key: value for key, value in _config(num_feedforward_networks=num_feedforward_networks,
                                             max_position_embeddings=32).items()
        if key not in ("model_type", "architectures")})
    reference = transformers.MobileBertForPreTraining(config).eval()
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            # Random values everywhere, including every NoNorm and the padding row,
            # so that no rule of the graph is invisible on this checkpoint.
            parameter.copy_(torch.randn_like(parameter) * (0.15 if parameter.dim() > 1 else 0.3)
                            + (1.0 if name.endswith("LayerNorm.weight") else 0.0))
    config.architectures = ["MobileBertForPreTraining"]
    directory.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(directory)
    torch.save(reference.state_dict(), directory / "pytorch_model.bin")
    return reference


def _run_engine(plan: bytes, ids, segments, window: int, vocab: int):
    import tensorrt as trt
    import torch

    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    inputs = {name: torch.zeros(window, dtype=torch.int32, device="cuda")
              for name in ("input_ids", "token_type_ids", "attention_mask")}
    inputs["input_ids"][: len(ids)] = torch.tensor(ids, dtype=torch.int32)
    inputs["token_type_ids"][: len(ids)] = torch.tensor(segments, dtype=torch.int32)
    inputs["attention_mask"][: len(ids)] = 1
    logits = torch.empty((window, vocab), dtype=torch.float32, device="cuda")
    relation = torch.empty((2,), dtype=torch.float32, device="cuda")
    for name, tensor in (*inputs.items(), ("prediction_logits", logits),
                         ("seq_relationship_logits", relation)):
        context.set_tensor_address(name, tensor.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return logits[: len(ids)].cpu(), relation.cpu()


@pytest.mark.trt
@pytest.mark.gpu
@pytest.mark.parametrize("num_feedforward_networks", [1, 3])
def test_graph_matches_transformers_on_random_checkpoints(tmp_path: Path,
                                                          num_feedforward_networks) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.fail("the TensorRT graph test needs a CUDA GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = _random_checkpoint(tmp_path, num_feedforward_networks=num_feedforward_networks,
                                   seed=num_feedforward_networks)
    weights = model_module._open_torch_checkpoint(tmp_path)
    cfg = config_module.resolve(tmp_path, weights)
    model_module._require_weights(weights, cfg)
    window = 12
    plan = model_module.build_engine(weights, cfg, seq_len=window)

    rng = np.random.default_rng(7)
    ids = [2] + rng.integers(5, _VOCAB, size=6).tolist() + [3] + [9, 3]
    segments = [0] * 8 + [1, 1]
    logits, relation = _run_engine(plan, ids, segments, window, _VOCAB)
    with torch.no_grad():
        # Unpadded, as the reference runs one sequence.
        expected = reference(input_ids=torch.tensor([ids]),
                             token_type_ids=torch.tensor([segments]))
    want_logits = expected.prediction_logits[0]
    want_relation = expected.seq_relationship_logits[0]
    scale = float(want_logits.abs().max())
    logit_error = float((logits - want_logits).abs().max())
    relation_error = float((relation - want_relation).abs().max())
    print(f"ffn={num_feedforward_networks}: logit max |d| {logit_error:.3e} of scale "
          f"{scale:.3e}; relation max |d| {relation_error:.3e} of "
          f"{float(want_relation.abs().max()):.3e}")
    assert logit_error <= 1e-4 * max(scale, 1.0)
    assert relation_error <= 1e-4 * max(float(want_relation.abs().max()), 1.0)
