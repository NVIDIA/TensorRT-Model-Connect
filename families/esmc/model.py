# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build an ESMC bundle.

One engine, one output: the vocabulary logits for every row of the window.
Which rows the caller asked about is the runtime's business, so the engine
returns the whole fixed-length window and the pipeline selects from it.

Three things about this checkpoint decide the shape of this file:

* the embedding is a plain table lookup. There is no ``token_dropout`` on the
  way ESMC was trained and no position table in the file - position enters
  only as the rotation inside each layer - so the only thing between the ids
  and the first block is ``gather``;
* the masked-LM head keeps its own decoder. ``tie_word_embeddings`` is false
  and ``lm_head.decoder.weight`` is in the file, so it is read, not filled in
  from the embedding table the way ``families/esm`` has to;
* the vocabulary is wider than the tokenizer. ``vocab_size`` is 64 and the
  ``tokenizer.json`` carries 33 tokens, so ``vocab.txt`` is padded out to the
  engine width with placeholders that no input produces.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorrt as trt

from . import config as config_module, encoder_builder, graph as g

_SUPPORTED_TASKS = ("masked_text_to_token_scores",)

# Masked-out scores are driven to this before the softmax: far below any real
# score and far above where a subtract-the-max softmax underflows.
_MASK_FLOOR = -1.0e4

# Work type per build precision: (numpy, TensorRT). fp16 keeps the int32
# inputs and the float32 logits, so the runtime and the bundle do not change.
_WORK_TYPES = {"fp32": (np.float32, trt.float32), "fp16": (np.float16, trt.float16)}


def _open_checkpoint(model_dir: Path) -> dict:
    """Load an ESMC checkpoint into a dictionary of NumPy arrays.

    Both published checkpoints are one ``model.safetensors``; the sharded form
    is read because a merged 6B-style checkpoint is the other layout the same
    loader will be asked for. PyTorch binaries are refused by name rather than
    guessed at: this family declares one checkpoint format.
    """
    directory = Path(model_dir)
    safetensors = directory / "model.safetensors"
    if safetensors.is_file():
        from safetensors.numpy import load_file

        return load_file(str(safetensors))

    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        from safetensors.numpy import load_file

        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as err:
            raise ValueError(f"esmc checkpoint index {index_path.name} is invalid JSON") from err
        if not isinstance(index, dict):
            raise ValueError(f"esmc checkpoint index {index_path.name} must be a JSON object")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"esmc checkpoint index {index_path.name} has no valid weight_map")
        if any(
            not isinstance(shard, str) or not shard.strip() or Path(shard).name != shard
            for shard in weight_map.values()
        ):
            raise ValueError(
                "safetensors shard names in index must be non-empty checkpoint-local filenames"
            )
        weights = {}
        for shard in sorted(set(weight_map.values())):
            shard_path = directory / shard
            if not shard_path.is_file():
                raise FileNotFoundError(f"missing esmc checkpoint shard: {shard_path}")
            weights.update(load_file(str(shard_path)))
        return weights

    binary = directory / "pytorch_model.bin"
    if binary.is_file():
        raise ValueError(
            f"esmc reads safetensors checkpoints only; {binary.name} needs converting to "
            f"model.safetensors"
        )

    raise FileNotFoundError(
        f"missing esmc checkpoint: neither {safetensors.name} nor "
        f"{index_path.name} found in {directory}"
    )


def _prefix(weights: dict) -> str:
    """``EsmcForMaskedLM`` saves under ``esmc.``; a bare ``EsmcModel`` does not."""
    for candidate in ("esmc.", ""):
        if f"{candidate}embed_tokens.weight" in weights:
            return candidate
    raise KeyError("esmc checkpoint has no embed_tokens.weight")


def _expected_tensors(prefix: str, cfg: dict) -> set[str]:
    """Every tensor this graph reads, named from the geometry it was given."""
    names = {
        f"{prefix}embed_tokens.weight",
        f"{prefix}norm.weight",
        "lm_head.dense.weight",
        "lm_head.dense.bias",
        "lm_head.layer_norm.weight",
        "lm_head.layer_norm.bias",
        "lm_head.decoder.weight",
        "lm_head.decoder.bias",
    }
    for index in range(cfg["num_layers"]):
        layer = f"{prefix}layers.{index}."
        names |= {
            f"{layer}input_layernorm.weight",
            f"{layer}input_layernorm.bias",
            f"{layer}post_attention_layernorm.weight",
            f"{layer}post_attention_layernorm.bias",
            f"{layer}self_attn.q_proj.weight",
            f"{layer}self_attn.k_proj.weight",
            f"{layer}self_attn.v_proj.weight",
            f"{layer}self_attn.o_proj.weight",
            f"{layer}self_attn.q_norm.weight",
            f"{layer}self_attn.k_norm.weight",
            f"{layer}mlp.gate_proj.weight",
            f"{layer}mlp.up_proj.weight",
            f"{layer}mlp.down_proj.weight",
        }
    return names


def _require_tensors(weights: dict, prefix: str, cfg: dict) -> None:
    """The checkpoint must be exactly the tensors the graph reads.

    Both directions matter. A missing tensor would otherwise surface as a
    ``KeyError`` from inside the graph, and an extra one means the file is not
    this model: a stack that carries another head on top, or a variant whose
    projections have a bias this graph silently drops.
    """
    expected = _expected_tensors(prefix, cfg)
    missing = sorted(expected - weights.keys())
    if missing:
        raise KeyError("esmc checkpoint is missing expected tensors: " + ", ".join(missing))
    unexpected = sorted(weights.keys() - expected)
    if unexpected:
        raise ValueError(
            "esmc checkpoint has tensors this graph does not read: " + ", ".join(unexpected)
        )

    hidden, intermediate, vocab = cfg["hidden_size"], cfg["intermediate_size"], cfg["vocab_size"]
    shapes = {
        f"{prefix}embed_tokens.weight": (vocab, hidden),
        f"{prefix}norm.weight": (hidden,),
        "lm_head.dense.weight": (hidden, hidden),
        "lm_head.dense.bias": (hidden,),
        "lm_head.layer_norm.weight": (hidden,),
        "lm_head.layer_norm.bias": (hidden,),
        "lm_head.decoder.weight": (vocab, hidden),
        "lm_head.decoder.bias": (vocab,),
    }
    for index in range(cfg["num_layers"]):
        layer = f"{prefix}layers.{index}."
        shapes.update(
            {
                f"{layer}input_layernorm.weight": (hidden,),
                f"{layer}input_layernorm.bias": (hidden,),
                f"{layer}post_attention_layernorm.weight": (hidden,),
                f"{layer}post_attention_layernorm.bias": (hidden,),
                f"{layer}self_attn.q_proj.weight": (hidden, hidden),
                f"{layer}self_attn.k_proj.weight": (hidden, hidden),
                f"{layer}self_attn.v_proj.weight": (hidden, hidden),
                f"{layer}self_attn.o_proj.weight": (hidden, hidden),
                f"{layer}self_attn.q_norm.weight": (hidden,),
                f"{layer}self_attn.k_norm.weight": (hidden,),
                f"{layer}mlp.gate_proj.weight": (intermediate, hidden),
                f"{layer}mlp.up_proj.weight": (intermediate, hidden),
                f"{layer}mlp.down_proj.weight": (hidden, intermediate),
            }
        )
    for key, shape in shapes.items():
        actual = np.asarray(weights[key]).shape
        if actual != shape:
            raise ValueError(f"esmc tensor {key} is {actual}, not {shape}")


def _builder(verbose):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    # TF32 caps a matmul graph near 0.99998 cosine by default, which hides
    # real graph error behind a plausible-looking number.
    config.clear_flag(trt.BuilderFlag.TF32)
    return builder, network, config


def build_engine(weights, cfg, *, seq_len, precision="fp32", verbose=False) -> bytes:
    builder, network, config = _builder(verbose)
    # The work type of every layer. Layer-norm statistics are taken in float32
    # either way (graph.add_layer_norm), and the engine I/O stays the same.
    work_np, work_trt = _WORK_TYPES[precision]
    prefix = cfg["prefix"]
    head_dim = cfg["head_dim"]

    input_ids = network.add_input("input_ids", trt.int32, (seq_len,))
    attention_mask = network.add_input("attention_mask", trt.int32, (seq_len,))

    table = np.asarray(weights[f"{prefix}embed_tokens.weight"], dtype=work_np)
    embedded = network.add_gather(
        g.add_constant(network, table.shape, table, dtype=work_np), input_ids, 0
    ).get_output(0)

    keep = network.add_cast(attention_mask, work_trt).get_output(0)
    column = network.add_shuffle(keep)
    column.reshape_dims = (seq_len, 1)
    keep_column = column.get_output(0)

    # Padded rows are zeroed before the first block. The attention mask would
    # hide them anyway; zeroing keeps them from feeding the norms above the
    # float16 range on a long padded tail.
    hidden = g.add_product(network, embedded, keep_column)

    # mask[i] * mask[j]: a pair counts only if both ends are real. Masking the
    # keys alone would leave the padded query rows attending normally.
    row = network.add_shuffle(keep)
    row.reshape_dims = (1, seq_len)
    pair = g.add_product(network, keep_column, row.get_output(0))
    square = network.add_shuffle(pair)
    square.reshape_dims = (1, seq_len, seq_len)
    ones = g.add_constant(network, (1, 1, 1), np.full((1,), 1.0), dtype=work_np)
    dropped = g.add_sub(network, ones, square.get_output(0))
    additive = g.add_product(
        network,
        dropped,
        g.add_constant(network, (1, 1, 1), np.full((1,), _MASK_FLOOR), dtype=work_np),
    )

    cos_table, sin_table = g.rotary_tables(
        seq_len, head_dim, theta=cfg["rope_theta"], dtype=work_np
    )
    cos = g.add_constant(network, (1, seq_len, head_dim), cos_table, dtype=work_np)
    sin = g.add_constant(network, (1, seq_len, head_dim), sin_table, dtype=work_np)

    hidden = encoder_builder.build_layers(
        network, hidden, weights, cfg, mask=additive, cos=cos, sin=sin, dtype=work_np
    )

    head = g.add_gelu(
        network,
        g.add_linear(
            network,
            hidden,
            weights["lm_head.dense.weight"],
            weights["lm_head.dense.bias"],
            dtype=work_np,
        ),
        dtype=work_np,
    )
    head = g.add_layer_norm(
        network,
        head,
        weights["lm_head.layer_norm.weight"],
        weights["lm_head.layer_norm.bias"],
        cfg["eps"],
    )
    logits = g.add_linear(
        network,
        head,
        weights["lm_head.decoder.weight"],
        weights["lm_head.decoder.bias"],
        dtype=work_np,
    )
    if logits.dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("esmc engine build failed")
    return bytes(plan)


def build(request, writer) -> None:
    """Build one ESMC bundle."""
    if request.task not in _SUPPORTED_TASKS:
        raise ValueError("esmc supports only task=" + ", ".join(_SUPPORTED_TASKS))
    if request.backend not in {"trt", "trt_rtx"}:
        raise ValueError("esmc supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("esmc does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("esmc does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("esmc does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("esmc does not support context parallelism")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError("esmc does not take an image")
    if request.video_num_frames is not None:
        raise NotImplementedError("esmc does not support video_num_frames")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError(f"esmc supports quantization=none; got {request.quantization!r}")
    if request.fp32_layers:
        raise NotImplementedError("esmc does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("esmc does not support MTP speculative decoding")

    # fp32 (the default) or fp16. The reference's own fp16 error on these
    # checkpoints is 3.3e-02 to 1.0e-01 in logit space, so fp16 is built but
    # its gate is stated separately; see tests/test_e2e.py. The layer norms
    # keep float32 statistics because they must: the residual stream reaches
    # 1.0e+04, whose square overflows fp16.
    precision = str(request.precision).lower()
    if precision not in _WORK_TYPES:
        raise NotImplementedError(
            f"esmc builds fp32 or fp16; {precision} is not qualified for this family"
        )

    if request.max_sequence_length is None:
        raise ValueError("esmc needs max_sequence_length: the engine is fixed-length")
    seq_len = int(request.max_sequence_length)
    if seq_len < 2:
        raise ValueError("esmc needs room for at least <cls> and <eos>")

    model_dir = Path(request.model_dir)
    weights = _open_checkpoint(model_dir)
    prefix = _prefix(weights)
    cfg = config_module.resolve(model_dir, weights, prefix)
    _require_tensors(weights, prefix, cfg)
    if seq_len > cfg["max_positions"]:
        raise NotImplementedError(
            f"esmc was trained to {cfg['max_positions']} positions, not {seq_len}"
        )

    vocabulary, framing = _vocabulary(model_dir, cfg)

    writer.set_header(family="esmc", task=request.task, backend=request.backend)
    writer.add_bytes(
        "model.plan",
        build_engine(
            weights, cfg, seq_len=seq_len, precision=precision, verbose=bool(request.verbose)
        ),
    )
    writer.add_bytes("vocab.txt", vocabulary)
    writer.add_json(
        "runtime.json",
        {
            "max_sequence_length": seq_len,
            "vocab_size": cfg["vocab_size"],
            "pad_token_id": framing["pad_token_id"],
            "cls_token_id": framing["cls_token_id"],
            "eos_token_id": framing["eos_token_id"],
            "mask_token_id": framing["mask_token_id"],
            "unk_token_id": framing["unk_token_id"],
            "vocabulary_id": "esmc",
        },
    )


def _vocabulary(model_dir: Path, cfg: dict) -> tuple[bytes, dict]:
    """The bundle's vocabulary, plus the ids the runtime frames and selects with.

    ESMC publishes a ``tokenizer.json`` with 33 tokens - ``<cls>``, ``<pad>``,
    ``<eos>``, ``<unk>``, the 20 residues, ``.``, ``-``, ``|`` and ``<mask>`` -
    while the engine's vocabulary is 64 wide, because the checkpoint's head is.
    ``vocab.txt`` is therefore written to the full width: the 33 real tokens in
    id order, then placeholders no residue string reaches. The runtime tokenizer
    reads that file and the factory checks its line count against the engine,
    so the two cannot drift apart.
    """
    path = Path(model_dir) / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"esmc found no tokenizer.json in {model_dir}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    vocab = payload.get("model", {}).get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("esmc tokenizer.json has no model.vocab object")

    pairs = sorted((int(position), str(token)) for token, position in vocab.items())
    if [position for position, _ in pairs] != list(range(len(pairs))):
        raise ValueError("esmc tokenizer.json vocabulary ids are not contiguous from 0")
    tokens = [token for _, token in pairs]
    if len(tokens) > cfg["vocab_size"]:
        raise ValueError(
            f"esmc tokenizer.json has {len(tokens)} tokens but the head has "
            f"{cfg['vocab_size']} columns"
        )
    if any(not token or "\n" in token for token in tokens):
        raise ValueError("esmc tokenizer.json has an empty or multi-line token")
    for index in range(len(tokens), cfg["vocab_size"]):
        placeholder = f"<esmc_unused_{index}>"
        if placeholder in vocab or placeholder in tokens:
            raise ValueError(f"esmc placeholder {placeholder} collides with a real token")
        tokens.append(placeholder)

    special = {
        str(entry["content"]): int(entry["id"])
        for entry in payload.get("added_tokens", [])
        if isinstance(entry, dict) and entry.get("special")
    }
    framing = {}
    for name, token in (
        ("pad_token_id", "<pad>"),
        ("cls_token_id", "<cls>"),
        ("eos_token_id", "<eos>"),
        ("mask_token_id", "<mask>"),
        ("unk_token_id", "<unk>"),
    ):
        if token not in special:
            raise ValueError(f"esmc tokenizer.json does not declare {token} as a special token")
        index = special[token]
        if not 0 <= index < len(tokens) or tokens[index] != token:
            raise ValueError(
                f"esmc tokenizer.json declares {token} at id {index}, "
                "but model.vocab does not place it at that id"
            )
        framing[name] = index
    if framing["mask_token_id"] != cfg["mask_token_id"]:
        raise ValueError(
            f"esmc config says mask_token_id={cfg['mask_token_id']} but tokenizer.json puts "
            f"<mask> at {framing['mask_token_id']}"
        )
    if framing["pad_token_id"] != cfg["pad_token_id"]:
        raise ValueError(
            f"esmc config says pad_token_id={cfg['pad_token_id']} but tokenizer.json puts "
            f"<pad> at {framing['pad_token_id']}"
        )
    return ("\n".join(tokens) + "\n").encode("utf-8"), framing
