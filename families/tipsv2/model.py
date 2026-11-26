# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a TIPSv2 bundle.

The bundle carries **two** engines, because TIPSv2 is two towers that meet only at a dot product.
Fusing them would make every call run both, which is the wrong shape for the three tasks here:
embedding an image, embedding a text, and scoring one against the other.

Neither engine normalises its output and neither applies the temperature. The engine returns what
the checkpoint computes; the runtime owns the L2 normalisation and the division by the
temperature, so what comes out of the engine stays comparable with what the reference returns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorrt as trt

from . import config as config_module, text_builder, tokenizer_export, vision_builder

_SUPPORTED_TASKS = ("image_to_embedding", "text_to_embedding", "text_image_to_relevance")
_PRECISIONS = {"fp32": np.float32, "fp16": np.float16}


def _require_integer(value, what: str):
    """A build option this family compares against the checkpoint, or ``None``.

    ``int(448.5)`` is 448, so a request for a size this family cannot serve would be truncated
    into the one it can and built as if it had been asked for. Every number read from the
    checkpoint's own files is refused the same way when it is not an integer; an option from the
    caller is no different.
    """
    if value is None or (isinstance(value, int) and not isinstance(value, bool)):
        return value
    raise NotImplementedError(f"tipsv2 {what} must be an integer, not {value!r}")


def _checkpoint_path(model_dir: Path) -> Path:
    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Required safetensors checkpoint is missing: {path}")
    return path


def _checkpoint_header(path: Path) -> dict:
    """Every tensor's name, shape and dtype, read from the file's header alone.

    The header is a few tens of kilobytes at the front of the file; the tensors are gigabytes.
    Reading it first means a checkpoint whose shapes this family cannot build is refused before
    any of it is materialised.
    """
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("This family requires safetensors in the build environment") from error

    with safe_open(str(path), framework="numpy") as handle:
        return {
            # A `(shape, dtype)` pair, which is everything about a tensor that can be known
            # without reading it - and all `config.resolve` and `_require_header` need.
            key: (tuple(int(size) for size in handle.get_slice(key).get_shape()),
                  str(handle.get_slice(key).get_dtype()))
            for key in handle.keys()
        }


def _open_checkpoint(path: Path) -> dict:
    """Load the safetensors checkpoint this family publishes."""
    from safetensors.numpy import load_file

    return dict(load_file(str(path)))


def _expected_shapes(cfg: dict) -> dict:
    """The exact shape of every tensor the two graphs read.

    ``config.resolve`` reads block 0 and the tables; this covers the other twenty-six blocks,
    where a different MLP width or a scalar in place of a vector would otherwise be found by
    TensorRT - or not found at all, because a one-element bias broadcasts over a row and builds a
    graph that is quietly the wrong model.
    """
    vision, text = cfg["vision"], cfg["text"]
    width = vision["hidden_size"]
    shapes = {
        "vision_encoder.cls_token": (1, 1, width),
        "vision_encoder.register_tokens": (1, vision["num_register_tokens"], width),
        "vision_encoder.pos_embed": (1, vision["num_positions"], width),
        "vision_encoder.patch_embed.proj.weight":
            (width, 3, vision["patch_size"], vision["patch_size"]),
        "vision_encoder.patch_embed.proj.bias": (width,),
        "vision_encoder.norm.weight": (width,),
        "vision_encoder.norm.bias": (width,),
        "text_encoder.token_embedding.weight": (text["vocab_size"], text["hidden_size"]),
        "text_encoder.ln_final.weight": (text["hidden_size"],),
        "text_encoder.ln_final.bias": (text["hidden_size"],),
    }
    for index in range(vision["num_layers"]):
        layer = f"vision_encoder.blocks.{index}"
        shapes.update({
            f"{layer}.norm1.weight": (width,), f"{layer}.norm1.bias": (width,),
            f"{layer}.norm2.weight": (width,), f"{layer}.norm2.bias": (width,),
            f"{layer}.attn.qkv.weight": (3 * width, width),
            f"{layer}.attn.qkv.bias": (3 * width,),
            f"{layer}.attn.proj.weight": (width, width), f"{layer}.attn.proj.bias": (width,),
            f"{layer}.mlp.fc1.weight": (vision["intermediate_size"], width),
            f"{layer}.mlp.fc1.bias": (vision["intermediate_size"],),
            f"{layer}.mlp.fc2.weight": (width, vision["intermediate_size"]),
            f"{layer}.mlp.fc2.bias": (width,),
            f"{layer}.ls1.gamma": (width,), f"{layer}.ls2.gamma": (width,),
        })
    inner = text["hidden_size"]
    for index in range(text["num_layers"]):
        layer = f"text_encoder.transformer.resblocks.{index}"
        shapes.update({
            f"{layer}.ln_1.weight": (inner,), f"{layer}.ln_1.bias": (inner,),
            f"{layer}.ln_2.weight": (inner,), f"{layer}.ln_2.bias": (inner,),
            f"{layer}.attn.in_proj_weight": (3 * inner, inner),
            f"{layer}.attn.in_proj_bias": (3 * inner,),
            f"{layer}.attn.out_proj.weight": (inner, inner),
            f"{layer}.attn.out_proj.bias": (inner,),
            f"{layer}.mlp.c_fc.weight": (text["intermediate_size"], inner),
            f"{layer}.mlp.c_fc.bias": (text["intermediate_size"],),
            f"{layer}.mlp.c_proj.weight": (inner, text["intermediate_size"]),
            f"{layer}.mlp.c_proj.bias": (inner,),
        })
    return shapes


def _require_header(header: dict, cfg: dict) -> None:
    """Every tensor the two graphs read: present, float32 and exactly this shape.

    Checked from the file's header, so nothing is loaded before it is known to be loadable.
    """
    expected = _expected_shapes(cfg)
    missing = sorted(key for key in expected if key not in header)
    if missing:
        raise KeyError("tipsv2 checkpoint is missing expected tensors: " + ", ".join(missing[:8]))
    for key, shape in expected.items():
        actual_shape, dtype = header[key]
        if dtype not in ("F32", "float32"):
            # The published checkpoint is float32 throughout; another storage dtype would need its
            # own measurement before it could be called supported.
            raise NotImplementedError(
                f"tipsv2 is qualified for a float32 checkpoint; {key} is {dtype}")
        if actual_shape != shape:
            raise ValueError(
                f"tipsv2 checkpoint tensor {key} has shape {actual_shape}, expected {shape}")


def _require_weights(weights: dict, cfg: dict) -> None:
    """The same names, dtypes and shapes once loaded, and every value finite."""
    _require_header(
        {key: (tuple(int(size) for size in np.asarray(value).shape),
               str(np.asarray(value).dtype))
         for key, value in weights.items()},
        cfg)
    for key in _expected_shapes(cfg):
        if not np.isfinite(np.asarray(weights[key])).all():
            raise ValueError(f"tipsv2 checkpoint tensor {key} is not finite")


def _builder(verbose: bool):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    # TF32 caps a conv-and-matmul graph near 0.99998 cosine by default, which hides real graph
    # error behind a plausible-looking number.
    config.clear_flag(trt.BuilderFlag.TF32)
    return builder, network, config


def _serialize(builder, network, config, what: str) -> bytes:
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"tipsv2 {what} engine build failed")
    return bytes(plan)


def build_vision_engine(weights, cfg, work_np, *, verbose: bool = False) -> bytes:
    builder, network, config = _builder(verbose)
    size = cfg["vision"]["image_size"]
    # Float32 whatever the engine computes in: the runtime hands over the preprocessed pixels in
    # one dtype, so the bundle's input contract does not change with the build precision. The
    # tower casts to its own dtype before the patch convolution.
    source = network.add_input("pixel_values", trt.float32, (1, 3, size, size))
    embeds = vision_builder.build_vision_tower(network, source, weights, cfg["vision"],
                                               dtype=work_np)
    embeds.name = "image_embeds"
    network.mark_output(embeds)
    return _serialize(builder, network, config, "vision")


def build_text_engine(weights, cfg, work_np, *, verbose: bool = False) -> bytes:
    builder, network, config = _builder(verbose)
    input_ids = network.add_input("input_ids", trt.int32, (cfg["text"]["max_text_len"],))
    embeds = text_builder.build_text_tower(network, input_ids, weights, cfg["text"],
                                           dtype=work_np)
    embeds.name = "text_embeds"
    network.mark_output(embeds)
    return _serialize(builder, network, config, "text")


def build(request, writer) -> None:
    """Build one TIPSv2 bundle."""
    if request.task not in _SUPPORTED_TASKS:
        raise ValueError("tipsv2 supports only task=" + ", ".join(_SUPPORTED_TASKS))
    if request.backend not in {"trt", "trt_rtx"}:
        raise ValueError("tipsv2 supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("tipsv2 does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("tipsv2 does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("tipsv2 does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("tipsv2 does not support context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("tipsv2 does not support video_num_frames")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError(
            f"tipsv2 supports no quantization; got {request.quantization!r}")
    if request.fp32_layers:
        raise NotImplementedError("tipsv2 does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("tipsv2 does not support MTP speculative decoding")

    precision = str(request.precision).lower()
    if precision not in _PRECISIONS:
        raise NotImplementedError(
            f"tipsv2 builds {' or '.join(sorted(_PRECISIONS))}; {precision} is not qualified")
    work_np = _PRECISIONS[precision]

    model_dir = Path(request.model_dir)
    # Header first, tensors second. `config.resolve` reads shapes and nothing else, so the whole
    # geometry and every name, dtype and shape this family needs are settled from a few tens of
    # kilobytes at the front of the file - before gigabytes of it are materialised.
    path = _checkpoint_path(model_dir)
    header = _checkpoint_header(path)
    cfg = config_module.resolve(model_dir, header)
    _require_header(header, cfg)
    weights = _open_checkpoint(path)
    _require_weights(weights, cfg)
    vision, text = cfg["vision"], cfg["text"]

    height = _require_integer(request.image_height, "image_height")
    width = _require_integer(request.image_width, "image_width")
    image_size = height or vision["image_size"]
    if width is not None and width != image_size:
        raise NotImplementedError("tipsv2 needs a square input size")
    if image_size != vision["image_size"]:
        # One position row per patch of the trained grid. A different input size needs the table
        # interpolated, not just the image resized.
        raise NotImplementedError(
            f"tipsv2 input size is fixed at the checkpoint's {vision['image_size']}")

    window = _require_integer(request.max_sequence_length, "max_sequence_length")
    if window is not None and window != text["max_text_len"]:
        raise NotImplementedError(
            "tipsv2 prompt length is fixed at the checkpoint's window of "
            f"{text['max_text_len']}")

    tokenizer_asset, asset = _read_tokenizer(model_dir)
    if asset.pad_id != text["pad_token_id"]:
        # The engine derives its padding mask from this id; a tokenizer that pads with another one
        # would mask the wrong rows.
        raise ValueError(
            f"tipsv2 tokenizer pads with id {asset.pad_id} but config.json says "
            f"{text['pad_token_id']}")
    if len(asset.pieces) > text["vocab_size"]:
        raise ValueError(
            f"tipsv2 tokenizer has {len(asset.pieces)} pieces but the embedding table has "
            f"{text['vocab_size']} rows")

    writer.set_header(family="tipsv2", task=request.task, backend=request.backend)
    writer.add_bytes("vision.plan",
                     build_vision_engine(weights, cfg, work_np, verbose=bool(request.verbose)))
    writer.add_bytes("text.plan",
                     build_text_engine(weights, cfg, work_np, verbose=bool(request.verbose)))
    writer.add_bytes("tokenizer.spm", tokenizer_asset)
    writer.add_json(
        "runtime.json",
        {
            "input_image_h": image_size,
            "input_image_w": image_size,
            "resample": cfg["preprocess"]["resample"],
            "rescale_factor": cfg["preprocess"]["rescale_factor"],
            "max_text_len": text["max_text_len"],
            "pad_token_id": text["pad_token_id"],
            # The text engine gathers from a table of exactly this many rows. The runtime checks
            # the tokenizer it is given against it, so a bundle carrying a tokenizer with more
            # pieces than the table has rows is refused at load rather than indexing out of it.
            "vocab_size": text["vocab_size"],
            "embedding_dim": cfg["embedding_dim"],
            "temperature": cfg["temperature"],
            # Which Unicode release the bundled lowercase table came from. The checkpoint's own
            # tokenizer lowercases with the running Python's table, so a bundle says which one it
            # reproduces rather than leaving it to the host it loads on.
            "unicode_version": tokenizer_export.unicode_version(),
        },
    )


def _read_tokenizer(model_dir: Path):
    path = Path(model_dir) / "tokenizer.model"
    if not path.is_file():
        raise FileNotFoundError(f"missing tipsv2 tokenizer: {path}")
    return tokenizer_export.convert_tokenizer_model(path.read_bytes())
