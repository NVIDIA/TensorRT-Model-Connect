# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a VideoMAE masked-video bundle.

One float32 engine runs ``VideoMAEForPreTraining`` for one clip:

- ``pixel_values`` ``[1, frames, 3, size, size]``: the clip after the
  checkpoint's rescale and normalization;
- ``visible_index`` ``[V]`` and ``masked_index`` ``[M]``: the ascending tube
  numbers that condition the reconstruction and the ones to reconstruct, with
  ``V + M`` the number of tubes;
- ``logits`` ``[1, M, tubelet * patch * patch * 3]``: the prediction of each
  masked tube in its own normalization (``norm_pix_loss``), its values in
  (frame, row, column, channel) order.

The tubelet embedding is applied to the whole clip, the encoder sees only the
visible tubes, and the decoder sees the visible tubes followed by one mask
token per masked tube, as in the reference. Turning predictions back into
pixels is the runtime's job (``runtime/clip.cpp``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorrt as trt

from . import config as config_module, graph as g

_TASK = "masked_video_text_to_video"
# The masking ratio of the checkpoint's pre-training (90 % of the tubes, the
# same spatial pattern in every tubelet); it only sets the shapes the engine
# is tuned for, every split of the tubes runs.
_TUNED_MASK_RATIO = 0.9


def _open_checkpoint(model_dir: Path, cfg: dict) -> dict:
    """Load the safetensors checkpoint this family publishes, once the name, dtype and shape
    of every tensor in its header match the configuration."""
    try:
        import safetensors
    except ImportError as error:
        raise RuntimeError("videomae requires safetensors in the build environment") from error

    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Required safetensors checkpoint is missing: {path}")
    with safetensors.safe_open(str(path), framework="np") as handle:
        header = {}
        for key in handle.keys():
            tensor = handle.get_slice(key)
            header[key] = (tensor.get_dtype(), tuple(tensor.get_shape()))
        _require_tensors(header, cfg)
        return {key: handle.get_tensor(key) for key in header}


def _layer_shapes(prefix: str, width: int, intermediate: int) -> dict[str, tuple[int, ...]]:
    """The tensors of one ``VideoMAELayer`` and their exact shapes."""
    return {prefix + name: shape for name, shape in (
        ("attention.attention.query.weight", (width, width)),
        ("attention.attention.q_bias", (width,)),
        ("attention.attention.key.weight", (width, width)),
        ("attention.attention.value.weight", (width, width)),
        ("attention.attention.v_bias", (width,)),
        ("attention.output.dense.weight", (width, width)),
        ("attention.output.dense.bias", (width,)),
        ("intermediate.dense.weight", (intermediate, width)),
        ("intermediate.dense.bias", (intermediate,)),
        ("output.dense.weight", (width, intermediate)),
        ("output.dense.bias", (width,)),
        ("layernorm_before.weight", (width,)),
        ("layernorm_before.bias", (width,)),
        ("layernorm_after.weight", (width,)),
        ("layernorm_after.bias", (width,)),
    )}


def _expected_shapes(cfg: dict) -> dict[str, tuple[int, ...]]:
    """Every tensor the family runs, with the exact shape the configuration implies."""
    encoder, decoder = cfg["encoder"], cfg["decoder"]
    width, decoder_width = encoder["width"], decoder["width"]
    shapes = {
        "videomae.embeddings.patch_embeddings.projection.weight":
            (width, 3, cfg["tubelet_size"], cfg["patch_size"], cfg["patch_size"]),
        "videomae.embeddings.patch_embeddings.projection.bias": (width,),
        "encoder_to_decoder.weight": (decoder_width, width),
        "mask_token": (1, 1, decoder_width),
        "decoder.norm.weight": (decoder_width,),
        "decoder.norm.bias": (decoder_width,),
        "decoder.head.weight": (cfg["tube_values"], decoder_width),
        "decoder.head.bias": (cfg["tube_values"],),
    }
    if cfg["final_layernorm"]:
        shapes["videomae.layernorm.weight"] = (width,)
        shapes["videomae.layernorm.bias"] = (width,)
    for index in range(encoder["layers"]):
        shapes.update(_layer_shapes(f"videomae.encoder.layer.{index}.", width,
                                    encoder["intermediate"]))
    for index in range(decoder["layers"]):
        shapes.update(_layer_shapes(f"decoder.decoder_layers.{index}.", decoder_width,
                                    decoder["intermediate"]))
    return shapes


def _require_tensors(tensors: dict[str, tuple[str, tuple[int, ...]]], cfg: dict) -> None:
    """Check each tensor's name, safetensors dtype code and shape against the configuration."""
    shapes = _expected_shapes(cfg)
    missing = [key for key in shapes if key not in tensors]
    if missing:
        raise KeyError("videomae checkpoint is missing expected tensors: " + ", ".join(missing))
    unexpected = sorted(set(tensors) - set(shapes))
    if unexpected:
        raise KeyError("videomae checkpoint has tensors this family does not run: "
                       + ", ".join(unexpected[:8]))
    not_float32 = sorted(key for key in shapes if tensors[key][0] != "F32")
    if not_float32:
        raise TypeError("videomae builds float32 checkpoints only: " + ", ".join(not_float32[:8]))
    for key, shape in shapes.items():
        if tensors[key][1] != shape:
            raise ValueError(f"videomae tensor {key} has shape {tensors[key][1]}, expected {shape}")


def _require_weights(weights: dict, cfg: dict) -> None:
    """The same checks for tensors already in memory."""
    _require_tensors({key: ("F32" if value.dtype == np.float32 else str(value.dtype),
                            tuple(value.shape)) for key, value in weights.items()}, cfg)


def _builder(verbose: bool):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    # TF32 would round every float32 matrix product and convolution input to a
    # 10-bit mantissa; the family is a float32 build of a float32 checkpoint.
    config.clear_flag(trt.BuilderFlag.TF32)
    return builder, network, config


def _tuned_visible(cfg: dict) -> int:
    per_tubelet = cfg["grid"] * cfg["grid"]
    visible = (per_tubelet - int(_TUNED_MASK_RATIO * per_tubelet)) * (
        cfg["num_frames"] // cfg["tubelet_size"])
    return min(max(visible, 1), cfg["tubes"] - 1)


def build_engine(weights: dict, cfg: dict, *, verbose: bool = False) -> bytes:
    """Serialize the float32 engine of one VideoMAE pre-training checkpoint."""
    builder, network, config = _builder(verbose)
    frames, size = cfg["num_frames"], cfg["image_size"]
    encoder, decoder = cfg["encoder"], cfg["decoder"]
    eps = cfg["layer_norm_eps"]

    pixels = network.add_input("pixel_values", trt.float32, (1, frames, 3, size, size))
    visible = network.add_input("visible_index", trt.int32, (-1,))
    masked = network.add_input("masked_index", trt.int32, (-1,))

    # Tubelet embedding: a Conv3d whose stride equals its kernel, over [1, 3, T, H, W].
    channels_first = network.add_shuffle(pixels)
    channels_first.first_transpose = (0, 2, 1, 3, 4)
    projection = network.add_convolution_nd(
        channels_first.get_output(0), encoder["width"],
        (cfg["tubelet_size"], cfg["patch_size"], cfg["patch_size"]),
        trt.Weights(np.ascontiguousarray(
            weights["videomae.embeddings.patch_embeddings.projection.weight"])),
        trt.Weights(np.ascontiguousarray(
            weights["videomae.embeddings.patch_embeddings.projection.bias"])))
    projection.stride_nd = (cfg["tubelet_size"], cfg["patch_size"], cfg["patch_size"])
    tokens = network.add_shuffle(projection.get_output(0))
    tokens.reshape_dims = (1, encoder["width"], cfg["tubes"])
    tokens.second_transpose = (0, 2, 1)
    rows = g.add(network, tokens.get_output(0),
                 g.constant(network, g.sinusoid_table(cfg["tubes"], encoder["width"])[None]))

    # The encoder runs on the visible tubes only.
    rows = network.add_gather(rows, visible, 1).get_output(0)
    for index in range(encoder["layers"]):
        rows = g.transformer_layer(network, rows, weights, f"videomae.encoder.layer.{index}.",
                                   encoder["heads"], eps)
    if cfg["final_layernorm"]:
        rows = g.layer_norm(network, rows, weights["videomae.layernorm.weight"],
                            weights["videomae.layernorm.bias"], eps)
    rows = g.linear(network, rows, weights["encoder_to_decoder.weight"])

    # Decoder input: the visible tubes, then a mask token per masked tube, each
    # with the decoder position of its tube.
    positions = g.constant(network, g.sinusoid_table(cfg["tubes"], decoder["width"])[None])
    shown = g.add(network, rows, network.add_gather(positions, visible, 1).get_output(0))
    hidden = g.add(network, g.constant(network, weights["mask_token"]),
                   network.add_gather(positions, masked, 1).get_output(0))
    sequence = network.add_concatenation([shown, hidden])
    sequence.axis = 1
    rows = sequence.get_output(0)
    for index in range(decoder["layers"]):
        rows = g.transformer_layer(network, rows, weights, f"decoder.decoder_layers.{index}.",
                                   decoder["heads"], eps)
    rows = g.last_rows(network, rows, visible, masked, decoder["width"])
    # nn.LayerNorm's default epsilon: the decoder norm does not take layer_norm_eps.
    rows = g.layer_norm(network, rows, weights["decoder.norm.weight"],
                        weights["decoder.norm.bias"], 1e-5)
    logits = g.linear(network, rows, weights["decoder.head.weight"], weights["decoder.head.bias"])
    logits.name = "logits"
    network.mark_output(logits)

    tuned = _tuned_visible(cfg)
    profile = builder.create_optimization_profile()
    profile.set_shape("visible_index", (1,), (tuned,), (cfg["tubes"] - 1,))
    profile.set_shape("masked_index", (1,), (cfg["tubes"] - tuned,), (cfg["tubes"] - 1,))
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("videomae engine build failed")
    return bytes(plan)


def build(request, writer) -> None:
    """Build one VideoMAE bundle for the masked_video_text_to_video Task."""
    if request.task != _TASK:
        raise ValueError(f"videomae supports only task={_TASK}")
    if request.backend != "trt":
        raise ValueError("videomae supports only backend=trt")
    if str(request.precision).lower() != "fp32":
        raise NotImplementedError(
            f"videomae builds the checkpoint's float32 only; {request.precision} is not qualified")
    if request.quantization not in (None, "none"):
        raise NotImplementedError(f"videomae does not support quantization {request.quantization!r}")
    if request.fp32_layers:
        raise NotImplementedError("videomae does not support mixed-precision layers")
    for name, value in (("max_batch_size", request.max_batch_size),
                        ("tensor_parallel_size", request.tensor_parallel_size),
                        ("context_parallel_size", request.context_parallel_size),
                        ("image_height", request.image_height),
                        ("image_width", request.image_width),
                        ("video_num_frames", request.video_num_frames)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"videomae needs an integer {name}; got {value!r}")
    if request.max_batch_size != 1:
        raise NotImplementedError("videomae reconstructs one clip per request")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("videomae does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("videomae does not support context parallelism")
    if request.dynamic_kv_cache:
        raise NotImplementedError("videomae does not support dynamic_kv_cache")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("videomae does not support MTP speculative decoding")
    if request.max_sequence_length is not None:
        # The tubes follow from the clip size; every split of them into shown and
        # reconstructed tubes runs on the one engine.
        raise NotImplementedError("videomae derives its sequence lengths from the clip size")

    model_dir = Path(request.model_dir)
    cfg = config_module.resolve(model_dir)
    for value, expected, name in (
        (request.image_height, cfg["image_size"], "image_height"),
        (request.image_width, cfg["image_size"], "image_width"),
        (request.video_num_frames, cfg["num_frames"], "video_num_frames"),
    ):
        if value is not None and value != expected:
            # The position tables have one row per tube of the trained clip.
            raise NotImplementedError(
                f"videomae {name} is fixed at the checkpoint's {expected}; got {value}")
    weights = _open_checkpoint(model_dir, cfg)

    writer.set_header(family="videomae", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", build_engine(weights, cfg, verbose=bool(request.verbose)))
    writer.add_json(
        "runtime.json",
        {
            "num_frames": cfg["num_frames"],
            "image_size": cfg["image_size"],
            "patch_size": cfg["patch_size"],
            "tubelet_size": cfg["tubelet_size"],
            "image_mean": cfg["image_mean"],
            "image_std": cfg["image_std"],
        },
    )
