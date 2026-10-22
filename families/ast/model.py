# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build an AST bundle.

One engine. AST is a ViT whose patches are cut from a mel spectrogram instead
of an image, so everything from the patch projection onwards is the encoder
stack this repo already builds; the family's own work is the Kaldi filter-bank
frontend, and that is arithmetic over the waveform rather than a graph.

The engine takes the normalised spectrogram the reference feature extractor
produces, in the reference's own ``[max_length, num_mel_bins]`` layout, so the
runtime's frontend can be compared against it directly. The analysis window and
the mel filters are computed once here and shipped beside the plan.
"""

from __future__ import annotations

from pathlib import Path

import ml_dtypes  # noqa: F401
import numpy as np
import tensorrt as trt

from . import config as config_module, encoder_builder, fbank, graph as g

_SUPPORTED_TASKS = ("audio_to_class_scores",)
_PREFIX = "audio_spectrogram_transformer"


def _open_safetensors(model_dir: Path):
    """Open the family's safetensors checkpoint."""
    from safetensors import safe_open

    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Required safetensors checkpoint is missing: {path}")
    return safe_open(str(path), framework="numpy")


def _load(model_dir: Path):
    """Read every tensor into memory this process owns.

    ``get_tensor`` can hand back a view over the memory-mapped file, and the
    handle is closed on the way out of this function. The copy is what makes
    the returned arrays outlive it.
    """
    with _open_safetensors(model_dir) as handle:
        shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()}
        weights = {key: np.array(handle.get_tensor(key), copy=True) for key in handle.keys()}
    return weights, shapes


def _require_weights(weights: dict) -> None:
    required = (
        f"{_PREFIX}.embeddings.cls_token",
        f"{_PREFIX}.embeddings.distillation_token",
        f"{_PREFIX}.embeddings.position_embeddings",
        f"{_PREFIX}.embeddings.patch_embeddings.projection.weight",
        f"{_PREFIX}.layernorm.weight",
        "classifier.layernorm.weight",
        "classifier.dense.weight",
    )
    missing = [key for key in required if key not in weights]
    if missing:
        raise KeyError("ast checkpoint is missing expected tensors: " + ", ".join(missing))


def _builder(precision, verbose):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    if precision == "fp32":
        # TF32 caps a conv-and-matmul graph near 0.99998 cosine by default,
        # which hides real graph error behind a plausible-looking number.
        config.clear_flag(trt.BuilderFlag.TF32)
    return builder, network, config


def _embed_patches(network, spectrogram, weights, cfg, *, dtype):
    """Cut the patch grid and prepend the two leading tokens."""
    rows = network.add_shuffle(spectrogram)
    # The projection sweeps frequency down the rows and time across them, so
    # the reference's [time, frequency] features are transposed on the way in.
    rows.reshape_dims = (1, 1, cfg["max_length"], cfg["num_mel_bins"])
    rows.second_transpose = (0, 1, 3, 2)
    grid = g.add_conv2d(
        network, rows.get_output(0),
        weights[f"{_PREFIX}.embeddings.patch_embeddings.projection.weight"],
        weights[f"{_PREFIX}.embeddings.patch_embeddings.projection.bias"],
        stride=(cfg["frequency_stride"], cfg["time_stride"]), dtype=dtype)

    flat = network.add_shuffle(grid)
    # [1, width, frequency, time] -> [patch, width], frequency-major, which is
    # the order the position table was trained in.
    flat.reshape_dims = (cfg["hidden_size"], cfg["num_patches"])
    flat.second_transpose = (1, 0)
    patches = flat.get_output(0)

    leading = [
        g.add_constant(network, (1, cfg["hidden_size"]),
                       weights[f"{_PREFIX}.embeddings.{name}"].reshape(1, cfg["hidden_size"]),
                       dtype=dtype)
        for name in ("cls_token", "distillation_token")
    ]
    sequence = network.add_concatenation([*leading, patches])
    sequence.axis = 0
    positions = g.add_constant(
        network, (cfg["seq_len"], cfg["hidden_size"]),
        weights[f"{_PREFIX}.embeddings.position_embeddings"].reshape(
            cfg["seq_len"], cfg["hidden_size"]), dtype=dtype)
    return g.add_sum(network, sequence.get_output(0), positions)


def _classify(network, hidden, weights, cfg, *, dtype):
    """Normalise every row, average the two leading ones, then score."""
    hidden = g.add_layer_norm(network, hidden, weights[f"{_PREFIX}.layernorm.weight"],
                              weights[f"{_PREFIX}.layernorm.bias"], cfg["eps"])
    # The normalisation runs before the pooling, over the whole sequence. Row
    # normalisation does not commute with the average, so the order is not free.
    leading = network.add_slice(hidden, (0, 0), (2, cfg["hidden_size"]), (1, 1)).get_output(0)
    pooled = network.add_reduce(leading, trt.ReduceOperation.AVG, 1 << 0, True).get_output(0)
    pooled = g.add_layer_norm(network, pooled, weights["classifier.layernorm.weight"],
                              weights["classifier.layernorm.bias"], cfg["eps"])
    return g.add_linear(network, pooled, weights["classifier.dense.weight"],
                        weights["classifier.dense.bias"], dtype=dtype)


def build_engine(weights, cfg, *, precision, verbose=False) -> bytes:
    builder, network, config = _builder(precision, verbose)
    work_np, work_trt = ((np.float16, trt.float16) if precision == "fp16"
                         else (np.float32, trt.float32))
    features = network.add_input("input_values", trt.float32,
                                 (cfg["max_length"], cfg["num_mel_bins"]))
    source = (features if features.dtype == work_trt
              else network.add_cast(features, work_trt).get_output(0))

    hidden = _embed_patches(network, source, weights, cfg, dtype=work_np)
    hidden = encoder_builder.build_layers(network, hidden, weights, _PREFIX, cfg, dtype=work_np)
    logits = _classify(network, hidden, weights, cfg, dtype=work_np)

    flat = network.add_shuffle(logits)
    flat.reshape_dims = (cfg["num_labels"],)
    logits = flat.get_output(0)
    if logits.dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("ast engine build failed")
    return bytes(plan)


def build(request, writer) -> None:
    """Build one AST bundle."""
    if request.task not in _SUPPORTED_TASKS:
        raise ValueError("ast supports only task=" + ", ".join(_SUPPORTED_TASKS))
    if request.backend not in {"trt", "trt_rtx"}:
        raise ValueError("ast supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("ast does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("ast does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("ast does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("ast does not support context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("ast does not support video_num_frames")
    if request.image_height is not None or request.image_width is not None:
        # AST's patches are cut from a spectrogram, so there is no image to size.
        raise NotImplementedError("ast takes audio, not an image size")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("ast does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("ast does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("ast does not support MTP speculative decoding")

    precision = str(request.precision).lower()
    if precision not in {"fp16", "fp32"}:
        raise ValueError(f"Unsupported ast precision: {precision}")

    model_dir = Path(request.model_dir)
    weights, shapes = _load(model_dir)
    _require_weights(weights)
    cfg = config_module.resolve(model_dir, shapes)

    if request.max_sequence_length is not None:
        if int(request.max_sequence_length) != cfg["max_length"]:
            # One position row per patch of the trained grid. A different
            # spectrogram length needs the table rebuilt, not resized.
            raise NotImplementedError(
                f"ast spectrogram length is fixed at the checkpoint's {cfg['max_length']}")

    writer.set_header(family="ast", task=request.task, backend=request.backend)
    writer.add_bytes("model.plan", build_engine(
        weights, cfg, precision=precision, verbose=bool(request.verbose)))
    writer.add_bytes("window.f32", fbank.hann_window(cfg["frame_length"]).tobytes())
    writer.add_bytes("mel_filters.f32", fbank.mel_filters(
        cfg["num_mel_bins"], cfg["n_fft"], float(cfg["sample_rate"]),
        low_freq=config_module.LOW_FREQ).tobytes())
    writer.add_json(
        "runtime.json",
        {
            "sample_rate": cfg["sample_rate"],
            "frame_length": cfg["frame_length"],
            "frame_shift": cfg["frame_shift"],
            "n_fft": cfg["n_fft"],
            "num_mel_bins": cfg["num_mel_bins"],
            "max_length": cfg["max_length"],
            "mel_floor": config_module.MEL_FLOOR,
            "preemphasis": config_module.PREEMPHASIS,
            "feature_mean": cfg["feature_mean"],
            "feature_scale": cfg["feature_scale"],
            "labels": cfg["labels"],
        },
    )
