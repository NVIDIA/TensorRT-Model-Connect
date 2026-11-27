# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the TIPSv2 geometry from the tensors, and refuse a ``config.json`` that contradicts them.

The checkpoint ships its own modelling code (``auto_map``), and that code reads ``config.json``
selectively: the **vision** tower's head count and MLP ratio come from a factory chosen by
``(hidden_size, num_hidden_layers)`` and ignore ``vision_config.num_attention_heads`` and
``vision_config.mlp_ratio`` entirely, while the **text** tower does read its four ``text_config``
numbers. A builder that trusts the file for the vision head count would reshape the same weights
into a different model on any checkpoint whose file and factory disagree.

So the geometry here is derived from tensor shapes and from the same factory table the reference
uses, and every number ``config.json`` also states is checked against it and refused on
disagreement. The file is treated as a claim; the tensors and the reference's own rules are the
evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# The reference picks the vision factory by geometry and takes the head count and the MLP ratio
# from it, not from config.json: `_VISION_FN_BY_GEOMETRY` in configuration_tips.py and the
# `vit_*` factories in image_encoder.py. Only so400m is qualified here; see `resolve`.
_VISION_FACTORY = {
    # (hidden_size, num_hidden_layers): (name, num_heads, intermediate_size)
    (1152, 27): ("vit_so400m", 16, 4304),
}

# Both are fixed by the reference's module construction, not by the file.
_VISION_EPS = 1e-6  # functools.partial(nn.LayerNorm, eps=1e-6) in VisionTransformer
_TEXT_EPS = 1e-5  # nn.LayerNorm default, used by ResidualAttentionBlock and ln_final
_POOLING_EPS = 1e-8  # GlobalAvgPooling default

# What the runtime's int32 fields hold. A config states numbers Python keeps exactly at any size;
# they have to fit where the bundle will put them.
_INT32_MAX = 2**31 - 1

# The vision MLP is nn.GELU() with its default approximate="none": the error-function GELU, which
# differs from the tanh approximation by about 1e-3. The text MLP is nn.ReLU. Neither is read from
# the file, so a file that claims something else is refused rather than silently ignored.
_VISION_ACT = "gelu"
_TEXT_ACT = "relu"

_RESAMPLE_BILINEAR = 2  # PIL.Image.Resampling.BILINEAR, as processor_config.json spells it


def _no_duplicate_keys(pairs):
    """``json.load``'s object hook: the last of two identical keys silently wins otherwise.

    A ``config.json`` that states ``hidden_size`` twice is making two claims, and the one the
    parser keeps is whichever came last. Both are refused instead.
    """
    seen: dict = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"tipsv2 configuration states {key!r} more than once")
        seen[key] = value
    return seen


def _load_json(path: Path) -> dict:
    loaded = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
    if not isinstance(loaded, dict):
        raise ValueError(f"tipsv2 expects an object in {path.name}")
    return loaded


def _shape(weights, key: str) -> tuple:
    """One tensor's shape, from a loaded array or from a ``(shape, dtype)`` header pair.

    Taking both is what lets this resolver run before the checkpoint is materialised: it reads
    shapes and nothing else, so ``model.py`` hands it the file's header first and the loaded
    tensors only once the geometry is settled.
    """
    if key not in weights:
        raise KeyError(f"tipsv2 checkpoint is missing '{key}'")
    entry = weights[key]
    shape = entry[0] if isinstance(entry, tuple) else entry.shape
    return tuple(int(size) for size in shape)


#: What ``dict.get`` cannot express: a key that is not there at all, as opposed to one whose value
#: is ``null``. A checkpoint may leave a claim out; a checkpoint that states ``null`` is making a
#: claim of nothing and is refused rather than silently treated as absent.
_ABSENT = object()


def _claim(stated: dict, key: str):
    return stated.get(key, _ABSENT)


def _section(config: dict, name: str) -> dict:
    """One of ``config.json``'s nested objects, absent-or-an-object.

    ``config.get(name) or {}`` reads ``null``, ``false``, ``0`` and ``[]`` as "not there", which
    is how an explicitly empty or malformed section would skip every check below it.
    """
    section = _claim(config, name)
    if section is _ABSENT:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"tipsv2 config.json {name} must be an object, not {section!r}")
    return section


def _agree(stated, derived, what: str) -> None:
    """Refuse a config.json claim that the tensors or the reference's own rules contradict.

    The claim must be an actual JSON integer. ``int()`` would accept ``1152.9`` and ``"1152"`` and
    then agree with a model that is neither.
    """
    if stated is _ABSENT:
        return
    if stated is None:
        raise ValueError(f"tipsv2 config.json states {what}=null; remove the key or give a value")
    if isinstance(stated, bool) or not isinstance(stated, int):
        raise ValueError(
            f"tipsv2 config.json {what} must be an integer, not {type(stated).__name__}")
    if stated != derived:
        raise ValueError(f"tipsv2 config.json says {what}={stated} but the model says {derived}")


def _count_blocks(weights, needle: str) -> int:
    indices = set()
    for key in weights:
        if key.startswith(needle):
            rest = key[len(needle):].split(".", 1)[0]
            if not rest.isdigit() or (len(rest) > 1 and rest[0] == "0"):
                raise ValueError(f"tipsv2 block index {rest!r} is not a canonical decimal")
            indices.add(int(rest))
    if not indices:
        raise KeyError(f"tipsv2 checkpoint has no blocks under '{needle}'")
    if indices != set(range(len(indices))):
        raise ValueError(f"tipsv2 block indices under '{needle}' are not contiguous")
    return len(indices)


def _require_bool(stated, expected: bool, what: str) -> None:
    """An absent claim is fine; `null` is a claim of nothing and anything but a JSON boolean is a
    claim this family cannot read."""
    if stated is _ABSENT:
        return
    if stated is None:
        raise ValueError(f"tipsv2 config.json states {what}=null; remove the key or give a value")
    if not isinstance(stated, bool):
        raise ValueError(f"tipsv2 config.json {what} must be a boolean, not {type(stated).__name__}")
    if stated is not expected:
        raise NotImplementedError(f"tipsv2 is qualified for {what}={expected}, not {stated}")


def _require_number(stated, what: str) -> float:
    if isinstance(stated, bool) or not isinstance(stated, (int, float)):
        raise ValueError(f"tipsv2 config.json {what} must be a number, not {type(stated).__name__}")
    value = float(stated)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"tipsv2 config.json {what} must be finite; got {stated!r}")
    return value


def _require_float32(stated, what: str) -> float:
    """A positive number that is still a usable positive number once stored as float32.

    ``runtime.json`` carries these to a C++ float, so being finite in float64 is not enough:
    1e-50 becomes exactly zero there and 1e39 becomes an infinity, and both reach arithmetic that
    divides by them. The value is checked in the type it will be stored in.
    """
    value = _require_number(stated, what)
    if value <= 0.0:
        raise ValueError(f"tipsv2 config.json {what} must be positive; got {stated!r}")
    narrowed = float(np.float32(value))
    if narrowed <= 0.0 or not np.isfinite(np.float32(value)):
        raise ValueError(
            f"tipsv2 config.json {what}={stated!r} is not a positive float32 (it becomes "
            f"{narrowed!r})")
    # The checkpoint's own value travels on; the runtime narrows it to the same float32 this
    # checked. Narrowing here as well would only make the bundle's JSON harder to read.
    return value


def _vision(weights, config: dict) -> dict:
    stated = _section(config, "vision_config")

    out_channels, channels, kernel_h, kernel_w = _shape(weights, "vision_encoder.patch_embed.proj.weight")
    if channels != 3:
        raise NotImplementedError(f"tipsv2 takes 3-channel images, not {channels}")
    if kernel_h != kernel_w:
        raise NotImplementedError("tipsv2 needs a square patch")
    _agree(_claim(stated, "num_channels"), channels, "vision_config.num_channels")
    _agree(_claim(stated, "patch_size"), kernel_h, "vision_config.patch_size")

    layers = _count_blocks(weights, "vision_encoder.blocks.")
    _agree(_claim(stated, "num_hidden_layers"), layers, "vision_config.num_hidden_layers")
    _agree(_claim(stated, "hidden_size"), out_channels, "vision_config.hidden_size")

    key = (out_channels, layers)
    if key not in _VISION_FACTORY:
        raise NotImplementedError(
            f"tipsv2 is qualified for the vision geometries {sorted(_VISION_FACTORY)}, "
            f"not hidden_size={out_channels} with {layers} layers")
    factory, heads, intermediate = _VISION_FACTORY[key]

    # The factory's MLP ratio has to land on the stored width; check the tensor rather than
    # recompute int(hidden * ratio), whose rounding is exactly what a ratio read from the file
    # would get wrong.
    fc1_out, fc1_in = _shape(weights, "vision_encoder.blocks.0.mlp.fc1.weight")
    if fc1_in != out_channels or fc1_out != intermediate:
        raise ValueError(
            f"tipsv2 {factory} expects an MLP of {out_channels} -> {intermediate}, "
            f"but the tensors say {fc1_in} -> {fc1_out}")
    # The reference ignores vision_config.num_attention_heads and vision_config.mlp_ratio; it
    # takes both from the factory. Refuse a file that disagrees with the factory rather than
    # build a different model than the file describes.
    _agree(_claim(stated, "num_attention_heads"), heads, "vision_config.num_attention_heads")
    ratio = _claim(stated, "mlp_ratio")
    if ratio is not _ABSENT:
        # The stated ratio has to land on the stored width the way the reference's
        # `int(dim * mlp_ratio)` would, which is the only use it could have had.
        stated_width = int(_require_number(ratio, "vision_config.mlp_ratio") * out_channels)
        if stated_width != intermediate:
            raise ValueError(
                f"tipsv2 config.json says vision_config.mlp_ratio={ratio}, which is an MLP of "
                f"{stated_width}, but {factory} and the tensors say {intermediate}")
    if out_channels % heads:
        raise ValueError(f"tipsv2 vision width {out_channels} is not divisible by {heads} heads")

    qkv_out, qkv_in = _shape(weights, "vision_encoder.blocks.0.attn.qkv.weight")
    if qkv_in != out_channels or qkv_out != 3 * out_channels:
        raise ValueError("tipsv2 vision attention projection is not a packed QKV tensor")
    if "vision_encoder.blocks.0.attn.qkv.bias" not in weights:
        # qkv_bias=True in every factory; a checkpoint without it is a different tower.
        raise KeyError("tipsv2 vision attention has no QKV bias")

    # swiglu would replace fc1/fc2 with w12/w3 and a different hidden width.
    _require_bool(_claim(stated, "use_swiglu_ffn"), False, "vision_config.use_swiglu_ffn")
    activation = stated.get("hidden_act", _VISION_ACT)
    if activation != _VISION_ACT:
        raise NotImplementedError(
            f"tipsv2 vision is qualified for the error-function {_VISION_ACT}, not {activation!r}")

    registers, register_width = _shape(weights, "vision_encoder.register_tokens")[-2:]
    if register_width != out_channels:
        raise ValueError("tipsv2 register token width disagrees with the encoder width")
    _agree(_claim(stated, "num_register_tokens"), registers, "vision_config.num_register_tokens")
    if registers < 1:
        # x_norm[:, 1 : registers + 1] would be empty and the patch rows would start one early.
        raise NotImplementedError("tipsv2 expects at least one register token")

    cls_width = _shape(weights, "vision_encoder.cls_token")[-1]
    if cls_width != out_channels:
        raise ValueError("tipsv2 class token width disagrees with the encoder width")

    num_positions, position_width = _shape(weights, "vision_encoder.pos_embed")[-2:]
    if position_width != out_channels:
        raise ValueError("tipsv2 position table width disagrees with the encoder width")
    # One row for the class token, then one per patch of the trained grid. The register tokens
    # are inserted after the table is added, so they carry no position of their own.
    grid = int(round((num_positions - 1) ** 0.5))
    if grid * grid != num_positions - 1:
        raise NotImplementedError(
            f"tipsv2 needs a class row plus a square patch grid, not {num_positions} positions")
    image_size = grid * kernel_h
    _agree(_claim(stated, "image_size"), image_size, "vision_config.image_size")

    return {
        "factory": factory,
        "hidden_size": out_channels,
        "intermediate_size": intermediate,
        "num_layers": layers,
        "num_heads": heads,
        "head_dim": out_channels // heads,
        "patch_size": kernel_h,
        "grid": grid,
        "num_positions": num_positions,
        "num_register_tokens": registers,
        "image_size": image_size,
        "eps": _VISION_EPS,
    }


def _text(weights, config: dict) -> dict:
    stated = _section(config, "text_config")

    vocab_size, hidden = _shape(weights, "text_encoder.token_embedding.weight")
    _agree(_claim(stated, "vocab_size"), vocab_size, "text_config.vocab_size")
    _agree(_claim(stated, "hidden_size"), hidden, "text_config.hidden_size")

    layers = _count_blocks(weights, "text_encoder.transformer.resblocks.")
    _agree(_claim(stated, "num_hidden_layers"), layers, "text_config.num_hidden_layers")

    intermediate, fc_in = _shape(weights, "text_encoder.transformer.resblocks.0.mlp.c_fc.weight")
    if fc_in != hidden:
        raise ValueError("tipsv2 text MLP input width disagrees with the encoder width")
    _agree(_claim(stated, "intermediate_size"), intermediate, "text_config.intermediate_size")

    packed, packed_width = _shape(weights, "text_encoder.transformer.resblocks.0.attn.in_proj_weight")
    if packed_width != hidden or packed != 3 * hidden:
        raise ValueError("tipsv2 text attention projection is not a packed QKV tensor")

    # The text tower reads its head count from the file: TIPSv2Config passes
    # text_config.num_attention_heads straight into nn.MultiheadAttention, and no tensor shape
    # carries it. An absent value would silently become the reference's 12-head default on a
    # 16-head checkpoint, so it is required.
    heads = stated.get("num_attention_heads")
    if heads is None:
        raise ValueError("tipsv2 text_config must state num_attention_heads")
    if isinstance(heads, bool) or not isinstance(heads, int):
        raise ValueError("tipsv2 text_config.num_attention_heads must be an integer")
    if heads <= 0 or hidden % heads:
        raise ValueError(f"tipsv2 text width {hidden} is not divisible by {heads} heads")

    activation = stated.get("hidden_act", _TEXT_ACT)
    if activation != _TEXT_ACT:
        raise NotImplementedError(
            f"tipsv2 text is qualified for {_TEXT_ACT}, not {activation!r}")
    # Without it the embedding is not scaled by sqrt(width) and every row changes.
    _require_bool(_claim(stated, "scale_sqrt_depth"), True, "text_config.scale_sqrt_depth")

    # The prompt window. The text tower has no learned position table - the sinusoid is computed
    # for any length - so this is a capacity, not a geometry: padding rows are excluded from the
    # keys, from the MLP and from the pooled average, so a prompt that fits gives the same
    # embedding at any window at least as long. It is still fixed into the engine, because the
    # reference's own tokenizer truncates at it.
    max_text_len = stated.get("max_position_embeddings")
    if max_text_len is None:
        raise ValueError("tipsv2 text_config must state max_position_embeddings")
    if isinstance(max_text_len, bool) or not isinstance(max_text_len, int) or max_text_len <= 0:
        raise ValueError("tipsv2 text_config.max_position_embeddings must be a positive integer")
    # The runtime holds the window in an int32 and the engine carries a sinusoid of
    # `window x width` float32 entries. Refuse a value either of them cannot represent here,
    # where the file that stated it is still in hand, rather than during the build or at load.
    if max_text_len > _INT32_MAX or max_text_len * hidden > _INT32_MAX:
        raise ValueError(
            f"tipsv2 text_config.max_position_embeddings {max_text_len} is past what this "
            f"family builds: the window and its {hidden}-wide position signal must both be "
            f"representable in an int32")

    # The pooled average divides by `count + epsilon`, in float32 whatever the engine builds in,
    # so an epsilon that narrows to zero would let an all-padding prompt divide zero by zero.
    #
    # **The reference never reads this key.** `TextEncoder` builds `GlobalAvgPooling` without an
    # epsilon argument, so its divisor is always the 1e-8 default and `configuration_tips.py`
    # does not carry `pooling_epsilon` into it. Obeying a stated value would therefore build an
    # engine that differs from its own reference by exactly as much as the value differs - a
    # config saying 1e-3 moves a short prompt's embedding two orders of magnitude past this
    # family's own gate, with nothing to show for it. So it is checked like every other claim the
    # file makes and refused when it disagrees.
    pooling_eps = _POOLING_EPS
    if "pooling_epsilon" in stated:
        stated_eps = _require_float32(stated["pooling_epsilon"], "text_config.pooling_epsilon")
        if stated_eps != pooling_eps:
            raise NotImplementedError(
                f"tipsv2 text_config.pooling_epsilon states {stated_eps}, but the reference's "
                f"GlobalAvgPooling always divides by count + {pooling_eps} and never reads the "
                f"key; an engine built with {stated_eps} would not match it")

    pad_token_id = stated.get("pad_token_id")
    if pad_token_id is None:
        raise ValueError("tipsv2 text_config must state pad_token_id")
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int) or pad_token_id < 0:
        raise ValueError("tipsv2 text_config.pad_token_id must be a non-negative integer")
    if pad_token_id >= vocab_size:
        raise ValueError(
            f"tipsv2 pad_token_id {pad_token_id} is outside the {vocab_size}-row embedding table")

    return {
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_layers": layers,
        "num_heads": heads,
        "head_dim": hidden // heads,
        "vocab_size": vocab_size,
        "max_text_len": max_text_len,
        "pad_token_id": pad_token_id,
        "pooling_eps": pooling_eps,
        "eps": _TEXT_EPS,
    }


def _preprocess(model_dir: Path, vision: dict) -> dict:
    """The image rule, read from ``processor_config.json``.

    TIPSv2 is the rare vision model with **no** ImageNet normalisation: the card and the processor
    both say the tower takes pixels in [0, 1]. Subtracting a mean here would move every pixel and
    nothing loudly, so ``do_normalize`` is required to be false rather than defaulted.
    """
    path = model_dir / "processor_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing tipsv2 processor config: {path}")
    loaded = _load_json(path)
    image = loaded.get("image_processor")
    if not isinstance(image, dict):
        raise ValueError("tipsv2 processor_config.json has no image_processor object")

    _require_bool(_claim(image, "do_resize"), True, "image_processor.do_resize")
    _require_bool(_claim(image, "do_rescale"), True, "image_processor.do_rescale")
    _require_bool(_claim(image, "do_normalize"), False, "image_processor.do_normalize")
    _require_bool(_claim(image, "do_center_crop"), False, "image_processor.do_center_crop")
    _require_bool(_claim(image, "do_convert_rgb"), True, "image_processor.do_convert_rgb")

    size = image.get("size")
    if not isinstance(size, dict) or "height" not in size or "width" not in size:
        raise ValueError("tipsv2 image_processor.size must state height and width")
    height, width = size["height"], size["width"]
    for name, value in (("height", height), ("width", width)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"tipsv2 image_processor.size.{name} must be a positive integer")
    if height != vision["image_size"] or width != vision["image_size"]:
        # One position row per patch of the trained grid; another size needs the table
        # interpolated, not just the image resized.
        raise NotImplementedError(
            f"tipsv2 is fixed at the checkpoint's {vision['image_size']}x{vision['image_size']}, "
            f"but the processor asks for {height}x{width}")

    resample = image.get("resample", _RESAMPLE_BILINEAR)
    if isinstance(resample, bool) or not isinstance(resample, int):
        raise ValueError("tipsv2 image_processor.resample must be an integer")
    if resample != _RESAMPLE_BILINEAR:
        # 2 is Pillow's bilinear, which the runtime reproduces coefficient for coefficient.
        raise NotImplementedError(
            f"tipsv2 is qualified for the bilinear resize ({_RESAMPLE_BILINEAR}), not {resample}")

    rescale = _require_float32(image.get("rescale_factor", 1.0 / 255.0),
                               "image_processor.rescale_factor")
    # The runtime multiplies an 8-bit sample by this, in float32. Check the product the way the
    # runtime computes it, at the brightest pixel there is, and keep it inside the range the
    # towers were trained on rather than merely finite.
    with np.errstate(over="ignore"):
        brightest = float(np.float32(255.0) * np.float32(rescale))
    if not np.isfinite(brightest) or not 0.0 < brightest <= 1.0:
        raise ValueError(
            f"tipsv2 image_processor.rescale_factor={rescale!r} sends a white pixel to "
            f"{brightest!r}; TIPSv2 takes pixels in (0, 1]")

    return {"image_size": vision["image_size"], "resample": resample, "rescale_factor": rescale}


def resolve(model_dir: Path, weights: dict) -> dict:
    """Return the geometry the builders and the runtime need."""
    model_dir = Path(model_dir)
    config = _load_json(model_dir / "config.json")

    # Both may be left out; neither may be stated as something else, and neither may be `null` -
    # a file that says `"architectures": null` is not a file that says nothing about them.
    architectures = _claim(config, "architectures")
    if architectures is not _ABSENT:
        if not isinstance(architectures, list) or not all(
                isinstance(name, str) for name in architectures):
            raise ValueError("tipsv2 config.json architectures must be a list of strings")
        if "TIPSv2Model" not in architectures:
            raise NotImplementedError(f"tipsv2 builds TIPSv2Model, not {architectures}")
    model_type = _claim(config, "model_type")
    if model_type is not _ABSENT:
        if not isinstance(model_type, str):
            raise ValueError("tipsv2 config.json model_type must be a string")
        if model_type != "tipsv2":
            # tipsv2_dpt reuses these towers under dense-prediction heads and is another family.
            raise NotImplementedError(f"tipsv2 builds model_type 'tipsv2', not {model_type!r}")

    vision = _vision(weights, config)
    text = _text(weights, config)
    if vision["hidden_size"] != text["hidden_size"]:
        # The two towers meet at a dot product; different widths would make it meaningless.
        raise ValueError(
            f"tipsv2 towers must share a width: vision {vision['hidden_size']}, "
            f"text {text['hidden_size']}")

    # The learned contrastive temperature. The reference returns it beside the embeddings and
    # never applies it; the runtime divides the cosine by it to get a softmax logit, so it has to
    # be positive and finite.
    # The runtime divides a cosine in [-1, 1] by this, in float32, so it has to stay a positive
    # float32 and the quotient has to stay finite.
    temperature = _require_float32(config.get("temperature_init_value"), "temperature_init_value")
    with np.errstate(over="ignore", divide="ignore"):
        inverted = np.float32(1.0) / np.float32(temperature)
    if not np.isfinite(inverted):
        raise ValueError(
            f"tipsv2 temperature_init_value={temperature!r} overflows float32 when inverted")

    return {
        "vision": vision,
        "text": text,
        "preprocess": _preprocess(model_dir, vision),
        "temperature": temperature,
        "embedding_dim": vision["hidden_size"],
    }
