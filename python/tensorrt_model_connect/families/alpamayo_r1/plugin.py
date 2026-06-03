"""Alpamayo-R1 family plugin — NVIDIA autonomous-driving VLA model.

Per https://huggingface.co/nvidia/Alpamayo-R1-10B (renamed Alpamayo 1 at
CES 2026), the model is a VLA (vision-language-action) network with:

  * **8.2B Cosmos-Reason2-8B (Qwen3-VL) backbone** for VLM understanding
    of camera frames + textual driving context. The Alpamayo checkpoint
    stores all backbone weights under a ``vlm.`` prefix
    (e.g. ``vlm.model.language_model.layers.0.self_attn.q_proj.weight``).
  * 2.3B action expert head — separate from the VLM, decodes trajectory
    tokens into a waypoint flow-matching diffusion. Not modeled in this
    plugin (out of scope for the LLM/VLM build).
  * Extended vocabulary (155 697 tokens vs. Cosmos-Reason2-8B's 151 936)
    — the extra ~3 760 tokens are a per-waypoint trajectory tokenizer
    that the VLM autoregressively emits before the action head decodes.

This plugin onboards the VLM portion only. The 10B model with the action
expert skipped acts as a standard Qwen3-VL chat VLM.

**Onboarding strategy.** The released Alpamayo-R1 checkpoint ships only
its own config.json (Alpamayo orchestration fields) — no Qwen3-VL
backbone config / tokenizer / vision encoder config. We work around
that by:

  1. Hardcoding the Cosmos-Reason2-8B backbone config (text_config +
     vision_config + rope_scaling) into this plugin as constants — same
     numbers as ``nvidia/Cosmos-Reason2-8B/config.json``.
  2. At ``load_weights`` time, mutating the in-memory ``ModelConfig`` to
     look like Qwen3-VL. ``vocab_size`` stays at Alpamayo's 155 697
     because the model's embedding + lm_head are sized for it.
  3. Wrapping the safetensors readers so that ``vlm.X`` keys are also
     accessible as ``X`` — this lets the unmodified qwen_vl loader find
     ``model.language_model.layers.N.*`` etc.
  4. Delegating ``load_weights`` and ``build_engine`` to the qwen_vl
     family with the augmented config + wrapped readers.

The tokenizer is loaded from the standard HF cache — users either pair
the Alpamayo checkpoint with a Cosmos-Reason2-8B tokenizer in the same
model_dir, or the runtime falls back to the embedded BPE in the bundle.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from ...config import ModelConfig
from ...checkpoint_mapper import (
    WeightDict,
    _ReaderCollection,
    _open_safetensors,
)


_VLM_PREFIX = "vlm."

# Cosmos-Reason2-8B text_config — the Qwen3-VL backbone Alpamayo wraps.
# Source: https://huggingface.co/nvidia/Cosmos-Reason2-8B/raw/main/config.json
_BACKBONE_TEXT_CONFIG = {
    "model_type": "qwen3_vl_text",
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_act": "silu",
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-06,
    "rope_theta": 5000000,
    "rope_scaling": {
        "mrope_interleaved": True,
        "mrope_section": [24, 20, 20],
        "rope_type": "default",
    },
    "attention_bias": False,
    "tie_word_embeddings": False,
}

# Cosmos-Reason2-8B vision_config (Qwen3-VL vision encoder).
_BACKBONE_VISION_CONFIG = {
    "model_type": "qwen3_vl",
    "depth": 27,
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "num_heads": 16,
    "in_channels": 3,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "num_position_embeddings": 2304,
    "out_hidden_size": 4096,
    "hidden_act": "gelu_pytorch_tanh",
    "deepstack_visual_indexes": [8, 16, 24],
}


class _PrefixedReader:
    """Proxy that prepends ``prefix`` before delegating get_tensor."""

    def __init__(self, inner, prefix: str) -> None:
        self._inner = inner
        self._prefix = prefix

    def get_tensor(self, name: str):
        return self._inner.get_tensor(self._prefix + name)

    def keys(self):
        for k in self._inner.keys():
            if k.startswith(self._prefix):
                yield k[len(self._prefix):]


def _wrap_readers_with_vlm_prefix(model_dir: Path) -> _ReaderCollection:
    """Open safetensors and alias ``vlm.X`` keys as ``X`` (and ``X``).

    Alpamayo stores its Qwen3-VL backbone under ``vlm.model.*`` and
    ``vlm.lm_head.*``. The qwen_vl loader looks for ``model.*`` and
    ``lm_head.*`` directly. This builds a ``_ReaderCollection`` whose
    ``tensor_map`` exposes both the original ``vlm.X`` and the aliased
    ``X`` keys — the aliased entries route into a ``_PrefixedReader``
    proxy that rewrites the ``get_tensor`` call back to ``vlm.X``.
    """

    readers = _open_safetensors(model_dir)
    aliased = dict(readers.tensor_map)
    for key, reader in list(readers.tensor_map.items()):
        if key.startswith(_VLM_PREFIX):
            stripped = key[len(_VLM_PREFIX):]
            if stripped not in aliased:
                aliased[stripped] = _PrefixedReader(reader, _VLM_PREFIX)
    return _ReaderCollection(list(readers), tensor_map=aliased)


def _augment_config_with_qwen3_vl_backbone(config: ModelConfig) -> None:
    """Mutate ``config`` in-place to expose the Qwen3-VL backbone fields.

    Alpamayo's released config.json has neither a Qwen3-VL ``text_config``
    nor a ``vision_config`` — they're not shipped with the checkpoint.
    Fill them in from the Cosmos-Reason2-8B constants above so the
    qwen_vl plugin sees a normal Qwen3-VL model with Alpamayo's vocab.
    """

    text = _BACKBONE_TEXT_CONFIG
    config.hidden_size = int(text["hidden_size"])
    config.intermediate_size = int(text["intermediate_size"])
    config.num_hidden_layers = int(text["num_hidden_layers"])
    config.num_attention_heads = int(text["num_attention_heads"])
    config.num_key_value_heads = int(text["num_key_value_heads"])
    config._head_dim = int(text["head_dim"])
    config.rms_norm_eps = float(text["rms_norm_eps"])
    config.rope_theta = float(text["rope_theta"])

    # Mirror into raw so plugin code that reads config.raw[...] sees the
    # same view (qwen_vl reads vision_config and rope_scaling from raw).
    config.raw.setdefault("vision_config", dict(_BACKBONE_VISION_CONFIG))
    config.raw.setdefault("rope_scaling", dict(text["rope_scaling"]))
    config.raw.setdefault("rope_theta", text["rope_theta"])
    config.raw.setdefault("hidden_act", text["hidden_act"])
    config.raw.setdefault("max_position_embeddings", text["max_position_embeddings"])
    config.raw.setdefault("attention_bias", text["attention_bias"])
    config.raw.setdefault("tie_word_embeddings", text["tie_word_embeddings"])
    config.raw.setdefault("model_type", "qwen3_vl")


def _qwen_vl_module():
    # Resolve via importlib so the unwrapped plugin module (not the
    # family-package wrapper) is patched for ``_open_safetensors``.
    return importlib.import_module(
        "tensorrt_model_connect.families.qwen_vl.plugin")


def _with_aliased_readers(model_dir: str, func):
    """Run ``func`` with qwen_vl's ``_open_safetensors`` monkey-patched
    to return the alpamayo-aware wrapped readers."""
    qmod = _qwen_vl_module()
    original = qmod._open_safetensors
    wrapped_dir = Path(model_dir)

    def _patched(*args, **kwargs):
        return _wrap_readers_with_vlm_prefix(wrapped_dir)

    qmod._open_safetensors = _patched
    try:
        return func()
    finally:
        qmod._open_safetensors = original


class AlpamayoR1Plugin:
    name = "alpamayo_r1"
    runtime_strategy = "vision_language"
    embed_input = True

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "alpamayo_r1"

    def load_weights(
        self, model_dir: str, config: ModelConfig,
        *, precision: str = "fp32",
    ) -> WeightDict:
        _augment_config_with_qwen3_vl_backbone(config)
        qmod = _qwen_vl_module()
        return _with_aliased_readers(
            model_dir, lambda: qmod.QwenVLPlugin().load_weights(model_dir, config))

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False, parallel_config=None,
    ) -> bytes:
        # Config may have been augmented already in load_weights; redo
        # defensively in case build_engine runs independently.
        _augment_config_with_qwen3_vl_backbone(config)
        qmod = _qwen_vl_module()
        return qmod.QwenVLPlugin().build_engine(
            config, weights, max_cache_length,
            precision=precision, quant_ctx=quant_ctx,
            verbose=verbose, parallel_config=parallel_config)

    def build_vision_engine(
        self, model_dir: str, config: ModelConfig, weights: WeightDict,
        *, precision: str = "fp32", verbose: bool = False,
    ) -> bytes | None:
        _augment_config_with_qwen3_vl_backbone(config)
        qmod = _qwen_vl_module()

        # qwen_vl._load_vision_weights iterates raw reader.keys() looking for
        # 'visual.*' or 'model.visual.*'. Alpamayo stores them at
        # 'vlm.model.visual.*' — one extra nesting level. Monkey-patch the
        # vision loader to also recognize the alpamayo prefix.
        original_load_vision = qmod._load_vision_weights

        def _alpamayo_load_vision(model_dir, config):
            from pathlib import Path
            from ...checkpoint_mapper import _open_safetensors as _real_open
            from ...checkpoint_mapper import _load_tensor, WeightDict
            readers = _real_open(Path(model_dir))
            weights_out = WeightDict()
            for reader in readers:
                for key in reader.keys():
                    if key.startswith("vlm.model.visual."):
                        canon = key[len("vlm.model."):]  # → "visual.*"
                        weights_out[canon] = _load_tensor([reader], key)
                    elif key.startswith("vlm.visual."):
                        canon = key[len("vlm."):]  # → "visual.*"
                        weights_out[canon] = _load_tensor([reader], key)
                    elif key.startswith("model.visual."):
                        canon = key[len("model."):]
                        weights_out[canon] = _load_tensor([reader], key)
                    elif key.startswith("visual."):
                        weights_out[key] = _load_tensor([reader], key)
            return weights_out

        qmod._load_vision_weights = _alpamayo_load_vision
        try:
            return _with_aliased_readers(
                model_dir,
                lambda: qmod.QwenVLPlugin().build_vision_engine(
                    model_dir, config, weights,
                    precision=precision, verbose=verbose))
        finally:
            qmod._load_vision_weights = original_load_vision

    def get_vl_config(self, config: ModelConfig) -> dict | None:
        _augment_config_with_qwen3_vl_backbone(config)
        qmod = _qwen_vl_module()
        return qmod.QwenVLPlugin().get_vl_config(config)


plugin = AlpamayoR1Plugin()
