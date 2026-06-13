"""Gemma-4 family plugin — multimodal (text + vision + audio) on top of a
Gemma-style text decoder.

Scope (initial scaffold):
  * Recognizes model_type == "gemma4" and the nested submodule types
    ("gemma4_text", "gemma4_vision", "gemma4_audio") so that nested-config
    dispatch in the harness still routes back here.
  * Architecture mirrors prior Gemma generations for the text tower:
      - RMSNorm with the (1 + gamma) * normalized convention.
      - sqrt(hidden_size) embedding scale.
      - Four norms per decoder layer (input_norm, post_attn_norm,
        pre_ffn_norm, post_ffn_norm) — Gemma-2 / Gemma-3 layout.
      - SwiGLU MLP with ``gelu_pytorch_tanh`` (mapped to "gelu_new").
      - Attention softcap (``attn_logit_softcapping``) and final logit
        softcap (``final_logit_softcapping``), when present in config.
  * Multimodal heads — vision and audio — are scaffolded with explicit
    NotImplementedError docstrings listing the missing details so they can
    be filled in once the released config + weight index are available.

OPEN QUESTIONS (need GPU validation / live HF config to confirm):
  * Exact HF weight prefixes. Likely candidates:
      - Text decoder:   ``model.language_model.layers.{i}.*`` or
                        ``model.text.layers.{i}.*``.
      - Vision encoder: ``model.vision_tower.*`` or ``model.vision.*``.
      - Audio encoder:  ``model.audio_tower.*`` or ``model.audio.*``.
      - Multimodal projector(s): ``model.multi_modal_projector.*`` /
                                 ``model.audio_projector.*``.
  * Image / audio token IDs in the tokenizer vocabulary.
  * Audio encoder: HF docs describe a 12-layer conv-subsampling encoder
    (hidden=1024, num_heads=8). The exact subsampling strides, conv
    kernel sizes, and downstream transformer norm placement need to be
    verified from the released modeling code.
  * Vision encoder: likely a SigLIP-2 variant (same as Gemma-3 vision)
    with the standard patch_size/image_size in vision_config; need to
    confirm whether Gemma-4 introduces a new tower.
  * Multimodal fusion: assumed to follow Gemma-3 / LLaVA-style embedding
    replacement at the text decoder's input (image_token_id /
    audio_token_id substitution), not cross-attention. Confirm with
    Gemma4ForConditionalGeneration forward source.

Because the audio + vision builders are stubs, ``build_vision_engine`` and
``build_audio_engine`` raise NotImplementedError today; the text decoder is
fully wired through the existing Gemma decoder builder and will produce a
valid engine for the text-only subset of inputs.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...config import ModelConfig
from ...checkpoint_mapper import (
    WeightDict,
    _has_tensor,
    _load_tensor,
    _open_safetensors,
    _transpose_2d,
)
from ...parallel_config import normalize_parallel_config


# Default image/audio token strings — placeholders; verify against released
# tokenizer.json once available.
_DEFAULT_IMAGE_TOKEN_STR = "<image_soft_token>"
_DEFAULT_AUDIO_TOKEN_STR = "<audio_soft_token>"
_DEFAULT_FIXED_IMAGE_SIZE = 896  # Gemma-3 default; verify for Gemma-4.


# Submodule model_types that may appear in nested config blocks.
_GEMMA4_TYPES = ("gemma4", "gemma4_text", "gemma4_vision", "gemma4_audio")


class Gemma4Plugin:
    """Multimodal Gemma-4 family plugin (text + vision + audio)."""

    name = "gemma4"
    runtime_strategy = "vision_language"
    embed_input = True

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def matches(self, model_type: str) -> bool:
        mt = (model_type or "").lower()
        if mt in _GEMMA4_TYPES:
            return True
        # Be tolerant of separators ("gemma-4", "gemma_4") and of the
        # ForConditionalGeneration architectures string accidentally
        # being passed in instead of the model_type slug.
        normalized = mt.replace("-", "").replace("_", "")
        return normalized.startswith("gemma4")

    # ------------------------------------------------------------------
    # Text decoder config extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _text_config(config: ModelConfig) -> dict:
        """Return the nested text_config block, or the top-level raw
        config when the harness has already unwrapped it.
        """
        text_cfg = config.raw.get("text_config")
        if isinstance(text_cfg, dict) and text_cfg:
            return text_cfg
        return config.raw

    @staticmethod
    def _vision_config(config: ModelConfig) -> dict | None:
        vc = config.raw.get("vision_config")
        if isinstance(vc, dict) and vc:
            return vc
        return None

    @staticmethod
    def _audio_config(config: ModelConfig) -> dict | None:
        ac = config.raw.get("audio_config")
        if isinstance(ac, dict) and ac:
            return ac
        return None

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(
        self, model_dir: str, config: ModelConfig,
        *, precision: str = "fp32",
    ) -> WeightDict:
        """Load Gemma-4 text decoder weights and apply Gemma-style fixes.

        Gemma-4 nests the text tower under ``model.language_model.*``
        (mirroring Gemma-3 / Qwen3-VL). We probe both that prefix and the
        flat ``model.*`` prefix for forward compatibility with any future
        repacked checkpoint.

        Gemma-specific transforms (carried over from the Gemma plugin):
          1. Add 1.0 to every RMSNorm gamma (``(1 + gamma) * normalized``).
          2. Scale the input embedding by ``sqrt(hidden_size)``.

        Vision / audio encoder weights are loaded by the dedicated
        ``_load_vision_weights`` / ``_load_audio_weights`` helpers, which
        ``build_vision_engine`` / ``build_audio_engine`` consume.
        """
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        text_cfg = self._text_config(config)
        hidden = int(text_cfg.get("hidden_size", config.hidden_size))
        int(text_cfg.get("vocab_size", config.vocab_size))
        num_layers = int(
            text_cfg.get("num_hidden_layers", config.num_hidden_layers))
        num_heads = int(
            text_cfg.get("num_attention_heads", config.num_attention_heads))
        num_kv_heads = int(
            text_cfg.get("num_key_value_heads", config.num_key_value_heads))
        head_dim = int(text_cfg.get("head_dim", 0)) or (hidden // max(num_heads, 1))

        # Probe weight key prefix — Gemma-4 conditional generation models
        # likely place the text tower under "model.language_model.*".
        prefix_candidates = (
            "model.language_model",
            "model",
        )
        text_prefix = None
        for cand in prefix_candidates:
            if _has_tensor(readers, f"{cand}.embed_tokens.weight"):
                text_prefix = cand
                break
        if text_prefix is None:
            # Last-resort fallback so build doesn't crash; callers will see
            # a clear KeyError below.
            text_prefix = "model"

        weights = WeightDict()

        # Embedding
        embed_key = f"{text_prefix}.embed_tokens.weight"
        embedding = _load_tensor(readers, embed_key)
        weights["embedding"] = embedding.astype(np.float32)

        attention_size = 0
        kv_attention_size = 0
        mlp_size = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"{text_prefix}.layers.{layer_idx}"

            # 4-norm layout (input/post_attn/pre_ffn/post_ffn)
            input_norm = _load_tensor(
                readers, f"{hf_prefix}.input_layernorm.weight")
            post_attn = _load_tensor(
                readers, f"{hf_prefix}.post_attention_layernorm.weight")
            weights[f"{prefix}.input_norm"] = (
                input_norm.astype(np.float32) + 1.0)
            weights[f"{prefix}.post_attn_norm"] = (
                post_attn.astype(np.float32) + 1.0)

            pre_ffn_key = f"{hf_prefix}.pre_feedforward_layernorm.weight"
            if _has_tensor(readers, pre_ffn_key):
                weights[f"{prefix}.pre_ffn_norm"] = (
                    _load_tensor(readers, pre_ffn_key).astype(np.float32) + 1.0)
            post_ffn_key = f"{hf_prefix}.post_feedforward_layernorm.weight"
            if _has_tensor(readers, post_ffn_key):
                weights[f"{prefix}.post_ffn_norm"] = (
                    _load_tensor(readers, post_ffn_key).astype(np.float32) + 1.0)

            # Q/K/V/O
            q_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.q_proj.weight")
            k_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.k_proj.weight")
            v_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.v_proj.weight")
            o_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.o_proj.weight")

            if attention_size == 0:
                attention_size = q_raw.shape[0]
            weights[f"{prefix}.w_q"] = _transpose_2d(q_raw, "q_proj")
            weights[f"{prefix}.w_k"] = _transpose_2d(k_raw, "k_proj")
            weights[f"{prefix}.w_v"] = _transpose_2d(v_raw, "v_proj")
            weights[f"{prefix}.w_o"] = _transpose_2d(o_raw, "o_proj")
            if kv_attention_size == 0:
                kv_attention_size = num_kv_heads * head_dim

            # Optional per-head q/k norms (Gemma-3 introduced these).
            q_norm_key = f"{hf_prefix}.self_attn.q_norm.weight"
            if _has_tensor(readers, q_norm_key):
                q_norm = _load_tensor(readers, q_norm_key).astype(np.float32) + 1.0
                weights[f"{prefix}.q_norm"] = np.tile(q_norm, num_heads)
            k_norm_key = f"{hf_prefix}.self_attn.k_norm.weight"
            if _has_tensor(readers, k_norm_key):
                k_norm = _load_tensor(readers, k_norm_key).astype(np.float32) + 1.0
                weights[f"{prefix}.k_norm"] = np.tile(k_norm, num_kv_heads)

            # SwiGLU MLP
            gate_raw = _load_tensor(readers, f"{hf_prefix}.mlp.gate_proj.weight")
            up_raw = _load_tensor(readers, f"{hf_prefix}.mlp.up_proj.weight")
            down_raw = _load_tensor(readers, f"{hf_prefix}.mlp.down_proj.weight")
            if mlp_size == 0:
                mlp_size = gate_raw.shape[0]
            weights[f"{prefix}.w_gate"] = _transpose_2d(gate_raw, "gate")
            weights[f"{prefix}.w_up"] = _transpose_2d(up_raw, "up")
            weights[f"{prefix}.w_down"] = _transpose_2d(down_raw, "down")

        # Final norm (Gemma "+1.0" rule applies)
        final_norm_key = f"{text_prefix}.norm.weight"
        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = (
                _load_tensor(readers, final_norm_key).astype(np.float32) + 1.0)
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        # LM head — Gemma generally ties to the embedding.
        lm_head_key = "lm_head.weight"
        if _has_tensor(readers, lm_head_key):
            weights["w_out"] = _transpose_2d(
                _load_tensor(readers, lm_head_key), "lm_head")
        else:
            weights["w_out"] = _transpose_2d(embedding.copy(), "embedding_tied")

        # Apply sqrt(hidden) embedding scale AFTER the LM head has been
        # tied — Gemma multiplies hidden_states by sqrt(hidden) right
        # after the embedding lookup, which is mathematically equivalent
        # to scaling the embedding table when LM head is untied.
        scale = math.sqrt(hidden)
        weights["embedding"] = weights["embedding"] * scale

        weights["_attention_size"] = attention_size  # type: ignore[assignment]
        weights["_kv_attention_size"] = kv_attention_size  # type: ignore[assignment]
        weights["_mlp_size"] = mlp_size  # type: ignore[assignment]

        return weights

    # ------------------------------------------------------------------
    # Text decoder build
    # ------------------------------------------------------------------

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
        debug_layer_outputs: bool = False,
        parallel_config=None,
    ) -> bytes:
        """Build the Gemma-4 text decoder TRT engine.

        Reuses the parameterized standard decoder builder with Gemma-style
        knobs:
          * ``mlp_type="swiglu"`` with ``activation="gelu_new"`` (Gemma
            actually uses gelu_pytorch_tanh; the builder treats
            "gelu_new" as the tanh-approx GELU variant.)
          * ``norm_type="rmsnorm"``.
          * Attention / final logit softcap when present in config.

        Parallel (TP) builds will be wired up once the TP infrastructure
        on main lands a Gemma-4-aware TP builder. Today this falls back
        to the standard single-device path with ``parallel_config`` left
        as-is for the perf agent's TP follow-up.
        """
        # Delegate to the local builders (text-decoder-only for now). We
        # do NOT import these at module top-level because they pull in
        # TensorRT, which the auto-discovery loop is willing to skip when
        # missing.
        from .standard_decoder_builder import build_standard_decoder_engine
        from .dual_profile_decoder_builder import (
            build_dual_profile_decoder_engine,
        )

        text_cfg = self._text_config(config)
        # Surface attn_logit_softcapping / final_logit_softcapping on
        # config.raw so the underlying builder can pick them up.
        config.raw.setdefault(
            "attn_logit_softcapping",
            text_cfg.get("attn_logit_softcapping"),
        )
        config.raw.setdefault(
            "final_logit_softcapping",
            text_cfg.get("final_logit_softcapping"),
        )

        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            raise NotImplementedError(
                "Gemma-4 tensor-parallel decoder build is not yet wired. "
                "Follow the Gemma TP precedent in "
                "families/gemma/dual_profile_decoder_tp_builder.py once "
                "the released config is verified.")

        decoder_engine_role = str(
            config.raw.get("_decoder_engine_role", "dual_profile"))
        if decoder_engine_role in ("dual_profile", "prefill") and not (
            debug_layer_outputs
            or bool(config.raw.get("dynamic_kv_cache", False))
        ):
            return build_dual_profile_decoder_engine(
                config, weights, max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                norm_type="rmsnorm",
                mlp_type="swiglu",
                position_type="rope",
                activation="gelu_new",
                verbose=verbose,
                profile_mode=(
                    "prefill" if decoder_engine_role == "prefill"
                    else "dual_profile"),
            )
        return build_standard_decoder_engine(
            config, weights, max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            norm_type="rmsnorm",
            mlp_type="swiglu",
            position_type="rope",
            activation="gelu_new",
            embed_input=True,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs,
        )

    # ------------------------------------------------------------------
    # Vision encoder
    # ------------------------------------------------------------------

    def build_vision_engine(
        self, model_dir: str, config: ModelConfig, weights: WeightDict,
        *, precision: str = "fp32", verbose: bool = False,
    ) -> bytes | None:
        """Build the Gemma-4 vision encoder TRT engine.

        Currently a stub — calls into ``vision_encoder_builder`` which
        raises NotImplementedError with a list of unresolved details
        (patch size, image size, normalization stats, projector layout).
        Returns None when ``vision_config`` is absent so the harness
        skips vision wiring for text-only checkpoints.
        """
        if self._vision_config(config) is None:
            return None
        from .vision_encoder_builder import build_gemma4_vision_engine

        vision_weights = _load_vision_weights(model_dir, config)
        return build_gemma4_vision_engine(
            self._vision_config(config), vision_weights,
            fixed_image_size=_DEFAULT_FIXED_IMAGE_SIZE,
            precision=precision, verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Audio encoder
    # ------------------------------------------------------------------

    def build_audio_engine(
        self, model_dir: str, config: ModelConfig, weights: WeightDict,
        *, precision: str = "fp32", verbose: bool = False,
    ) -> bytes | None:
        """Build the Gemma-4 audio encoder TRT engine.

        Currently a stub — calls into ``audio_encoder_builder`` which
        raises NotImplementedError. Returns None when ``audio_config``
        is absent.
        """
        if self._audio_config(config) is None:
            return None
        from .audio_encoder_builder import build_gemma4_audio_engine

        audio_weights = _load_audio_weights(model_dir, config)
        return build_gemma4_audio_engine(
            self._audio_config(config), audio_weights,
            precision=precision, verbose=verbose,
        )

    # ------------------------------------------------------------------
    # VL / multimodal config injection
    # ------------------------------------------------------------------

    def get_vl_config(self, config: ModelConfig) -> dict | None:
        """Return the VL config block emitted into the bundle config.json.

        Returns None when neither vision nor audio is configured. When
        either is present, the dict carries enough info for the C++
        runtime + Python preprocessor to assemble multimodal prompts.
        """
        vc = self._vision_config(config)
        ac = self._audio_config(config)
        if vc is None and ac is None:
            return None

        text_cfg = self._text_config(config)
        hidden_size = int(text_cfg.get("hidden_size", config.hidden_size))

        vl_cfg: dict = {
            "vision_output_dim": hidden_size,
            "preprocessor_type": "simple_chw",
            "vl_prompt_template": (
                "<start_of_turn>user\n"
                "{image_pads}{audio_pads}{prompt}<end_of_turn>\n"
                "<start_of_turn>model\n"
            ),
        }

        if vc is not None:
            patch_size = int(vc.get("patch_size", 14))
            image_size = int(vc.get("image_size", _DEFAULT_FIXED_IMAGE_SIZE))
            num_patches = (image_size // patch_size) ** 2
            vl_cfg.update({
                "image_token_id": config.raw.get("image_token_index", -1),
                "image_token_str": _DEFAULT_IMAGE_TOKEN_STR,
                "fixed_image_size": image_size,
                "num_image_pad_tokens": num_patches,
            })

        if ac is not None:
            vl_cfg.update({
                "audio_token_id": config.raw.get("audio_token_index", -1),
                "audio_token_str": _DEFAULT_AUDIO_TOKEN_STR,
                "audio_hidden_size": int(ac.get("hidden_size", 1024)),
                "audio_num_layers": int(ac.get("num_hidden_layers", 12)),
                "audio_num_heads": int(ac.get("num_attention_heads", 8)),
            })

        return vl_cfg


# ---------------------------------------------------------------------------
# Encoder weight loaders (vision + audio)
# ---------------------------------------------------------------------------

def _load_vision_weights(model_dir: str, config: ModelConfig) -> WeightDict:
    """Collect vision encoder weights for the Gemma-4 vision tower.

    Probes the two most likely prefixes seen across Google multimodal
    releases:
      * ``model.vision_tower.*`` (Gemma-3 layout).
      * ``model.vision.*``       (compact Gemma-4 candidate).
    Plus a multimodal projector under either
    ``model.multi_modal_projector.*`` or ``model.vision_projector.*``.
    """
    model_dir_path = Path(model_dir)
    readers = _open_safetensors(model_dir_path)

    weights = WeightDict()
    for reader in readers:
        for key in reader.keys():
            for prefix in (
                "model.vision_tower.",
                "model.vision.",
                "vision_tower.",
                "model.multi_modal_projector.",
                "model.vision_projector.",
            ):
                if key.startswith(prefix):
                    # Strip the leading "model." so downstream builders can
                    # be prefix-agnostic.
                    canon = key[len("model."):] if key.startswith("model.") else key
                    weights[canon] = _load_tensor([reader], key)
                    break
    return weights


def _load_audio_weights(model_dir: str, config: ModelConfig) -> WeightDict:
    """Collect audio encoder weights for the Gemma-4 audio tower.

    Probes ``model.audio_tower.*`` / ``model.audio.*`` plus an audio
    projector under ``model.audio_projector.*`` or
    ``model.multi_modal_projector.*`` (when shared with vision).
    """
    model_dir_path = Path(model_dir)
    readers = _open_safetensors(model_dir_path)

    weights = WeightDict()
    for reader in readers:
        for key in reader.keys():
            for prefix in (
                "model.audio_tower.",
                "model.audio.",
                "audio_tower.",
                "model.audio_projector.",
            ):
                if key.startswith(prefix):
                    canon = key[len("model."):] if key.startswith("model.") else key
                    weights[canon] = _load_tensor([reader], key)
                    break
    return weights


plugin = Gemma4Plugin()
