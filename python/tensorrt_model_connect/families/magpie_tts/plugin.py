# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MagpieTTS family plugin -- encoder-decoder text-to-speech model.

MagpieTTS is an encoder-decoder TTS model from a NeMo .nemo archive:
  - Text encoder: 6 causal self-attention layers (no KV cache), takes text token IDs,
    outputs encoder features [max_source_positions, hidden]
  - Decoder: 12 self-attention layers (with KV cache) + ASYMMETRIC cross-attention
    to encoder output, autoregressive, predicts 8 codebook tokens per frame
  - NanoCodec: HiFi-GAN conv decoder, converts codec tokens to 22kHz waveform
  - Weight source: NeMo .nemo archive (tar with model_weights.ckpt + model_config.yaml)

Real architecture (from NeMo model_config.yaml):
  - model_type: "decoder_ce"
  - embedding_dim: 768
  - encoder: 6 layers, d_model=768, d_ffn=3072, 12 heads, kernel_size=3, causal
  - decoder: 12 layers, d_model=768, d_ffn=3072, 12 SA heads, cross-attn: 1 head,
    d_head=128, d_memory=768, kernel_size=1, causal, has_xattn
  - LayerNorm everywhere (bias=False, i.e. beta=0)
  - Fused QKV: qkv_net.weight [3H, H]
  - FFN uses Conv1d: encoder kernel_size=3, decoder kernel_size=1
  - Baked speaker context: baked_context_embedding [5, 110*768]
  - Output: final_proj [16192, 768] with bias (8 codebooks * 2024 each)

Cross-attention design:
  The cross_k/cross_v inputs to the decoder engine are the RAW encoder output
  (same tensor copied to all layers). The per-layer K/V projections are baked
  into the decoder TRT graph. Cross-attention is ASYMMETRIC: 1 head, d_head=128.
  Two separate norms: norm_xattn_query (on decoder state) and norm_xattn_memory
  (on encoder output) before K/V projection.

Encoder uses embed_input=False (token IDs), decoder uses embed_input=True
(the C++ runtime sums 8 codebook embeddings on host, then copies to device).

Classifier-Free Guidance (CFG):
  The bundle bakes three config fields that control inference-time sampling:
    - magpie_temperature (default 0.6): decoder sampling temperature.
      This matches NeMo's short-form Magpie inference default.
    - magpie_cfg_scale (default 2.5): strength of text-conditioning amplification.
      Each decoder frame runs two forward passes — conditioned (real encoder output)
      and unconditional (null-text encoder output). Logits are blended:
        logits = uncond + cfg_scale * (cond - uncond)
      Scale 1.0 disables CFG (single pass). Scale 2.5 matches NeMo.
    - magpie_finished_limit_with_eot (default 0): optional hard-stop safety net.
      The TRT runtime keeps this disabled by default so short-form generation
      matches NeMo's standard do_tts() stop behavior. It can still be enabled
      explicitly via environment override for debugging.
  All can be adjusted at runtime via the config registry, e.g.
    trtmc run ... --set audio_magpie.temperature=0.6 \\
                 --set audio_magpie.cfg_scale=2.5 \\
                 --set audio_magpie.finished_limit=10
  Or via CLI flags:
    ./build/trtmc generate-audio bundle.bundle --prompt "text" --output out.wav --cfg-scale 2.5 --greedy
"""

from __future__ import annotations

import io
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
from tensorrt_model_connect import trt_compat

from .config import ModelConfig
from .checkpoint_mapper import WeightDict
from ...build_timing import timed_trt_compile
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)
from . import graph_ops
from . import magpie_tokenizer


trt = trt_compat.get_trt()

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_np(tensor) -> np.ndarray:
    """Convert a torch tensor or array to float32 numpy."""
    if hasattr(tensor, 'numpy'):
        return tensor.numpy().astype(np.float32)
    return np.asarray(tensor, dtype=np.float32)


def _t2d(tensor) -> np.ndarray:
    """Transpose [out, in] -> [in, out] for TRT matmul convention."""
    a = _to_np(tensor)
    if a.ndim == 2:
        return np.ascontiguousarray(a.T)
    return a


def _validate_supported_checkpoint_architecture(state_dict) -> None:
    """Reject upstream Magpie architectures the current runtime cannot execute."""
    codebooks = {
        int(match.group(1))
        for key in state_dict
        if (match := re.fullmatch(r"audio_embeddings\.(\d+)\.weight", key))
    }
    local_layers = {
        int(match.group(1))
        for key in state_dict
        if (match := re.match(r"local_transformer\.layers\.(\d+)\.", key))
    }
    projection_keys = {
        "local_transformer_in_projection.weight",
        "local_transformer_in_projection.bias",
    }
    if (
        codebooks != set(range(8))
        or local_layers != {0}
        or not projection_keys.issubset(state_dict)
    ):
        raise ValueError(
            "This Magpie runtime supports 8 codebooks and one local-transformer "
            "layer with an input projection; the selected checkpoint has "
            f"{len(codebooks)} codebooks and local-transformer layers "
            f"{sorted(local_layers)}. Pin a compatible checkpoint with the "
            "model manifest hf_revision field or trtmc build --model-revision."
        )


def _split_fused_qkv(fused_weight, hidden: int):
    """Split fused QKV weight [3H, H] into Q[H,H], K[H,H], V[H,H].

    Each is transposed to [H, H] (TRT matmul convention: [in, out]).
    Returns (w_q, w_k, w_v) each of shape [hidden, hidden].
    """
    w = _to_np(fused_weight)  # [3H, H]
    assert w.shape[0] == 3 * hidden and w.shape[1] == hidden, \
        f"Expected fused QKV [{3*hidden}, {hidden}], got {w.shape}"
    q, k, v = w[:hidden], w[hidden:2*hidden], w[2*hidden:]
    # Transpose each [H, H] -> [H, H] for TRT (rhs constant is [in, out])
    return (np.ascontiguousarray(q.T),
            np.ascontiguousarray(k.T),
            np.ascontiguousarray(v.T))


def _split_fused_kv(fused_weight, d_head: int):
    """Split fused cross-attn KV weight [2*d_head, H] into K[d_head,H], V[d_head,H].

    Each is transposed to [H, d_head] for TRT matmul convention.
    Returns (w_k, w_v) each of shape [hidden, d_head].
    """
    w = _to_np(fused_weight)  # [2*d_head, hidden]
    assert w.shape[0] == 2 * d_head, \
        f"Expected fused KV [{2*d_head}, *], got {w.shape}"
    k, v = w[:d_head], w[d_head:]
    # Transpose [d_head, H] -> [H, d_head]
    return (np.ascontiguousarray(k.T),
            np.ascontiguousarray(v.T))


def _squeeze_conv1d_to_linear(conv_weight):
    """Squeeze Conv1d weight [out, in, 1] -> linear [in, out] for TRT matmul.

    Only valid for kernel_size=1. Result is transposed for TRT convention.
    """
    w = _to_np(conv_weight)
    assert w.ndim == 3 and w.shape[2] == 1, \
        f"Expected Conv1d weight [out, in, 1], got shape {w.shape}"
    # [out, in, 1] -> squeeze -> [out, in] -> transpose -> [in, out]
    return np.ascontiguousarray(w[:, :, 0].T)


# ---------------------------------------------------------------------------
# IPA asset extraction for native C++ tokenizer
# ---------------------------------------------------------------------------

def _extract_ipa_assets(nemo_path: str) -> dict[str, bytes] | None:
    """Extract IPA tokenizer assets from a NeMo archive for bundle baking.

    Returns a dict with 4 keys ready for bundle sections:
      - magpie_ipa_phoneme_dict: TSV (word<TAB>ph1 ph2 ph3) per line
      - magpie_ipa_heteronyms: one word per line
      - magpie_ipa_vocab: one token per line (line index = token ID)
      - magpie_ipa_config: JSON config

    Returns None if NeMo/IPATokenizer is not available (graceful degradation).
    """
    import json

    # --- Step 1: Extract raw text files from the .nemo archive ---
    phoneme_dict_text = None
    heteronyms_text = None
    model_config_yaml = None

    path = Path(nemo_path)
    if path.is_dir():
        nemo_files = sorted(path.glob("*.nemo"))
        if nemo_files:
            path = nemo_files[0]

    try:
        import yaml
        with tarfile.open(str(path), 'r') as tar:
            for member in tar.getmembers():
                basename = Path(member.name).name
                if basename == "model_config.yaml":
                    f = tar.extractfile(member)
                    if f is not None:
                        model_config_yaml = yaml.safe_load(f.read())
                # The phoneme dict and heteronyms files are referenced in
                # model_config.yaml but also bundled in the archive
                if "ipa_cmudict" in basename and basename.endswith(".txt"):
                    f = tar.extractfile(member)
                    if f is not None:
                        phoneme_dict_text = f.read().decode("utf-8")
                if "heteronyms" in basename:
                    f = tar.extractfile(member)
                    if f is not None:
                        heteronyms_text = f.read().decode("utf-8")
    except Exception as e:
        print(f"[trtmc build]   Warning: failed to read .nemo for IPA: {e}",
              file=sys.stderr)
        return None

    # --- Step 2: If dict files not in archive, try filesystem paths from config ---
    if phoneme_dict_text is None and model_config_yaml is not None:
        text_tokenizer_cfg = model_config_yaml.get("text_tokenizer", {})
        dict_path = text_tokenizer_cfg.get("phoneme_dict", None)
        if dict_path and Path(dict_path).exists():
            phoneme_dict_text = Path(dict_path).read_text(encoding="utf-8")
        het_path = text_tokenizer_cfg.get("heteronyms", None)
        if het_path and Path(het_path).exists():
            heteronyms_text = Path(het_path).read_text(encoding="utf-8")

    if phoneme_dict_text is None:
        print("[trtmc build]   Warning: could not find IPA phoneme dict",
              file=sys.stderr)
        return None

    # --- Step 3: Parse the phoneme dict into TSV format ---
    # NeMo IPA dict format: "WORD  ɪpɑprənʌnsɪeɪʃən" or "WORD(N)  ..."
    # Each pronunciation is a single IPA string (characters are individual tokens).
    # We normalize to: "word<TAB>pronunciation_string\n"
    # Multiple lines for the same word = multiple pronunciations.
    tsv_lines = []
    for line in phoneme_dict_text.splitlines():
        line = line.strip()
        if not line or line.startswith(";;;"):
            continue
        # Split on two-or-more spaces (NeMo format) or first whitespace block
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        word_raw = parts[0].strip()
        pronunciation = parts[1].strip()
        # Strip variant number: "WORD(2)" -> "word"
        if "(" in word_raw:
            word_raw = word_raw[:word_raw.index("(")]
        word = word_raw.lower()
        if word and pronunciation:
            tsv_lines.append(f"{word}\t{pronunciation}")
    phoneme_dict_tsv = "\n".join(tsv_lines) + "\n"

    # --- Step 4: Get authoritative vocab from NeMo IPATokenizer ---
    vocab_text = None
    grapheme_prefix = "#"
    eos_id = -1
    ignore_ambiguous = True

    try:
        tokenizer, text_vocab_size = magpie_tokenizer.load_tokenizer(path)
        # Extract vocab: _id2token is the authoritative mapping
        if hasattr(tokenizer, '_id2token'):
            id2token = tokenizer._id2token
            vocab_lines = []
            for i in range(len(id2token)):
                vocab_lines.append(str(id2token[i]))
            vocab_text = "\n".join(vocab_lines) + "\n"

        # Extract config from tokenizer / g2p attributes
        g2p = getattr(tokenizer, 'g2p', None)
        if g2p and hasattr(g2p, 'grapheme_prefix'):
            gp = getattr(g2p, 'grapheme_prefix', '')
            if gp:
                grapheme_prefix = str(gp)
            else:
                grapheme_prefix = ""  # NeMo uses no prefix
        else:
            grapheme_prefix = ""  # NeMo default: no grapheme prefix
        if g2p and hasattr(g2p, 'ignore_ambiguous_words'):
            ignore_ambiguous = bool(g2p.ignore_ambiguous_words)
        # EOS is text_vocab_size + 1 (NeMo convention, set by caller)
        eos_id = text_vocab_size + 1 if text_vocab_size else -1
    except Exception as e:
        raise RuntimeError(
            f"NeMo IPATokenizer failed to load — this is required for MagpieTTS "
            f"bundle builds. Install NeMo toolkit: pip install nemo_toolkit[tts]\n"
            f"Error: {e}"
        ) from e

    if vocab_text is None:
        raise RuntimeError(
            "NeMo IPATokenizer loaded but _id2token vocab is missing. "
            "The NeMo installation may be incomplete or incompatible. "
            "Install NeMo toolkit: pip install nemo_toolkit[tts]"
        )

    # --- Step 5: Build config JSON ---
    config_json = json.dumps({
        "grapheme_prefix": grapheme_prefix,
        "eos_id": eos_id,
        "ignore_ambiguous_words": 1 if ignore_ambiguous else 0,
    })

    # --- Step 6: Build heteronyms text ---
    if heteronyms_text is None:
        heteronyms_text = ""
    else:
        # Normalize: one word per line, lowercase
        het_lines = []
        for line in heteronyms_text.splitlines():
            word = line.strip().lower()
            if word:
                het_lines.append(word)
        heteronyms_text = "\n".join(het_lines) + "\n" if het_lines else ""

    return {
        "magpie_ipa_phoneme_dict": phoneme_dict_tsv.encode("utf-8"),
        "magpie_ipa_heteronyms": heteronyms_text.encode("utf-8"),
        "magpie_ipa_vocab": vocab_text.encode("utf-8"),
        "magpie_ipa_config": config_json.encode("utf-8"),
    }


# ---------------------------------------------------------------------------
# NeMo archive loading
# ---------------------------------------------------------------------------

def _load_nemo_archive(path: str):
    """Load model_weights.ckpt and model_config.yaml from a .nemo archive.

    The .nemo file is a tar archive containing:
      - model_weights.ckpt (PyTorch checkpoint)
      - model_config.yaml (OmegaConf YAML)

    Returns (state_dict, config_dict).
    """
    import torch
    import yaml

    nemo_path = Path(path)

    # If path is a directory, look for .nemo files inside it
    if nemo_path.is_dir():
        nemo_files = sorted(nemo_path.glob("*.nemo"))
        if nemo_files:
            nemo_path = nemo_files[0]
        else:
            raise FileNotFoundError(
                f"No .nemo file found in {path}")

    state_dict = None
    config_dict = None

    with tarfile.open(str(nemo_path), 'r') as tar:
        for member in tar.getmembers():
            basename = Path(member.name).name
            if basename == "model_weights.ckpt":
                f = tar.extractfile(member)
                if f is not None:
                    buf = io.BytesIO(f.read())
                    state_dict = torch.load(
                        buf, map_location="cpu", weights_only=False)
            elif basename == "model_config.yaml":
                f = tar.extractfile(member)
                if f is not None:
                    config_dict = yaml.safe_load(f.read())

    if state_dict is None:
        raise FileNotFoundError(
            f"model_weights.ckpt not found in {nemo_path}")
    if config_dict is None:
        raise FileNotFoundError(
            f"model_config.yaml not found in {nemo_path}")

    return state_dict, config_dict


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------

class MagpieTTSPlugin:
    name = "magpie_tts"
    runtime_strategy = "text_to_audio_magpie"

    def __init__(self):
        self._audio_config: dict = {}

    def matches(self, model_type: str) -> bool:
        return model_type.lower() in ("magpie_tts", "decoder_ce")

    def load_weights(self, model_dir: str, config: ModelConfig) -> WeightDict:
        """Load MagpieTTS weights from .nemo archive."""
        weights = WeightDict()

        # Resolve to the actual .nemo file path (model_dir may be a temp dir
        # with a symlink to the .nemo archive).
        nemo_path = Path(model_dir)
        if nemo_path.is_dir():
            nemo_files = sorted(nemo_path.glob("*.nemo"))
            if nemo_files:
                nemo_path = nemo_files[0].resolve()
        self._nemo_path = str(nemo_path)

        state_dict, nemo_cfg = _load_nemo_archive(model_dir)
        _validate_supported_checkpoint_architecture(state_dict)

        # Extract config from model_config.yaml
        enc_cfg = nemo_cfg.get("encoder", {})
        dec_cfg = nemo_cfg.get("decoder", {})

        hidden = int(nemo_cfg.get("embedding_dim", enc_cfg.get("d_model", 768)))
        enc_layers = int(enc_cfg.get("n_layers", 6))
        dec_layers = int(dec_cfg.get("n_layers", 12))
        enc_heads = int(enc_cfg.get("sa_n_heads", 12))
        dec_heads = int(dec_cfg.get("sa_n_heads", 12))
        enc_ffn = int(enc_cfg.get("d_ffn", 3072))
        dec_ffn = int(dec_cfg.get("d_ffn", 3072))
        enc_kernel_size = int(enc_cfg.get("kernel_size", 3))
        xa_n_heads = int(dec_cfg.get("xa_n_heads", 1))
        xa_d_head = int(dec_cfg.get("xa_d_head", 128))
        max_positions = 2048  # default from position_embeddings shape

        # Infer from actual weight shapes
        if "text_embedding.weight" in state_dict:
            te = _to_np(state_dict["text_embedding.weight"])
            text_vocab_size = te.shape[0]
            hidden = te.shape[1]
        else:
            text_vocab_size = int(nemo_cfg.get("text_vocab_size", 2380))

        if "encoder.position_embeddings.weight" in state_dict:
            pe = _to_np(state_dict["encoder.position_embeddings.weight"])
            max_positions = pe.shape[0]

        # Count codebooks from audio_embeddings.{i}.weight
        num_codebooks = 0
        while f"audio_embeddings.{num_codebooks}.weight" in state_dict:
            num_codebooks += 1
        if num_codebooks == 0:
            num_codebooks = 8

        codebook_size = 2024
        if "audio_embeddings.0.weight" in state_dict:
            codebook_size = _to_np(state_dict["audio_embeddings.0.weight"]).shape[0]

        # Store metadata
        weights["_enc_layers"] = enc_layers
        weights["_dec_layers"] = dec_layers
        weights["_enc_heads"] = enc_heads
        weights["_dec_heads"] = dec_heads
        weights["_enc_ffn"] = enc_ffn
        weights["_dec_ffn"] = dec_ffn
        weights["_hidden_size"] = hidden
        weights["_num_codebooks"] = num_codebooks
        weights["_codebook_size"] = codebook_size
        # audio_magpie.max_source_positions — build-time cap on the encoder
        # position embedding (smaller = proportionally less cross-attention
        # compute, e.g. 256 vs 2048 ≈ 8× less). Supplied through --set or a
        # config profile. The shared builder only forwards opaque
        # family_build_options; this family owns the audio_magpie namespace.
        # 0 means "keep the model default".
        family_options = config.raw.get("_family_build_options", {})
        audio_options = (
            family_options.get("audio_magpie", {})
            if isinstance(family_options, dict) else {}
        )
        override_max_pos = int(audio_options.get("max_source_positions", 0) or 0)
        if 0 < override_max_pos < max_positions:
            print(f"[trtmc build]   Shrinking max_source_positions: "
                  f"{max_positions} -> {override_max_pos}", file=sys.stderr)
            max_positions = override_max_pos
            # Truncate encoder position embedding to new size
            if "encoder.position_embeddings.weight" in state_dict:
                pe = _to_np(state_dict["encoder.position_embeddings.weight"])
                state_dict["encoder.position_embeddings.weight"] = pe[:max_positions]

        weights["_max_source_positions"] = max_positions
        weights["_text_vocab_size"] = text_vocab_size
        weights["_xa_n_heads"] = xa_n_heads
        weights["_xa_d_head"] = xa_d_head
        weights["_enc_kernel_size"] = enc_kernel_size

        # --- Encoder weights ---
        weights["enc_pos_embedding"] = _to_np(
            state_dict["encoder.position_embeddings.weight"])

        for i in range(enc_layers):
            src = f"encoder.layers.{i}"
            pfx = f"enc_layer.{i}"

            # Fused QKV: [3H, H] -> split + transpose
            w_q, w_k, w_v = _split_fused_qkv(
                state_dict[f"{src}.self_attention.qkv_net.weight"], hidden)
            weights[f"{pfx}.w_q"] = w_q
            weights[f"{pfx}.w_k"] = w_k
            weights[f"{pfx}.w_v"] = w_v

            # Output projection: [H, H] -> transpose to [H, H]
            weights[f"{pfx}.w_o"] = _t2d(
                state_dict[f"{src}.self_attention.o_net.weight"])

            # LayerNorm gamma (bias=False, so beta=0)
            weights[f"{pfx}.attn_norm"] = _to_np(
                state_dict[f"{src}.norm_self.weight"])

            # Conv1d FFN with kernel_size=3: keep 3D shape [out, in, K]
            # for TRT convolution
            weights[f"{pfx}.ffn_conv1_weight"] = _to_np(
                state_dict[f"{src}.pos_ff.proj.conv.weight"])  # [3072, 768, 3]
            weights[f"{pfx}.ffn_conv2_weight"] = _to_np(
                state_dict[f"{src}.pos_ff.o_net.conv.weight"])  # [768, 3072, 3]

            # FFN LayerNorm gamma
            weights[f"{pfx}.ffn_norm"] = _to_np(
                state_dict[f"{src}.norm_pos_ff.weight"])

        # Encoder final LayerNorm gamma
        weights["enc_final_norm"] = _to_np(
            state_dict["encoder.norm_out.weight"])

        # --- Decoder weights ---
        weights["dec_pos_embedding"] = _to_np(
            state_dict["decoder.position_embeddings.weight"])

        for i in range(dec_layers):
            src = f"decoder.layers.{i}"
            pfx = f"layer.{i}"

            # Self-attention: fused QKV [3H, H]
            w_q, w_k, w_v = _split_fused_qkv(
                state_dict[f"{src}.self_attention.qkv_net.weight"], hidden)
            weights[f"{pfx}.w_q"] = w_q
            weights[f"{pfx}.w_k"] = w_k
            weights[f"{pfx}.w_v"] = w_v

            weights[f"{pfx}.w_o"] = _t2d(
                state_dict[f"{src}.self_attention.o_net.weight"])

            # Self-attention LayerNorm gamma
            weights[f"{pfx}.input_norm"] = _to_np(
                state_dict[f"{src}.norm_self.weight"])

            # Cross-attention: ASYMMETRIC (1 head, d_head=128)
            # Q: [128, 768] -> transpose to [768, 128]
            weights[f"{pfx}.cross_w_q"] = _t2d(
                state_dict[f"{src}.cross_attention.q_net.weight"])

            # Fused KV: [256, 768] -> split K[128,768] + V[128,768],
            # transpose each to [768, 128]
            cross_w_k, cross_w_v = _split_fused_kv(
                state_dict[f"{src}.cross_attention.kv_net.weight"], xa_d_head)
            weights[f"{pfx}.cross_w_k"] = cross_w_k
            weights[f"{pfx}.cross_w_v"] = cross_w_v

            # Cross O: [768, 128] -> transpose to [128, 768]
            weights[f"{pfx}.cross_w_o"] = _t2d(
                state_dict[f"{src}.cross_attention.o_net.weight"])

            # Cross-attention norms (two separate norms)
            weights[f"{pfx}.norm_xattn_query"] = _to_np(
                state_dict[f"{src}.norm_xattn_query.weight"])
            weights[f"{pfx}.norm_xattn_memory"] = _to_np(
                state_dict[f"{src}.norm_xattn_memory.weight"])

            # FFN: Conv1d kernel_size=1 -> squeeze to linear
            weights[f"{pfx}.w_fc1"] = _squeeze_conv1d_to_linear(
                state_dict[f"{src}.pos_ff.proj.conv.weight"])  # [3072,768,1]->[768,3072]
            weights[f"{pfx}.w_fc2"] = _squeeze_conv1d_to_linear(
                state_dict[f"{src}.pos_ff.o_net.conv.weight"])  # [768,3072,1]->[3072,768]

            # FFN LayerNorm gamma
            weights[f"{pfx}.post_attn_norm"] = _to_np(
                state_dict[f"{src}.norm_pos_ff.weight"])

        # Decoder final LayerNorm gamma
        weights["final_norm"] = _to_np(state_dict["decoder.norm_out.weight"])

        # Output projection: [16192, 768] with bias
        weights["w_out"] = _t2d(state_dict["final_proj.weight"])
        weights["w_out_bias"] = _to_np(state_dict["final_proj.bias"])

        # --- Embeddings ---
        weights["text_embedding"] = _to_np(
            state_dict["text_embedding.weight"])
        for cb in range(num_codebooks):
            weights[f"audio_embedding_{cb}"] = _to_np(
                state_dict[f"audio_embeddings.{cb}.weight"])

        # --- Local transformer weights (codebook AR sampling) ---
        if "local_transformer.position_embeddings.weight" in state_dict:
            weights["lt_pos_embedding"] = _to_np(
                state_dict["local_transformer.position_embeddings.weight"])
            weights["lt_in_proj_w"] = _to_np(
                state_dict["local_transformer_in_projection.weight"]).T
            weights["lt_in_proj_b"] = _to_np(
                state_dict["local_transformer_in_projection.bias"])
            lt_src = "local_transformer.layers.0"
            weights["lt_norm_self"] = _to_np(
                state_dict[f"{lt_src}.norm_self.weight"])
            weights["lt_qkv_net"] = _to_np(
                state_dict[f"{lt_src}.self_attention.qkv_net.weight"]).T
            weights["lt_o_net"] = _to_np(
                state_dict[f"{lt_src}.self_attention.o_net.weight"]).T
            weights["lt_norm_ff"] = _to_np(
                state_dict[f"{lt_src}.norm_pos_ff.weight"])
            weights["lt_ff_proj"] = _to_np(
                state_dict[f"{lt_src}.pos_ff.proj.conv.weight"]).squeeze(-1).T
            weights["lt_ff_out"] = _to_np(
                state_dict[f"{lt_src}.pos_ff.o_net.conv.weight"]).squeeze(-1).T
            for cb in range(num_codebooks):
                weights[f"lt_out_proj_w_{cb}"] = _to_np(
                    state_dict[f"local_transformer_out_projections.{cb}.weight"]).T
                weights[f"lt_out_proj_b_{cb}"] = _to_np(
                    state_dict[f"local_transformer_out_projections.{cb}.bias"])
            lt_hidden = weights["lt_pos_embedding"].shape[1]
            weights["_lt_hidden"] = lt_hidden
            weights["_lt_max_positions"] = weights["lt_pos_embedding"].shape[0]
            weights["_lt_d_head"] = lt_hidden
            weights["_lt_ffn_dim"] = weights["lt_ff_proj"].shape[1]

        # Baked speaker context embedding
        if "baked_context_embedding.weight" in state_dict:
            weights["baked_context_embedding"] = _to_np(
                state_dict["baked_context_embedding.weight"])
        if "baked_context_embedding_len" in state_dict:
            bce_len = state_dict["baked_context_embedding_len"]
            if hasattr(bce_len, 'numpy'):
                weights["baked_context_lengths"] = bce_len.numpy().astype(
                    np.int32)
            else:
                weights["baked_context_lengths"] = np.asarray(
                    bce_len, dtype=np.int32)

        num_speakers = 0
        if "baked_context_embedding.weight" in state_dict:
            num_speakers = _to_np(
                state_dict["baked_context_embedding.weight"]).shape[0]

        # Download NanoCodec .nemo separately and extract codec weights
        try:
            from huggingface_hub import hf_hub_download
            codec_nemo = hf_hub_download(
                "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps",
                "nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo")
            codec_sd, _ = _load_nemo_archive(codec_nemo)
            weights["_codec_state_dict"] = {
                k: _to_np(v) for k, v in codec_sd.items()
                if k.startswith("audio_decoder.")
                or k.startswith("vector_quantizer.")
            }
        except Exception as e:
            print(f"[trtmc build]   Warning: could not load NanoCodec: {e}",
                  file=sys.stderr)
            # Fallback: try to extract from MagpieTTS state_dict
            codec_keys = {
                k[len("codec."):]: _to_np(v)
                for k, v in state_dict.items()
                if k.startswith("codec.") and (
                    k.startswith("codec.audio_decoder.")
                    or k.startswith("codec.vector_quantizer.")
                )
            }
            if codec_keys:
                weights["_codec_state_dict"] = codec_keys

        # Cache audio config
        self._audio_config = {
            "magpie_tts": True,
            "sample_rate": 22050,
            "magpie_num_codebooks": num_codebooks,
            "magpie_codebook_size": codebook_size,
            "magpie_fps": 21.5,
            "magpie_num_speakers": num_speakers,
            "magpie_encoder_layers": enc_layers,
            "magpie_decoder_layers": dec_layers,
            "magpie_hidden_size": hidden,
            "magpie_text_vocab_size": text_vocab_size,
            "magpie_max_source_positions": max_positions,
            "magpie_xa_n_heads": xa_n_heads,
            "magpie_xa_d_head": xa_d_head,
            "magpie_enc_kernel_size": enc_kernel_size,
            "magpie_temperature": 0.6,
            "magpie_cfg_scale": 2.5,
            "magpie_finished_limit_with_eot": 0,
        }

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
        debug_layer_outputs: bool = False,
        parallel_config=None,
    ) -> bytes:
        """Build MagpieTTS decoder TRT engine with dynamic seq_len.

        Uses optimization profiles so the SAME engine handles both:
          - Autoregressive decode: seq_len=1 (profile 0, optimized)
          - Batched prefill: seq_len=ctx_len (profile 1)
        No separate prefill engine needed — saves ~400MB bundle space.
        """
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            require_tensorrt_11_for_tensor_parallel(
                parallel, feature="MagpieTTS tensor-parallel decoder builds")
            if quant_ctx is not None:
                raise ValueError("MagpieTTS tensor-parallel builds do not support quantization")
            from .decoder_tp_builder import build_magpie_tp_decoder_engine
            return build_magpie_tp_decoder_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                debug_layer_outputs=debug_layer_outputs,
                parallel_config=parallel,
            )

        dec_layers = weights["_dec_layers"]
        dec_heads = weights["_dec_heads"]
        dec_ffn = weights["_dec_ffn"]
        hidden = weights["_hidden_size"]
        num_codebooks = weights["_num_codebooks"]
        codebook_size = weights["_codebook_size"]
        max_source_positions = weights["_max_source_positions"]
        xa_n_heads = weights["_xa_n_heads"]
        xa_d_head = weights["_xa_d_head"]
        head_dim = hidden // dec_heads
        output_size = num_codebooks * codebook_size
        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(
                f"Unsupported MagpieTTS precision {precision!r}; "
                "expected fp32 or fp16")
        requested_fp32_layers = {
            int(layer) for layer in config.raw.get("_fp32_layers", ())
        } if precision == "fp16" else set()

        # Determine max prefill length from baked context
        ctx_len = 1
        if "baked_context_lengths" in weights:
            ctx_lengths = np.asarray(
                weights["baked_context_lengths"], dtype=np.int32)
            if ctx_lengths.size > 0:
                ctx_len = max(int(ctx_lengths.max()), 1)

        W = max_cache_length  # shorthand for cache size

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()
        trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        trt_config.clear_flag(trt.BuilderFlag.TF32)

        # Dynamic inputs: seq_len varies from 1 (decode) to ctx_len (prefill)
        input_embed = network.add_input(
            "input_embed", trt.float32, (-1, hidden))
        position_id = network.add_input(
            "position_id", trt.int32, (-1,))
        # 3D mask: [1, seq_len, max_cache + seq_len]
        attention_mask = network.add_input(
            "attention_mask", trt.float32, (1, -1, -1))

        # Per-layer KV cache inputs (fixed shape)
        cache_k_inputs, cache_v_inputs = [], []
        for i in range(dec_layers):
            cache_k_inputs.append(network.add_input(
                graph_ops.layer_tensor_name("cache_k", i),
                work_trt_dtype, (max_cache_length, hidden)))
            cache_v_inputs.append(network.add_input(
                graph_ops.layer_tensor_name("cache_v", i),
                work_trt_dtype, (max_cache_length, hidden)))

        # Per-layer cross-attention inputs (fixed shape)
        # Keep the public cross-memory ABI in FP32 like the text encoder
        # output, then cast at the compute boundary below.
        cross_kv_dtype = trt.float32
        cross_k_inputs, cross_v_inputs = [], []
        for i in range(dec_layers):
            cross_k_inputs.append(network.add_input(
                graph_ops.layer_tensor_name("cross_k", i),
                cross_kv_dtype, (max_source_positions, hidden)))
            cross_v_inputs.append(network.add_input(
                graph_ops.layer_tensor_name("cross_v", i),
                cross_kv_dtype, (max_source_positions, hidden)))

        # Cross-attention prior for monotonic alignment (NeMo inference)
        # Shape [1, 1, max_source_positions] — broadcasts over [xa_heads, seq, max_src]
        # Layers 3-9 get the prior; others get None (vanilla attention).
        cross_attn_prior = network.add_input(
            "cross_attn_prior", trt.float32, (1, 1, max_source_positions))
        if work_trt_dtype != trt.float32:
            input_embed = network.add_cast(
                input_embed, work_trt_dtype).get_output(0)
        prior_layers = set(range(3, 10))  # layers 3,4,5,6,7,8,9

        # Optimization profiles
        # Profile 0: autoregressive (seq_len=1) — common case, optimized
        ar_profile = builder.create_optimization_profile()
        ar_profile.set_shape("input_embed",
            (1, hidden), (1, hidden), (ctx_len, hidden))
        ar_profile.set_shape("position_id",
            (1,), (1,), (ctx_len,))
        ar_profile.set_shape("attention_mask",
            (1, 1, W + 1), (1, 1, W + 1), (1, ctx_len, W + ctx_len))
        trt_config.add_optimization_profile(ar_profile)

        # Profile 1: prefill (seq_len=ctx_len) — one-time, optimized for bulk
        pf_profile = builder.create_optimization_profile()
        pf_profile.set_shape("input_embed",
            (1, hidden), (ctx_len, hidden), (ctx_len, hidden))
        pf_profile.set_shape("position_id",
            (1,), (ctx_len,), (ctx_len,))
        pf_profile.set_shape("attention_mask",
            (1, 1, W + 1), (1, ctx_len, W + ctx_len), (1, ctx_len, W + ctx_len))
        trt_config.add_optimization_profile(pf_profile)

        # Learned positional embedding
        dec_pos_np = weights["dec_pos_embedding"]
        pos_table = graph_ops.add_constant(
            network, dec_pos_np.shape, dec_pos_np, dtype=work_np_dtype)
        pos_embed = network.add_gather(pos_table, position_id, 0)

        # hidden_state = input_embed + positional_embedding
        hidden_state = network.add_elementwise(
            input_embed, pos_embed.get_output(0),
            trt.ElementWiseOperation.SUM).get_output(0)

        eps_tensor = graph_ops.add_constant(
            network, (1, 1), np.array([1e-5], dtype=work_np_dtype),
            dtype=work_np_dtype)
        fp32_eps_tensor = graph_ops.add_constant(
            network, (1, 1), np.array([1e-5], dtype=np.float32),
            dtype=np.float32) if requested_fp32_layers else None
        xa_scale_tensor = graph_ops.add_constant(
            network, (1, 1, 1),
            np.array([1.0 / np.sqrt(max(xa_d_head, 1))],
                     dtype=work_np_dtype),
            dtype=work_np_dtype)

        if debug_layer_outputs:
            _mark_debug_output(network, hidden_state, "debug_embed")

        present_k_outputs, present_v_outputs = [], []
        alignment_layers = [3, 4, 5, 6]  # layers used for alignment estimation
        alignment_weights = []
        for layer_idx in range(dec_layers):
            prefix = f"layer.{layer_idx}"
            use_fp32_layer = layer_idx in requested_fp32_layers
            layer_np_dtype = np.float32 if use_fp32_layer else work_np_dtype
            layer_trt_dtype = trt.float32 if use_fp32_layer else work_trt_dtype
            layer_hidden = hidden_state
            layer_cache_k = cache_k_inputs[layer_idx]
            layer_cache_v = cache_v_inputs[layer_idx]
            layer_cross_k = cross_k_inputs[layer_idx]
            layer_cross_v = cross_v_inputs[layer_idx]
            layer_mask = attention_mask
            layer_scale = xa_scale_tensor
            layer_prior = (
                cross_attn_prior if layer_idx in prior_layers else None)
            if layer_hidden.dtype != layer_trt_dtype:
                layer_hidden = network.add_cast(
                    layer_hidden, layer_trt_dtype).get_output(0)
            for name, tensor in (
                ("cache_k", layer_cache_k),
                ("cache_v", layer_cache_v),
                ("cross_k", layer_cross_k),
                ("cross_v", layer_cross_v),
                ("mask", layer_mask),
                ("scale", layer_scale),
            ):
                if tensor.dtype != layer_trt_dtype:
                    tensor = network.add_cast(
                        tensor, layer_trt_dtype).get_output(0)
                if name == "cache_k":
                    layer_cache_k = tensor
                elif name == "cache_v":
                    layer_cache_v = tensor
                elif name == "cross_k":
                    layer_cross_k = tensor
                elif name == "cross_v":
                    layer_cross_v = tensor
                elif name == "mask":
                    layer_mask = tensor
                else:
                    layer_scale = tensor
            if layer_prior is not None and layer_prior.dtype != layer_trt_dtype:
                layer_prior = network.add_cast(
                    layer_prior, layer_trt_dtype).get_output(0)
            result = _add_magpie_decoder_layer(
                network=network, hidden=layer_hidden,
                cache_k=layer_cache_k,
                cache_v=layer_cache_v,
                cross_k=layer_cross_k,
                cross_v=layer_cross_v,
                attention_mask=layer_mask,
                xa_scale_tensor=layer_scale,
                eps_tensor=(fp32_eps_tensor if use_fp32_layer else eps_tensor),
                weights=weights, prefix=prefix,
                hidden_size=hidden, num_heads=dec_heads, head_dim=head_dim,
                ffn_dim=dec_ffn, max_cache_length=max_cache_length,
                max_source_positions=max_source_positions,
                xa_n_heads=xa_n_heads, xa_d_head=xa_d_head,
                cross_attn_prior=layer_prior, dtype=layer_np_dtype)
            hidden_state = result["hidden"]
            present_k_outputs.append(result["present_k"])
            present_v_outputs.append(result["present_v"])
            # Collect cross-attn weights from alignment layers for averaging
            if layer_idx in alignment_layers:
                alignment_weight = result["cross_attn_weights"]
                if alignment_weight.dtype != trt.float32:
                    alignment_weight = network.add_cast(
                        alignment_weight, trt.float32).get_output(0)
                alignment_weights.append(alignment_weight)
            # Mark last layer's cross-attn weights as output
            if layer_idx == dec_layers - 1:
                _mark_debug_output(network, result["cross_attn_weights"],
                                   "cross_attn_weights")
            if debug_layer_outputs:
                _mark_debug_output(network, hidden_state,
                                   f"debug_hidden_{layer_idx}")

        # Average alignment weights from layers 3-6 → output for C++ alignment tracking
        # Each weight tensor has shape [xa_heads, 1, max_source_positions].
        # Average across layers first, then reduce over heads to get
        # [1, 1, max_source_positions] — matching the C++ buffer allocation.
        if len(alignment_weights) >= 2:
            avg = alignment_weights[0]
            for aw in alignment_weights[1:]:
                avg = network.add_elementwise(
                    avg, aw, trt.ElementWiseOperation.SUM).get_output(0)
            n_align = graph_ops.add_constant(
                network, (1, 1, 1),
                np.array([1.0 / len(alignment_weights)], dtype=np.float32),
                dtype=np.float32)
            avg = network.add_elementwise(
                avg, n_align, trt.ElementWiseOperation.PROD).get_output(0)
            # Reduce over heads (axis 0) → [1, 1, max_source_positions]
            avg_over_heads = network.add_reduce(
                avg, trt.ReduceOperation.AVG, 1 << 0, True)
            _mark_debug_output(network, avg_over_heads.get_output(0),
                               "alignment_weights")

        # Final LayerNorm
        if hidden_state.dtype != work_trt_dtype:
            hidden_state = network.add_cast(
                hidden_state, work_trt_dtype).get_output(0)
        hidden_state = graph_ops.add_layer_norm(
            network, hidden_state, hidden,
            weights["final_norm"],
            np.zeros(hidden, dtype=np.float32), eps_tensor,
            dtype=work_np_dtype)

        # Output pre-logits hidden state for local transformer
        _mark_debug_output(network, hidden_state, "decoder_hidden")

        # Output logits: [seq_len, output_size]
        logits = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, hidden, output_size, weights["w_out"],
            dtype=work_np_dtype)
        logits = graph_ops.add_bias_sum(
            network, logits, output_size, weights["w_out_bias"],
            dtype=work_np_dtype)
        if logits.dtype != trt.float32:
            logits = network.add_cast(logits, trt.float32).get_output(0)
        logits.name = "logits"
        network.mark_output(logits)

        # Present KV outputs: [seq_len, hidden]
        for i in range(dec_layers):
            present_k = present_k_outputs[i]
            present_v = present_v_outputs[i]
            if present_k.dtype != work_trt_dtype:
                present_k = network.add_cast(
                    present_k, work_trt_dtype).get_output(0)
                present_v = network.add_cast(
                    present_v, work_trt_dtype).get_output(0)
            present_k.name = graph_ops.layer_tensor_name("present_k", i)
            present_v.name = graph_ops.layer_tensor_name("present_v", i)
            network.mark_output(present_k)
            network.mark_output(present_v)

        if verbose:
            print(f"[trtmc build] Building MagpieTTS decoder "
                  f"({dec_layers}L, h={hidden}, SA heads={dec_heads}, "
                  f"XA heads={xa_n_heads} d_head={xa_d_head}, "
                  f"ffn={dec_ffn}, cache={max_cache_length}, "
                  f"output={num_codebooks}x{codebook_size}, "
                  f"prefill_ctx_len={ctx_len})",
                  file=sys.stderr)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT MagpieTTS decoder engine build failed")
        return bytes(plan)

    def build_vision_engine(
        self, model_dir: str, config: ModelConfig, weights: WeightDict,
        *, precision: str = "fp32", verbose: bool = False,
    ) -> bytes | None:
        """Build MagpieTTS text encoder engine (stored as vision_engine_plan)."""
        encoder_precision = (
            "fp32"
            if precision == "fp16" and 12 in {
                int(layer) for layer in config.raw.get("_fp32_layers", ())
            }
            else precision
        )
        return _build_magpie_encoder(
            weights, precision=encoder_precision, verbose=verbose)

    def build_extra_engines(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        verbose: bool = False,
        build_timing: dict | None = None,
    ) -> dict:
        """Build extra bundle sections: codec engine + embedding tables."""
        result = {}
        num_codebooks = weights["_num_codebooks"]
        selected_fp32_components = {
            int(layer) for layer in config.raw.get("_fp32_layers", ())
        } if precision == "fp16" else set()

        # Audio embedding tables: 8 codebook tables concatenated
        # Layout: table0 || table1 || ... || table7
        # Each table: [codebook_size, hidden] float32
        embed_parts = []
        for cb in range(num_codebooks):
            key = f"audio_embedding_{cb}"
            if key in weights:
                embed_parts.append(
                    np.asarray(weights[key], dtype=np.float32).ravel())
        if embed_parts:
            result["magpie_audio_embed"] = np.concatenate(embed_parts).tobytes()

        # Text embedding table: [text_vocab_size, hidden] float32
        if "text_embedding" in weights:
            result["magpie_text_embed"] = np.asarray(
                weights["text_embedding"], dtype=np.float32).ravel().tobytes()

        # Baked context embedding: [num_speakers, frames*hidden] float32
        if "baked_context_embedding" in weights:
            result["magpie_context_embed"] = np.asarray(
                weights["baked_context_embedding"],
                dtype=np.float32).ravel().tobytes()

        # Baked context lengths: [num_speakers] int32
        if "baked_context_lengths" in weights:
            result["magpie_context_lengths"] = np.asarray(
                weights["baked_context_lengths"],
                dtype=np.int32).ravel().tobytes()

        # Build local transformer TRT engine (codebook AR sampling)
        if "_lt_hidden" in weights:
            lt_hidden = weights["_lt_hidden"]
            num_cb = weights["_num_codebooks"]
            if verbose:
                print(f"[trtmc build]   Building local transformer engine "
                      f"(hidden={lt_hidden}, 1 layer, {num_cb} codebooks) ...",
                      file=sys.stderr)
            with timed_trt_compile(build_timing, "extra_magpie_local_transformer"):
                lt_plan = _build_local_transformer_engine(
                    weights,
                    precision=(
                        "fp32" if 13 in selected_fp32_components else precision),
                    verbose=verbose)
            result["lt_engine_plan"] = lt_plan
            result["lt_in_projection"] = np.concatenate([
                weights["lt_in_proj_w"].ravel(),
                weights["lt_in_proj_b"].ravel(),
            ]).astype(np.float32).tobytes()
            out_proj_parts = []
            for cb in range(num_cb):
                out_proj_parts.append(weights[f"lt_out_proj_w_{cb}"].ravel())
                out_proj_parts.append(weights[f"lt_out_proj_b_{cb}"].ravel())
            result["lt_out_projections"] = np.concatenate(
                out_proj_parts).astype(np.float32).tobytes()
            result["lt_position_embedding"] = weights[
                "lt_pos_embedding"].astype(np.float32).ravel().tobytes()

        # Build NanoCodec (HiFi-GAN) TRT engine
        codec_sd = weights.get("_codec_state_dict")
        if codec_sd:
            max_codec_frames = min(max_cache_length, 512)
            max_codec_frames = ((max_codec_frames + 63) // 64) * 64

            if verbose:
                print(f"[trtmc build]   Building NanoCodec engine "
                      f"(max_frames={max_codec_frames}) ...",
                      file=sys.stderr)

            from .nanocodec_builder import build_nanocodec_decoder_engine
            with timed_trt_compile(build_timing, "extra_nanocodec_audio_decoder"):
                codec_plan = build_nanocodec_decoder_engine(
                    codec_sd, max_frames=max_codec_frames,
                    precision=(
                        "fp32" if 14 in selected_fp32_components else precision),
                    verbose=verbose)
            result["codec_engine_plan"] = codec_plan

        # Extract and bake IPA tokenizer assets (native C++ tokenizer)
        if hasattr(self, '_nemo_path') and self._nemo_path:
            try:
                ipa_assets = _extract_ipa_assets(self._nemo_path)
                if ipa_assets is not None:
                    for key, data in ipa_assets.items():
                        result[key] = data
                    if verbose:
                        for key, data in ipa_assets.items():
                            print(f"[trtmc build]   Baked {key} "
                                  f"({len(data)} bytes)", file=sys.stderr)
            except Exception as e:
                print(f"[trtmc build]   Warning: IPA extraction failed, "
                      f"Python bridge will be needed at runtime: {e}",
                      file=sys.stderr)

        return result

    def get_audio_config(self, config: ModelConfig) -> dict:
        """Return audio config dict injected into bundle config.json."""
        return self._audio_config.copy()


plugin = MagpieTTSPlugin()


# ---------------------------------------------------------------------------
# Encoder engine builder (causal self-attention + Conv1d FFN)
# ---------------------------------------------------------------------------

def _build_magpie_encoder(
        weights: WeightDict, *, precision: str = "fp32",
        verbose: bool = False) -> bytes:
    """Build the MagpieTTS text encoder TRT engine.

    Input: input_ids [max_source_positions] (int32)
    Output: encoder_output [max_source_positions, hidden_size]

    The encoder is CAUSAL (self-attention uses causal mask). This is unusual
    but confirmed by the NeMo config (is_causal: true). Uses learned positional
    embeddings and Conv1d FFN with kernel_size=3.
    """
    enc_layers = weights["_enc_layers"]
    enc_heads = weights["_enc_heads"]
    enc_ffn = weights["_enc_ffn"]
    hidden = weights["_hidden_size"]
    max_pos = weights["_max_source_positions"]
    enc_kernel_size = weights.get("_enc_kernel_size", 3)
    if precision == "fp16":
        work_np_dtype = np.float16
    elif precision == "fp32":
        work_np_dtype = np.float32
    else:
        raise ValueError(
            f"Unsupported MagpieTTS encoder precision {precision!r}; "
            "expected fp32 or fp16")

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    tc = builder.create_builder_config()
    tc.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    tc.clear_flag(trt.BuilderFlag.TF32)

    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([1e-5], dtype=work_np_dtype),
        dtype=work_np_dtype)

    # Inputs
    input_ids = network.add_input(
        "input_ids", trt.int32, (max_pos,))

    # Embedding tables
    text_embed_table = graph_ops.add_constant(
        network,
        (weights["_text_vocab_size"], hidden),
        weights["text_embedding"], dtype=work_np_dtype)
    pos_embed_table = graph_ops.add_constant(
        network,
        (max_pos, hidden),
        weights["enc_pos_embedding"], dtype=work_np_dtype)

    # Gather embeddings: text + positional
    text_embed = network.add_gather(text_embed_table, input_ids, 0)
    hs = network.add_elementwise(
        text_embed.get_output(0), pos_embed_table,
        trt.ElementWiseOperation.SUM).get_output(0)

    # Encoder self-attention layers (CAUSAL, no KV cache)
    for li in range(enc_layers):
        pfx = f"enc_layer.{li}"

        # Pre-attention LayerNorm
        normed = graph_ops.add_layer_norm(
            network, hs, hidden,
            weights[f"{pfx}.attn_norm"],
            np.zeros(hidden, dtype=np.float32), eps_tensor,
            dtype=work_np_dtype)

        # Causal self-attention (full sequence, with mask)
        attn = _add_causal_self_attention(
            network, normed,
            w_q=weights[f"{pfx}.w_q"],
            w_k=weights[f"{pfx}.w_k"],
            w_v=weights[f"{pfx}.w_v"],
            w_o=weights[f"{pfx}.w_o"],
            hidden_size=hidden, num_heads=enc_heads,
            seq_length=max_pos, dtype=work_np_dtype)

        # Residual
        hs = network.add_elementwise(
            hs, attn, trt.ElementWiseOperation.SUM).get_output(0)

        # Pre-FFN LayerNorm
        normed2 = graph_ops.add_layer_norm(
            network, hs, hidden,
            weights[f"{pfx}.ffn_norm"],
            np.zeros(hidden, dtype=np.float32), eps_tensor,
            dtype=work_np_dtype)

        # Conv1d FFN with kernel_size=3
        ffn_out = _add_conv1d_ffn(
            network, normed2,
            conv1_weight=weights[f"{pfx}.ffn_conv1_weight"],
            conv2_weight=weights[f"{pfx}.ffn_conv2_weight"],
            in_channels=hidden, mid_channels=enc_ffn,
            seq_length=max_pos, kernel_size=enc_kernel_size,
            dtype=work_np_dtype)

        # Residual
        hs = network.add_elementwise(
            hs, ffn_out, trt.ElementWiseOperation.SUM).get_output(0)

    # Final LayerNorm
    hs = graph_ops.add_layer_norm(
        network, hs, hidden,
        weights["enc_final_norm"],
        np.zeros(hidden, dtype=np.float32), eps_tensor,
        dtype=work_np_dtype)

    output = hs
    if output.dtype != trt.float32:
        output = network.add_cast(output, trt.float32).get_output(0)
    output.name = "encoder_output"
    network.mark_output(output)

    if verbose:
        print(f"[trtmc build] Building MagpieTTS encoder "
              f"({enc_layers}L, h={hidden}, heads={enc_heads}, "
              f"ffn={enc_ffn}, kernel={enc_kernel_size}, causal=True)",
              file=sys.stderr)

    plan = builder.build_serialized_network(network, tc)
    if plan is None:
        raise RuntimeError("TensorRT MagpieTTS encoder engine build failed")
    return bytes(plan)


def _add_causal_self_attention(
    network, hidden, *, w_q, w_k, w_v, w_o,
    hidden_size, num_heads, seq_length, dtype=np.float32,
):
    """Full-sequence causal self-attention (no KV cache).

    Input hidden: [seq_length, hidden_size]
    Output: [seq_length, hidden_size]

    Uses a lower-triangular causal mask constant.
    """
    head_dim = hidden_size // num_heads

    # Q, K, V projections: [seq, hidden] @ [hidden, hidden] = [seq, hidden]
    q = graph_ops.add_matmul_rhs_constant(
        network, hidden, hidden_size, hidden_size, w_q, dtype=dtype)
    k = graph_ops.add_matmul_rhs_constant(
        network, hidden, hidden_size, hidden_size, w_k, dtype=dtype)
    v = graph_ops.add_matmul_rhs_constant(
        network, hidden, hidden_size, hidden_size, w_v, dtype=dtype)

    context_flat = graph_ops.add_attention_from_rows(
        network, q, k, v,
        num_heads=num_heads, head_dim=head_dim,
        q_seq=seq_length, kv_seq=seq_length,
        causal=True)

    # Output projection
    out = graph_ops.add_matmul_rhs_constant(
        network, context_flat, hidden_size, hidden_size, w_o, dtype=dtype)

    return out


def _add_conv1d_ffn(
    network, hidden, *, conv1_weight, conv2_weight,
    in_channels, mid_channels, seq_length, kernel_size,
    dtype=np.float32,
):
    """Conv1d FFN: conv1d(kernel) -> GELU -> conv1d(kernel).

    Input hidden: [seq_length, in_channels]
    conv1_weight: [mid_channels, in_channels, kernel_size]
    conv2_weight: [in_channels, mid_channels, kernel_size]
    Output: [seq_length, in_channels]

    For TRT, we reshape 1D conv to 2D: input [1, C, 1, L], kernel [out, in, 1, K].
    Causal padding: pad left by (kernel_size - 1), pad right by 0.
    """
    # Transpose input from [seq, C] to [C, seq], then reshape to 4D [1, C, 1, seq]
    hs_t = network.add_shuffle(hidden)
    hs_t.first_transpose = trt.Permutation([1, 0])
    hs_4d = network.add_shuffle(hs_t.get_output(0))
    hs_4d.reshape_dims = (1, in_channels, 1, seq_length)

    # Conv1: [mid_channels, in_channels, K] -> [mid_channels, in_channels, 1, K]
    c1_w = np.ascontiguousarray(conv1_weight, dtype=dtype)
    c1_w_4d = np.ascontiguousarray(
        c1_w.reshape(mid_channels, in_channels, 1, kernel_size))

    # Causal padding: pad left by (K-1), right by 0
    # TRT padding_nd is symmetric, so we use pre_padding and post_padding
    c1 = network.add_convolution_nd(
        hs_4d.get_output(0), num_output_maps=mid_channels,
        kernel_shape=(1, kernel_size),
        kernel=trt.Weights(c1_w_4d),
        bias=trt.Weights(np.zeros(mid_channels, dtype=dtype)))
    c1.pre_padding = (0, kernel_size - 1)
    c1.post_padding = (0, 0)
    # Output: [1, mid_channels, 1, seq_length]

    # Squeeze to 2D for GELU activation: [mid_channels, seq_length]
    c1_sq = network.add_shuffle(c1.get_output(0))
    c1_sq.reshape_dims = (mid_channels, seq_length)

    # Transpose to [seq_length, mid_channels] for GELU
    c1_t = network.add_shuffle(c1_sq.get_output(0))
    c1_t.first_transpose = trt.Permutation([1, 0])
    act = graph_ops.add_activation(
        network, c1_t.get_output(0), "gelu_new", dtype=dtype)

    # Transpose back to [mid_channels, seq_length] for conv2
    act_t = network.add_shuffle(act)
    act_t.first_transpose = trt.Permutation([1, 0])
    act_4d = network.add_shuffle(act_t.get_output(0))
    act_4d.reshape_dims = (1, mid_channels, 1, seq_length)

    # Conv2: [in_channels, mid_channels, K] -> [in_channels, mid_channels, 1, K]
    c2_w = np.ascontiguousarray(conv2_weight, dtype=dtype)
    c2_w_4d = np.ascontiguousarray(
        c2_w.reshape(in_channels, mid_channels, 1, kernel_size))

    c2 = network.add_convolution_nd(
        act_4d.get_output(0), num_output_maps=in_channels,
        kernel_shape=(1, kernel_size),
        kernel=trt.Weights(c2_w_4d),
        bias=trt.Weights(np.zeros(in_channels, dtype=dtype)))
    c2.pre_padding = (0, kernel_size - 1)
    c2.post_padding = (0, 0)
    # Output: [1, in_channels, 1, seq_length]

    # Reshape back to [seq_length, in_channels]
    c2_sq = network.add_shuffle(c2.get_output(0))
    c2_sq.reshape_dims = (in_channels, seq_length)
    c2_t = network.add_shuffle(c2_sq.get_output(0))
    c2_t.first_transpose = trt.Permutation([1, 0])

    return c2_t.get_output(0)


# ---------------------------------------------------------------------------
# Decoder layer helper (self-attn + asymmetric cross-attn + linear FFN)
# ---------------------------------------------------------------------------

def _add_magpie_decoder_layer(
    *, network, hidden, cache_k, cache_v, cross_k, cross_v,
    attention_mask, xa_scale_tensor, eps_tensor,
    weights, prefix,
    hidden_size, num_heads, head_dim, ffn_dim, max_cache_length,
    max_source_positions, xa_n_heads, xa_d_head,
    cross_attn_prior=None,
    dtype=np.float32,
):
    """Single MagpieTTS decoder layer with dynamic seq_len support.

    Input hidden has shape [seq_len, hidden_size] where seq_len is dynamic:
      - seq_len=1 during autoregressive decode
      - seq_len=ctx_len during batched prefill

    Self-attention: concat(cache_k, present_k) with 3D causal mask.
    Cross-attention: 1 head, d_head=128 (ASYMMETRIC).
    FFN: GELU MLP (Conv1d k=1 squeezed to linear).
    All norms: LayerNorm (bias=False, beta=0).
    """
    attention_size = hidden_size
    xa_attention_size = xa_n_heads * xa_d_head  # 1 * 128 = 128

    # --- Self-attention ---
    normed = graph_ops.add_layer_norm(
        network, hidden, hidden_size,
        weights[f"{prefix}.input_norm"],
        np.zeros(hidden_size, dtype=np.float32), eps_tensor, dtype=dtype)

    q = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, attention_size,
        weights[f"{prefix}.w_q"], dtype=dtype)
    k = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, attention_size,
        weights[f"{prefix}.w_k"], dtype=dtype)
    v = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, attention_size,
        weights[f"{prefix}.w_v"], dtype=dtype)

    present_k = k
    present_v = v

    # Concat with KV cache: [max_cache, attn] + [seq_len, attn] -> [max_cache+seq_len, attn]
    ak = network.add_concatenation([cache_k, k])
    ak.axis = 0
    av = network.add_concatenation([cache_v, v])
    av.axis = 0

    mask_4d = graph_ops.add_3d_mask_to_4d(network, attention_mask)
    cf = graph_ops.add_attention_from_rows(
        network, q, ak.get_output(0), av.get_output(0),
        num_heads=num_heads, head_dim=head_dim,
        q_seq=None, kv_seq=None,
        mask=mask_4d)

    # Output projection + residual
    sa = graph_ops.add_matmul_rhs_constant(
        network, cf, attention_size, hidden_size,
        weights[f"{prefix}.w_o"], dtype=dtype)
    psa = network.add_elementwise(
        hidden, sa, trt.ElementWiseOperation.SUM).get_output(0)

    # --- Asymmetric cross-attention ---
    cn_query = graph_ops.add_layer_norm(
        network, psa, hidden_size,
        weights[f"{prefix}.norm_xattn_query"],
        np.zeros(hidden_size, dtype=np.float32), eps_tensor, dtype=dtype)

    cn_memory_k = graph_ops.add_layer_norm(
        network, cross_k, hidden_size,
        weights[f"{prefix}.norm_xattn_memory"],
        np.zeros(hidden_size, dtype=np.float32), eps_tensor, dtype=dtype)
    cn_memory_v = graph_ops.add_layer_norm(
        network, cross_v, hidden_size,
        weights[f"{prefix}.norm_xattn_memory"],
        np.zeros(hidden_size, dtype=np.float32), eps_tensor, dtype=dtype)

    cq = graph_ops.add_matmul_rhs_constant(
        network, cn_query, hidden_size, xa_attention_size,
        weights[f"{prefix}.cross_w_q"], dtype=dtype)
    ck_proj = graph_ops.add_matmul_rhs_constant(
        network, cn_memory_k, hidden_size, xa_attention_size,
        weights[f"{prefix}.cross_w_k"], dtype=dtype)
    cv_proj = graph_ops.add_matmul_rhs_constant(
        network, cn_memory_v, hidden_size, xa_attention_size,
        weights[f"{prefix}.cross_w_v"], dtype=dtype)

    # Cross-attention stays decomposed because the engine exposes attention
    # probabilities for alignment and optionally reweights them with a prior.
    # Native IAttention only returns context.
    # Q: [seq_len, xa_size] -> [xa_heads, seq_len, xa_d_head]
    cqh = network.add_shuffle(cq)
    cqh.reshape_dims = (-1, xa_n_heads, xa_d_head)
    cqh.second_transpose = trt.Permutation([1, 0, 2])
    # K/V: [max_src, xa_size] -> [xa_heads, max_src, xa_d_head] (fixed)
    ckh = network.add_shuffle(ck_proj)
    ckh.reshape_dims = (max_source_positions, xa_n_heads, xa_d_head)
    ckh.second_transpose = trt.Permutation([1, 0, 2])
    cvh = network.add_shuffle(cv_proj)
    cvh.reshape_dims = (max_source_positions, xa_n_heads, xa_d_head)
    cvh.second_transpose = trt.Permutation([1, 0, 2])

    # Scores: [xa_heads, seq_len, max_src]
    score_q = cqh.get_output(0)
    score_k = ckh.get_output(0)
    score_v = cvh.get_output(0)
    score_scale = xa_scale_tensor
    score_prior = cross_attn_prior
    cs = network.add_elementwise(
        network.add_matrix_multiply(
            score_q, trt.MatrixOperation.NONE,
            score_k, trt.MatrixOperation.TRANSPOSE).get_output(0),
        score_scale, trt.ElementWiseOperation.PROD)
    csm = network.add_softmax(cs.get_output(0))
    csm.axes = 1 << 2

    # Apply attention prior (NeMo inference path: multiply + re-normalize)
    # Prior shape: [1, 1, max_src] — broadcasts over [xa_heads, seq_len, max_src]
    if score_prior is not None:
        # attn_prob = softmax(scores) * prior
        attn_weighted = network.add_elementwise(
            csm.get_output(0), score_prior,
            trt.ElementWiseOperation.PROD)
        # re-normalize: attn_prob / sum(attn_prob, dim=-1, keepdim=True)
        sum_layer = network.add_reduce(
            attn_weighted.get_output(0),
            trt.ReduceOperation.SUM, 1 << 2, True)  # reduce last dim, keep dims
        eps_norm = graph_ops.add_constant(
            network, (1, 1, 1), np.array([1e-8], dtype=dtype),
            dtype=dtype)
        sum_safe = network.add_elementwise(
            sum_layer.get_output(0), eps_norm,
            trt.ElementWiseOperation.SUM)
        csm_final = network.add_elementwise(
            attn_weighted.get_output(0), sum_safe.get_output(0),
            trt.ElementWiseOperation.DIV)
    else:
        csm_final = csm

    # Context: [xa_heads, seq_len, xa_d_head] -> [seq_len, xa_size]
    cc = network.add_matrix_multiply(
        csm_final.get_output(0), trt.MatrixOperation.NONE,
        score_v, trt.MatrixOperation.NONE)
    context = cc.get_output(0)
    ccf = network.add_shuffle(context)
    ccf.first_transpose = trt.Permutation([1, 0, 2])
    ccf.reshape_dims = (-1, xa_attention_size)

    ca = graph_ops.add_matmul_rhs_constant(
        network, ccf.get_output(0), xa_attention_size, hidden_size,
        weights[f"{prefix}.cross_w_o"], dtype=dtype)
    pca = network.add_elementwise(
        psa, ca, trt.ElementWiseOperation.SUM).get_output(0)

    # --- GELU MLP (linear, since decoder Conv1d has kernel_size=1) ---
    fn = graph_ops.add_layer_norm(
        network, pca, hidden_size,
        weights[f"{prefix}.post_attn_norm"],
        np.zeros(hidden_size, dtype=np.float32), eps_tensor, dtype=dtype)

    fc1 = graph_ops.add_matmul_rhs_constant(
        network, fn, hidden_size, ffn_dim, weights[f"{prefix}.w_fc1"],
        dtype=dtype)
    act = graph_ops.add_activation(network, fc1, "gelu_new", dtype=dtype)
    fc2 = graph_ops.add_matmul_rhs_constant(
        network, act, ffn_dim, hidden_size, weights[f"{prefix}.w_fc2"],
        dtype=dtype)

    out = network.add_elementwise(
        pca, fc2, trt.ElementWiseOperation.SUM).get_output(0)

    return {"hidden": out, "present_k": present_k, "present_v": present_v,
            "cross_attn_weights": csm.get_output(0)}


# ---------------------------------------------------------------------------
# Local transformer engine builder (1-layer, codebook AR sampling)
# ---------------------------------------------------------------------------

def _build_local_transformer_engine(  # pragma: no cover
    weights: WeightDict, *, precision: str = "fp32", verbose: bool = False,
) -> bytes:
    """Build TRT engine for the local transformer (AR codebook sampling).

    Tiny 1-layer transformer with KV cache. Called 8 times per frame
    (once per codebook). Input is projected decoder hidden [1, lt_hidden].
    Output is hidden state [1, lt_hidden] (out_proj applied externally).
    """
    lt_hidden = weights["_lt_hidden"]
    lt_d_head = weights["_lt_d_head"]
    lt_ffn_dim = weights["_lt_ffn_dim"]
    lt_max_cache = 8
    attention_window = lt_max_cache + 1
    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "fp32":
        work_np_dtype, work_trt_dtype = np.float32, trt.float32
    else:
        raise ValueError(
            f"Unsupported Magpie local precision {precision!r}; "
            "expected fp32 or fp16")

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)

    input_embed = network.add_input(
        "input_embed", trt.float32, (1, lt_hidden))
    position_id = network.add_input(
        "position_id", trt.int32, (1,))
    attention_mask = network.add_input(
        "attention_mask", trt.float32, (1, attention_window))
    cache_k = network.add_input(
        "cache_k_0", trt.float32, (lt_max_cache, lt_hidden))
    cache_v = network.add_input(
        "cache_v_0", trt.float32, (lt_max_cache, lt_hidden))
    if work_trt_dtype != trt.float32:
        input_embed = network.add_cast(
            input_embed, work_trt_dtype).get_output(0)
        attention_mask = network.add_cast(
            attention_mask, work_trt_dtype).get_output(0)
        cache_k = network.add_cast(cache_k, work_trt_dtype).get_output(0)
        cache_v = network.add_cast(cache_v, work_trt_dtype).get_output(0)

    pos_np = weights["lt_pos_embedding"]
    pos_table = graph_ops.add_constant(
        network, pos_np.shape, pos_np, dtype=work_np_dtype)
    pos_embed = network.add_gather(pos_table, position_id, 0)

    hidden_state = network.add_elementwise(
        input_embed, pos_embed.get_output(0),
        trt.ElementWiseOperation.SUM).get_output(0)

    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([1e-5], dtype=work_np_dtype),
        dtype=work_np_dtype)

    # Self-attention (1 head, d_head=lt_hidden, causal)
    normed = graph_ops.add_layer_norm(
        network, hidden_state, lt_hidden,
        weights["lt_norm_self"],
        np.zeros(lt_hidden, dtype=np.float32), eps_tensor,
        dtype=work_np_dtype)

    qkv = graph_ops.add_matmul_rhs_constant(
        network, normed, lt_hidden, 3 * lt_hidden,
        weights["lt_qkv_net"], dtype=work_np_dtype)

    q_slice = network.add_slice(qkv, (0, 0), (1, lt_hidden), (1, 1))
    k_slice = network.add_slice(qkv, (0, lt_hidden), (1, lt_hidden), (1, 1))
    v_slice = network.add_slice(qkv, (0, 2 * lt_hidden), (1, lt_hidden), (1, 1))

    present_k = k_slice.get_output(0)
    present_v = v_slice.get_output(0)

    ak = network.add_concatenation([cache_k, present_k])
    ak.axis = 0
    av = network.add_concatenation([cache_v, present_v])
    av.axis = 0

    mask_4d = graph_ops.add_2d_mask_to_4d(network, attention_mask)
    cf = graph_ops.add_attention_from_rows(
        network,
        q_slice.get_output(0),
        ak.get_output(0),
        av.get_output(0),
        num_heads=1,
        head_dim=lt_d_head,
        q_seq=1,
        kv_seq=attention_window,
        mask=mask_4d)

    sa = graph_ops.add_matmul_rhs_constant(
        network, cf, lt_hidden, lt_hidden,
        weights["lt_o_net"], dtype=work_np_dtype)
    psa = network.add_elementwise(
        hidden_state, sa, trt.ElementWiseOperation.SUM).get_output(0)

    # FFN (GELU MLP)
    fn = graph_ops.add_layer_norm(
        network, psa, lt_hidden,
        weights["lt_norm_ff"],
        np.zeros(lt_hidden, dtype=np.float32), eps_tensor,
        dtype=work_np_dtype)

    fc1 = graph_ops.add_matmul_rhs_constant(
        network, fn, lt_hidden, lt_ffn_dim, weights["lt_ff_proj"],
        dtype=work_np_dtype)
    act = graph_ops.add_activation(
        network, fc1, "gelu_new", dtype=work_np_dtype)
    fc2 = graph_ops.add_matmul_rhs_constant(
        network, act, lt_ffn_dim, lt_hidden, weights["lt_ff_out"],
        dtype=work_np_dtype)

    out = network.add_elementwise(
        psa, fc2, trt.ElementWiseOperation.SUM).get_output(0)

    if out.dtype != trt.float32:
        out = network.add_cast(out, trt.float32).get_output(0)
        present_k = network.add_cast(
            present_k, trt.float32).get_output(0)
        present_v = network.add_cast(
            present_v, trt.float32).get_output(0)
    out.name = "lt_output"
    network.mark_output(out)
    present_k.name = "present_k_0"
    network.mark_output(present_k)
    present_v.name = "present_v_0"
    network.mark_output(present_v)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT local transformer engine build failed")
    return bytes(plan)


# ---------------------------------------------------------------------------
# Debug output helper
# ---------------------------------------------------------------------------

def _mark_debug_output(network, tensor, name):
    identity = network.add_identity(tensor)
    cast = network.add_cast(identity.get_output(0), trt.float32)
    out = cast.get_output(0)
    out.name = name
    network.mark_output(out)
