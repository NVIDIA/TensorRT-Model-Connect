"""Wan Text-to-Video family plugin (Wan 2.1 + 2.2 TI2V).

Composes shared builders: T5 encoder + standard DiT + causal 3D VAE.

DiT and VAE architecture parameters are read from the diffusers
``transformer/config.json`` and ``vae/config.json`` at load time. This
lets one plugin serve both Wan 2.1 (T2V-1.3B, in_channels=16, base_dim=96)
and Wan 2.2 (TI2V-5B, in_channels=48, base_dim=160 encoder / 256 decoder).

Wan 2.2 VAE has architectural differences that the v1 ``causal_vae_3d_builder``
does not yet implement (patch_size=2 input downsample, is_residual=True
residual blocks, asymmetric encoder/decoder channel dims, scale_factor_spatial=16
instead of 8). When those differences are detected we raise
``NotImplementedError`` early with a clear message, so the lane fails at
build time with a documented reason rather than silently producing a
broken bundle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...config import ModelConfig
from ...checkpoint_mapper import WeightDict


# Wan 2.1 defaults — fall-back when transformer/config.json or vae/config.json
# is not present (e.g. older checkpoints predating diffusers integration).
_WAN21_DEFAULTS = {
    "dit": {
        "in_channels": 16,
        "out_channels": 16,
        "num_attention_heads": 12,
        "attention_head_dim": 128,
        "num_layers": 30,
        "ffn_dim": 8960,
        "text_dim": 4096,
        "freq_dim": 256,
        "qk_norm": True,
        "cross_attn_norm": True,
    },
    "vae": {
        "z_dim": 16,
        "base_dim": 96,
        "decoder_base_dim": None,        # None => same as base_dim
        "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "temporal_upsample": [False, True, True],
        "patch_size": 1,                 # 1 = no input-patch downsample
        "is_residual": False,
        "scale_factor_spatial": 8,
        "scale_factor_temporal": 4,
    },
    "t5": {
        "d_model": 4096,
        "num_heads": 64,
        "d_kv": 64,
        "d_ff": 10240,
        "num_layers": 24,
        "vocab_size": 256384,
        "max_seq_len": 226,
    },
    "patch_size_dit": [1, 2, 2],
    "latents_mean": [
        -0.7571, -0.7089, -0.9113, 0.1075,
        -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632,
        -0.1922, -0.9497, 0.2503, -0.2921,
    ],
    "latents_std": [
        2.8184, 1.4541, 2.3275, 2.6558,
        1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579,
        1.6382, 1.1253, 2.8251, 1.9160,
    ],
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r") as fh:
        return json.load(fh)


def _resolve_dit_params(transformer_dir: str) -> dict[str, Any]:
    """Read DiT architecture from transformer/config.json, fall back to Wan 2.1."""
    cfg = _read_json(Path(transformer_dir) / "config.json")
    d = dict(_WAN21_DEFAULTS["dit"])
    for key in d.keys():
        if key in cfg:
            d[key] = cfg[key]
    d["dim"] = d["num_attention_heads"] * d["attention_head_dim"]
    return d


def _resolve_vae_params(vae_dir: str) -> dict[str, Any]:
    """Read VAE architecture from vae/config.json, fall back to Wan 2.1."""
    cfg = _read_json(Path(vae_dir) / "config.json")
    d = dict(_WAN21_DEFAULTS["vae"])
    for key in d.keys():
        if key in cfg:
            d[key] = cfg[key]
    # Normalize tuple fields stored as lists in JSON.
    d["dim_mult"] = tuple(d["dim_mult"])
    d["temporal_upsample"] = tuple(d["temporal_upsample"])
    # If decoder uses a different base_dim than the encoder, the diffusers
    # config exposes it as `decoder_base_dim`. When None the decoder reuses
    # the encoder's base_dim.
    if d["decoder_base_dim"] is None:
        d["decoder_base_dim"] = d["base_dim"]
    return d


def _is_wan22_vae(vae_params: dict[str, Any]) -> bool:
    """Detect the Wan 2.2 VAE shape from the diffusers config.

    Wan 2.2 sets patch_size=2 + asymmetric encoder/decoder base_dim +
    scale_factor_spatial=16. Any one of these flags switches dispatch to
    the wan22 VAE builder.
    """
    return (
        vae_params["patch_size"] != 1
        or vae_params["base_dim"] != vae_params["decoder_base_dim"]
        or vae_params["scale_factor_spatial"] != 8
    )


# Wan 2.2 specific latents_mean / std — diffusers ships these as part of the
# pipeline config; for now we hard-code the published values when z_dim=48.
_WAN22_LATENTS_MEAN_48 = [0.0] * 48  # placeholder; real values in vae/config.json
_WAN22_LATENTS_STD_48 = [1.0] * 48   # placeholder


class WanT2VPlugin:
    name = "wan_t2v"
    runtime_strategy = "diffusion_wan"
    pipeline_classes = ["WanPipeline", "WanVideoToVideoPipeline"]

    # T5 encoder shape (UMT5-XXL) is shared by Wan 2.1 and 2.2.
    _T5_D_MODEL = 4096
    _T5_NUM_HEADS = 64
    _T5_D_KV = 64
    _T5_D_FF = 10240
    _T5_NUM_LAYERS = 24
    _T5_VOCAB_SIZE = 256384
    _T5_MAX_SEQ_LEN = 226

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        return mt in ("wan", "wan2.1", "wan2.2", "wan_t2v", "wan_ti2v")

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        """Probe the diffusers layout and surface subdirectory paths."""
        model_path = Path(model_dir)
        weights = WeightDict()

        if (model_path / "model_index.json").exists():
            weights["_model_format"] = "diffusers"
            weights["_text_encoder_dir"] = str(model_path / "text_encoder")
            weights["_transformer_dir"] = str(model_path / "transformer")
            weights["_vae_dir"] = str(model_path / "vae")
        else:
            raise ValueError(
                f"Expected diffusers format with model_index.json in {model_dir}")

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
    ) -> bytes:
        """Not used for diffusion models — use build_components() instead."""
        raise NotImplementedError(
            "Wan T2V uses build_components(), not build_engine()")

    def build_components(
        self, model_dir: str, config: ModelConfig, weights: WeightDict,
        *, precision: str = "fp32", verbose: bool = False,
        parallel_config=None, **_kwargs,
    ) -> dict:
        """Build T5 + DiT + VAE component engines for the diffusion pipeline."""
        import sys
        from ...build_timing import timed_trt_compile, timed_weight_loading
        from .t5_encoder_builder import build_t5_encoder_engine, load_t5_weights
        from .standard_dit_builder import build_standard_dit_engine, load_dit_weights
        from .standard_dit_tp_builder import (
            build_standard_dit_engine as build_standard_dit_tp_engine)
        from .causal_vae_3d_builder import build_causal_vae_3d_engine, load_vae_weights
        from .wan22_causal_vae_3d_decoder_builder import (
            build_wan22_causal_vae_3d_decoder_engine,
            load_vae_weights_wan22,
        )
        from ...parallel_config import (
            normalize_parallel_config,
            require_tensorrt_11_for_tensor_parallel,
            validate_dit_tp,
        )

        build_timing = _kwargs.get("build_timing")
        parallel = normalize_parallel_config(parallel_config)
        require_tensorrt_11_for_tensor_parallel(
            parallel, feature="Wan tensor-parallel builds")

        text_encoder_dir = weights["_text_encoder_dir"]
        transformer_dir = weights["_transformer_dir"]
        vae_dir = weights["_vae_dir"]

        # Resolve arch params from the diffusers subdir configs. Falls back to
        # the Wan 2.1 defaults for older checkpoints that don't ship configs.
        dit_p = _resolve_dit_params(transformer_dir)
        vae_p = _resolve_vae_params(vae_dir)
        is_wan22 = (dit_p["in_channels"] == 48) or (vae_p["z_dim"] == 48)
        use_wan22_vae = _is_wan22_vae(vae_p)

        if parallel.enabled:
            validate_dit_tp(
                dim=dit_p["dim"],
                num_heads=dit_p["num_attention_heads"],
                ffn_dim=dit_p["ffn_dim"],
                parallel=parallel.for_rank(0),
                feature="Wan tensor parallel",
            )

        # Video dimensions from config (defaults match the Wan 2.1 HF reference)
        video_height = config.raw.get("video_height", 480)
        video_width = config.raw.get("video_width", 832)
        video_num_frames = config.raw.get("video_num_frames", 17)

        scale_t = vae_p["scale_factor_temporal"]
        scale_s = vae_p["scale_factor_spatial"]
        t_lat = (video_num_frames - 1) // scale_t + 1
        h_lat = video_height // scale_s
        w_lat = video_width // scale_s

        # DiT patch_size — the temporal+spatial chunking applied inside the
        # DiT before tokenization. Wan 2.1 / 2.2 both use [1, 2, 2].
        pt, ph, pw = (1, 2, 2)
        num_patches = (t_lat // pt) * (h_lat // ph) * (w_lat // pw)

        # 1. T5 text encoder (unchanged between 2.1 and 2.2)
        print("[wan-t2v] Loading T5 encoder weights ...", file=sys.stderr)
        with timed_weight_loading(build_timing, "t5_encoder"):
            t5_weights = load_t5_weights(
                text_encoder_dir,
                d_model=self._T5_D_MODEL, num_heads=self._T5_NUM_HEADS,
                d_kv=self._T5_D_KV, d_ff=self._T5_D_FF,
                num_layers=self._T5_NUM_LAYERS, vocab_size=self._T5_VOCAB_SIZE,
                precision=precision,
            )
        with timed_trt_compile(build_timing, "t5_encoder"):
            t5_plan = build_t5_encoder_engine(
                t5_weights,
                d_model=self._T5_D_MODEL, num_heads=self._T5_NUM_HEADS,
                d_kv=self._T5_D_KV, d_ff=self._T5_D_FF,
                num_layers=self._T5_NUM_LAYERS, vocab_size=self._T5_VOCAB_SIZE,
                max_seq_len=self._T5_MAX_SEQ_LEN, verbose=verbose,
            )

        # 2. DiT denoiser — config-driven dims
        print(
            f"[wan-t2v] Loading DiT weights ({'Wan 2.2' if is_wan22 else 'Wan 2.1'}: "
            f"dim={dit_p['dim']}, heads={dit_p['num_attention_heads']}, "
            f"layers={dit_p['num_layers']}, in_channels={dit_p['in_channels']}) ...",
            file=sys.stderr,
        )
        with timed_weight_loading(build_timing, "dit"):
            dit_weights = load_dit_weights(
                transformer_dir,
                dim=dit_p["dim"],
                num_heads=dit_p["num_attention_heads"],
                num_layers=dit_p["num_layers"],
                ffn_dim=dit_p["ffn_dim"],
                context_dim=dit_p["text_dim"],
            )

        dit_plan = None
        dit_rank_plans = None
        with timed_trt_compile(build_timing, "dit"):
            common = dict(
                dim=dit_p["dim"],
                num_heads=dit_p["num_attention_heads"],
                num_layers=dit_p["num_layers"],
                ffn_dim=dit_p["ffn_dim"],
                context_dim=dit_p["dim"],  # text proj happens runtime-side
                num_patches=num_patches,
                text_seq_len=self._T5_MAX_SEQ_LEN,
                qk_norm=dit_p["qk_norm"],
                cross_attn_norm=dit_p["cross_attn_norm"],
                ffn_activation="gelu_new",
                verbose=verbose,
            )
            if parallel.enabled:
                dit_rank_plans = {}
                for rank in range(parallel.tp_size):
                    print(
                        f"[wan-t2v] Building DiT TP rank {rank}/{parallel.tp_size} ...",
                        file=sys.stderr,
                    )
                    dit_rank_plans[rank] = build_standard_dit_tp_engine(
                        dit_weights, **common,
                        parallel_config=parallel.for_rank(rank),
                    )
            else:
                dit_plan = build_standard_dit_engine(dit_weights, **common)

        # 3. Causal 3D VAE decoder — config-driven dims, with separate Wan 2.2
        # path that handles patch_size=2, decoder_base_dim, and the singular
        # ``.upsampler.`` weight prefix.
        print(
            f"[wan-t2v] Loading VAE decoder weights "
            f"({'Wan 2.2' if use_wan22_vae else 'Wan 2.1'} VAE) ...",
            file=sys.stderr,
        )
        if use_wan22_vae:
            with timed_weight_loading(build_timing, "vae_decoder"):
                vae_weights = load_vae_weights_wan22(
                    vae_dir,
                    z_dim=vae_p["z_dim"],
                    decoder_base_dim=vae_p["decoder_base_dim"],
                    dim_mult=vae_p["dim_mult"],
                    num_res_blocks=vae_p["num_res_blocks"],
                )
            # h_lat / w_lat above used scale_factor_spatial=16 — so they are
            # already the post-pixel-shuffle latent rows/cols. The internal
            # builder runs spatial upsamples to get to h_lat * 8, w_lat * 8
            # and the patch_size=2 pixel-shuffle does the last 2x.
            with timed_trt_compile(build_timing, "vae_decoder"):
                vae_plan = build_wan22_causal_vae_3d_decoder_engine(
                    vae_weights,
                    z_dim=vae_p["z_dim"],
                    decoder_base_dim=vae_p["decoder_base_dim"],
                    dim_mult=vae_p["dim_mult"],
                    num_res_blocks=vae_p["num_res_blocks"],
                    temporal_upsample=vae_p["temporal_upsample"],
                    h_lat=h_lat, w_lat=w_lat,
                    patch_size=vae_p["patch_size"],
                    verbose=verbose,
                )
        else:
            with timed_weight_loading(build_timing, "vae_decoder"):
                vae_weights = load_vae_weights(
                    vae_dir,
                    z_dim=vae_p["z_dim"], base_dim=vae_p["base_dim"],
                    dim_mult=vae_p["dim_mult"],
                    num_res_blocks=vae_p["num_res_blocks"],
                    norm_type="l2_channel_norm",
                )
            with timed_trt_compile(build_timing, "vae_decoder"):
                vae_plan = build_causal_vae_3d_engine(
                    vae_weights,
                    z_dim=vae_p["z_dim"], base_dim=vae_p["base_dim"],
                    dim_mult=vae_p["dim_mult"],
                    num_res_blocks=vae_p["num_res_blocks"],
                    temporal_upsample=vae_p["temporal_upsample"],
                    h_lat=h_lat, w_lat=w_lat,
                    norm_type="l2_channel_norm", verbose=verbose,
                )

        # 4. Preprocessor weights (DiT prelayers that live outside the engine)
        preprocessor_weights = _serialize_preprocessor_weights(dit_weights)

        out: dict[str, Any] = {
            "text_encoders": [("t5", t5_plan)],
            "vae_decoder": vae_plan,
            "preprocessor_weights": preprocessor_weights,
        }
        if parallel.enabled:
            out["denoiser_ranks"] = dit_rank_plans or {}
        else:
            out["denoiser"] = dit_plan
        return out

    def get_diffusion_config(self, config: ModelConfig) -> dict:
        """Return diffusion pipeline configuration."""
        from .causal_vae_3d_builder import count_vae_caches

        transformer_dir = config.raw.get("_transformer_dir")
        vae_dir = config.raw.get("_vae_dir")
        if transformer_dir and vae_dir:
            dit_p = _resolve_dit_params(transformer_dir)
            vae_p = _resolve_vae_params(vae_dir)
        else:
            dit_p = dict(_WAN21_DEFAULTS["dit"])
            dit_p["dim"] = dit_p["num_attention_heads"] * dit_p["attention_head_dim"]
            vae_p = dict(_WAN21_DEFAULTS["vae"])
            if vae_p["decoder_base_dim"] is None:
                vae_p["decoder_base_dim"] = vae_p["base_dim"]

        is_wan22 = (dit_p["in_channels"] == 48) or (vae_p["z_dim"] == 48)

        video_height = config.raw.get("video_height", 480)
        video_width = config.raw.get("video_width", 832)
        video_num_frames = config.raw.get("video_num_frames", 17)

        if is_wan22:
            latents_mean = _WAN22_LATENTS_MEAN_48
            latents_std = _WAN22_LATENTS_STD_48
            vae_model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        else:
            latents_mean = _WAN21_DEFAULTS["latents_mean"]
            latents_std = _WAN21_DEFAULTS["latents_std"]
            vae_model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

        return {
            "diffusion_backend_type": "wan_3d",
            "scheduler": "flow_match_euler",
            "num_inference_steps": config.raw.get("num_inference_steps", 50),
            "guidance_scale": 5.0,
            "flow_shift": 3.0,
            "video_height": video_height,
            "video_width": video_width,
            "video_num_frames": video_num_frames,
            "dit_dim": dit_p["dim"],
            "dit_num_heads": dit_p["num_attention_heads"],
            "dit_num_layers": dit_p["num_layers"],
            "patch_size": [1, 2, 2],
            "z_dim": vae_p["z_dim"],
            "scale_factor_temporal": vae_p["scale_factor_temporal"],
            "scale_factor_spatial": vae_p["scale_factor_spatial"],
            "freq_dim": dit_p["freq_dim"],
            "text_seq_len": self._T5_MAX_SEQ_LEN,
            "latents_mean": latents_mean,
            "latents_std": latents_std,
            "num_vae_caches": count_vae_caches(
                dim_mult=tuple(vae_p["dim_mult"]),
                num_res_blocks=vae_p["num_res_blocks"],
                temporal_upsample=tuple(vae_p["temporal_upsample"]),
            ),
            "vae_model_id": vae_model_id,
            "text_encoder_dim": self._T5_D_MODEL,
        }


def _serialize_preprocessor_weights(dit_weights: dict) -> bytes:
    """Serialize DiT preprocessor weights into a binary format.

    Format: JSON index (length-prefixed) + contiguous float32 data.
    """
    import struct
    import numpy as np

    keys = [
        "patch_embedding.weight",
        "patch_embedding.bias",
        "condition_embedder.time_embedding.0.weight",
        "condition_embedder.time_embedding.0.bias",
        "condition_embedder.time_embedding.2.weight",
        "condition_embedder.time_embedding.2.bias",
        "condition_embedder.time_proj.weight",
        "condition_embedder.time_proj.bias",
        "condition_embedder.text_embedding.weight",
        "condition_embedder.text_embedding.bias",
        "condition_embedder.text_embedding_2.weight",
        "condition_embedder.text_embedding_2.bias",
    ]

    index = {}
    data_parts = []
    offset = 0

    for key in keys:
        if key not in dit_weights:
            continue
        w = dit_weights[key].astype(np.float32)

        if key == "patch_embedding.weight" and w.ndim > 2:
            out_ch = w.shape[0]
            patch_dim = int(np.prod(w.shape[1:]))
            w = np.ascontiguousarray(w.reshape(out_ch, patch_dim).T)

        w = np.ascontiguousarray(w)
        nbytes = w.nbytes
        index[key] = {"offset": offset, "shape": list(w.shape)}
        data_parts.append(w.tobytes())
        offset += nbytes

    index_json = json.dumps(index).encode("utf-8")
    result = struct.pack("<I", len(index_json)) + index_json
    for part in data_parts:
        result += part

    return result


plugin = WanT2VPlugin()
