"""Wan2.1 Text-to-Video family plugin.

Composes shared builders: T5 encoder + standard DiT + causal 3D VAE.
"""

from __future__ import annotations

from .config import ModelConfig
from .checkpoint_mapper import WeightDict


class WanT2VPlugin:
    name = "wan_t2v"
    runtime_strategy = "diffusion_wan"
    pipeline_classes = ["WanPipeline", "WanVideoToVideoPipeline"]

    # Wan2.1-T2V-1.3B architecture params
    _T5_D_MODEL = 4096
    _T5_NUM_HEADS = 64
    _T5_D_KV = 64
    _T5_D_FF = 10240
    _T5_NUM_LAYERS = 24
    _T5_VOCAB_SIZE = 256384
    _T5_MAX_SEQ_LEN = 226

    _DIT_DIM = 1536
    _DIT_NUM_HEADS = 12
    _DIT_NUM_LAYERS = 30
    _DIT_FFN_DIM = 8960
    _DIT_CONTEXT_DIM = 4096
    _DIT_FREQ_DIM = 256

    _VAE_Z_DIM = 16
    _VAE_BASE_DIM = 96
    _VAE_DIM_MULT = (1, 2, 4, 4)
    _VAE_NUM_RES_BLOCKS = 2
    _VAE_TEMPORAL_UPSAMPLE = (False, True, True)

    _PATCH_SIZE = [1, 2, 2]
    _SCALE_FACTOR_TEMPORAL = 4
    _SCALE_FACTOR_SPATIAL = 8

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        return mt in ("wan", "wan2.1", "wan_t2v")

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        """Load weights from all three subdirectories."""
        from pathlib import Path

        model_path = Path(model_dir)
        weights = WeightDict()

        # Detect diffusers-format: has model_index.json + subdirs
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
        """Build all three component engines."""
        from ...build_timing import timed_trt_compile, timed_weight_loading
        from .t5_encoder_builder import build_t5_encoder_engine, load_t5_weights
        from .standard_dit_builder import build_standard_dit_engine, load_dit_weights
        from .standard_dit_tp_builder import (
            build_standard_dit_engine as build_standard_dit_tp_engine)
        from .causal_vae_3d_builder import build_causal_vae_3d_engine, load_vae_weights
        from ...parallel_config import (
            normalize_parallel_config,
            require_tensorrt_11_for_tensor_parallel,
            validate_dit_tp,
        )
        build_timing = _kwargs.get("build_timing")
        parallel = normalize_parallel_config(parallel_config)
        require_tensorrt_11_for_tensor_parallel(
            parallel, feature="Wan tensor-parallel builds")
        if parallel.enabled:
            validate_dit_tp(
                dim=self._DIT_DIM,
                num_heads=self._DIT_NUM_HEADS,
                ffn_dim=self._DIT_FFN_DIM,
                parallel=parallel.for_rank(0),
                feature="Wan tensor parallel",
            )

        text_encoder_dir = weights["_text_encoder_dir"]
        transformer_dir = weights["_transformer_dir"]
        vae_dir = weights["_vae_dir"]

        # Video dimensions from config (480x832@17fr matches HF reference)
        video_height = config.raw.get("video_height", 480)
        video_width = config.raw.get("video_width", 832)
        video_num_frames = config.raw.get("video_num_frames", 17)

        t_lat = (video_num_frames - 1) // self._SCALE_FACTOR_TEMPORAL + 1
        h_lat = video_height // self._SCALE_FACTOR_SPATIAL
        w_lat = video_width // self._SCALE_FACTOR_SPATIAL
        pt, ph, pw = self._PATCH_SIZE
        num_patches = (t_lat // pt) * (h_lat // ph) * (w_lat // pw)

        # 1. T5 text encoder
        import sys
        print("[wan-t2v] Loading T5 encoder weights ...", file=sys.stderr)
        with timed_weight_loading(build_timing, "t5_encoder"):
            t5_weights = load_t5_weights(
                text_encoder_dir,
                d_model=self._T5_D_MODEL,
                num_heads=self._T5_NUM_HEADS,
                d_kv=self._T5_D_KV,
                d_ff=self._T5_D_FF,
                num_layers=self._T5_NUM_LAYERS,
                vocab_size=self._T5_VOCAB_SIZE,
                precision=precision,
            )
        with timed_trt_compile(build_timing, "t5_encoder"):
            t5_plan = build_t5_encoder_engine(
                t5_weights,
                d_model=self._T5_D_MODEL,
                num_heads=self._T5_NUM_HEADS,
                d_kv=self._T5_D_KV,
                d_ff=self._T5_D_FF,
                num_layers=self._T5_NUM_LAYERS,
                vocab_size=self._T5_VOCAB_SIZE,
                max_seq_len=self._T5_MAX_SEQ_LEN,
                verbose=verbose,
            )

        # 2. DiT denoiser
        print("[wan-t2v] Loading DiT weights ...", file=sys.stderr)
        with timed_weight_loading(build_timing, "dit"):
            dit_weights = load_dit_weights(
                transformer_dir,
                dim=self._DIT_DIM,
                num_heads=self._DIT_NUM_HEADS,
                num_layers=self._DIT_NUM_LAYERS,
                ffn_dim=self._DIT_FFN_DIM,
                context_dim=self._DIT_CONTEXT_DIM,
            )
        # Note: context_dim=dim (1536) because the text embedding projection
        # (4096->1536) is handled externally in the runner, so cross-attn
        # K/V weights are [dim, dim].
        dit_plan = None
        dit_rank_plans = None
        with timed_trt_compile(build_timing, "dit"):
            if parallel.enabled:
                dit_rank_plans = {}
                for rank in range(parallel.tp_size):
                    print(
                        f"[wan-t2v] Building DiT TP rank {rank}/{parallel.tp_size} ...",
                        file=sys.stderr,
                    )
                    dit_rank_plans[rank] = build_standard_dit_tp_engine(
                        dit_weights,
                        dim=self._DIT_DIM,
                        num_heads=self._DIT_NUM_HEADS,
                        num_layers=self._DIT_NUM_LAYERS,
                        ffn_dim=self._DIT_FFN_DIM,
                        context_dim=self._DIT_DIM,
                        num_patches=num_patches,
                        text_seq_len=self._T5_MAX_SEQ_LEN,
                        qk_norm=True,
                        cross_attn_norm=True,
                        ffn_activation="gelu_new",
                        verbose=verbose,
                        parallel_config=parallel.for_rank(rank),
                    )
            else:
                dit_plan = build_standard_dit_engine(
                    dit_weights,
                    dim=self._DIT_DIM,
                    num_heads=self._DIT_NUM_HEADS,
                    num_layers=self._DIT_NUM_LAYERS,
                    ffn_dim=self._DIT_FFN_DIM,
                    context_dim=self._DIT_DIM,
                    num_patches=num_patches,
                    text_seq_len=self._T5_MAX_SEQ_LEN,
                    qk_norm=True,
                    cross_attn_norm=True,
                    ffn_activation="gelu_new",
                    verbose=verbose,
                )

        # 3. Causal 3D VAE decoder
        print("[wan-t2v] Loading VAE decoder weights ...", file=sys.stderr)
        with timed_weight_loading(build_timing, "vae_decoder"):
            vae_weights = load_vae_weights(
                vae_dir,
                z_dim=self._VAE_Z_DIM,
                base_dim=self._VAE_BASE_DIM,
                dim_mult=self._VAE_DIM_MULT,
                num_res_blocks=self._VAE_NUM_RES_BLOCKS,
                norm_type="l2_channel_norm",
            )
        with timed_trt_compile(build_timing, "vae_decoder"):
            vae_plan = build_causal_vae_3d_engine(
                vae_weights,
                z_dim=self._VAE_Z_DIM,
                base_dim=self._VAE_BASE_DIM,
                dim_mult=self._VAE_DIM_MULT,
                num_res_blocks=self._VAE_NUM_RES_BLOCKS,
                temporal_upsample=self._VAE_TEMPORAL_UPSAMPLE,
                h_lat=h_lat,
                w_lat=w_lat,
                norm_type="l2_channel_norm",
                verbose=verbose,
            )

        # 4. Extract preprocessor weights for C++ runtime
        #    These are the DiT weights that are NOT in the TRT engine graph:
        #    patch embedding, timestep MLP, text projection.
        preprocessor_weights = _serialize_preprocessor_weights(dit_weights)

        out = {
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

        # Must match the dimensions used in build_components() for TRT
        video_height = config.raw.get("video_height", 480)
        video_width = config.raw.get("video_width", 832)
        video_num_frames = config.raw.get("video_num_frames", 17)

        return {
            "diffusion_backend_type": "wan_3d",
            "scheduler": "flow_match_euler",
            "num_inference_steps": config.raw.get("num_inference_steps", 50),
            "guidance_scale": 5.0,
            "flow_shift": 3.0,
            "video_height": video_height,
            "video_width": video_width,
            "video_num_frames": video_num_frames,
            "dit_dim": self._DIT_DIM,
            "dit_num_heads": self._DIT_NUM_HEADS,
            "dit_num_layers": self._DIT_NUM_LAYERS,
            "patch_size": self._PATCH_SIZE,
            "z_dim": self._VAE_Z_DIM,
            "scale_factor_temporal": self._SCALE_FACTOR_TEMPORAL,
            "scale_factor_spatial": self._SCALE_FACTOR_SPATIAL,
            "freq_dim": self._DIT_FREQ_DIM,
            "text_seq_len": self._T5_MAX_SEQ_LEN,
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
            "num_vae_caches": count_vae_caches(
                dim_mult=self._VAE_DIM_MULT,
                num_res_blocks=self._VAE_NUM_RES_BLOCKS,
                temporal_upsample=self._VAE_TEMPORAL_UPSAMPLE,
            ),
            "vae_model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "text_encoder_dim": self._T5_D_MODEL,
        }


def _serialize_preprocessor_weights(dit_weights: dict) -> bytes:
    """Serialize DiT preprocessor weights into a binary format.

    Format: JSON index (length-prefixed) + contiguous float32 data.
    The index maps weight names to {offset, shape} in the data blob.

    Weights stored (all float32, linear weights already transposed [in, out]):
        patch_embedding.weight, patch_embedding.bias
        condition_embedder.time_embedding.0.weight/bias
        condition_embedder.time_embedding.2.weight/bias
        condition_embedder.time_proj.weight/bias
        condition_embedder.text_embedding.weight/bias
    """
    import json
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

        # patch_embedding.weight is Conv3D [out_ch, in_ch, kt, kh, kw].
        # Flatten to [out_ch, patch_dim] then transpose to [patch_dim, out_ch]
        # so C++ can use it directly as matmul: patches @ weight -> hidden.
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
    # Format: [4-byte index length][index JSON][contiguous float32 data]
    result = struct.pack("<I", len(index_json)) + index_json
    for part in data_parts:
        result += part

    return result


plugin = WanT2VPlugin()
