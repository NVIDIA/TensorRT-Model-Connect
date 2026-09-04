# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.1 Text-to-Video family-local build."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from .config import ModelConfig
from .checkpoint_mapper import WeightDict


from .parallel import ParallelConfig

if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _WanT2VModel:
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

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
    ) -> WeightDict:
        """Load weights from all three subdirectories."""
        model_path = Path(model_dir)
        weights = WeightDict()

        # Detect diffusers-format: has model_index.json + subdirs
        if (model_path / "model_index.json").exists():
            weights["_model_format"] = "diffusers"
            weights["_text_encoder_dir"] = str(model_path / "text_encoder")
            weights["_transformer_dir"] = str(model_path / "transformer")
            weights["_vae_dir"] = str(model_path / "vae")
        else:
            raise ValueError(f"Expected diffusers format with model_index.json in {model_dir}")

        scheduler_path = model_path / "scheduler" / "scheduler_config.json"
        if scheduler_path.exists():
            scheduler_config = json.loads(scheduler_path.read_text())
            weights["_scheduler_config"] = scheduler_config
            config.raw["_scheduler_config"] = scheduler_config

        return weights

    def build_components(
        self,
        model_dir: str,
        config: ModelConfig,
        weights: WeightDict,
        *,
        precision: str = "fp32",
        verbose: bool = False,
        parallel_config=None,
    ) -> dict:
        """Build all three component engines."""
        from .timing import timed_trt_compile, timed_weight_loading
        from .t5_encoder_builder import build_t5_encoder_engine, load_t5_weights
        from .standard_dit_builder import build_standard_dit_engine, load_dit_weights
        from .standard_dit_tp_builder import (
            build_standard_dit_engine as build_standard_dit_tp_engine,
        )
        from .standard_dit_cp_builder import (
            build_standard_dit_engine as build_standard_dit_cp_engine,
        )
        from .causal_vae_3d_builder import build_causal_vae_3d_engine, load_vae_weights
        from .parallel import (
            normalize_parallel_config,
            validate_dit_tp,
        )

        build_timing = None
        parallel = normalize_parallel_config(parallel_config)
        parallel.validate()
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

        requested_fp32_layers = frozenset(
            int(layer) for layer in config.raw.get("_fp32_layers", ())
        )
        unsupported_fp32_layers = sorted(
            layer for layer in requested_fp32_layers if layer != self._T5_NUM_LAYERS
        )
        if unsupported_fp32_layers:
            raise ValueError(
                "Wan T2V fp32_layers supports only selector "
                f"{self._T5_NUM_LAYERS}, which selects the complete T5 "
                f"encoder; unsupported selectors: {unsupported_fp32_layers}"
            )
        t5_precision = "fp32" if self._T5_NUM_LAYERS in requested_fp32_layers else precision

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
                precision=t5_precision,
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
                precision=t5_precision,
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
            if parallel.cp_enabled:
                print(
                    f"[wan-t2v] Building shared DiT CP{parallel.cp_size} plan ...",
                    file=sys.stderr,
                )
                dit_plan = build_standard_dit_cp_engine(
                    dit_weights,
                    dim=self._DIT_DIM,
                    num_heads=self._DIT_NUM_HEADS,
                    num_layers=self._DIT_NUM_LAYERS,
                    ffn_dim=self._DIT_FFN_DIM,
                    context_dim=self._DIT_DIM,
                    num_patches=num_patches,
                    text_seq_len=self._T5_MAX_SEQ_LEN,
                    precision=precision,
                    verbose=verbose,
                    parallel_config=parallel,
                )
            elif parallel.enabled:
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
                    precision=precision,
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
            )
        vae_build_options = {
            "z_dim": self._VAE_Z_DIM,
            "base_dim": self._VAE_BASE_DIM,
            "dim_mult": self._VAE_DIM_MULT,
            "num_res_blocks": self._VAE_NUM_RES_BLOCKS,
            "temporal_upsample": self._VAE_TEMPORAL_UPSAMPLE,
            "h_lat": h_lat,
            "w_lat": w_lat,
            "norm_type": "l2_channel_norm",
            "precision": precision,
            "verbose": verbose,
        }
        with timed_trt_compile(build_timing, "vae_decoder"):
            vae_plan = build_causal_vae_3d_engine(
                vae_weights,
                **vae_build_options,
            )
        with timed_trt_compile(build_timing, "vae_decoder_first_frame"):
            vae_first_frame_plan = build_causal_vae_3d_engine(
                vae_weights,
                **vae_build_options,
                first_frame_only=True,
            )

        # 4. Extract preprocessor weights for C++ runtime
        #    These are the DiT weights that are NOT in the TRT engine graph:
        #    patch embedding, timestep MLP, text projection.
        preprocessor_weights = _serialize_preprocessor_weights(dit_weights)

        out = {
            "text_encoders": [("t5", t5_plan)],
            "vae_decoder": vae_plan,
            "vae_decoder_first_frame": vae_first_frame_plan,
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

        scheduler_cfg = config.raw.get("_scheduler_config", {})
        scheduler_class = str(scheduler_cfg.get("_class_name", ""))
        scheduler_name = (
            "unipc_multistep"
            if scheduler_class == "UniPCMultistepScheduler"
            else "flow_match_euler"
        )
        if scheduler_name == "unipc_multistep":
            supported = (
                int(scheduler_cfg.get("solver_order", 2)) == 2
                and str(scheduler_cfg.get("solver_type", "bh2")) == "bh2"
                and str(scheduler_cfg.get("prediction_type", "flow_prediction"))
                == "flow_prediction"
                and bool(scheduler_cfg.get("use_flow_sigmas", True))
            )
            if not supported:
                raise ValueError("Wan T2V supports only order-2 BH2 UniPC flow prediction")

        # Must match the dimensions used in build_components() for TRT
        video_height = config.raw.get("video_height", 480)
        video_width = config.raw.get("video_width", 832)
        video_num_frames = config.raw.get("video_num_frames", 17)

        return {
            "diffusion_backend_type": "wan_3d",
            "scheduler": scheduler_name,
            "num_inference_steps": config.raw.get("num_inference_steps", 50),
            "guidance_scale": 5.0,
            "flow_shift": float(scheduler_cfg.get("flow_shift", scheduler_cfg.get("shift", 1.0))),
            "unipc_lower_order_final": int(bool(scheduler_cfg.get("lower_order_final", True))),
            "use_dynamic_shifting": int(bool(scheduler_cfg.get("use_dynamic_shifting", False))),
            "base_shift": float(scheduler_cfg.get("base_shift", 0.5)),
            "max_shift": float(scheduler_cfg.get("max_shift", 1.15)),
            "base_image_seq_len": int(scheduler_cfg.get("base_image_seq_len", 256)),
            "max_image_seq_len": int(scheduler_cfg.get("max_image_seq_len", 4096)),
            "shift_terminal": float(scheduler_cfg.get("shift_terminal") or 0.0),
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
                -0.7571,
                -0.7089,
                -0.9113,
                0.1075,
                -0.1745,
                0.9653,
                -0.1517,
                1.5508,
                0.4134,
                -0.0715,
                0.5517,
                -0.3632,
                -0.1922,
                -0.9497,
                0.2503,
                -0.2921,
            ],
            "latents_std": [
                2.8184,
                1.4541,
                2.3275,
                2.6558,
                1.2196,
                1.7708,
                2.6052,
                2.0743,
                3.2687,
                2.1526,
                2.8652,
                1.5579,
                1.6382,
                1.1253,
                2.8251,
                1.9160,
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


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one Wan text-to-video bundle."""
    if request.max_sequence_length is not None:
        raise NotImplementedError("wan_t2v does not support max_sequence_length")

    if request.task != "image_generation":
        raise ValueError("wan_t2v supports only task=image_generation")
    if request.precision not in {"fp16", "fp32"}:
        raise ValueError("Wan T2V supports only precision=fp16 or fp32")
    if request.max_batch_size != 1:
        raise NotImplementedError("Wan T2V requires max_batch_size=1")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("Wan T2V does not support quantization")

    height = int(request.image_height or 480)
    width = int(request.image_width or 832)
    frames = int(request.video_num_frames or 17)
    if height % 16 or width % 16:
        raise ValueError("Wan T2V image dimensions must be divisible by 16")
    if (frames - 1) % 4:
        raise ValueError("Wan T2V video_num_frames must equal 4*n+1")
    model_dir = Path(request.model_dir)
    config = ModelConfig(
        model_type="wan",
        raw={
            "video_height": height,
            "video_width": width,
            "video_num_frames": frames,
            "_fp32_layers": request.fp32_layers,
        },
    )
    parallel = ParallelConfig(
        tp_size=int(request.tensor_parallel_size),
        cp_size=int(request.context_parallel_size),
    )
    parallel.validate()
    model = _WanT2VModel()
    weights = model.load_weights(str(model_dir), config)
    components = model.build_components(
        str(model_dir),
        config,
        weights,
        precision=request.precision,
        verbose=request.verbose,
        parallel_config=parallel,
    )

    writer.set_header(family="wan_t2v", task=request.task, backend="trt")
    writer.add_bytes("text_encoder.0.plan", components["text_encoders"][0][1])
    if parallel.enabled:
        plans = components["denoiser_ranks"]
        for rank in range(parallel.tp_size):
            writer.add_bytes(f"denoiser.rank{rank}.plan", plans[rank])
    else:
        writer.add_bytes("denoiser.plan", components["denoiser"])
    writer.add_bytes("vae.plan", components["vae_decoder"])
    writer.add_bytes("vae.first_frame.plan", components["vae_decoder_first_frame"])
    writer.add_bytes("preprocessor.weights", components["preprocessor_weights"])
    writer.add_bytes("tokenizer.json", (model_dir / "tokenizer/tokenizer.json").read_bytes())
    runtime = model.get_diffusion_config(config)
    runtime.update(
        {
            "parallel_mode": parallel.mode,
            "parallel_size": parallel.world_size,
            "num_text_encoders": 1,
            "max_batch_size": {"dit": 1, "text_encoder": 1, "vae": 1},
            "tokenizer_add_special_tokens": False,
            "tokenizer_prefix_ids": [],
            "tokenizer_suffix_ids": [1],
        }
    )
    writer.add_json("runtime.json", runtime)
