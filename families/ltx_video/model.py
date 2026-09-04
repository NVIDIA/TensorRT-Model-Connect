# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LTX-Video family plugin.

Builds a native TRTMC bundle for Lightricks LTX-Video:
  - T5-XXL text encoder as a TensorRT network plan
  - LTXVideoTransformer3DModel denoiser as a raw TensorRT plan
  - AutoencoderKLLTXVideo decoder as a raw TensorRT plan

The generated runtime path is C++ + TensorRT only. The denoiser and VAE
engines are constructed directly with the TensorRT network API.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from .checkpoint_mapper import WeightDict
from .config import ModelConfig


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _LTXVideoModel:
    pipeline_classes = ["LTXPipeline"]

    # The linked Lightricks/LTX-Video checkpoint is the 2B text-to-video model.
    _T5_D_MODEL = 4096
    _T5_NUM_HEADS = 64
    _T5_D_KV = 64
    _T5_D_FF = 10240
    _T5_NUM_LAYERS = 24
    _T5_VOCAB_SIZE = 32128
    _T5_MAX_SEQ_LEN = 128

    _DIT_IN_CHANNELS = 128
    _DIT_OUT_CHANNELS = 128
    _DIT_DIM = 2048
    _DIT_NUM_HEADS = 32
    _DIT_NUM_LAYERS = 28

    _VAE_Z_DIM = 128
    _SCALE_FACTOR_TEMPORAL = 8
    _SCALE_FACTOR_SPATIAL = 32
    _PATCH_SIZE = [1, 1, 1]

    # HF model-card examples use 480x704 for this repository.
    _DEFAULT_HEIGHT = 480
    _DEFAULT_WIDTH = 704
    _DEFAULT_NUM_FRAMES = 161
    _DEFAULT_FRAME_RATE = 25
    _DEFAULT_NUM_STEPS = 50
    _DEFAULT_GUIDANCE_SCALE = 3.0
    _DEFAULT_GUIDANCE_RESCALE = 0.0
    _DEFAULT_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"

    def load_weights(self, model_dir: str, config: ModelConfig) -> WeightDict:
        model_path = Path(model_dir)
        model_index_path = model_path / "model_index.json"
        if not model_index_path.exists():
            raise ValueError(f"Expected diffusers format with model_index.json in {model_dir}")

        model_index = json.loads(model_index_path.read_text())
        pipeline_class = str(model_index.get("_class_name", ""))
        if pipeline_class not in self.pipeline_classes:
            raise ValueError(f"Expected LTX pipeline class, got {pipeline_class!r}")

        weights = WeightDict()
        weights["_model_format"] = "diffusers"
        weights["_pipeline_class"] = pipeline_class
        weights["_text_encoder_dir"] = str(model_path / "text_encoder")
        weights["_transformer_dir"] = str(model_path / "transformer")
        weights["_vae_dir"] = str(model_path / "vae")
        weights["_tokenizer_dir"] = str(model_path / "tokenizer")

        for key, rel in (
            ("_text_encoder_config", "text_encoder/config.json"),
            ("_transformer_config", "transformer/config.json"),
            ("_vae_config", "vae/config.json"),
            ("_scheduler_config", "scheduler/scheduler_config.json"),
        ):
            path = model_path / rel
            if not path.is_file():
                raise FileNotFoundError(f"LTX-Video checkpoint file is missing: {path}")
            weights[key] = json.loads(path.read_text())
            config.raw[key] = weights[key]

        config.raw["_pipeline_class"] = pipeline_class
        latents_mean, latents_std = _load_ltx_vae_latent_stats(model_path / "vae")
        weights["_vae_latents_mean"] = latents_mean
        weights["_vae_latents_std"] = latents_std
        config.raw["_vae_latents_mean"] = latents_mean
        config.raw["_vae_latents_std"] = latents_std
        return weights

    def build_components(
        self,
        model_dir: str,
        config: ModelConfig,
        weights: WeightDict,
        *,
        precision: str = "fp32",
        verbose: bool = False,
    ) -> dict:
        del model_dir
        from .t5_encoder_builder import build_t5_encoder_engine, load_t5_weights

        t5_cfg = weights.get("_text_encoder_config", {})
        transformer_cfg = weights.get("_transformer_config", {})

        height = int(config.raw.get("video_height", self._DEFAULT_HEIGHT))
        width = int(config.raw.get("video_width", self._DEFAULT_WIDTH))
        num_frames = int(config.raw.get("video_num_frames", self._DEFAULT_NUM_FRAMES))
        frame_rate = int(config.raw.get("frame_rate", self._DEFAULT_FRAME_RATE))

        latent_frames = (num_frames - 1) // self._SCALE_FACTOR_TEMPORAL + 1
        latent_height = height // self._SCALE_FACTOR_SPATIAL
        latent_width = width // self._SCALE_FACTOR_SPATIAL
        sequence_length = latent_frames * latent_height * latent_width

        t5_d_model = int(t5_cfg.get("d_model", self._T5_D_MODEL))
        t5_num_heads = int(t5_cfg.get("num_heads", self._T5_NUM_HEADS))
        t5_d_kv = int(t5_cfg.get("d_kv", self._T5_D_KV))
        t5_d_ff = int(t5_cfg.get("d_ff", self._T5_D_FF))
        t5_num_layers = int(t5_cfg.get("num_layers", self._T5_NUM_LAYERS))
        t5_vocab_size = int(t5_cfg.get("vocab_size", self._T5_VOCAB_SIZE))
        requested_fp32_layers = frozenset(
            int(layer) for layer in config.raw.get("_fp32_layers", ())
        )
        invalid_fp32_layers = sorted(
            layer for layer in requested_fp32_layers if layer < 0 or layer > t5_num_layers
        )
        if invalid_fp32_layers:
            raise ValueError(
                "LTX-Video fp32_layers contains unknown T5 selectors: "
                f"{invalid_fp32_layers}; expected 0-{t5_num_layers}, where "
                f"{t5_num_layers} selects the complete T5 encoder"
            )
        t5_component_fp32 = precision == "fp16" and t5_num_layers in requested_fp32_layers
        t5_precision = "fp32" if t5_component_fp32 else precision
        t5_fp32_layers = tuple(sorted(requested_fp32_layers - {t5_num_layers}))

        print("[ltx-video] Loading T5 encoder weights ...", file=sys.stderr)
        t5_weights = load_t5_weights(
            weights["_text_encoder_dir"],
            precision=t5_precision,
            fp32_layers=t5_fp32_layers,
            d_model=t5_d_model,
            num_heads=t5_num_heads,
            d_kv=t5_d_kv,
            d_ff=t5_d_ff,
            num_layers=t5_num_layers,
            vocab_size=t5_vocab_size,
        )
        t5_plan = build_t5_encoder_engine(
            t5_weights,
            d_model=t5_d_model,
            num_heads=t5_num_heads,
            d_kv=t5_d_kv,
            d_ff=t5_d_ff,
            num_layers=t5_num_layers,
            vocab_size=t5_vocab_size,
            max_seq_len=self._T5_MAX_SEQ_LEN,
            precision=t5_precision,
            fp32_layers=t5_fp32_layers,
            verbose=verbose,
        )

        if precision not in {"fp16", "fp32"}:
            raise ValueError("LTX-Video supports only precision=fp16 or fp32")
        compute_precision = precision

        print(
            "[ltx-video] Compiling LTX denoiser "
            f"(tokens={sequence_length}, latent={latent_frames}x"
            f"{latent_height}x{latent_width}) ...",
            file=sys.stderr,
        )
        denoiser_plan = _compile_ltx_denoiser_engine(
            weights["_transformer_dir"],
            latent_frames=latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            text_seq_len=self._T5_MAX_SEQ_LEN,
            text_dim=t5_d_model,
            frame_rate=frame_rate,
            precision=compute_precision,
            in_channels=int(transformer_cfg.get("in_channels", self._DIT_IN_CHANNELS)),
            verbose=verbose,
        )

        print("[ltx-video] Compiling LTX VAE decoder ...", file=sys.stderr)
        vae_plan = _compile_ltx_vae_decoder_engine(
            weights["_vae_dir"],
            latent_frames=latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            latent_channels=self._VAE_Z_DIM,
            precision=compute_precision,
            verbose=verbose,
        )

        return {
            "text_encoders": [("t5", t5_plan)],
            "denoiser": denoiser_plan,
            "vae_decoder": vae_plan,
        }

    def get_diffusion_config(self, config: ModelConfig) -> dict:
        transformer_cfg = config.raw.get("_transformer_config", {})
        scheduler_cfg = config.raw.get("_scheduler_config", {})
        vae_cfg = config.raw.get("_vae_config", {})

        height = int(config.raw.get("video_height", self._DEFAULT_HEIGHT))
        width = int(config.raw.get("video_width", self._DEFAULT_WIDTH))
        num_frames = int(config.raw.get("video_num_frames", self._DEFAULT_NUM_FRAMES))

        return {
            "diffusion_backend_type": "ltx_video",
            "scheduler": "flow_match_euler",
            "num_inference_steps": int(
                config.raw.get("num_inference_steps", self._DEFAULT_NUM_STEPS)
            ),
            "guidance_scale": float(config.raw.get("guidance_scale", self._DEFAULT_GUIDANCE_SCALE)),
            "guidance_rescale": float(
                config.raw.get("guidance_rescale", self._DEFAULT_GUIDANCE_RESCALE)
            ),
            "video_height": height,
            "video_width": width,
            "video_num_frames": num_frames,
            "frame_rate": int(config.raw.get("frame_rate", self._DEFAULT_FRAME_RATE)),
            "negative_prompt": str(
                config.raw.get("negative_prompt", self._DEFAULT_NEGATIVE_PROMPT)
            ),
            "z_dim": int(transformer_cfg.get("in_channels", self._DIT_IN_CHANNELS)),
            "dit_dim": int(transformer_cfg.get("num_attention_heads", self._DIT_NUM_HEADS))
            * int(transformer_cfg.get("attention_head_dim", 64)),
            "dit_num_heads": int(transformer_cfg.get("num_attention_heads", self._DIT_NUM_HEADS)),
            "dit_num_layers": int(transformer_cfg.get("num_layers", self._DIT_NUM_LAYERS)),
            "patch_size": self._PATCH_SIZE,
            "scale_factor_temporal": int(
                vae_cfg.get("temporal_compression_ratio", self._SCALE_FACTOR_TEMPORAL)
            ),
            "scale_factor_spatial": int(
                vae_cfg.get("spatial_compression_ratio", self._SCALE_FACTOR_SPATIAL)
            ),
            "text_seq_len": self._T5_MAX_SEQ_LEN,
            "text_encoder_dim": self._T5_D_MODEL,
            "flow_shift": float(scheduler_cfg.get("shift", 1.0)),
            "use_dynamic_shifting": int(bool(scheduler_cfg.get("use_dynamic_shifting", True))),
            "base_shift": float(scheduler_cfg.get("base_shift", 0.95)),
            "max_shift": float(scheduler_cfg.get("max_shift", 2.05)),
            "base_image_seq_len": int(scheduler_cfg.get("base_image_seq_len", 1024)),
            "max_image_seq_len": int(scheduler_cfg.get("max_image_seq_len", 4096)),
            "shift_terminal": float(scheduler_cfg.get("shift_terminal", 0.1)),
            "latents_mean": list(config.raw.get("_vae_latents_mean", [])),
            "latents_std": list(config.raw.get("_vae_latents_std", [])),
            "vae_scaling_factor": float(vae_cfg.get("scaling_factor", 1.0)),
        }


def _load_ltx_vae_latent_stats(vae_dir: Path) -> tuple[list[float], list[float]]:
    from safetensors import safe_open

    for path in sorted(vae_dir.glob("*.safetensors")):
        with safe_open(path, framework="np", device="cpu") as reader:
            keys = set(reader.keys())
            if "latents_mean" not in keys or "latents_std" not in keys:
                continue
            mean = reader.get_tensor("latents_mean").astype("float32").reshape(-1)
            std = reader.get_tensor("latents_std").astype("float32").reshape(-1)
            return mean.tolist(), std.tolist()
    raise ValueError("LTX-Video VAE checkpoint does not contain latent statistics")


def _compile_ltx_denoiser_engine(
    transformer_dir: str,
    *,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    text_seq_len: int,
    text_dim: int,
    frame_rate: int,
    precision: str,
    in_channels: int,
    verbose: bool,
) -> bytes:
    del text_dim
    from .ltx_dit_builder import build_ltx_dit_engine, load_ltx_dit_weights

    weights = load_ltx_dit_weights(transformer_dir, precision=precision)
    return build_ltx_dit_engine(
        weights,
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        text_seq_len=text_seq_len,
        in_channels=in_channels,
        frame_rate=frame_rate,
        precision=precision,
        verbose=verbose,
    )


def _compile_ltx_vae_decoder_engine(
    vae_dir: str,
    *,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    latent_channels: int,
    precision: str,
    verbose: bool,
) -> bytes:
    from .ltx_vae_builder import build_ltx_vae_decoder_engine, load_ltx_vae_weights

    weights = load_ltx_vae_weights(vae_dir, precision=precision)
    return build_ltx_vae_decoder_engine(
        weights,
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        latent_channels=latent_channels,
        precision=precision,
        verbose=verbose,
    )


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one LTX-Video image-generation bundle."""
    if request.max_sequence_length is not None:
        raise NotImplementedError("ltx_video does not support max_sequence_length")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "image_generation":
        raise ValueError("ltx_video supports only task=image_generation")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("LTX-Video requires tensor_parallel_size=1")
    if request.max_batch_size != 1:
        raise NotImplementedError("LTX-Video requires max_batch_size=1")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("LTX-Video does not support quantization")

    model_dir = Path(request.model_dir)
    height = int(request.image_height or _LTXVideoModel._DEFAULT_HEIGHT)
    width = int(request.image_width or _LTXVideoModel._DEFAULT_WIDTH)
    frames = int(request.video_num_frames or _LTXVideoModel._DEFAULT_NUM_FRAMES)
    if height % _LTXVideoModel._SCALE_FACTOR_SPATIAL:
        raise ValueError("LTX-Video image_height must be divisible by 32")
    if width % _LTXVideoModel._SCALE_FACTOR_SPATIAL:
        raise ValueError("LTX-Video image_width must be divisible by 32")
    if (frames - 1) % _LTXVideoModel._SCALE_FACTOR_TEMPORAL:
        raise ValueError("LTX-Video video_num_frames must equal 8*n+1")
    config = ModelConfig(
        model_type="ltx_video",
        raw={
            "_fp32_layers": request.fp32_layers,
            "video_height": height,
            "video_width": width,
            "video_num_frames": frames,
        },
    )
    model = _LTXVideoModel()
    weights = model.load_weights(str(model_dir), config)
    components = model.build_components(
        str(model_dir),
        config,
        weights,
        precision=request.precision,
        verbose=request.verbose,
    )
    text_encoders = components["text_encoders"]
    if len(text_encoders) != 1:
        raise RuntimeError("LTX-Video must produce exactly one text encoder")

    writer.set_header(family="ltx_video", task=request.task, backend="trt")
    writer.add_bytes("text_encoder.0.plan", text_encoders[0][1])
    writer.add_bytes("denoiser.plan", components["denoiser"])
    writer.add_bytes("vae.plan", components["vae_decoder"])
    tokenizer_path = model_dir / "tokenizer/tokenizer.json"
    writer.add_bytes("tokenizer.json", tokenizer_path.read_bytes())

    runtime = model.get_diffusion_config(config)
    runtime.update(
        {
            "max_batch_size": {
                "dit": 1,
                "text_encoder": 1,
                "vae": 1,
            },
            "tokenizer_add_special_tokens": False,
            "tokenizer_prefix_ids": [],
            "tokenizer_suffix_ids": [],
        }
    )
    writer.add_json("runtime.json", runtime)
