"""PixArt-Sigma family plugin for Torch-TRT.

Compiles three components via torch_tensorrt:
  1. T5-XXL text encoder (exported on CPU to save GPU VRAM)
  2. PixArt DiT denoiser (28 transformer blocks)
  3. AutoencoderKL VAE decoder

Bundle sections match the C++ WanPipeline naming convention:
  text_encoder_0_plan, denoiser_plan, vae_decoder_plan

Note: C++ runtime dispatch uses runtime_strategy="diffusion_pixart" which
routes to WanPlugin. The torch-trt DiT engine includes preprocessing
(patch embed, caption projection, timestep embedding) inside the engine,
unlike the raw TRT path where these are serialized as preprocessor_weights.
Full C++ runtime compatibility requires adapting WanPipeline to detect
this difference — deferred to a follow-up.
"""

from __future__ import annotations

import gc
import json
import sys

import torch

from ..config import ModelConfig


class PixArtTorchTrtPlugin:
    name = "pixart"
    runtime_strategy = "diffusion"
    bundle_runtime_strategy = "torchtrt_diffusion"

    _T5_MAX_SEQ_LEN = 120
    _VAE_LATENT_CHANNELS = 4
    _VAE_SCALING_FACTOR = 0.13025
    _VAE_SCALE_FACTOR = 8
    _DIT_IN_CHANNELS = 4
    _DIT_OUT_CHANNELS = 8
    _DIT_PATCH_SIZE = 2
    _IMAGE_HEIGHT = 1024
    _IMAGE_WIDTH = 1024

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        return mt in (
            "pixart", "pixart_sigma", "pixart_alpha",
            "pixartsigmapipeline", "pixartalphapipeline",
        )

    def load_model(self, model_dir, config, max_cache_length, *, dtype=None):
        raise NotImplementedError(
            "PixArt uses build_components(), not load_model()")

    def get_export_args(self, model, config, max_cache_length, *,
                        precision="fp16"):
        raise NotImplementedError(
            "PixArt uses build_components(), not get_export_args()")

    def build_components(
        self,
        model_dir: str,
        config: ModelConfig,
        compile_fn,
        *,
        precision: str = "fp16",
        verbose: bool = False,
    ) -> dict:
        """Build all three component engines via torch_tensorrt.

        Uses CPU-side torch.export for all components to save GPU VRAM.
        The TRT builder only needs workspace memory (~2-4 GB) on GPU.

        Args:
            model_dir: Path to diffusers-format model directory.
            config: ModelConfig (from model_index.json).
            compile_fn: Function(wrapper, export_args, *, trt_inputs=None,
                        verbose=False) -> bytes.
            precision: "fp16" or "bf16".
            verbose: Enable detailed logging.

        Returns:
            dict with keys:
              "sections": list of BundleSection (engine plans)
              "runtime_strategy": str for bundle config
              "diffusion_config": dict of diffusion pipeline params
        """
        from pathlib import Path
        from ..bundle_writer import BundleSection
        from ..strategies.diffusion import (
            T5EncoderWrapper,
            PixArtDiTWrapper,
            VAEDecoderWrapper,
            _TrtSafeAttnProcessor,
        )

        model_path = Path(model_dir)
        compute_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

        # Read component configs
        t5_config = json.loads(
            (model_path / "text_encoder" / "config.json").read_text())
        dit_config = json.loads(
            (model_path / "transformer" / "config.json").read_text())

        t5_d_model = t5_config.get("d_model", 4096)
        dit_num_heads = dit_config.get("num_attention_heads", 16)
        dit_head_dim = dit_config.get("attention_head_dim", 72)
        dit_dim = dit_num_heads * dit_head_dim
        dit_num_layers = dit_config.get("num_layers", 28)

        h_lat = self._IMAGE_HEIGHT // self._VAE_SCALE_FACTOR
        w_lat = self._IMAGE_WIDTH // self._VAE_SCALE_FACTOR

        sections = []

        # --- 1. T5 text encoder (CPU export for VRAM savings) ---
        print("[trtmc build] Loading T5 text encoder (on CPU) ...",
              file=sys.stderr)
        from transformers import T5EncoderModel
        t5_model = T5EncoderModel.from_pretrained(
            str(model_path / "text_encoder"),
            torch_dtype=compute_dtype,
        )
        t5_model.eval()

        t5_wrapper = T5EncoderWrapper(t5_model)
        t5_wrapper.eval()

        t5_cpu_args = (
            torch.zeros(1, self._T5_MAX_SEQ_LEN, dtype=torch.int32),
            torch.ones(1, self._T5_MAX_SEQ_LEN, dtype=torch.int32),
        )
        t5_cuda_args = (
            torch.zeros(1, self._T5_MAX_SEQ_LEN, dtype=torch.int32,
                        device="cuda"),
            torch.ones(1, self._T5_MAX_SEQ_LEN, dtype=torch.int32,
                        device="cuda"),
        )

        print(f"[trtmc build] Compiling T5 encoder "
              f"(d_model={t5_d_model}, seq_len={self._T5_MAX_SEQ_LEN}) ...",
              file=sys.stderr)
        t5_engine = compile_fn(
            t5_wrapper, t5_cpu_args,
            trt_inputs=t5_cuda_args, verbose=verbose)
        sections.append(BundleSection("text_encoder_0_plan", t5_engine))

        del t5_model, t5_wrapper, t5_engine
        gc.collect()

        # --- 2. PixArt DiT denoiser (CPU export) ---
        print(f"[trtmc build] Loading PixArt DiT "
              f"(dim={dit_dim}, layers={dit_num_layers}) ...",
              file=sys.stderr)
        from diffusers import PixArtTransformer2DModel
        dit_model = PixArtTransformer2DModel.from_pretrained(
            str(model_path / "transformer"),
            torch_dtype=compute_dtype,
        )
        dit_model.eval()

        dit_wrapper = PixArtDiTWrapper(
            dit_model,
            in_channels=self._DIT_IN_CHANNELS,
            out_channels=self._DIT_OUT_CHANNELS,
        )
        dit_wrapper.eval()

        # encoder_attention_mask: 1=real token, 0=padding. The wrapper
        # converts this to additive bias (-10000 for padding) and passes
        # it as 3D to the model (bypassing its own 2D->3D conversion).
        # Use a mixed mask for tracing (not all-ones) to prevent folding.
        dit_mask = torch.ones(1, self._T5_MAX_SEQ_LEN, dtype=compute_dtype)
        dit_mask[0, self._T5_MAX_SEQ_LEN // 2:] = 0.0  # half real, half pad
        dit_cpu_args = (
            torch.randn(1, self._DIT_IN_CHANNELS, h_lat, w_lat,
                        dtype=compute_dtype),
            torch.randn(1, self._T5_MAX_SEQ_LEN, t5_d_model,
                        dtype=compute_dtype),
            torch.tensor([1.0], dtype=compute_dtype),
            dit_mask,
        )
        dit_cuda_args = tuple(t.cuda() for t in dit_cpu_args)

        print(f"[trtmc build] Compiling DiT denoiser "
              f"(latent={h_lat}x{w_lat}) ...", file=sys.stderr)
        dit_engine = compile_fn(
            dit_wrapper, dit_cpu_args,
            trt_inputs=dit_cuda_args, verbose=verbose)
        sections.append(BundleSection("denoiser_plan", dit_engine))

        del dit_model, dit_wrapper, dit_engine
        gc.collect()

        # --- 3. VAE decoder (CPU export) ---
        print("[trtmc build] Loading VAE decoder ...", file=sys.stderr)
        from diffusers import AutoencoderKL
        vae_model = AutoencoderKL.from_pretrained(
            str(model_path / "vae"),
            torch_dtype=compute_dtype,
        )
        vae_model.eval()

        vae_model.set_attn_processor(_TrtSafeAttnProcessor())
        vae_wrapper = VAEDecoderWrapper(vae_model, self._VAE_SCALING_FACTOR)
        vae_wrapper.eval()

        vae_cpu_args = (
            torch.randn(1, self._VAE_LATENT_CHANNELS, h_lat, w_lat,
                        dtype=compute_dtype),
        )
        vae_cuda_args = tuple(t.cuda() for t in vae_cpu_args)

        print(f"[trtmc build] Compiling VAE decoder "
              f"(latent={h_lat}x{w_lat} -> "
              f"image={self._IMAGE_HEIGHT}x{self._IMAGE_WIDTH}) ...",
              file=sys.stderr)
        vae_engine = compile_fn(
            vae_wrapper, vae_cpu_args,
            trt_inputs=vae_cuda_args, verbose=verbose,
            workspace_size=4 << 30)
        sections.append(BundleSection("vae_decoder_plan", vae_engine))

        del vae_model, vae_wrapper, vae_engine
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "sections": sections,
            "runtime_strategy": self.bundle_runtime_strategy,
            "diffusion_config": self._get_diffusion_config(
                dit_dim, dit_num_heads, dit_num_layers, t5_d_model,
                h_lat, w_lat),
        }

    def _get_diffusion_config(self, dit_dim, dit_num_heads, dit_num_layers,
                              t5_d_model, h_lat, w_lat):
        """Return diffusion pipeline config fields for the bundle."""
        return {
            "diffusion_backend_type": "wan_3d",
            "scheduler": "dpmsolver_multistep",
            "num_inference_steps": 20,
            "guidance_scale": 4.5,
            "image_height": self._IMAGE_HEIGHT,
            "image_width": self._IMAGE_WIDTH,
            "video_height": self._IMAGE_HEIGHT,
            "video_width": self._IMAGE_WIDTH,
            "video_num_frames": 1,
            "dit_dim": dit_dim,
            "dit_num_heads": dit_num_heads,
            "dit_num_layers": dit_num_layers,
            "patch_size": [1, self._DIT_PATCH_SIZE, self._DIT_PATCH_SIZE],
            "z_dim": self._VAE_LATENT_CHANNELS,
            "scale_factor_temporal": 1,
            "scale_factor_spatial": self._VAE_SCALE_FACTOR,
            "freq_dim": 256,
            "text_seq_len": self._T5_MAX_SEQ_LEN,
            "latents_mean": [],
            "latents_std": [],
            "num_vae_caches": 0,
            "vae_model_id": "",
            "text_encoder_dim": t5_d_model,
            "vae_scaling_factor": self._VAE_SCALING_FACTOR,
            "use_rope": 0,
        }


plugin = PixArtTorchTrtPlugin()
