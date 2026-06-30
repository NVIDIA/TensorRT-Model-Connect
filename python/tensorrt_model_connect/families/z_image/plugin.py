"""Z-Image-Turbo family plugin.

Tongyi-MAI/Z-Image-Turbo: text-to-image diffusion model from Alibaba.
Architecture: Qwen3 text encoder + ZImage DiT (unified attention) + AutoencoderKL VAE.
Uses FlowMatchEulerDiscreteScheduler with dynamic mu shifting.

Components:
  text_encoder: Qwen3Model (36 layers, hidden=2560, uses hidden_states[-2])
  transformer: ZImageTransformer2DModel (30 layers, dim=3840, 30 heads, SwiGLU FFN,
               unified single-stream attention, tanh-gated AdaLN, 3-axis RoPE)
  vae: AutoencoderKL (FLUX-style, 16 latent channels, shift_factor=0.1159, scaling_factor=0.3611)
  scheduler: FlowMatchEulerDiscreteScheduler (dynamic mu based on image_seq_len)

Key differences from FLUX:
  - Latent size: h_lat = 2 * (H // (vae_scale * 2)), same for width.
    For 1024x1024: vae_scale=8, so h_lat = 2*(1024//16) = 128.
  - Patchify: 2x2 patches on h_lat x w_lat -> 4096 patches (for 1024x1024).
  - Timestep: pipeline does (1000 - raw_t) / 1000, then transformer multiplies by t_scale=1000.
  - Noise pred negation: pipeline negates transformer output before scheduler step.
  - AdaLN: per-layer uses Linear only (no SiLU), FinalLayer uses SiLU+Linear.
  - Gates use tanh activation.
  - norm2 is POST-norm (on attention/FFN output), not pre-norm.
  - FinalLayer uses LayerNorm (not RMSNorm).
"""

from __future__ import annotations

import sys

from .config import ModelConfig
from .weights import WeightDict


class ZImagePlugin:
    name = "z_image"
    runtime_strategy = "diffusion_zimage"
    pipeline_classes = ["ZImagePipeline"]

    # Z-Image Turbo architecture params
    _DIT_DIM = 3840
    _DIT_NUM_HEADS = 30
    _DIT_NUM_LAYERS = 30
    _DIT_NUM_REFINER_LAYERS = 2
    _DIT_FFN_DIM = 10240  # int(3840 / 3 * 8)
    _DIT_HEAD_DIM = 128
    _ADALN_EMBED_DIM = 256

    # Qwen3 text encoder params
    _TEXT_HIDDEN = 2560
    _TEXT_NUM_LAYERS = 36
    _TEXT_NUM_HEADS = 32
    _TEXT_NUM_KV_HEADS = 8
    _TEXT_HEAD_DIM = 128
    _TEXT_INTERMEDIATE = 9728
    _TEXT_VOCAB = 151936
    _TEXT_ROPE_THETA = 1000000.0
    _TEXT_MAX_SEQ_LEN = 512
    _TEXT_OUTPUT_LAYER = -2

    _VAE_LATENT_CHANNELS = 16
    _VAE_SCALING_FACTOR = 0.3611
    _VAE_SHIFT_FACTOR = 0.1159
    _VAE_SCALE_FACTOR = 8  # vae_scale_factor = 2**(len(block_out_channels)-1) = 2**3 = 8

    _PATCH_SIZE = [1, 2, 2]
    _ROPE_THETA = 256.0
    _AXES_DIMS = [32, 48, 48]
    _AXES_LENS = [1536, 512, 512]
    _T_SCALE = 1000.0

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        return mt in ("z_image", "zimage", "z-image", "zimagepipeline")

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
    ) -> WeightDict:
        from pathlib import Path

        model_path = Path(model_dir)
        weights = WeightDict()

        if (model_path / "model_index.json").exists():
            weights["_model_format"] = "diffusers"
            weights["_text_encoder_dir"] = str(model_path / "text_encoder")
            weights["_transformer_dir"] = str(model_path / "transformer")
            weights["_vae_dir"] = str(model_path / "vae")
            weights["_tokenizer_dir"] = str(model_path / "tokenizer")
            weights["_model_dir"] = str(model_path)
        else:
            raise ValueError(f"Expected diffusers format with model_index.json in {model_dir}")

        return weights

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
    ) -> bytes:
        raise NotImplementedError("Z-Image uses build_components(), not build_engine()")

    def build_components(
        self,
        model_dir: str,
        config: ModelConfig,
        weights: WeightDict,
        *,
        precision: str = "fp32",
        verbose: bool = False,
        parallel_config=None,
        max_batch_size: int = 1,
        **_kwargs,
    ) -> dict:
        """Build REAL TRT engines for all Z-Image components."""
        from ...build_timing import timed_trt_compile, timed_weight_loading
        from .model.components.text_encoder import (
            build_qwen3_encoder_engine,
            load_qwen3_encoder_weights,
        )
        from .model.components.dit import build_z_image_dit_engine, load_z_image_dit_weights
        from .model.parallel import (
            build_z_image_dit_engine as build_z_image_dit_tp_engine,
        )
        from .model.components.vae import build_vae_2d_decoder_engine
        from ...parallel_config import (
            normalize_parallel_config,
            require_tensorrt_11_for_tensor_parallel,
            validate_dit_tp,
        )

        build_timing = _kwargs.get("build_timing")
        parallel = normalize_parallel_config(parallel_config)
        # TP + batch>1 is out of scope for this PR series.
        if max_batch_size > 1 and parallel.enabled:
            raise NotImplementedError(
                "Z-Image tensor-parallel + max_batch_size > 1 is not supported "
                "in this release; build with either TP=1 or max_batch_size=1."
            )
        require_tensorrt_11_for_tensor_parallel(parallel, feature="Z-Image tensor-parallel builds")

        # Per-component batch policy (Decisions C / E).
        dit_mbs = int(max_batch_size)
        dit_opt = min(dit_mbs, 4)
        te_mbs = min(dit_mbs * 2, 8)
        te_opt = min(te_mbs, 4)
        vae_mbs = 1
        if parallel.enabled:
            validate_dit_tp(
                dim=self._DIT_DIM,
                num_heads=self._DIT_NUM_HEADS,
                ffn_dim=self._DIT_FFN_DIM,
                parallel=parallel.for_rank(0),
                feature="Z-Image tensor parallel",
            )

        text_encoder_dir = weights["_text_encoder_dir"]
        transformer_dir = weights["_transformer_dir"]
        vae_dir = weights["_vae_dir"]

        image_height = config.raw.get("image_height", 1024)
        image_width = config.raw.get("image_width", 1024)

        # CRITICAL: HF prepare_latents does:
        #   height = 2 * (int(height) // (vae_scale_factor * 2))
        # vae_scale_factor = 8, so h_lat = 2 * (1024 // 16) = 128
        h_lat = 2 * (image_height // (self._VAE_SCALE_FACTOR * 2))
        w_lat = 2 * (image_width // (self._VAE_SCALE_FACTOR * 2))

        ph, pw = self._PATCH_SIZE[1], self._PATCH_SIZE[2]
        num_patches = (h_lat // ph) * (w_lat // pw)

        print(
            f"[z-image] Latent size: {h_lat}x{w_lat}, "
            f"patches: {num_patches} ({h_lat // ph}x{w_lat // pw})",
            file=sys.stderr,
        )

        # 1. Qwen3 text encoder
        print("[z-image] Loading Qwen3 text encoder weights ...", file=sys.stderr)
        with timed_weight_loading(build_timing, "qwen3_encoder"):
            te_weights = load_qwen3_encoder_weights(
                text_encoder_dir,
                hidden_size=self._TEXT_HIDDEN,
                num_layers=self._TEXT_NUM_LAYERS,
                num_heads=self._TEXT_NUM_HEADS,
                num_kv_heads=self._TEXT_NUM_KV_HEADS,
                intermediate_size=self._TEXT_INTERMEDIATE,
                vocab_size=self._TEXT_VOCAB,
            )
        with timed_trt_compile(build_timing, "qwen3_encoder"):
            te_plan = build_qwen3_encoder_engine(
                te_weights,
                hidden_size=self._TEXT_HIDDEN,
                num_layers=self._TEXT_NUM_LAYERS,
                num_heads=self._TEXT_NUM_HEADS,
                num_kv_heads=self._TEXT_NUM_KV_HEADS,
                head_dim=self._TEXT_HEAD_DIM,
                intermediate_size=self._TEXT_INTERMEDIATE,
                vocab_size=self._TEXT_VOCAB,
                max_seq_len=self._TEXT_MAX_SEQ_LEN,
                rope_theta=self._TEXT_ROPE_THETA,
                output_layer=self._TEXT_OUTPUT_LAYER,
                verbose=verbose,
                max_batch_size=te_mbs,
                opt_batch_size=te_opt,
            )

        # 2. Z-Image DiT denoiser
        print("[z-image] Loading Z-Image DiT weights ...", file=sys.stderr)
        with timed_weight_loading(build_timing, "z_image_dit"):
            dit_weights = load_z_image_dit_weights(
                transformer_dir,
                dim=self._DIT_DIM,
                num_heads=self._DIT_NUM_HEADS,
                num_layers=self._DIT_NUM_LAYERS,
                num_refiner_layers=self._DIT_NUM_REFINER_LAYERS,
                ffn_dim=self._DIT_FFN_DIM,
            )
        dit_plan = None
        dit_rank_plans = None
        with timed_trt_compile(build_timing, "z_image_dit"):
            if parallel.enabled:
                dit_rank_plans = {}
                for rank in range(parallel.tp_size):
                    print(
                        f"[z-image] Building DiT TP rank {rank}/{parallel.tp_size} ...",
                        file=sys.stderr,
                    )
                    dit_rank_plans[rank] = build_z_image_dit_tp_engine(
                        dit_weights,
                        dim=self._DIT_DIM,
                        num_heads=self._DIT_NUM_HEADS,
                        num_layers=self._DIT_NUM_LAYERS,
                        num_refiner_layers=self._DIT_NUM_REFINER_LAYERS,
                        ffn_dim=self._DIT_FFN_DIM,
                        num_patches=num_patches,
                        text_seq_len=self._TEXT_MAX_SEQ_LEN,
                        head_dim=self._DIT_HEAD_DIM,
                        adaln_embed_dim=self._ADALN_EMBED_DIM,
                        verbose=verbose,
                        parallel_config=parallel.for_rank(rank),
                    )
            else:
                dit_plan = build_z_image_dit_engine(
                    dit_weights,
                    dim=self._DIT_DIM,
                    num_heads=self._DIT_NUM_HEADS,
                    num_layers=self._DIT_NUM_LAYERS,
                    num_refiner_layers=self._DIT_NUM_REFINER_LAYERS,
                    ffn_dim=self._DIT_FFN_DIM,
                    num_patches=num_patches,
                    text_seq_len=self._TEXT_MAX_SEQ_LEN,
                    head_dim=self._DIT_HEAD_DIM,
                    adaln_embed_dim=self._ADALN_EMBED_DIM,
                    verbose=verbose,
                    max_batch_size=dit_mbs,
                    opt_batch_size=dit_opt,
                )

        # 3. VAE decoder
        print("[z-image] Building VAE decoder engine ...", file=sys.stderr)
        vae_plan = build_vae_2d_decoder_engine(
            vae_dir,
            latent_channels=self._VAE_LATENT_CHANNELS,
            h_lat=h_lat,
            w_lat=w_lat,
            scaling_factor=self._VAE_SCALING_FACTOR,
            shift_factor=self._VAE_SHIFT_FACTOR,
            verbose=verbose,
            build_timing=build_timing,
            timing_component="vae_decoder",
        )

        # 4. Serialize preprocessor weights for C++ runtime
        preprocessor_weights = _serialize_preprocessor_weights(dit_weights)

        out = {
            "text_encoders": [("qwen3", te_plan)],
            "vae_decoder": vae_plan,
            "preprocessor_weights": preprocessor_weights,
        }
        if parallel.enabled:
            out["denoiser_ranks"] = dit_rank_plans or {}
        else:
            out["denoiser"] = dit_plan
        if max_batch_size > 1:
            out["max_batch_size_envelope"] = {
                "dit": dit_mbs,
                "text_encoder": te_mbs,
                "vae": vae_mbs,
            }
        return out

    def diffusion_bundle_sections(
        self, components: dict, *, parallel_config=None
    ) -> list[tuple[str, bytes]]:
        from ...parallel_config import normalize_parallel_config, rank_denoiser_section

        parallel = normalize_parallel_config(parallel_config)
        sections: list[tuple[str, bytes]] = []
        for index, (_name, plan) in enumerate(components["text_encoders"]):
            sections.append((f"text_encoder_{index}_plan", plan))
        if parallel.enabled:
            denoiser_rank_plans = components["denoiser_ranks"]
            for rank in range(parallel.tp_size):
                plan = denoiser_rank_plans.get(rank)
                if plan is None:
                    plan = denoiser_rank_plans.get(str(rank))
                if plan is None:
                    raise ValueError(f"Missing Z-Image tensor-parallel denoiser rank {rank}")
                sections.append((rank_denoiser_section(rank), plan))
        else:
            sections.append(("denoiser_plan", components["denoiser"]))
        sections.append(("vae_decoder_plan", components["vae_decoder"]))
        sections.append(("preprocessor_weights", components["preprocessor_weights"]))
        return sections

    def diffusion_bundle_config(self, config: ModelConfig, *, components: dict) -> dict:
        cfg = self.get_diffusion_config(config)
        cfg["num_text_encoders"] = len(components["text_encoders"])
        return cfg

    def diffusion_tokenizer_add_special_tokens(
        self,
        model_dir_path,
        *,
        detect_tokenizer_add_special_tokens,
    ) -> bool:
        from pathlib import Path

        model_dir = Path(model_dir_path)
        for tok_subdir in ("tokenizer_2", "tokenizer"):
            tok_dir = model_dir / tok_subdir
            if tok_dir.is_dir():
                return bool(detect_tokenizer_add_special_tokens(tok_dir))
        return bool(detect_tokenizer_add_special_tokens(model_dir))

    def diffusion_tokenizer_bundle_sections(
        self,
        model_dir_path,
        *,
        ensure_tokenizer_json,
    ) -> list[tuple[str, bytes]]:
        from pathlib import Path

        model_dir = Path(model_dir_path)
        token_filenames = (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "spiece.model",
            "tokenizer.model",
        )
        sections: list[tuple[str, bytes]] = []
        embedded: set[str] = set()
        for tok_subdir in ("tokenizer_2", "tokenizer"):
            tokenizer_dir = model_dir / tok_subdir
            if not tokenizer_dir.is_dir():
                continue
            if not (tokenizer_dir / "tokenizer.json").exists():
                ensure_tokenizer_json(tokenizer_dir)
            for filename in token_filenames:
                if filename in embedded:
                    continue
                file_path = tokenizer_dir / filename
                if file_path.exists():
                    sections.append((filename, file_path.read_bytes()))
                    embedded.add(filename)

        clip_file_map = {
            "tokenizer.json": "clip_tokenizer.json",
            "vocab.json": "clip_vocab.json",
            "merges.txt": "clip_merges.txt",
            "tokenizer_config.json": "clip_tokenizer_config.json",
            "special_tokens_map.json": "clip_special_tokens_map.json",
        }
        clip_tokenizer_dir = model_dir / "tokenizer"
        if clip_tokenizer_dir.is_dir() and (model_dir / "tokenizer_2").is_dir():
            for src_name, dst_name in clip_file_map.items():
                file_path = clip_tokenizer_dir / src_name
                if file_path.exists():
                    sections.append((dst_name, file_path.read_bytes()))
        return sections

    def get_diffusion_config(self, config: ModelConfig) -> dict:
        image_height = config.raw.get("image_height", 1024)
        image_width = config.raw.get("image_width", 1024)

        # HF scheduler config has shift=3.0 and use_dynamic_shifting=False.
        # The pipeline calculates mu and passes it to set_timesteps(mu=mu),
        # but since use_dynamic_shifting=False, the mu is IGNORED and shift=3.0
        # is always used. The pipeline also sets scheduler.sigma_min = 0.0.

        return {
            "diffusion_backend_type": "z_image_2d",
            "scheduler": "flow_match_euler",
            "num_inference_steps": 9,
            "guidance_scale": 0.0,
            "flow_shift": 3.0,  # HF scheduler shift=3.0 (use_dynamic_shifting=False)
            "video_height": image_height,
            "video_width": image_width,
            "video_num_frames": 1,
            "dit_dim": self._DIT_DIM,
            "dit_num_heads": self._DIT_NUM_HEADS,
            "dit_num_layers": self._DIT_NUM_LAYERS,
            "patch_size": self._PATCH_SIZE,
            "z_dim": self._VAE_LATENT_CHANNELS,
            "scale_factor_temporal": 1,
            "scale_factor_spatial": self._VAE_SCALE_FACTOR,  # Just vae_scale_factor=8, NOT *2
            "freq_dim": self._ADALN_EMBED_DIM,
            "text_seq_len": self._TEXT_MAX_SEQ_LEN,
            "latents_mean": [],
            "latents_std": [],
            "num_vae_caches": 0,
            "vae_model_id": "Tongyi-MAI/Z-Image-Turbo",
            "text_encoder_dim": self._TEXT_HIDDEN,
            # Z-Image-specific flags
        }


def _serialize_preprocessor_weights(dit_weights: WeightDict) -> bytes:
    """Serialize Z-Image preprocessor weights for C++ runtime."""
    import json
    import struct
    import numpy as np

    keys_map = {
        "t_embedder.mlp.0.weight": "t_emb.0.weight",
        "t_embedder.mlp.0.bias": "t_emb.0.bias",
        "t_embedder.mlp.2.weight": "t_emb.2.weight",
        "t_embedder.mlp.2.bias": "t_emb.2.bias",
        "cap_embedder.norm.weight": "cap_norm.weight",
        "cap_embedder.proj.weight": "cap_proj.weight",
        "cap_embedder.proj.bias": "cap_proj.bias",
        "x_embedder.weight": "x_embedder.weight",
        "x_embedder.bias": "x_embedder.bias",
        "cap_pad_token": "cap_pad_token",
        "x_pad_token": "x_pad_token",
    }

    index = {}
    data_parts = []
    offset = 0

    for canonical_key, dit_key in keys_map.items():
        if dit_key not in dit_weights:
            continue
        w = dit_weights[dit_key]
        if not isinstance(w, np.ndarray):
            w = np.array(w, dtype=np.float32)
        w = np.ascontiguousarray(w.astype(np.float32))
        nbytes = w.nbytes
        index[canonical_key] = {"offset": offset, "shape": list(w.shape)}
        data_parts.append(w.tobytes())
        offset += nbytes

    index_json = json.dumps(index).encode("utf-8")
    result = struct.pack("<I", len(index_json)) + index_json
    for part in data_parts:
        result += part

    return result


plugin = ZImagePlugin()
