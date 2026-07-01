"""PixArt-Sigma / PixArt-Alpha family plugin.

Supports PixArt-Sigma and PixArt-Alpha text-to-image diffusion models.
Architecture: T5-XXL text encoder + PixArt DiT (ada_norm_single) + AutoencoderKL VAE.

Components:
  text_encoder: T5EncoderModel (T5-XXL, d_model=4096, 24 layers)
  transformer: PixArtTransformer2DModel (28 layers, dim=1152, 16 heads,
               ada_norm_single with per-block scale_shift_table,
               fixed 2D sinusoidal position embeddings — no RoPE)
  vae: AutoencoderKL (4 latent channels, block_out_channels=[128,256,512,512])
  scheduler: DPMSolverMultistepScheduler (dpmsolver++, epsilon prediction)

Key differences from Wan/FLUX:
  - No RoPE: uses fixed 2D sinusoidal position embeddings (buffer, not learned)
  - ada_norm_single: per-block scale_shift_table[6, dim] + global timestep
  - 4-channel latent VAE (vs 16-channel for FLUX/Z-Image)
  - Cross-attention has no norm and no gate (plain residual add)
  - caption_projection: T5 4096 -> 1152 (Linear + GELU + Linear)
  - DPM-Solver++ scheduler (not flow matching)
"""

from __future__ import annotations

import sys

from .config import ModelConfig
from .weights import WeightDict


class PixArtPlugin:
    name = "pixart"
    runtime_strategy = "diffusion_pixart"
    pipeline_classes = [
        "PixArtSigmaPipeline",
        "PixArtAlphaPipeline",
        # Older diffusers may use Transformer2DModel instead of
        # PixArtTransformer2DModel — the pipeline class still matches.
    ]

    # T5-XXL text encoder params
    _T5_D_MODEL = 4096
    _T5_NUM_HEADS = 64
    _T5_D_KV = 64
    _T5_D_FF = 10240
    _T5_NUM_LAYERS = 24
    _T5_VOCAB_SIZE = 32128
    _T5_MAX_SEQ_LEN = 120  # PixArt default max_sequence_length

    # PixArt DiT params (XL-2 configuration)
    _DIT_DIM = 1152  # 16 heads * 72 head_dim
    _DIT_NUM_HEADS = 16
    _DIT_HEAD_DIM = 72
    _DIT_NUM_LAYERS = 28
    _DIT_FFN_DIM = 4608  # 4 * 1152
    _DIT_CAPTION_CHANNELS = 4096  # T5 output dim before projection
    _DIT_CROSS_ATTN_DIM = 1152  # after caption_projection
    _DIT_PATCH_SIZE = 2
    _DIT_IN_CHANNELS = 4
    _DIT_OUT_CHANNELS = 8

    # VAE params
    _VAE_LATENT_CHANNELS = 4
    _VAE_SCALING_FACTOR = 0.13025
    _VAE_SCALE_FACTOR = 8  # 2^(len(block_out_channels)-1) = 2^3

    # Image dimensions
    _IMAGE_HEIGHT = 1024
    _IMAGE_WIDTH = 1024

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        return mt in ("pixart", "pixart_sigma", "pixart_alpha", "pixartsigma", "pixartalpha")

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
    ) -> WeightDict:
        """Load weight paths from diffusers-format directory."""
        from pathlib import Path

        model_path = Path(model_dir)
        weights = WeightDict()

        if (model_path / "model_index.json").exists():
            weights["_model_format"] = "diffusers"
            weights["_text_encoder_dir"] = str(model_path / "text_encoder")
            weights["_transformer_dir"] = str(model_path / "transformer")
            weights["_vae_dir"] = str(model_path / "vae")
        else:
            raise ValueError(f"Expected diffusers format with model_index.json in {model_dir}")

        # Read transformer config for exact architecture params
        import json

        transformer_config_path = model_path / "transformer" / "config.json"
        if transformer_config_path.exists():
            tc = json.loads(transformer_config_path.read_text())
            weights["_transformer_config"] = tc

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
        raise NotImplementedError("PixArt uses build_components(), not build_engine()")

    def build_components(
        self,
        model_dir: str,
        config: ModelConfig,
        weights: WeightDict,
        *,
        precision: str = "fp32",
        verbose: bool = False,
        parallel_config=None,
        **_kwargs,
    ) -> dict:
        """Build all three component engines."""
        from ...build_timing import timed_trt_compile, timed_weight_loading
        from .model.components.text_encoder import build_t5_encoder_engine, load_t5_weights
        from .model.model import build_standard_dit_engine
        from .model.parallel import (
            build_standard_dit_engine as build_standard_dit_tp_engine,
        )
        from .model.components.vae import build_vae_2d_decoder_engine
        from ...parallel_config import (
            normalize_parallel_config,
            require_tensorrt_11_for_tensor_parallel,
        )
        import json
        from pathlib import Path

        build_timing = _kwargs.get("build_timing")
        parallel = normalize_parallel_config(parallel_config)
        require_tensorrt_11_for_tensor_parallel(parallel, feature="PixArt tensor-parallel builds")

        text_encoder_dir = weights["_text_encoder_dir"]
        transformer_dir = weights["_transformer_dir"]
        vae_dir = weights["_vae_dir"]

        # Read transformer config for exact params
        tc = weights.get("_transformer_config", {})
        num_heads = tc.get("num_attention_heads", self._DIT_NUM_HEADS)
        head_dim = tc.get("attention_head_dim", self._DIT_HEAD_DIM)
        dit_dim = num_heads * head_dim
        num_layers = tc.get("num_layers", self._DIT_NUM_LAYERS)
        tc.get("in_channels", self._DIT_IN_CHANNELS)
        patch_size = tc.get("patch_size", self._DIT_PATCH_SIZE)
        cross_attn_dim = tc.get("cross_attention_dim", dit_dim)
        ffn_dim = dit_dim * 4  # PixArt uses 4x multiplier

        # Read T5 config from text encoder directory
        t5_config_path = Path(text_encoder_dir) / "config.json"
        t5_cfg = {}
        if t5_config_path.exists():
            t5_cfg = json.loads(t5_config_path.read_text())

        t5_d_model = t5_cfg.get("d_model", self._T5_D_MODEL)
        t5_num_heads = t5_cfg.get("num_heads", self._T5_NUM_HEADS)
        t5_d_kv = t5_cfg.get("d_kv", self._T5_D_KV)
        t5_d_ff = t5_cfg.get("d_ff", self._T5_D_FF)
        t5_num_layers = t5_cfg.get("num_layers", self._T5_NUM_LAYERS)
        t5_vocab_size = t5_cfg.get("vocab_size", self._T5_VOCAB_SIZE)

        # Image and latent dimensions
        img_h = config.raw.get("image_height", self._IMAGE_HEIGHT)
        img_w = config.raw.get("image_width", self._IMAGE_WIDTH)
        h_lat = img_h // self._VAE_SCALE_FACTOR
        w_lat = img_w // self._VAE_SCALE_FACTOR
        num_patches = (h_lat // patch_size) * (w_lat // patch_size)

        print(
            f"[pixart] DiT: dim={dit_dim}, heads={num_heads}, "
            f"layers={num_layers}, patches={num_patches} "
            f"({h_lat // patch_size}x{w_lat // patch_size})",
            file=sys.stderr,
        )

        # 1. T5 text encoder
        print("[pixart] Loading T5 encoder weights ...", file=sys.stderr)
        with timed_weight_loading(build_timing, "t5_encoder"):
            t5_weights = load_t5_weights(
                text_encoder_dir,
                d_model=t5_d_model,
                num_heads=t5_num_heads,
                d_kv=t5_d_kv,
                d_ff=t5_d_ff,
                num_layers=t5_num_layers,
                vocab_size=t5_vocab_size,
                precision=precision,
            )
        with timed_trt_compile(build_timing, "t5_encoder"):
            t5_plan = build_t5_encoder_engine(
                t5_weights,
                d_model=t5_d_model,
                num_heads=t5_num_heads,
                d_kv=t5_d_kv,
                d_ff=t5_d_ff,
                num_layers=t5_num_layers,
                vocab_size=t5_vocab_size,
                max_seq_len=self._T5_MAX_SEQ_LEN,
                verbose=verbose,
            )

        # 2. DiT denoiser (no RoPE — uses fixed sinusoidal position embeddings)
        print("[pixart] Loading PixArt DiT weights ...", file=sys.stderr)
        with timed_weight_loading(build_timing, "pixart_dit"):
            dit_weights = _load_pixart_dit_weights(
                transformer_dir,
                dim=dit_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                ffn_dim=ffn_dim,
                cross_attn_dim=cross_attn_dim,
            )

        dit_plan = None
        dit_rank_plans = None
        with timed_trt_compile(build_timing, "pixart_dit"):
            if parallel.enabled:
                dit_rank_plans = {}
                for rank in range(parallel.tp_size):
                    print(
                        f"[pixart] Building PixArt DiT TP rank {rank}/{parallel.tp_size} ...",
                        file=sys.stderr,
                    )
                    dit_rank_plans[rank] = build_standard_dit_tp_engine(
                        dit_weights,
                        dim=dit_dim,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        ffn_dim=ffn_dim,
                        context_dim=cross_attn_dim,
                        num_patches=num_patches,
                        text_seq_len=self._T5_MAX_SEQ_LEN,
                        qk_norm=False,
                        cross_attn_norm=False,
                        ffn_activation="gelu_approximate",
                        use_rope=False,
                        verbose=verbose,
                        parallel_config=parallel.for_rank(rank),
                    )
            else:
                dit_plan = build_standard_dit_engine(
                    dit_weights,
                    dim=dit_dim,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    ffn_dim=ffn_dim,
                    context_dim=cross_attn_dim,
                    num_patches=num_patches,
                    text_seq_len=self._T5_MAX_SEQ_LEN,
                    verbose=verbose,
                )

        # 3. VAE decoder
        print("[pixart] Building VAE decoder engine ...", file=sys.stderr)
        vae_plan = build_vae_2d_decoder_engine(
            vae_dir,
            latent_channels=self._VAE_LATENT_CHANNELS,
            h_lat=h_lat,
            w_lat=w_lat,
            scaling_factor=self._VAE_SCALING_FACTOR,
            shift_factor=0.0,
            verbose=verbose,
            build_timing=build_timing,
            timing_component="vae_decoder",
        )

        # 4. Serialize preprocessor weights
        preprocessor_weights = _serialize_preprocessor_weights(dit_weights, t5_d_model, dit_dim)

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
                    raise ValueError(f"Missing PixArt tensor-parallel denoiser rank {rank}")
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
        """Return diffusion pipeline configuration."""
        tc = config.raw.get("_transformer_config", {})

        img_h = config.raw.get("image_height", self._IMAGE_HEIGHT)
        img_w = config.raw.get("image_width", self._IMAGE_WIDTH)

        num_heads = tc.get("num_attention_heads", self._DIT_NUM_HEADS)
        head_dim = tc.get("attention_head_dim", self._DIT_HEAD_DIM)
        dit_dim = num_heads * head_dim
        num_layers = tc.get("num_layers", self._DIT_NUM_LAYERS)

        return {
            "diffusion_backend_type": "wan_3d",
            "scheduler": "dpmsolver_multistep",
            "num_inference_steps": 20,
            "guidance_scale": 4.5,
            "image_height": img_h,
            "image_width": img_w,
            "video_height": img_h,
            "video_width": img_w,
            "video_num_frames": 1,
            "dit_dim": dit_dim,
            "dit_num_heads": num_heads,
            "dit_num_layers": num_layers,
            "patch_size": [1, self._DIT_PATCH_SIZE, self._DIT_PATCH_SIZE],
            "z_dim": self._VAE_LATENT_CHANNELS,
            "scale_factor_temporal": 1,
            "scale_factor_spatial": self._VAE_SCALE_FACTOR,
            "freq_dim": 256,  # Sinusoidal timestep embedding dim
            "text_seq_len": self._T5_MAX_SEQ_LEN,
            # Empty: DDIM models skip Wan-style denormalization.
            # VAE scaling (1/scaling_factor) is handled in the 2D VAE decode.
            "latents_mean": [],
            "latents_std": [],
            "num_vae_caches": 0,
            "vae_model_id": "",
            "text_encoder_dim": self._T5_D_MODEL,
            "vae_scaling_factor": self._VAE_SCALING_FACTOR,
            "use_rope": 0,  # PixArt uses fixed sinusoidal pos embed
        }


def _load_pixart_dit_weights(
    model_dir: str,
    *,
    dim: int,
    num_heads: int,
    num_layers: int,
    ffn_dim: int,
    cross_attn_dim: int,
) -> WeightDict:
    """Load PixArt DiT weights and map to standard naming.

    PixArt uses 'transformer_blocks.{i}' prefix while the standard DiT
    builder expects 'blocks.{i}'. This function loads weights with the
    PixArt naming and maps them to the standard convention.
    """
    from pathlib import Path
    from .weights import WeightDict, _open_safetensors, _load_tensor, _has_tensor

    import numpy as np

    model_path = Path(model_dir)
    readers = _open_safetensors(model_path)
    weights = WeightDict()

    def _t(name: str) -> np.ndarray:
        """Load and transpose [out, in] -> [in, out]."""
        w = _load_tensor(readers, name)
        return np.ascontiguousarray(w.T, dtype=np.float32)

    def _f(name: str) -> np.ndarray:
        """Load flat (1D) weight."""
        return _load_tensor(readers, name).astype(np.float32)

    def _maybe_f(name: str) -> np.ndarray | None:
        if _has_tensor(readers, name):
            return _f(name)
        return None

    for i in range(num_layers):
        # PixArt prefix -> standard prefix
        src = f"transformer_blocks.{i}"
        dst = f"blocks.{i}"

        # Per-block scale_shift_table: [6, dim] -> [1, 6, dim]
        sst = _load_tensor(readers, f"{src}.scale_shift_table")
        weights[f"{dst}.scale_shift_table"] = sst.astype(np.float32).reshape(1, 6, dim)

        # Self-attention (all projections have bias in PixArt)
        for proj in ("to_q", "to_k", "to_v"):
            weights[f"{dst}.attn1.{proj}.weight"] = _t(f"{src}.attn1.{proj}.weight")
            b = _maybe_f(f"{src}.attn1.{proj}.bias")
            if b is not None:
                weights[f"{dst}.attn1.{proj}.bias"] = b

        weights[f"{dst}.attn1.to_out.0.weight"] = _t(f"{src}.attn1.to_out.0.weight")
        b = _maybe_f(f"{src}.attn1.to_out.0.bias")
        if b is not None:
            weights[f"{dst}.attn1.to_out.0.bias"] = b

        # Cross-attention
        for proj in ("to_q", "to_k", "to_v"):
            weights[f"{dst}.attn2.{proj}.weight"] = _t(f"{src}.attn2.{proj}.weight")
            b = _maybe_f(f"{src}.attn2.{proj}.bias")
            if b is not None:
                weights[f"{dst}.attn2.{proj}.bias"] = b

        weights[f"{dst}.attn2.to_out.0.weight"] = _t(f"{src}.attn2.to_out.0.weight")
        b = _maybe_f(f"{src}.attn2.to_out.0.bias")
        if b is not None:
            weights[f"{dst}.attn2.to_out.0.bias"] = b

        # FFN: ff.net.0.proj (GELU Linear) + ff.net.2 (output Linear)
        # Map to standard naming: ffn.net.0.proj / ffn.net.2
        weights[f"{dst}.ffn.net.0.proj.weight"] = _t(f"{src}.ff.net.0.proj.weight")
        b = _maybe_f(f"{src}.ff.net.0.proj.bias")
        if b is not None:
            weights[f"{dst}.ffn.net.0.proj.bias"] = b

        weights[f"{dst}.ffn.net.2.weight"] = _t(f"{src}.ff.net.2.weight")
        b = _maybe_f(f"{src}.ff.net.2.bias")
        if b is not None:
            weights[f"{dst}.ffn.net.2.bias"] = b

    # Final output: scale_shift_table [2, dim] -> [1, 2, dim]
    sst_final = _load_tensor(readers, "scale_shift_table")
    weights["scale_shift_table"] = sst_final.astype(np.float32).reshape(1, 2, dim)

    # Final projection
    weights["proj_out.weight"] = _t("proj_out.weight")
    b = _maybe_f("proj_out.bias")
    if b is not None:
        weights["proj_out.bias"] = b

    # Preprocessor weights (used externally, not in TRT engine)
    # Patch embedding Conv2d
    if _has_tensor(readers, "pos_embed.proj.weight"):
        weights["pos_embed.proj.weight"] = _load_tensor(readers, "pos_embed.proj.weight").astype(
            np.float32
        )
    if _has_tensor(readers, "pos_embed.proj.bias"):
        weights["pos_embed.proj.bias"] = _load_tensor(readers, "pos_embed.proj.bias").astype(
            np.float32
        )

    # Timestep embedder (adaln_single)
    _adaln_keys = [
        "adaln_single.emb.timestep_embedder.linear_1.weight",
        "adaln_single.emb.timestep_embedder.linear_1.bias",
        "adaln_single.emb.timestep_embedder.linear_2.weight",
        "adaln_single.emb.timestep_embedder.linear_2.bias",
        "adaln_single.linear.weight",
        "adaln_single.linear.bias",
    ]
    for key in _adaln_keys:
        if _has_tensor(readers, key):
            w = _load_tensor(readers, key).astype(np.float32)
            if w.ndim == 2:
                weights[key] = np.ascontiguousarray(w.T, dtype=np.float32)
            else:
                weights[key] = w

    # Caption projection (T5 4096 -> dit_dim)
    _caption_keys = [
        "caption_projection.linear_1.weight",
        "caption_projection.linear_1.bias",
        "caption_projection.linear_2.weight",
        "caption_projection.linear_2.bias",
    ]
    for key in _caption_keys:
        if _has_tensor(readers, key):
            w = _load_tensor(readers, key).astype(np.float32)
            if w.ndim == 2:
                weights[key] = np.ascontiguousarray(w.T, dtype=np.float32)
            else:
                weights[key] = w

    return weights


def _serialize_preprocessor_weights(
    dit_weights: WeightDict,
    t5_dim: int,
    dit_dim: int,
) -> bytes:
    """Serialize PixArt preprocessor weights into binary format.

    Format: JSON index (length-prefixed) + contiguous float32 data.

    Preprocessor weights stored:
        pos_embed.proj.weight, pos_embed.proj.bias
        adaln_single.emb.timestep_embedder.linear_1.weight/bias
        adaln_single.emb.timestep_embedder.linear_2.weight/bias
        adaln_single.linear.weight/bias
        caption_projection.linear_1.weight/bias
        caption_projection.linear_2.weight/bias

    These are mapped to Wan-compatible key names where possible so the
    C++ parse_preprocessor_weights() can load them:
        pos_embed.proj -> patch_embedding
        timestep_embedder -> condition_embedder.time_embedding
        adaln_single.linear -> condition_embedder.time_proj
        caption_projection -> condition_embedder.text_embedding
    """
    import json
    import struct
    import numpy as np

    key_map = {
        # Patch embedding -> patch_embedding (Wan-compatible)
        "pos_embed.proj.weight": "patch_embedding.weight",
        "pos_embed.proj.bias": "patch_embedding.bias",
        # Timestep MLP -> condition_embedder.time_embedding
        "adaln_single.emb.timestep_embedder.linear_1.weight": "condition_embedder.time_embedding.0.weight",
        "adaln_single.emb.timestep_embedder.linear_1.bias": "condition_embedder.time_embedding.0.bias",
        "adaln_single.emb.timestep_embedder.linear_2.weight": "condition_embedder.time_embedding.2.weight",
        "adaln_single.emb.timestep_embedder.linear_2.bias": "condition_embedder.time_embedding.2.bias",
        # adaln_single.linear -> condition_embedder.time_proj
        "adaln_single.linear.weight": "condition_embedder.time_proj.weight",
        "adaln_single.linear.bias": "condition_embedder.time_proj.bias",
        # Caption projection -> condition_embedder.text_embedding
        "caption_projection.linear_1.weight": "condition_embedder.text_embedding.weight",
        "caption_projection.linear_1.bias": "condition_embedder.text_embedding.bias",
        "caption_projection.linear_2.weight": "condition_embedder.text_embedding_2.weight",
        "caption_projection.linear_2.bias": "condition_embedder.text_embedding_2.bias",
    }

    index = {}
    data_parts = []
    offset = 0

    for src_key, dst_key in key_map.items():
        if src_key not in dit_weights:
            continue
        w = dit_weights[src_key]
        if not isinstance(w, np.ndarray):
            w = np.array(w, dtype=np.float32)
        w = np.ascontiguousarray(w.astype(np.float32))

        # Patch embedding is Conv2d [out_ch, C, kH, kW].
        # The C++ patchify produces patches in (C, kH, kW) order
        # (see WanDiffusionBackend::patchify — C loops outermost).
        # So the weight stays in (out_ch, C, kH, kW) order — just
        # flatten to [out_ch, patch_dim] then transpose to [patch_dim, out_ch].
        if src_key == "pos_embed.proj.weight" and w.ndim > 2:
            out_ch = w.shape[0]
            patch_dim = int(np.prod(w.shape[1:]))
            w = np.ascontiguousarray(w.reshape(out_ch, patch_dim).T)

        nbytes = w.nbytes
        index[dst_key] = {"offset": offset, "shape": list(w.shape)}
        data_parts.append(w.tobytes())
        offset += nbytes

    index_json = json.dumps(index).encode("utf-8")
    result = struct.pack("<I", len(index_json)) + index_json
    for part in data_parts:
        result += part

    return result


plugin = PixArtPlugin()
