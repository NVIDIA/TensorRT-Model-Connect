"""Diffusion pipeline runner — pure-Python TRT inference for diffusion models.

Generic runner that loads N text encoder engines + denoiser + VAE from a
bundle. Scheduler selected by config. CFG handled generically.

Follows the same pattern as debug_runner.py TrtRunner for decoder models.
"""

from __future__ import annotations

import json
import struct
import sys

import numpy as np

from tensorrt_model_connect import trt_compat
from .schedulers import get_scheduler

trt = trt_compat.get_trt() if trt_compat.is_available() else None

try:
    try:
        from cuda import cudart
    except ImportError:
        # cuda-python >= 13.x uses cuda.bindings
        from cuda.bindings import runtime as cudart
    HAS_CUDA = trt is not None
except ImportError:
    HAS_CUDA = False


def _check_cuda(status):
    """Check CUDA status and raise on error."""
    if isinstance(status, tuple):
        err = status[0]
        if err.value != 0:
            raise RuntimeError(f"CUDA error: {err}")
        return status[1] if len(status) > 1 else None
    return status


class DiffusionRunner:
    """Generic diffusion pipeline runner with TRT engines.

    Loads text encoder(s), denoiser, and VAE decoder engines from a bundle.
    Runs the full text -> denoise -> decode pipeline.
    """

    def __init__(self, bundle_path: str):
        """Load engines and config from a .trtfb bundle.

        Args:
            bundle_path: Path to the diffusion .trtfb bundle.
        """
        if not HAS_CUDA:
            raise RuntimeError("CUDA/TensorRT not available for DiffusionRunner")

        self.bundle_path = bundle_path
        self.config = {}
        self._engines = {}
        self._contexts = {}

        # Parse bundle
        self._load_bundle(bundle_path)

        # Initialize scheduler
        scheduler_name = self.config.get("scheduler", "flow_match_euler")
        scheduler_kwargs = {}
        if "flow_shift" in self.config:
            scheduler_kwargs["shift"] = self.config["flow_shift"]
        try:
            self.scheduler = get_scheduler(scheduler_name, **scheduler_kwargs)
        except ValueError:
            self.scheduler = None

        # CUDA stream
        self.stream = _check_cuda(cudart.cudaStreamCreate())

    def _load_bundle(self, path: str) -> None:
        """Parse bundle file and deserialize TRT engines."""
        with open(path, "rb") as f:
            magic = f.read(8)
            assert magic == b"TRTFB\x00\x01\x00", f"Bad magic: {magic}"
            json_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(json_len).decode("utf-8"))
            data_start = 16 + json_len

        sections = header.get("sections", {})

        # Load config
        if "config.json" in sections:
            sec = sections["config.json"]
            with open(path, "rb") as f:
                f.seek(data_start + sec["offset"])
                self.config = json.loads(f.read(sec["size"]).decode("utf-8"))

        # Load engine plans and deserialize
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        for sec_name, sec_info in sections.items():
            if sec_name.endswith("_plan"):
                with open(path, "rb") as f:
                    f.seek(data_start + sec_info["offset"])
                    plan_data = f.read(sec_info["size"])

                engine = runtime.deserialize_cuda_engine(plan_data)
                if engine is None:
                    raise RuntimeError(
                        f"Failed to deserialize engine: {sec_name}")

                context = engine.create_execution_context()
                name = sec_name.replace("_plan", "")
                self._engines[name] = engine
                self._contexts[name] = context

    def _run_engine(
        self,
        engine_name: str,
        inputs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Run a single TRT engine with the given inputs.

        Returns dict of output name -> numpy array.
        """
        engine = self._engines[engine_name]
        context = self._contexts[engine_name]

        # Allocate device buffers and set tensor addresses
        device_buffers = {}
        host_outputs = {}

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            shape = engine.get_tensor_shape(name)
            # Convert shape tuple to positive ints
            shape = tuple(max(1, s) for s in shape)
            dtype = engine.get_tensor_dtype(name)

            np_dtype = trt.nptype(dtype)
            size_bytes = int(np.prod(shape)) * np.dtype(np_dtype).itemsize

            if mode == trt.TensorIOMode.INPUT:
                if name in inputs:
                    h_data = np.ascontiguousarray(
                        inputs[name].astype(np_dtype))
                else:
                    h_data = np.zeros(shape, dtype=np_dtype)

                d_ptr = _check_cuda(cudart.cudaMalloc(size_bytes))
                _check_cuda(cudart.cudaMemcpyAsync(
                    d_ptr, h_data.ctypes.data, size_bytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self.stream))
                device_buffers[name] = d_ptr
                context.set_tensor_address(name, d_ptr)
            else:
                d_ptr = _check_cuda(cudart.cudaMalloc(size_bytes))
                device_buffers[name] = d_ptr
                host_outputs[name] = (shape, np_dtype, size_bytes)
                context.set_tensor_address(name, d_ptr)

        # Execute
        context.execute_async_v3(self.stream)
        _check_cuda(cudart.cudaStreamSynchronize(self.stream))

        # Copy outputs to host
        results = {}
        for name, (shape, np_dtype, size_bytes) in host_outputs.items():
            h_out = np.empty(shape, dtype=np_dtype)
            _check_cuda(cudart.cudaMemcpy(
                h_out.ctypes.data, device_buffers[name], size_bytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
            results[name] = h_out

        # Free device buffers
        for d_ptr in device_buffers.values():
            cudart.cudaFree(d_ptr)

        return results

    def encode_text(self, input_ids: np.ndarray) -> np.ndarray:
        """Run text encoder(s) on input token IDs.

        Args:
            input_ids: [1, seq_len] int32 token IDs.

        Returns:
            Text embeddings [1, seq_len, text_dim].
        """
        # Find text encoder engine(s)
        te_names = sorted(
            k for k in self._engines if k.startswith("text_encoder"))

        if not te_names:
            raise RuntimeError("No text encoder engine in bundle")

        te_engine = self._engines[te_names[0]]
        input_shape = tuple(max(1, s) for s in te_engine.get_tensor_shape("input_ids"))
        seq_len = input_shape[1] if len(input_shape) >= 2 else input_ids.shape[1]

        padded_ids = np.zeros((1, seq_len), dtype=np.int32)
        copy_len = min(seq_len, input_ids.shape[1])
        padded_ids[:, :copy_len] = input_ids[:, :copy_len].astype(np.int32)

        # Match the engine's expected attention-mask dtype/semantics.
        ids_flat = padded_ids.flatten()
        mask_dtype = trt.nptype(te_engine.get_tensor_dtype("attention_mask"))
        if np.issubdtype(mask_dtype, np.integer):
            attn_mask = (padded_ids != 0).astype(mask_dtype)
        else:
            attn_mask = np.where(ids_flat != 0, 1.0, 0.0).astype(mask_dtype)
            attn_mask = attn_mask.reshape(padded_ids.shape)

        # Run first (primary) text encoder
        results = self._run_engine(te_names[0], {
            "input_ids": padded_ids,
            "attention_mask": attn_mask,
        })
        embeddings = results.get("text_embeddings")
        if embeddings is None:
            embeddings = results.get("output0")
        if embeddings is None:
            output_names = [name for name in results]
            if not output_names:
                raise RuntimeError("Text encoder engine has no outputs")
            embeddings = results[output_names[0]]

        # Wan-style diffusion paths have no text attention mask in the DiT and
        # benefit from zeroing padding rows.
        out_seq_len = embeddings.shape[1]
        valid_mask = (ids_flat[:out_seq_len] != 0).astype(np.float32)
        valid_mask = valid_mask.reshape(1, out_seq_len, 1)
        embeddings = embeddings * valid_mask

        # Truncate to actual content length + padding matching HF's convention.
        # HF WanPipeline uses max_sequence_length=226 by default.
        # The DiT cross-attention has no mask, so sequence length affects
        # softmax normalization.
        max_text_len = self.config.get("text_seq_len", 512)
        if embeddings.shape[1] > max_text_len:
            embeddings = embeddings[:, :max_text_len, :]

        return embeddings

    def denoise(
        self,
        latents: np.ndarray,
        text_embeddings: np.ndarray,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
    ) -> np.ndarray:
        """Run the denoising loop.

        Args:
            latents: Initial noise [1, C, T, H, W].
            text_embeddings: Text encoder output [1, seq_len, text_dim].
            num_inference_steps: Number of denoising steps.
            guidance_scale: CFG guidance scale.

        Returns:
            Denoised latents [1, C, T, H, W].
        """
        if self.scheduler is None:
            raise RuntimeError(
                f"Unsupported scheduler for DiffusionRunner: {self.config.get('scheduler', '')}")
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Get model config for computing patches, RoPE, etc.
        dim = self.config.get("dit_dim", 1536)
        num_heads = self.config.get("dit_num_heads", 12)
        head_dim = dim // num_heads
        patch_size = self.config.get("patch_size", [1, 2, 2])

        _, c, t_lat, h_lat, w_lat = latents.shape
        pt, ph, pw = patch_size
        text_seq_len = text_embeddings.shape[1]

        for step_idx, timestep in enumerate(timesteps):
            # Prepare inputs for denoiser
            # Patchify latents: [1, C, T, H, W] -> [num_patches, patch_dim]
            hidden = self._patchify(latents, patch_size)

            # Compute timestep embedding (sinusoidal + MLP)
            temb = self._compute_timestep_embedding(float(timestep))

            # Compute 3D RoPE
            rope_cos, rope_sin = self._compute_3d_rope(
                t_lat // pt, h_lat // ph, w_lat // pw, head_dim)

            # Text conditioning: project to denoiser dim
            text_proj = text_embeddings.reshape(text_seq_len, -1)

            # For CFG: run with text and without text (null conditioning)
            if guidance_scale > 1.0:
                # Batch: [hidden_states, hidden_states_uncond]
                # For simplicity, run two passes
                noise_pred_text = self._run_engine("denoiser", {
                    "hidden_states": hidden,
                    "timestep_embedding": temb,
                    "encoder_hidden_states": text_proj,
                    "rotary_cos": rope_cos,
                    "rotary_sin": rope_sin,
                })["output"]

                # Null conditioning (zeros)
                null_text = np.zeros_like(text_proj)
                noise_pred_uncond = self._run_engine("denoiser", {
                    "hidden_states": hidden,
                    "timestep_embedding": temb,
                    "encoder_hidden_states": null_text,
                    "rotary_cos": rope_cos,
                    "rotary_sin": rope_sin,
                })["output"]

                # CFG
                noise_pred = (noise_pred_uncond +
                              guidance_scale * (noise_pred_text - noise_pred_uncond))
            else:
                noise_pred = self._run_engine("denoiser", {
                    "hidden_states": hidden,
                    "timestep_embedding": temb,
                    "encoder_hidden_states": text_proj,
                    "rotary_cos": rope_cos,
                    "rotary_sin": rope_sin,
                })["output"]

            # Unpatchify: [num_patches, patch_dim] -> [1, C, T, H, W]
            noise_pred_spatial = self._unpatchify(
                noise_pred, patch_size, c, t_lat, h_lat, w_lat)

            # Scheduler step
            latents = self.scheduler.step(
                noise_pred_spatial, float(timestep), latents, step_idx)

            if step_idx % 10 == 0:
                print(f"  Step {step_idx+1}/{len(timesteps)} "
                      f"(t={float(timestep):.1f})", file=sys.stderr)

        return latents

    def decode_video(
        self,
        latents: np.ndarray,
    ) -> np.ndarray:
        """Decode latents to video frames using the VAE decoder.

        Args:
            latents: Denoised latents [1, C, T_lat, H_lat, W_lat].

        Returns:
            Video frames [1, 3, T, H, W] in [0, 1] range.
        """
        # Denormalize latents
        latents_mean = np.array(
            self.config.get("latents_mean", [0.0] * 16), dtype=np.float32)
        latents_std = np.array(
            self.config.get("latents_std", [1.0] * 16), dtype=np.float32)

        # latents = latents * std + mean (undo normalization)
        latents = (latents * latents_std.reshape(1, -1, 1, 1, 1) +
                   latents_mean.reshape(1, -1, 1, 1, 1))

        # Frame-by-frame VAE decode
        _, c, t_lat, h_lat, w_lat = latents.shape
        frames = []

        # Initialize caches (zeros)
        vae_engine = self._engines.get("vae_decoder")
        if vae_engine is None:
            raise RuntimeError("No VAE decoder engine in bundle")

        # Enumerate cache inputs
        cache_states = {}
        for i in range(vae_engine.num_io_tensors):
            name = vae_engine.get_tensor_name(i)
            if name.startswith("cache_") and not name.startswith("cache_out_"):
                shape = tuple(vae_engine.get_tensor_shape(name))
                cache_states[name] = np.zeros(shape, dtype=np.float32)

        for t in range(t_lat):
            # Extract single latent frame
            latent_frame = latents[:, :, t:t+1, :, :]

            # Build input dict
            inputs = {"latent_frame": latent_frame}
            inputs.update(cache_states)

            # Run VAE
            results = self._run_engine("vae_decoder", inputs)

            # Extract frame and updated caches
            frame = results["video_frame"]
            frames.append(frame)

            # Update caches
            for name in cache_states:
                out_name = name.replace("cache_", "cache_out_")
                if out_name in results:
                    cache_states[name] = results[out_name]

        # Stack frames along temporal dim
        video = np.concatenate(frames, axis=2)

        # Trim extra frames from frame-0's temporal upsample.
        # The TRT engine always runs temporal pixel-shuffle (even for frame 0
        # with zero caches), producing scale_factor_temporal output frames per
        # input frame. HF skips pixel-shuffle for frame 0, producing only 1.
        # Trim the first (scale_factor_temporal - 1) frames to match HF.
        sft = self.config.get("scale_factor_temporal", 1)
        trim = sft - 1
        if trim > 0 and video.shape[2] > trim:
            video = video[:, :, trim:, :, :]

        # Clamp to [0, 1]
        video = np.clip((video + 1.0) / 2.0, 0.0, 1.0)

        return video

    def generate(
        self,
        input_ids: np.ndarray,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        video_num_frames: int = 81,
        video_height: int = 480,
        video_width: int = 832,
        seed: int = 42,
    ) -> np.ndarray:
        """Full generation pipeline: text encode -> denoise -> VAE decode.

        Args:
            input_ids: Text token IDs [1, seq_len].
            num_inference_steps: Denoising steps.
            guidance_scale: CFG guidance scale.
            video_num_frames: Output video frames.
            video_height: Output video height.
            video_width: Output video width.
            seed: Random seed for initial noise.

        Returns:
            Video frames [1, 3, T, H, W] float32 in [0, 1].
        """
        # 1. Encode text
        print("[diffusion] Encoding text ...", file=sys.stderr)
        text_embeddings = self.encode_text(input_ids)

        # 2. Prepare latents (random noise)
        z_dim = self.config.get("z_dim", 16)
        temporal_compression = self.config.get("scale_factor_temporal", 4)
        spatial_compression = self.config.get("scale_factor_spatial", 8)

        t_lat = (video_num_frames - 1) // temporal_compression + 1
        h_lat = video_height // spatial_compression
        w_lat = video_width // spatial_compression

        rng = np.random.default_rng(seed)
        latents = rng.standard_normal(
            (1, z_dim, t_lat, h_lat, w_lat)).astype(np.float32)

        # 3. Denoise
        print(f"[diffusion] Denoising ({num_inference_steps} steps) ...",
              file=sys.stderr)
        denoised = self.denoise(
            latents, text_embeddings, num_inference_steps, guidance_scale)

        # 4. VAE decode
        print("[diffusion] Decoding video ...", file=sys.stderr)
        video = self.decode_video(denoised)

        return video

    # --- Internal helpers ---

    def _patchify(
        self, x: np.ndarray, patch_size: list[int],
    ) -> np.ndarray:
        """Convert [1, C, T, H, W] to [num_patches, C*pt*ph*pw]."""
        _, c, t, h, w = x.shape
        pt, ph, pw = patch_size
        nt, nh, nw = t // pt, h // ph, w // pw

        # Reshape and permute
        x = x.reshape(1, c, nt, pt, nh, ph, nw, pw)
        x = x.transpose(0, 2, 4, 6, 1, 3, 5, 7)  # [1, nt, nh, nw, c, pt, ph, pw]
        x = x.reshape(nt * nh * nw, c * pt * ph * pw)
        return x.astype(np.float32)

    def _unpatchify(
        self, x: np.ndarray, patch_size: list[int],
        c: int, t: int, h: int, w: int,
    ) -> np.ndarray:
        """Convert [num_patches, pt*ph*pw*C] to [1, C, T, H, W].

        The DiT proj_out produces output ordered as [pt, ph, pw, C]
        (matching HF's WanTransformer3DModel convention).
        """
        pt, ph, pw = patch_size
        nt, nh, nw = t // pt, h // ph, w // pw

        # Output dim is ordered [pt, ph, pw, c] (C varies fastest)
        x = x.reshape(nt, nh, nw, pt, ph, pw, c)
        x = x.transpose(6, 0, 3, 1, 4, 2, 5)  # [c, nt, pt, nh, ph, nw, pw]
        x = x.reshape(1, c, t, h, w)
        return x.astype(np.float32)

    def _compute_timestep_embedding(self, timestep: float) -> np.ndarray:
        """Compute timestep embedding using weights from the bundle.

        Returns [1, dim * 6] for DiT block modulation.
        """
        freq_dim = self.config.get("freq_dim", 256)

        # Sinusoidal embedding
        half = freq_dim // 2
        freqs = np.exp(
            -np.log(10000.0) * np.arange(half, dtype=np.float64) / half)
        args = timestep * freqs
        embed = np.concatenate([np.cos(args), np.sin(args)]).astype(np.float32)
        embed = embed.reshape(1, freq_dim)

        # MLP: Linear -> SiLU -> Linear (produces [1, dim*6])
        # These weights are stored in the DiT weights dict but we need them
        # at runtime. For now, return a placeholder that will be computed
        # from cached weights.
        # TODO: Load MLP weights into the runner
        return embed  # Placeholder — actual implementation needs MLP weights

    def _compute_3d_rope(
        self, nt: int, nh: int, nw: int, head_dim: int,
        theta: float = 10000.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute 3D RoPE cos/sin tables for DiT.

        Returns (cos, sin) each of shape [num_patches, head_dim].
        """
        # Split head_dim into temporal, height, width components
        # Wan uses: t_dim = head_dim - 2*(head_dim//6), h_dim = w_dim = head_dim//6
        h_dim = w_dim = 2 * (head_dim // 6)
        t_dim = head_dim - h_dim - w_dim

        def _get_1d_rope(dim: int, max_len: int):
            half = dim // 2
            freqs = 1.0 / (theta ** (np.arange(half, dtype=np.float64) / half))
            positions = np.arange(max_len, dtype=np.float64)
            angles = np.outer(positions, freqs)
            cos = np.cos(angles)
            sin = np.sin(angles)
            # Repeat interleave: [max_len, half] -> [max_len, dim]
            cos = np.repeat(cos, 2, axis=1)
            sin = np.repeat(sin, 2, axis=1)
            return cos.astype(np.float32), sin.astype(np.float32)

        t_cos, t_sin = _get_1d_rope(t_dim, max(nt, 1024))
        h_cos, h_sin = _get_1d_rope(h_dim, max(nh, 1024))
        w_cos, w_sin = _get_1d_rope(w_dim, max(nw, 1024))

        # Build 3D grid: [nt, nh, nw] positions
        cos_parts = []
        sin_parts = []
        for ti in range(nt):
            for hi in range(nh):
                for wi in range(nw):
                    cos_row = np.concatenate([
                        t_cos[ti], h_cos[hi], w_cos[wi]])
                    sin_row = np.concatenate([
                        t_sin[ti], h_sin[hi], w_sin[wi]])
                    cos_parts.append(cos_row)
                    sin_parts.append(sin_row)

        rope_cos = np.stack(cos_parts)  # [num_patches, head_dim]
        rope_sin = np.stack(sin_parts)

        return rope_cos, rope_sin
