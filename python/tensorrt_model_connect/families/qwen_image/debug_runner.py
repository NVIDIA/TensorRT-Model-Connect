"""Qwen Image-owned TRT debug runner."""

from __future__ import annotations

from typing import Any

import numpy as np

from tensorrt_model_connect import trt_compat


trt = trt_compat.get_trt() if trt_compat.is_available() else None

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    try:
        from cuda import cudart  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover - exercised in TRT-free test envs
        cudart = None  # type: ignore[assignment]

def _check_cuda(status):
    """Raise on CUDA error."""
    if cudart is None:
        raise RuntimeError("cuda-python is required for family debug_runner execution")
    if hasattr(cudart, "cudaError_t"):
        success = cudart.cudaError_t.cudaSuccess
    else:
        success = 0
    if status != success:
        raise RuntimeError(f"CUDA error: {status}")

def _trt_nptype_safe(dtype: trt.DataType):
    """Resolve TRT dtype to a NumPy dtype, including BF16 fallback."""
    try:
        return trt.nptype(dtype)
    except TypeError:
        if dtype == trt.bfloat16:
            return np.uint16
        raise

def _require_trt_runtime() -> None:
    if trt is None:
        raise ImportError("tensorrt is required for family debug_runner execution")
    if cudart is None:
        raise ImportError("cuda-python is required for family debug_runner execution")



# ---------------------------------------------------------------------------
# Qwen-Image debug runner
# ---------------------------------------------------------------------------


# Hardcoded T2I prompt template (mirror of diffusers
# QwenImagePipeline.prompt_template_encode). The 34-token prefix that this
# template adds is later stripped from the text-encoder output via the
# ``drop_idx`` mechanism. Bundles record both fields under
# ``tokenizer.prompt_template_kind`` / ``prompt_template_drop_idx``.
_QWEN_IMAGE_T2I_PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, "
    "size, texture, quantity, text, spatial relationships of the objects "
    "and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
_QWEN_IMAGE_T2I_DROP_IDX = 34


def load_engine_from_bundle(
    bundle_path: str,
    section_name: str = "engine_plan",
) -> tuple[bytes, dict]:
    """Load this family's engine plan bytes and bundle metadata."""
    import json
    import struct

    with open(bundle_path, "rb") as f:
        magic = f.read(8)
        if magic != b"TRTFB\x00\x01\x00":
            raise ValueError(f"Not a valid .trtfb bundle: {bundle_path}")
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        sections = header.get("sections", {})
        engine_meta = sections.get(section_name)
        if engine_meta is None:
            raise KeyError(
                f"Bundle {bundle_path!r} does not contain section {section_name!r}")
        f.seek(16 + header_len + engine_meta["offset"])
        engine_plan = f.read(engine_meta["size"])

    return engine_plan, header

def load_section_from_bundle(bundle_path: str, section_name: str) -> bytes | None:
    """Load a named raw section from this family's .trtfb bundle."""
    import json
    import struct

    with open(bundle_path, "rb") as f:
        magic = f.read(8)
        if magic != b"TRTFB\x00\x01\x00":
            raise ValueError(f"Not a valid .trtfb bundle: {bundle_path}")
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        sections = header.get("sections", {})
        meta = sections.get(section_name)
        if meta is None:
            return None
        f.seek(16 + header_len + meta["offset"])
        return f.read(meta["size"])

def load_config_from_bundle(bundle_path: str) -> dict:
    """Load and parse this family's config.json from a .trtfb bundle."""
    import json

    data = load_section_from_bundle(bundle_path, "config.json")
    if data is None:
        return {}
    return json.loads(data.decode("utf-8"))


class QwenImageDebugRunner:
    """Pure-Python TRT runner for Qwen-Image (T2I) ``.trtfb`` bundles.

    Loads the bundle, deserialises the three engines (``text_encoder_0``,
    ``denoiser``, ``vae_decoder``), and implements the full T2I pipeline:
    prompt template + tokenize + text encode + flow-match Euler denoising
    with true-CFG + VAE decode.

    This mirrors what the C++ pipeline does, so the HF parity gate
    validates the entire Python-side pipeline. Any later C++ divergence
    is purely C++-side.

    Engines baked into the bundle are static-shape at 1024x1024:
      text_encoder_0:  input_ids[1024] int32, attention_mask[1024] f32
                       -> last_hidden_state[1024, 3584] f32
      denoiser:        img_patched[1, 4096, 64] f32,
                       txt_hidden[1, 1024, 3584] f32, timestep[1] f32
                       -> noise_patched[1, 4096, 64] f32
      vae_decoder:     latent[1, 16, 1, 128, 128] f32
                       -> image[1, 3, 1, 1024, 1024] f32 (range ~[-1, 1])

    Trace: ARCH-FAM-001, UD-FAM-QWEN-IMAGE-01, IT-QWEN-IMAGE-DEBUG-RUN-001.
    """

    def __init__(self, bundle_path: str | Any) -> None:
        _require_trt_runtime()

        import json
        import struct
        from pathlib import Path

        self.bundle_path = Path(bundle_path)
        if not self.bundle_path.exists():
            raise FileNotFoundError(f"bundle not found: {self.bundle_path}")

        # --- Parse bundle header. ----------------------------------------
        with open(self.bundle_path, "rb") as f:
            magic = f.read(8)
            if magic != b"TRTFB\x00\x01\x00":
                raise ValueError(
                    f"not a valid .trtfb bundle: magic={magic!r}")
            header_len = struct.unpack("<Q", f.read(8))[0]
            header_bytes = f.read(header_len)
        header = json.loads(header_bytes.decode("utf-8"))
        self._header = header
        sections = header.get("sections", {})
        data_start = 16 + header_len

        def _read_section(name: str) -> bytes:
            meta = sections.get(name)
            if meta is None:
                raise KeyError(
                    f"section {name!r} not found; available: "
                    f"{list(sections.keys())}"
                )
            with open(self.bundle_path, "rb") as fh:
                fh.seek(data_start + meta["offset"])
                return fh.read(meta["size"])

        # --- Load embedded config.json (the variant schema). -------------
        config_bytes = _read_section("config.json")
        self.config: dict[str, Any] = json.loads(config_bytes.decode("utf-8"))

        # --- Deserialise the three engines. ------------------------------
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        def _load_engine(plan_name: str):
            plan = _read_section(plan_name)
            engine = runtime.deserialize_cuda_engine(plan)
            if engine is None:
                raise RuntimeError(f"failed to deserialise {plan_name}")
            context = engine.create_execution_context()
            return engine, context

        self._text_engine, self._text_ctx = _load_engine("text_encoder_0_plan")
        self._dit_engine, self._dit_ctx = _load_engine("denoiser_plan")
        self._vae_engine, self._vae_ctx = _load_engine("vae_decoder_plan")

        # --- Parse preprocessor weights (latents_mean / latents_std). ----
        from .qwen_image_preprocessor import (
            load_qwen_image_preprocessor_weights,
        )
        pp_bytes = _read_section("preprocessor_weights")
        self._preprocessor = load_qwen_image_preprocessor_weights(pp_bytes)

        # --- Materialise tokenizer files to a temp dir, then load it. ----
        import tempfile
        self._tokenizer_dir = tempfile.mkdtemp(
            prefix="qwen_image_tokenizer_")
        tok_dir = Path(self._tokenizer_dir)
        for tok_file in (
            "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "vocab.json", "merges.txt",
        ):
            data = _read_section(tok_file)
            (tok_dir / tok_file).write_bytes(data)

        # Lazy-imported to avoid forcing transformers as a hard dep when this
        # module is imported just for the decoder runners.
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tok_dir), trust_remote_code=False,
        )

        # --- CUDA stream. -----------------------------------------------
        err, self.stream = cudart.cudaStreamCreate()
        _check_cuda(err)

        # --- Derive shape constants from the engine signatures. ----------
        # Text encoder: input_ids[max_text_input], output[max_text_input, hidden]
        ti_shape = tuple(self._text_engine.get_tensor_shape("input_ids"))
        to_shape = tuple(
            self._text_engine.get_tensor_shape("last_hidden_state"))
        self._text_max_tokens = int(ti_shape[0])
        self._text_hidden = int(to_shape[1])

        # Denoiser: img_patched [1, n_img, in_ch], txt_hidden [1, n_txt, 3584]
        img_shape = tuple(self._dit_engine.get_tensor_shape("img_patched"))
        txt_shape = tuple(self._dit_engine.get_tensor_shape("txt_hidden"))
        self._n_img = int(img_shape[1])
        self._in_ch_packed = int(img_shape[2])
        self._n_txt = int(txt_shape[1])
        # img tokens = h_lat//2 * w_lat//2; for square images h_lat = w_lat.
        side = int(round((self._n_img) ** 0.5))
        if side * side != self._n_img:
            raise NotImplementedError(
                f"non-square image latents not supported (n_img={self._n_img})"
            )
        # Packed-token grid is half the latent grid (patch_size=2).
        self._packed_h = side
        self._packed_w = side
        self._latent_h = side * 2
        self._latent_w = side * 2
        # Bundle config provides VAE channel count (= packed_ch / 4).
        self._latent_ch = int(self.config["vae"]["latent_channels"])

        # VAE: image [1, 3, 1, H, W] — record H/W for sanity-check.
        vae_in_shape = tuple(self._vae_engine.get_tensor_shape("latent"))
        vae_out_shape = tuple(self._vae_engine.get_tensor_shape("image"))
        self._vae_latent_shape = vae_in_shape  # (1, 16, 1, 128, 128)
        self._vae_image_shape = vae_out_shape  # (1, 3, 1, 1024, 1024)

        # Scheduler config from bundle (matches diffusers
        # scheduler_config.json for Qwen-Image-2512).
        self._sched_cfg = dict(self.config.get("diffusion", {}))

    # ------------------------------------------------------------------
    # Engine execution helper (single forward pass, static shapes).
    # ------------------------------------------------------------------

    def _run_engine(
        self,
        engine,
        context,
        inputs: dict[str, np.ndarray],
        output_names: list[str],
    ) -> dict[str, np.ndarray]:
        """Run one TRT execution with the given host-side inputs.

        Allocates device buffers for every IO tensor, H2D-copies inputs,
        executes, D2H-copies outputs, frees device buffers. Static-shape
        only.
        """
        d_buffers: dict[str, int] = {}
        out_specs: dict[str, tuple[tuple[int, ...], np.dtype, int]] = {}
        try:
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                mode = engine.get_tensor_mode(name)
                shape = tuple(int(d) for d in engine.get_tensor_shape(name))
                dtype_trt = engine.get_tensor_dtype(name)
                dtype_np = _trt_nptype_safe(dtype_trt)
                nbytes = int(np.prod(shape)) * np.dtype(dtype_np).itemsize

                err, d_ptr = cudart.cudaMalloc(nbytes)
                _check_cuda(err)
                d_buffers[name] = d_ptr
                context.set_tensor_address(name, int(d_ptr))

                if mode == trt.TensorIOMode.INPUT:
                    if name not in inputs:
                        raise KeyError(
                            f"missing input {name!r} for engine "
                            f"(expected {list(inputs.keys())})"
                        )
                    host_in = np.ascontiguousarray(
                        inputs[name].astype(dtype_np)
                    )
                    if host_in.shape != shape:
                        raise ValueError(
                            f"input {name!r} shape mismatch: got "
                            f"{host_in.shape}, expected {shape}"
                        )
                    err = cudart.cudaMemcpyAsync(
                        int(d_ptr),
                        host_in.ctypes.data,
                        nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                        self.stream,
                    )
                    if isinstance(err, tuple):
                        err = err[0]
                    _check_cuda(err)
                else:
                    out_specs[name] = (shape, np.dtype(dtype_np), nbytes)

            ok = context.execute_async_v3(self.stream)
            if not ok:
                raise RuntimeError("execute_async_v3 returned False")

            results: dict[str, np.ndarray] = {}
            for name in output_names:
                shape, dtype_np, nbytes = out_specs[name]
                host_out = np.empty(shape, dtype=dtype_np)
                err = cudart.cudaMemcpyAsync(
                    host_out.ctypes.data,
                    int(d_buffers[name]),
                    nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self.stream,
                )
                if isinstance(err, tuple):
                    err = err[0]
                _check_cuda(err)
                results[name] = host_out

            err = cudart.cudaStreamSynchronize(self.stream)
            if isinstance(err, tuple):
                err = err[0]
            _check_cuda(err)

            return results
        finally:
            for d_ptr in d_buffers.values():
                cudart.cudaFree(d_ptr)

    # ------------------------------------------------------------------
    # Pipeline stages.
    # ------------------------------------------------------------------

    def _encode_prompt(self, prompt: str) -> np.ndarray:
        """Tokenize + run text encoder + apply drop_idx + pad to ``n_txt``.

        Returns ``txt_hidden`` shaped ``[1, n_txt, hidden]`` ready to feed
        the denoiser. Padding rows past the valid length are zero (matching
        the additive-mask convention; the baked denoiser does not consume
        a text mask).
        """
        # 1) Tokenize (template + prompt) to the text encoder's max length.
        text = _QWEN_IMAGE_T2I_PROMPT_TEMPLATE.format(prompt)
        # Diffusers' tokenizer call uses `padding=True` and a max-length
        # cap; here the engine is fixed at self._text_max_tokens, so pad
        # to that length exactly.
        enc = self.tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            max_length=self._text_max_tokens,
            truncation=True,
        )
        input_ids = enc["input_ids"][0].astype(np.int32)  # [max_seq]
        attn_mask01 = enc["attention_mask"][0].astype(np.int64)  # 0/1
        valid_len = int(attn_mask01.sum())

        # Additive mask for the engine: 0 for valid, -1e9 for pad.
        additive_mask = np.zeros(self._text_max_tokens, dtype=np.float32)
        if valid_len < self._text_max_tokens:
            additive_mask[valid_len:] = -1e9

        # 2) Run text encoder.
        out = self._run_engine(
            self._text_engine, self._text_ctx,
            inputs={
                "input_ids": input_ids,
                "attention_mask": additive_mask,
            },
            output_names=["last_hidden_state"],
        )
        hidden = out["last_hidden_state"]  # [max_seq, hidden]
        # Sanity-check: NaN-free.
        if not np.isfinite(hidden).all():
            raise RuntimeError("text encoder output contains NaN/Inf")

        # 3) Drop the first drop_idx (=34) rows of the *valid* prompt and
        # pad to n_txt with zeros. Matches diffusers
        # ``[e[drop_idx:] for e in split_hidden_states]`` followed by the
        # batch-level zero-pad to the max length (here n_txt=1024).
        drop = _QWEN_IMAGE_T2I_DROP_IDX
        kept_len = max(0, valid_len - drop)
        if kept_len == 0:
            raise ValueError(
                f"prompt too short after dropping template prefix "
                f"(valid_len={valid_len}, drop_idx={drop})"
            )
        padded = np.zeros((self._n_txt, self._text_hidden), dtype=np.float32)
        copy_len = min(kept_len, self._n_txt)
        padded[:copy_len] = hidden[drop:drop + copy_len]
        return padded.reshape(1, self._n_txt, self._text_hidden)

    def _prepare_latents(self, seed: int) -> np.ndarray:
        """Sample initial packed latents matching the denoiser shape.

        Returns ``[1, n_img, in_ch_packed]`` float32 — already patchified
        (per diffusers ``_pack_latents``).
        """
        import torch  # lazy: torch is only required for the diffusion path
        # Diffusers prepare_latents shape (T2I): (B, 1, C, H, W). For the
        # T2I pipeline F=1 collapses; we sample at (B, C, H, W) directly so
        # numel matches QwenImagePipeline.prepare_latents (which calls
        # randn_tensor on (B, 1, C, H, W) then _pack_latents reshapes
        # ignoring the leading F=1 axis).
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        latents = torch.randn(
            (1, self._latent_ch, self._latent_h, self._latent_w),
            generator=gen, dtype=torch.float32,
        ).numpy()
        # Pack patches: mirror of QwenImagePipeline._pack_latents
        # which does: latents.view(B, C, H/2, 2, W/2, 2)
        #         -> permute(0, 2, 4, 1, 3, 5)
        #         -> reshape(B, (H/2)*(W/2), C*4).
        b, c, h, w = latents.shape
        v = latents.reshape(b, c, h // 2, 2, w // 2, 2)
        v = v.transpose(0, 2, 4, 1, 3, 5)
        packed = v.reshape(b, (h // 2) * (w // 2), c * 4)
        return packed.astype(np.float32)

    def _run_denoiser(
        self,
        img_packed: np.ndarray,
        txt_hidden: np.ndarray,
        timestep_norm: float,
    ) -> np.ndarray:
        """One denoiser forward.

        Inputs / outputs all packed-token shaped [1, n_img, in_ch_packed].
        ``timestep_norm`` is in [0, 1] (the engine scales by 1000 internally
        in the sinusoidal timestep embedding).
        """
        ts = np.asarray([float(timestep_norm)], dtype=np.float32)
        out = self._run_engine(
            self._dit_engine, self._dit_ctx,
            inputs={
                "img_patched": img_packed.astype(np.float32),
                "txt_hidden": txt_hidden.astype(np.float32),
                "timestep": ts,
            },
            output_names=["noise_patched"],
        )
        return out["noise_patched"]

    def _vae_decode(self, latents_packed: np.ndarray) -> np.ndarray:
        """Un-patchify, un-normalise, then VAE-decode.

        Inputs: ``latents_packed`` shape ``[1, n_img, in_ch_packed]``.
        Returns: image array ``[3, H, W]`` uint8 in [0, 255].
        """
        # 1) Unpack patches: (B, n_img, C*4) -> (B, C, 1, H_lat, W_lat).
        b, num_patches, ch_packed = latents_packed.shape
        if b != 1:
            raise NotImplementedError("batch>1 not supported")
        h_packed = self._packed_h
        w_packed = self._packed_w
        c_unpacked = ch_packed // 4
        # Mirror QwenImagePipeline._unpack_latents:
        # (B, n_patches, ch*4) -> (B, h_pack, w_pack, ch, 2, 2)
        v = latents_packed.reshape(
            b, h_packed, w_packed, c_unpacked, 2, 2
        )
        # permute (0, 3, 1, 4, 2, 5)
        v = v.transpose(0, 3, 1, 4, 2, 5)
        # reshape (B, ch, 1, H_lat, W_lat)
        v = v.reshape(b, c_unpacked, 1, h_packed * 2, w_packed * 2)

        # 2) Un-normalise per channel (the bundle stores raw
        # vae.config.latents_std and latents_mean — diffusers internally does
        # latents_std = 1/raw, then z = z / inverted + mean which is
        # equivalent to z * raw + mean).
        mean = self._preprocessor["latents_mean"].astype(np.float32)
        std = self._preprocessor["latents_std"].astype(np.float32)
        if mean.shape[0] != c_unpacked or std.shape[0] != c_unpacked:
            raise ValueError(
                f"latents_mean/std channel mismatch: "
                f"got {mean.shape} vs c={c_unpacked}"
            )
        m = mean.reshape(1, c_unpacked, 1, 1, 1)
        s = std.reshape(1, c_unpacked, 1, 1, 1)
        latents5d = (v.astype(np.float32) * s) + m

        # 3) VAE decode.
        out = self._run_engine(
            self._vae_engine, self._vae_ctx,
            inputs={"latent": latents5d},
            output_names=["image"],
        )
        image = out["image"]  # [1, 3, 1, H, W], range ~[-1, 1]
        if image.ndim == 5:
            image = image[0, :, 0]  # [3, H, W]
        elif image.ndim == 4:
            image = image[0]  # [3, H, W]
        else:
            raise ValueError(f"unexpected VAE image shape {image.shape}")

        # 4) Convert [-1, 1] -> uint8 [0, 255].
        image = np.clip(image, -1.0, 1.0)
        img_u8 = np.rint((image + 1.0) * 127.5).astype(np.uint8)
        return img_u8

    # ------------------------------------------------------------------
    # Public entry point.
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        negative_prompt: str = " ",
        num_inference_steps: int = 50,
        cfg_scale: float = 4.0,
        height: int | None = None,
        width: int | None = None,
        seed: int = 42,
    ) -> np.ndarray:
        """Run the full T2I pipeline and return a ``[3, H, W]`` uint8 image.

        ``height`` / ``width`` MUST match the engine's baked latent grid
        (1024x1024 for the shipped bundle). If left None, defaults are read
        from the bundle's ``image.default_height`` / ``image.default_width``.
        """
        if height is None:
            height = int(self.config["image"]["default_height"])
        if width is None:
            width = int(self.config["image"]["default_width"])

        expected_h = self._latent_h * 8
        expected_w = self._latent_w * 8
        if height != expected_h or width != expected_w:
            raise ValueError(
                f"height/width must match the engine's baked grid; got "
                f"{height}x{width}, expected {expected_h}x{expected_w}"
            )

        # 1) Encode positive + negative prompts.
        pos_hidden = self._encode_prompt(prompt)
        neg_hidden = self._encode_prompt(negative_prompt)

        import torch  # lazy: torch is only required for the diffusion path
        # 2) Prepare scheduler. Use diffusers FlowMatchEulerDiscreteScheduler
        # to match the reference exactly (dynamic mu shifting, time_shift_type,
        # set_begin_index, etc. — too many edge cases to reimplement).
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

        sched_cfg = self._sched_cfg
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=int(sched_cfg.get("num_train_timesteps", 1000)),
            shift=float(sched_cfg.get("shift", 1.0)),
            use_dynamic_shifting=bool(
                sched_cfg.get("use_dynamic_shifting", True)),
            base_image_seq_len=int(sched_cfg.get("base_image_seq_len", 256)),
            max_image_seq_len=int(
                sched_cfg.get("max_image_seq_len", 8192)),
            base_shift=0.5,
            max_shift=1.15,
            shift_terminal=None,
            time_shift_type="exponential",
        )

        # mu shift matching pipeline_qwenimage.py:calculate_shift().
        def _calc_mu(image_seq_len: int) -> float:
            base = sched_cfg.get("base_image_seq_len", 256)
            mx = sched_cfg.get("max_image_seq_len", 4096)
            base_shift = 0.5
            max_shift = 1.15
            m = (max_shift - base_shift) / (mx - base)
            b = base_shift - m * base
            return image_seq_len * m + b

        mu = _calc_mu(self._n_img)
        sigmas = np.linspace(
            1.0, 1.0 / num_inference_steps, num_inference_steps
        )
        scheduler.set_timesteps(
            sigmas=sigmas.tolist(), mu=mu, device="cpu",
        )
        scheduler.set_begin_index(0)

        # 3) Sample initial latents (packed).
        latents = self._prepare_latents(seed)
        # diffusers passes ``torch.Tensor`` to scheduler.step. We keep a
        # torch copy for the scheduler interaction and a numpy copy for
        # engine I/O — converting per-step is cheap relative to a denoiser
        # forward.
        latents_t = torch.from_numpy(latents).to(torch.float32)

        timesteps = scheduler.timesteps
        do_cfg = cfg_scale > 1.0 and negative_prompt is not None

        for i, t in enumerate(timesteps):
            t_norm = float(t.item()) / 1000.0
            latents_np = latents_t.numpy().astype(np.float32)

            noise_pos = self._run_denoiser(
                latents_np, pos_hidden, t_norm,
            )
            if do_cfg:
                noise_neg = self._run_denoiser(
                    latents_np, neg_hidden, t_norm,
                )
                # Diffusers true-CFG:
                #   comb = neg + cfg * (pos - neg)
                #   noise = comb * (||pos|| / ||comb||)   (per-token renorm)
                comb = noise_neg + cfg_scale * (noise_pos - noise_neg)
                pos_norm = np.linalg.norm(
                    noise_pos, axis=-1, keepdims=True
                )
                comb_norm = np.linalg.norm(comb, axis=-1, keepdims=True)
                # Avoid division by zero.
                comb_norm = np.maximum(comb_norm, 1e-8)
                noise = comb * (pos_norm / comb_norm)
            else:
                noise = noise_pos

            noise_t = torch.from_numpy(noise.astype(np.float32))
            latents_t = scheduler.step(
                noise_t, t, latents_t, return_dict=False,
            )[0]
            latents_t = latents_t.to(torch.float32)

        # 4) VAE-decode the final packed latents.
        final_packed = latents_t.numpy().astype(np.float32)
        return self._vae_decode(final_packed)

    # ------------------------------------------------------------------
    # Cleanup.
    # ------------------------------------------------------------------

    def __del__(self):
        # Best-effort cleanup; called during interpreter shutdown.
        if cudart is None:
            return
        stream = getattr(self, "stream", None)
        if stream is not None:
            try:
                cudart.cudaStreamDestroy(stream)
            except Exception:  # noqa: BLE001 - shutdown path
                pass
        tok_dir = getattr(self, "_tokenizer_dir", None)
        if tok_dir is not None:
            import shutil
            try:
                shutil.rmtree(tok_dir, ignore_errors=True)
            except Exception:  # noqa: BLE001 - shutdown path
                pass
