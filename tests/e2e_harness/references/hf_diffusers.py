"""HF Diffusers reference backend.

Runs HuggingFace diffusers pipeline as a subprocess for GPU isolation,
producing per-stage reference outputs for comparison against TRT.

Supports Wan-style text-to-video pipelines with per-stage extraction
(T5 encoding, single DiT step, full denoising loop, VAE decode).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .. import _case_artifact_dir
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[3]


def _ref_subprocess_env() -> dict:
    """Build env for HF reference subprocesses.

    On GB300, torch's bundled cuBLAS is incompatible with the driver,
    causing CUBLAS_STATUS_INVALID_VALUE on every matmul. We LD_PRELOAD
    the system cuBLAS to fix this.
    """
    env = os.environ.copy()
    sys_cublas = "/usr/local/cuda/lib64/libcublas.so.13"
    sys_cublaslt = "/usr/local/cuda/lib64/libcublasLt.so.13"
    if os.path.exists(sys_cublas) and os.path.exists(sys_cublaslt):
        existing = env.get("LD_PRELOAD", "")
        preload = f"{sys_cublas}:{sys_cublaslt}"
        env["LD_PRELOAD"] = f"{preload}:{existing}" if existing else preload
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def _resolve_cached_model_ref(hf_id: str) -> str:
    """Prefer a local snapshot and patch tokenizer configs incompatible with current transformers."""
    if not hf_id:
        return hf_id
    local_path = Path(hf_id)
    if local_path.exists():
        return hf_id
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(hf_id, local_files_only=True))
    except Exception:
        return hf_id

    tok_cfg = snapshot / "tokenizer" / "tokenizer_config.json"
    if not tok_cfg.exists():
        return str(snapshot)

    try:
        cfg = json.loads(tok_cfg.read_text())
    except Exception:
        return str(snapshot)

    if not isinstance(cfg.get("extra_special_tokens"), list):
        return str(snapshot)

    patched_root = Path(tempfile.gettempdir()) / "trtmc_hf_patched" / hashlib.sha256(
        str(snapshot).encode("utf-8")).hexdigest()
    patched_cfg = patched_root / "tokenizer" / "tokenizer_config.json"
    if not patched_cfg.exists():
        shutil.copytree(snapshot, patched_root, dirs_exist_ok=True)
        patched = json.loads(patched_cfg.read_text())
        patched.pop("extra_special_tokens", None)
        patched_cfg.write_text(json.dumps(patched, indent=2))
    return str(patched_root)


def _resolve_input_path(path: str | None, ctx: RunContext) -> str | None:
    """Resolve e2e asset paths against engine, repo, and tests/e2e roots."""
    if not path:
        return None
    p = Path(path)
    if p.is_absolute():
        return str(p)
    for base in (ctx.engine_dir, str(PROJECT_DIR), str(PROJECT_DIR / "tests" / "e2e")):
        candidate = Path(base) / path
        if candidate.is_file():
            return str(candidate)
    return str(p)


def _ltx_initial_latents_path(case: E2ECase, ctx: RunContext) -> str:
    if ctx.artifacts_dir:
        base_dir = _case_artifact_dir(ctx.artifacts_dir, case.name)
    else:
        base_dir = os.path.join(tempfile.gettempdir(), "trtmc_ltx_latents", case.name)
    return os.path.join(base_dir, "initial_latents.raw")


def _qwen_image_initial_latents_path(case: E2ECase, ctx: RunContext) -> str:
    """Mirror of the runner-side helper. Both subprocesses MUST point at the
    same path; the runner writes the raw fp32 bytes, the HF reference reads
    them back so both pipelines start from identical noise (E2E shared-latents
    path, mirrors the LTX precedent)."""
    if ctx.artifacts_dir:
        base_dir = _case_artifact_dir(ctx.artifacts_dir, case.name)
    else:
        base_dir = os.path.join(
            tempfile.gettempdir(), "trtmc_qwen_image_latents", case.name)
    return os.path.join(base_dir, "initial_latents.raw")


class HfDiffusersReference:
    """Reference backend using HuggingFace diffusers pipelines."""

    @property
    def backend_name(self) -> str:
        return "hf_diffusers"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        dispatch = {
            "t5_encode": self._run_t5_encode,
            "dit_step": self._run_dit_step,
            "end_to_end": self._run_full_pipeline,
            "end_to_end_video": self._run_full_pipeline,
            "generate": self._run_full_pipeline,
            "debug_pipeline": self._run_debug_pipeline,
            "vae_decode": self._run_full_pipeline,
            "frame_quality": self._run_full_pipeline,
            "crossover_ref_t5_trt_dit": self._run_crossover_noop,
            "crossover_trt_t5_ref_dit": self._run_crossover_noop,
        }
        handler = dispatch.get(stage.name)
        if handler is None:
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Unknown diffusers reference stage: {stage.name}"},
            )
        return handler(case, stage, ctx)

    def _run_t5_encode(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF T5 encoding and save output for comparison.

        Supports Wan (WanPipeline), Flux (FluxPipeline), and generic
        diffusers pipelines. Falls back gracefully if the pipeline class
        is not available.
        """
        model_id = case.hf_id
        model_ref = _resolve_cached_model_ref(model_id)
        prompt = case.inputs.get("prompt", "A cat sitting on a beach")
        python = ctx.reference_python_path() or sys.executable
        max_length = 120 if case.family == "pixart" else 512
        model_type = str(case.metadata.get("model_type", "")).lower()

        # Save to artifacts_dir so the file persists for comparator access
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        os.makedirs(model_dir, exist_ok=True)
        output_path = os.path.join(model_dir, "hf_t5_output.npy")

        script = f"""
import torch, numpy as np, sys
import transformers

model_id = {model_id!r}
model_ref = {model_ref!r}
prompt = {prompt!r}
output_path = {output_path!r}
max_length = {max_length}

transformers.logging.set_verbosity_error()


def _retie_wan_text_encoder(pipe):
    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is None:
        return
    shared = getattr(text_encoder, "shared", None)
    encoder = getattr(text_encoder, "encoder", None)
    embed_tokens = getattr(encoder, "embed_tokens", None) if encoder is not None else None
    if shared is None or embed_tokens is None:
        return
    if hasattr(text_encoder, "tie_weights"):
        text_encoder.tie_weights()
    if shared.weight.shape == embed_tokens.weight.shape and shared.weight.data_ptr() != embed_tokens.weight.data_ptr():
        # Some transformers versions report embed_tokens as missing even though
        # it should be tied to shared; enforce tying explicitly.
        text_encoder.encoder.embed_tokens = text_encoder.shared
    if shared.weight.data_ptr() != text_encoder.encoder.embed_tokens.weight.data_ptr():
        raise RuntimeError("Wan text_encoder embeddings are not tied after load")


# Try pipeline classes in order: Wan, Flux, generic DiffusionPipeline
pipe = None
pipeline_order = ["WanPipeline", "FluxPipeline", "DiffusionPipeline"]
if {model_type!r} in ("flux.2", "flux2"):
    pipeline_order = ["Flux2Pipeline", "WanPipeline", "FluxPipeline", "DiffusionPipeline"]
for cls_name in pipeline_order:
    try:
        import diffusers
        diffusers.logging.set_verbosity_error()
        cls = getattr(diffusers, cls_name, None)
        if cls is None:
            continue
        load_kwargs = dict(torch_dtype=torch.float32, low_cpu_mem_usage=True)
        if cls_name == "WanPipeline":
            # Prefer full materialization for robust tied-embedding loading.
            # Newer diffusers can reject this for models with keep_in_fp32_modules,
            # so we fall back to low_cpu_mem_usage=True.
            try:
                pipe = cls.from_pretrained(
                    model_ref, torch_dtype=torch.float32, low_cpu_mem_usage=False)
            except ValueError as e:
                if "keep_in_fp32_modules" not in str(e):
                    raise
                pipe = cls.from_pretrained(model_ref, **load_kwargs)
        else:
            pipe = cls.from_pretrained(model_ref, **load_kwargs)
        if cls_name == "WanPipeline":
            _retie_wan_text_encoder(pipe)
        print(f"Loaded {{cls_name}}", file=sys.stderr)
        break
    except Exception as e:
        print(f"{{cls_name}} failed: {{e}}", file=sys.stderr)
        continue

if pipe is None:
    print("ERROR: no diffusers pipeline could load this model", file=sys.stderr)
    sys.exit(1)

# Extract text encoder and tokenizer (different pipelines use different names)
text_encoder = getattr(pipe, "text_encoder", None)
if text_encoder is None:
    text_encoder = getattr(pipe, "text_encoder_2", None)

tokenizer = getattr(pipe, "tokenizer", None)
if tokenizer is None:
    tokenizer = getattr(pipe, "tokenizer_2", None)

if text_encoder is None or tokenizer is None:
    print("ERROR: pipeline has no text_encoder/tokenizer", file=sys.stderr)
    sys.exit(1)

tokens = tokenizer(prompt, return_tensors="pt", padding="max_length",
                    max_length=max_length, truncation=True)
with torch.no_grad():
    t5_out = text_encoder(
        input_ids=tokens.input_ids,
        attention_mask=tokens.attention_mask,
    )[0]

np.save(output_path, t5_out.numpy())
print(f"shape={{list(t5_out.shape)}}")
print(f"mean={{float(t5_out.mean()):.6f}}")
"""
        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=600,
            env=_ref_subprocess_env())
        elapsed = time.monotonic() - t0

        data: dict = {
            "returncode": result.returncode,
            "stdout": result.stdout,
        }
        if os.path.exists(output_path):
            data["output_path"] = output_path

        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=result.stdout,
            timing_s=elapsed,
            metadata={"backend": "hf_diffusers"},
        )

    def _run_dit_step(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF DiT forward pass for comparison."""
        # This is implicitly compared in the debug_pipeline flow
        return self._run_debug_pipeline(case, stage, ctx)

    def _run_debug_pipeline(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """The debug_pipeline script runs BOTH TRT and HF internally.

        For the reference side, we report the HF results from the same script.
        The diffusion comparator handles the joint output.
        """
        # Return a marker output indicating that debug_pipeline handles
        # both sides. The comparator will use the TRT runner's output
        # which contains both TRT and HF comparison results.
        return StageOutput(
            stage_name=stage.name,
            data={
                "backend": "hf_diffusers",
                "note": "debug_pipeline runs HF internally; comparison embedded in TRT output",
            },
            timing_s=0.0,
            metadata={"backend": "hf_diffusers"},
        )

    def _run_crossover_noop(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Crossover stages are self-contained in the runner (mix TRT + HF).

        The runner subprocess handles both the HF and TRT components itself,
        so the reference side returns a no-op marker.
        """
        return StageOutput(
            stage_name=stage.name,
            data={
                "backend": "hf_diffusers",
                "note": "Crossover stage is self-contained in runner; "
                        "reference is embedded in the runner subprocess",
            },
            timing_s=0.0,
            metadata={"backend": "hf_diffusers"},
        )

    def _run_full_pipeline(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run full HF diffusers pipeline to generate reference frames."""
        if case.family == "sana_wm":
            return self._run_sana_wm_pipeline(case, stage, ctx)

        model_id = case.hf_id
        model_ref = _resolve_cached_model_ref(model_id)
        prompt = case.inputs.get("prompt", "A cat sitting on a beach")
        num_steps = case.inputs.get("num_inference_steps", 30)
        image_height = case.inputs.get("image_height", 1024)
        image_width = case.inputs.get("image_width", image_height)
        video_height = case.inputs.get("video_height", 480)
        video_width = case.inputs.get("video_width", 832)
        video_num_frames = case.inputs.get("video_num_frames", 17)
        python = ctx.reference_python_path() or sys.executable
        ltx_initial_latents_raw = _ltx_initial_latents_path(case, ctx)
        qwen_image_initial_latents_raw = _qwen_image_initial_latents_path(case, ctx)
        model_type = str(case.metadata.get("model_type", "")).lower()

        # Save frames to artifacts_dir so they persist for comparator access
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        frames_dir = os.path.join(model_dir, "hf_frames")
        os.makedirs(frames_dir, exist_ok=True)

        family = case.family
        # Qwen-Image (and forward-compat Edit-mode variants) drive the HF
        # reference via ``diffusers.QwenImagePipeline`` with ``true_cfg_scale``
        # (NOT ``guidance_scale``). Manifest provides ``negative_prompt``,
        # ``cfg_scale``, ``height``, ``width``, ``num_inference_steps``,
        # ``seed`` — mirror what the TRT runner emits to ``trtmc run``.
        qi_negative_prompt = case.inputs.get("negative_prompt", " ")
        qi_cfg_scale = float(
            case.inputs.get("cfg_scale", case.inputs.get("guidance_scale", 4.0)))
        qi_height = int(
            case.inputs.get("height", case.inputs.get("image_height", image_height)))
        qi_width = int(
            case.inputs.get("width", case.inputs.get("image_width", image_width)))
        script = f"""
import torch
import numpy as np
from PIL import Image
import os
import transformers

transformers.logging.set_verbosity_error()

family = {family!r}
model_type = {model_type!r}
model_id = {model_id!r}
model_ref = {model_ref!r}
prompt = {prompt!r}
num_steps = {num_steps}
image_height = {image_height}
image_width = {image_width}
video_height = {video_height}
video_width = {video_width}
video_num_frames = {video_num_frames}
frames_dir = {frames_dir!r}
seed = {int(case.inputs.get("seed", case.determinism.get("seed", 42)))}
ltx_guidance_scale = {float(case.inputs.get("guidance_scale", 3.0))}
wan_guidance_scale = {float(case.inputs.get("guidance_scale", 5.0))}
z_image_guidance_scale = {float(case.inputs.get("guidance_scale", 0.0))}
qi_negative_prompt = {qi_negative_prompt!r}
qi_cfg_scale = {qi_cfg_scale}
qi_height = {qi_height}
qi_width = {qi_width}
ltx_initial_latents_raw = {ltx_initial_latents_raw!r}
qwen_image_initial_latents_raw = {qwen_image_initial_latents_raw!r}

if family in ("qwen_image",):
    # Text-to-image via QwenImagePipeline. The class lookup forward-compats
    # to QwenImageEditPipeline / QwenImageEditPlusPipeline so a future Edit
    # manifest just works after wiring an ``image`` input.
    import diffusers
    diffusers.logging.set_verbosity_error()
    pipeline_cls = None
    for cls_name in ("QwenImageEditPlusPipeline", "QwenImageEditPipeline",
                     "QwenImagePipeline"):
        cls = getattr(diffusers, cls_name, None)
        if cls is None:
            continue
        # Pick Edit variants only when the manifest passes an image input;
        # for plain T2I, fall through to QwenImagePipeline.
        if "Edit" in cls_name and not {bool(case.inputs.get("image"))}:
            continue
        pipeline_cls = cls
        break
    if pipeline_cls is None:
        raise RuntimeError(
            "diffusers does not expose QwenImagePipeline; upgrade diffusers")
    # bf16 matches the PSNR-validated Python E2E gate.
    pipe = pipeline_cls.from_pretrained(model_ref, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    # Shared-initial-latents path: load the raw fp32 bytes written by the
    # runner-side helper, reshape to UNPACKED [1, 16, h_lat, w_lat], call
    # the pipeline's static ``_pack_latents`` to produce the packed
    # [1, n_img, 64] tensor diffusers expects when ``latents=`` is supplied
    # to ``__call__``. Both subprocesses thereby start from byte-identical
    # noise — eliminates the std::mt19937 vs torch.Generator divergence.
    qi_latents = None
    if os.path.exists(qwen_image_initial_latents_raw):
        vae_scale = 8
        latent_channels = 16
        h_lat = qi_height // vae_scale
        w_lat = qi_width // vae_scale
        # diffusers' ``prepare_latents`` rounds latent H/W to a multiple
        # of 2 because the DiT packs 2x2 spatial blocks; mirror that to
        # match the dimensions ``_pack_latents`` will operate on.
        pack_h_lat = 2 * (h_lat // 2)
        pack_w_lat = 2 * (w_lat // 2)
        raw = np.fromfile(qwen_image_initial_latents_raw, dtype=np.float32)
        expected_size = latent_channels * pack_h_lat * pack_w_lat
        if raw.size != expected_size:
            raise RuntimeError(
                f"Qwen-Image shared latents size {{raw.size}} does not "
                f"match expected [1, {{latent_channels}}, {{pack_h_lat}}, "
                f"{{pack_w_lat}}] = {{expected_size}}")
        # _pack_latents expects [B, C, H, W] (collapses internally to the
        # ``height // 2, 2, width // 2, 2`` grid before permute+reshape).
        unpacked = torch.from_numpy(raw).view(
            1, latent_channels, pack_h_lat, pack_w_lat).to(
                device="cuda", dtype=torch.bfloat16)
        qi_latents = pipeline_cls._pack_latents(
            unpacked, 1, latent_channels, pack_h_lat, pack_w_lat)
    output = pipe(
        prompt=prompt,
        negative_prompt=qi_negative_prompt,
        true_cfg_scale=qi_cfg_scale,
        num_inference_steps=num_steps,
        height=qi_height,
        width=qi_width,
        latents=qi_latents,
        generator=torch.Generator("cuda").manual_seed(seed),
    )
    frames = output.images
elif family in ("flux",):
    if model_type in ("flux.2", "flux2"):
        from diffusers import Flux2Pipeline
        # FLUX.2's model card recommends bf16 for diffusers inference; full
        # float32 can OOM on CI runners after TensorRT engine rebuilds.
        flux_dtype = torch.bfloat16
        pipe = Flux2Pipeline.from_pretrained(
            model_ref, torch_dtype=flux_dtype, low_cpu_mem_usage=True)
    else:
        from diffusers import FluxPipeline
        # Keep legacy FLUX.1 reference in float32; bf16/fp16 can trip GB300
        # cuBLAS issues in older diffusers/torch combinations.
        flux_dtype = torch.float32
        pipe = FluxPipeline.from_pretrained(model_ref, torch_dtype=flux_dtype)
    pipe.to("cuda")
    kwargs = dict(
        prompt=prompt,
        num_inference_steps=num_steps,
        height=image_height, width=image_width,
        generator=torch.Generator("cuda").manual_seed(seed),
    )
    if model_type in ("flux.2", "flux2"):
        kwargs["guidance_scale"] = 3.5
    output = pipe(**kwargs)
    frames = output.images
elif family in ("z_image",):
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(
        model_ref, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
    pipe.to("cuda")
    output = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        height=image_height, width=image_width,
        guidance_scale=z_image_guidance_scale,
        generator=torch.Generator("cuda").manual_seed(seed),
    )
    frames = output.images
elif family in ("pixart",):
    from diffusers import PixArtSigmaPipeline
    pipe = PixArtSigmaPipeline.from_pretrained(model_ref, torch_dtype=torch.float32)
    pipe.to("cuda")
    output = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        height=image_height, width=image_width,
        generator=torch.Generator("cuda").manual_seed(seed),
    )
    frames = output.images
elif family in ("ltx_video",):
    from diffusers import LTXPipeline
    pipe = LTXPipeline.from_pretrained(model_ref, torch_dtype=torch.float32)
    pipe.to("cuda")
    ltx_latents = None
    if os.path.exists(ltx_initial_latents_raw):
        packed = np.fromfile(ltx_initial_latents_raw, dtype=np.float32)
        channels = int(pipe.transformer.config.in_channels)
        if packed.size % channels != 0:
            raise RuntimeError(
                f"invalid LTX initial latent size {{packed.size}} for {{channels}} channels")
        ltx_latents = torch.from_numpy(
            packed.reshape(1, packed.size // channels, channels)).to(
                device="cuda", dtype=torch.float32)
    output = pipe(
        prompt=prompt,
        negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
        num_inference_steps=num_steps,
        height=video_height,
        width=video_width,
        num_frames=video_num_frames,
        guidance_scale=ltx_guidance_scale,
        latents=ltx_latents,
        generator=torch.Generator("cuda").manual_seed(seed),
    )
    frames = output.frames[0]
else:
    # Default: Wan-style text-to-video
    import ftfy  # noqa: F401 — required by WanPipeline prompt cleaning
    from diffusers import WanPipeline
    try:
        pipe = WanPipeline.from_pretrained(
            model_ref, torch_dtype=torch.float32, low_cpu_mem_usage=False)
    except ValueError as e:
        if "keep_in_fp32_modules" not in str(e):
            raise
        pipe = WanPipeline.from_pretrained(
            model_ref, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    if hasattr(pipe, "text_encoder"):
        te = pipe.text_encoder
        if hasattr(te, "tie_weights"):
            te.tie_weights()
        shared = getattr(te, "shared", None)
        embed = getattr(getattr(te, "encoder", None), "embed_tokens", None)
        if shared is not None and embed is not None and shared.weight.shape == embed.weight.shape:
            if shared.weight.data_ptr() != embed.weight.data_ptr():
                te.encoder.embed_tokens = te.shared
    pipe.to("cuda")
    output = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        height=video_height,
        width=video_width,
        num_frames=video_num_frames,
        guidance_scale=wan_guidance_scale,
        generator=torch.Generator("cuda").manual_seed(seed),
    )
    frames = output.frames[0]

for i, frame in enumerate(frames):
    if isinstance(frame, Image.Image):
        frame.save(os.path.join(frames_dir, f"frame_{{i:04d}}.png"))
    else:
        img = Image.fromarray(np.uint8(frame * 255) if frame.max() <= 1.0 else np.uint8(frame))
        img.save(os.path.join(frames_dir, f"frame_{{i:04d}}.png"))

print(f"Generated {{len(frames)}} frames")
"""
        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=3600,
            env=_ref_subprocess_env())
        elapsed = time.monotonic() - t0

        frame_files = sorted(Path(frames_dir).glob("frame_*.png"))

        # Persist stderr for debugging
        if result.stderr:
            stderr_path = os.path.join(model_dir, "hf_diffusion_full_pipeline_stderr.log")
            try:
                with open(stderr_path, "w") as f:
                    f.write(result.stderr)
            except OSError:
                pass
            if result.returncode != 0:
                logger.error("HF diffusers full pipeline failed (rc=%d): %s",
                             result.returncode, result.stderr[-500:])

        data: dict = {
            "returncode": result.returncode,
            "num_frames": len(frame_files),
            "frames_dir": frames_dir,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

        # Compute frame statistics for reference
        if frame_files:
            try:
                import numpy as np
                from PIL import Image

                all_pixels = []
                for fp in frame_files:
                    img = Image.open(fp).convert("RGB")
                    arr = np.array(img, dtype=np.float32) / 255.0
                    all_pixels.append(arr.flatten())
                combined = np.concatenate(all_pixels)
                data["frame_stats"] = {
                    "count": len(frame_files),
                    "mean": float(np.mean(combined)),
                    "std": float(np.std(combined)),
                    "min": float(np.min(combined)),
                    "max": float(np.max(combined)),
                }
            except Exception as e:
                data["frame_stats_error"] = str(e)

        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=result.stdout,
            timing_s=elapsed,
            metadata={"backend": "hf_diffusers"},
        )

    def _run_sana_wm_pipeline(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run SANA-WM through the official script used as the oracle."""
        prompt = case.inputs.get("prompt", "")
        prompt_file = _resolve_input_path(case.inputs.get("prompt_file"), ctx)
        if prompt_file and os.path.isfile(prompt_file):
            prompt = Path(prompt_file).read_text(encoding="utf-8")

        image_path = _resolve_input_path(
            case.inputs.get("image")
            or case.inputs.get("test_image")
            or case.inputs.get("image_path"),
            ctx,
        )

        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        frames_dir = os.path.join(model_dir, "hf_frames")
        output_dir = os.path.join(model_dir, "hf_sana_wm_output")
        shutil.rmtree(frames_dir, ignore_errors=True)
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        if prompt_file and os.path.isfile(prompt_file):
            prompt_path = prompt_file
        else:
            prompt_path = os.path.join(model_dir, "hf_sana_wm_prompt.txt")
            Path(prompt_path).write_text(prompt, encoding="utf-8")

        python = ctx.reference_python_path() or sys.executable
        script = PROJECT_DIR / "inference_video_scripts" / "inference_sana_wm.py"
        cmd = [
            python, str(script),
            "--prompt", str(prompt_path),
            "--output_dir", frames_dir,
            "--translation_speed", str(case.inputs.get("translation_speed", 0.055)),
            "--rotation_speed_deg", str(case.inputs.get("rotation_speed_deg", 1.2)),
            "--num_frames", str(case.inputs.get("video_num_frames", 321)),
        ]
        num_steps = (
            case.inputs.get("num_inference_steps")
            or case.inputs.get("num_steps")
            or case.inputs.get("step")
        )
        if num_steps is not None:
            cmd.extend(["--step", str(num_steps)])
        cfg_scale = case.inputs.get("cfg_scale")
        if cfg_scale is None:
            cfg_scale = case.inputs.get("guidance_scale")
        if cfg_scale is not None:
            cmd.extend(["--cfg_scale", str(cfg_scale)])
        camera_path = _resolve_input_path(
            case.inputs.get("camera") or case.inputs.get("camera_path"), ctx
        )
        if camera_path:
            cmd.extend(["--camera", camera_path])
        else:
            cmd.extend([
                "--action",
                str(case.inputs.get("action", "w-80,jw-40,w-40,lw-60,w-100")),
            ])
        intrinsics = case.inputs.get("camera_intrinsics")
        if intrinsics is None:
            intrinsics = case.inputs.get("intrinsics")
        if intrinsics is not None:
            if isinstance(intrinsics, str):
                value = _resolve_input_path(intrinsics, ctx) or intrinsics
            elif isinstance(intrinsics, (list, tuple)):
                value = ",".join(str(v) for v in intrinsics)
            else:
                value = str(intrinsics)
            cmd.extend(["--intrinsics", value])
        fps = case.inputs.get("fps")
        if fps is not None:
            cmd.extend(["--fps", str(fps)])
        flow_shift = case.inputs.get("flow_shift")
        if flow_shift is not None:
            cmd.extend(["--flow_shift", str(flow_shift)])
        if image_path:
            cmd.extend(["--image", image_path])
        else:
            cmd.extend(["--image", ""])
        if case.inputs.get("no_action_overlay"):
            cmd.append("--no_action_overlay")
        if case.inputs.get("no_refiner"):
            cmd.append("--no_refiner")
        env = _ref_subprocess_env()
        sana_script = case.inputs.get("sana_wm_script")
        if sana_script:
            env["SANA_WM_SCRIPT"] = str(sana_script)

        t0 = time.monotonic()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600,
            env=env)
        elapsed = time.monotonic() - t0

        frame_files = sorted(Path(frames_dir).glob("frame_*.png"))
        if result.stderr:
            stderr_path = os.path.join(model_dir, "hf_sana_wm_stderr.log")
            try:
                with open(stderr_path, "w") as f:
                    f.write(result.stderr)
            except OSError:
                pass
            if result.returncode != 0:
                logger.error("HF SANA-WM pipeline failed (rc=%d): %s",
                             result.returncode, result.stderr[-500:])

        data: dict = {
            "returncode": result.returncode,
            "num_frames": len(frame_files),
            "frames_dir": frames_dir,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

        if frame_files:
            try:
                import numpy as np
                from PIL import Image

                all_pixels = []
                for fp in frame_files:
                    img = Image.open(fp).convert("RGB")
                    arr = np.array(img, dtype=np.float32) / 255.0
                    all_pixels.append(arr.flatten())
                combined = np.concatenate(all_pixels)
                data["frame_stats"] = {
                    "count": len(frame_files),
                    "mean": float(np.mean(combined)),
                    "std": float(np.std(combined)),
                    "min": float(np.min(combined)),
                    "max": float(np.max(combined)),
                }
            except Exception as e:
                data["frame_stats_error"] = str(e)

        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=result.stdout,
            timing_s=elapsed,
            metadata={"backend": "hf_diffusers", "command": cmd},
        )


plugin = HfDiffusersReference()
