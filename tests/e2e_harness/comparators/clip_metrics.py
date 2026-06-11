"""CLIP-based semantic metrics for image diffusion reference comparison.

Provides reference-relative semantic parity that works WITHOUT shared
initial latents (unlike PSNR/SSIM). Captures prompt fidelity and
image-image semantic similarity.

Intentionally blind to fine perceptual quality — MUST be paired with
PSNR/SSIM (shared-latent models) or LPIPS for full quality coverage.

Scope: text-to-image and image-to-image models only.
       Video models are excluded (video_num_frames > 1).
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Pin model identity here so CI scores never drift silently across upgrades.
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

# CLIP text encoder context length. Prompts longer than this are truncated.
CLIP_MAX_TOKENS = 77

# Number of frames sampled per directory when computing metrics.
_DEFAULT_MAX_FRAMES = 4


class ClipMetrics(NamedTuple):
    trt_prompt_clipscore: float
    hf_prompt_clipscore: float
    prompt_clipscore_delta: float
    trt_hf_image_clip_cosine: float
    prompt_truncated: bool


@functools.lru_cache(maxsize=1)
def _load_clip():
    """Lazy-load open_clip on CPU / fp32. Returns bundle or None on ImportError."""
    try:
        import open_clip
    except ImportError:
        logger.warning(
            "open_clip not installed — CLIP metrics will be skipped. "
            "Install with: pip install open-clip-torch"
        )
        return None

    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED
    )
    # fp32 + eval for determinism across environments.
    model = model.eval().float()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    logger.debug("Loaded CLIP %s / %s", CLIP_MODEL, CLIP_PRETRAINED)
    return model, preprocess, tokenizer


def _embed_images(paths: list[Path]):
    """Return L2-normalised image embeddings tensor [N, D] or None."""
    bundle = _load_clip()
    if bundle is None or not paths:
        return None
    import torch
    from PIL import Image

    model, preprocess, _ = bundle
    tensors = []
    for p in paths:
        try:
            tensors.append(preprocess(Image.open(p).convert("RGB")))
        except Exception as exc:
            logger.warning("CLIP: could not open %s: %s", p, exc)
    if not tensors:
        return None
    with torch.no_grad():
        batch = torch.stack(tensors)
        emb = model.encode_image(batch).float()
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb  # [N, D]


def _embed_text(prompt: str) -> tuple:
    """Return (L2-normalised text embedding [D], was_truncated: bool) or (None, False)."""
    bundle = _load_clip()
    if bundle is None:
        return None, False
    import torch

    model, _, tokenizer = bundle
    # Detect truncation by encoding with a large context length and comparing.
    full_tokens = tokenizer([prompt], context_length=10_000)[0]
    truncated = int((full_tokens != 0).sum()) > CLIP_MAX_TOKENS

    with torch.no_grad():
        tok = tokenizer([prompt])
        emb = model.encode_text(tok).float()
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0], truncated  # [D], bool


def compute_clip_metrics(
    trt_dir: str,
    ref_dir: str,
    prompt: str | None,
    max_frames: int = _DEFAULT_MAX_FRAMES,
) -> ClipMetrics | None:
    """Compute CLIP semantic metrics between TRT and HF reference image dirs.

    Returns None when:
    - open_clip is not installed
    - prompt is empty / None
    - no frame_*.png files found in either directory

    Caller should treat None as "skipped", not as a failure.
    """
    if not prompt:
        logger.debug("CLIP metrics skipped: no prompt provided")
        return None

    trt_frames = sorted(Path(trt_dir).glob("frame_*.png"))[:max_frames]
    ref_frames = sorted(Path(ref_dir).glob("frame_*.png"))[:max_frames]

    if not trt_frames:
        logger.debug("CLIP metrics skipped: no TRT frames in %s", trt_dir)
        return None
    if not ref_frames:
        logger.debug("CLIP metrics skipped: no reference frames in %s", ref_dir)
        return None

    trt_emb = _embed_images(trt_frames)  # [Nt, D]
    ref_emb = _embed_images(ref_frames)  # [Nr, D]
    txt_emb, truncated = _embed_text(prompt)  # [D], bool

    if trt_emb is None or ref_emb is None or txt_emb is None:
        return None

    # Prompt-fidelity CLIPScore: 100 * max(cos(text, image), 0), mean over frames.
    trt_cos = float((trt_emb @ txt_emb).mean())
    hf_cos = float((ref_emb @ txt_emb).mean())
    trt_score = 100.0 * max(trt_cos, 0.0)
    hf_score = 100.0 * max(hf_cos, 0.0)

    # Image-image semantic similarity: frame-by-frame cosine, averaged.
    n = min(len(trt_frames), len(ref_frames))
    img_cos = float((trt_emb[:n] * ref_emb[:n]).sum(dim=-1).mean())

    if truncated:
        logger.warning(
            "CLIP: prompt exceeds %d tokens and was truncated — "
            "clipscore reflects only the first ~%d tokens: %.60s…",
            CLIP_MAX_TOKENS,
            CLIP_MAX_TOKENS,
            prompt,
        )

    return ClipMetrics(
        trt_prompt_clipscore=trt_score,
        hf_prompt_clipscore=hf_score,
        prompt_clipscore_delta=trt_score - hf_score,
        trt_hf_image_clip_cosine=img_cos,
        prompt_truncated=truncated,
    )
