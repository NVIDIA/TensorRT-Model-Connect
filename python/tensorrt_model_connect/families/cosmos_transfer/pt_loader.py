"""Custom .pt checkpoint loader for nvidia/Cosmos-Transfer1-7B.

Cosmos-Transfer1 ships its weights as raw PyTorch state-dicts (``*.pt`` files)
produced by NVIDIA's internal training stack, NOT as ``safetensors`` and NOT
as the HuggingFace ``diffusers`` layout. So the standard
``checkpoint_mapper._open_safetensors`` / ``_load_tensor`` helpers do not work
out of the box.

Expected on-disk layout (HF repo nvidia/Cosmos-Transfer1-7B, 155 files):

    base_model.pt              - main DiT denoiser (~7B params)
    edge_control.pt            - Canny edge ControlNet branch
    depth_control.pt           - depth-map ControlNet branch
    seg_control.pt             - segmentation-mask ControlNet branch (optional)
    vis_control.pt             - blurred-RGB ControlNet branch (optional)
    keypoint_control.pt        - human-keypoint ControlNet branch (optional)
    4kupscaler_control.pt      - 720p->4k upscaler ControlNet (separate task)
    t5_text_encoder.pt         - frozen Google T5-XXL encoder weights
    cosmos_tokenizer/           - VAE (Cosmos-Tokenizer CV8x8x8) sub-dir or
                                  flat *.pt files (varies by release).

Naming variants observed across Cosmos releases:
  - ``ctrl_<modality>.pt``           (cosmos-transfer1 GitHub)
  - ``<modality>_control.pt``        (HF repo, post-tarball repack)
  - ``base_model.pt`` vs ``model.pt`` (some releases use the latter)

This module exposes:

    load_pt_state_dict(path) -> dict[str, np.ndarray]
        Strict ``torch.load(weights_only=True)`` wrapper that returns a numpy
        WeightDict (linear projections are NOT transposed here — the sub-
        builders handle that, matching what wan_t2v / flux do).

    discover_checkpoints(model_dir) -> dict[str, Path]
        Scans the model dir for the known .pt filenames listed above and
        returns ``{role: path}`` (role in {base, edge, depth, seg, vis,
        keypoint, upscaler, t5, vae}). Missing roles are simply absent from
        the dict — the plugin decides which are required.

torch.load policy
-----------------
We pass ``weights_only=True`` so the loader refuses arbitrary Python objects;
the only allowed values are tensors / nested dicts / lists / numeric primitives.
This is required by NVIDIA security guidelines (the .pt files come from an
external repo) and is supported by torch >= 1.13.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


# Canonical role -> list of filename candidates (first match wins).
# Order matters: newer / preferred names listed first.
_CHECKPOINT_ROLES: dict[str, tuple[str, ...]] = {
    "base": ("base_model.pt", "model.pt", "dit.pt"),
    "edge": ("edge_control.pt", "ctrl_edge.pt"),
    "depth": ("depth_control.pt", "ctrl_depth.pt"),
    "seg": ("seg_control.pt", "ctrl_seg.pt", "segmentation_control.pt"),
    "vis": ("vis_control.pt", "ctrl_vis.pt", "blur_control.pt"),
    "keypoint": ("keypoint_control.pt", "ctrl_keypoint.pt"),
    "upscaler": ("4kupscaler_control.pt", "ctrl_upscaler.pt"),
    "t5": ("t5_text_encoder.pt", "text_encoder.pt"),
    "vae": ("cosmos_tokenizer.pt", "vae.pt", "tokenizer.pt"),
}


# The five canonical ControlNet modalities (excluding the 4K upscaler, which
# is a separate post-processing task).
CONTROLNET_MODALITIES: tuple[str, ...] = (
    "edge", "depth", "seg", "vis", "keypoint",
)


def discover_checkpoints(model_dir: str | Path) -> dict[str, Path]:
    """Scan ``model_dir`` for known Cosmos-Transfer .pt files.

    Returns a dict mapping role -> existing file path. Roles whose
    candidate filenames are all missing are absent from the dict.
    Sub-directory lookups: ``cosmos_tokenizer/*.pt`` is recognized
    for the VAE role (some releases ship it as a directory of shards).
    """
    model_path = Path(model_dir)
    found: dict[str, Path] = {}

    for role, candidates in _CHECKPOINT_ROLES.items():
        for fname in candidates:
            p = model_path / fname
            if p.is_file():
                found[role] = p
                break

    # VAE may also live under a sub-dir.
    if "vae" not in found:
        vae_dir = model_path / "cosmos_tokenizer"
        if vae_dir.is_dir():
            # Prefer the decoder if present; otherwise grab the first .pt.
            for cand in ("decoder.pt", "model.pt", "tokenizer.pt"):
                p = vae_dir / cand
                if p.is_file():
                    found["vae"] = p
                    break
            else:
                pts = sorted(vae_dir.glob("*.pt"))
                if pts:
                    found["vae"] = pts[0]

    return found


def load_pt_state_dict(
    path: str | Path,
    *,
    map_location: str = "cpu",
    strip_prefixes: tuple[str, ...] = ("module.", "_orig_mod."),
) -> "dict[str, np.ndarray]":
    """Load a Cosmos ``.pt`` checkpoint and return a flat ``{name: ndarray}``.

    Behavior:
      * ``torch.load(weights_only=True)`` — refuses pickled Python objects.
      * Unwraps common top-level wrappers: ``{"model": ...}``,
        ``{"state_dict": ...}``, ``{"ema": ...}`` (preferring "ema" if present
        since Cosmos releases the EMA copy for inference).
      * Strips DDP / torch.compile prefixes (``module.``, ``_orig_mod.``).
      * Converts tensors to ``numpy.ndarray`` via ``.detach().cpu().numpy()``
        on the appropriate dtype. Sub-builders are responsible for any
        transpose / reshape required by TensorRT.

    Args:
        path: Path to the .pt file.
        map_location: torch device for the load. Default 'cpu' so this works
            on builder machines without CUDA.
        strip_prefixes: List of key prefixes to strip. The first matching
            prefix wins per key.

    Returns:
        Flat dict mapping weight name to a NumPy array. Empty dict if the
        file is missing.
    """
    import numpy as np
    import torch

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Cosmos .pt checkpoint not found: {p}")

    raw = torch.load(str(p), map_location=map_location, weights_only=True)

    # Top-level unwrap. Cosmos checkpoints in the wild use one of these
    # layouts. We prefer the EMA copy when both are present, since that is
    # what NVIDIA uses for inference.
    if isinstance(raw, dict):
        for key in ("ema", "model_ema", "state_dict_ema",
                    "model", "state_dict", "module"):
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break

    if not isinstance(raw, dict):
        raise ValueError(
            f"Unexpected .pt structure in {p}: top-level is "
            f"{type(raw).__name__}, expected dict")

    out: dict[str, np.ndarray] = {}
    for k, v in raw.items():
        if not isinstance(v, torch.Tensor):
            # Skip metadata entries (ints, strs, etc.).
            continue
        name = str(k)
        for prefix in strip_prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        arr = v.detach().cpu()
        # Promote bfloat16 to float32 for numpy (no native np.bfloat16).
        if arr.dtype == torch.bfloat16:
            arr = arr.to(torch.float32)
        out[name] = arr.numpy()

    return out
