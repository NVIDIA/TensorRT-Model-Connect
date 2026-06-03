"""Stage the ``bytedance-research/Lance`` HF repo into a buildable directory.

The Lance repo is not a flat HF checkpoint: the LLM lives under ``Lance_3B/``
(or ``Lance_3B_Video/``) as ``llm_config.json`` + ``model.safetensors``, and the
Qwen2.5-VL ViT is a separate ``Qwen2.5-VL-ViT/`` dir. The builder and the
``lance`` family plugin expect a single directory with a ``config.json`` whose
``model_type`` routes to the Lance plugin, plus the ViT at ``vision/``.

This module is the single source of truth for that staging, used by both
``engine_builder._resolve_model`` (so ``trtmc build bytedance-research/Lance``
works directly) and ``scripts/prepare_lance_model.py``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# variant -> LLM subdir in the Lance repo
VARIANT_DIRS = {"image": "Lance_3B", "video": "Lance_3B_Video"}
# Files copied (symlinked) to the top level of the staged dir.
_TOP_LEVEL_FILES = [
    "model.safetensors",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
    "tokenizer_config.json",
]
# Only fetch what the understanding path needs (not the video checkpoint or VAE).
DOWNLOAD_PATTERNS = ["Lance_3B/*", "Lance_3B_Video/*", "Qwen2.5-VL-ViT/*"]


def is_lance_repo(path: Path) -> bool:
    """True if ``path`` is an unpacked Lance repo (any variant + the ViT)."""
    path = Path(path)
    if not (path / "Qwen2.5-VL-ViT" / "vit.safetensors").exists():
        return False
    return any((path / d / "llm_config.json").exists() for d in VARIANT_DIRS.values())


def _symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    os.symlink(target, link)


def stage_lance_repo(src: Path, out: Path, variant: str = "image") -> Path:
    """Stage the Lance repo at ``src`` into ``out`` and return ``out``.

    Writes ``config.json`` (= the variant's ``llm_config.json`` with
    ``model_type`` stamped ``"lance"`` so it routes to the Lance plugin) and
    symlinks the LLM weights, tokenizer files, and the ViT at ``vision/``.
    """
    src, out = Path(src), Path(out)
    if variant not in VARIANT_DIRS:
        raise ValueError(f"unknown Lance variant {variant!r}; choices: {sorted(VARIANT_DIRS)}")
    llm_dir = src / VARIANT_DIRS[variant]
    vit_path = src / "Qwen2.5-VL-ViT" / "vit.safetensors"
    if not (llm_dir / "llm_config.json").exists():
        raise FileNotFoundError(f"{llm_dir}/llm_config.json not found (not a Lance repo?)")
    if not vit_path.exists():
        raise FileNotFoundError(f"{vit_path} not found (not a Lance repo?)")

    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((llm_dir / "llm_config.json").read_text())
    cfg["model_type"] = "lance"
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    for name in _TOP_LEVEL_FILES:
        srcf = llm_dir / name
        if srcf.exists():
            _symlink(srcf, out / name)
        elif name == "model.safetensors":
            raise FileNotFoundError(f"{srcf} not found")

    _symlink(vit_path, out / "vision" / "model.safetensors")
    return out


def download_lance_repo(repo_id: str = "bytedance-research/Lance") -> Path:
    """Download the Lance repo (understanding-relevant files only).

    Cache-aware: falls back to ``local_files_only`` so an already-cached repo
    resolves offline. Non-Lance repos simply fetch nothing matching.
    """
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(repo_id=repo_id, allow_patterns=DOWNLOAD_PATTERNS))
    except Exception:
        return Path(snapshot_download(
            repo_id=repo_id, allow_patterns=DOWNLOAD_PATTERNS, local_files_only=True))


def resolve_and_stage_lance(model_id_or_path: str, variant: str = "image") -> str | None:
    """Return a staged, buildable Lance dir, or None if this isn't a Lance repo.

    Handles both a local unpacked Lance repo dir and a remote HF repo id. The
    staged dir is placed next to the source snapshot so it is reused across
    builds. Returns None (cheaply) when ``model_id_or_path`` is not Lance.
    """
    local = Path(model_id_or_path)
    if local.is_dir():
        if not is_lance_repo(local):
            return None
        src = local
    else:
        # Remote: a Lance-pattern snapshot_download is cache-aware and fetches
        # nothing for non-Lance repos; the is_lance_repo check then rejects them.
        try:
            src = download_lance_repo(model_id_or_path)
        except Exception:
            return None
        if not is_lance_repo(src):
            return None

    out = src.parent / f"{src.name}__trtmc_lance_{variant}"
    return str(stage_lance_repo(src, out, variant=variant))
