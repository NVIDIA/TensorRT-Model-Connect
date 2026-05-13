"""MagpieTTS IPA tokenizer support owned by the Magpie family builder."""

from __future__ import annotations

import os
import pathlib
import tarfile


def _extract_nemo_assets(nemo_path: pathlib.Path, extract_dir: pathlib.Path) -> None:
    """Extract phoneme dict / heteronym files from a .nemo archive."""
    with tarfile.open(str(nemo_path), "r") as tar:
        for member in tar.getmembers():
            basename = pathlib.Path(member.name).name
            if basename in ("model_weights.ckpt", "model_config.yaml"):
                continue
            if member.isfile():
                dest = extract_dir / basename
                if not dest.exists():
                    f = tar.extractfile(member)
                    if f is not None:
                        dest.write_bytes(f.read())


def _resolve_asset_dir() -> pathlib.Path:
    candidates = [
        pathlib.Path(os.environ.get("XDG_CACHE_HOME", "")).expanduser() / "trtmc_nemo_assets"
        if os.environ.get("XDG_CACHE_HOME", "").strip()
        else None,
        pathlib.Path.home() / ".cache" / "trtmc_nemo_assets",
        pathlib.Path("/tmp/trtmc_nemo_assets"),
    ]

    for candidate in candidates:
        if candidate is None:
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except PermissionError:
            continue

    raise PermissionError("Could not create writable cache dir for Magpie tokenizer assets")


def load_tokenizer(nemo_path: str | pathlib.Path, lang_key: str = "english_phoneme"):
    """Load the NeMo IPATokenizer from the MagpieTTS archive config.

    This mirrors the CLI bridge in scripts/magpie_tokenizer.py, but keeps the
    build-time tokenizer dependency local to the Magpie model family.
    """
    import logging
    logging.disable(logging.WARNING)

    import yaml

    path = pathlib.Path(nemo_path)
    nemo_cfg = None
    with tarfile.open(str(path), "r") as tar:
        for member in tar.getmembers():
            if pathlib.Path(member.name).name == "model_config.yaml":
                f = tar.extractfile(member)
                if f is not None:
                    nemo_cfg = yaml.safe_load(f.read())
                break
    if nemo_cfg is None:
        raise FileNotFoundError(f"model_config.yaml not found in {path}")

    text_vocab_size = int(nemo_cfg.get("text_vocab_size", 2378))
    text_tokenizers = nemo_cfg.get("text_tokenizers", {})

    if lang_key not in text_tokenizers:
        available = list(text_tokenizers.keys())
        raise ValueError(
            f"Language '{lang_key}' not found. Available: {available}")

    tok_cfg = dict(text_tokenizers[lang_key])
    target = str(tok_cfg.get("_target_", ""))

    if target == "AutoTokenizer":
        raise RuntimeError(
            f"Magpie tokenizer '{lang_key}' requires HF AutoTokenizer path, "
            "which is not supported in trtmc runtime yet. Use english_phoneme."
        )

    try:
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
    except Exception as exc:
        raise RuntimeError(
            "Missing Magpie tokenizer dependencies. Install: "
            "nemo_toolkit[tts]==2.7.0"
        ) from exc

    asset_dir = _resolve_asset_dir()
    _extract_nemo_assets(path, asset_dir)

    if "g2p" in tok_cfg and "phoneme_probability" in tok_cfg["g2p"]:
        tok_cfg["g2p"]["phoneme_probability"] = 1.0

    if "g2p" in tok_cfg:
        g2p_cfg = tok_cfg["g2p"]
        for key in ("phoneme_dict", "heteronyms"):
            val = g2p_cfg.get(key)
            if isinstance(val, str) and val.startswith("nemo:"):
                filename = val.split(":")[-1]
                local_path = asset_dir / filename
                if local_path.exists():
                    g2p_cfg[key] = str(local_path)

    oc = OmegaConf.create(tok_cfg)
    sub_tok = instantiate(oc)

    return sub_tok, text_vocab_size
