"""MagpieTTS IPA tokenizer support owned by the Magpie family builder."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import tarfile
import warnings


os.environ.setdefault("NEMO_LOG_LEVEL", "ERROR")
warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MagpieTTS IPA tokenizer bridge for trtmc C++ runtime"
    )
    parser.add_argument(
        "--nemo-path",
        required=True,
        help="Path to MagpieTTS .nemo archive or directory containing one",
    )
    parser.add_argument("--check", action="store_true", help="Validate tokenizer can be loaded")
    parser.add_argument(
        "--op", choices=["encode", "decode"], default="", help="Tokenizer operation"
    )
    parser.add_argument("--text-file", default="", help="Input text file for encode")
    parser.add_argument("--ids", default="", help="Comma-separated token IDs for decode")
    parser.add_argument(
        "--lang",
        default="english_phoneme",
        help="Language key from text_tokenizers (default: english_phoneme)",
    )
    return parser.parse_args()


def _repo_id_from_hf_cache_path(path: pathlib.Path) -> str | None:
    """Extract repo id from HF cache paths like models--org--repo/blobs/..."""
    for part in path.parts:
        if part.startswith("models--"):
            encoded = part[len("models--") :]
            if "--" in encoded:
                return encoded.replace("--", "/")
    return None


def _looks_like_repo_id(text: str) -> bool:
    if text.startswith("/") or text.startswith(".") or "://" in text:
        return False
    parts = text.split("/")
    return len(parts) == 2 and all(parts)


def _download_nemo_for_repo(repo_id: str) -> pathlib.Path:
    """Resolve a .nemo archive from HF cache (and network if permitted)."""
    from huggingface_hub import snapshot_download

    offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")
    attempts = (True,) if offline else (True, False)

    last_exc: Exception | None = None
    for local_only in attempts:
        try:
            snapshot_dir = snapshot_download(
                repo_id=repo_id,
                allow_patterns=["*.nemo"],
                local_files_only=local_only,
            )
            nemo_files = sorted(pathlib.Path(snapshot_dir).glob("*.nemo"))
            if nemo_files:
                return nemo_files[0]
            raise FileNotFoundError(f"No .nemo files found in snapshot: {snapshot_dir}")
        except Exception as exc:  # pragma: no cover - exercised in integration
            last_exc = exc

    detail = f"{last_exc}" if last_exc is not None else "unknown error"
    raise FileNotFoundError(
        f"Could not resolve .nemo for repo '{repo_id}' from cache or hub: {detail}"
    )


def _resolve_nemo_path(path: str) -> pathlib.Path:
    """Resolve .nemo archive from file/dir, HF cache blob path, or repo id."""
    p = pathlib.Path(path)

    try:
        if p.is_dir():
            nemo_files = sorted(p.glob("*.nemo"))
            if nemo_files:
                return nemo_files[0]
            raise FileNotFoundError(f"No .nemo file found in directory: {path}")
        if p.is_file():
            return p
    except PermissionError as exc:
        repo_id = _repo_id_from_hf_cache_path(p)
        if repo_id:
            return _download_nemo_for_repo(repo_id)
        raise FileNotFoundError(f"NeMo path exists but is not accessible: {path} ({exc})") from exc

    if _looks_like_repo_id(path):
        return _download_nemo_for_repo(path)

    repo_id = _repo_id_from_hf_cache_path(p)
    if repo_id:
        return _download_nemo_for_repo(repo_id)

    raise FileNotFoundError(f".nemo archive does not exist or is invalid: {path}")


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

    This keeps the build-time tokenizer dependency local to the Magpie model
    family while supporting module execution for CLI use.
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
        raise ValueError(f"Language '{lang_key}' not found. Available: {available}")

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
            "Missing Magpie tokenizer dependencies. Install: nemo_toolkit[tts]==2.7.0"
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


def parse_ids_csv(ids_text: str) -> list[int]:
    ids_text = ids_text.strip()
    if not ids_text:
        return []
    out: list[int] = []
    for part in ids_text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def main() -> int:
    args = parse_args()

    nemo_path = _resolve_nemo_path(args.nemo_path)
    if not nemo_path.exists():
        print(f".nemo archive does not exist: {nemo_path}", file=sys.stderr)
        return 2

    try:
        tokenizer, text_vocab_size = load_tokenizer(nemo_path, args.lang)
    except Exception as exc:
        print(f"Failed to load MagpieTTS tokenizer: {exc}", file=sys.stderr)
        return 3

    if args.check:
        return 0

    eos_id = text_vocab_size + 1

    if args.op == "encode":
        if not args.text_file:
            print("--text-file is required for encode", file=sys.stderr)
            return 4
        text = pathlib.Path(args.text_file).read_text(encoding="utf-8")
        ids = tokenizer.encode(text)
        ids.append(eos_id)
        print(" ".join(str(i) for i in ids))
        return 0

    if args.op == "decode":
        ids = parse_ids_csv(args.ids)
        try:
            decoded = tokenizer.decode(ids)
        except Exception:
            decoded = " ".join(str(i) for i in ids)
        print(decoded)
        return 0

    print("--op is required unless --check is set", file=sys.stderr)
    return 6


if __name__ == "__main__":
    raise SystemExit(main())
