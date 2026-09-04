# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load the exact IPA tokenizer embedded in a MagpieTTS NeMo archive."""

from __future__ import annotations

import logging
import tarfile
import tempfile
from pathlib import Path


def load_tokenizer(nemo_path: str | Path, lang_key: str = "english_phoneme"):
    """Instantiate the checkpoint-owned NeMo IPA tokenizer."""
    import yaml
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    logging.disable(logging.WARNING)
    archive = Path(nemo_path)
    if not archive.is_file():
        raise FileNotFoundError(f"MagpieTTS NeMo archive does not exist: {archive}")

    model_config = None
    with tarfile.open(archive, "r") as tar:
        for member in tar.getmembers():
            if Path(member.name).name == "model_config.yaml":
                stream = tar.extractfile(member)
                if stream is not None:
                    model_config = yaml.safe_load(stream.read())
                break
    if not isinstance(model_config, dict):
        raise FileNotFoundError(f"model_config.yaml not found in {archive}")

    text_vocab_size = int(model_config["text_vocab_size"])
    tokenizers = model_config["text_tokenizers"]
    if lang_key not in tokenizers:
        raise ValueError(f"MagpieTTS checkpoint does not define tokenizer {lang_key!r}")
    tokenizer_config = dict(tokenizers[lang_key])
    if str(tokenizer_config.get("_target_", "")) == "AutoTokenizer":
        raise ValueError("MagpieTTS native runtime requires the english_phoneme IPA tokenizer")

    with tempfile.TemporaryDirectory(prefix="magpie-ipa-") as directory:
        asset_dir = Path(directory)
        with tarfile.open(archive, "r") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                name = Path(member.name).name
                stream = tar.extractfile(member)
                if stream is not None:
                    (asset_dir / name).write_bytes(stream.read())

        g2p = tokenizer_config.get("g2p")
        if isinstance(g2p, dict):
            if "phoneme_probability" in g2p:
                g2p["phoneme_probability"] = 1.0
            for key in ("phoneme_dict", "heteronyms"):
                value = g2p.get(key)
                if isinstance(value, str) and value.startswith("nemo:"):
                    path = asset_dir / value.rsplit(":", 1)[-1]
                    if not path.is_file():
                        raise FileNotFoundError(
                            f"MagpieTTS tokenizer asset is missing: {path.name}"
                        )
                    g2p[key] = str(path)
        tokenizer = instantiate(OmegaConf.create(tokenizer_config))

    return tokenizer, text_vocab_size
