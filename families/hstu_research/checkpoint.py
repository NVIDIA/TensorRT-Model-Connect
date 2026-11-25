# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert complete public Meta-style research weights without executing Hub code."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import urllib.request
import numpy as np
from .config import Architecture, Config, ServingProfile

DEFAULT_REPOSITORY = "tzchen07/hstu-checkpoints-2026-06-10"
DEFAULT_REVISION = "fb6530eba870e842cfb236737586a4188ec3faef"
DEFAULT_ARCHIVE = "run26_ndcg1849.tar.gz"
DEFAULT_ARCHIVE_SHA256 = "9c4edcecb5e15c513ec7baa343f1a1e2ee0b8a200242e811612f55205a74c9ac"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def expected_shapes(a):
    shapes = dict(
        item_embeddings=(a.vocab_size, a.hidden_size),
        position_embeddings=(a.max_sequence_length, a.hidden_size),
        causal_mask=(a.max_sequence_length, a.max_sequence_length),
    )
    for i in range(a.num_layers):
        prefix = f"layers.{i}."
        shapes.update(
            {
                prefix + "uvqk": (
                    a.hidden_size,
                    a.num_heads * (2 * a.value_dim + 2 * a.attention_dim),
                ),
                prefix + "time_bias": (a.time_buckets + 1,),
                prefix + "position_bias": (2 * a.max_sequence_length - 1,),
                prefix + "output_weight": (a.num_heads * a.value_dim, a.hidden_size),
                prefix + "output_bias": (a.hidden_size,),
            }
        )
    return shapes


def load_weights(directory, config):
    with np.load(Path(directory) / "hstu_research_weights.npz", allow_pickle=False) as archive:
        expected = expected_shapes(config.architecture)
        if set(archive.files) != set(expected):
            raise ValueError("checkpoint tensor keys differ from the complete supported model")
        weights = {key: np.array(archive[key], copy=True) for key in archive.files}
    for key, shape in expected.items():
        value = weights[key]
        if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
            raise ValueError("invalid checkpoint tensor: " + key)
    n = config.architecture.max_sequence_length
    if not np.array_equal(weights["causal_mask"], np.triu(np.ones((n, n), np.float32), 1)):
        raise ValueError("checkpoint uses an unsupported attention mask")
    return weights


def parse_gin(text):
    values = {}
    for line in text.splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = ast.literal_eval(value.strip())
    return values


def validate_behavior(gin):
    required = {
        "hstu_encoder.normalization": "rel_bias",
        "hstu_encoder.linear_config": "uvqk",
        "hstu_encoder.linear_activation": "silu",
        "hstu_encoder.concat_ua": False,
        "hstu_encoder.enable_relative_attention_bias": True,
        "train_fn.user_embedding_norm": "l2_norm",
        "train_fn.item_l2_norm": True,
        "train_fn.l2_norm_eps": 1e-6,
        "train_fn.interaction_module_type": "DotProduct",
        "train_fn.embedding_module_type": "local",
        "train_fn.input_compression_budget": None,
        "train_fn.enable_time_decay": False,
    }
    required.update(
        {
            "hstu_encoder." + key: False
            for key in (
                "enable_prism",
                "enable_ssd",
                "enable_fluid",
                "enable_multi_token",
                "enable_ctx_cross_attn",
                "enable_ctx_gate",
                "enable_ctx_output_residual",
            )
        }
    )
    for key, value in required.items():
        if key not in gin or gin[key] != value:
            raise ValueError("unsupported checkpoint behavior: " + key)


def architecture_from_state(raw, gin):
    item = raw["_embedding_module._item_emb.weight"]
    n = raw["_input_features_preproc._pos_emb.weight"].shape[0]
    a = Architecture(
        hidden_size=item.shape[1],
        num_heads=gin["hstu_encoder.num_heads"],
        attention_dim=gin["hstu_encoder.dqk"],
        value_dim=gin["hstu_encoder.dv"],
        num_layers=gin["hstu_encoder.num_blocks"],
        vocab_size=item.shape[0],
        max_sequence_length=n,
        max_history_length=gin["train_fn.max_sequence_length"],
    )
    return a


def map_weights(raw, a):
    mapping = {
        "_embedding_module._item_emb.weight": "item_embeddings",
        "_input_features_preproc._pos_emb.weight": "position_embeddings",
        "_attn_mask": "causal_mask",
    }
    for i in range(a.num_layers):
        prefix = f"_hstu._attention_layers.{i}."
        mapping.update(
            {
                prefix + old: f"layers.{i}." + new
                for old, new in (
                    ("_uvqk", "uvqk"),
                    ("_rel_attn_bias._ts_w", "time_bias"),
                    ("_rel_attn_bias._pos_w", "position_bias"),
                    ("_o.weight", "output_weight"),
                    ("_o.bias", "output_bias"),
                )
            }
        )
    if set(mapping) != set(raw):
        raise ValueError("checkpoint contains missing or unsupported model components")
    weights = {}
    for old, new in mapping.items():
        array = raw[old].detach().cpu().numpy()
        if new.endswith("output_weight"):
            array = array.T
        weights[new] = np.array(array, dtype=np.float32, copy=True)
    return weights


def convert(source, output, *, history_length=None, candidates=64, provenance=None):
    import torch

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("conversion output directory must be empty")
    # The archive includes NumPy RNG state. Allow only these standard NumPy types,
    # never arbitrary globals named by the downloaded pickle.
    reconstruct = np.core.multiarray._reconstruct
    allowed = [
        (reconstruct, "numpy._core.multiarray._reconstruct"),
        (reconstruct, "numpy.core.multiarray._reconstruct"),
        np.ndarray,
        np.dtype,
        np.dtypes.UInt32DType,
    ]
    with torch.serialization.safe_globals(allowed):
        state = torch.load(source, map_location="cpu", weights_only=True)
    gin = parse_gin(state["gin_config_str"])
    validate_behavior(gin)
    raw = state["model_state_dict"]
    if not all(name.startswith("module.") for name in raw):
        raise ValueError("expected the documented DDP checkpoint schema")
    raw = {name.removeprefix("module."): value for name, value in raw.items()}
    a = architecture_from_state(raw, gin)
    weights = map_weights(raw, a)
    profile = ServingProfile(
        history_length=a.max_history_length if history_length is None else history_length,
        candidates=candidates,
    )
    config = Config(
        architecture=a, serving=profile, bf16_projections="all", fp32_block_indices=(0,)
    )
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "hstu_research_weights.npz", **weights)
    config.save(output)
    load_weights(output, config)
    receipt = dict(
        kind="public-trained-research-checkpoint",
        source_sha256=digest(source),
        weights_sha256=digest(output / "hstu_research_weights.npz"),
        epoch=int(state["epoch"]),
        training_config=gin,
        **(provenance or {}),
    )
    (output / "checkpoint_provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return config


def download_default(output, *, history_length=None, candidates=64):
    with tempfile.TemporaryDirectory(prefix="hstu-public-checkpoint-") as temporary:
        root = Path(temporary)
        archive = root / DEFAULT_ARCHIVE
        url = f"https://huggingface.co/{DEFAULT_REPOSITORY}/resolve/{DEFAULT_REVISION}/{DEFAULT_ARCHIVE}"
        urllib.request.urlretrieve(url, archive)
        if digest(archive) != DEFAULT_ARCHIVE_SHA256:
            raise ValueError("default checkpoint archive checksum mismatch")
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers() if m.isfile() and m.name.startswith("HSTU-")]
            if len(members) != 1:
                raise ValueError("unexpected default archive contents")
            checkpoint = root / "checkpoint.pt"
            with tar.extractfile(members[0]) as source, checkpoint.open("wb") as target:
                import shutil

                shutil.copyfileobj(source, target)
        return convert(
            checkpoint,
            output,
            history_length=history_length,
            candidates=candidates,
            provenance=dict(
                repository=DEFAULT_REPOSITORY,
                revision=DEFAULT_REVISION,
                archive=DEFAULT_ARCHIVE,
                archive_sha256=digest(archive),
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--download-default", action="store_true")
    parser.add_argument("--history-length", type=int)
    parser.add_argument("--candidates", type=int, default=64)
    args = parser.parse_args()
    if bool(args.source) == args.download_default:
        parser.error("choose --source or --download-default")
    options = dict(history_length=args.history_length, candidates=args.candidates)
    if args.source:
        convert(args.source, args.output, **options)
    else:
        download_default(args.output, **options)


if __name__ == "__main__":
    main()
