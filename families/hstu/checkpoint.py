# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create a reproducible public validation fixture, never pretrained weights."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .config import Config, REVISION, UPSTREAM


def expected_shapes(c):
    d = c.hidden_size
    return {
        "item_embeddings": (c.vocab_size, d),
        "position_embeddings": (c.max_history_length + c.max_decode_steps + 1, d),
        "uvqk": (d, 4 * d),
        "output_weight": (d, d),
        "output_bias": (d,),
        "head_weight": (d, 1),
        "head_bias": (1,),
    }


def load_weights(path, c):
    with np.load(Path(path) / "hstu_weights.npz", allow_pickle=False) as f:
        if set(f.files) != set(expected_shapes(c)):
            raise ValueError("checkpoint keys differ from supported HSTU reference")
        values = {name: np.array(f[name], copy=True) for name in f.files}
    for name, shape in expected_shapes(c).items():
        v = values[name]
        if v.shape != shape or v.dtype != np.float32 or not np.isfinite(v).all():
            raise ValueError(f"invalid checkpoint tensor: {name}")
    return values


def create_fixture(path, candidates=50, seed=20261124, *, config=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError("fixture output directory must be empty")
    c = config or Config(candidates=candidates)
    rng = np.random.default_rng(seed)
    weights = {
        name: rng.normal(0, 0.02, shape).astype(np.float32)
        for name, shape in expected_shapes(c).items()
    }
    weights["output_weight"] *= np.float32(3.0)
    weights["head_weight"] *= np.float32(8.0)
    weights["output_bias"].fill(0)
    weights["head_bias"].fill(0)
    np.savez(path / "hstu_weights.npz", **weights)
    c.save(path)
    (path / "checkpoint_provenance.json").write_text(
        json.dumps(
            dict(
                kind="deterministic-validation-fixture",
                pretrained=False,
                seed=seed,
                architecture_source=UPSTREAM,
                architecture_revision=REVISION,
                sha256=hashlib.sha256((path / "hstu_weights.npz").read_bytes()).hexdigest(),
            ),
            indent=2,
        )
        + "\n"
    )
    return c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--candidates", type=int, choices=(50, 64), default=50)
    p.add_argument("--seed", type=int, default=20261124)
    p.add_argument(
        "--config", type=Path, help="Directory containing a validated architecture/serving config"
    )
    a = p.parse_args()
    create_fixture(
        a.output, a.candidates, a.seed, config=Config.load(a.config) if a.config else None
    )


if __name__ == "__main__":
    main()
