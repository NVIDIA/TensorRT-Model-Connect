# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert all 86 tensors of the official MLPerf Inference v6.0 dense checkpoint."""

import argparse
import json
from pathlib import Path
import numpy as np
from .config import Config, ServingProfile

DEFAULT_URL = "https://inference.mlcommons-storage.org/dlrmv3_trained_checkpoint/non_sparse.ckpt"
LICENSE_URL = "https://inference.mlcommons-storage.org/dlrmv3_trained_checkpoint/LICENSE.txt"
PRE = "_hstu_transducer._input_preprocessor."
POST = "_hstu_transducer._output_postprocessor."
POS = "_hstu_transducer._positional_encoder."
BLOCK = "_hstu_transducer._stu_module._stu_layers."
HEAD = "_multitask_module._prediction_module."


def expected_shapes():
    shapes = {}
    for prefix, width, hidden in (
        (PRE + "_content_embedding_mlp.", 512, 256),
        (PRE + "_additional_embedding_mlp.", 512, 256),
        (PRE + "_action_embedding_mlp.", 40, 256),
        ("_item_embedding_mlp.", 1024, 512),
    ):
        shapes.update(
            {
                prefix + "0.weight": (hidden, width),
                prefix + "0.bias": (hidden,),
                prefix + "1.weight": (hidden,),
                prefix + "1.bias": (hidden,),
                prefix + "2.weight": (512, hidden),
                prefix + "2.bias": (512,),
                prefix + "3.weight": (512,),
                prefix + "3.bias": (512,),
            }
        )
    shapes.update(
        {
            PRE + "_action_encoder._action_embedding_table": (5, 8),
            PRE + "_action_encoder._combined_action_weights": (5,),
            PRE + "_action_encoder._target_action_embedding_table": (1, 40),
            PRE + "_batched_contextual_linear_weights": (1, 512, 512),
            PRE + "_batched_contextual_linear_bias": (1, 512),
            POS + "_position_embeddings_weight": (8192, 512),
            POS + "_timestamp_embeddings_weight": (2049, 512),
            POST + "_period_units": (1, 2),
            POST + "_units_per_period": (1, 2),
            POST + "_time_feature_combiner.weight": (512, 516),
            POST + "_time_feature_combiner.bias": (512,),
            POST + "_layer_norm.weight": (512,),
            POST + "_layer_norm.bias": (512,),
            HEAD + "0.weight": (512, 512),
            HEAD + "0.bias": (512,),
            HEAD + "1.weight": (512,),
            HEAD + "1.bias": (512,),
            HEAD + "2.weight": (1, 512),
            HEAD + "2.bias": (1,),
        }
    )
    for i in range(5):
        prefix = BLOCK + str(i) + "."
        shapes.update(
            {
                prefix + "_uvqk_weight": (512, 2048),
                prefix + "_uvqk_beta": (2048,),
                prefix + "_output_weight": (1536, 512),
            }
        )
        for name in (
            "_input_norm_weight",
            "_input_norm_bias",
            "_output_norm_weight",
            "_output_norm_bias",
        ):
            shapes[prefix + name] = (512,)
    return shapes


def validate_weights(weights):
    expected = expected_shapes()
    if set(weights) != set(expected):
        raise ValueError("checkpoint must contain exactly the 86 official dense tensors")
    for name, shape in expected.items():
        value = weights[name]
        dtype = np.int64 if name.endswith("_combined_action_weights") else np.float32
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError("invalid checkpoint tensor: " + name)
    if not np.array_equal(
        weights[PRE + "_action_encoder._combined_action_weights"], [1, 2, 4, 8, 16]
    ):
        raise ValueError("unsupported action bitmask encoding")
    for name in (POST + "_period_units", POST + "_units_per_period"):
        if (weights[name] <= 0).any():
            raise ValueError("invalid timestamp periods")
    return weights


def load_weights(directory, config=None):
    with np.load(Path(directory) / "dlrmv3_weights.npz", allow_pickle=False) as archive:
        return validate_weights(
            {name: np.array(archive[name], copy=True) for name in archive.files}
        )


def convert(source, output, *, history_length=512, candidates=128):
    import torch

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("conversion output must be empty")
    state = torch.load(source, map_location="cpu", weights_only=True)
    weights = validate_weights(
        {k: v.detach().cpu().numpy().copy() for k, v in state["dense_dict"].items()}
    )
    config = Config(serving=ServingProfile(history_length=history_length, candidates=candidates))
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "dlrmv3_weights.npz", **weights)
    config.save(output)
    (output / "checkpoint_provenance.json").write_text(
        json.dumps(
            dict(
                model="MLPerf Inference v6.0 DLRMv3",
                source=DEFAULT_URL,
                license=LICENSE_URL,
                scope="complete dense model; sparse embedding lookup is caller-owned",
                tensor_count=len(weights),
                tensor_bytes=sum(v.nbytes for v in weights.values()),
            ),
            indent=2,
        )
        + "\n"
    )
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--history-length", type=int, default=512)
    parser.add_argument("--candidates", type=int, default=128)
    args = parser.parse_args()
    convert(
        args.source, args.output, history_length=args.history_length, candidates=args.candidates
    )


if __name__ == "__main__":
    main()
