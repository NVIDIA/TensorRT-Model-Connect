# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate an aligned OpenFold3 eager reference from the prepared bundle inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from benchmark_reference import _fresh_batch, _load_batch, _seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--precision",
        choices=("fp16", "bf16"),
        default="bf16",
        help="reference autocast precision (default: bf16, the qualified accuracy oracle)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the OpenFold3 reference requires CUDA")

    from openfold3.core.metrics.aggregate_confidence_ranking import get_confidence_scores
    from openfold3.core.runners.writer import OF3OutputWriter
    from openfold3.core.utils.tensor_utils import tensor_tree_map
    from openfold3.projects.of3_all_atom.model import OpenFold3
    from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry

    config = OF3ProjectEntry().get_model_config_with_presets(["predict", "low_mem"])
    config.unlock()
    config.architecture.shared.diffusion.no_full_rollout_samples = 1
    config.lock()
    model = OpenFold3(config)
    state = torch.load(arguments.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.cuda().eval()

    _seed()
    cpu_batch = _load_batch(arguments.query, arguments.components, arguments.features)
    gpu_batch = tensor_tree_map(lambda tensor: tensor.cuda(), cpu_batch)
    _seed()
    autocast_dtype = torch.float16 if arguments.precision == "fp16" else torch.bfloat16
    with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype):
        batch, outputs = model(_fresh_batch(gpu_batch))
        positions = outputs["atom_positions_predicted"]
        finite = bool(torch.isfinite(positions).all().item())
        max_abs = float(torch.nan_to_num(positions).abs().max().item())
        print(
            f"OpenFold3 {arguments.precision}-mixed coordinate check: "
            f"finite={finite}, max_abs={max_abs:.6g} Angstrom",
            flush=True,
        )
        if not finite or max_abs > 10_000.0:
            raise RuntimeError(
                f"OpenFold3 {arguments.precision}-mixed produced an invalid diffusion rollout; "
                "use the qualified BF16 reference oracle"
            )
        confidence = get_confidence_scores(
            batch=batch,
            outputs=outputs,
            config=config,
            compute_per_sample=False,
        )
    batch["seed"] = [42]
    writer = OF3OutputWriter(
        output_dir=arguments.output_dir,
        structure_format="cif",
        full_confidence_output_format="json",
        full_confidence_output_dtype="float32",
        write_full_confidence_scores=True,
    )
    writer.write_all_outputs(batch, outputs, confidence)
    print(arguments.output_dir)


if __name__ == "__main__":
    main()
