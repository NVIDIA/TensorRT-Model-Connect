# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the aligned eager or torch.compile OpenFold3 reference."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch


def _load_batch(query: Path, components: Path, features: Path) -> dict:
    from biotite.structure.info.ccd import set_ccd_path
    from openfold3.core.data.framework.data_module import DataModuleConfig, InferenceDataModule
    from openfold3.core.data.pipelines.preprocessing.template import TemplatePreprocessorSettings
    from openfold3.core.data.tools.colabfold_msa_server import MsaComputationSettings
    from openfold3.projects.of3_all_atom.config.dataset_configs import (
        InferenceDatasetSpec,
        InferenceJobConfig,
    )
    from openfold3.projects.of3_all_atom.config.inference_query_format import InferenceQuerySet

    set_ccd_path(components)
    query_set = InferenceQuerySet.from_json(query)
    job = InferenceJobConfig(
        query_set=query_set,
        seeds=[42],
        template_preprocessor_settings=TemplatePreprocessorSettings(mode="predict"),
    )
    module = InferenceDataModule(
        DataModuleConfig(
            datasets=[InferenceDatasetSpec(config=job)],
            batch_size=1,
            epoch_len=1,
            num_workers=0,
        ),
        use_msa_server=False,
        use_templates=False,
        msa_computation_settings=MsaComputationSettings(),
    )
    module.prepare_data()
    module.setup()
    batch = next(iter(module.predict_dataloader()))
    if not bool(batch["valid_sample"].item()):
        raise RuntimeError("OpenFold3 rejected the benchmark request")
    with np.load(features, allow_pickle=False) as prepared:
        atom_count = int(batch["atom_mask"].shape[-1])
        for name in (
            "ref_pos",
            "ref_mask",
            "ref_element",
            "ref_charge",
            "ref_atom_name_chars",
            "ref_space_uid",
            "atom_mask",
            "atom_to_token_index",
            "token_mask",
            "restype",
            "profile",
            "deletion_mean",
            "token_bonds",
            "msa",
            "has_deletion",
            "deletion_value",
            "msa_mask",
        ):
            value = prepared[name]
            if name.startswith("ref_") or name in {"atom_mask", "atom_to_token_index"}:
                value = value[:, :atom_count]
            replacement = torch.as_tensor(value, dtype=batch[name].dtype)
            if replacement.shape != batch[name].shape:
                raise RuntimeError(f"prepared OpenFold3 feature {name!r} differs from reference")
            batch[name] = replacement
    return batch


def _seed() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def _fresh_batch(batch: dict) -> dict:
    from openfold3.core.utils.tensor_utils import tensor_tree_map

    return tensor_tree_map(lambda tensor: tensor.clone(), batch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("eager", "compile"), required=True)
    parser.add_argument(
        "--precision",
        choices=("fp16", "bf16"),
        default="bf16",
        help="reference autocast precision (default: bf16, the qualified accuracy oracle)",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.warmup < 1 or arguments.iterations < 1:
        parser.error("--warmup and --iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("the OpenFold3 reference benchmark requires CUDA")

    from openfold3.core.metrics.aggregate_confidence_ranking import get_confidence_scores
    from openfold3.core.utils.tensor_utils import tensor_tree_map
    from openfold3.projects.of3_all_atom.model import OpenFold3
    from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry

    project_entry = OF3ProjectEntry()
    # Keep both reference baselines on the same resident execution path. The
    # upstream ``low_mem`` preset uses Python refcount assertions around tensor
    # offload, which are intentionally incompatible with torch.compile wrappers.
    config = project_entry.get_model_config_with_presets(["predict"])
    config.unlock()
    config.architecture.shared.diffusion.no_full_rollout_samples = 1
    config.lock()
    model = OpenFold3(config)
    state = torch.load(arguments.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.cuda().eval()
    if arguments.mode == "compile":
        # Dynamo intentionally bypasses functools.cache wrappers while tracing.
        # OpenFold3's cached NumPy-to-tensor quaternion lookup then mutates a
        # guarded global in the frame that created the guard. Keep this tiny
        # constant lookup eager so the remainder of the model can be compiled.
        from openfold3.core.utils import rigid_utils

        rigid_utils._get_quat = torch.compiler.disable(rigid_utils._get_quat)
        model = torch.compile(model, dynamic=False)
    _seed()
    cpu_batch = _load_batch(arguments.query, arguments.components, arguments.features)
    gpu_batch = tensor_tree_map(lambda tensor: tensor.cuda(), cpu_batch)

    def predict() -> tuple[dict, dict]:
        _seed()
        batch, outputs = model(_fresh_batch(gpu_batch))
        outputs["confidence_scores"] = get_confidence_scores(
            batch=batch,
            outputs=outputs,
            config=config,
            compute_per_sample=False,
        )
        return batch, outputs

    autocast_dtype = torch.float16 if arguments.precision == "fp16" else torch.bfloat16
    with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype):
        for _ in range(arguments.warmup):
            predict()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        latencies: list[float] = []
        for _ in range(arguments.iterations):
            started = time.perf_counter()
            _, outputs = predict()
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - started)
        if not torch.isfinite(outputs["atom_positions_predicted"]).all():
            raise RuntimeError("OpenFold3 reference produced non-finite coordinates")

    document = {
        "schema_version": 1,
        "backend": f"pytorch-{arguments.mode}",
        "precision": f"{arguments.precision}-mixed",
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "token_count": int(gpu_batch["token_mask"].shape[-1]),
        "atom_count": int(gpu_batch["atom_mask"].shape[-1]),
        "warmup_runs": arguments.warmup,
        "measured_runs": arguments.iterations,
        "latency_ms": [value * 1000.0 for value in latencies],
        "mean_latency_ms": float(np.mean(latencies) * 1000.0),
        "p50_latency_ms": float(np.median(latencies) * 1000.0),
        "min_latency_ms": float(np.min(latencies) * 1000.0),
        "max_latency_ms": float(np.max(latencies) * 1000.0),
        "stddev_latency_ms": float(np.std(latencies) * 1000.0),
        "throughput_samples_per_second": float(1.0 / np.mean(latencies)),
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
        "engine_or_compile_setup_excluded": True,
        "preprocessing_and_checkpoint_load_excluded": True,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(document, sort_keys=True))


if __name__ == "__main__":
    main()
