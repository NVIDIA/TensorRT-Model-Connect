# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure the reference's complete request path in eager PyTorch or with torch.compile.

One timed call runs what one native ``predict_structure`` Task call runs: featurization of the sequence,
``EsmFold2Model.fold`` (trunk, diffusion sampler, confidence head), the ranking of all samples by pTM and the
structure text of every sample (``output_to_pdb``, which copies the coordinates to the host; the reference has no
mmCIF writer). The call starts and ends with a CUDA synchronization. Model loading and the warmup calls (the first
one compiles in compile mode) are excluded. Peak memory is the device's ``memory.used`` from ``nvidia-smi``, sampled
every 100 ms over each length's calls, the counter the native benchmark's telemetry reads.

The sequence of length L is drawn uniformly over the 20 standard residues with ``numpy.random.default_rng(L)``;
``--write-fasta`` writes the same sequences for the native benchmark (``trtmc-bench``).

    python -m families.esmfold2.tests.benchmark_reference --checkpoint <snapshot> --mode eager \
        --lengths 128,256,512 --output reference.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

RESIDUES = "ACDEFGHIKLMNPQRSTVWY"


def benchmark_sequence(length: int) -> str:
    """The deterministic benchmark sequence of ``length`` residues."""

    if length < 2:
        raise ValueError("a benchmark sequence needs at least 2 residues")
    rng = np.random.default_rng(length)
    return "".join(RESIDUES[index] for index in rng.integers(0, len(RESIDUES), size=length))


class DeviceMemorySampler:
    """Largest ``nvidia-smi`` ``memory.used`` (MiB) of one device, named by its UUID, while the context is open.

    ``samples`` counts the queries that returned a number; without one there is no peak to report.
    """

    def __init__(self, uuid: str, interval: float = 0.1) -> None:
        self.uuid = uuid
        self.interval = interval
        self.peak_mib = 0.0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        try:
            completed = subprocess.run(
                ["nvidia-smi", f"--id={self.uuid}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False, timeout=30,
            )
            value = float(completed.stdout.split()[0]) if completed.returncode == 0 else math.nan
        except (OSError, subprocess.TimeoutExpired, IndexError, ValueError):
            return
        if math.isfinite(value):
            self.peak_mib = max(self.peak_mib, value)
            self.samples += 1

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def __enter__(self) -> DeviceMemorySampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def _lengths(text: str) -> list[int]:
    values = [int(item) for item in text.split(",") if item.strip()]
    if not values or any(value < 2 for value in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("expected distinct lengths of at least 2 residues, comma separated")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path, help="local snapshot of biohub/ESMFold2(-Fast)")
    parser.add_argument("--mode", choices=("eager", "compile"), required=True)
    parser.add_argument("--lengths", required=True, type=_lengths)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--write-fasta", type=Path, help="also write len<L>.fasta for the native benchmark")
    parser.add_argument("--loops", type=_positive, default=3)
    parser.add_argument("--steps", type=_positive, default=50)
    parser.add_argument("--samples", type=_positive, default=32)
    parser.add_argument("--warmup", type=_positive, default=2)
    parser.add_argument("--iterations", type=_positive, default=5)
    args = parser.parse_args()

    if args.write_fasta is not None:
        args.write_fasta.mkdir(parents=True, exist_ok=True)
        for length in args.lengths:
            (args.write_fasta / f"len{length}.fasta").write_text(f">random_{length}\n{benchmark_sequence(length)}\n")

    import torch
    import transformers
    from transformers.models.esmfold2.modeling_esmfold2 import EsmFold2Model
    from transformers.models.esmfold2.protein_utils import output_to_pdb, prepare_protein_features

    if not torch.cuda.is_available():
        raise SystemExit("the reference benchmark needs a CUDA device")
    device = torch.cuda.current_device()
    # nvidia-smi numbers devices without CUDA_VISIBLE_DEVICES, so the sampler names the device by its UUID.
    uuid = f"GPU-{torch.cuda.get_device_properties(device).uuid}"
    load_start = time.perf_counter()
    model = EsmFold2Model.from_pretrained(args.checkpoint, dtype=torch.bfloat16, local_files_only=True)
    model = model.to("cuda").eval()
    if args.mode == "compile":
        torch._dynamo.config.cache_size_limit = 64
        model.compile()
        model.structure_head.compile()
        model.confidence_head.compile()
    torch.cuda.synchronize()
    environment = {
        "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device), "mode": args.mode, "load_seconds": time.perf_counter() - load_start,
        "loops": args.loops, "steps": args.steps, "samples": args.samples, "warmup": args.warmup,
        "iterations": args.iterations,
    }
    print(json.dumps(environment), flush=True)

    def request(sequence: str) -> tuple[float, list[str]]:
        torch.cuda.synchronize()
        start = time.perf_counter()
        features = prepare_protein_features(sequence, device="cuda")
        output = model.fold(**features, num_loops=args.loops, num_sampling_steps=args.steps,
                            num_diffusion_samples=args.samples)
        # The native call rejects non-finite outputs too, so the check is timed on both sides.
        keys = ("sample_atom_coords", "plddt", "ptm", "complex_plddt")
        if not bool(torch.stack([torch.isfinite(output[key]).all() for key in keys]).all()):
            raise RuntimeError("the reference returned non-finite coordinates or confidences")
        order = torch.argsort(output["ptm"].float().reshape(-1), descending=True).tolist()
        if len(order) != args.samples:
            raise RuntimeError(f"the reference returned {len(order)} samples, expected {args.samples}")
        structures = [output_to_pdb(output, features, sample_idx=sample) for sample in order]
        torch.cuda.synchronize()
        return time.perf_counter() - start, structures

    results = {"environment": environment, "lengths": []}
    for length in args.lengths:
        sequence = benchmark_sequence(length)
        record: dict = {"residues": length}
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.no_grad(), DeviceMemorySampler(uuid) as sampler:
                record["warmup_seconds"] = [request(sequence)[0] for _ in range(args.warmup)]
                timed = [request(sequence) for _ in range(args.iterations)]
            record["seconds"] = [seconds for seconds, _ in timed]
            # Structure texts of one call, one per sample (a native request returns all of them too).
            record["pdb_texts_per_call"] = len(timed[-1][1])
            record["p50_ms"] = 1000 * statistics.median(record["seconds"])
            record["samples_per_s"] = args.samples * 1000 / record["p50_ms"]
            record["memory_samples"] = sampler.samples
            record["peak_device_mib"] = sampler.peak_mib if sampler.samples else None
            record["max_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        except torch.cuda.OutOfMemoryError as error:
            record["error"] = f"out of memory: {str(error).splitlines()[0]}"
        torch.cuda.empty_cache()
        print(json.dumps({key: value for key, value in record.items() if key != "seconds"}), flush=True)
        results["lengths"].append(record)
        args.output.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
