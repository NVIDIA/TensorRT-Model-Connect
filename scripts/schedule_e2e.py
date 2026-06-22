#!/usr/bin/env python3
"""Schedule E2E tests across GPUs and workers for balanced load.

Reads pytest test IDs (one per line on stdin), model manifests, and optional
timing estimates, then distributes work across GPU worker slots by estimated
critical-path load.

Usage:
    pytest tests/test_e2e.py --co -q | grep test_e2e | \\
        python scripts/schedule_e2e.py --num-gpus 4 --workers-per-gpu 4

Output (JSON to stdout):
    {
      "0": [["tests/...test_e2e[model-a]", ...], ["tests/...test_e2e[model-b]", ...], ...],
      "1": [["tests/...test_e2e[model-c]", ...], ...],
      ...
    }
    Key = GPU index, value = list of worker queues (each queue is a list of test IDs).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Size classification
# ---------------------------------------------------------------------------

# Strategies that are inherently GPU-heavy regardless of param count.
_HEAVY_STRATEGIES = frozenset({
    "vision_language",
    "speech_to_speech",
})

_EXCLUSIVE_GPU_RESOURCE = "exclusive_gpu"
_SMALL_TEST_WEIGHT = 90.0
_LARGE_TEST_WEIGHT = 300.0
_EXCLUSIVE_GPU_TEST_WEIGHT = 900.0
_TIMING_ESTIMATES_FILE = "timing_estimates.json"


def _param_billions(hf_id: str) -> float | None:
    """Extract approximate parameter count in billions from HF ID string."""
    # Match patterns like "7B", "0.6B", "30B-A3B", "1.3B", "350M"
    for m in re.finditer(r"(\d+\.?\d*)\s*([BbMm])", hf_id):
        val = float(m.group(1))
        unit = m.group(2).upper()
        if unit == "M":
            return val / 1000.0
        return val
    return None


def classify_size(manifest: dict) -> str:
    """Classify a model as 'large' or 'small' based on manifest metadata.

    Large = likely to use significant GPU memory (>= 3B params, or heavy
    strategy like diffusion/VL/speech-to-speech).
    """
    strategy = str(manifest.get("runtime_strategy", "") or "")
    hf_id = str(manifest.get("hf_id", "") or "")

    # Heavy strategies are always large. Diffusion strategies are named by
    # backend family, e.g. diffusion_flux and diffusion_ltx.
    if strategy == "diffusion" or strategy.startswith("diffusion_") or strategy in _HEAVY_STRATEGIES:
        return "large"

    # MoE with real weights (not the 15M toy)
    if strategy == "decoder_moe":
        params = _param_billions(hf_id)
        if params is not None and params >= 1.0:
            return "large"

    # Hybrid models (Nemotron-H)
    if strategy == "hybrid_mamba_attention":
        return "large"

    # Check param count from HF ID
    params = _param_billions(hf_id)
    if params is not None and params >= 3.0:
        return "large"

    # bark-large has "bark" in the name but no size suffix — treat as large
    name = str(manifest.get("name", "") or "")
    if "bark-large" in name:
        return "large"

    return "small"


def classify_parallel_resource(manifest: dict) -> str:
    """Return the requested parallel scheduling resource tier."""
    if manifest.get("e2e_parallel_resource") == _EXCLUSIVE_GPU_RESOURCE:
        return _EXCLUSIVE_GPU_RESOURCE
    return "shared"


def _load_manifests(manifest_dir: Path) -> dict[str, dict]:
    manifests: dict[str, dict] = {}
    manifest_paths = {
        *manifest_dir.glob("*.json"),
        *manifest_dir.glob("*/manifests/*.json"),
    }
    for f in sorted(manifest_paths):
        try:
            m = json.loads(f.read_text())
            manifests[m["name"]] = m
        except (json.JSONDecodeError, KeyError):
            continue
    return manifests


def _load_timing_estimates(path: Path | None) -> dict[str, float]:
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict) and isinstance(data.get("estimates_s"), dict):
        data = data["estimates_s"]
    if not isinstance(data, dict):
        return {}

    estimates: dict[str, float] = {}
    for name, value in data.items():
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            continue
        if seconds > 0:
            estimates[str(name)] = seconds
    return estimates


def _default_timing_estimates_path(manifest_dir: Path) -> Path:
    return manifest_dir.parent / _TIMING_ESTIMATES_FILE


def _model_name_from_test_id(test_id: str) -> str:
    match = re.search(r"\[(.+?)\]", test_id)
    return match.group(1) if match else ""


def _bundle_group_key(test_id: str, manifests: dict[str, dict]) -> str:
    """Return a stable scheduling key for tests that build the same bundle."""
    manifest = manifests.get(_model_name_from_test_id(test_id), {})
    bundle = str(manifest.get("bundle", "") or "").strip()
    if bundle:
        return f"bundle:{bundle}"
    return f"single:{test_id}"


def _group_by_bundle(
    test_ids: list[str],
    manifests: dict[str, dict],
) -> list[tuple[int, list[str]]]:
    """Group tests that share a bundle so they stay in one worker queue."""
    groups: list[tuple[int, list[str]]] = []
    key_to_group: dict[str, int] = {}
    for idx, test_id in enumerate(test_ids):
        key = _bundle_group_key(test_id, manifests)
        if key not in key_to_group:
            key_to_group[key] = len(groups)
            groups.append((idx, []))
        groups[key_to_group[key]][1].append(test_id)
    return groups


def split_by_parallel_resource(
    test_ids: list[str],
    manifest_dir: Path,
) -> tuple[list[str], list[str]]:
    """Split tests into exclusive-GPU and shared-resource groups."""
    manifests = _load_manifests(manifest_dir)
    exclusive_tests: list[str] = []
    shared_tests: list[str] = []
    for tid in test_ids:
        manifest = manifests.get(_model_name_from_test_id(tid), {})
        if classify_parallel_resource(manifest) == _EXCLUSIVE_GPU_RESOURCE:
            exclusive_tests.append(tid)
        else:
            shared_tests.append(tid)
    return exclusive_tests, shared_tests


def _test_weight(
    test_id: str,
    manifests: dict[str, dict],
    timing_estimates: dict[str, float] | None = None,
) -> float:
    name = _model_name_from_test_id(test_id)
    if timing_estimates and name in timing_estimates:
        return timing_estimates[name]

    manifest = manifests.get(name, {})
    if classify_parallel_resource(manifest) == _EXCLUSIVE_GPU_RESOURCE:
        return _EXCLUSIVE_GPU_TEST_WEIGHT
    if classify_size(manifest) == "large":
        return _LARGE_TEST_WEIGHT
    return _SMALL_TEST_WEIGHT


def _estimated_gpu_loads(
    assignments: dict[str, list[list[str]]],
    manifest_dir: Path,
    timing_estimates: dict[str, float] | None = None,
) -> dict[int, int]:
    """Estimate per-GPU schedule load for phase-aware balancing."""
    manifests = _load_manifests(manifest_dir)
    loads: dict[int, int] = {}
    for gpu_id, workers in assignments.items():
        loads[int(gpu_id)] = int(sum(
            _test_weight(test_id, manifests, timing_estimates)
            for worker in workers
            for test_id in worker
        ))
    return loads


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


def schedule(
    test_ids: list[str],
    manifest_dir: Path,
    num_gpus: int,
    workers_per_gpu: int,
    gpu_load_offsets: dict[int, int] | None = None,
    timing_estimates: dict[str, float] | None = None,
) -> dict[str, list[list[str]]]:
    """Produce balanced GPU×worker assignments.

    Algorithm:
    1. Classify exclusive-GPU tests and reserve whole-GPU workers for them.
    2. Weight each test by observed timing estimates when available, otherwise
       by coarse manifest size.
    3. Assign tests longest-processing-time first to the worker slot with the
       lowest estimated end-to-end load, including any prior exclusive-GPU
       offset for that GPU.
    """
    manifests = _load_manifests(manifest_dir)
    if timing_estimates is None:
        timing_estimates = _load_timing_estimates(
            _default_timing_estimates_path(manifest_dir))

    # Classify
    exclusive_tests: list[str] = []
    large_tests: list[str] = []
    small_tests: list[str] = []
    for tid in test_ids:
        # Extract model name from "tests/test_e2e.py::test_e2e[model-name]"
        name = _model_name_from_test_id(tid)
        m = manifests.get(name, {})
        if classify_parallel_resource(m) == _EXCLUSIVE_GPU_RESOURCE:
            exclusive_tests.append(tid)
        elif classify_size(m) == "large":
            large_tests.append(tid)
        else:
            small_tests.append(tid)

    def weight(tid: str) -> float:
        return _test_weight(tid, manifests, timing_estimates)

    def group_weight(group: tuple[int, list[str]]) -> float:
        return sum(weight(tid) for tid in group[1])

    # Reserve whole GPUs for exclusive tests. This keeps high-memory build
    # cases from colliding with unrelated workers while preserving parallelism
    # on the remaining GPUs.
    result: dict[str, list[list[str]]] = {}
    reserved_gpu_count = min(len(exclusive_tests), num_gpus)
    if exclusive_tests:
        exclusive_slots = [
            [0.0, gpu_id, 0, []]
            for gpu_id in range(max(reserved_gpu_count, 1))
        ]
        grouped_exclusive = _group_by_bundle(exclusive_tests, manifests)
        for group in sorted(grouped_exclusive, key=lambda item: (-group_weight(item), item[0])):
            slot = min(exclusive_slots, key=lambda item: (item[0], item[1]))
            slot[0] += group_weight(group)
            slot[3].extend(group[1])
        for _, gpu_id, _, tests_for_gpu in exclusive_slots:
            if tests_for_gpu:
                result[str(gpu_id)] = [tests_for_gpu]

    shared_gpu_ids = list(range(reserved_gpu_count, num_gpus))
    if not shared_gpu_ids:
        shared_gpu_ids = list(range(num_gpus))
        for gpu_id in shared_gpu_ids:
            result.setdefault(str(gpu_id), [[]])

    # Schedule shared work using longest-processing-time first across worker
    # slots.  Each slot starts with the prior exclusive-GPU load for that GPU,
    # matching the pipelined runner: shared workers on a GPU can only launch
    # after that GPU's exclusive queue completes.
    slots: list[list[object]] = []
    for gpu_id in shared_gpu_ids:
        for worker_idx in range(workers_per_gpu):
            slots.append([
                float((gpu_load_offsets or {}).get(gpu_id, 0)),
                gpu_id,
                worker_idx,
                [],
            ])

    grouped_shared = _group_by_bundle([*large_tests, *small_tests], manifests)
    for group in sorted(grouped_shared, key=lambda item: (-group_weight(item), item[0])):
        slot = min(slots, key=lambda item: (item[0], item[1], item[2]))
        slot[0] = float(slot[0]) + group_weight(group)
        slot[3].extend(group[1])

    for gpu_id in shared_gpu_ids:
        workers: list[list[str]] = [[] for _ in range(workers_per_gpu)]
        for _, slot_gpu_id, worker_idx, worker_tests in slots:
            if slot_gpu_id == gpu_id:
                workers[int(worker_idx)] = list(worker_tests)
        shared_workers = [w for w in workers if w]
        if str(gpu_id) in result:
            result[str(gpu_id)][0].extend([tid for worker in shared_workers for tid in worker])
        else:
            result[str(gpu_id)] = shared_workers

    return result


def schedule_phases(
    test_ids: list[str],
    manifest_dir: Path,
    num_gpus: int,
    workers_per_gpu: int,
    timing_estimates: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    """Schedule exclusive-GPU tests before shared tests.

    `exclusive_gpu` manifests need a whole GPU while they run. When all GPUs
    are reserved by that set, mixing shared tests into the same worker queues
    serializes the rest of the suite. Running two phases preserves the resource
    contract while letting shared tests use the full worker fan-out afterward.
    """
    exclusive_tests, shared_tests = split_by_parallel_resource(test_ids, manifest_dir)
    if timing_estimates is None:
        timing_estimates = _load_timing_estimates(
            _default_timing_estimates_path(manifest_dir))
    phases: list[dict[str, object]] = []
    exclusive_gpu_loads: dict[int, int] = {}
    if exclusive_tests:
        exclusive_schedule = schedule(
            exclusive_tests,
            manifest_dir,
            num_gpus=num_gpus,
            workers_per_gpu=1,
            timing_estimates=timing_estimates,
        )
        exclusive_gpu_loads = _estimated_gpu_loads(
            exclusive_schedule, manifest_dir, timing_estimates)
        phases.append({
            "name": "exclusive_gpu",
            "schedule": exclusive_schedule,
        })
    if shared_tests:
        phases.append({
            "name": "shared",
            "schedule": schedule(
                shared_tests,
                manifest_dir,
                num_gpus=num_gpus,
                workers_per_gpu=workers_per_gpu,
                gpu_load_offsets=exclusive_gpu_loads,
                timing_estimates=timing_estimates,
            ),
        })
    return phases


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument(
        "--split-exclusive-phases",
        action="store_true",
        help="emit sequential exclusive_gpu and shared phases",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tests" / "e2e" / "models",
    )
    parser.add_argument(
        "--timing-estimates",
        type=Path,
        default=None,
        help="JSON model-name to estimated seconds mapping",
    )
    args = parser.parse_args()
    timing_estimates_path = (
        args.timing_estimates
        if args.timing_estimates is not None
        else _default_timing_estimates_path(args.manifest_dir)
    )
    timing_estimates = _load_timing_estimates(timing_estimates_path)

    # Read test IDs from stdin
    test_ids = [line.strip() for line in sys.stdin if line.strip()]
    if not test_ids:
        print("ERROR: No test IDs on stdin.", file=sys.stderr)
        sys.exit(1)

    if args.split_exclusive_phases:
        phases = schedule_phases(
            test_ids,
            args.manifest_dir,
            args.num_gpus,
            args.workers_per_gpu,
            timing_estimates=timing_estimates,
        )
        print(
            f"Schedule: {len(test_ids)} tests across {args.num_gpus} GPUs "
            f"x {args.workers_per_gpu} workers/GPU in {len(phases)} phase(s)",
            file=sys.stderr,
        )
        if timing_estimates:
            print(
                f"  Timing estimates: {timing_estimates_path} "
                f"({len(timing_estimates)} models)",
                file=sys.stderr,
            )
        for phase in phases:
            name = str(phase["name"])
            schedule_for_phase = phase["schedule"]
            assert isinstance(schedule_for_phase, dict)
            phase_total = sum(
                len(worker)
                for workers in schedule_for_phase.values()
                for worker in workers
            )
            phase_workers = sum(len(workers) for workers in schedule_for_phase.values())
            print(
                f"  Phase {name}: {phase_total} tests across {phase_workers} workers",
                file=sys.stderr,
            )
            _print_schedule_summary(
                schedule_for_phase,
                args.manifest_dir,
                timing_estimates,
            )
        json.dump({"phases": phases}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    assignments = schedule(
        test_ids,
        args.manifest_dir,
        args.num_gpus,
        args.workers_per_gpu,
        timing_estimates=timing_estimates,
    )

    print(f"Schedule: {len(test_ids)} tests across {args.num_gpus} GPUs "
          f"x {args.workers_per_gpu} workers/GPU", file=sys.stderr)
    if timing_estimates:
        print(
            f"  Timing estimates: {timing_estimates_path} "
            f"({len(timing_estimates)} models)",
            file=sys.stderr,
        )
    _print_schedule_summary(assignments, args.manifest_dir, timing_estimates)

    # Output JSON to stdout
    json.dump(assignments, sys.stdout, indent=2)
    sys.stdout.write("\n")


def _print_schedule_summary(
    assignments: dict[str, list[list[str]]],
    manifest_dir: Path,
    timing_estimates: dict[str, float] | None = None,
) -> None:
    """Print a human-readable schedule summary to stderr."""
    total_large = 0
    total_small = 0
    total_estimated = 0.0
    manifests = _load_manifests(manifest_dir)

    for gpu_id, workers in sorted(assignments.items(), key=lambda x: int(x[0])):
        n_tests = sum(len(w) for w in workers)
        n_large = 0
        n_exclusive = 0
        estimated = 0.0
        for w in workers:
            for tid in w:
                name = _model_name_from_test_id(tid)
                manifest = manifests.get(name, {})
                if classify_parallel_resource(manifest) == _EXCLUSIVE_GPU_RESOURCE:
                    n_exclusive += 1
                if classify_size(manifest) == "large":
                    n_large += 1
                estimated += _test_weight(tid, manifests, timing_estimates)
        n_small = n_tests - n_large
        total_large += n_large
        total_small += n_small
        total_estimated += estimated
        print(f"  GPU {gpu_id}: {n_tests} tests ({n_large}L + {n_small}S) "
              f"across {len(workers)} workers"
              f"{f' [{n_exclusive} exclusive]' if n_exclusive else ''}"
              f", estimated {estimated / 60:.1f}m",
              file=sys.stderr)
    print(
        f"  Total: {total_large} large + {total_small} small, "
        f"estimated {total_estimated / 60:.1f}m",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
