# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and qualify the original native HSTU provider through public C++ APIs.

Both source checkouts are explicit inputs; this command never downloads them.
Results qualify seeded model correctness, artifact packaging and cache behavior,
not trained recommendation quality, FlexKV integration or serving performance.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import struct
import subprocess

if not __debug__:
    raise RuntimeError(
        "HSTU native qualification requires Python assertions; "
        "run without -O, -OO, or PYTHONOPTIMIZE"
    )

from families.hstu.native_attention_build import (
    native_attention_notices, source_directory, verify_source,
)
from families.hstu.tests.cache_e2e import _append, _history, _keyed, _request_step
from families.hstu.tests.fixtures import ACTION_IDS, ITEM_IDS, SEED, make_checkpoint
from families.hstu.tests.reference import reference_receipt, run_reference
from families.hstu.tests.session_e2e import _check_result, _verify_session_transitions
from tensorrt_model_connect import BuildRequest, build
from tensorrt_model_connect.build import content_cache_key
from tensorrt_model_connect.bundle_writer import BUNDLE_MAGIC


HERE = Path(__file__).resolve().parent


def _json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sections(bundle):
    """Read the public bundle framing without loading a native library."""
    data = bundle.read_bytes()
    assert data.startswith(BUNDLE_MAGIC)
    start = len(BUNDLE_MAGIC) + 8
    size = struct.unpack_from("<Q", data, len(BUNDLE_MAGIC))[0]
    assert 0 < size <= len(data) - start
    header = json.loads(data[start:start + size])
    assert (header["family"], header["task"], header["backend"]) == ("hstu", "recommendation", "trt")
    payload = data[start + size:]
    result = {}
    for name, extent in header["sections"].items():
        offset, length = extent["offset"], extent["length"]
        assert 0 <= offset <= len(payload) and 0 <= length <= len(payload) - offset
        result[name] = payload[offset:offset + length]
    return result


def _no_absolute_paths(value):
    if isinstance(value, dict):
        for key, item in value.items():
            _no_absolute_paths(key)
            _no_absolute_paths(item)
    elif isinstance(value, list):
        for item in value:
            _no_absolute_paths(item)
    elif isinstance(value, str):
        assert not Path(value).is_absolute(), "packaged manifest contains an absolute path"


def verify_bundle(bundle, mode, private_roots):
    sections = _sections(bundle)
    manifest = json.loads(sections["attention_native.json"])
    runtime = json.loads(sections["runtime.json"])
    assert manifest["provider"] == "original_cuda_m64"
    assert manifest["attention_mode"] == mode
    assert runtime["attention_implementation"] == "nvidia_hstu"
    assert runtime["enable_history_cache"] == (mode == "paged")
    assert "native_kernel_source" not in runtime
    assert sections["attention_native.NOTICE"] == native_attention_notices(manifest["original"])
    assert manifest["notices_content_key"] == content_cache_key(
        "hstu-native-notices-v1", sections["attention_native.NOTICE"])
    assert manifest["library_content_key"] == content_cache_key(
        "hstu-native-library-v1", sections["attention_native.so"])
    _no_absolute_paths(manifest)
    _no_absolute_paths(runtime)
    # Inspect generated binary bytes as well as JSON: debug/line information can
    # disclose compiler input paths even when the manifest has been sanitized.
    for name in ("attention_native.json", "attention_native.so", "runtime.json"):
        for root in private_roots:
            assert str(root.resolve()).encode() not in sections[name], (name, "private build path")
    return {"bundle_content_key": content_cache_key("hstu-qualified-bundle-v1", bundle.read_bytes()),
            "section_content_keys": {name: content_cache_key("hstu-qualified-section-v1", name.encode(), value)
                                     for name, value in sections.items()},
            "provider": manifest["provider"], "mode": mode,
            "notices_verified": True, "private_paths_absent": True}


def _execute(binary, runtime_root, bundle, payload, directory):
    directory.mkdir()
    request, output = directory / "request.json", directory / "result.json"
    _json(request, payload)
    command = [str(binary), "--bundle", str(bundle), "--runtime-root", str(runtime_root),
               "--input-json", str(request), "--output-json", str(output)]
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(filter(None, (
        str(runtime_root), environment.get("LD_LIBRARY_PATH", ""))))
    completed = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=300)
    (directory / "native.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    _json(directory / "command.json", {"argv": command, "returncode": completed.returncode})
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(output.read_text())
    assert "error" not in result, result.get("error")
    return result


def _request(batch):
    sequences = []
    for user in range(batch):
        length = 200 - 3 * user
        sequences.append(_keyed({
            "history_item_ids": [ITEM_IDS[(index + user) % len(ITEM_IDS)] for index in range(length)],
            "history_action_ids": [ACTION_IDS[(index + user) % len(ACTION_IDS)] for index in range(length)],
            "candidate_item_ids": [ITEM_IDS[(index + user + 7) % len(ITEM_IDS)] for index in range(256)],
        }, f"user-{user}"))
    return {"sequences": sequences}


def _trace(request, graphs):
    initial = request["sequences"]
    appended = [_append(sequence) for sequence in initial]
    candidates = deepcopy(initial)
    for sequence in candidates:
        sequence["candidate_item_ids"].reverse()
    readonly = [_keyed(sequence, f"readonly-{index}", read_only=True)
                for index, sequence in enumerate(appended)]
    return {
        "cuda_graphs": graphs,
        "cache": {"max_bytes": 64 << 20, "max_entries": 32,
                  "storage_max_bytes": 64 << 20, "write_through": True},
        "steps": [
            _request_step("cold", *initial), _request_step("hit", *initial),
            _request_step("candidate-change", *candidates), _request_step("append", *appended),
            _request_step("append-hit", *appended),
            _request_step("readonly", *readonly), _request_step("readonly-again", *readonly),
            {"name": "clear-memory", "action": "clear_memory"},
            _request_step("storage", *appended),
            {"name": "invalidate", "action": "invalidate", "identity": appended[0]["cache"]},
            _request_step("invalidated", *appended),
        ],
    }


def _compare(actual, expected, thresholds, label, comparisons):
    assert len(actual["sequences"]) == len(expected["sequences"]), label
    for index, (left, right) in enumerate(zip(actual["sequences"], expected["sequences"])):
        comparisons.extend(_check_result(left, right, thresholds, f"{label}/{index}"))


def _check_cache(audit, trace, initial, oracle, thresholds, comparisons, label):
    assert len(audit["steps"]) == len(trace["steps"])
    for step, row in zip(trace["steps"], audit["steps"]):
        assert row["name"] == step["name"]
        if "action" in step:
            continue
        assert "error" not in row and "baseline_error" not in row, row
        expected = oracle(step["request"])
        for path in ("actual", "baseline"):
            _compare(row[path], expected, thresholds, f"{label}/{row['name']}/{path}", comparisons)
        for index, (actual, baseline) in enumerate(zip(row["actual"]["sequences"], row["baseline"]["sequences"])):
            assert baseline["cache"]["source"] == "disabled"
            report = actual["cache"]
            history = _history(initial[index])
            if step["name"] in {"hit", "candidate-change", "append"}:
                reused = history
            elif step["name"] in {"append-hit", "storage", "invalidated"}:
                reused = history + 2
            else:
                reused = 0
            if step["name"] == "invalidated" and index == 0:
                reused = 0
            sequence = step["request"]["sequences"][index]
            assert report["reused_history_tokens"] == reused, (label, step["name"], index, report)
            assert report["history_tokens"] == _history(sequence)
            assert report["computed_tokens"] == _history(sequence) + 256 - reused
            if step["name"] == "cold":
                assert report["source"] == "miss" and report["published"]
            if step["name"] == "append":
                assert report["reason"] == "append_history"
            if step["name"] == "storage":
                assert report["source"] == "storage"
            if step["name"].startswith("readonly"):
                assert not report["published"]
                assert row["stats_before"]["publications"] == row["stats_after"]["publications"]
    assert audit["stats"]["storage_hits"] >= len(initial)
    assert audit["stats"]["load_failures"] == audit["stats"]["store_failures"] == 0


def run(source, reference_source, runtime_root, output):
    source, reference_source, runtime_root, output = (
        path.resolve() for path in (source, reference_source, runtime_root, output))
    original = verify_source(source_directory(source))
    reference = reference_receipt(reference_source)
    binaries = {name: runtime_root / name for name in
                ("trtmc-hstu", "hstu_cache_sequence_runner", "hstu_session_runner")}
    for binary in binaries.values():
        assert binary.is_file() and os.access(binary, os.X_OK), f"missing native runner: {binary}"
    output.mkdir(parents=True, exist_ok=False)
    thresholds = json.loads((HERE / "thresholds/hstu-cache-hundreds-bf16.json").read_text())["threshold_overrides"]
    receipt = {"seed": SEED, "source": original, "reference": reference,
               "thresholds": thresholds, "bundles": {}, "comparisons": [],
               "scope": "public C++ correctness and packaging; no latency or FlexKV claim"}
    bundles, checkpoints = {}, {}
    for mode in ("dense", "paged"):
        checkpoint = output / f"checkpoint-{mode}"
        config = make_checkpoint(
            checkpoint, hidden_size=256, num_heads=4, head_dim=64, num_layers=2,
            max_sequence_length=1024, position_buckets=1024, scaling_seqlen=1024,
            time_buckets=0, prediction_head=[128, 2], enable_history_cache=mode == "paged",
            embedding_tables=[{"name": "item", "role": "item", "num_embeddings": len(ITEM_IDS)},
                              {"name": "action", "role": "action", "num_embeddings": len(ACTION_IDS)}],
            attention_implementation="nvidia_hstu", native_kernel_source=str(source),
        )
        bundle = output / f"{mode}.bundle"
        print(f"Building {mode} native bundle", flush=True)
        build(BuildRequest(model_dir=checkpoint, output_path=bundle, family="hstu",
                           task="recommendation", precision="bf16", max_batch_size=8))
        receipt["bundles"][mode] = verify_bundle(
            bundle, mode, (source, output, HERE.parents[2]))
        bundles[mode], checkpoints[mode] = bundle, checkpoint
    assert (checkpoints["dense"] / "model.safetensors").read_bytes() == (
        checkpoints["paged"] / "model.safetensors").read_bytes()
    memo = {}

    def oracle(request):
        inputs = deepcopy(request)
        for sequence in inputs["sequences"]:
            sequence.pop("cache", None)
        key = json.dumps(inputs, sort_keys=True)
        if key not in memo:
            memo[key] = run_reference(checkpoints["dense"], inputs,
                                      upstream_root=reference_source, precision="bf16")
            _json(output / f"oracle-{len(memo):03d}.json", {"request": inputs, "result": memo[key]})
        return memo[key]

    for batch in (1, 8):
        request = _request(batch)
        label = f"dense-b{batch}"
        actual = _execute(binaries["trtmc-hstu"], runtime_root, bundles["dense"], request, output / label)
        _compare(actual, oracle(request), thresholds, label, receipt["comparisons"])
        for graphs in (False, True):
            label = f"paged-b{batch}-graphs{int(graphs)}"
            print(f"Checking {label}", flush=True)
            trace = _trace(request, graphs)
            audit = _execute(binaries["hstu_cache_sequence_runner"], runtime_root,
                             bundles["paged"], trace, output / label)
            _check_cache(audit, trace, request["sequences"], oracle, thresholds,
                         receipt["comparisons"], label)
    initial = _request(1)["sequences"][0]
    candidates = initial["candidate_item_ids"]
    initial["candidate_item_ids"] = []
    settings = {"initial_history": initial, "candidate_item_ids": candidates,
                "fork_item_ids": [ITEM_IDS[5], ITEM_IDS[6]], "append_action_id": ACTION_IDS[4],
                "num_steps": 20, "cache_max_bytes": 64 << 20, "cuda_graphs": True, **thresholds}
    print("Checking native sessions and branches", flush=True)
    audit = _execute(binaries["hstu_session_runner"], runtime_root, bundles["paged"],
                     settings, output / "session")
    for row in audit["steps"]:
        for path in ("actual", "baseline"):
            _compare(row[path], oracle(row["request"]), thresholds,
                     f"session/{row['name']}/{path}", receipt["comparisons"])
    _verify_session_transitions(audit, initial, candidates, config)
    receipt.update(complete=True, batches=[1, 8], cache_graph_modes=[False, True], session_steps=20)
    _json(output / "qualification.json", receipt)
    print(f"Passed {len(receipt['comparisons'])} output-field comparisons: {output / 'qualification.json'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Pinned FBGEMM checkout with CUTLASS initialized")
    parser.add_argument("--reference-source", required=True, type=Path, help="Pinned NVIDIA recsys-examples checkout")
    parser.add_argument("--runtime-root", required=True, type=Path, help="Built C++ runtime and HSTU test runners")
    parser.add_argument("--output", required=True, type=Path, help="New local directory for bundles, requests and evidence")
    args = parser.parse_args(argv)
    run(args.source, args.reference_source, args.runtime_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
