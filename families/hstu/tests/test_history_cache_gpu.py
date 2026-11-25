# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify the inherited history lifecycle on both trained-model precisions."""

import json
import os
from pathlib import Path
import struct
import threading
import time

import numpy as np
import pytest
from tools.variant_coverage import covers_variant

from families.hstu.triton.native_client import NativeClient, snapshot_metadata
from families.hstu.config import Config
from families.hstu.reference import example_inputs

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module", params=[("fp32", 1), ("fp32", 4), ("bf16", 1), ("bf16", 4)])
def workload(request, dlrm_root, dlrm_bridge_root):
    root = dlrm_root
    runtime = os.environ["TRTMC_RUNTIME_ROOT"]
    precision, batch = request.param
    directory = Path(root) / f"dlrmv3-default-{precision}"
    return dict(
        bundle=directory / "model.bundle", runtime_root=runtime, bridge_root=dlrm_bridge_root
    ), example_inputs(Config.load(directory), batch)


def row_bytes(metadata, kv_only=False):
    specs = metadata["stages"][0]["outputs"]
    return sum(
        int(np.prod(spec["shape"][1:])) * np.dtype(spec["dtype"]).itemsize
        for spec in specs
        if not kv_only or spec["name"] in ("history_k_b", "history_v_b")
    )


def kv_bytes(metadata):
    return row_bytes(metadata, kv_only=True)


def rewrite(snapshot, **metadata):
    length = struct.unpack_from("<Q", snapshot, 8)[0]
    header = snapshot_metadata(snapshot)
    header.update(metadata)
    encoded = json.dumps(header).encode()
    return b"HSTUKV2\0" + struct.pack("<Q", len(encoded)) + encoded + snapshot[16 + length :]


def prepared(workload):
    options, inputs = workload
    with NativeClient(**options) as producer:
        names = [s["name"] for s in producer.metadata["stages"][0]["inputs"]]
        batch = next(iter(inputs.values())).shape[0]
        snapshot = producer.export_history(
            {name: inputs[name] for name in names}, ["revision-1"] * batch
        )
        baseline = producer.score(inputs)
    request = {name: value for name, value in inputs.items() if name not in names}
    return options, snapshot, baseline, request


@covers_variant(precision="fp32")
@covers_variant(precision="bf16")
def test_export_destroy_restore_then_score_skips_history_and_refreshes_candidates(workload):
    options, snapshot, baseline, request = prepared(workload)
    from families.hstu.triton.generate_repository import runtime_metadata

    assert (
        snapshot_metadata(snapshot)["bundle_id"]
        == runtime_metadata(options["bundle"], options["bridge_root"])["bundle_id"]
    )
    with NativeClient(**options, hydration=True, kv_cache_size_bytes=134217728) as consumer:
        handles = consumer.restore_history(snapshot)
        assert consumer.restore_history(snapshot) == handles
        before = consumer.stats()
        assert before["history_prepare_calls_total"] == before["decode_step_calls_total"] == 0
        assert before["resident_entries"] <= len(handles)
        revisions = snapshot_metadata(snapshot)["revisions"]
        for _ in range(2):
            output = consumer.score_with_history(handles, revisions, request)
            for name in baseline:
                np.testing.assert_array_equal(output[name], baseline[name])
        after = consumer.stats()
        assert after["history_prepare_calls_total"] == 0
        assert after["request_prepare_calls_total"] == 2 and after["decode_step_calls_total"] == 2
        assert after["history_pack_bytes_total"] == (
            0 if len(handles) == 1 else 2 * len(handles) * row_bytes(consumer.metadata)
        )
        assert after["used_bytes"] == after["active_sequences"] == 0
        assert consumer.history_info(handles[0])["leased"] is False
        # A new row order must carry its corresponding history handle.
        order = list(reversed(range(len(handles))))
        permuted = consumer.score_with_history(
            [handles[i] for i in order],
            [revisions[i] for i in order],
            {n: v[order] for n, v in request.items()},
        )
        for name in baseline:
            np.testing.assert_array_equal(permuted[name], baseline[name][order])
        # Reuse history with changed candidates; every request must prepare fresh context.
        count = consumer.metadata["candidates"]
        changed_request = {
            n: (v[:, ::-1].copy() if v.shape[1] == count else v.copy()) for n, v in request.items()
        }
        changed_full = dict(workload[1], **changed_request)
        changed_reference = consumer.score(changed_full)
        changed_cached = consumer.score_with_history(handles, revisions, changed_request)
        for name in baseline:
            np.testing.assert_array_equal(changed_cached[name], changed_reference[name])
        # Return to the previous candidates after both cached and cold bindings changed.
        again = consumer.score_with_history(handles, revisions, request)
        for name in baseline:
            np.testing.assert_array_equal(again[name], baseline[name])
        consumer.release_history(handles)
        consumer.release_history(handles)
        assert consumer.stats()["resident_entries"] == 0
        with pytest.raises(RuntimeError, match="handle"):
            consumer.score_with_history(handles, revisions, request)


def test_corrupt_wrong_identity_and_revision_fail_without_gpu_execution(workload):
    options, snapshot, baseline, request = prepared(workload)
    with NativeClient(**options, hydration=True, kv_cache_size_bytes=134217728) as consumer:
        handles = consumer.restore_history(snapshot)
        corrupt = bytearray(snapshot)
        corrupt[-1] ^= 1
        for bad in [
            bytes(corrupt),
            snapshot[:-1],
            snapshot + b"x",
            rewrite(snapshot, bundle_id="0" * 64),
            rewrite(snapshot, batch=0),
        ]:
            with pytest.raises(RuntimeError):
                consumer.restore_history(bad)
        assert consumer.stats()["restore_failures_total"] == 5
        with pytest.raises(RuntimeError, match="revision"):
            consumer.score_with_history(handles, ["wrong"] * len(handles), request)
        assert consumer.stats()["decode_step_calls_total"] == 0
        with NativeClient(**options, hydration=True, kv_cache_size_bytes=134217728) as other:
            with pytest.raises(RuntimeError, match="another instance"):
                other.history_info(handles[0])
        output = consumer.score_with_history(
            handles, snapshot_metadata(snapshot)["revisions"], request
        )
        for name in baseline:
            np.testing.assert_array_equal(output[name], baseline[name])


def test_exact_budget_leases_deduplication_and_lru(workload):
    options, inputs = workload
    for invalid in (-1, 0, 1.5, True):
        with pytest.raises(RuntimeError, match="positive integer"):
            NativeClient(**options, hydration=True, kv_cache_size_bytes=invalid)
    single = {n: v[:1].copy() for n, v in inputs.items()}
    with NativeClient(**options) as producer:
        metadata = producer.metadata
        names = [s["name"] for s in metadata["stages"][0]["inputs"]]
        snapshot = producer.export_history({n: single[n] for n in names}, ["r1"])
    scratch = metadata["max_batch_size"] * kv_bytes(metadata)
    with pytest.raises(RuntimeError, match="no capacity"):
        NativeClient(
            **options, hydration=True, kv_cache_size_bytes=scratch + kv_bytes(metadata) - 1
        )
    with pytest.raises(RuntimeError, match="smaller"):
        NativeClient(**options, hydration=True, kv_cache_memory_ratio=1e-12)
    with NativeClient(
        **options,
        hydration=True,
        kv_cache_size_bytes=scratch + 2 * kv_bytes(metadata),
        max_entries=2,
        lease_seconds=10,
    ) as consumer:
        first = consumer.restore_history(snapshot)
        second = consumer.restore_history(rewrite(snapshot, revisions=["r2"]))
        assert consumer.restore_history(snapshot) == first
        assert consumer.stats()["resident_entries"] == 2
        with pytest.raises(RuntimeError, match="capacity exhausted"):
            consumer.restore_history(rewrite(snapshot, revisions=["r3"]))
        consumer.release_history(first)
        third = consumer.restore_history(rewrite(snapshot, revisions=["r3"]))
        # Successful use releases r2's early-hydration lease, making it an LRU victim.
        request = {n: v for n, v in single.items() if n not in names}
        consumer.score_with_history(second, ["r2"], request)
        consumer.restore_history(rewrite(snapshot, revisions=["r4"]))
        with pytest.raises(RuntimeError, match="handle"):
            consumer.history_info(second[0])
        assert consumer.history_info(third[0])["ready"]
        stats = consumer.stats()
        assert stats["evictions_total"] == stats["allocation_failures"] == 1
        assert stats["reserved_bytes"] == stats["budget_bytes"] == scratch + 2 * kv_bytes(metadata)


def test_expiry_and_concurrent_stats(workload):
    options, snapshot, baseline, request = prepared(workload)
    with NativeClient(
        **options, hydration=True, kv_cache_size_bytes=134217728, ttl_seconds=0.1, lease_seconds=0
    ) as consumer:
        handles = consumer.restore_history(snapshot)
        time.sleep(0.15)
        with pytest.raises(RuntimeError, match="handle"):
            consumer.history_info(handles[0])
        assert consumer.stats()["resident_entries"] == 0
    with NativeClient(**options, hydration=True, kv_cache_size_bytes=134217728) as consumer:
        handles = consumer.restore_history(snapshot)
        samples, failures, stop = [], [], threading.Event()

        def poll():
            while not stop.wait(0.001):
                try:
                    samples.append(consumer.stats())
                except Exception as error:
                    failures.append(str(error))

        thread = threading.Thread(target=poll)
        thread.start()
        try:
            deadline = time.monotonic() + 5
            count = 0
            while count < 3 or (
                not any(s["active_sequences"] == len(handles) for s in samples)
                and time.monotonic() < deadline
            ):
                count += 1
                consumer.score_with_history(
                    handles, snapshot_metadata(snapshot)["revisions"], request
                )
        finally:
            stop.set()
            thread.join()
        assert not failures
        assert any(s["active_sequences"] == len(handles) for s in samples)
        assert consumer.stats()["active_sequences"] == 0


def test_zero_history_and_padded_candidates_preserve_cold_and_restored_scores(workload):
    from ..checkpoint import load_weights
    from ..reference import Reference

    options, original = workload
    inputs = {name: value.copy() for name, value in original.items()}
    inputs["history_lengths"][:] = 0
    inputs["num_candidates"][:] = 1
    if len(inputs["num_candidates"]) > 1:
        inputs["num_candidates"][0] = 0
    config = Config.load(Path(options["bundle"]).parent)
    expected = Reference(config, load_weights(Path(options["bundle"]).parent)).score(inputs)
    options, snapshot, cold, request = prepared((options, inputs))
    precision = "bf16" if "bf16" in str(options["bundle"]) else "fp32"
    batch = next(iter(inputs.values())).shape[0]
    gate = json.loads(
        (
            Path(__file__).parent / "thresholds" / f"dlrmv3-{precision}-default-b{batch}.json"
        ).read_text()
    )["threshold_overrides"]
    np.testing.assert_allclose(
        cold["scores"], expected["scores"], atol=gate["score_atol"], rtol=gate["score_rtol"]
    )
    assert np.count_nonzero(cold["scores"][:, 1:]) == 0
    with NativeClient(**options, hydration=True, kv_cache_size_bytes=134217728) as consumer:
        handles = consumer.restore_history(snapshot)
        actual = consumer.score_with_history(handles, ["revision-1"] * batch, request)
        np.testing.assert_array_equal(actual["scores"], cold["scores"])
        assert consumer.stats()["history_prepare_calls_total"] == 0
