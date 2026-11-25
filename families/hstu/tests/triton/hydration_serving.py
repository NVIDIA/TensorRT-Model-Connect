# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
import time
import urllib.request
import numpy as np
import tritonclient.http as http
import tritonclient.grpc as grpc
from tritonclient.utils import np_to_triton_dtype
from families.hstu.triton.native_client import NativeClient
from families.hstu.config import Config
from families.hstu.reference import example_inputs

from .common import root, runtime_root

admin = http.InferenceServerClient("localhost:8000")
results = []


def call(client, module, model, command, arrays, outputs):
    tensors = []
    for name, value in dict(
        arrays, CACHE_CONTROL=np.array([json.dumps(command).encode()], dtype=object)
    ).items():
        value = np.ascontiguousarray(value)
        tensor = module.InferInput(name, value.shape, np_to_triton_dtype(value.dtype))
        tensor.set_data_from_numpy(value)
        tensors.append(tensor)
    return client.infer(
        model, tensors, outputs=[module.InferRequestedOutput(name) for name in outputs]
    )


def scrape(model):
    text = urllib.request.urlopen("http://localhost:8002/metrics").read().decode()
    values = {}
    for line in text.splitlines():
        if line.startswith("hstu_kv_cache_") and f'model="{model}"' in line:
            key = line.split("{", 1)[0].removeprefix("hstu_kv_cache_")
            values[key] = float(line.rsplit(" ", 1)[1])
    return text, values


def wait_metrics(model, predicate):
    for _ in range(200):
        text, values = scrape(model)
        if predicate(values):
            return text, values
        time.sleep(0.01)
    raise AssertionError((model, values))


for profile in ("default", "short"):
    for precision in ("fp32", "bf16"):
        variant = f"dlrmv3-{profile}-{precision}"
        model = "hydrated_" + variant.replace("-", "_")
        if admin.is_model_ready(model):
            admin.unload_model(model)
            wait_metrics(model, lambda v: not v)
        admin.load_model(model)
        cases = [1, 4] if profile == "default" else [4]
        c = Config.load(root / "output" / variant)
        per_history = 5 * 4 * (c.serving.history_length + 1) * 256 * 4
        with NativeClient(root / "output" / variant / "model.bundle", runtime_root) as native:
            names = [s["name"] for s in native.metadata["stages"][0]["inputs"]]
            output_names = [s["name"] for s in native.metadata["outputs"]]
            for case in cases:
                features = example_inputs(Config.load(root / "output" / variant), case)
                batch = case
                expected = native.score(features)
                history = {k: v for k, v in features.items() if k in names}
                snapshot = native.export_history(history, ["history-v1"] * batch)
                request = {k: v for k, v in features.items() if k not in names}
                for module, protocol, port in [(http, "http", 8000), (grpc, "grpc", 8001)]:
                    client = module.InferenceServerClient(f"localhost:{port}")
                    status = call(
                        client,
                        module,
                        model,
                        {"operation": "restore"},
                        {"CACHE_SNAPSHOT": np.frombuffer(snapshot, dtype=np.uint8)},
                        ["CACHE_STATUS"],
                    )
                    handles = json.loads(status.as_numpy("CACHE_STATUS")[0])["handles"]
                    text, before = wait_metrics(model, lambda v: v.get("resident_entries") == batch)
                    (root / "output" / variant / f"{case}-{protocol}-resident.prom").write_text(
                        text
                    )
                    assert (
                        before["resident_bytes"] == per_history * batch
                        and before["used_bytes"] == 0
                    )
                    assert before["history_prepare_calls_total"] == 0
                    assert before["restoring_bytes"] == 0
                    duplicate = call(
                        client,
                        module,
                        model,
                        {"operation": "restore"},
                        {"CACHE_SNAPSHOT": np.frombuffer(snapshot, dtype=np.uint8)},
                        ["CACHE_STATUS"],
                    )
                    assert json.loads(duplicate.as_numpy("CACHE_STATUS")[0])["handles"] == handles
                    stop = threading.Event()
                    samples = []

                    def poll():
                        while not stop.wait(0.002):
                            samples.append(scrape(model)[1])

                    thread = threading.Thread(target=poll)
                    thread.start()
                    try:
                        # Busy gauges are asynchronous and short requests can fall
                        # between publications. Keep the same request under load
                        # until activity is observed, with a strict five-second
                        # observation deadline and at least twenty comparisons.
                        repetitions = 0
                        deadline = time.monotonic() + 5
                        while repetitions < 20 or (
                            not any(x.get("active_sequences") == batch for x in samples)
                            and time.monotonic() < deadline
                        ):
                            repetitions += 1
                            response = call(
                                client,
                                module,
                                model,
                                {
                                    "operation": "score",
                                    "handles": handles,
                                    "revisions": ["history-v1"] * batch,
                                },
                                request,
                                output_names + ["CACHE_STATUS"],
                            )
                            for key, value in expected.items():
                                np.testing.assert_array_equal(response.as_numpy(key), value)
                    finally:
                        stop.set()
                        thread.join()
                    text, after = wait_metrics(
                        model,
                        lambda v: (
                            v.get("decode_step_calls_total")
                            == before["decode_step_calls_total"] + repetitions
                        ),
                    )
                    assert (
                        after["history_prepare_calls_total"] == 0 and after["active_sequences"] == 0
                    )
                    assert any(x.get("active_sequences") == batch for x in samples), (
                        variant,
                        case,
                        protocol,
                        repetitions,
                        samples[-10:],
                    )
                    bad = bytearray(snapshot)
                    bad[-1] ^= 1
                    try:
                        call(
                            client,
                            module,
                            model,
                            {"operation": "restore"},
                            {"CACHE_SNAPSHOT": np.frombuffer(bytes(bad), dtype=np.uint8)},
                            ["CACHE_STATUS"],
                        )
                    except Exception:
                        pass
                    else:
                        raise AssertionError("corrupt remote snapshot accepted")
                    call(
                        client,
                        module,
                        model,
                        {"operation": "release", "handles": handles},
                        {},
                        ["CACHE_STATUS"],
                    )
                    text, idle = wait_metrics(model, lambda v: v.get("resident_entries") == 0)
                    (root / "output" / variant / f"{case}-{protocol}-idle.prom").write_text(text)
                    assert idle["used_bytes"] == idle["active_sequences"] == 0
                    results.append(
                        dict(
                            variant=variant,
                            case=case,
                            protocol=protocol,
                            bit_identical=True,
                            resident_bytes=before["resident_bytes"],
                            active_sample=True,
                            history_launches=0,
                            decode_launches=repetitions,
                        )
                    )
                    client.close()
            # Export through Triton itself as well as native snapshot import.
            response = call(
                admin,
                http,
                model,
                {"operation": "export", "revisions": ["history-v1"] * batch},
                history,
                ["CACHE_STATUS", "CACHE_SNAPSHOT"],
            )
            remote = response.as_numpy("CACHE_SNAPSHOT").tobytes()
            assert remote == snapshot
        admin.unload_model(model)
        wait_metrics(model, lambda v: not v)
        admin.load_model(model)
        response = call(
            admin,
            http,
            model,
            {"operation": "restore"},
            {"CACHE_SNAPSHOT": np.frombuffer(snapshot, dtype=np.uint8)},
            ["CACHE_STATUS"],
        )
        new_handles = json.loads(response.as_numpy("CACHE_STATUS")[0])["handles"]
        assert new_handles != handles
        call(
            admin,
            http,
            model,
            {"operation": "score", "handles": new_handles, "revisions": ["history-v1"] * batch},
            request,
            output_names,
        )
        admin.unload_model(model)
        wait_metrics(model, lambda v: not v)
        (root / "triton-qualification.json").write_text(json.dumps(results, indent=2))
        print(
            variant,
            f"passed HTTP/gRPC batches {cases}, live metrics, export, corruption and reload",
            flush=True,
        )
# Exercise a positive allocation-failure metric through a real Triton endpoint.
# A native fixed-scratch score cannot create post-load cache pressure; snapshot
# restoration can fill the same runtime's leased-entry budget deterministically.
model = "hydrated_dlrmv3_default_fp32"
admin.load_model(model)
handles = []
try:
    checkpoint = root / "output/dlrmv3-default-fp32"
    features = example_inputs(Config.load(checkpoint), 4)
    with NativeClient(checkpoint / "model.bundle", runtime_root) as producer:
        history = {s["name"]: features[s["name"]] for s in producer.metadata["stages"][0]["inputs"]}
        for iteration in range(3):
            snapshot = producer.export_history(
                history, [f"pressure-{iteration}-{row}" for row in range(4)]
            )
            if iteration < 2:
                response = call(
                    admin,
                    http,
                    model,
                    {"operation": "restore"},
                    {"CACHE_SNAPSHOT": np.frombuffer(snapshot, dtype=np.uint8)},
                    ["CACHE_STATUS"],
                )
                handles += json.loads(response.as_numpy("CACHE_STATUS")[0])["handles"]
            else:
                try:
                    call(
                        admin,
                        http,
                        model,
                        {"operation": "restore"},
                        {"CACHE_SNAPSHOT": np.frombuffer(snapshot, dtype=np.uint8)},
                        ["CACHE_STATUS"],
                    )
                except Exception:
                    pass
                else:
                    raise AssertionError("leased-entry cache pressure must reject restoration")
    text, values = wait_metrics(model, lambda v: v.get("allocation_failures", 0) == 1)
    assert values["resident_entries"] == 8
    assert (
        values["reserved_bytes"]
        == 12 * 5 * 4 * (Config.load(checkpoint).serving.history_length + 1) * 256 * 4
    )
    assert values["restoring_bytes"] == values["used_bytes"] == 0
    (root / "positive-allocation-failure.prom").write_text(text)
    (root / "positive-allocation-failure.json").write_text(json.dumps(values, indent=2))
    call(admin, http, model, {"operation": "release", "handles": handles}, {}, ["CACHE_STATUS"])
    print("positive allocation-failure metric under leased-entry pressure PASS", flush=True)
finally:
    admin.unload_model(model)
admin.close()
