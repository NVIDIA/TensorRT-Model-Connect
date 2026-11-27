# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
import numpy as np
import tritonclient.http as http
import tritonclient.grpc as grpc
from families.hstu.triton.native_client import NativeClient
from families.hstu.config import Config
from families.hstu.reference import example_inputs
from .common import FIELDS, infer, make_model, root, runtime_root, scrape, wait_metrics

admin = http.InferenceServerClient("localhost:8000")
records = []
failures = []


def save():
    (root / "native-cache-qualification.json").write_text(
        json.dumps(dict(records=records, failures=failures), indent=2) + "\n"
    )


def check(tag, work):
    try:
        values = work()
        records.append(dict(test=tag, passed=True, observations=values))
        print(tag, "PASS", flush=True)
    except Exception as error:
        records.append(dict(test=tag, passed=False, error=str(error)))
        failures.append(tag)
        print(tag, "FAIL", str(error), flush=True)
    save()


def rejection(name, variant, parameters):
    make_model(name, variant, parameters)
    try:
        admin.load_model(name)
    except Exception as error:
        assert admin.is_server_live(), "Server became unavailable during a negative config test"
        return dict(parameters=parameters, rejected=True, error=str(error))
    else:
        text, samples, fields = scrape(name)
        (root / (name + ".prom")).write_text(text)
        raise AssertionError(
            "Invalid/insufficient config loaded successfully: "
            + json.dumps(dict(parameters=parameters, metrics=fields))
        )
    finally:
        admin.unload_model(name)


for profile in ("default", "short"):
    for precision in ("fp32", "bf16"):
        variant = f"dlrmv3-{profile}-{precision}"
        suffix = variant.replace("-", "_")
        c = Config.load(root / "output" / variant)
        per_history = 5 * 4 * (c.serving.history_length + 1) * 256 * 4
        requests = {
            batch: example_inputs(Config.load(root / "output" / variant), batch)
            for batch in ((1, 4) if profile == "default" else (4,))
        }
        with NativeClient(root / "output" / variant / "model.bundle", runtime_root) as reference:
            expected = {
                batch: reference.score(requests[batch])
                for batch in ((1, 4) if profile == "default" else (4,))
            }

        def explicit_budget():
            name = "cap_" + suffix
            make_model(name, variant, {"kv_cache_size_bytes": 134217728})
            results = []
            try:
                admin.load_model(name)
                for batch in (1, 4) if profile == "default" else (4,):
                    for module, protocol, port in ((http, "http", 8000), (grpc, "grpc", 8001)):
                        with module.InferenceServerClient(f"localhost:{port}") as client:
                            response = infer(client, module, name, requests[batch], expected[batch])
                            for key, value in expected[batch].items():
                                np.testing.assert_array_equal(response.as_numpy(key), value)
                        # Publication is asynchronous; wait for the completed request's
                        # peak and idle gauges before checking the same expected values.
                        text, samples, fields = wait_metrics(
                            name,
                            lambda values: (
                                all(key in values for key in FIELDS)
                                and values["used_bytes"] == values["active_sequences"] == 0
                                and values["peak_bytes"] >= per_history * batch
                            ),
                        )
                        (root / "output" / variant / f"cap-b{batch}-{protocol}.prom").write_text(
                            text
                        )
                        assert fields["supported"] == 1, fields
                        assert fields["budget_bytes"] == 134217728, fields
                        assert 0 < fields["reserved_bytes"] <= fields["budget_bytes"], fields
                        assert fields["used_bytes"] == fields["active_sequences"] == 0, fields
                        assert fields["peak_bytes"] >= per_history * batch, fields
                        assert fields["allocation_failures"] == 0, fields
                        results.append(
                            dict(
                                batch=batch,
                                protocol=protocol,
                                bit_identical=True,
                                metrics=fields,
                                metric_names=sorted(set(row["name"] for row in samples)),
                            )
                        )
            finally:
                admin.unload_model(name)
            wait_metrics(name, lambda values: not values)
            return results

        check(variant + ":explicit_budget_and_http_grpc", explicit_budget)

        def ratio_budget():
            results = []
            for label, ratio in (("one", 0.01), ("five", 0.05)):
                name = "ratio_" + label + "_" + suffix
                make_model(name, variant, {"kv_cache_memory_ratio": ratio})
                try:
                    admin.load_model(name)
                    response = infer(admin, http, name, requests[4], expected[4])
                    for key, value in expected[4].items():
                        np.testing.assert_array_equal(response.as_numpy(key), value)
                    text, samples, fields = wait_metrics(
                        name, lambda values: values.get("budget_bytes", 0) > 0
                    )
                    (root / "output" / variant / f"ratio-{label}.prom").write_text(text)
                    results.append(dict(ratio=ratio, metrics=fields))
                finally:
                    admin.unload_model(name)
            factor = results[1]["metrics"]["budget_bytes"] / results[0]["metrics"]["budget_bytes"]
            assert 4.5 < factor < 5.5, results
            return dict(samples=results, budget_factor=factor)

        check(variant + ":memory_ratio_takes_effect", ratio_budget)
        check(
            variant + ":insufficient_bytes",
            lambda: rejection("tiny_bytes_" + suffix, variant, {"kv_cache_size_bytes": 1}),
        )
        check(
            variant + ":insufficient_ratio",
            lambda: rejection("tiny_ratio_" + suffix, variant, {"kv_cache_memory_ratio": "1e-12"}),
        )
        check(
            variant + ":mutually_exclusive",
            lambda: rejection(
                "both_" + suffix,
                variant,
                {"kv_cache_size_bytes": 134217728, "kv_cache_memory_ratio": 0.01},
            ),
        )

        if variant == "dlrmv3-default-fp32":

            def live_metrics():
                name = "live_cache_metrics"
                make_model(name, variant, {"kv_cache_size_bytes": 134217728})
                samples = []
                stop = threading.Event()
                errors = []

                def poll():
                    while not stop.wait(0.005):
                        try:
                            samples.append(scrape(name)[2])
                        except Exception as error:
                            errors.append(str(error))

                try:
                    admin.load_model(name)
                    thread = threading.Thread(target=poll)
                    thread.start()
                    try:
                        with grpc.InferenceServerClient("localhost:8001") as client:
                            for _ in range(24):
                                infer(client, grpc, name, requests[4], expected[4])
                    finally:
                        stop.set()
                        thread.join()
                    text, rows, idle = wait_metrics(
                        name,
                        lambda values: (
                            values.get("active_sequences") == values.get("used_bytes") == 0
                        ),
                    )
                    (root / "live-cache-metrics.prom").write_text(text)
                    observations = dict(
                        polls=len(samples),
                        poll_errors=errors,
                        max_active_sequences=max(
                            (sample.get("active_sequences", 0) for sample in samples), default=0
                        ),
                        max_used_bytes=max(
                            (sample.get("used_bytes", 0) for sample in samples), default=0
                        ),
                        idle=idle,
                    )
                    (root / "live-cache-metric-samples.json").write_text(
                        json.dumps(dict(observations=observations, samples=samples), indent=2)
                        + "\n"
                    )
                    assert not errors, observations
                    assert (
                        observations["max_active_sequences"] > 0
                        and observations["max_used_bytes"] > 0
                    ), observations
                    return observations
                finally:
                    admin.unload_model(name)

            check("live_metric_activity", live_metrics)

for key, values in [
    ("kv_cache_memory_ratio", ["-1", "1.1", "nan", "inf", "abc", "0.05junk"]),
    ("kv_cache_size_bytes", ["-1", "134217728.5", "abc", "134217728junk", "18446744073709551616"]),
]:
    for index, value in enumerate(values):
        check(
            "invalid:" + key + ":" + value,
            lambda key=key, value=value, index=index: rejection(
                f"invalid_{key}_{index}", "dlrmv3-default-fp32", {key: value}
            ),
        )


def multiple_models():
    observations = []
    names = []
    features = example_inputs(Config.load(root / "output/dlrmv3-default-fp32"), 4)
    for index, budget in enumerate((134217728, 268435456)):
        name = f"isolated_cache_{index}"
        names.append(name)
        make_model(name, "dlrmv3-default-fp32", {"kv_cache_size_bytes": budget})
    try:
        for name in names:
            admin.load_model(name)
        for name, budget in zip(names, (134217728, 268435456)):
            infer(admin, http, name, features, ["scores"])
            text, samples, fields = wait_metrics(
                name, lambda values: values.get("budget_bytes", 0) > 0
            )
            (root / (name + ".prom")).write_text(text)
            assert fields["budget_bytes"] == budget, (name, fields)
            observations.append(dict(model=name, metrics=fields))
    finally:
        for name in names:
            admin.unload_model(name)
    return observations


check("multiple_model_metric_isolation", multiple_models)


def multiple_instances():
    name = "two_cache_instances"
    make_model(name, "dlrmv3-default-fp32", {"kv_cache_size_bytes": 134217728}, instances=2)
    features = example_inputs(Config.load(root / "output/dlrmv3-default-fp32"), 4)
    try:
        admin.load_model(name)
        for _ in range(8):
            infer(admin, http, name, features, ["scores"])
        text, samples, fields = wait_metrics(name, lambda values: values.get("budget_bytes", 0) > 0)
        (root / (name + ".prom")).write_text(text)
        observations = dict(
            instance_count=2,
            metrics=fields,
            metric_samples=samples,
            inference_statistics=admin.get_inference_statistics(model_name=name),
        )
        (root / "multi-instance-observations.json").write_text(
            json.dumps(observations, indent=2) + "\n"
        )
        assert fields["budget_bytes"] == 2 * 134217728, observations
        return observations
    finally:
        admin.unload_model(name)


check("multiple_instance_metric_accounting", multiple_instances)
admin.close()
save()
print(json.dumps(dict(checks=len(records), failed=failures), indent=2), flush=True)
raise SystemExit(bool(failures))
