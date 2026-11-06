# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A compile reference answers only from compiled code: it fails over to no answers, never to eager ones."""

import pytest

torch = pytest.importorskip("torch")

from trtmc_perf_serving.backends.base import BackendError, BackendUnavailable  # noqa: E402
from trtmc_perf_serving.backends.reference import ReferenceBackend, common  # noqa: E402
from trtmc_perf_serving.backends.reference.common import ReferenceSpec  # noqa: E402

FAMILY = """
import torch


class Adapter:
    def __init__(self, spec, host):
        self.spec, self.host = spec, host
        self.layer = host.maybe_compile(torch.nn.Linear(2, 2), spec)

    def invoke(self, request, artifact_base):
        if request.get("fail") == "rejected":
            raise self.host.Error("not this request")
        if request.get("fail"):
            raise RuntimeError("inductor lowering failed\\n  Explanation: the user stack")
        if request.get("compiled", True):
            self.layer(torch.zeros(1, 2))
        return self.host.invocation({"ok": True}, 1.0)
"""


@pytest.fixture
def family(tmp_path, monkeypatch):
    adapter = tmp_path / "fam" / "reference" / "adapter.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text(FAMILY)
    compiled, counts = [], {"compiled_graphs": 0, "graph_breaks": 0}

    def compile(function):  # a compiled function counts its graph when it first runs, as Dynamo does
        compiled.append(function)

        def run(*args, **kwargs):
            counts["compiled_graphs"] = max(counts["compiled_graphs"], 1)
            return function(*args, **kwargs)
        return run

    monkeypatch.setattr(common.torch, "compile", compile)
    monkeypatch.setattr(common, "compile_counts", lambda: dict(counts))
    monkeypatch.setattr("trtmc_perf_serving.backends.reference.strict_recompiles", lambda: None)

    def backend(mode):
        return ReferenceBackend(ReferenceSpec(operation="world_model", model="m", mode=mode, device="cpu",
                                              adapter=str(adapter)))
    return backend, compiled, counts


def test_compile_mode_compiles_the_family_modules_and_reports_its_graphs(family, tmp_path):
    backend, compiled, _ = family
    invocation = backend("compile").invoke({}, tmp_path / "out")
    assert len(compiled) == 1 and invocation.extra["compiled_graphs"] == 1 and invocation.extra["graph_breaks"] == 0
    invocation = backend("eager").invoke({}, tmp_path / "out")
    assert len(compiled) == 1 and "compiled_graphs" not in invocation.extra  # eager compiles nothing


def test_a_failing_compile_reference_stops_serving_but_a_rejected_request_does_not(family, tmp_path):
    backend, _, _ = family
    compile_mode, eager = backend("compile"), backend("eager")
    with pytest.raises(BackendError, match="not this request"):
        compile_mode.invoke({"fail": "rejected"}, tmp_path / "out")
    with pytest.raises(BackendUnavailable, match="torch.compile reference failed: RuntimeError: inductor lowering") \
            as stopped:
        compile_mode.invoke({"fail": True}, tmp_path / "out")
    assert "Explanation" not in str(stopped.value)  # the client gets the first line; the server log the rest
    with pytest.raises(RuntimeError, match="inductor lowering") as raised:  # eager: that request alone fails
        eager.invoke({"fail": True}, tmp_path / "out")
    assert not isinstance(raised.value, BackendUnavailable)


def test_a_compile_reference_whose_compiled_modules_never_ran_stops_serving(family, tmp_path):
    backend, _, _ = family
    with pytest.raises(BackendUnavailable, match="compiled no graph"):
        backend("compile").invoke({"compiled": False}, tmp_path / "out")


def test_recompiling_past_the_limit_fails_instead_of_running_eager(monkeypatch):
    import torch._dynamo

    monkeypatch.setattr(torch._dynamo.config, "fail_on_recompile_limit_hit", False)
    common.strict_recompiles()
    assert torch._dynamo.config.fail_on_recompile_limit_hit is True


def test_compile_counts_read_dynamo_counters(monkeypatch):
    from collections import Counter, defaultdict

    from torch._dynamo import utils

    counters = defaultdict(Counter)
    counters["stats"]["unique_graphs"] = 3
    counters["graph_break"]["a"], counters["graph_break"]["b"] = 2, 1
    monkeypatch.setattr(utils, "counters", counters)
    assert common.compile_counts() == {"compiled_graphs": 3, "graph_breaks": 3}
