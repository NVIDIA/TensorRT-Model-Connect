from __future__ import annotations

import subprocess

import numpy as np
import pytest

from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec
from tests.e2e_harness.manifest_loader import get_case_by_name
from tests.e2e_harness.orchestrator import _append_manifest_config_args
from tests.e2e_harness.runners.text_generation import TextGenerationCausalRunner


def test_multi_device_qwen_manifests_declare_tp_build_and_mpirun_runtime() -> None:
    for case_prefix in ("qwen3-0.6b-fp16", "qwen3-4b-instruct-2507"):
        for tp_size in (2, 4, 8):
            case = get_case_by_name(f"{case_prefix}-tp{tp_size}")
            assert case is not None
            assert case.family == "qwen"
            assert case.runtime_strategy == "decoder_kv_cache"
            assert case.metadata["ci_tier"] == "multi_device"
            assert "TensorRT 11.0+" in case.metadata["notes"]
            assert case.metadata["build_args"]["parallel"] == {
                "mode": "tensor_parallel",
                "tp_size": tp_size,
            }
            assert case.metadata["distributed_runtime"]["enabled"] is True
            assert case.metadata["distributed_runtime"]["world_size"] == tp_size
            assert case.metadata["distributed_runtime"]["capture_gpu_memory"] is True
            assert case.metadata["distributed_runtime"]["debug_logits"] is True
            assert [req.kind for req in case.preflight] == [
                "binary_exists",
                "command_available",
                "gpu_count_min",
            ]
            assert case.preflight[-1].args["count"] == tp_size


def test_e2e_build_args_append_parallel_config_sets() -> None:
    cmd = ["python", "-m", "tensorrt_model_connect.__main__", "build"]
    _append_manifest_config_args(cmd, {
        "parallel": {"mode": "tensor_parallel", "tp_size": 2},
        "config_overrides": {"runtime.foo": "bar"},
    })
    assert cmd[-6:] == [
        "--set",
        "runtime.foo=bar",
        "--set",
        "parallel.mode=tensor_parallel",
        "--set",
        "parallel.tp_size=2",
    ]


def test_text_runner_wraps_distributed_runtime_with_mpirun(monkeypatch, tmp_path) -> None:
    case = E2ECase(
        name="qwen-tp",
        hf_id="Qwen/Qwen3-0.6B",
        family="qwen",
        runtime_strategy="decoder_kv_cache",
        bundle="qwen-tp.trtfb",
        inputs={"prompt": "city", "max_new_tokens": 1},
        metadata={
            "distributed_runtime": {
                "enabled": True,
                "world_size": 2,
                "export_env": ["LD_LIBRARY_PATH"],
            },
        },
    )
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        binary_path="/build/trtmc",
        engine_dir="/engines",
        ld_library_path="/trt:/nccl",
    )
    seen: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="[1,0]<stdout>:Paris\n[1,1]<stdout>:Paris\n",
            stderr=(
                "[1,0]<stderr>:[trtmc.load_timing] label=\"engine_plan_tp_rank0\" "
                "load_deserialize_ms=1.5\n"
                "[1,1]<stderr>:[trtmc.load_timing] label=\"engine_plan_tp_rank1\" "
                "load_deserialize_ms=2.5\n"
                "[1,0]<stderr>:[trtmc] Pipeline loaded "
                "(strategy=decoder_kv_cache, backend=trt_new_runtime)\n"
                "[1,0]<stderr>:[trtmc.timing] prefill_ms=3.0 decode_ms=4.0 "
                "total_ms=7.0\n"
            ),
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    text, _, meta = TextGenerationCausalRunner()._run_cpp_binary(
        ctx, "/engines/qwen-tp.trtfb", "city", 1, case=case, inputs=case.inputs)

    assert text == "Paris"
    assert seen["cmd"][:8] == [
        "mpirun",
        "--tag-output",
        "-np",
        "2",
        "-x",
        "LD_LIBRARY_PATH",
        "-x",
        "TRTMC_NCCL_RENDEZVOUS",
    ]
    assert seen["cmd"][8:] == [
        "/build/trtmc",
        "run",
        "/engines/qwen-tp.trtfb",
        "--prompt",
        "city",
        "--max-new-tokens",
        "1",
    ]
    assert seen["env"]["LD_LIBRARY_PATH"] == "/trt:/nccl"
    assert seen["env"]["TRTMC_NCCL_RENDEZVOUS"] == str(
        tmp_path / "qwen-tp" / "qwen-tp.nccl_rendezvous.bin")
    assert meta["rank_zero_stdout"] == "Paris"
    assert meta["trt_load_deserialize_s"] == 0.004
    assert meta["trt_engine_s"] == 0.007


def test_text_runner_collects_distributed_debug_logits_with_mpirun(
    monkeypatch, tmp_path
) -> None:
    case = E2ECase(
        name="qwen-tp",
        hf_id="Qwen/Qwen3-0.6B",
        family="qwen",
        runtime_strategy="decoder_kv_cache",
        bundle="qwen-tp.trtfb",
        inputs={"prompt": "city", "max_new_tokens": 1},
        metadata={
            "distributed_runtime": {
                "enabled": True,
                "world_size": 2,
                "export_env": ["LD_LIBRARY_PATH"],
            },
        },
    )
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        binary_path="/build/trtmc",
        engine_dir="/engines",
        ld_library_path="/trt:/nccl",
        runtime_python="/python",
    )
    expected_logits = tmp_path / "qwen-tp" / "trt_full_logits.npy"
    seen: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["env"] = kwargs.get("env")
        expected_logits.parent.mkdir(parents=True, exist_ok=True)
        np.save(expected_logits, np.zeros((1, 2), dtype=np.float32))
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=(
                "[1,0]<stdout>:OK rank=0 steps=1 vocab=2\n"
                "[1,0]<stdout>:TRTMC_DEBUG_META "
                "{\"generated_text\":\"Paris\",\"full_text\":\"city Paris\","
                "\"generated_token_count\":1,\"distributed_rank\":0}\n"
                "[1,1]<stdout>:OK rank=1 steps=1 vocab=2\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    logits_path, _, meta = TextGenerationCausalRunner()._run_debug_runner_logits(
        ctx, "/engines/qwen-tp.trtfb", "city", 1, case=case, phase="full")

    assert logits_path == str(expected_logits)
    assert seen["cmd"][:8] == [
        "mpirun",
        "--tag-output",
        "-np",
        "2",
        "-x",
        "LD_LIBRARY_PATH",
        "-x",
        "TRTMC_NCCL_RENDEZVOUS",
    ]
    assert seen["cmd"][8:10] == ["/python", "-c"]
    assert "TensorParallelNcclGroup" in seen["cmd"][10]
    assert seen["env"]["LD_LIBRARY_PATH"] == "/trt:/nccl"
    assert seen["env"]["TRTMC_NCCL_RENDEZVOUS"] == str(
        tmp_path / "qwen-tp" / "qwen-tp.debug_full.nccl_rendezvous.bin")
    assert meta["rank_zero_stdout"].startswith("OK rank=0")
    assert meta["generated_text"] == "Paris"
    assert meta["distributed_rank"] == 0


def test_distributed_full_generation_requires_debug_logits(monkeypatch, tmp_path) -> None:
    case = E2ECase(
        name="qwen-tp",
        hf_id="Qwen/Qwen3-0.6B",
        family="qwen",
        runtime_strategy="decoder_kv_cache",
        bundle="qwen-tp.trtfb",
        inputs={"prompt": "city", "max_new_tokens": 1},
        metadata={
            "distributed_runtime": {
                "enabled": True,
                "world_size": 2,
                "debug_logits": True,
            },
        },
    )
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        binary_path="/build/trtmc",
        engine_dir="/engines",
    )
    runner = TextGenerationCausalRunner()

    def fake_cpp_binary(*args, **kwargs):
        return "Paris", 0.01, {"returncode": 0}

    def fake_debug_logits(*args, **kwargs):
        return None, 0.01, {"returncode": 0, "error": "logits file not created"}

    monkeypatch.setattr(runner, "_run_cpp_binary", fake_cpp_binary)
    monkeypatch.setattr(runner, "_run_debug_runner_logits", fake_debug_logits)

    with pytest.raises(
        RuntimeError,
        match="Distributed debug logits requested.*logits file not created",
    ):
        runner.run_stage(case, StageSpec(name="full_generation"), ctx)
