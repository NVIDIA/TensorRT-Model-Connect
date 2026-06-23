"""Tests for Hugging Face reference helper logic."""

from __future__ import annotations

import json
import subprocess

import pytest

from tests.e2e_harness.references import hf_transformers
from tests.e2e_harness.references.hf_transformers import (
    _decode_vl_generated_text,
    _json_output_reader,
    _npy_output_reader,
    _read_text_artifact,
    run_reference_subprocess,
)


class _FakeProcessor:
    def __init__(self, mapping: dict[tuple[int, ...], str]) -> None:
        self.mapping = mapping

    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return self.mapping[tuple(int(token) for token in token_ids)]


def test_vl_decode_uses_generated_suffix_for_full_sequences() -> None:
    processor = _FakeProcessor({
        (101, 102): "prompt",
        (201, 202): "blue",
    })

    assert _decode_vl_generated_text(processor, [101, 102, 201, 202], 2) == "blue"


def test_vl_decode_falls_back_when_model_returns_generated_only_ids() -> None:
    processor = _FakeProcessor({
        (): "",
        (201, 202): "blue",
    })

    assert _decode_vl_generated_text(processor, [201, 202], 4) == "blue"


def test_run_reference_subprocess_loads_artifacts(monkeypatch, tmp_path) -> None:
    json_path = tmp_path / "out.json"
    npy_path = tmp_path / "out.npy"
    text_path = tmp_path / "out.txt"

    def _fake_run(cmd, **kwargs):
        import numpy as np

        assert cmd == ["/ref/python", "-c", "print('ok')"]
        assert kwargs["timeout"] == 5
        assert kwargs["env"]["LD_LIBRARY_PATH"] == "/libs"
        json_path.write_text(json.dumps({"answer": 42}), encoding="utf-8")
        np.save(npy_path, np.array([1, 2, 3], dtype=np.int32))
        text_path.write_text("done", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="OK\n", stderr="warn\n")

    monkeypatch.setattr(hf_transformers.subprocess, "run", _fake_run)

    out = run_reference_subprocess(
        command=["/ref/python", "-c", "print('ok')"],
        timeout_s=5,
        label="hf_helper",
        artifact_dir=str(tmp_path),
        case_name="case-a",
        stage_name="full_generation",
        env={"LD_LIBRARY_PATH": "/libs"},
        output_readers=(
            _json_output_reader(str(json_path)),
            _npy_output_reader(str(npy_path), "values", path_key="values_path"),
        ),
        text_reader=lambda: _read_text_artifact(str(text_path)),
        include_stdio_metadata=True,
        metadata={"trust_remote_code": False},
        failure_label="HF helper",
    )

    assert out.stage_name == "full_generation"
    assert out.data["answer"] == 42
    assert out.data["values_path"] == str(npy_path)
    assert out.data["values"].tolist() == [1, 2, 3]
    assert out.text == "done"
    assert out.metadata == {
        "returncode": 0,
        "stdout": "OK\n",
        "stderr": "warn\n",
        "trust_remote_code": False,
    }


def test_run_reference_subprocess_nonzero_saves_full_stderr(
    monkeypatch, tmp_path
) -> None:
    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 7, stdout="", stderr="bad stderr")

    monkeypatch.setattr(hf_transformers.subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError) as excinfo:
        run_reference_subprocess(
            command=["/ref/python", "-c", "raise SystemExit(7)"],
            timeout_s=5,
            label="hf_helper",
            artifact_dir=str(tmp_path),
            case_name="case-a",
            stage_name="full_generation",
            failure_label="HF helper",
        )

    assert "HF helper failed for case-a (rc=7)" in str(excinfo.value)
    assert "bad stderr" in str(excinfo.value)
    log_path = tmp_path / "case-a" / "hf_helper_stderr.log"
    assert log_path.read_text(encoding="utf-8") == "bad stderr"


def test_run_reference_subprocess_timeout_saves_stderr(monkeypatch, tmp_path) -> None:
    def _fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"], stderr="late stderr")

    monkeypatch.setattr(hf_transformers.subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError) as excinfo:
        run_reference_subprocess(
            command=["/ref/python", "-c", "while True: pass"],
            timeout_s=5,
            label="hf_helper",
            artifact_dir=str(tmp_path),
            case_name="case-a",
            stage_name="full_generation",
            failure_label="HF helper",
        )

    assert "HF helper timed out for case-a" in str(excinfo.value)
    assert "late stderr" in str(excinfo.value)
    log_path = tmp_path / "case-a" / "hf_helper_stderr.log"
    assert log_path.read_text(encoding="utf-8") == "late stderr"
