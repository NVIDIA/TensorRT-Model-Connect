# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import stat
from types import SimpleNamespace

from tools.reference import transformers_text
from tools import trtmc_reference


def _prepare_work(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "answers.json").write_text(
        json.dumps({"requests": [{"sample_id": "one", "answer": "A"}]}),
        encoding="utf-8",
    )
    (path / "prompts.jsonl").write_text(
        json.dumps({"sample_id": "one", "prompt": "question"}) + "\n",
        encoding="utf-8",
    )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_kind": "mmlu_json",
                "files": {
                    "answers": str(path / "answers.json"),
                    "prompts": str(path / "prompts.jsonl"),
                },
            }
        ),
        encoding="utf-8",
    )


def _args(work_dir: Path, cache_dir: Path, *extra: str):
    return trtmc_reference.build_parser().parse_args(
        [
            "run",
            "--model",
            "org/model",
            "--family",
            "family",
            "--reference-family",
            "causal",
            "--work-dir",
            str(work_dir),
            "--cache-dir",
            str(cache_dir),
            *extra,
        ]
    )


def test_reference_cache_reuses_same_settings_across_work_directories(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_dir = tmp_path / "cache"
    first = tmp_path / "first"
    second = tmp_path / "second"
    _prepare_work(first)
    _prepare_work(second)
    calls: list[Path] = []

    def fake_reference(args) -> None:
        work_dir = Path(args.work_dir)
        calls.append(work_dir)
        artifact = work_dir / "hf_artifacts" / "one.bin"
        artifact.parent.mkdir()
        artifact.write_bytes(b"reference")
        (work_dir / "hf_predictions.json").write_text(
            json.dumps(
                {
                    "responses": [
                        {
                            "sample_id": "one",
                            "output_text": "A",
                            "artifact": str(artifact),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (work_dir / "hf_raw.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        trtmc_reference.task_eval,
        "run_hf_reference",
        fake_reference,
    )

    assert trtmc_reference.run_reference(_args(first, cache_dir)) == "generated"
    assert trtmc_reference.run_reference(_args(second, cache_dir)) == "reused"

    assert calls == [first]
    assert (first / "hf_predictions.json").is_symlink()
    assert (second / "hf_predictions.json").is_symlink()
    assert not (second / "hf_predictions.json").readlink().is_absolute()
    payload = json.loads(
        (second / "hf_predictions.json").read_text(encoding="utf-8")
    )
    assert Path(payload["responses"][0]["artifact"]).read_bytes() == b"reference"
    assert json.loads((second / "hf_cache.json").read_text(encoding="utf-8"))[
        "status"
    ] == "reused"
    entries = [path for path in cache_dir.iterdir() if not path.name.startswith(".")]
    assert len(entries) == 1
    assert stat.S_IMODE(entries[0].stat().st_mode) & 0o055 == 0o055


def test_reference_cache_key_changes_with_inference_setting(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_dir = tmp_path / "cache"
    first = tmp_path / "first"
    second = tmp_path / "second"
    _prepare_work(first)
    _prepare_work(second)
    calls = 0

    def fake_reference(args) -> None:
        nonlocal calls
        calls += 1
        Path(args.work_dir, "hf_predictions.json").write_text(
            json.dumps(
                {"responses": [{"sample_id": "one", "output_text": "A"}]}
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        trtmc_reference.task_eval,
        "run_hf_reference",
        fake_reference,
    )

    trtmc_reference.run_reference(_args(first, cache_dir, "--seed", "1"))
    trtmc_reference.run_reference(_args(second, cache_dir, "--seed", "2"))

    assert calls == 2
    assert len([path for path in cache_dir.iterdir() if not path.name.startswith(".")]) == 2


def test_reference_cache_can_adopt_an_existing_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_dir = tmp_path / "cache"
    work_dir = tmp_path / "existing"
    _prepare_work(work_dir)
    (work_dir / "hf_predictions.json").write_text(
        json.dumps({"responses": [{"sample_id": "one", "output_text": "A"}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        trtmc_reference.task_eval,
        "run_hf_reference",
        lambda _args: (_ for _ in ()).throw(AssertionError("must not infer")),
    )

    status = trtmc_reference.run_reference(
        _args(work_dir, cache_dir, "--adopt-existing")
    )

    assert status == "adopted"
    assert (work_dir / "hf_predictions.json").is_symlink()


def test_causal_reference_uses_native_transformers_entrypoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_dir = tmp_path / "cache"
    work_dir = tmp_path / "native"
    _prepare_work(work_dir)
    arguments = _args(work_dir, cache_dir)
    arguments.reference_family = "causal_base_continuation"
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        assert kwargs["stderr"] == trtmc_reference.subprocess.STDOUT
        predictions = Path(command[command.index("--predictions") + 1])
        raw_output = Path(command[command.index("--raw-output") + 1])
        metadata = Path(command[command.index("--repro-metadata") + 1])
        predictions.write_text(
            json.dumps(
                {"responses": [{"sample_id": "one", "output_text": "A"}]}
            ),
            encoding="utf-8",
        )
        raw_output.write_text("{}\n", encoding="utf-8")
        metadata.write_text(
            json.dumps(
                {
                    "command": [
                        command[0],
                        command[1],
                        "--sample-id",
                        "{sample_id}",
                    ]
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        trtmc_reference.task_eval,
        "run_hf_reference",
        lambda _args: (_ for _ in ()).throw(AssertionError("wrapper was used")),
    )
    monkeypatch.setattr(trtmc_reference.subprocess, "run", fake_run)

    assert trtmc_reference.run_reference(arguments) == "generated"

    command = captured["command"]
    assert command[1].endswith("tools/reference/transformers_text.py")
    assert "task_eval.py" not in " ".join(command)
    assert (work_dir / "hf_native_run.log").is_symlink()
    assert (work_dir / "hf_native_repro.json").is_symlink()


def test_transformers_reference_metadata_is_direct_and_sample_selectable(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "reproduction.json"
    arguments = transformers_text.build_parser().parse_args(
        [
            "--model",
            "org/model",
            "--prompts",
            str(tmp_path / "prompts.jsonl"),
            "--answers",
            str(tmp_path / "answers.json"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--predictions",
            str(tmp_path / "predictions.json"),
            "--raw-output",
            str(tmp_path / "raw.jsonl"),
            "--repro-metadata",
            str(metadata),
            "--local-files-only",
        ]
    )

    transformers_text._write_reproduction_metadata(arguments)

    payload = json.loads(metadata.read_text(encoding="utf-8"))
    command = payload["command"]
    assert command[1].endswith("tools/reference/transformers_text.py")
    assert command[command.index("--sample-id") + 1] == "{sample_id}"
    assert command[command.index("--prompts") + 1] == "{work_dir}/prompts.jsonl"
    assert "task_eval.py" not in " ".join(command)
