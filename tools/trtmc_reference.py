#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run and cache model reference inference independently from task evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
for import_root in (REPO_ROOT, PYTHON_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tools import task_eval  # noqa: E402


CACHE_SCHEMA = "trtmc.reference-cache/v1"
CACHE_IMPLEMENTATION = 1
REFERENCE_CACHE_IDENTITY_IMPLEMENTATION = 2
_CACHE_METADATA = "reference.json"
_WORK_METADATA = "hf_cache.json"
_NATIVE_RUN_LOG = "hf_native_run.log"
_NATIVE_REPRO_METADATA = "hf_native_repro.json"
_TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".md", ".txt"}
_NATIVE_TEXT_DATASET_KINDS = {"mmlu_five_shot_json", "text_generation_json"}
_NATIVE_ENCODER_DATASET_KINDS = {"sts_pair_jsonl"}
_NATIVE_PLUGIN_DATASET_KINDS = {
    "diffusion_prompt_json",
    "image_classification_json",
    "prompted_segmentation_json",
    "reranking_json",
    "semantic_segmentation_json",
    "time_series_csv",
}
_NATIVE_VLM_DATASET_KINDS = {"vlm_chat_json", "vlm_unified_json"}
_NATIVE_ELF_DATASET_KINDS = {
    "conditional_text_jsonl",
    "unconditional_text_json",
}
_NATIVE_SPEECH_DATASET_KINDS = {"asr_chat_json", "seedtts_json"}
_TRANSFORMERS_TEXT_RUNNER = REPO_ROOT / "tools" / "reference" / "transformers_text.py"
_TRANSFORMERS_ENCODER_RUNNER = (
    REPO_ROOT / "tools" / "reference" / "transformers_encoder.py"
)
_PLUGIN_REFERENCE_RUNNER = REPO_ROOT / "tools" / "reference" / "plugin_reference.py"
_TRANSFORMERS_VLM_RUNNER = REPO_ROOT / "tools" / "reference" / "transformers_vlm.py"
_ELF_PREPARED_RUNNER = REPO_ROOT / "tools" / "reference" / "elf_prepared.py"
_SPEECH_REFERENCE_RUNNER = REPO_ROOT / "tools" / "reference" / "speech.py"
_IGNORED_INPUT_NAMES = {
    "build.log",
    "eval_result.json",
    "hf_cache.json",
    "hf_run.log",
    "summary.json",
    "summary.md",
    "trtfb_predictions.json",
    "trtfb_raw.jsonl",
    "trtfb_run.log",
    "visual_review.html",
}
_NATIVE_RUNNER_VARIANT_TASK_KEYS = {
    "family",
    "model_max_new_tokens",
    "reference_backend",
    "reference_family",
    "user_contract",
}


class ReferenceError(RuntimeError):
    """Reference inference or cache materialization failed."""


def _is_reference_output(name: str) -> bool:
    return (
        (name.startswith("hf_") and name not in {"hf_run.log", _WORK_METADATA})
        or name == "shared_initial_latents"
    )


def _is_non_input(name: str) -> bool:
    return (
        _is_reference_output(name)
        or name.startswith("trtfb_")
        or name in _IGNORED_INPUT_NAMES
    )


def _input_files(work_dir: Path) -> Iterable[Path]:
    for path in sorted(work_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(work_dir)
        if _is_non_input(relative.parts[0]):
            continue
        if relative.name in _IGNORED_INPUT_NAMES:
            continue
        yield path


def _normalize_identity_manifest(content: str) -> str:
    manifest = json.loads(content)
    task_config = manifest.get("task_eval", {})
    if not isinstance(task_config, dict):
        return content
    if "model_manifest" in task_config:
        task_config["model_manifest"] = "<REFERENCE_CACHE_IDENTITY>"
    dataset_kind = str(manifest.get("dataset_kind", "") or "")
    if native_reference_runner_for_dataset_kind(dataset_kind) is not None:
        for name in _NATIVE_RUNNER_VARIANT_TASK_KEYS:
            task_config.pop(name, None)
    return json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    )


def _hash_file(
    hasher: Any,
    path: Path,
    work_dir: Path,
    *,
    reference_cache_identity: str = "",
) -> None:
    relative = path.relative_to(work_dir)
    hasher.update(str(relative).encode())
    if path.suffix.lower() in _TEXT_SUFFIXES and path.stat().st_size <= 32 * 1024 * 1024:
        content = path.read_text(encoding="utf-8", errors="replace")
        if relative == Path("manifest.json"):
            if reference_cache_identity:
                content = _normalize_identity_manifest(content)
        hasher.update(content.replace(str(work_dir), "<WORK_DIR>").encode())
        return
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    native_runner = _native_reference_runner(args)
    settings = {
        "implementation": CACHE_IMPLEMENTATION,
        "native_runner": _native_runner_identity(native_runner),
        "python": str(Path(sys.executable).resolve()),
        "model": args.model,
        "model_revision": args.model_revision,
        "family": args.family,
        "reference_family": args.reference_family,
        "dtype": args.dtype,
        "device": args.device,
        "device_map": args.device_map,
        "attn_impl": args.attn_impl,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
        "do_sample": args.do_sample,
        "apply_chat_template": args.apply_chat_template,
        "elf_reference_repo": args.elf_reference_repo,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "min_p": args.min_p,
        "seed": args.seed,
    }
    reference_cache_identity = str(
        getattr(args, "reference_cache_identity", "") or ""
    )
    if reference_cache_identity:
        settings["reference_cache_identity"] = reference_cache_identity
        settings["reference_cache_identity_implementation"] = (
            REFERENCE_CACHE_IDENTITY_IMPLEMENTATION
        )
    return settings


def native_reference_runner_for_dataset_kind(dataset_kind: str) -> Path | None:
    if dataset_kind in _NATIVE_TEXT_DATASET_KINDS:
        return _TRANSFORMERS_TEXT_RUNNER
    if dataset_kind in _NATIVE_ENCODER_DATASET_KINDS:
        return _TRANSFORMERS_ENCODER_RUNNER
    if dataset_kind in _NATIVE_PLUGIN_DATASET_KINDS:
        return _PLUGIN_REFERENCE_RUNNER
    if dataset_kind in _NATIVE_VLM_DATASET_KINDS:
        return _TRANSFORMERS_VLM_RUNNER
    if dataset_kind in _NATIVE_ELF_DATASET_KINDS:
        return _ELF_PREPARED_RUNNER
    if dataset_kind in _NATIVE_SPEECH_DATASET_KINDS:
        return _SPEECH_REFERENCE_RUNNER
    return None


def _native_reference_runner(args: argparse.Namespace) -> Path | None:
    manifest_path = Path(args.work_dir).resolve() / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_kind = str(manifest.get("dataset_kind", "") or "")
    return native_reference_runner_for_dataset_kind(dataset_kind)


def _native_runner_identity(path: Path | None) -> str:
    if path is None:
        return ""
    return f"{path.name}:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def reference_key(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    work_dir = Path(args.work_dir).resolve()
    settings = _settings(args)
    hasher = hashlib.sha256()
    hasher.update(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode())
    reference_cache_identity = str(
        getattr(args, "reference_cache_identity", "") or ""
    )
    for path in _input_files(work_dir):
        _hash_file(
            hasher,
            path,
            work_dir,
            reference_cache_identity=reference_cache_identity,
        )
    return hasher.hexdigest(), settings


def _cache_entry(cache_dir: Path, key: str) -> Path:
    return cache_dir / key


def _load_cache_metadata(entry: Path, key: str) -> dict[str, Any] | None:
    path = entry / _CACHE_METADATA
    if not path.is_file():
        return None
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        metadata.get("schema_version") != CACHE_SCHEMA
        or metadata.get("key") != key
        or not (entry / "files" / "hf_predictions.json").is_file()
    ):
        return None
    return metadata


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _clear_reference_outputs(work_dir: Path) -> None:
    for path in work_dir.iterdir():
        if _is_reference_output(path.name):
            _remove_path(path)


def _materialize(entry: Path, work_dir: Path) -> None:
    files_dir = entry / "files"
    for source in sorted(files_dir.iterdir()):
        target = work_dir / source.name
        if target.exists() or target.is_symlink():
            _remove_path(target)
        relative_source = os.path.relpath(source, target.parent)
        target.symlink_to(relative_source, target_is_directory=source.is_dir())


def _rewrite_cached_paths(root: Path, old: Path, new: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        if path.name == _NATIVE_RUN_LOG:
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        updated = content.replace(str(old), str(new))
        if updated != content:
            path.write_text(updated, encoding="utf-8")


def _reference_outputs(work_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(work_dir.iterdir())
        if _is_reference_output(path.name)
    ]


def _restore_moved_outputs(moved: Iterable[tuple[Path, Path]]) -> None:
    for source, destination in reversed(list(moved)):
        if source.exists() or source.is_symlink():
            shutil.move(str(source), str(destination))


def _make_cache_readable(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            continue
        mode = path.stat().st_mode
        path.chmod(mode | (0o055 if path.is_dir() else 0o044))


def _move_outputs_to_cache(
    *,
    work_dir: Path,
    cache_dir: Path,
    entry: Path,
    metadata: Mapping[str, Any],
) -> None:
    outputs = _reference_outputs(work_dir)
    if not any(path.name == "hf_predictions.json" for path in outputs):
        raise ReferenceError("reference inference did not produce hf_predictions.json")

    stage = Path(tempfile.mkdtemp(prefix=f".{entry.name}.", dir=cache_dir))
    files_dir = stage / "files"
    files_dir.mkdir()
    moved: list[tuple[Path, Path]] = []
    try:
        for source in outputs:
            destination = files_dir / source.name
            shutil.move(str(source), str(destination))
            moved.append((destination, source))
        _rewrite_cached_paths(files_dir, work_dir, entry / "files")
        (stage / _CACHE_METADATA).write_text(
            json.dumps(dict(metadata), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _make_cache_readable(stage)
        stage.rename(entry)
    except BaseException:
        _restore_moved_outputs(moved)
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _write_work_metadata(
    work_dir: Path,
    *,
    key: str,
    status: str,
    entry: Path | None,
) -> None:
    payload = {
        "schema_version": CACHE_SCHEMA,
        "key": key,
        "status": status,
    }
    if entry is not None:
        payload["entry"] = str(entry)
    (work_dir / _WORK_METADATA).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _run_without_cache(args: argparse.Namespace, key: str) -> str:
    _run_reference_inference(args)
    work_dir = Path(args.work_dir).resolve()
    if not task_eval.predictions_file_valid(
        work_dir / "hf_predictions.json",
        work_dir / "answers.json",
    ):
        raise ReferenceError("reference inference produced invalid predictions")
    _write_work_metadata(work_dir, key=key, status="generated", entry=None)
    print("Reference result: generated", flush=True)
    return "generated"


def _native_reference_command(
    args: argparse.Namespace,
    runner: Path,
) -> list[str]:
    work_dir = Path(args.work_dir).resolve()
    command = [
        sys.executable,
        str(runner),
        "--model",
        str(args.model),
        "--prompts",
        str(work_dir / "prompts.jsonl"),
        "--answers",
        str(work_dir / "answers.json"),
        "--manifest",
        str(work_dir / "manifest.json"),
        "--predictions",
        str(work_dir / args.predictions),
        "--raw-output",
        str(work_dir / args.raw_output),
        "--repro-metadata",
        str(work_dir / _NATIVE_REPRO_METADATA),
        "--dtype",
        str(args.dtype),
        "--device",
        str(args.device),
    ]
    if args.reference_family:
        command.extend(["--reference-family", str(args.reference_family)])
    revision_runners = {
        _TRANSFORMERS_TEXT_RUNNER,
        _TRANSFORMERS_ENCODER_RUNNER,
        _TRANSFORMERS_VLM_RUNNER,
        _SPEECH_REFERENCE_RUNNER,
    }
    if runner in revision_runners and args.model_revision:
        command.extend(["--model-revision", str(args.model_revision)])
    if runner == _ELF_PREPARED_RUNNER and args.elf_reference_repo:
        command.extend(["--elf-reference-repo", str(args.elf_reference_repo)])
    if runner == _SPEECH_REFERENCE_RUNNER and args.family:
        command.extend(["--family", str(args.family)])
    for flag, value in (
        ("--device-map", args.device_map),
        ("--attn-impl", args.attn_impl),
        ("--max-new-tokens", args.max_new_tokens),
        ("--temperature", args.temperature),
        ("--top-k", args.top_k),
        ("--top-p", args.top_p),
        ("--seed", args.seed),
    ):
        if value not in (None, ""):
            command.extend([flag, str(value)])
    for enabled, flag in (
        (args.trust_remote_code, "--trust-remote-code"),
        (args.local_files_only, "--local-files-only"),
        (args.do_sample, "--do-sample"),
        (args.apply_chat_template, "--apply-chat-template"),
    ):
        if enabled:
            command.append(flag)
    return command


def _run_reference_inference(args: argparse.Namespace) -> None:
    runner = _native_reference_runner(args)
    if runner is None:
        task_eval.run_hf_reference(args)
        return
    work_dir = Path(args.work_dir).resolve()
    command = _native_reference_command(args, runner)
    log_path = work_dir / _NATIVE_RUN_LOG
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {shlex.join(command)}\n")
        log_file.flush()
        process = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    if process.returncode != 0:
        raise ReferenceError(
            f"native reference failed with rc={process.returncode}; see {log_path}"
        )


def run_reference(args: argparse.Namespace) -> str:
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    key, settings = reference_key(args)
    if not args.cache_dir:
        return _run_without_cache(args, key)

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    entry = _cache_entry(cache_dir, key)
    cached = None if args.force else _load_cache_metadata(entry, key)
    if cached is not None:
        _clear_reference_outputs(work_dir)
        _materialize(entry, work_dir)
        _write_work_metadata(work_dir, key=key, status="reused", entry=entry)
        print(f"Reference result: reused {key[:12]}", flush=True)
        return "reused"

    if entry.exists():
        shutil.rmtree(entry)

    adopt = args.adopt_existing and task_eval.predictions_file_valid(
        work_dir / "hf_predictions.json",
        work_dir / "answers.json",
    )
    if not adopt:
        _clear_reference_outputs(work_dir)
        _run_reference_inference(args)
    if not task_eval.predictions_file_valid(
        work_dir / "hf_predictions.json",
        work_dir / "answers.json",
    ):
        raise ReferenceError("reference inference produced invalid predictions")

    status = "adopted" if adopt else "generated"
    metadata = {
        "schema_version": CACHE_SCHEMA,
        "key": key,
        "settings": settings,
        "status": status,
    }
    _move_outputs_to_cache(
        work_dir=work_dir,
        cache_dir=cache_dir,
        entry=entry,
        metadata=metadata,
    )
    _materialize(entry, work_dir)
    _write_work_metadata(work_dir, key=key, status=status, entry=entry)
    print(f"Reference result: {status} {key[:12]}", flush=True)
    return status


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--min-p", type=float)
    parser.add_argument("--seed", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and cache a model reference.")
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--family", default="")
    parser.add_argument("--reference-family", default="")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument(
        "--reference-cache-identity",
        default="",
        help=(
            "Explicit identity shared by TRTMC variants with the same reference "
            "contract. Native-runner cache keys normalize variant-only task metadata "
            "while preserving prepared inputs and effective inference settings."
        ),
    )
    parser.add_argument("--predictions", default="hf_predictions.json")
    parser.add_argument("--raw-output", default="hf_raw.jsonl")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="")
    parser.add_argument("--attn-impl", default="")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--elf-reference-repo", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--adopt-existing", action="store_true")
    add_generation_args(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_reference(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
