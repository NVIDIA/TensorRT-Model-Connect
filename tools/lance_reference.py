#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a repeatable Lance x2t_image benchmark through the pinned upstream code."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
from typing import Any, Iterator, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-repo", required=True, type=Path)
    parser.add_argument("--reference-commit", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--model-subdir", default="Lance_3B")
    parser.add_argument("--vit-subdir", default="Qwen2.5-VL-ViT")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--instruction",
        default="Look at the image carefully and answer the question.",
    )
    parser.add_argument("--max-new-tokens", required=True, type=int)
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--resolution", default="image_768res")
    parser.add_argument("--height", default=768, type=int)
    parser.add_argument("--width", default=768, type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _git_revision(repository: Path) -> str:
    repository = repository.resolve()
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository}",
            "-C",
            str(repository),
            "rev-parse",
            "HEAD",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"cannot resolve Lance checkout revision: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _check_reference(repository: Path, expected_commit: str) -> str:
    repository = repository.resolve()
    source = repository / "inference_lance.py"
    if not source.is_file():
        raise FileNotFoundError(f"Lance checkout has no inference_lance.py: {repository}")
    revision = _git_revision(repository)
    if revision != expected_commit:
        raise RuntimeError(
            "Lance checkout revision does not match release.yaml: "
            f"expected {expected_commit}, found {revision}"
        )
    for arguments in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repository}",
                "-C",
                str(repository),
                *arguments,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError("Lance checkout has tracked local changes")
    return revision


def _snapshot_revision(path: Path) -> str:
    parts = path.resolve().parts
    try:
        index = parts.index("snapshots")
    except ValueError:
        return "local-path"
    return parts[index + 1] if index + 1 < len(parts) else "unresolved"


def _model_paths(arguments: argparse.Namespace) -> tuple[Path, Path, str]:
    model_source = Path(arguments.model)
    if model_source.exists():
        root = model_source.resolve()
    else:
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                repo_id=arguments.model,
                revision=arguments.revision,
                local_files_only=arguments.local_files_only,
                allow_patterns=[
                    f"{arguments.model_subdir}/**",
                    f"{arguments.vit_subdir}/**",
                ],
            )
        )
    model_path = root / arguments.model_subdir
    vit_path = root / arguments.vit_subdir
    required = (
        model_path / "llm_config.json",
        model_path / "model.safetensors",
        vit_path / "vit.safetensors",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Lance snapshot is incomplete: " + ", ".join(missing))
    return model_path, vit_path, _snapshot_revision(root)


def _dataset_payload(*, image: Path, prompt: str, instruction: str, count: int) -> dict[str, Any]:
    sample = {
        "interleave_array": [str(image.resolve()), [instruction, prompt, ""]],
        "element_dtype_array": ["image", "text"],
        "istarget_in_interleave": [0, 1],
    }
    return {f"{index:04d}": dict(sample) for index in range(count)}


def _decord_image_only_stub() -> dict[str, types.ModuleType]:
    """Provide import compatibility while rejecting unsupported video use."""

    class VideoReader:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError(
                "The aarch64 Lance x2t-image reference does not provide decord; "
                "video workloads require an official decord build"
            )

    decord = types.ModuleType("decord")
    video_reader = types.ModuleType("decord.video_reader")
    decord.VideoReader = VideoReader
    decord.video_reader = video_reader
    video_reader.VideoReader = VideoReader

    def unsupported_cpu(*_args, **_kwargs):
        raise RuntimeError("decord CPU contexts are unavailable for this image-only reference")

    decord.cpu = unsupported_cpu
    return {"decord": decord, "decord.video_reader": video_reader}


def _load_upstream(reference_repo: Path) -> Any:
    try:
        import decord  # noqa: F401
    except ImportError:
        sys.modules.update(_decord_image_only_stub())
    source = reference_repo / "inference_lance.py"
    spec = importlib.util.spec_from_file_location("trtmc_lance_upstream", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import upstream Lance source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def _arguments(values: Sequence[str]) -> Iterator[None]:
    previous = sys.argv
    sys.argv = list(values)
    try:
        yield
    finally:
        sys.argv = previous


def _upstream_argv(
    arguments: argparse.Namespace,
    model_path: Path,
    vit_path: Path,
    dataset: Path,
    result_directory: Path,
) -> list[str]:
    return [
        "inference_lance.py",
        "--model_path",
        str(model_path),
        "--vit_path",
        str(vit_path),
        "--val_dataset_config_file",
        str(dataset),
        "--vit_type",
        "qwen_2_5_vl_original",
        "--llm_qk_norm",
        "true",
        "--llm_qk_norm_und",
        "true",
        "--llm_qk_norm_gen",
        "true",
        "--tie_word_embeddings",
        "false",
        "--validation_num_timesteps",
        "30",
        "--validation_timestep_shift",
        "3.5",
        "--copy_init_moe",
        "true",
        "--max_num_frames",
        "121",
        "--max_latent_size",
        "64",
        "--latent_patch_size",
        "1",
        "1",
        "1",
        "--visual_und",
        "true",
        "--visual_gen",
        "true",
        "--vae_model_type",
        "wan",
        "--apply_qwen_2_5_vl_pos_emb",
        "true",
        "--apply_chat_template",
        "false",
        "--cfg_type",
        "0",
        "--validation_data_seed",
        "42",
        "--video_height",
        str(arguments.height),
        "--video_width",
        str(arguments.width),
        "--num_frames",
        "1",
        "--task",
        "x2t_image",
        "--save_path_gen",
        str(result_directory),
        "--resolution",
        arguments.resolution,
        "--text_template",
        "true",
        "--cfg_text_scale",
        "4.0",
        "--use_KVcache",
        "true",
        "--enhance_prompt",
        "false",
    ]


def _run_upstream(
    arguments: argparse.Namespace,
    model_path: Path,
    vit_path: Path,
    dataset: Path,
    result_directory: Path,
) -> tuple[list[float], list[str]]:
    import torch

    reference_repo = arguments.reference_repo.resolve()
    sys.path.insert(0, str(reference_repo))
    upstream = _load_upstream(reference_repo)
    upstream.MAX_GENERATION_LENGTH = arguments.max_new_tokens
    original = upstream.validate_on_fixed_batch
    samples: list[float] = []
    answers: list[str] = []

    def measured(*args: Any, **kwargs: Any) -> Any:
        torch.cuda.synchronize()
        started = time.perf_counter()
        value = original(*args, **kwargs)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
        inference_args = kwargs.get("inference_args")
        prompt_data = getattr(inference_args, "prompt_data_dict", {})
        raw = list(prompt_data.values())[-1] if prompt_data else ""
        answers.append(upstream.normalize_understanding_answer(str(raw)))
        return value

    upstream.validate_on_fixed_batch = measured
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        os.environ.pop(name, None)
    argv = _upstream_argv(arguments, model_path, vit_path, dataset, result_directory)
    with _working_directory(reference_repo), _arguments(argv):
        upstream.main()
    expected = arguments.warmup + arguments.iterations
    if len(samples) != expected:
        raise RuntimeError(f"Lance produced {len(samples)} samples; expected {expected}")
    return samples[arguments.warmup :], answers[arguments.warmup :]


def run(arguments: argparse.Namespace) -> int:
    if arguments.warmup < 0 or arguments.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if arguments.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    if not arguments.image.is_file():
        raise FileNotFoundError(f"Lance input image does not exist: {arguments.image}")
    reference_revision = _check_reference(
        arguments.reference_repo.resolve(), arguments.reference_commit
    )
    model_path, vit_path, model_revision = _model_paths(arguments)
    count = arguments.warmup + arguments.iterations
    with tempfile.TemporaryDirectory(prefix="trtmc-perf-lance-") as temporary:
        root = Path(temporary)
        dataset = root / "x2t_image.json"
        dataset.write_text(
            json.dumps(
                _dataset_payload(
                    image=arguments.image,
                    prompt=arguments.prompt,
                    instruction=arguments.instruction,
                    count=count,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        samples, answers = _run_upstream(arguments, model_path, vit_path, dataset, root / "results")
    payload = {
        "samples_ms": samples,
        "text": answers[0] if answers else "",
        "model_revision": model_revision,
        "reference_revision": reference_revision,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"lance reference error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
