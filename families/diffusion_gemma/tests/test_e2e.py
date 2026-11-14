# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-checkpoint native generation and fixed-canvas reference qualification."""

from __future__ import annotations

import json
import os
import string
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build


FAMILY = "diffusion_gemma"
ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "manifests/diffusiongemma-26b-a4b-fp32-512.json").read_text())
CASES = {case["name"]: case for case in MANIFEST["testcases"]}


def _csv(values) -> set[str]:
    return {item.strip() for value in values for item in str(value).split(",") if item.strip()}


def pytest_generate_tests(metafunc) -> None:
    if "case_name" not in metafunc.fixturenames:
        return
    config = metafunc.config
    models = _csv(config.getoption("--e2e-model") or [])
    model_file = config.getoption("--e2e-models-file")
    if model_file:
        models |= {
            line.strip() for line in Path(model_file).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    cases = _csv(config.getoption("--e2e-testcase") or [])
    explicit = bool(models or cases)
    selected = [
        name for name in CASES
        if (not models or {FAMILY, MANIFEST["name"], name} & models)
        and (not cases or name in cases)
    ]
    direct_paths = {ROOT.resolve(), ROOT.parent.resolve(), Path(__file__).resolve()}
    direct = any(Path(str(value).split("::", 1)[0]).resolve() in direct_paths for value in config.args)
    requested_family = bool(models & {FAMILY, MANIFEST["name"]})
    requested_case = any(value.startswith((FAMILY, "diffusiongemma-")) for value in cases)
    if explicit and not selected and (direct or requested_family or requested_case):
        raise pytest.UsageError(f"No diffusion_gemma E2E testcase matches the explicit selectors: {sorted(models | cases)}")
    if not explicit:
        selected = [pytest.param(name, marks=pytest.mark.skip(
            reason="Exact checkpoint E2E requires an explicit E2E selector"
        )) for name in selected]
    metafunc.parametrize("case_name", selected)


def _path(variable: str) -> Path:
    value = os.environ.get(variable)
    assert value, f"Selected {FAMILY} E2E requires {variable}"
    # Preserve virtual-environment Python symlinks and their isolated dependencies.
    path = Path(value).absolute()
    assert path.exists(), f"{variable} does not exist: {path}"
    return path


def _checkpoint(manifest: dict) -> Path:
    if os.environ.get("TRTMC_DIFFUSION_GEMMA_MODEL_DIR"):
        path = _path("TRTMC_DIFFUSION_GEMMA_MODEL_DIR").resolve()
    else:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(
            manifest["hf_id"], revision=manifest["hf_revision"], local_files_only=True,
        )).resolve()
    assert path.name == manifest["hf_revision"], (
        f"Selected {FAMILY} E2E requires the snapshot directory for revision "
        f"{manifest['hf_revision']}, got {path}"
    )
    return path


def normalized_answer(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(value.strip().strip(string.punctuation + "\u2018\u2019\u201c\u201d").split())


def _run(arguments: list[str], *, output: Path, environment: dict[str, str]) -> None:
    with output.open("w", encoding="utf-8") as stream:
        subprocess.run(arguments, check=True, stdout=stream, stderr=subprocess.STDOUT,
                       env=environment, timeout=21600)


def _read(directory: Path, metadata: dict, name: str):
    import numpy as np

    record = metadata["tensors"][name]
    assert record["dtype"] == "float32"
    result = np.fromfile(directory / record["file"], dtype="<f4")
    assert result.size == int(np.prod(record["shape"]))
    return result.reshape(record["shape"])


def reference_reusable(reference: Path, recorded: dict, *, checkpoint: Path,
                       request: dict, precision: str, max_sequence_length: int,
                       canvas_length: int) -> bool:
    from families.diffusion_gemma.tests.original_reference import TRANSFORMERS_REVISION

    embedding_dtype = {"bf16": "bfloat16", "fp32": "float32"}[precision]
    return (
        recorded.get("precision") == precision
        and recorded.get("embedding_dtype") == embedding_dtype
        and recorded.get("max_sequence_length") == max_sequence_length
        and recorded.get("canvas_length") == canvas_length
        and recorded.get("request") == request
        and recorded.get("checkpoint_config") == json.loads((checkpoint / "config.json").read_text())
        and recorded.get("checkpoint_index") == json.loads((checkpoint / "model.safetensors.index.json").read_text())
        and recorded.get("transformers_revision") == TRANSFORMERS_REVISION
        and bool(recorded.get("tensors"))
        and all((reference / tensor["file"]).is_file() for tensor in recorded["tensors"].values())
    )


def compare_observations(native: Path, reference: Path) -> dict:
    """Gate values, token choices, and the complete immutable encoder prefix.

    BF16 kernels may group reductions differently. The gates bound their
    accumulated error while rejecting shape errors, nonfinite values, shuffled
    logits, omitted self-conditioning, and changed encoder caches. Thresholds
    are fixed here and are also exercised against deliberately wrong outputs.
    """
    import numpy as np

    actual = json.loads((native / "metadata.json").read_text())
    expected = json.loads((reference / "metadata.json").read_text())
    assert actual["cache_readonly"] is True
    assert expected["cache_readonly"] is True
    assert actual["decoder_steps"] == expected["decoder_steps"] >= 2
    layers = expected["checkpoint_config"]["text_config"]["num_hidden_layers"]
    required = {
        f"encoder_{kind}_{layer}" for kind in ("key", "value") for layer in range(layers)
    } | {
        f"step_{step}_{kind}" for step in range(expected["decoder_steps"])
        for kind in ("raw_logits", "self_conditioning")
    }
    assert required <= set(expected["tensors"])
    names = set(expected["tensors"]) - {"raw_logits"}
    assert names <= set(actual["tensors"])
    metrics = {}
    for name in sorted(names):
        got, want = _read(native, actual, name), _read(reference, expected, name)
        assert got.shape == want.shape, name
        assert np.isfinite(got).all() and np.isfinite(want).all(), name
        difference = got.astype(np.float64) - want.astype(np.float64)
        rms = float(np.sqrt(np.mean(difference * difference)))
        reference_rms = float(np.sqrt(np.mean(want.astype(np.float64) ** 2)))
        relative_rms = rms / max(reference_rms, 1e-6)
        maximum = float(np.max(np.abs(difference)))
        assert relative_rms <= 0.01, (name, relative_rms)
        limit = 1.0 if name.endswith("raw_logits") else 0.125
        assert maximum <= limit, (name, maximum, limit)
        result = {"relative_rms": relative_rms, "maximum_absolute_error": maximum}
        if name.endswith("raw_logits"):
            agreement = float(np.mean(got.argmax(axis=-1) == want.argmax(axis=-1)))
            assert agreement >= 0.95, (name, agreement)
            result["argmax_agreement"] = agreement
        metrics[name] = result
    return metrics


@pytest.mark.e2e
@pytest.mark.gpu
def test_checkpoint_native_and_reference(case_name: str, tmp_path: Path) -> None:
    import numpy as np
    from transformers import AutoTokenizer

    case = CASES[case_name]
    manifest = MANIFEST
    checkpoint = _checkpoint(manifest)
    native_binary = _path("TRTMC_BINARY")
    runtime_root = _path("TRTMC_RUNTIME_ROOT")
    native_build = _path("TRTMC_NATIVE_BUILD_DIR")
    diagnostic = native_build / "test_diffusion_gemma_diagnostic"
    consumer = native_build / "test_diffusion_gemma_sdk_cpp"
    assert diagnostic.is_file() and consumer.is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    artifacts = Path(os.environ.get("TRTMC_E2E_ARTIFACT_DIR", tmp_path)) / FAMILY / case_name
    artifacts.mkdir(parents=True, exist_ok=True)
    if os.environ.get("TRTMC_DIFFUSION_GEMMA_BUNDLE"):
        bundle = _path("TRTMC_DIFFUSION_GEMMA_BUNDLE")
    else:
        bundle = artifacts / manifest["bundle"]
        build(BuildRequest(
            model_dir=checkpoint, output_path=bundle, family=manifest["family"],
            task=manifest["task"], precision=manifest["precision"],
            max_sequence_length=manifest["max_sequence_length"],
            tensor_parallel_size=manifest["tensor_parallel_size"],
        ))
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = str(runtime_root) + ":" + environment.get("LD_LIBRARY_PATH", "")
    _run([str(native_binary), "inspect", str(bundle)], output=artifacts / "inspect.log",
         environment=environment)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": case["prompt"]}],
        add_generation_prompt=True, tokenize=True, enable_thinking=False, return_dict=False,
    )
    config = json.loads((checkpoint / "config.json").read_text())
    canvases = np.random.default_rng(case["seed"]).integers(
        0, config["text_config"]["vocab_size"], size=(2, config["canvas_length"]),
        dtype=np.int32,
    ).tolist()
    diagnostic_request = artifacts / "diagnostic-request.json"
    diagnostic_request.write_text(json.dumps({
        "input_ids": prompt_ids, "decoder_input_ids": canvases,
        "temperatures": [0.8, 0.4 + 0.4 * 47 / 48],
    }))
    native, reference = artifacts / "native", artifacts / "reference"
    _run([str(diagnostic), str(bundle), str(diagnostic_request), str(native), str(runtime_root)],
         output=artifacts / "native-diagnostic.log", environment=environment)
    native_metadata = json.loads((native / "metadata.json").read_text())
    assert native_metadata["precision"] == manifest["precision"]
    assert native_metadata["max_sequence_length"] == manifest["max_sequence_length"]
    assert native_metadata["canvas_length"] == config["canvas_length"]
    reference_python = os.environ.get("TRTMC_DIFFUSION_GEMMA_REFERENCE_PYTHON", sys.executable)
    reuse = False
    reference_metadata = reference / "metadata.json"
    if os.environ.get("TRTMC_DIFFUSION_GEMMA_REUSE_REFERENCE") == "1" and reference_metadata.is_file():
        recorded = json.loads(reference_metadata.read_text())
        reuse = reference_reusable(
            reference, recorded, checkpoint=checkpoint,
            request=json.loads(diagnostic_request.read_text()), precision=manifest["precision"],
            max_sequence_length=manifest["max_sequence_length"], canvas_length=config["canvas_length"],
        )
    if not reuse:
        _run([reference_python, str(ROOT / "original_reference.py"), str(checkpoint),
              str(diagnostic_request), str(reference), "--precision", manifest["precision"],
              "--max-sequence-length", str(manifest["max_sequence_length"])],
             output=artifacts / "official-reference.log", environment=environment)
    recorded = json.loads(reference_metadata.read_text())
    assert reference_reusable(
        reference, recorded, checkpoint=checkpoint,
        request=json.loads(diagnostic_request.read_text()), precision=manifest["precision"],
        max_sequence_length=manifest["max_sequence_length"], canvas_length=config["canvas_length"],
    )
    metrics = compare_observations(native, reference)
    (artifacts / "reference-metrics.json").write_text(json.dumps(metrics, indent=2))
    generation_request = artifacts / "generation-request.json"
    generation_request.write_text(json.dumps({
        "input_ids": prompt_ids, "max_new_tokens": case["max_new_tokens"], "seed": case["seed"],
    }))
    generation_output = artifacts / "generation-output.json"
    _run([str(consumer), str(bundle), str(runtime_root), str(generation_request), str(generation_output)],
         output=artifacts / "native-generation.log", environment=environment)
    result = json.loads(generation_output.read_text())
    assert 0 < len(result["token_ids"]) <= case["max_new_tokens"]
    assert normalized_answer(result["text"]) == normalized_answer(case["expected_answer"])
    assert result["prefill_ms"] >= 0 and result["decode_ms"] > 0
