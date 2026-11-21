# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for DeBERTa-v2/v3.

**The gate is the feature values.** Seven errors were built on purpose and
measured against this checkpoint, on the passage this case ships:

=========================================  ==========  ============
rule broken                                maxabs      cosine
=========================================  ==========  ============
(correct build)                            4.29e-05    1.00000012
linear relative positions, not log buckets 0.0685      0.99999666
absolute position embeddings added         4.801       0.99581569
scale sqrt(d) instead of sqrt(3d)          7.83        0.94018942
c2p term dropped                           13.37       0.85509247
p2c term dropped                           19.62       0.27363199
c2p and p2c indices swapped                19.91       0.44196913
relative embeddings not layer-normalised   20.32       0.41130102
=========================================  ==========  ============

**The passage is long on purpose.** Below ``position_buckets / 2`` - 128 rows
here - the log bucketing *is* the identity, so on a short input the bucketing
row above measures exactly the baseline and the gate tests nothing. At 148
tokens it costs 0.0685. A 0.999 cosine floor would still miss it at
0.99999666, which is why the values are gated and the cosine is only a
loud-failure guard.

The fp16 case uses its own feature tolerance, set from the fp16 noise
measured on this case; see ``_FP16_FEATURE_TOLERANCE``.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "deberta_v2"
TASKS = frozenset({"text_to_token_features"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"


def _case_index() -> dict[str, tuple[Path, dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] in TASKS
        for case in manifest["testcases"]:
            name = str(case["name"])
            assert name not in result
            result[name] = (path, manifest, case)
    return result


CASES = _case_index()


def _selected_cases(config) -> tuple[list[str], bool]:
    model_filters = set()
    for raw in config.getoption("--e2e-model") or []:
        model_filters.update((item.strip() for item in str(raw).split(",") if item.strip()))
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        model_filters.update(
            (
                line.strip()
                for line in Path(models_file).read_text(encoding="utf-8").splitlines()
                if line.strip() and (not line.lstrip().startswith("#"))
            )
        )
    testcase_filters = set()
    for raw in config.getoption("--e2e-testcase") or []:
        testcase_filters.update((item.strip() for item in str(raw).split(",") if item.strip()))
    if not model_filters and (not testcase_filters):
        return (sorted(CASES), False)
    selected = []
    for name, (_, manifest, _) in CASES.items():
        model_match = (
            not model_filters
            or FAMILY in model_filters
            or name in model_filters
            or (manifest["name"] in model_filters)
        )
        testcase_match = not testcase_filters or name in testcase_filters
        if model_match and testcase_match:
            selected.append(name)
    return (sorted(selected), True)


def pytest_generate_tests(metafunc) -> None:
    for fixture in ("case_name", "fp16_case_name"):
        if fixture not in metafunc.fixturenames:
            continue
        names, enabled = _selected_cases(metafunc.config)
        if fixture == "fp16_case_name":
            names = [name for name in names if CASES[name][1]["precision"] == "fp16"]
        parameters = names
        if not enabled:
            parameters = [
                pytest.param(
                    name,
                    marks=pytest.mark.skip(
                        reason="direct E2E requires one of the three explicit E2E selectors"
                    ),
                )
                for name in names
            ]
        metafunc.parametrize(fixture, parameters, ids=names)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected {FAMILY} E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected {FAMILY} E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get(f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    if explicit:
        return _required_path(explicit, f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            repo_id=manifest["hf_id"], revision=manifest.get("hf_revision"), local_files_only=True
        )
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}"
        ) from error
    return Path(snapshot)


def _runtime() -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    assert torch.cuda.device_count() >= 1, f"selected {FAMILY} E2E requires one GPU"
    return (binary, runtime_root)


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            max_sequence_length=manifest.get("max_sequence_length"),
            max_batch_size=int(manifest.get("max_batch_size", 1)),
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
            quantization=manifest.get("quantization"),
            fp32_layers=tuple((int(layer) for layer in manifest.get("fp32_layers", ()))),
        )
    )


def _run_json(binary: Path, runtime_root: Path, bundle: Path, case: dict,
              command: str, *arguments: str) -> dict:
    invocation = [str(binary), command, str(bundle), "--runtime-root", str(runtime_root),
                  *arguments]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        (value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    )
    completed = subprocess.run(invocation, check=True, capture_output=True, text=True, env=env,
                               timeout=int(case.get("runtime_timeout_s", 3600)))
    payloads = []
    for line in completed.stdout.splitlines():
        start = line.find("{")
        if start >= 0:
            try:
                payloads.append(json.loads(line[start:]))
            except json.JSONDecodeError:
                pass
    assert payloads, f"native {command} returned no JSON: {completed.stdout[-1000:]}"
    assert all((payload == payloads[0] for payload in payloads))
    return payloads[0]


# Measured on this case: a correct build is within 4.29e-05 of the reference
# and the weakest break the gate is meant to catch - the relative-position
# bucketing - is 0.0685. This sits 116x above the noise and 13x below that
# break.
_FEATURE_TOLERANCE = 5e-3
# Kept only to fail early and loudly on a gross error; measured, it cannot
# see the bucketing break at all.
_COSINE_FLOOR = 0.999

# fp16 cannot use the tolerance above. Measured on this case (B200 and B300,
# TensorRT 11.1 and 11.4, 24 builds), a correct fp16 engine is 4.5e-02 to
# 7.8e-02 off the reference, and an fp16 engine scaled by sqrt(d) instead of
# sqrt(3d) 7.81 to 7.86 (12 builds). 2.5e-01 sits 3.2x above the first and 31x
# below the second; the cosine floor is shared (fp16 noise moves it by under
# 1e-05).
#
# The bucketing break (0.0685 in fp32) is inside the fp16 noise and is not
# gated here: an fp16 build with linear relative positions is 6.9e-02 to
# 8.9e-02 off on this passage. The index matrices come from rel_position,
# which does not depend on the precision, so the fp32 case gates them for both.
#
# test_fp16_gate_rejects_a_wrong_attention_scale builds both and checks this.
_FP16_FEATURE_TOLERANCE = 2.5e-1


def _reference_ids(model_dir: Path, text: str) -> list[int]:
    """The framing the checkpoint was trained with, built explicitly.

    The installed transformers builds this tokenizer with a post-processor
    that adds no special tokens, so ``tokenizer(text)`` returns the body
    alone. The framing is applied here, and the family is required to produce
    the same ids, rather than letting a tokenizer-version quirk decide what
    the gate compares.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    body = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [tokenizer.cls_token_id, *body, tokenizer.sep_token_id]


def _reference_features(model_dir: Path, ids: list[int], window: int):
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(str(model_dir), dtype=torch.float32).eval()
    padded = list(ids) + [0] * (window - len(ids))
    mask = [1] * len(ids) + [0] * (window - len(ids))
    with torch.no_grad():
        hidden = model(input_ids=torch.tensor([padded]),
                       attention_mask=torch.tensor([mask])).last_hidden_state[0]
    return hidden[: len(ids)].tolist()


def _cosine(left, right) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    assert left_norm > 0.0 and right_norm > 0.0
    return dot / (left_norm * right_norm)


def _check_features(native, reference, precision: str) -> tuple[float, float]:
    """Assert the gate of ``precision`` on the real rows; return max abs and cosine."""
    assert len(native) == len(reference)
    worst = max(max(abs(a - b) for a, b in zip(row, want))
                for row, want in zip(native, reference))
    cosine = _cosine([v for row in native for v in row],
                     [v for row in reference for v in row])
    assert cosine >= _COSINE_FLOOR, f"{FAMILY} feature cosine {cosine}"
    tolerance = _FP16_FEATURE_TOLERANCE if precision == "fp16" else _FEATURE_TOLERANCE
    assert worst <= tolerance, (
        f"{FAMILY} {precision} features differ from the reference by {worst}"
    )
    return worst, cosine


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime()
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)

    prompt = str(case["prompt"])
    payload = _run_json(binary, runtime_root, bundle, case, "encode", "--task",
                        "text_to_token_features", "--text", prompt)

    # The tokenizer is gated before the features are. The runtime converts
    # spm.model at build time and frames the result itself, so an error in
    # either would otherwise show up only as a feature mismatch with no
    # indication of where it came from.
    expected_ids = _reference_ids(model_dir, prompt)
    tokens = payload["tokens"]
    assert [int(token["token_id"]) for token in tokens] == expected_ids, (
        f"{FAMILY} token ids differ from the reference tokenizer"
    )
    assert [int(token["token_index"]) for token in tokens] == list(range(len(expected_ids)))
    assert all(int(token["input_index"]) == 0 for token in tokens)

    # Long enough that the bucketing is not the identity; see the module
    # docstring for what a short passage would fail to test.
    assert len(expected_ids) > int(manifest["max_sequence_length"]) // 4, (
        "this case is meant to exercise the relative-position bucketing"
    )

    # The matrix is tokens by features, and says so: a transposed result
    # would otherwise have the right element count and the wrong meaning.
    assert payload["feature_kind"] == "token"
    assert payload["axes"] == ["token", "feature"]
    shape = [int(value) for value in payload["shape"]]
    assert shape[0] == len(expected_ids), (
        f"{FAMILY} returned {shape[0]} rows for {len(expected_ids)} tokens"
    )
    assert shape[1] == int(payload["dim"])

    values = [float(value) for value in payload["values"]]
    assert len(values) == shape[0] * shape[1]
    columns = shape[1]
    native = [values[row * columns:(row + 1) * columns] for row in range(shape[0])]
    reference = _reference_features(model_dir, expected_ids,
                                    int(manifest["max_sequence_length"]))

    worst, cosine = _check_features(native, reference, manifest["precision"])
    print(
        f"{FAMILY} {manifest['precision']} parity: {shape[0]} tokens x {shape[1]} features, "
        f"feature maxabs {worst:.3e}, cosine {cosine:.8f}"
    )


def _synthetic(seed: int, noise: float, rows: int = 4) -> tuple[list, list]:
    """Unit-scale reference features, and the same features with uniform noise of ``noise``."""
    import random

    rng = random.Random(seed)
    reference = [[rng.gauss(0.0, 1.0) for _ in range(64)] for _ in range(rows)]
    features = [[value + rng.uniform(-noise, noise) for value in row] for row in reference]
    return features, reference


def test_fp16_gate_takes_fp16_noise_and_fails_a_larger_error() -> None:
    """fp16-sized noise passes the fp16 gate; one feature off by twice the tolerance fails it."""
    features, reference = _synthetic(0, 2e-2)
    features[2][5] = reference[2][5] + 7.8e-2
    _check_features(features, reference, "fp16")
    features[1][7] = reference[1][7] + 2 * _FP16_FEATURE_TOLERANCE
    with pytest.raises(AssertionError, match="fp16 features differ"):
        _check_features(features, reference, "fp16")


def test_fp32_gate_is_unchanged() -> None:
    """fp32 keeps its own tolerance: fp16-sized noise fails it."""
    features, reference = _synthetic(0, 4.3e-5)
    _check_features(features, reference, "fp32")
    features, reference = _synthetic(0, 2e-2)
    with pytest.raises(AssertionError, match="fp32 features differ"):
        _check_features(features, reference, "fp32")


def test_fp16_gate_rejects_a_wrong_attention_scale(fp16_case_name: str) -> None:
    """A correct fp16 engine passes the fp16 gate; one scaled by sqrt(d) instead of sqrt(3d) fails it.

    Both engines are built from the case checkpoint and run in this process on
    the case passage, padded to the window the way the runtime pads it.
    """
    import tensorrt as trt
    import torch

    from .. import config as config_module, model

    _, manifest, case = CASES[fp16_case_name]
    assert manifest["precision"] == "fp16"
    model_dir = _model_dir(manifest)
    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    window = int(manifest["max_sequence_length"])
    ids = _reference_ids(model_dir, str(case["prompt"]))
    reference = _reference_features(model_dir, ids, window)
    weights = model._open_checkpoint(model_dir)
    cfg = config_module.resolve(model_dir, weights, model._prefix(weights))

    input_ids = torch.zeros(window, dtype=torch.int32, device="cuda")
    input_ids[: len(ids)] = torch.tensor(ids, dtype=torch.int32)
    attention_mask = torch.zeros(window, dtype=torch.int32, device="cuda")
    attention_mask[: len(ids)] = 1

    def run(build_cfg) -> list[list[float]]:
        plan = model.build_engine(weights, build_cfg, seq_len=window, precision="fp16")
        engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
        context = engine.create_execution_context()
        hidden = torch.empty((window, cfg["hidden_size"]), dtype=torch.float32, device="cuda")
        for name, tensor in (("input_ids", input_ids), ("attention_mask", attention_mask),
                             ("last_hidden_state", hidden)):
            context.set_tensor_address(name, tensor.data_ptr())
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return hidden[: len(ids)].cpu().tolist()

    worst, _ = _check_features(run(cfg), reference, "fp16")
    broken = run(dict(cfg, scale_factor=1))
    broken_worst = max(max(abs(a - b) for a, b in zip(row, want))
                       for row, want in zip(broken, reference))
    assert broken_worst > _FP16_FEATURE_TOLERANCE
    with pytest.raises(AssertionError):
        _check_features(broken, reference, "fp16")
    print(
        f"{FAMILY} fp16 gate: correct maxabs {worst:.3e}; sqrt(d) scale maxabs "
        f"{broken_worst:.3e}"
    )
