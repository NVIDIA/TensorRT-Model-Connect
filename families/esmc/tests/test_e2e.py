# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for ESMC.

**The gate is the logit values.** The breaks this is meant to catch are the
ones that load every tensor and still run. Measured against the fp32 reference
on the sequence the 300M case ships (NVIDIA A100-SXM4-40GB, TensorRT
11.1.0.106, CUDA 13.0, torch 2.12.0, transformers 5.24.0), worst absolute
difference over the masked row:

=========================================================  ==========
build                                                      maxabs
=========================================================  ==========
correct, TensorRT fp32                                     5.5e-05
correct, TensorRT fp16                                     1.3e-01
correct, fp64 NumPy transcription of this graph             1.3e-04
attention branch not divided by sqrt(layers / 36)          5.2e-02
MLP built as erf-gelu(up(x)) instead of SwiGLU             6.2e+00
decoder taken from the embedding table instead of its own  2.0e+01
query/key normalised per head instead of over the width    2.0e+01
query/key not normalised at all                            2.2e+01
=========================================================  ==========

The first two rows come from engines and the reference. The fp64 row and the
four break rows come from an fp64 NumPy transcription of the same graph, which
matches the reference to 1.3e-04 and therefore measures a formula error rather
than a precision one. The fp32 gate of ``_LOGIT_TOLERANCE`` sits about 100x
above the correct build and 10x below the weakest break, and
``test_fp32_gate_rejects_unscaled_residuals`` builds that weakest one in
TensorRT to keep the claim honest.

The fp16 gate is a different shape. The correct fp16 engine is 1.3e-01 off
(fp16's own noise on a residual stream that reaches 1.0e+04; the reference's
fp16 model is 1.1e-01 off by itself), so ``_FP16_LOGIT_TOLERANCE`` cannot sit
below the weakest break: an fp16 engine with the residual scale dropped is
1.5e-01, *inside* the fp16 noise, and no fp16 threshold separates the two. The
fp16 gate therefore catches the gross structural breaks (the last three rows,
all O(1) or worse) and the fp32 case is what gates the residual scale. Both
checkpoints keep top-1 agreement at the masked row in fp32 and fp16.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "esmc"
TASKS = frozenset({"masked_text_to_token_scores"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"

# <mask> is id 32 in every ESMC vocabulary; the tokenizer's own definition is
# what the family reads, and this constant only describes the case prompts.
_MASK_ID = 32

# The two supported checkpoint variants have distinct published geometry.
# Check explicit local overrides too: using one directory for both sides of
# parity otherwise lets a 300M model pass a case advertised as 600M.
_CHECKPOINT_GEOMETRY = {
    "biohub/ESMC-300M": (30, 960, 15, 2560),
    "biohub/ESMC-600M": (36, 1152, 18, 3072),
}


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
    for fixture in ("case_name", "fp16_case_name", "fp32_case_name"):
        if fixture not in metafunc.fixturenames:
            continue
        names, enabled = _selected_cases(metafunc.config)
        if fixture in ("fp16_case_name", "fp32_case_name"):
            precision = "fp16" if fixture == "fp16_case_name" else "fp32"
            names = [name for name in names if CASES[name][1]["precision"] == precision]
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
        path = _required_path(explicit, f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        geometry = tuple(
            config.get(key)
            for key in (
                "num_hidden_layers",
                "hidden_size",
                "num_attention_heads",
                "intermediate_size",
            )
        )
        assert (
            config.get("model_type") == FAMILY
            and geometry == _CHECKPOINT_GEOMETRY[manifest["hf_id"]]
        ), f"local ESMC checkpoint does not match selected {manifest['hf_id']}"
        return path
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


def _run_json(
    binary: Path, runtime_root: Path, bundle: Path, case: dict, command: str, *arguments: str
) -> dict:
    invocation = [
        str(binary),
        command,
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        *arguments,
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        (value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    )
    completed = subprocess.run(
        invocation,
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=int(case.get("runtime_timeout_s", 3600)),
    )
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


# Logits, not features: the lm_head puts these on a wider scale than a hidden
# row, so the tolerance is stated against the reference logits directly.
# Measured: 5.5e-05 (300M) and 4.4e-05 (600M) on A100. The weakest structural
# break is 5.2e-02, so this sits between them by more than an order of
# magnitude on both sides.
_LOGIT_TOLERANCE = 5e-3
# Kept only to fail early and loudly on a gross error.
_COSINE_FLOOR = 0.999

# fp16 cannot use the tolerance above. Measured on the 300M case (A100), a
# correct fp16 engine is 1.3e-01 off the fp32 reference at the masked row
# (600M: 4.7e-02, L4 300M: 1.1e-01), and the same engine with the residual
# scale dropped is 1.5e-01 - inside that noise, so no fp16 threshold separates
# them. This tolerance therefore covers a correct build with 2.4x to 6.4x of
# margin and rejects the O(1) structural breaks. The fp32 case gates the
# residual scale; see test_fp32_gate_rejects_unscaled_residuals.
_FP16_LOGIT_TOLERANCE = 3.0e-1


def _reference_ids(model_dir: Path, text: str) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    return list(tokenizer(text)["input_ids"])


def _reference_logits(model_dir: Path, ids: list[int], window: int):
    import torch
    from transformers import AutoModelForMaskedLM

    model = AutoModelForMaskedLM.from_pretrained(str(model_dir), dtype=torch.float32).eval()
    padded = list(ids) + [1] * (window - len(ids))
    mask = [1] * len(ids) + [0] * (window - len(ids))
    with torch.no_grad():
        scores = model(
            input_ids=torch.tensor([padded]), attention_mask=torch.tensor([mask])
        ).logits[0]
    return scores[: len(ids)].tolist()


def _cosine(left, right) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    assert left_norm > 0.0 and right_norm > 0.0
    return dot / (left_norm * right_norm)


def _worst_error(native, reference) -> tuple[float, int]:
    """Largest absolute difference, and how many rows disagree on the top-1 id."""
    worst = max(max(abs(a - b) for a, b in zip(row, want)) for row, want in zip(native, reference))
    disagreeing = sum(
        1
        for row, want in zip(native, reference)
        if max(range(len(row)), key=row.__getitem__) != max(range(len(want)), key=want.__getitem__)
    )
    return worst, disagreeing


def _check_logits(native, reference, precision: str) -> tuple[float, float]:
    """Assert the gate of ``precision`` on the selected rows; return max abs and cosine."""
    assert len(native) == len(reference)
    worst, disagreeing = _worst_error(native, reference)
    cosine = _cosine([v for row in native for v in row], [v for row in reference for v in row])
    print(
        f"{FAMILY} {precision} parity: logit maxabs {worst:.3e}, cosine {cosine:.8f}, "
        f"top-1 rows differing {disagreeing}/{len(native)}"
    )
    assert cosine >= _COSINE_FLOOR, f"{FAMILY} logit cosine {cosine}"
    tolerance = _FP16_LOGIT_TOLERANCE if precision == "fp16" else _LOGIT_TOLERANCE
    assert worst <= tolerance, f"{FAMILY} {precision} logits differ from the reference by {worst}"
    return worst, cosine


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime()
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)

    prompt = str(case["prompt"])
    payload = _run_json(
        binary,
        runtime_root,
        bundle,
        case,
        "encode",
        "--task",
        "masked_text_to_token_scores",
        "--text",
        prompt,
    )

    # The tokenizer is gated before the logits are. ESMC's tokenizer is a
    # 33-entry BPE with no merges, and the family reproduces its rules in C++;
    # an error there would otherwise surface only as a value mismatch.
    expected_ids = _reference_ids(model_dir, prompt)
    positions = payload["positions"]
    assert payload["score_kind"] == "logit"
    assert payload["axes"] == ["selected_position", "vocabulary_id"]
    assert payload["vocabulary_id"] == "esmc"

    # One <mask> in, one row out: the Task returns the masked positions when
    # the caller marked any.
    masked = [index for index, value in enumerate(expected_ids) if value == _MASK_ID]
    assert len(masked) == 1, "this case is meant to exercise the masked-position path"
    assert [int(token["token_index"]) for token in positions] == masked
    assert [int(token["token_id"]) for token in positions] == [
        expected_ids[index] for index in masked
    ]
    assert all(int(token["input_index"]) == 0 for token in positions)

    shape = [int(value) for value in payload["shape"]]
    assert shape[0] == len(masked)
    # The engine's vocabulary is 64 wide; the tokenizer only names 33 of them.
    assert shape[1] == 64
    values = [float(value) for value in payload["logits"]]
    assert len(values) == shape[0] * shape[1]
    columns = shape[1]
    native = [values[row * columns : (row + 1) * columns] for row in range(shape[0])]

    reference_rows = _reference_logits(
        model_dir, expected_ids, int(manifest["max_sequence_length"])
    )
    reference = [reference_rows[index] for index in masked]
    assert columns == len(reference[0])

    _check_logits(native, reference, manifest["precision"])


def _synthetic(seed: int, noise: float, rows: int = 1) -> tuple[list, list]:
    """Reference logits on the scale of this case, and scores with random noise of size ``noise``."""
    import random

    rng = random.Random(seed)
    reference = [[rng.gauss(0.0, 4.0) for _ in range(64)] for _ in range(rows)]
    scores = [[value + rng.uniform(-noise, noise) for value in row] for row in reference]
    return scores, reference


def test_fp16_gate_takes_fp16_noise_and_fails_a_structural_break() -> None:
    """fp16-sized noise passes the fp16 gate; an error of the residual-scale size fails it."""
    scores, reference = _synthetic(0, 5.0e-2)
    _check_logits(scores, reference, "fp16")
    scores[0][3] = reference[0][3] + 0.5
    with pytest.raises(AssertionError, match="fp16 logits differ"):
        _check_logits(scores, reference, "fp16")


def test_fp32_gate_is_unchanged() -> None:
    """fp32 keeps its own tolerance: fp16-sized noise fails it."""
    scores, reference = _synthetic(0, 1e-5)
    _check_logits(scores, reference, "fp32")
    scores, reference = _synthetic(0, 5.0e-2)
    with pytest.raises(AssertionError, match="fp32 logits differ"):
        _check_logits(scores, reference, "fp32")


def test_fp32_gate_rejects_unscaled_residuals(fp32_case_name: str) -> None:
    """A correct fp32 engine passes the gate; one that leaves out the residual scale fails it.

    Both engines are built from the case checkpoint and run in this process on
    the case prompt, padded to the window the way the runtime pads it. The case
    must be the 30-layer checkpoint: at 36 layers the residual scale is exactly
    1.0, so dropping it changes nothing and there is nothing to discriminate.
    """
    _, manifest, case = CASES[fp32_case_name]
    assert manifest["precision"] == "fp32"
    model_dir = _model_dir(manifest)

    import tensorrt as trt
    import torch

    from .. import config as config_module, model

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    window = int(manifest["max_sequence_length"])
    ids = _reference_ids(model_dir, str(case["prompt"]))
    masked = [index for index, value in enumerate(ids) if value == _MASK_ID]
    reference_rows = _reference_logits(model_dir, ids, window)
    reference = [reference_rows[index] for index in masked]
    weights = model._open_checkpoint(model_dir)
    cfg = config_module.resolve(model_dir, weights, model._prefix(weights))
    if cfg["residual_scale"] == 1.0:
        assert manifest["hf_id"] == "biohub/ESMC-600M", "300M must exercise the residual scale"
        pytest.skip("this checkpoint's residual scale is 1.0, so the break is invisible")

    input_ids = torch.ones(window, dtype=torch.int32, device="cuda")
    input_ids[: len(ids)] = torch.tensor(ids, dtype=torch.int32)
    attention_mask = torch.zeros(window, dtype=torch.int32, device="cuda")
    attention_mask[: len(ids)] = 1

    def run(build_cfg) -> list[list[float]]:
        plan = model.build_engine(weights, build_cfg, seq_len=window, precision="fp32")
        engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
        logits = torch.empty((window, cfg["vocab_size"]), dtype=torch.float32, device="cuda")
        context = engine.create_execution_context()
        for name, tensor in (
            ("input_ids", input_ids),
            ("attention_mask", attention_mask),
            ("logits", logits),
        ):
            context.set_tensor_address(name, tensor.data_ptr())
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        rows = logits.cpu().tolist()
        return [rows[index] for index in masked]

    correct, _ = _check_logits(run(cfg), reference, "fp32")
    broken = run(dict(cfg, residual_scale=1.0))
    with pytest.raises(AssertionError, match="fp32 logits differ"):
        _check_logits(broken, reference, "fp32")
    print(f"{FAMILY} fp32 gate: correct maxabs {correct:.3e}; residual scale dropped fails it")
