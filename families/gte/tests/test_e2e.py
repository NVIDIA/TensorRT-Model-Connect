# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for GTE-v1.5.

**The gate is the pooled [CLS] values.** Eleven errors were built on purpose
and measured against this checkpoint at 512 positions:

==========================================  ==============  ============
rule broken                                 [CLS] maxabs    [CLS] cosine
==========================================  ==============  ============
(correct build)                             2.92e-06        1.00000000
rotary table in float64 instead of float32  2.86e-06        1.00000000
tanh gelu instead of the exact erf gelu     2.25e-03        0.99999971
NTK base doubled, frequencies not divided   4.17e-02        0.99991537
rotary NTK scaling ignored (base 500000)    2.59e-01        0.99814874
qkv projection bias dropped                 6.58e-01        0.97404702
interleaved rotary instead of split-half    1.43            0.91632873
attention mask dropped (pads attend)        1.59            0.79614703
embedding LayerNorm dropped                 2.04            0.86376769
mean pooling instead of [CLS]               2.89            0.95699990
qkv split head-major instead of q,k,v       7.09            0.12725731
gelu applied to up instead of gate          9.45            0.20465821
pre-norm block instead of post-norm         134             0.43299265
==========================================  ==============  ============

Two things that table is honest about:

* A **cosine floor cannot be the gate here.** The tanh-gelu build keeps
  0.9999997 of cosine and the NTK-base-only build keeps 0.9999154, so any
  floor loose enough not to fire on build noise would pass both. The floor
  below is kept only to fail early and loudly on a gross error; the values
  are what is gated.
* The float64 rotary table is a **break this gate does not catch**. The
  table itself is wrong by 3.9e-05 against the reference's own
  ``cos_cached``, and the pooled vector moves by 2.86e-06 - the correct
  build's noise. ``families/gte/rope.py`` says so rather than claiming the
  gate covers it, and ``tests/test_model.py`` asserts on the table instead.

``_EMBEDDING_TOLERANCE`` is 3e-4: 103x above a correct build's 2.92e-06 and
7.5x below the weakest break this gate does catch.

The fp16 case uses its own tolerance, set from the fp16 noise measured on
this case; see ``_FP16_EMBEDDING_TOLERANCE``. The FP8 case (fp16 work type,
``quantization: fp8``) has its own tolerance and cosine floor; see
``_FP8_EMBEDDING_TOLERANCE``.
"""

from __future__ import annotations

from tools.e2e_evidence import record_evidence

import json
import math
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "gte"
TASKS = frozenset({"text_to_embedding"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"


def _case_index() -> dict[str, tuple[Path, dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] in TASKS
        # The checkpoint's auto_map points at modules that live in a second
        # repository, Alibaba-NLP/new-impl. Offline CI stages only what the
        # manifest declares, so an undeclared dependency fails as a missing
        # import rather than as a missing checkpoint. The declaration carries
        # no revision on purpose: auto_map names no revision either, so
        # transformers resolves the modules at "main", and staging a pinned
        # commit caches it under the sha without writing the refs/main alias,
        # which leaves the offline lookup failing exactly as if nothing had
        # been staged.
        declared = {entry["repo_id"] for entry in manifest["hf_dependencies"]}
        assert declared == {"Alibaba-NLP/new-impl"}, (
            f"{path.name} must declare the repository its auto_map names"
        )
        assert all("revision" not in entry for entry in manifest["hf_dependencies"]), (
            f"{path.name} must declare its auto_map dependency without a revision"
        )
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
    for fixture in ("case_name", "fp16_case_name", "fp8_case_name"):
        if fixture not in metafunc.fixturenames:
            continue
        names, enabled = _selected_cases(metafunc.config)
        if fixture == "fp16_case_name":
            names = [name for name in names if _gate(CASES[name][1]) == "fp16"]
        if fixture == "fp8_case_name":
            names = [name for name in names if _gate(CASES[name][1]) == "fp8"]
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


# Measured on this case: a correct build is within 2.92e-06 of the reference
# and the weakest break this gate catches - the tanh gelu - is 2.25e-03.
# 3e-4 sits 103x above the noise and 7.5x below that break.
_EMBEDDING_TOLERANCE = 3e-4
# Kept only to fail early and loudly on a gross error. It cannot be the gate:
# two of the measured breaks keep more than 0.9999 of cosine.
_COSINE_FLOOR = 0.999

# fp16 cannot use the tolerance above. Measured on this case (B200 and B300,
# TensorRT 11.1 and 11.4, 14 builds), a correct fp16 engine is 5.2e-03 to
# 8.3e-03 off the reference, and fp16 engines with the breaks of the table
# are: NTK scaling ignored 2.5e-01 to 2.7e-01, NTK base doubled 4.1e-02 to
# 4.5e-02. 2.5e-02 sits 3.0x above the worst correct build, 10x below the
# first break and 1.6x below the second.
#
# The tanh-gelu break (2.25e-03 in fp32) is inside the fp16 noise and is not
# gated here: an fp16 build with it is 5.6e-03 to 8.3e-03 off, the range of a
# correct one. The gelu form is chosen by the same code at every precision,
# so the fp32 case gates it for both.
#
# test_fp16_gate_rejects_ignored_ntk_scaling builds both and checks this.
_FP16_EMBEDDING_TOLERANCE = 2.5e-2

# FP8 (projections quantized, fp16 work type) cannot use either tolerance
# above. Measured on this case (B200 and B300, TensorRT 11.1 and 11.4, 8
# builds), a correct FP8 engine is 0.262 off the reference in every build
# (cosine 0.9956). FP8 engines with deliberate errors: the activation scales
# of the wrong layers (the table read in reverse layer order) 3.53, every
# activation scale 8x too small 5.20. 0.8 sits 3.1x above the correct build
# and 4.4x below the first break.
#
# Rotary breaks are inside the FP8 noise (NTK scaling ignored 0.297, NTK base
# doubled 0.252) and are not gated here; the rotary code is the same at every
# precision, so the fp32 and fp16 cases gate it.
#
# test_fp8_gate_rejects_scales_of_the_wrong_layers builds both and checks this.
_FP8_EMBEDDING_TOLERANCE = 0.8
# Correct FP8 builds keep 0.9956 of cosine and the two breaks above 0.71 and
# 0.57; the floor only fails early on a gross error, like the fp32 one.
_FP8_COSINE_FLOOR = 0.98


def _gate(manifest: dict) -> str:
    """The gate of a case: ``fp8`` for a quantized case, otherwise its precision."""
    return "fp8" if manifest.get("quantization") == "fp8" else str(manifest["precision"])


def _reference(model_dir: Path, text: str):
    """The reference's own pooled vector, pooled the way its module list says."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    # trust_remote_code on both calls, not only the model: this checkpoint's
    # config carries an auto_map, so the tokenizer loader resolves it too and
    # would otherwise stop to ask on a terminal that is not there.
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True,
                                      dtype=torch.float32).eval()
    # The remote code registers its rotary tables with persistent=False, so
    # they are not in model.safetensors and the loader has nothing to put in
    # them. Under the pinned transformers they come back as uninitialised
    # memory rather than as the module's own values, which makes the
    # reference silently wrong - one run returned an inv_freq of 5e-16 and
    # the next an index error. Rebuilding them calls the checkpoint's own
    # constructor again; it reads no weights and changes no arithmetic.
    model.embeddings._init_rope(model.config)
    model.embeddings.register_buffer(
        "position_ids", torch.arange(model.config.max_position_embeddings), persistent=False)
    inv_freq = model.embeddings.rotary_emb.inv_freq
    assert torch.isfinite(inv_freq).all() and float(inv_freq.max()) <= 1.0, (
        f"{FAMILY} reference rotary buffers did not survive loading: {inv_freq[:4].tolist()}"
    )

    batch = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state[0]
    # 1_Pooling/config.json sets pooling_mode_cls_token and clears every
    # other mode, and modules.json lists no Normalize step, so the published
    # vector is row 0 with its length intact.
    return hidden[0].tolist()


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    assert left_norm > 0.0 and right_norm > 0.0
    return dot / (left_norm * right_norm)


def _check_embedding(values, reference, precision: str, text: str) -> tuple[float, float]:
    """Assert the gate of ``precision`` (fp32, fp16 or fp8) on one pooled vector.

    Returns the max abs difference and the cosine.
    """
    delta = max(abs(a - b) for a, b in zip(values, reference))
    cosine = _cosine(values, reference)
    floor = _FP8_COSINE_FLOOR if precision == "fp8" else _COSINE_FLOOR
    record_evidence("thresholds", {"cosine_min": floor})
    assert cosine >= floor, f"{FAMILY} {precision} embedding cosine {cosine} for {text!r}"
    tolerance = {"fp16": _FP16_EMBEDDING_TOLERANCE, "fp8": _FP8_EMBEDDING_TOLERANCE}.get(
        precision, _EMBEDDING_TOLERANCE)
    record_evidence("thresholds", {"embedding_max_abs": tolerance, "cosine_min": floor})
    assert delta <= tolerance, (
        f"{FAMILY} {precision} embedding for {text!r} differs from the reference by {delta}"
    )
    return delta, cosine


def _native(binary, runtime_root, bundle, case, text: str, *arguments: str) -> dict:
    return _run_json(binary, runtime_root, bundle, case, "embed", "--task",
                     "text_to_embedding", "--text", text, *arguments)


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    record_evidence("checkpoint", {"hf_id": manifest["hf_id"], "hf_revision": manifest.get("hf_revision")})
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime()
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)

    vectors = []
    worst = 0.0
    worst_cosine = 1.0
    for text in (case["query"], case["document"], case["negative_document"]):
        payload = _native(binary, runtime_root, bundle, case, str(text))
        record_evidence("inputs", {"text": str(text)})
        record_evidence("native", payload)
        reference = _reference(model_dir, str(text))
        record_evidence("reference", {"values": reference})
        values = [float(value) for value in payload["values"]]
        assert len(values) == len(reference), (
            f"{FAMILY} returned {len(values)} values, reference {len(reference)}"
        )
        delta, cosine = _check_embedding(values, reference, _gate(manifest), str(text))
        worst = max(worst, delta)
        worst_cosine = min(worst_cosine, cosine)

        # 1_Pooling/config.json sets pooling_mode_cls_token, and modules.json
        # is Transformer then Pooling with no Normalize, so the vector keeps
        # its length and says so. Mean pooling the same rows instead costs
        # 0.043 of cosine against this reference, so these two strings are
        # the difference between two plausible vectors.
        assert payload["pooling"] == "cls"
        assert payload["normalization"] == "none"
        # Three GTE-v1.5 checkpoints share this family and do not share a
        # space, and none of them names the one it is in.
        assert payload["embedding_space"] == ""
        norm = math.sqrt(sum(value * value for value in values))
        assert abs(norm - 1.0) > 1e-3, (
            f"{FAMILY} reports normalization=none but returned a unit vector: {norm}"
        )
        vectors.append(values)

    # This checkpoint ships no role prompt - no prompt template, no Prompt
    # module, and a README that embeds both sides with one call - so a role
    # is refused rather than accepted and ignored. A runtime that accepted
    # one would be claiming an asymmetry the checkpoint does not have.
    for role in ("query", "document"):
        with pytest.raises(subprocess.CalledProcessError):
            _native(binary, runtime_root, bundle, case, str(case["query"]), "--role", role)

    # Retrieval is reported, not gated: it is a property of the checkpoint
    # rather than of this build, and one pair is not a benchmark.
    query, document, negative = vectors
    relevant = _cosine(query, document)
    irrelevant = _cosine(query, negative)
    assert relevant > irrelevant, (
        f"{FAMILY} scored the unrelated document higher: {relevant} <= {irrelevant}"
    )
    print(
        f"{FAMILY} {_gate(manifest)} parity: {len(document)} values, "
        f"embedding maxabs {worst:.3e}, "
        f"cosine {worst_cosine:.8f}, query-vs-answer {relevant:.6f} "
        f"vs query-vs-other {irrelevant:.6f}"
    )


def _synthetic(seed: int, noise: float) -> tuple[list[float], list[float]]:
    """A reference vector on the scale of this case, and one with random noise of size ``noise``."""
    import random

    rng = random.Random(seed)
    reference = [rng.gauss(0.0, 0.5) for _ in range(768)]
    values = [value + rng.uniform(-noise, noise) for value in reference]
    return values, reference


def test_fp16_gate_takes_fp16_noise_and_fails_the_ntk_break() -> None:
    """fp16-sized noise passes the fp16 gate; an error of the ignored-NTK size fails it."""
    values, reference = _synthetic(0, 8.5e-3)
    _check_embedding(values, reference, "fp16", "synthetic")
    values[3] = reference[3] + 0.25
    with pytest.raises(AssertionError, match="fp16 embedding"):
        _check_embedding(values, reference, "fp16", "synthetic")


def test_fp8_gate_takes_fp8_noise_and_fails_the_wrong_scales() -> None:
    """FP8-sized error passes the FP8 gate; an error of the wrong-layer-scales size fails it."""
    values, reference = _synthetic(0, 5e-3)
    values[3] = reference[3] + 0.27
    _check_embedding(values, reference, "fp8", "synthetic")
    with pytest.raises(AssertionError, match="fp16 embedding"):
        _check_embedding(values, reference, "fp16", "synthetic")
    values[5] = reference[5] + 3.5
    with pytest.raises(AssertionError, match="fp8 embedding"):
        _check_embedding(values, reference, "fp8", "synthetic")


def test_fp32_gate_is_unchanged() -> None:
    """fp32 keeps its own tolerance: fp16-sized noise fails it."""
    values, reference = _synthetic(0, 3e-6)
    _check_embedding(values, reference, "fp32", "synthetic")
    values, reference = _synthetic(0, 8.5e-3)
    with pytest.raises(AssertionError, match="fp32 embedding"):
        _check_embedding(values, reference, "fp32", "synthetic")


def _engine_vectors(plan: bytes, cfg: dict, tokenizer, texts: list[str],
                    window: int) -> list[list[float]]:
    """Pooled [CLS] vectors of ``plan`` for ``texts``, padded the way the runtime pads them."""
    import tensorrt as trt
    import torch

    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    vectors = []
    for text in texts:
        ids = tokenizer(text)["input_ids"]
        input_ids = torch.full((window,), cfg["pad_token_id"], dtype=torch.int32,
                               device="cuda")
        input_ids[: len(ids)] = torch.tensor(ids, dtype=torch.int32)
        attention_mask = torch.zeros(window, dtype=torch.int32, device="cuda")
        attention_mask[: len(ids)] = 1
        hidden = torch.empty((window, cfg["hidden_size"]), dtype=torch.float32,
                             device="cuda")
        for name, tensor in (("input_ids", input_ids), ("attention_mask", attention_mask),
                             ("last_hidden_state", hidden)):
            context.set_tensor_address(name, tensor.data_ptr())
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        vectors.append(hidden[0].cpu().tolist())
    return vectors


def test_fp16_gate_rejects_ignored_ntk_scaling(fp16_case_name: str) -> None:
    """A correct fp16 engine passes the fp16 gate; one that ignores the NTK rotary scaling fails.

    Both engines are built from the case checkpoint and run in this process on
    the three case texts, padded to the window the way the runtime pads them.
    """
    import torch
    from transformers import AutoTokenizer

    from .. import config as config_module, model

    _, manifest, case = CASES[fp16_case_name]
    assert manifest["precision"] == "fp16"
    model_dir = _model_dir(manifest)
    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    window = int(manifest["max_sequence_length"])
    weights = model._open_checkpoint(model_dir)
    cfg = config_module.resolve(model_dir, weights)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    texts = [str(case[key]) for key in ("query", "document", "negative_document")]
    references = [_reference(model_dir, text) for text in texts]

    def run(build_cfg) -> list[list[float]]:
        plan = model.build_engine(weights, build_cfg, seq_len=window, precision="fp16")
        return _engine_vectors(plan, cfg, tokenizer, texts, window)

    worst = max(_check_embedding(values, reference, "fp16", text)[0]
                for values, reference, text in zip(run(cfg), references, texts))
    broken = run(dict(cfg, rope_scaling_factor=1.0))
    broken_worst = max(max(abs(a - b) for a, b in zip(values, reference))
                       for values, reference in zip(broken, references))
    with pytest.raises(AssertionError, match="fp16 embedding"):
        for values, reference, text in zip(broken, references, texts):
            _check_embedding(values, reference, "fp16", text)
    print(
        f"{FAMILY} fp16 gate: correct maxabs {worst:.3e}; NTK scaling ignored "
        f"maxabs {broken_worst:.3e}"
    )


def test_fp8_gate_rejects_scales_of_the_wrong_layers(fp8_case_name: str) -> None:
    """A correct FP8 engine passes the FP8 gate; one with another layer's activation scales fails.

    Both engines are built from the case checkpoint with the committed scale table (the
    broken one reads the table in reverse layer order) and run in this process on the
    three case texts.
    """
    import torch
    from transformers import AutoTokenizer

    from .. import config as config_module, model, quantization

    _, manifest, case = CASES[fp8_case_name]
    assert manifest["precision"] == "fp16" and manifest["quantization"] == "fp8"
    model_dir = _model_dir(manifest)
    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    window = int(manifest["max_sequence_length"])
    weights = model._open_checkpoint(model_dir)
    cfg = config_module.resolve(model_dir, weights)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    texts = [str(case[key]) for key in ("query", "document", "negative_document")]
    references = [_reference(model_dir, text) for text in texts]

    correct = quantization.calibrate_gte(weights, cfg, "fp8")
    names = quantization.projections(cfg)
    swapped = {
        name: quantization._LayerScales(
            correct.scales[names[-1 - index]].input_scale, correct.scales[name].weight)
        for index, name in enumerate(names)
    }

    def run(quant_ctx) -> list[list[float]]:
        plan = model.build_engine(weights, cfg, seq_len=window, precision="fp16",
                                  quant_ctx=quant_ctx)
        return _engine_vectors(plan, cfg, tokenizer, texts, window)

    worst = max(_check_embedding(values, reference, "fp8", text)[0]
                for values, reference, text in zip(run(correct), references, texts))
    broken = run(quantization.GteQuantContext(scales=swapped))
    broken_worst = max(max(abs(a - b) for a, b in zip(values, reference))
                       for values, reference in zip(broken, references))
    with pytest.raises(AssertionError, match="fp8 embedding"):
        for values, reference, text in zip(broken, references, texts):
            _check_embedding(values, reference, "fp8", text)
    print(
        f"{FAMILY} fp8 gate: correct maxabs {worst:.3e}; scales of the wrong layers "
        f"maxabs {broken_worst:.3e}"
    )
