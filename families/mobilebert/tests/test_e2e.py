# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime and official-reference E2E for MobileBERT.

Every case builds (once per manifest) the bundle of the pinned checkpoint, runs
the native CLI, and compares with the reference: ``MobileBertForPreTraining``
of transformers in float32 on the CPU, fed the ids of the checkpoint's own
``tokenizer.json`` (loaded with the ``tokenizers`` library). transformers
5.2.0's ``AutoTokenizer`` is not the tokenizer reference here: this repository
has no ``tokenizer_config.json``, and transformers then builds a BertTokenizer
with ``do_lower_case=False`` that ignores the file's lowercasing normaliser, so
"The" becomes ``[UNK]`` for this uncased model.

**The gates are the logit values**, checked against deliberate breaks of the
graph built from this checkpoint (``test_gate_rejects_known_breaks``):

* masked LM: the largest absolute logit difference over the selected rows;
* next sentence: the two logits are near 2.7e8 and differ by about 1e4, so the
  gate is on the relative error of each logit and on the error of their
  difference relative to that difference, and the label must agree. The
  published head scores ``next`` for every pair, the unrelated case included,
  exactly as the reference does: the case checks parity, not discrimination.

The tolerances and the measurements behind them are next to their constants.
The case text, the token ids the native tokenizer produces and the selected
positions are checked exactly.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "mobilebert"
TASKS = frozenset({"masked_text_to_token_scores", "text_pair_to_pretraining_relation_scores"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
MASK_ID = 103

# Measured on a B300 (TensorRT 11.4): a correct float32 engine is within 6.5e-05
# of the reference on every case of the manifest (37 rows of the Unicode case the
# widest); the weakest break of test_gate_rejects_known_breaks moves the masked
# rows by 1.1e+01. 1e-3 sits 15x above the first and four orders of magnitude
# below the second.
_LOGIT_TOLERANCE = 1e-3
# Next sentence, same engines. The logits are near 3e8, where one float32 step is
# 32: a correct engine is within 2.1e-07 of each logit, relative, and within one
# step of their difference (2.6e-03 of a 1.2e4 difference). The weakest breaks
# move the logits by 1.4e-02 relative and their difference by 2.8e-01 of itself.
# 1e-5 and 5e-2 leave room for a few float32 steps and stay well below both.
_RELATION_RTOL = 1e-5
_MARGIN_RTOL = 5e-2


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
        model_filters.update(item.strip() for item in str(raw).split(",") if item.strip())
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        model_filters.update(
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#"))
    testcase_filters = set()
    for raw in config.getoption("--e2e-testcase") or []:
        testcase_filters.update(item.strip() for item in str(raw).split(",") if item.strip())
    if not model_filters and not testcase_filters:
        return sorted(CASES), False
    selected = []
    for name, (_, manifest, _) in CASES.items():
        model_match = (not model_filters or FAMILY in model_filters or name in model_filters
                       or manifest["name"] in model_filters)
        if model_match and (not testcase_filters or name in testcase_filters):
            selected.append(name)
    return sorted(selected), True


def pytest_generate_tests(metafunc) -> None:
    for fixture in ("case_name", "manifest_name"):
        if fixture not in metafunc.fixturenames:
            continue
        names, enabled = _selected_cases(metafunc.config)
        if fixture == "manifest_name":
            names = sorted({CASES[name][1]["name"] for name in names})
        parameters = names if enabled else [
            pytest.param(name, marks=pytest.mark.skip(
                reason="direct E2E requires one of the three explicit E2E selectors"))
            for name in names]
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
        snapshot = snapshot_download(repo_id=manifest["hf_id"],
                                     revision=manifest.get("hf_revision"), local_files_only=True)
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
    return binary, runtime_root


_BUNDLES: dict[str, Path] = {}


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    """Builds each selected manifest's bundle once for the module."""

    def get(manifest: dict) -> Path:
        name = manifest["name"]
        if name not in _BUNDLES:
            bundle = tmp_path_factory.mktemp(name) / manifest["bundle"]
            build(BuildRequest(
                model_dir=_model_dir(manifest), output_path=bundle, family=FAMILY,
                task=manifest["task"], precision=manifest["precision"],
                max_sequence_length=manifest.get("max_sequence_length"),
                max_batch_size=int(manifest.get("max_batch_size", 1)),
                tensor_parallel_size=int(manifest["tensor_parallel_size"]),
                quantization=manifest.get("quantization")))
            _BUNDLES[name] = bundle
        return _BUNDLES[name]

    return get


def _run(command: list[str], runtime_root: Path, timeout: int = 600) -> str:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    completed = subprocess.run(command, check=False, capture_output=True, text=True, env=env,
                               timeout=timeout)
    assert completed.returncode == 0, (
        f"{command[0]} failed ({completed.returncode}): {completed.stderr[-2000:]}")
    return completed.stdout


def _cli_json(binary: Path, runtime_root: Path, bundle: Path, *arguments: str) -> dict:
    stdout = _run([str(binary), "encode", str(bundle), "--runtime-root", str(runtime_root),
                   *arguments], runtime_root)
    payloads = []
    for line in stdout.splitlines():
        start = line.find("{")
        if start >= 0:
            try:
                payloads.append(json.loads(line[start:]))
            except json.JSONDecodeError:
                pass
    assert payloads, f"native encode returned no JSON: {stdout[-1000:]}"
    return payloads[-1]


class Reference:
    """transformers float32 on the CPU, with the checkpoint's own tokenizer.json."""

    def __init__(self, model_dir: Path):
        import torch
        from tokenizers import Tokenizer
        from transformers import MobileBertForPreTraining

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.model = MobileBertForPreTraining.from_pretrained(
            str(model_dir), dtype=torch.float32).eval()

    def run(self, ids: list[int], segments: list[int] | None = None):
        import torch

        segments = segments or [0] * len(ids)
        with torch.no_grad():
            out = self.model(input_ids=torch.tensor([ids]),
                             token_type_ids=torch.tensor([segments]))
        return out.prediction_logits[0].tolist(), out.seq_relationship_logits[0].tolist()


_REFERENCES: dict[str, Reference] = {}


def _reference(model_dir: Path) -> Reference:
    if str(model_dir) not in _REFERENCES:
        _REFERENCES[str(model_dir)] = Reference(model_dir)
    return _REFERENCES[str(model_dir)]


def _fill_window(reference: Reference, sentence: str, window: int) -> str:
    """``[MASK]``, repeats of ``sentence`` and ``a`` fillers, ``[MASK]``: exactly ``window`` ids.

    The first row after [CLS] reads the zero left-hand neighbour of no padding,
    the last row before [SEP] the last learned position and the zero right-hand
    neighbour of the window's end.
    """
    def length(body: str) -> int:
        return len(reference.tokenizer.encode(f"[MASK] {body} [MASK]").ids)

    body = sentence
    while length(f"{body} {sentence}") <= window:
        body = f"{body} {sentence}"
    while length(f"{body} a") <= window:
        body = f"{body} a"
    text = f"[MASK] {body} [MASK]"
    assert length(body) == window, (length(body), window)
    return text


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norms = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norms


def _top(row: list[float], k: int = 5) -> list[int]:
    return sorted(range(len(row)), key=lambda index: -row[index])[:k]


def _check_masked(native: list[list[float]], reference: list[list[float]]) -> float:
    """Assert the masked-LM gate; return the largest absolute difference."""
    assert len(native) == len(reference) and native
    assert all(len(row) == len(want) for row, want in zip(native, reference))
    # max() keeps a finite difference over a NaN one, so non-finite values are refused first.
    assert all(math.isfinite(value) for rows in (native, reference) for row in rows
               for value in row), f"{FAMILY} logits are not all finite"
    worst = max(max(abs(a - b) for a, b in zip(row, want))
                for row, want in zip(native, reference))
    # Every logit within the tolerance also bounds the ranking: two tokens can
    # change places only if their reference logits are within twice the
    # tolerance. Comparing top-k lists exactly would fail on such near-ties.
    assert worst <= _LOGIT_TOLERANCE, f"{FAMILY} logits differ from the reference by {worst}"
    return worst


def _check_relation(native: list[float], reference: list[float]) -> tuple[float, float]:
    """Assert the next-sentence gate; return (relative logit error, margin error)."""
    assert len(native) == len(reference) == 2
    assert all(math.isfinite(value) for value in (*native, *reference)), (
        f"{FAMILY} relation logits are not all finite")
    relative = max(abs(a - b) / abs(b) for a, b in zip(native, reference))
    margin = reference[0] - reference[1]
    margin_error = abs((native[0] - native[1]) - margin) / abs(margin)
    assert relative <= _RELATION_RTOL, f"{FAMILY} relation logits differ by {relative} relative"
    assert margin_error <= _MARGIN_RTOL, f"{FAMILY} relation margin differs by {margin_error}"
    assert (native[0] >= native[1]) == (reference[0] >= reference[1]), "relation label differs"
    return relative, margin_error


def _masked_rows(payload: dict) -> list[list[float]]:
    rows, columns = (int(value) for value in payload["shape"])
    values = [float(value) for value in payload["logits"]]
    assert len(values) == rows * columns
    return [values[row * columns:(row + 1) * columns] for row in range(rows)]


def test_official_checkpoint_e2e(case_name: str, bundles) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime()
    bundle = bundles(manifest)
    reference = _reference(model_dir)

    if "first" in case:
        first, second = str(case["first"]), str(case["second"])
        payload = _cli_json(binary, runtime_root, bundle, "--task",
                            "text_pair_to_pretraining_relation_scores", "--text", first,
                            "--text-pair", second)
        assert payload["labels"] == ["next", "not_next"]
        assert payload["score_kind"] == "logit"
        encoding = reference.tokenizer.encode(first, second)
        _, want = reference.run(encoding.ids, encoding.type_ids)
        native = [float(value) for value in payload["scores"]]
        relative, margin = _check_relation(native, want)
        assert payload["top_class"] == (0 if want[0] >= want[1] else 1), (
            f"{FAMILY} relation top_class differs from the reference")
        print(f"{case_name}: native {native}, reference {want}; relative {relative:.2e}, "
              f"margin error {margin:.2e}, label {payload['labels'][payload['top_class']]}")
        return

    if "token_ids" in case:
        ids = [int(value) for value in case["token_ids"]]
        payload = _cli_json(binary, runtime_root, bundle, "--task",
                            "masked_text_to_token_scores", "--token-ids", json.dumps(ids))
        framed_here = False
    else:
        text = (str(case["text"]) if "text" in case else
                _fill_window(reference, str(case["fill_window"]),
                             int(manifest["max_sequence_length"])))
        ids = reference.tokenizer.encode(text).ids
        payload = _cli_json(binary, runtime_root, bundle, "--task",
                            "masked_text_to_token_scores", "--text", text)
        framed_here = True

    assert payload["score_kind"] == "logit"
    assert payload["axes"] == ["selected_position", "vocabulary_id"]
    assert payload["vocabulary_id"] == FAMILY
    masked = [index for index, value in enumerate(ids) if value == MASK_ID]
    selected = masked or list(range(len(ids)))
    positions = payload["positions"]
    # The native tokenizer is checked through the ids it reports for every row.
    assert [int(token["token_index"]) for token in positions] == selected
    assert [int(token["token_id"]) for token in positions] == [ids[i] for i in selected]
    for token in positions:
        index = int(token["token_index"])
        inserted = framed_here and index in (0, len(ids) - 1)
        assert int(token["input_index"]) == (-1 if inserted else 0)
        assert token["byte_offsets"] is None

    native = _masked_rows(payload)
    rows, _ = reference.run(ids)
    want = [rows[index] for index in selected]
    worst = _check_masked(native, want)
    top = [_top(row, 1)[0] for row in native[:4]]
    print(f"{case_name}: {len(ids)} ids, {len(selected)} rows x {len(native[0])} vocabulary, "
          f"logit max |d| {worst:.3e}, cosine "
          f"{_cosine(sum(native, []), sum(want, [])):.9f}, top-1 "
          f"{[reference.tokenizer.id_to_token(i) for i in top]}")


def _float32(values) -> list[float]:
    """The float32 each printed score stands for; the printers differ in digits only."""
    import numpy as np

    return np.asarray(values, dtype=np.float64).astype(np.float32).tolist()


def test_sdk_consumers_match_the_cli(manifest_name: str, bundles) -> None:
    """The public C and C++ SDK serve both Tasks of one bundle, equal to the CLI."""
    manifest = next(m for _, m, _ in CASES.values() if m["name"] == manifest_name)
    binary, runtime_root = _runtime()
    native_build = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"),
                                  "TRTMC_NATIVE_BUILD_DIR")
    bundle = bundles(manifest)
    masked_case = next(case for case in manifest["testcases"] if case.get("premerge"))
    pair_case = next(case for case in manifest["testcases"] if "first" in case)
    arguments = [str(masked_case["text"]), str(pair_case["first"]), str(pair_case["second"])]
    masked = _cli_json(binary, runtime_root, bundle, "--task", "masked_text_to_token_scores",
                       "--text", arguments[0])
    relation = _cli_json(binary, runtime_root, bundle, "--task",
                         "text_pair_to_pretraining_relation_scores", "--text", arguments[1],
                         "--text-pair", arguments[2])
    for language in ("c", "cpp"):
        consumer = native_build / f"test_{FAMILY}_sdk_{language}"
        assert consumer.is_file(), f"missing {consumer}"
        output = json.loads(_run([str(consumer), str(bundle), str(runtime_root), *arguments],
                                 runtime_root).strip().splitlines()[-1])
        assert output["masked"]["shape"] == masked["shape"]
        assert output["masked"]["vocabulary_id"] == masked["vocabulary_id"]
        assert [(p["token_id"], p["token_index"], p["input_index"])
                for p in output["masked"]["positions"]] == [
            (p["token_id"], p["token_index"], p["input_index"]) for p in masked["positions"]]
        assert _float32(output["masked"]["logits"]) == _float32(masked["logits"])
        assert output["relation"]["labels"] == relation["labels"]
        assert output["relation"]["vocabulary_id"] == relation["vocabulary_id"]
        assert _float32(output["relation"]["scores"]) == _float32(relation["scores"])
        print(f"sdk {language}: identical to the CLI on both Tasks")


def test_gate_rejects_known_breaks(manifest_name: str, monkeypatch) -> None:
    """Engines with plausible graph breaks fail the gates that a correct engine passes.

    Each break loads every tensor and runs. They are built in this process from
    the case checkpoint with a 64-row window and run on the premerge text and the
    next-sentence pair, padded the way the runtime pads.
    """
    import numpy as np
    import tensorrt as trt
    import torch

    from .. import config as config_module, graph, model

    manifest = next(m for _, m, _ in CASES.values() if m["name"] == manifest_name)
    model_dir = _model_dir(manifest)
    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    reference = _reference(model_dir)
    masked_case = next(case for case in manifest["testcases"] if case.get("premerge"))
    pair_case = next(case for case in manifest["testcases"] if "first" in case)
    single = reference.tokenizer.encode(str(masked_case["text"])).ids
    pair = reference.tokenizer.encode(str(pair_case["first"]), str(pair_case["second"]))
    masked = [index for index, value in enumerate(single) if value == MASK_ID]
    want_rows, _ = reference.run(single)
    _, want_relation = reference.run(pair.ids, pair.type_ids)
    weights = model._open_torch_checkpoint(model_dir)
    cfg = config_module.resolve(model_dir, weights)
    window = 64

    def run(engine, ids, segments):
        context = engine.create_execution_context()
        tensors = {name: torch.zeros(window, dtype=torch.int32, device="cuda")
                   for name in ("input_ids", "token_type_ids", "attention_mask")}
        tensors["input_ids"][: len(ids)] = torch.tensor(ids, dtype=torch.int32)
        tensors["token_type_ids"][: len(ids)] = torch.tensor(segments, dtype=torch.int32)
        tensors["attention_mask"][: len(ids)] = 1
        tensors["prediction_logits"] = torch.empty((window, cfg["vocab_size"]),
                                                   dtype=torch.float32, device="cuda")
        tensors["seq_relationship_logits"] = torch.empty((2,), dtype=torch.float32,
                                                         device="cuda")
        for name, tensor in tensors.items():
            context.set_tensor_address(name, tensor.data_ptr())
        assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return (tensors["prediction_logits"].cpu().tolist(),
                tensors["seq_relationship_logits"].cpu().tolist())

    def gates(name: str) -> tuple[bool, bool]:
        plan = model.build_engine(weights, cfg, seq_len=window)
        engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
        rows, _ = run(engine, single, [0] * len(single))
        _, relation = run(engine, pair.ids, pair.type_ids)
        native_rows = [rows[i] for i in masked]
        reference_rows = [want_rows[i] for i in masked]
        worst = max(max(abs(a - b) for a, b in zip(row, want))
                    for row, want in zip(native_rows, reference_rows))
        relative = max(abs(a - b) / abs(b) for a, b in zip(relation, want_relation))
        margin = want_relation[0] - want_relation[1]
        margin_error = abs((relation[0] - relation[1]) - margin) / abs(margin)
        try:
            _check_masked(native_rows, reference_rows)
            masked_ok = True
        except AssertionError:
            masked_ok = False
        try:
            _check_relation(relation, want_relation)
            relation_ok = True
        except AssertionError:
            relation_ok = False
        print(f"{name}: masked max |d| {worst:.3e} ({'pass' if masked_ok else 'fail'}); "
              f"relation relative {relative:.3e}, margin error {margin_error:.3e} "
              f"({'pass' if relation_ok else 'fail'})")
        return masked_ok, relation_ok

    assert gates("correct engine") == (True, True)

    def swapped_trigram(net, tokens, seq_len, width):
        # [previous, this, next] instead of the reference's [next, this, previous].
        zero = net.constant(np.zeros((1, width), np.float32))
        following = net.concat([net.rows(tokens, 1, seq_len - 1), zero], axis=0)
        preceding = net.concat([zero, net.rows(tokens, 0, seq_len - 1)], axis=0)
        return net.concat([preceding, tokens, following], axis=1)

    with monkeypatch.context() as patch:
        patch.setattr(graph, "_trigram", swapped_trigram)
        assert gates("trigram neighbours swapped") == (False, False)

    head_weight = graph.head_weight
    with monkeypatch.context() as patch:
        # The extra dense matrix read before the decoder instead of after it.
        patch.setattr(graph, "head_weight", lambda weights_, cfg_: np.roll(
            head_weight(weights_, cfg_), -cfg_["embedding_size"], axis=1))
        assert gates("masked-LM projection halves swapped")[0] is False

    embeddings = graph._embeddings

    def no_segments(net, weights_, cfg_, input_ids, token_type_ids, keep_column, seq_len):
        # Every row reads segment 0's embedding.
        zero = net.elementwise(token_type_ids, token_type_ids, trt.ElementWiseOperation.SUB)
        return embeddings(net, weights_, cfg_, input_ids, zero, keep_column, seq_len)

    with monkeypatch.context() as patch:
        patch.setattr(graph, "_embeddings", no_segments)
        assert gates("segment embeddings ignored")[1] is False
