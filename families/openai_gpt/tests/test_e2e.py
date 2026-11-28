# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned checkpoint -> bundle -> native ``text_continuation`` -> transformers proof.

Oracles (one per case):

* ``token_exact``: the float32 bundle generates exactly the token IDs of greedy decoding
  with the float32 transformers model (TF32 off on both sides; one forward pass of the whole
  sequence per token, see ``_greedy_reference``), and its text is the reference tokenizer's
  decoding of those IDs.
* ``sampling_support``: seeded sampling repeats for the same seed, and every sampled
  token is one that transformers' sampling allows at its step: its temperature,
  top-k and top-p processors applied to the float32 logits of the same prefix.

Inputs: ``text`` goes through the native tokenizer, and the reference tokenizer's IDs of
the same text, sent as token IDs, must continue identically; ``token_ids`` repeats the
case's ``filler`` text and cuts its IDs so that the prefix and the continuation fill
every row of the bundle's KV cache. The direct C and C++ SDK callers must return the
command line's result, and the runtime must refuse a request it cannot serve.
"""

from __future__ import annotations

import gc
import json
import os
import re
import struct
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence

_TEST_DIR = Path(__file__).resolve().parent
_FAMILY = _TEST_DIR.parent.name
_TASK = "text_continuation"
_ORACLES = ("token_exact", "sampling_support")
_INPUTS = ("text", "token_ids")
_SAMPLING = ("temperature", "top_k", "top_p", "seed")


def _load_cases() -> dict[str, tuple[dict, dict]]:
    cases: dict[str, tuple[dict, dict]] = {}
    for path in sorted((_TEST_DIR / "manifests").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == _FAMILY and manifest["task"] == _TASK, path
        assert manifest["precision"] == "fp32", path
        assert isinstance(manifest["max_sequence_length"], int), path
        assert manifest["tensor_parallel_size"] == 1, path
        assert manifest["bundle"] == f"{manifest['name']}.bundle", path
        assert isinstance(manifest["hf_id"], str) and isinstance(manifest["hf_revision"], str), path
        for case in manifest["testcases"]:
            assert case["oracle"] in _ORACLES and case["input"] in _INPUTS, case["name"]
            # A text case has a prompt, a token-ID case a filler, and never both.
            source = "prompt" if case["input"] == "text" else "filler"
            assert source in case and {"prompt", "filler"} - {source} - set(case), case["name"]
            sampling = case["oracle"] == "sampling_support"
            assert all((name in case) == sampling for name in _SAMPLING), case["name"]
            assert case["name"] not in cases, case["name"]
            cases[case["name"]] = (manifest, case)
    assert cases, f"{_FAMILY} has no E2E cases"
    return cases


_CASES = _load_cases()


def _csv_values(values: list[str]) -> set[str]:
    return {item.strip() for value in values for item in str(value).split(",") if item.strip()}


def _selection(config) -> tuple[set[str], set[str]]:
    """Names selected by --e2e-model or --e2e-models-file, and testcases by --e2e-testcase."""
    models = _csv_values(config.getoption("--e2e-model", default=[]) or [])
    models_file = config.getoption("--e2e-models-file", default=None)
    if models_file:
        path = Path(models_file)
        assert path.is_file(), f"E2E models file does not exist: {path}"
        models |= {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    return models, _csv_values(config.getoption("--e2e-testcase", default=[]) or [])


def _collected_cases(items) -> set[str]:
    """The ``case_name`` parameter of every ``test_*e2e`` item collected in this session."""
    cases = set()
    for item in items:
        parameters = getattr(getattr(item, "callspec", None), "params", {})
        if re.fullmatch(r"test_.*e2e", getattr(item, "originalname", "") or "") and (
            "case_name" in parameters
        ):
            cases.add(str(parameters["case_name"]))
    return cases


def _require_selected(case_name: str, manifest: dict, config, collected: set[str]) -> None:
    models, testcases = _selection(config)
    if os.environ.get("TRTMC_E2E") != "1" and not (models or testcases):
        pytest.skip("real family E2E requires TRTMC_E2E=1 or an explicit E2E selection")
    if models and not ({_FAMILY, manifest["name"], case_name} & models):
        pytest.skip(f"{case_name} was not selected")
    if testcases and case_name not in testcases:
        # Testcases narrow the selection. A selection that shares no case with what
        # --e2e-model selects in this family, or, without --e2e-model, with the E2E cases
        # collected in this session, is wrong and must not pass by running none.
        selected = {
            name for name, (other, _) in _CASES.items() if {_FAMILY, other["name"], name} & models
        }
        if models and not selected & testcases:
            pytest.fail(
                f"--e2e-testcase {sorted(testcases)} matches none of the selected {_FAMILY} "
                f"cases: {sorted(selected)}"
            )
        if not models and not testcases & collected:
            pytest.fail(
                f"--e2e-testcase {sorted(testcases)} matches no E2E case collected in this session"
            )
        pytest.skip(f"{case_name} was not selected")


@pytest.fixture(scope="session")
def _torch():
    import torch

    assert torch.cuda.is_available(), "selected E2E requires CUDA"
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    # The float32 reference must be true float32 (NGC images default matmuls to TF32).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield torch
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def _environment() -> tuple[Path, Path, Path]:
    binary = Path(os.environ.get("TRTMC_BINARY", ""))
    runtime_root = Path(os.environ.get("TRTMC_RUNTIME_ROOT", ""))
    build_dir = Path(os.environ.get("TRTMC_NATIVE_BUILD_DIR", ""))
    assert binary.is_file() and os.access(binary, os.X_OK), "selected E2E requires TRTMC_BINARY"
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file(), "requires TRTMC_RUNTIME_ROOT"
    assert (runtime_root / f"libtrtmc_model_{_FAMILY}.so").is_file(), runtime_root
    assert build_dir.is_dir(), "the SDK callers require TRTMC_NATIVE_BUILD_DIR"
    return binary, runtime_root, build_dir


def _checkpoint(manifest: dict) -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    # The pre-merge runs offline with a cache holding just the snapshot's files; a lookup
    # that may reach the Hub (for example to list the revision's files) fails there.
    hf_id, revision = manifest["hf_id"], manifest["hf_revision"]
    try:
        path = Path(snapshot_download(repo_id=hf_id, revision=revision, local_files_only=True))
    except LocalEntryNotFoundError as error:
        raise AssertionError(
            f"{hf_id} at {revision} is not in the Hugging Face cache; download it first "
            f"(hf download {hf_id} --revision {revision})"
        ) from error
    assert (path / "config.json").is_file(), path
    return path


@pytest.fixture(scope="session")
def _bundles(tmp_path_factory):
    """One bundle per manifest and session: the cases of a manifest share its build inputs."""
    return {"root": tmp_path_factory.mktemp("bundles")}


def _bundle(bundles: dict, manifest: dict, model_dir: Path) -> Path:
    if manifest["name"] in bundles:
        return bundles[manifest["name"]]
    bundle = bundles["root"] / manifest["bundle"]
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=_FAMILY,
            task=_TASK,
            precision=manifest["precision"],
            max_sequence_length=manifest["max_sequence_length"],
        )
    )
    assert bundle.is_file() and bundle.stat().st_size > 0, bundle
    bundles[manifest["name"]] = bundle
    return bundle


@pytest.fixture(scope="session")
def _reference_models():
    """transformers float32 models by checkpoint, loaded once per session."""
    models: dict[Path, object] = {}
    yield models
    models.clear()
    gc.collect()


def _reference_model(models: dict, model_dir: Path, torch):
    if model_dir not in models:
        from transformers import AutoModelForCausalLM

        models[model_dir] = (
            AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.float32)
            .to("cuda")
            .eval()
        )
    return models[model_dir]


def _prefix(case: dict, manifest: dict, model_dir: Path) -> tuple[list[str], list[int]]:
    """Command-line input arguments and the reference token IDs of the same prefix."""
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    if case["input"] == "text":
        ids = tokenizer.encode(case["prompt"]).ids
        # transformers rebuilds this tokenizer (README, "Tokenizer"); on this ordinary text
        # it agrees with tokenizer.json.
        auto = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        assert auto(case["prompt"])["input_ids"] == ids
        return ["--prompt", case["prompt"]], ids
    # Prefix plus continuation fill the cache: the last token is never fed back.
    count = manifest["max_sequence_length"] - case["max_new_tokens"] + 1
    ids: list[int] = []
    while len(ids) < count:
        ids += tokenizer.encode(case["filler"]).ids
    ids = ids[:count]
    return ["--token-ids", json.dumps(ids)], ids


def _cli(binary: Path, runtime_root: Path, bundle: Path, inputs: list[str], controls: dict):
    command = [str(binary), "run", str(bundle), "--runtime-root", str(runtime_root), *inputs]
    for name, value in controls.items():
        command += [f"--{name.replace('_', '-')}", str(value)]
    record_evidence(
        "commands", {"argv": [arg if len(arg) < 400 else arg[:400] + "..." for arg in command]}
    )
    return subprocess.run(command, capture_output=True, text=True, timeout=1800)


def _run_native(binary: Path, runtime_root: Path, bundle: Path, inputs: list[str], controls: dict):
    completed = _cli(binary, runtime_root, bundle, inputs, controls)
    assert completed.returncode == 0, completed.stderr[-4000:]
    payload = json.loads(completed.stdout)
    record_evidence("native", {**payload, "stderr": completed.stderr[-4000:]})
    return payload


def _controls(case: dict, **changes) -> dict:
    controls = {"max_new_tokens": case["max_new_tokens"]}
    controls.update({name: case[name] for name in _SAMPLING if name in case})
    controls.update(changes)
    return controls


def _sdk(build_dir: Path, runtime_root: Path, bundle: Path, mode: str, source: str, controls: dict):
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), environment.get("LD_LIBRARY_PATH", "")) if value
    )
    outputs = []
    for language in ("c", "cpp"):
        executable = build_dir / f"test_{_FAMILY}_sdk_{language}"
        assert executable.is_file(), f"missing family SDK caller: {executable}"
        completed = subprocess.run(
            [str(executable), str(bundle), str(runtime_root), mode, source]
            + [f"{name}={value}" for name, value in controls.items()],
            capture_output=True,
            text=True,
            timeout=1800,
            env=environment,
        )
        assert completed.returncode == 0, completed.stderr[-4000:]
        output = json.loads(completed.stdout)
        record_evidence(f"sdk_{language}_{mode}", output)
        outputs.append(output)
    return outputs


def _greedy_reference(model, prompt_ids: list[int], max_new_tokens: int, torch) -> list[int]:
    """Greedy decoding with one forward pass of the whole sequence per new token.

    OpenAIGPTLMHeadModel has no KV cache. transformers 5.2.0 ``generate()`` nevertheless
    passes, from its second step on, the cached-decoding ``position_ids`` of the newest token
    alone together with the whole sequence, and the model then adds that one position
    embedding to every token (README, "Validation"). The forward pass of the whole sequence
    with the default positions is the model's own definition.
    """
    sequence = list(prompt_ids)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = model(torch.tensor([sequence], device="cuda")).logits[0, -1]
            # torch.argmax takes the first maximum, as the native greedy sampler does.
            sequence.append(int(torch.argmax(logits)))
    return sequence[len(prompt_ids):]


def _sampling_support(model, prompt_ids: list[int], native_ids: list[int], case: dict, torch):
    """Per sampled token: whether transformers' sampling processors keep it at its step."""
    from transformers.generation.logits_process import (
        LogitsProcessorList,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    processors = LogitsProcessorList(
        [
            TemperatureLogitsWarper(float(case["temperature"])),
            TopKLogitsWarper(int(case["top_k"])),
            TopPLogitsWarper(float(case["top_p"])),
        ]
    )
    sequence = torch.tensor([prompt_ids + native_ids], device="cuda")
    with torch.inference_mode():
        logits = model(sequence).logits[0, len(prompt_ids) - 1 : -1].float()
    allowed = []
    for step, token in enumerate(native_ids):
        prefix = sequence[:, : len(prompt_ids) + step]
        scores = processors(prefix, logits[step : step + 1].clone())
        allowed.append(bool(torch.isfinite(scores[0, token]).item()))
    return allowed


def _artifact(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def _check(name: str, label: str, actual, expected, scope: str = "independent_reference") -> dict:
    return {
        "name": name,
        "label": label,
        "scope": scope,
        "actual": actual,
        "operator": "==",
        "expected": expected,
        "passed": actual == expected,
    }


def _refused(binary: Path, runtime_root: Path, bundle: Path, inputs: list[str], controls: dict):
    completed = _cli(binary, runtime_root, bundle, inputs, controls)
    record_evidence("contract", {"refused": completed.returncode != 0, "stderr": completed.stderr[-400:]})
    return completed.returncode != 0


@pytest.mark.parametrize("case_name", sorted(_CASES))
def test_e2e(case_name: str, request, tmp_path: Path, _bundles, _reference_models) -> None:
    manifest, case = _CASES[case_name]
    _require_selected(case_name, manifest, request.config, _collected_cases(request.session.items))
    # CUDA is required only by a selected case, so it is set up after the selection.
    torch = request.getfixturevalue("_torch")
    record_evidence("inputs", {"manifest": manifest, "case": case})
    binary, runtime_root, build_dir = _environment()
    model_dir = _checkpoint(manifest)
    record_evidence(
        "checkpoint",
        {"model_dir": str(model_dir), "hf_id": manifest["hf_id"], "hf_revision": manifest["hf_revision"]},
    )
    inputs, prompt_ids = _prefix(case, manifest, model_dir)
    record_evidence(
        "inputs",
        {"prompt": case.get("prompt", case.get("filler")), "prompt_token_ids": prompt_ids},
    )
    with evidence_stage("build"):
        bundle = _bundle(_bundles, manifest, model_dir)
    controls = _controls(case)
    with evidence_stage("native"):
        native = _run_native(binary, runtime_root, bundle, inputs, controls)
        rerun = _run_native(binary, runtime_root, bundle, inputs, controls)
        # The reference tokenizer's IDs of the prefix must continue exactly as the text.
        framed = _run_native(
            binary, runtime_root, bundle, ["--token-ids", json.dumps(prompt_ids)], controls
        )
        token_file = tmp_path / "prefix.i32"
        token_file.write_bytes(struct.pack(f"<{len(prompt_ids)}i", *prompt_ids))
        sdk = _sdk(build_dir, runtime_root, bundle, "tokens", str(token_file), controls)
        if case["input"] == "text":
            sdk += _sdk(build_dir, runtime_root, bundle, "text", case["prompt"], controls)
    native_ids = native["token_ids"]

    with evidence_stage("compare"):
        # The checkpoint declares no EOS token, so every request generates all its tokens.
        assert len(native_ids) == case["max_new_tokens"], len(native_ids)
        assert rerun["token_ids"] == native_ids and rerun["text"] == native["text"]
        assert framed["token_ids"] == native_ids and framed["text"] == native["text"]
        for output in sdk:
            assert output["token_ids"] == native_ids and output["text"] == native["text"], output
        # Requests the bundle cannot serve are refused, not truncated.
        capacity = manifest["max_sequence_length"]
        refusals = {
            "a prefix one token too long for the cache": _refused(
                binary, runtime_root, bundle,
                ["--token-ids", json.dumps([prompt_ids[0]] * (capacity - case["max_new_tokens"] + 2))],
                controls,
            ),
            "a token ID outside the vocabulary": _refused(
                binary, runtime_root, bundle, ["--token-ids", json.dumps([40478])], controls
            ),
            "an empty prefix": _refused(binary, runtime_root, bundle, ["--prompt", ""], controls),
            "zero new tokens": _refused(
                binary, runtime_root, bundle, inputs, _controls(case, max_new_tokens=0)
            ),
        }
        record_evidence("diagnostics", {"refused": refusals})
        assert all(refusals.values()), refusals

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    model = _reference_model(_reference_models, model_dir, torch)
    native_file = _artifact(tmp_path / "native.json", {"token_ids": native_ids, "text": native["text"]})
    if case["oracle"] == "token_exact":
        with evidence_stage("reference"):
            reference_ids = _greedy_reference(model, prompt_ids, case["max_new_tokens"], torch)
        reference_text = tokenizer.decode(reference_ids, skip_special_tokens=True)
        record_evidence("reference", {"reference_ids": reference_ids, "reference_text": reference_text})
        matching = sum(a == b for a, b in zip(native_ids, reference_ids))
        checks = [
            _check("token_count", "Generated tokens", len(native_ids), len(reference_ids)),
            _check("matching_tokens", "Token IDs equal to greedy transformers float32", matching, len(reference_ids)),
            _check("text_match", "Text equal to the reference decoding (1 = equal)", int(native["text"] == reference_text), 1),
        ]
        reference_file = _artifact(
            tmp_path / "reference.json", {"token_ids": reference_ids, "text": reference_text}
        )
    else:
        with evidence_stage("reference"):
            allowed = _sampling_support(model, prompt_ids, native_ids, case, torch)
        other_seed = _run_native(
            binary, runtime_root, bundle, inputs, _controls(case, seed=case["seed"] + 1)
        )
        record_evidence("reference", {"allowed_by_transformers": allowed, "other_seed_ids": other_seed["token_ids"]})
        checks = [
            _check("token_count", "Generated tokens", len(native_ids), case["max_new_tokens"], "contract"),
            _check("allowed_fraction", "Sampled tokens that transformers' processors keep", sum(allowed) / len(allowed), 1.0),
            _check("same_seed_repeats", "Same seed, same tokens", rerun["token_ids"] == native_ids, True, "contract"),
            # A sampler that ignored the seed (or always decoded greedily) would repeat itself.
            _check("other_seed_differs", "Another seed, other tokens", other_seed["token_ids"] != native_ids, True, "contract"),
            _check("text_match", "Text equal to the reference decoding (1 = equal)", int(native["text"] == tokenizer.decode(native_ids, skip_special_tokens=True)), 1),
        ]
        reference_file = _artifact(
            tmp_path / "reference.json", {"allowed_by_transformers": allowed}
        )
    record_evidence(
        "reference_comparison",
        {
            "label": case_name,
            "scope": "independent_reference",
            "enforced": True,
            "native": native_file,
            "reference": reference_file,
            "checks": checks,
        },
    )
    with evidence_stage("compare"):
        failed = [check for check in checks if not check["passed"]]
        assert not failed, failed


def test_cases_cover_the_contract() -> None:
    oracles = {case["oracle"] for _, case in _CASES.values()}
    inputs = {case["input"] for _, case in _CASES.values()}
    assert oracles == set(_ORACLES) and inputs == set(_INPUTS)
    # Every case runs before merge, and one fills every row of the KV cache.
    assert all(case.get("premerge") is True for _, case in _CASES.values())
    assert any(case["input"] == "token_ids" for _, case in _CASES.values())
