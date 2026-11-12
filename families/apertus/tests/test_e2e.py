# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned checkpoint -> bundle -> native ``text_continuation`` -> official reference proof.

Oracles (one per case):

* ``token_exact``: an FP32 bundle generates exactly the token IDs of greedy
  ``transformers`` generation in FP32 (TF32 disabled on both sides).
* ``teacher_forced_near_tie``: a BF16 bundle may leave the FP32 path only at
  near-ties. The FP32 reference is run on the prompt plus the native tokens;
  every native token must be within ``near_tie_logit_tolerance`` of that
  step's maximum FP32 logit. The tolerance is calibrated on the official
  ``transformers`` BF16 generation scored the same way.
* ``sampling_contract``: seeded sampling is deterministic, stays within the
  vocabulary and the token budget, and decodes to the reported text.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence

_TEST_DIR = Path(__file__).resolve().parent
_FAMILY = _TEST_DIR.parent.name
_TASK = "text_continuation"
_ORACLES = ("token_exact", "teacher_forced_near_tie", "sampling_contract")
_INPUTS = ("chat_token_ids", "text", "random_token_ids")
# The official transformers BF16 generation of this checkpoint, scored by its FP32 run,
# chose tokens up to 0.347 logit below the FP32 maximum (15 prompts, 788 tokens; 12 steps
# off the FP32 argmax). The tolerance is that bound rounded up to one significant digit.
_NEAR_TIE_LOGIT_TOLERANCE = 0.4


def _load_cases() -> dict[str, tuple[dict, dict]]:
    cases: dict[str, tuple[dict, dict]] = {}
    for path in sorted((_TEST_DIR / "manifests").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == _FAMILY and manifest["task"] == _TASK, path
        assert manifest["precision"] in {"bf16", "fp32"}, path
        assert isinstance(manifest["max_sequence_length"], int), path
        # One GPU, no tensor parallelism; the bundle file name is the manifest's.
        assert manifest["tensor_parallel_size"] == 1, path
        assert manifest["bundle"] == f"{manifest['name']}.bundle", path
        assert ("hf_id" in manifest) != ("generated_checkpoint" in manifest), path
        for case in manifest["testcases"]:
            assert case["oracle"] in _ORACLES and case["input"] in _INPUTS, case["name"]
            assert case["name"] not in cases, case["name"]
            cases[case["name"]] = (manifest, case)
    assert cases, f"{_FAMILY} has no E2E cases"
    return cases


_CASES = _load_cases()


def _selected(config) -> set[str]:
    values = list(config.getoption("--e2e-model", default=[]) or [])
    values += list(config.getoption("--e2e-testcase", default=[]) or [])
    selected = {item.strip() for value in values for item in str(value).split(",") if item.strip()}
    models_file = config.getoption("--e2e-models-file", default=None)
    if models_file:
        lines = Path(models_file).read_text(encoding="utf-8").splitlines()
        selected |= {line.strip() for line in lines if line.strip() and not line.startswith("#")}
    return selected


def _require_selected(case_name: str, manifest: dict, config) -> None:
    selected = _selected(config)
    if os.environ.get("TRTMC_E2E") != "1" and not selected:
        pytest.skip("real family E2E requires TRTMC_E2E=1 or an explicit E2E selection")
    if selected and not ({_FAMILY, manifest["name"], case_name} & selected):
        pytest.skip(f"{case_name} was not selected")


def _environment():
    binary = Path(os.environ.get("TRTMC_BINARY", ""))
    runtime_root = Path(os.environ.get("TRTMC_RUNTIME_ROOT", ""))
    assert binary.is_file() and os.access(binary, os.X_OK), "selected E2E requires TRTMC_BINARY"
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file(), "requires TRTMC_RUNTIME_ROOT"
    assert (runtime_root / f"libtrtmc_model_{_FAMILY}.so").is_file(), runtime_root
    import torch

    assert torch.cuda.is_available(), "selected E2E requires CUDA"
    # The FP32 reference must be true FP32 (NGC images default matmuls to TF32).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.get_float32_matmul_precision() == "highest"
    return binary, runtime_root, torch


def _tiny_tokenizer() -> dict:
    """A byte-level tokenizer.json with the checkpoint's split expression and no merges."""
    units, extra = [], 256
    for value in range(256):
        printable = 33 <= value <= 126 or 161 <= value <= 172 or value >= 174
        units.append(chr(value if printable else extra))
        extra += 0 if printable else 1
    specials = ["<unk>", "<s>", "</s>", "<pad>"]
    vocab = {token: index for index, token in enumerate(specials + units)}
    added = [
        {
            "id": index,
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for index, token in enumerate(specials)
    ]
    split = (
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|"
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|"
        r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": split},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
            ],
            "pair": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 1}},
            ],
            "special_tokens": {"<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}},
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": True,
            "vocab": vocab,
            "merges": [],
        },
    }


def _generated_checkpoint(spec: dict, directory: Path, torch) -> Path:
    """Seeded random ``ApertusForCausalLM`` saved like a released checkpoint."""
    from transformers import ApertusConfig, ApertusForCausalLM

    torch.manual_seed(int(spec["seed"]))
    config = ApertusConfig(**spec["config"])
    config.architectures = ["ApertusForCausalLM"]
    model = ApertusForCausalLM(config).to(torch.float32).eval()
    with torch.no_grad():
        for layer in model.model.layers:
            # Learned xIELU parameters of the released checkpoint span this range.
            layer.mlp.act_fn.alpha_p.uniform_(-1.0, 4.0)
            layer.mlp.act_fn.alpha_n.uniform_(-1.0, 4.0)
    model.save_pretrained(directory, safe_serialization=True)
    (directory / "tokenizer.json").write_text(json.dumps(_tiny_tokenizer()), encoding="utf-8")
    del model
    return directory


def _checkpoint(manifest: dict, tmp_path: Path, torch) -> Path:
    if "generated_checkpoint" in manifest:
        return _generated_checkpoint(
            manifest["generated_checkpoint"], tmp_path / "checkpoint", torch
        )
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            repo_id=manifest["hf_id"], revision=manifest["hf_revision"], local_files_only=True
        )
    )
    assert (path / "config.json").is_file(), path
    return path


def _prompt_text(case: dict) -> str:
    filler = case.get("filler")
    if filler is None:
        return case["prompt"]
    paragraphs = [filler["sentence"].format(index=index) for index in range(filler["count"])]
    paragraphs.insert(filler["insert_at"], filler["needle"])
    return "\n".join(paragraphs) + "\n" + case["prompt"]


def _inputs(case: dict, model_dir: Path) -> tuple[list[str], list[int]]:
    """CLI input arguments and the token IDs the reference consumes for the same prefix."""
    if case["input"] == "random_token_ids":
        import numpy as np

        spec = case["random_tokens"]
        rng = np.random.default_rng(int(spec["seed"]))
        ids = [1, *rng.integers(4, spec["vocab_size"], size=spec["count"] - 1).tolist()]
        return ["--token-ids", json.dumps(ids)], ids
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    text = _prompt_text(case)
    if case["input"] == "text":
        return ["--prompt", text], tokenizer(text)["input_ids"]
    # The default system message carries today's date; a case pins it for reproducibility.
    messages = [{"role": "user", "content": text}]
    if "system_prompt" in case:
        messages.insert(0, {"role": "system", "content": case["system_prompt"]})
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids
    return ["--token-ids", json.dumps(list(ids))], list(ids)


# One bundle per manifest and session: the cases of a manifest share its build inputs.
_BUNDLES: dict[str, Path] = {}


def _bundle(manifest: dict, model_dir: Path, tmp_path_factory) -> Path:
    bundle = _BUNDLES.get(manifest["name"])
    if bundle is not None and "generated_checkpoint" not in manifest:
        return bundle
    bundle = tmp_path_factory.mktemp(manifest["name"]) / manifest["bundle"]
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
    _BUNDLES[manifest["name"]] = bundle
    return bundle


def _run_native(
    binary: Path, runtime_root: Path, bundle: Path, inputs: list[str], case: dict
) -> dict:
    command = [
        str(binary),
        "run",
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        *inputs,
        "--max-new-tokens",
        str(case["max_new_tokens"]),
    ]
    for field in ("temperature", "top_k", "top_p", "seed"):
        if field in case:
            command += [f"--{field.replace('_', '-')}", str(case[field])]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=1800)
    record_evidence(
        "commands", {"argv": [arg if len(arg) < 400 else arg[:400] + "..." for arg in command]}
    )
    payload = json.loads(completed.stdout)
    record_evidence("native", {**payload, "stderr": completed.stderr[-4000:]})
    return payload


def _reference(
    model_dir: Path,
    prompt_ids: list[int],
    native_ids: list[int],
    case: dict,
    torch,
    precisions: tuple[str, ...],
) -> dict:
    """Greedy official generation and FP32 teacher-forced scores of a token path."""
    from transformers import AutoModelForCausalLM

    result: dict = {}
    for precision in precisions:
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[precision]
        model = (
            AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=dtype)
            .to("cuda")
            .eval()
        )
        prompt = torch.tensor([prompt_ids], device="cuda")
        with torch.inference_mode():
            generated = model.generate(
                prompt,
                attention_mask=torch.ones_like(prompt),
                max_new_tokens=int(case["max_new_tokens"]),
                do_sample=False,
            )
            result[f"{precision}_ids"] = generated[0, len(prompt_ids) :].tolist()
            if precision == "fp32":
                for label, path in (
                    ("native", native_ids),
                    ("official_bf16", result.get("bf16_ids")),
                ):
                    if path:
                        result[f"{label}_fp32_gaps"] = _gaps(model, prompt_ids, path, torch)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return result


def _gaps(model, prompt_ids: list[int], path: list[int], torch) -> list[float]:
    """Per step: max FP32 logit minus the FP32 logit of the token the path chose."""
    sequence = torch.tensor([prompt_ids + path], device="cuda")
    # The last len(path) + 1 rows start at the prompt's last position.
    logits = model(sequence, logits_to_keep=len(path) + 1).logits[0, : len(path)].float()
    chosen = logits.gather(1, torch.tensor(path, device="cuda")[:, None])[:, 0]
    return (logits.max(dim=1).values - chosen).tolist()


def _thresholds(case_name: str) -> dict:
    path = _TEST_DIR / "thresholds" / f"{case_name}.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]


@pytest.mark.parametrize("case_name", sorted(_CASES))
def test_e2e(case_name: str, request, tmp_path: Path, tmp_path_factory) -> None:
    manifest, case = _CASES[case_name]
    _require_selected(case_name, manifest, request.config)
    record_evidence("inputs", {"manifest": manifest, "case": case})
    binary, runtime_root, torch = _environment()
    model_dir = _checkpoint(manifest, tmp_path, torch)
    record_evidence(
        "checkpoint",
        {
            "model_dir": str(model_dir),
            "hf_id": manifest.get("hf_id"),
            "hf_revision": manifest.get("hf_revision"),
        },
    )
    cli_inputs, prompt_ids = _inputs(case, model_dir)
    record_evidence("inputs", {"prompt_tokens": len(prompt_ids)})
    # Long-context cases must reach past the checkpoint's original RoPE window.
    assert len(prompt_ids) >= int(case.get("min_prompt_tokens", 1)), len(prompt_ids)
    assert len(prompt_ids) + case["max_new_tokens"] <= manifest["max_sequence_length"]
    with evidence_stage("build"):
        bundle = _bundle(manifest, model_dir, tmp_path_factory)
    record_evidence("inputs", {"bundle": str(bundle), "bundle_bytes": bundle.stat().st_size})
    with evidence_stage("native"):
        payload = _run_native(binary, runtime_root, bundle, cli_inputs, case)
        reruns = [
            _run_native(binary, runtime_root, bundle, cli_inputs, case)
            for _ in range(int(case.get("determinism_reruns", 0)))
        ]
        # The family tokenizer must frame a text prefix exactly as the official tokenizer:
        # the same prefix as reference token IDs must give the same continuation.
        framed = None
        if case["input"] == "text":
            framed = _run_native(
                binary, runtime_root, bundle, ["--token-ids", json.dumps(prompt_ids)], case
            )
    native_ids = payload["token_ids"]
    with evidence_stage("compare"):
        assert 1 <= len(native_ids) <= case["max_new_tokens"]
        assert all(rerun["token_ids"] == native_ids for rerun in reruns), "native rerun differs"
        if framed is not None:
            assert framed["token_ids"] == native_ids and framed["text"] == payload["text"]
    if case["oracle"] == "sampling_contract":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        vocab_size = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))[
            "vocab_size"
        ]
        record_evidence("reference", {"mode": "contract_only", "oracle": case["oracle"]})
        with evidence_stage("compare"):
            assert all(0 <= token < vocab_size for token in native_ids)
            assert payload["text"] == tokenizer.decode(native_ids, skip_special_tokens=True)
        return
    precisions = ("bf16", "fp32") if case["oracle"] == "teacher_forced_near_tie" else ("fp32",)
    with evidence_stage("reference"):
        reference = _reference(model_dir, prompt_ids, native_ids, case, torch, precisions)
    record_evidence("reference", reference)
    with evidence_stage("compare"):
        if "generated_checkpoint" not in manifest:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            assert payload["text"] == tokenizer.decode(native_ids, skip_special_tokens=True)
        if case["oracle"] == "token_exact":
            assert native_ids == reference["fp32_ids"]
            return
        tolerance = float(
            _thresholds(case_name).get("near_tie_logit_tolerance", _NEAR_TIE_LOGIT_TOLERANCE)
        )
        record_evidence("thresholds", {"near_tie_logit_tolerance": tolerance})
        assert max(reference["native_fp32_gaps"]) <= tolerance, reference["native_fp32_gaps"]


def test_cases_are_well_formed_and_cover_the_contract() -> None:
    oracles = {case["oracle"] for _, case in _CASES.values()}
    assert oracles == set(_ORACLES)
    assert any(case.get("premerge") for _, case in _CASES.values())
    for manifest, case in _CASES.values():
        if case["oracle"] == "token_exact":
            assert manifest["precision"] == "fp32", case["name"]
        if case["oracle"] == "teacher_forced_near_tie":
            assert manifest["precision"] == "bf16", case["name"]
        if case["input"] == "random_token_ids":
            assert "generated_checkpoint" in manifest, case["name"]
