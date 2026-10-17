# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare pinned model-card behavior through the public native Task SDK.

This opt-in harness is intentionally separate from CPU unit tests. It requires
a real checkpoint, a built bundle, a compiled family SDK client, and CUDA.
Sampling is validated separately with shared draws, never by equating seeds.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import subprocess
from pathlib import Path


if __package__:
    from .lfm25_semantic_contract import POLICY, evaluate, reference_parts
else:
    from lfm25_semantic_contract import POLICY, evaluate, reference_parts


CHECKPOINT = "LiquidAI/LFM2.5-2.6B"
REVISION = "654f9463ce32b05d0429d76fe1f580b27d4c1ac0"
FAMILY = "lfm2"
CONTEXT = 131072
QUICKSTART_TOKENS = 512
LANGUAGES = {
    "en": "Answer in English: what is the capital of France?",
    "ar": "أجب بالعربية: ما عاصمة فرنسا؟",
    "zh": "请用中文回答：法国的首都是哪里？",
    "fr": "Réponds en français : quelle est la capitale de la France ?",
    "de": "Antworte auf Deutsch: Was ist die Hauptstadt von Frankreich?",
    "it": "Rispondi in italiano: qual è la capitale della Francia?",
    "ja": "日本語で答えてください。フランスの首都はどこですか？",
    "ko": "한국어로 답하세요. 프랑스의 수도는 어디인가요?",
    "pt": "Responda em português: qual é a capital da França?",
    "es": "Responde en español: ¿cuál es la capital de Francia?",
    "vi": "Trả lời bằng tiếng Việt: thủ đô của Pháp là gì?",
    "th": "ตอบเป็นภาษาไทย: เมืองหลวงของฝรั่งเศสคืออะไร?",
    "id": "Jawab dalam bahasa Indonesia: apa ibu kota Prancis?",
    "hi": "हिंदी में उत्तर दें: फ्रांस की राजधानी क्या है?",
    "ru": "Ответь по-русски: какая столица Франции?",
    "pl": "Odpowiedz po polsku: jaka jest stolica Francji?",
}


def cases():
    yield "single-token-continuation", {
        "prompt": "Hello", "config": {"max_new_tokens": 32, "temperature": 0.0,
                                         "top_k": 1, "repetition_penalty": 1.0},
    }
    yield "raw-capital-completion", {
        "prompt": "The capital of France is",
        "config": {"max_new_tokens": 64, "temperature": 0.0, "top_k": 1, "repetition_penalty": 1.0},
    }
    yield "model-card-quickstart", {
        "messages": [{"role": "user", "content": "What is C. elegans?"}],
        "config": {"max_new_tokens": QUICKSTART_TOKENS, "temperature": 0.0,
                   "top_k": 1, "repetition_penalty": 1.0},
    }
    yield "model-card-sampling", {
        "messages": [{"role": "user", "content": "What is C. elegans?"}],
        "config": {"max_new_tokens": QUICKSTART_TOKENS, "temperature": 0.1,
                   "top_k": 50, "top_p": 1.0, "min_p": 0.0,
                   "repetition_penalty": 1.1, "seed": 42},
    }
    yield "sampled-answer-completion", {
        "messages": [{"role": "user", "content": "What is C. elegans?"}],
        "config": {"max_new_tokens": 2048, "temperature": 0.1,
                   "top_k": 50, "top_p": 1.0, "min_p": 0.0,
                   "repetition_penalty": 1.1, "seed": 42},
    }
    fixtures = json.loads((Path(__file__).parent / "fixtures/lfm25-conversation.json").read_text())
    wanted = {"single-user", "system-tools", "json-tool-request", "tool-result",
              "strip-prior-reasoning", "preserve-prior-reasoning", "parallel-tools"}
    for case in fixtures["cases"]:
        if case["name"] not in wanted:
            continue
        request = {key: case[key] for key in ("messages", "tools") if key in case}
        request["config"] = {"max_new_tokens": 1024, "temperature": 0.0, "top_k": 1,
                             "repetition_penalty": 1.0}
        if "preserve_thinking" in case:
            request["config"]["preserve_thinking"] = case["preserve_thinking"]
        yield case["name"], request
    yield "positive-temperature-tool", {
        "messages": [{"role": "system", "content":
                      "Use the provided tool to satisfy the request."},
                     {"role": "user", "content": "Set the cabin temperature to 22 degrees Celsius."}],
        "tools": [{"name": "set_temperature", "description": "Set the cabin temperature in Celsius.",
                   "parameters": {"type": "object", "properties": {"temperature": {"type": "number"}},
                                  "required": ["temperature"]}}],
        "config": {"max_new_tokens": 1024, "temperature": 0.0, "top_k": 1, "repetition_penalty": 1.0},
    }
    # Keep the original weak request as reference-behavior evidence, alongside
    # this explicit positive call and the pinned card's candidate dialogue.
    card_tool = {
        "name": "get_candidate_status",
        "description": "Retrieves the current status of a candidate in the recruitment process",
        "parameters": {"type": "object", "properties": {"candidate_id": {
            "type": "string", "description": "Unique identifier for the candidate"}},
            "required": ["candidate_id"]},
    }
    card_user = {"role": "user", "content": "What is the current status of candidate ID 12345?"}
    generation = {"max_new_tokens": 1024, "temperature": 0.0, "top_k": 1,
                  "repetition_penalty": 1.0}
    yield "model-card-tool-call", {"messages": [card_user], "tools": [card_tool],
                                   "config": generation}
    yield "model-card-tool-json", {
        "messages": [{"role": "system", "content":
                      "Return tool calls as JSON objects with name and arguments."}, card_user],
        "tools": [card_tool], "config": generation,
    }
    card_result = [{"candidate_id": "12345", "status": "Interview Scheduled",
                    "position": "Clinical Research Associate", "date": "2023-11-20"}]
    yield "model-card-tool-result", {
        "messages": [card_user, {"role": "assistant",
            "content": "Checking the current status of candidate ID 12345.",
            "tool_calls": [{"type": "function", "function": {
                "name": "get_candidate_status", "arguments": {"candidate_id": "12345"}}}]},
            {"role": "tool", "content": json.dumps(card_result)}],
        "tools": [card_tool], "config": generation,
    }
    for language, prompt in LANGUAGES.items():
        yield "language-" + language, {"messages": [{"role": "user", "content": prompt}],
              "config": {"max_new_tokens": 1024, "temperature": 0.0, "top_k": 1,
                         "repetition_penalty": 1.0}}


def full_context_request(tokenizer):
    """Full token-capacity boundary; generated content is diagnostic only."""
    prefix = tokenizer("The access code is 7319.\n", add_special_tokens=True).input_ids
    filler = tokenizer("This line is background information.\n", add_special_tokens=False).input_ids
    suffix = tokenizer("\nThe access code is", add_special_tokens=False).input_ids
    reserve = 8
    count = CONTEXT - reserve - len(prefix) - len(suffix)
    assert count > 0 and filler
    ids = prefix + (filler * ((count + len(filler) - 1) // len(filler)))[:count] + suffix
    assert len(ids) + reserve == CONTEXT
    return {"token_ids": ids, "config": {"max_new_tokens": reserve, "temperature": 0.0,
                                         "top_k": 1, "repetition_penalty": 1.0}}


def prepare_transport_request(request):
    """Supply SDK IDs for an ID-free HF history without changing its content.

    ID-free tool results consume preceding calls in their declared order.
    Explicit result IDs keep their association; duplicate call/result IDs and
    unpaired ID-free results are rejected rather than assigned an invented link.
    """
    output = copy.deepcopy(request)
    messages = output.get("messages", [])
    reserved, explicit_calls = set(), set()

    def identifier(item, field):
        value = item.get(field)
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError("tool transport IDs must be strings")
        return value

    for message in messages:
        for call in message.get("tool_calls", []):
            call_id = identifier(call, "id")
            if call_id:
                if call_id in explicit_calls:
                    raise ValueError("duplicate explicit tool-call ID")
                explicit_calls.add(call_id)
                reserved.add(call_id)
        if message.get("role") == "tool":
            result_id = identifier(message, "tool_call_id")
            if result_id:
                reserved.add(result_id)

    pending, returned = [], set()
    next_id = 0
    for message in messages:
        for call in message.get("tool_calls", []):
            call_id = identifier(call, "id")
            if not call_id:
                while "call_" + str(next_id) in reserved:
                    next_id += 1
                call_id = "call_" + str(next_id)
                next_id += 1
                reserved.add(call_id)
                call["id"] = call_id
            pending.append(call_id)
        if message.get("role") == "tool":
            result_id = identifier(message, "tool_call_id")
            if not result_id:
                if not pending:
                    raise ValueError("ID-free tool result has no preceding unmatched call")
                result_id = pending[0]
                message["tool_call_id"] = result_id
            if result_id in returned:
                raise ValueError("duplicate tool-result ID")
            returned.add(result_id)
            if result_id in pending:
                pending.remove(result_id)
    return output


def artifact_metadata(path):
    """Report file names and sizes; content identity is collected externally."""
    path = Path(path)
    return {"name": path.name, "bytes": path.stat().st_size}


def capture_native_metadata(bundle, runtime_root, client, reference_precision, inspector=None):
    """Inspect through the generic SDK Bundle reader without loading engines."""
    inspector = Path(inspector or client)
    inspected = subprocess.run([str(inspector), "--inspect-bundle", str(bundle)],
                               capture_output=True, text=True, check=True)
    metadata = json.loads(inspected.stdout)
    if metadata.get("family") != FAMILY:
        raise ValueError("native bundle belongs to a different model family")
    runtime_root = Path(runtime_root)
    libraries = ["libtrtmc_model_" + FAMILY + ".so", "libtrtmc_backend_trt.so",
                 "libtrtmc_c.so", "libtrtmc_core.so", "libtrtmc_runtime.so"]
    identity = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_precision": reference_precision,
        "bundle_metadata": metadata,
        "precision_matches_reference": metadata.get("precision") == reference_precision,
        "bundle": artifact_metadata(bundle),
        "client": artifact_metadata(client),
        "inspector": artifact_metadata(inspector),
        "runtime_libraries": {name: artifact_metadata(runtime_root / name) for name in libraries},
        "metadata_scope": "Names, sizes, and declared bundle configuration only; content identity and linker environment overrides are not inspected.",
    }
    return identity


def capture_reference_execution_identity(model, torch):
    """Describe the loaded reference path without selecting a different backend."""
    import sys

    module_name = type(model).__module__
    module = sys.modules.get(module_name)
    source = getattr(module, "__file__", None)
    config = getattr(model, "config", None)

    def loaded_member(owner, name):
        # Reading an existing namespace must not invoke a lazy module loader.
        try:
            return vars(owner).get(name)
        except TypeError:
            return None

    def callable_identity(value):
        if not callable(value):
            return None
        return {"module": getattr(value, "__module__", None),
                "qualname": getattr(value, "__qualname__", None)}

    available_callables = {
        "model_forward": getattr(type(model), "forward", None),
        "eager_attention_forward": loaded_member(module, "eager_attention_forward"),
        "sdpa_attention_forward": loaded_member(
            sys.modules.get("transformers.integrations.sdpa_attention"), "sdpa_attention_forward"),
        "causal_conv1d_fn": loaded_member(module, "causal_conv1d_fn"),
        "causal_conv1d_update": loaded_member(module, "causal_conv1d_update"),
    }

    def setting(path):
        value = torch
        for component in path.split("."):
            value = getattr(value, component, None)
            if value is None:
                return None
        return value() if callable(value) else value

    flags = (
        "get_float32_matmul_precision",
        "are_deterministic_algorithms_enabled",
        "backends.cuda.matmul.allow_tf32",
        "backends.cuda.matmul.allow_fp16_reduced_precision_reduction",
        "backends.cuda.matmul.allow_bf16_reduced_precision_reduction",
        "backends.cuda.matmul.allow_fp16_accumulation",
        "backends.cuda.flash_sdp_enabled",
        "backends.cuda.mem_efficient_sdp_enabled",
        "backends.cuda.math_sdp_enabled",
        "backends.cuda.cudnn_sdp_enabled",
        "backends.cuda.fp16_bf16_reduction_math_sdp_allowed",
    )
    identity = {
        "model_class": module_name + "." + type(model).__qualname__,
        "invocation_observed": False,
        "selection_scope": "Configured names and already-loaded callable/API availability only; no dispatch resolution or invocation observation.",
        "available_callables": {name: callable_identity(value)
                                for name, value in available_callables.items()},
        "attention_implementation": getattr(config, "_attn_implementation", None),
        "experts_implementation": getattr(config, "_experts_implementation", None),
        "model_dtype": str(model.dtype) if hasattr(model, "dtype") else None,
        "model_module": {
            "name": module_name,
            "source": artifact_metadata(source) if source and Path(source).is_file() else None,
        },
        "causal_conv": {
            "fast_path_available": getattr(module, "is_fast_path_available", None),
            "forward_function_available": callable(getattr(module, "causal_conv1d_fn", None)),
            "update_function_available": callable(getattr(module, "causal_conv1d_update", None)),
        },
        "torch_backend_flags": {name: setting(name) for name in flags},
        "backend_flag_scope": "Enabled settings and optional fast-path availability; these do not prove which kernel executed. Unavailable settings are null.",
        "source_scope": "Model-module filename and size only; this observation does not verify source content or loaded bytecode.",
    }
    return identity


def run_native_case(client, bundle, runtime_root, output_dir, name, request):
    """Persist the native process evidence before propagating any failure."""
    request_path = output_dir / (name + ".request.json")
    request_path.write_text(json.dumps(prepare_transport_request(request), ensure_ascii=False))
    command = [str(client), str(bundle), str(runtime_root), str(request_path)]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:
        (output_dir / (name + ".native-process.json")).write_text(json.dumps({
            "returncode": None, "stdout": "", "stderr": str(error),
        }, indent=2))
        raise
    (output_dir / (name + ".native-process.json")).write_text(json.dumps({
        "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr,
    }, indent=2, ensure_ascii=False))
    completed.check_returncode()
    payload = json.loads(completed.stdout)
    (output_dir / (name + ".native.json")).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def reference_case(model, tokenizer, request, name, torch):
    """Run one original reference case; callers retain any exception as failure."""
    if "token_ids" in request:
        ids = request["token_ids"]
    elif "prompt" in request:
        ids = tokenizer(request["prompt"], add_special_tokens=True).input_ids
        if name == "single-token-continuation":
            assert len(ids) == 1, "short-prefix regression must contain one input token"
    else:
        options = {"tools": request.get("tools", []),
                   "preserve_thinking": request["config"].get("preserve_thinking", False)}
        rendered = tokenizer.apply_chat_template(request["messages"], tokenize=False,
                                                  add_generation_prompt=True, **options)
        ids = tokenizer(rendered, add_special_tokens=False).input_ids
    input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
    sampling = name in {"model-card-sampling", "sampled-answer-completion"}
    generation = {"max_new_tokens": request["config"]["max_new_tokens"],
                  "do_sample": sampling, "repetition_penalty": 1.0}
    if sampling:
        torch.manual_seed(request["config"]["seed"])
        generation.update({key: request["config"][key] for key in
                           ("temperature", "top_k", "top_p", "min_p", "repetition_penalty")})
    with torch.inference_mode():
        generated = model.generate(input_ids=input_ids, **generation)
    return len(ids), generated[0, len(ids):].tolist()


def assert_contract(name, native, reference, tokenizer, eos, request, input_count):
    return evaluate(name, native, reference_parts(tokenizer, reference, request),
                    reference, eos, request, input_count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-precision", choices=("bf16", "fp16"), default="bf16",
                        help="Original reference dtype; supply a native bundle built in the same precision")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--inspector", type=Path, help="SDK client with --inspect-bundle; defaults to --client")
    parser.add_argument("--metadata-only", "--identity-only", dest="metadata_only", action="store_true", help="Inspect CPU artifact metadata without inference")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--full-context", action="store_true")
    parser.add_argument("--list", action="store_true", help="List selected cases without loading weights")
    args = parser.parse_args()
    selected = list(cases())
    if args.case:
        names = {name for name, _ in selected}
        if args.full_context:
            names.add("full-context")
        unknown = set(args.case) - names
        if unknown:
            parser.error("unknown cases: " + ", ".join(sorted(unknown)))
        selected = [(name, value) for name, value in selected if name in args.case]
    if args.list:
        names = [name for name, _ in selected]
        if args.full_context and (not args.case or "full-context" in args.case):
            names.append("full-context")
        print("\n".join(names))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "native-metadata.json").exists() or (
            not args.metadata_only and any(args.output.iterdir())):
        parser.error("use a new empty output directory to preserve earlier strict/semantic receipts")
    native_metadata = capture_native_metadata(args.bundle, args.runtime_root, args.client,
                                               args.reference_precision, args.inspector)
    native_metadata["capture_timing"] = "metadata_only" if args.metadata_only else "before_validation"
    (args.output / "native-metadata.json").write_text(json.dumps(native_metadata, indent=2))
    if not native_metadata["precision_matches_reference"]:
        raise ValueError("native runtime precision does not match the requested reference precision")
    if args.metadata_only:
        print("Native artifact metadata captured; no inference was run.")
        return

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    expected_template = json.loads((Path(__file__).parent / "fixtures/lfm25-conversation.json").read_text())
    assert (args.model_dir / "chat_template.jinja").read_bytes() == expected_template["template"].encode("utf-8"), "checkpoint template content differs from the pinned fixture"
    if args.full_context and (not args.case or "full-context" in args.case):
        selected.append(("full-context", full_context_request(tokenizer)))
    args.output.mkdir(parents=True, exist_ok=True)
    reference_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.reference_precision]
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
                                                dtype=reference_dtype).eval().to("cuda")
    reference_execution = capture_reference_execution_identity(model, torch)
    (args.output / "reference-execution.json").write_text(json.dumps(reference_execution, indent=2))
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    receipts = []
    failures = []

    def save_receipts():
        (args.output / "semantic-parity.json").write_text(json.dumps({
            "acceptance_policy": POLICY,
            "token_identity_scope": "Exact IDs are diagnostics, not a semantic acceptance gate.",
            "checkpoint": CHECKPOINT, "revision": REVISION,
            "reference_precision": args.reference_precision,
            "native_metadata": native_metadata,
            "reference_execution": reference_execution,
            "transformers": transformers.__version__, "torch": torch.__version__, "cases": receipts,
            "checkpoint_identity_boundary": "Exact template content checked; caller must supply pinned checkpoint weights.",
        }, indent=2))

    for name, request in selected:
        request_path = args.output / (name + ".reference-request.json")
        request_path.write_text(json.dumps(request, ensure_ascii=False))
        try:
            input_count, reference = reference_case(model, tokenizer, request, name, torch)
        except Exception as error:
            row = {"case": name, "semantic_contract": "failed", "failure_stage": "reference",
                   "reference_error_type": type(error).__name__, "reference_error": str(error)}
            receipts.append(row)
            (args.output / (name + ".reference-error.json")).write_text(json.dumps(row, indent=2))
            failures.append(name + ": reference execution failed")
            save_receipts()
            print(name, "FAIL: reference execution:", str(error), flush=True)
            continue
        row = {"case": name, "generation_config": request["config"],
               "reference_do_sample": name in {"model-card-sampling", "sampled-answer-completion"},
               "input_tokens": input_count, "reference_ids": reference,
               "reference_reached_eos": bool(reference and reference[-1] in eos),
               "reference_text": tokenizer.decode(reference, skip_special_tokens=False),
               "native_returncode": None, "semantic_contract": "not_run"}
        receipts.append(row)
        (args.output / (name + ".reference.json")).write_text(json.dumps(row, indent=2, ensure_ascii=False))
        save_receipts()
        print(name, "reference complete", flush=True)
        torch.cuda.empty_cache()
        try:
            native = run_native_case(args.client, args.bundle, args.runtime_root, args.output, name, request)
            row.update(native_returncode=0, native_ids=native["token_ids"],
                       exact_ids=native["token_ids"] == reference)
        except Exception as error:
            process_path = args.output / (name + ".native-process.json")
            process = json.loads(process_path.read_text()) if process_path.is_file() else {}
            row.update(native_returncode=process.get("returncode"), semantic_contract="failed",
                       failure_stage="native", native_error_type=type(error).__name__,
                       native_error=str(error))
            failures.append(name + ": native process failed")
            save_receipts()
            print(name, "FAIL: native process", flush=True)
            continue
        save_receipts()
        try:
            row["semantic_checks"] = assert_contract(name, native, reference, tokenizer, eos,
                                                       request, input_count)
        except Exception as error:
            row.update(semantic_contract="failed", semantic_error=str(error))
            failures.append(name + ": " + str(error))
            save_receipts()
            print(name, "FAIL:", str(error), flush=True)
            continue
        row["semantic_contract"] = "passed"
        save_receipts()
        print(name, "PASS", flush=True)

    if failures:
        raise AssertionError("Model-card contract failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
