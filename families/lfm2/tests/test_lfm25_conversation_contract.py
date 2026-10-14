# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import subprocess
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).resolve().parent
ROOT = TEST_DIR.parents[2]
FIXTURES = json.loads((TEST_DIR / "fixtures/lfm25-conversation.json").read_text())


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    binary = tmp_path_factory.mktemp("conversation") / "probe"
    subprocess.run([shutil.which("c++"), "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT), "-I", str(ROOT / "core/runtime/include"),
                    str(TEST_DIR / "cpp/lfm25_conversation_probe.cpp"),
                    str(TEST_DIR.parent / "runtime/conversation.cpp"), "-o", str(binary)],
                   check=True, capture_output=True, text=True)

    def invoke(record):
        completed = subprocess.run([str(binary)], input=json.dumps(record) + "\n",
                                   capture_output=True, text=True, check=True)
        return json.loads(completed.stdout)
    return invoke


@pytest.mark.parametrize("case", FIXTURES["cases"], ids=lambda case: case["name"])
def test_native_message_template_is_exact_hf(probe, case):
    assert probe(case) == {"rendered": case["rendered"]}


@pytest.mark.parametrize("body,expected", [
    ('[set_temperature(temperature=22)]', [("set_temperature", {"temperature": 22})]),
    ('[{"name":"set_temperature","arguments":{"temperature":22}}]', [("set_temperature", {"temperature": 22})]),
    ('[{"function":{"name":"set_temperature","arguments":"{\\"temperature\\":22}"}}]', [("set_temperature", {"temperature": 22})]),
    ("[record(text='it\\'s ok', enabled=True, items=[1, False, None], data={'name': '日本語'})]",
     [("record", {"text": "it's ok", "enabled": True, "items": [1, False, None], "data": {"name": "日本語"}})]),
    ('[first(), second(value=-1.25)]', [("first", {}), ("second", {"value": -1.25})]),
    (r"[record(text='\u2603\U0001F600\x41')]", [("record", {"text": "☃😀A"})]),
    (r"[record(text='\ud83d\ude00')]", [("record", {"text": "😀"})]),
])
def test_parse_pythonic_and_json_tool_calls(probe, body, expected):
    output = probe({"decoded": '<think>Use tools.</think><|tool_call_start|>' + body + '<|tool_call_end|>'})
    assert output["parts"][0] == {"type": "reasoning", "text": "Use tools."}
    calls = output["parts"][1:]
    assert [(call["name"], json.loads(call["arguments_json"])) for call in calls] == expected
    assert all(call["state"] == 1 for call in calls)
    assert all(call["call_id"] for call in calls)
    assert len({call["call_id"] for call in calls}) == len(calls)


@pytest.mark.parametrize("body,state", [
    ('<|tool_call_start|>[set_temperature(temperature=', 2),
    ('<|tool_call_start|>[bad(x=1, x=2)]<|tool_call_end|>', 3),
    ('<|tool_call_start|>[bad(x=__import__("os"))]<|tool_call_end|>', 3),
    ('<|tool_call_start|>[bad(x=)]<|tool_call_end|>', 3),
    ('<|tool_call_start|>[bad(x.y=1)]<|tool_call_end|>', 3),
    ('<|tool_call_start|>[.bad(x=1)]<|tool_call_end|>', 3),
    ('<|tool_call_start|>[bad(x="first\nsecond")]<|tool_call_end|>', 3),
    (r'<|tool_call_start|>[bad(x="\ud800")]<|tool_call_end|>', 3),
    ('<|tool_call_start|>[]<|tool_call_end|>', 3),
])
def test_preserve_invalid_tool_output_without_executing(probe, body, state):
    call = probe({"decoded": body})["parts"][0]
    assert call["state"] == state
    assert call["arguments_json"] in body


def test_dense_generation_prefix_and_text_parts(probe):
    result = probe({"decoded": "Plan.</think>Done.", "starts_in_reasoning": True})
    assert result["parts"] == [{"type": "reasoning", "text": "Plan."}, {"type": "text", "text": "Done."}]


def test_unmarked_json_tool_response_requires_declared_tools(probe):
    text = '{"name":"set_temperature","arguments":{"temperature":22}}'
    assert probe({"decoded": text})["parts"] == [{"type": "text", "text": text}]
    call = probe({"decoded": text, "allow_json_tools": True})["parts"][0]
    assert call["state"] == 1 and call["name"] == "set_temperature"
    assert json.loads(call["arguments_json"]) == {"temperature": 22}


def test_transport_ids_avoid_tool_call_and_result_history(probe):
    request = {
        "decoded": "<|tool_call_start|>[first(), second(value=22)]<|tool_call_end|>",
        "used_call_ids": ["call_0", "call_2"],
        "used_result_ids": ["call_1", "call_4"],
    }
    parts = probe(request)["parts"]
    assert [part["call_id"] for part in parts] == ["call_3", "call_5"]
    assert [part["name"] for part in parts] == ["first", "second"]
    assert [json.loads(part["arguments_json"]) for part in parts] == [{}, {"value": 22}]
    assert parts == probe(request)["parts"], "transport IDs must be deterministic"


def test_transport_ids_reserve_later_model_ids(probe):
    request = {"decoded": '<|tool_call_start|>[first()]<|tool_call_end|>'
               '<|tool_call_start|>[{"id":"call_0","name":"second","arguments":{}}]<|tool_call_end|>'}
    assert [part["call_id"] for part in probe(request)["parts"]] == ["call_1", "call_0"]


def test_transport_ids_replace_colliding_or_duplicate_model_ids(probe):
    request = {
        "decoded": '<|tool_call_start|>' + json.dumps([
            {"id": "old", "name": "first", "arguments": {}},
            {"id": "duplicate", "name": "second", "arguments": {}},
            {"id": "duplicate", "name": "third", "arguments": {}},
            {"id": "unique", "name": "fourth", "arguments": {}},
        ]) + '<|tool_call_end|>',
        "used_result_ids": ["old", "call_0"],
    }
    parts = probe(request)["parts"]
    assert [part["call_id"] for part in parts] == ["call_1", "call_2", "call_3", "unique"]
    assert all(part["state"] == 1 for part in parts)


def test_incomplete_transport_does_not_invent_semantic_fields(probe):
    part = probe({"decoded": "<|tool_call_start|>[first(value="})["parts"][0]
    assert part["state"] == 2
    assert part["call_id"] == "" and part["name"] == ""
    assert part["arguments_json"] == "[first(value="


def test_native_failure_receipt_is_saved_before_raising(tmp_path):
    from families.lfm2.tests.lfm25_contract_validation import run_native_case

    client = tmp_path / "failing-client"
    client.write_text("#!/usr/bin/env python3\nimport sys\nprint('native stdout')\n"
                      "print('native diagnostic', file=sys.stderr)\nsys.exit(7)\n")
    client.chmod(0o755)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        run_native_case(client, "unused.bundle", tmp_path, tmp_path, "failure", {"messages": []})
    assert caught.value.returncode == 7
    assert json.loads((tmp_path / "failure.native-process.json").read_text()) == {
        "returncode": 7, "stdout": "native stdout\n", "stderr": "native diagnostic\n",
    }


@pytest.mark.parametrize("case", FIXTURES["cases"], ids=lambda case: case["name"])
def test_input_transport_ids_leave_pinned_template_unchanged(probe, case):
    from families.lfm2.tests.lfm25_contract_validation import prepare_transport_request

    adapted = prepare_transport_request(case)
    assert probe(adapted) == {"rendered": case["rendered"]}
    assert adapted == prepare_transport_request(adapted)


def test_input_transport_ids_preserve_explicit_ids_and_order_results():
    from families.lfm2.tests.lfm25_contract_validation import prepare_transport_request

    def call(name):
        return {"function": {"name": name, "arguments": {}}}
    request = {"messages": [
        {"role": "assistant", "tool_calls": [call("first"), {"id": "call_0", **call("second")}]},
        {"role": "tool", "tool_call_id": "call_0", "content": "second result"},
        {"role": "tool", "content": "first result"},
    ]}
    output = prepare_transport_request(request)
    assert [c["id"] for c in output["messages"][0]["tool_calls"]] == ["call_1", "call_0"]
    assert output["messages"][1]["tool_call_id"] == "call_0"
    assert output["messages"][2]["tool_call_id"] == "call_1"
    assert "id" not in request["messages"][0]["tool_calls"][0]


@pytest.mark.parametrize("messages", [
    [{"role": "tool", "content": "orphan"}],
    [{"role": "assistant", "tool_calls": [{"id": "duplicate"}, {"id": "duplicate"}]}],
    [{"role": "tool", "tool_call_id": "same"}, {"role": "tool", "tool_call_id": "same"}],
])
def test_input_transport_rejects_ambiguous_associations(messages):
    from families.lfm2.tests.lfm25_contract_validation import prepare_transport_request

    with pytest.raises(ValueError):
        prepare_transport_request({"messages": messages})


def test_malformed_native_json_keeps_successful_process_receipt(tmp_path):
    from families.lfm2.tests.lfm25_contract_validation import run_native_case

    client = tmp_path / "malformed-client"
    client.write_text("#!/usr/bin/env python3\nprint('not JSON')\n")
    client.chmod(0o755)
    with pytest.raises(json.JSONDecodeError):
        run_native_case(client, "unused.bundle", tmp_path, tmp_path, "malformed", {"messages": []})
    receipt = json.loads((tmp_path / "malformed.native-process.json").read_text())
    assert receipt == {"returncode": 0, "stdout": "not JSON\n", "stderr": ""}


def test_native_metadata_records_sizes_and_runtime_precision(tmp_path):
    from families.lfm2.tests.lfm25_contract_validation import capture_native_metadata, FAMILY

    bundle = tmp_path / "model.bundle"
    bundle.write_bytes(b"bundle bytes")
    client = tmp_path / "identity-client"
    metadata = {"family": FAMILY, "task": "text_generation", "backend": "trt",
                "precision": "bf16", "max_cache_length": 128000}
    client.write_text("#!/usr/bin/env python3\nprint(" + repr(json.dumps(metadata)) + ")\n")
    client.chmod(0o755)
    libraries = ["libtrtmc_model_" + FAMILY + ".so", "libtrtmc_backend_trt.so",
                 "libtrtmc_c.so", "libtrtmc_core.so", "libtrtmc_runtime.so"]
    for name in libraries:
        (tmp_path / name).write_bytes(name.encode())
    identity = capture_native_metadata(bundle, tmp_path, client, "bf16")
    assert identity["bundle"] == {"name": bundle.name, "bytes": len(b"bundle bytes")}
    assert identity["client"] == {"name": client.name, "bytes": client.stat().st_size}
    assert set(identity["runtime_libraries"]) == set(libraries)
    for name, item in identity["runtime_libraries"].items():
        assert item == {"name": name, "bytes": len(name.encode())}
    assert identity["reference_precision"] == "bf16"
    assert identity["bundle_metadata"]["precision"] == "bf16"
    assert identity["precision_matches_reference"] is True
    assert str(tmp_path) not in json.dumps(identity)
    mismatch = capture_native_metadata(bundle, tmp_path, client, "fp16")
    assert mismatch["precision_matches_reference"] is False


def test_reference_and_native_failures_do_not_hide_later_cases(monkeypatch, tmp_path):
    import sys
    from contextlib import nullcontext
    from types import ModuleType, SimpleNamespace
    from families.lfm2.tests import lfm25_contract_validation as contract

    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir()
    (model_dir / "chat_template.jinja").write_bytes(FIXTURES["template"].encode("utf-8"))
    # Setup and inference dependencies are fake; the real case orchestration,
    # subprocess execution, receipt persistence, and final failure are exercised.
    monkeypatch.setattr(contract, "capture_native_metadata", lambda *args: {
        "precision_matches_reference": True, "test_fixture": True})
    monkeypatch.setattr(contract, "capture_reference_execution_identity",
                        lambda *args: {"test_fixture": True})
    monkeypatch.setattr(contract, "assert_contract", lambda *args: {"test_fixture": True})

    class Generated:
        def __getitem__(self, _):
            return self

        def tolist(self):
            return [9]

    class Model:
        generation_config = SimpleNamespace(eos_token_id=9)

        def eval(self):
            return self

        def to(self, _):
            return self

        def generate(self, input_ids, **_):
            if input_ids == [[1]]:
                raise RuntimeError("injected reference failure")
            return Generated()

    torch = ModuleType("torch")
    torch.__version__ = "unit-test"
    torch.long, torch.bfloat16, torch.float16 = "long", "bf16", "fp16"
    torch.set_num_threads = lambda _: None
    torch.tensor = lambda values, **_: values
    torch.inference_mode = nullcontext
    torch.cuda = SimpleNamespace(empty_cache=lambda: None)
    transformers = ModuleType("transformers")
    transformers.__version__ = "unit-test"
    transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=lambda *args, **kwargs: Model())
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs:
        SimpleNamespace(decode=lambda *args, **kwargs: "done"))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    selected = [(name, {"token_ids": [token], "config": {"max_new_tokens": 1}})
                for name, token in (("reference-failure", 1), ("native-failure", 2), ("valid", 3))]
    monkeypatch.setattr(contract, "cases", lambda: iter(selected))
    client = tmp_path / "client"
    client.write_text("#!/usr/bin/env python3\nimport json,sys\nr=json.load(open(sys.argv[3]))\n"
                      "print('malformed' if r['token_ids']==[2] else json.dumps({'token_ids':[9]}))\n")
    client.chmod(0o755)
    output = tmp_path / "receipts"
    monkeypatch.setattr(sys, "argv", ["contract", "--model-dir", str(model_dir),
        "--bundle", str(tmp_path / "unused.bundle"), "--runtime-root", str(tmp_path),
        "--client", str(client), "--output", str(output)])
    with pytest.raises(AssertionError, match="Model-card contract failures"):
        contract.main()
    receipt = json.loads((output / "semantic-parity.json").read_text())
    assert receipt["reference_execution"] == {"test_fixture": True}
    assert json.loads((output / "reference-execution.json").read_text()) == {"test_fixture": True}
    rows = receipt["cases"]
    assert [row["case"] for row in rows] == [name for name, _ in selected]
    assert [row["semantic_contract"] for row in rows] == ["failed", "failed", "passed"]
    assert rows[0]["failure_stage"] == "reference"
    assert rows[1]["failure_stage"] == "native" and rows[1]["native_returncode"] == 0
    assert (output / "reference-failure.reference-error.json").is_file()
    assert (output / "native-failure.native-process.json").is_file()
