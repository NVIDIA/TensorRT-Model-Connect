# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU controls for passive capture, exact replay history, and verdict isolation."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from families.lfm2_moe.tests import lfm25_row88_diagnostics as diagnostic


def target():
    return json.loads(diagnostic._FIXTURE.read_text())


def capture():
    fixture = target()
    result = diagnostic.Row88Capture()
    result.prompt_ids = fixture["prompt_ids"]
    result.calls = len(fixture["reference_prefix"])
    logits = np.zeros(128000, dtype=np.float32)
    logits[fixture["reference_prefix"][88]] = 1
    result.report.update(status="captured", logits=logits)
    return result


def test_passive_hook_records_only_row88_and_never_replaces_outputs(monkeypatch):
    torch = pytest.importorskip("torch")
    observed = []
    monkeypatch.setattr(
        diagnostic, "record_evidence", lambda name, value: observed.append((name, value))
    )

    class Model(torch.nn.Module):
        def forward(self, value):
            return SimpleNamespace(logits=value)

    model = Model()
    probe = diagnostic.Row88Capture()
    probe.attach(model)
    value = torch.zeros((1, 1, 7), dtype=torch.bfloat16)
    for index in range(94):
        value.fill_(index)
        result = model(value)
        assert result.logits is value
    probe.finish([1, 2])
    assert probe.calls == 94
    assert not model._forward_hooks
    assert probe.report["status"] == "captured"
    assert probe.report["dtype"] == "torch.bfloat16"
    assert probe.report["logits"].tolist() == [88.0] * 7
    assert [name for name, _ in observed] == [
        "reference_row88",
        "reference_state_rows",
        "reference_layer0_row1",
    ]
    assert observed[1][1]["status"] == "unavailable"
    assert not model._forward_pre_hooks


def test_hook_failure_and_missing_row_do_not_raise(monkeypatch):
    monkeypatch.setattr(diagnostic, "record_evidence", lambda *args: None)
    probe = diagnostic.Row88Capture()
    probe.attach(SimpleNamespace())
    probe.finish([1])
    assert probe.report["status"] == "capture_error"
    probe = diagnostic.Row88Capture()
    assert probe._observe(None, (), SimpleNamespace()) is None
    probe.finish([1])
    assert probe.report["status"] == "not_captured"
    probe = diagnostic.Row88Capture()
    probe.calls = 88
    assert probe._observe(None, (), SimpleNamespace()) is None
    probe.finish([1])
    assert probe.report["status"] == "capture_error"


def test_only_exact_pinned_full_card_case_is_observed():
    fixture = target()
    manifest = {key: fixture[key] for key in ("hf_id", "hf_revision")}
    case = {"name": fixture["case"]}
    assert diagnostic.capture_for(manifest, case) is not None
    assert diagnostic.capture_for({**manifest, "hf_revision": "other"}, case) is None
    assert diagnostic.capture_for(manifest, {"name": "other"}) is None


def replay(
    monkeypatch,
    tmp_path,
    *,
    wrong_prompt=False,
    wrong_position=False,
    states=False,
    observed_failure=False,
):
    fixture = target()
    calls = []
    monkeypatch.setenv("TRTMC_NATIVE_BUILD_DIR", str(tmp_path / "build"))

    def run(command, **options):
        calls.append((command, options))
        if command[0] == "cmake":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "--encode-prompt" in command:
            ids = fixture["prompt_ids"] + ([17] if wrong_prompt else [])
            return SimpleNamespace(returncode=0, stdout=json.dumps({"prompt_ids": ids}), stderr="")
        requests = [json.loads(line) for line in options["input"].splitlines()]
        assert requests[0]["tokens"] == fixture["prompt_ids"]
        assert requests[0]["reset"] and requests[0]["decode_phase"] is False
        assert requests[1]["tokens"] == fixture["reference_prefix"][:88]
        assert requests[1]["decode_phase"] is True
        logits = np.zeros(128000, dtype=np.float32)
        logits[fixture["native_prefix"][88]] = 1
        logits.tofile(requests[1]["output"])
        responses = [
            {"ready": True},
            {"position": 16, "vocab_size": 128000},
            {"position": 105 if wrong_position else 104, "vocab_size": 128000},
        ]
        if states and not observed_failure:
            assert len(requests) == 4
            assert requests[2]["reset"] and requests[2]["decode_phase"] is False
            assert requests[2]["tokens"] == requests[0]["tokens"]
            assert requests[3]["decode_phase"] is True
            assert requests[3]["tokens"] == requests[1]["tokens"]
            logits.tofile(requests[3]["output"])
            native_state_files(tmp_path, directory=Path(requests[2]["state_capture"]))
            responses.extend(responses[1:3])
        return SimpleNamespace(
            returncode=1 if observed_failure else 0,
            stdout="\n".join(map(json.dumps, responses)),
            stderr="",
        )

    monkeypatch.setattr(diagnostic.subprocess, "run", run)
    report = {}
    probe = capture()
    if states:
        probe.state = SimpleNamespace(error=None, rows=[{}] * 89)
    diagnostic._collect_native(
        report,
        probe,
        tmp_path / "bundle",
        "What is C. elegans?",
        fixture["native_prefix"],
        fixture["reference_prefix"],
        tmp_path,
    )
    return report, calls


def test_exact_native_prompt_and_phase_history_admission(monkeypatch, tmp_path):
    report, calls = replay(monkeypatch, tmp_path)
    assert report["status"] == "admitted"
    assert all(report["admission"].values())
    assert len(calls) == 3
    assert len(report["shared_history_ids"]) == 104
    assert "not intercepted" in report["prompt_scope"]


def test_observed_replay_reuses_one_process_and_identical_reset_history(monkeypatch, tmp_path):
    observed = []
    monkeypatch.setattr(
        diagnostic, "record_evidence", lambda name, value: observed.append((name, value))
    )
    report, calls = replay(monkeypatch, tmp_path, states=True)
    assert report["status"] == "admitted" and all(report["admission"].values())
    assert len(calls) == 3  # build, prompt encoding, one native process for both replays
    assert observed[0][0] == "native_state_rows"
    assert observed[0][1]["status"] == "admitted"
    assert observed[0][1]["kv_reconstruction_complete"]


def test_late_observer_process_failure_preserves_unobserved_baseline(monkeypatch, tmp_path):
    observed = []
    monkeypatch.setattr(
        diagnostic, "record_evidence", lambda name, value: observed.append((name, value))
    )
    report, _ = replay(monkeypatch, tmp_path, states=True, observed_failure=True)
    assert report["status"] == "baseline_captured_observer_failed"
    assert report["native_probe_returncode"] == 1
    assert all(report["admission"].values())
    assert report["native_logits"].shape == (128000,)
    assert report["native_logits"][target()["native_prefix"][88]] == 1
    assert observed[0][0] == "native_state_rows"
    assert observed[0][1]["status"] == "capture_error"


@pytest.mark.parametrize("wrong_prompt,wrong_position", [(True, False), (False, True)])
def test_wrong_prompt_or_state_cannot_be_admitted(
    monkeypatch, tmp_path, wrong_prompt, wrong_position
):
    report, calls = replay(
        monkeypatch, tmp_path, wrong_prompt=wrong_prompt, wrong_position=wrong_position
    )
    assert report["status"] == "target_not_reproduced"
    assert len(calls) == (2 if wrong_prompt else 3)


@pytest.mark.parametrize("which", ["reference", "native", "missing_capture", "calls"])
def test_wrong_target_prefix_or_capture_never_launches_native(monkeypatch, tmp_path, which):
    fixture = target()
    reference, native = fixture["reference_prefix"][:], fixture["native_prefix"][:]
    probe = capture()
    if which == "reference":
        reference[0] += 1
    elif which == "native":
        native[0] += 1
    elif which == "missing_capture":
        probe.report["status"] = "not_captured"
    else:
        probe.calls += 1
    monkeypatch.setattr(
        diagnostic.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("no replay on wrong target"),
    )
    report = {}
    diagnostic._collect_native(
        report, probe, tmp_path / "bundle", "prompt", native, reference, tmp_path
    )
    assert report["status"] == "target_not_reproduced"


def test_diagnostic_exception_persists_without_masking_existing_semantic_failure(
    monkeypatch, tmp_path
):
    from families.lfm2_moe.tests import test_e2e as e2e
    from tools import e2e_evidence

    fixture = target()
    recorder = e2e_evidence.Evidence(
        tmp_path / "evidence/lfm2_moe/row88",
        family="lfm2_moe",
        case="row88",
        source_revision="fixture",
    )
    monkeypatch.setattr(e2e, "_require_selected", lambda *args: None)
    monkeypatch.setattr(
        e2e, "_required_environment", lambda *args: (Path("native"), tmp_path, None)
    )
    monkeypatch.setattr(e2e, "_checkpoint", lambda *args: tmp_path)
    monkeypatch.setattr(e2e, "_build_bundle", lambda *args: None)
    monkeypatch.setattr(e2e, "_assert_rank_sections", lambda *args: None)
    native = {"token_ids": fixture["native_prefix"], "text": "full native output retained"}
    monkeypatch.setattr(e2e, "_run_native", lambda *args: native)
    monkeypatch.setattr(
        e2e,
        "_hf_reference",
        lambda *args, **kwargs: (fixture["reference_prefix"], "full reference", "decoded"),
    )

    def broken(*args):
        raise RuntimeError("diagnostic setup failed")

    def semantic_failure(*args):
        assert recorder.data["paired_row88"]["status"] == "diagnostic_error"
        raise AssertionError("unchanged semantic failure")

    monkeypatch.setattr(diagnostic, "_collect_native", broken)
    monkeypatch.setattr(e2e, "_assert_correctness", semantic_failure)
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        with pytest.raises(AssertionError, match="unchanged semantic failure"):
            e2e.test_e2e(fixture["case"], SimpleNamespace(config=None), tmp_path)
        recorder.finish("failed", failure="unchanged semantic failure")
    finally:
        e2e_evidence._ACTIVE.reset(token)
    saved = json.loads((recorder.directory / "evidence.json").read_text())
    assert saved["paired_row88"]["error_type"] == "RuntimeError"
    assert saved["native"]["text"] == native["text"]
    assert saved["reference"]["reference_text"] == "full reference"
    assert saved["failure_stage"] == "compare"


@pytest.mark.parametrize("fails", [False, True])
def test_existing_generate_arguments_and_reference_failure_are_preserved(
    monkeypatch, tmp_path, fails
):
    import sys
    from types import ModuleType
    from families.lfm2_moe.tests import test_e2e as e2e

    torch = pytest.importorskip("torch")
    fixture, observed, calls = target(), [], []
    monkeypatch.setattr(
        diagnostic, "record_evidence", lambda name, value: observed.append((name, value))
    )

    class Model(torch.nn.Module):
        device, dtype = "cuda", torch.bfloat16

        def __init__(self):
            super().__init__()
            self.config = state_config()

        def to(self, device):
            assert device == "cuda"
            return self

        def forward(self):
            return SimpleNamespace(logits=torch.ones((1, 1, 7), dtype=torch.bfloat16))

        def generate(self, **kwargs):
            calls.append(kwargs)
            for _ in range(89):
                self()
            if fails:
                raise RuntimeError("original generate failure")
            return np.array([fixture["prompt_ids"] + fixture["reference_prefix"]])

    class Inputs(dict):
        def to(self, device):
            return self

    model = Model()
    inputs = Inputs(input_ids=np.array([fixture["prompt_ids"]]), attention_mask=np.ones((1, 16)))
    monkeypatch.setattr(e2e, "_render_prompt", lambda *args: inputs)
    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: model
    )
    transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: SimpleNamespace(
            decode=lambda *args, **kwargs: "text"
        )
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    probe = diagnostic.Row88Capture()
    case = {"max_new_tokens": 8192, "repetition_penalty": 1.0, "use_chat_template": True}
    if fails:
        with pytest.raises(RuntimeError, match="original generate failure"):
            e2e._hf_reference(
                tmp_path, {"precision": "bf16"}, case, "prompt", [1], torch, diagnostic=probe
            )
    else:
        result = e2e._hf_reference(
            tmp_path, {"precision": "bf16"}, case, "prompt", [1], torch, diagnostic=probe
        )
        assert result[0] == fixture["reference_prefix"]
    assert len(calls) == 1
    assert calls[0]["input_ids"] is inputs["input_ids"]
    assert calls[0]["attention_mask"] is inputs["attention_mask"]
    assert {key: value for key, value in calls[0].items() if key not in inputs} == {
        "max_new_tokens": 8192,
        "do_sample": False,
        "repetition_penalty": 1.0,
    }
    assert not model._forward_hooks
    assert not model._forward_pre_hooks
    assert observed[0][1]["status"] == "captured"


def state_config():
    return SimpleNamespace(
        hidden_size=2048,
        num_key_value_heads=8,
        num_attention_heads=32,
        conv_L_cache=3,
        layer_types=[
            "full_attention" if i in [2, 6, 10, 14, 18, 21] else "conv" for i in range(24)
        ],
    )


def test_original_state_copies_actual_layout_without_cache_aliases():
    torch = pytest.importorskip("torch")
    fixture = target()

    class Lfm2MoeHybridConvCache:
        def __init__(self):
            # Actual logical BHND views with noncontiguous BSHD backing storage.
            backing = torch.arange(16 * 8 * 64, dtype=torch.float32).reshape(1, 16, 8, 64)
            self.key_cache = [backing.bfloat16().transpose(1, 2).clone() for _ in range(24)]
            self.value_cache = [item.clone() for item in self.key_cache]
            self.conv_cache = [torch.ones((1, 2048, 3), dtype=torch.bfloat16) for _ in range(24)]

        def get_seq_length(self):
            return self.key_cache[2].shape[-2]

    cache = Lfm2MoeHybridConvCache()
    state = diagnostic.StateRows(state_config())
    state.before(
        0,
        (),
        {
            "input_ids": torch.tensor([fixture["prompt_ids"]]),
            "cache_position": torch.arange(16),
            "attention_mask": torch.ones((1, 16)),
        },
    )
    state.after(0, SimpleNamespace(past_key_values=cache))
    assert state.error is None and len(state.rows) == 1
    before = state.report()
    expected = diagnostic._bits(cache.key_cache[2])[0]
    np.testing.assert_array_equal(before["kv_history_bits"][0, 0], expected)
    assert before["rows"][0]["attention"][0]["stride"] == list(cache.key_cache[2].stride())
    for value in (*cache.key_cache, *cache.value_cache, *cache.conv_cache):
        value.zero_()
    np.testing.assert_array_equal(before["kv_history_bits"][0, 0], expected)
    assert np.all(before["conv_history_bits"] == 0x3F80)
    assert before["kv_history_bits"].dtype == before["conv_history_bits"].dtype == np.uint16


def test_actual_original_cache_class_storage_and_append_on_cpu():
    torch = pytest.importorskip("torch")
    original = pytest.importorskip("transformers.models.lfm2_moe.modeling_lfm2_moe")
    config = original.Lfm2MoeConfig(
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=8,
        conv_L_cache=3,
        num_hidden_layers=24,
        layer_types=state_config().layer_types,
    )
    cache = original.Lfm2MoeHybridConvCache(
        config, max_batch_size=1, dtype=torch.bfloat16, device="cpu"
    )
    state = diagnostic.StateRows(config)
    fixture = target()
    for step, tokens in enumerate((fixture["prompt_ids"], [fixture["reference_prefix"][0]])):
        state.before(
            step,
            (),
            {
                "input_ids": torch.tensor([tokens]),
                "past_key_values": cache,
                "cache_position": torch.arange(16) if step == 0 else torch.tensor([16]),
            },
        )
        values = torch.full((1, 8, len(tokens), 64), step + 1, dtype=torch.bfloat16)
        for layer in state.attention:
            cache.update(values.clone(), values.clone(), layer)
        for layer in state.conv:
            cache.conv_cache[layer].copy_(
                torch.tensor([1 + step, 2 + step, 3 + step], dtype=torch.bfloat16)
            )
        state.after(step, SimpleNamespace(past_key_values=cache))
    assert state.error is None and cache.get_seq_length() == 17
    assert all(row["kv_prefix_unchanged"] and row["conv_prefix_unchanged"] for row in state.rows)
    assert state.compacted_length == 17 and state.conv_taps == 4
    assert state.conv_descriptors[0]["shape"] == [1, 2048, 3]


def test_prefix_mutation_preserves_actual_anomaly_without_fabricating_history():
    state = diagnostic.StateRows(state_config())
    first = np.zeros((6, 2, 8, 16, 64), dtype=np.uint16)
    first_row = {}
    state._append(0, first, first_row)
    assert first_row["kv_prefix_unchanged"]
    next_values = np.zeros((6, 2, 8, 17, 64), dtype=np.uint16)
    next_values[2, 1, 3, 7, 5] = 0x8000  # Signed zero is a real bit change.
    next_row = {}
    state._append(1, next_values, next_row)
    assert next_row["kv_prefix_unchanged"] is False
    assert state.compacted_length == 16
    assert state.anomaly == {
        "row": 1,
        "position": 17,
        "coordinate": [2, 1, 3, 7, 5],
        "layer": 10,
        "kind": "value",
        "differing_words": 1,
    }
    next_values.fill(0)
    assert state.anomaly_values[2, 1, 3, 7, 5] == 0x8000


def test_conv_history_requires_actual_two_tap_shift_and_preserves_anomaly():
    state = diagnostic.StateRows(state_config())
    first = np.broadcast_to(np.array([1, 2, 3], dtype=np.uint16), (18, 2048, 3)).copy()
    first_row = {"conv_prefix_unchanged": None}
    state._append_conv(0, first, first_row)
    second = np.broadcast_to(np.array([2, 3, 4], dtype=np.uint16), first.shape).copy()
    second_row = {"conv_prefix_unchanged": None}
    state._append_conv(1, second, second_row)
    assert first_row["conv_prefix_unchanged"] and second_row["conv_prefix_unchanged"]
    np.testing.assert_array_equal(state.conv_values[0, 0, :4], [1, 2, 3, 4])
    third = np.broadcast_to(np.array([3, 4, 5], dtype=np.uint16), first.shape).copy()
    third[17, 2047, 0] = 0x8000
    third_row = {"conv_prefix_unchanged": None}
    state._append_conv(2, third, third_row)
    assert third_row["conv_prefix_unchanged"] is False and state.conv_taps == 4
    assert state.conv_anomaly == {
        "row": 2,
        "position": 18,
        "coordinate": [17, 2047, 0],
        "layer": 23,
        "differing_words": 1,
    }
    third.fill(0)
    assert state.conv_anomaly_values[17, 2047, 0] == 0x8000


def test_state_snapshot_error_does_not_replace_row88_capture(monkeypatch):
    torch = pytest.importorskip("torch")
    records = []
    monkeypatch.setattr(
        diagnostic, "record_evidence", lambda name, value: records.append((name, value))
    )
    probe = diagnostic.Row88Capture()
    probe.state = diagnostic.StateRows(state_config())
    probe.calls = 88
    output = SimpleNamespace(logits=torch.ones((1, 1, 7), dtype=torch.bfloat16))
    assert probe._observe(None, (), output) is None
    probe.finish(target()["prompt_ids"])
    assert probe.report["status"] == "captured"
    assert records[1][0] == "reference_state_rows"
    assert records[1][1]["status"] == "unavailable"
    assert records[1][1]["error_type"] == "AttributeError"


def native_state_files(tmp_path, *, corrupt_shape=False, directory=None):
    directory = directory or tmp_path / "state"
    directory.mkdir()
    np.zeros((6, 2, 8, 104, 64), dtype=np.uint16).tofile(directory / "kv.u16")
    np.zeros((18, 2048, 91), dtype=np.uint16).tofile(directory / "conv.u16")
    metadata = {
        "attention_layers": [2, 6, 10, 14, 18, 21],
        "conv_layers": [i for i in range(24) if i not in [2, 6, 10, 14, 18, 21]],
        "rows_captured": 89,
        "kv_compacted_length": 104,
        "kv_history_shape": [6, 2, 8, 103 if corrupt_shape else 104, 64],
        "conv_shape": [18, 2048, 91],
        "conv_compacted_taps": 91,
        "conv_prefix_anomaly": None,
        "conv_prefix_checks": [None] + [True] * 88,
        "positions": list(range(16, 105)),
        "layout_admitted": True,
        "stream_admitted": True,
        "present_equal": True,
        "error_type": None,
        "prefix_anomaly": None,
        "present_anomaly": None,
        "prefix_checks": [None] + [True] * 88,
        "rows": [
            {
                "position": 16 + index,
                "write_index": 15 + index,
                "token": target()["prompt_ids"][-1]
                if index == 0
                else target()["reference_prefix"][index - 1],
                "phase": 0 if index == 0 else 1,
                "stream": 16,
                "layout_rechecked": True,
            }
            for index in range(89)
        ],
    }
    metadata["descriptors"] = state_descriptors()
    (directory / "metadata.json").write_text(json.dumps(metadata))
    return directory


def state_descriptors():
    records = []
    address = 4096
    for index in range(6):
        for kind in ("k", "v"):
            for prefix, mode in (("cache", "INPUT"), ("present", "OUTPUT")):
                records.append(
                    {
                        "name": f"{prefix}_{kind}_{index}",
                        "mode": mode,
                        "shape": [1, 8, 128000, 64],
                        "strides": [65536000, 8192000, 64, 1],
                        "address": address,
                    }
                )
            address += 1 << 30
    for index in range(18):
        for prefix, mode in (("conv_state", "INPUT"), ("present_conv", "OUTPUT")):
            records.append(
                {
                    "name": f"{prefix}_{index}",
                    "mode": mode,
                    "shape": [2048, 3],
                    "strides": [3, 1],
                    "address": address,
                }
            )
            address += 1 << 30
    for item in records:
        item.update(
            dtype="BF16",
            format="LINEAR",
            profile=0,
            vectorized_dimension=-1,
            scalar_bytes=2,
            components_per_element=-1,
            bytes_per_component=-1,
        )
    return records


@pytest.mark.parametrize("wrong", ["pitch", "alias", "missing", "recheck"])
def test_actual_layout_records_are_required_even_when_global_flags_pass(tmp_path, wrong):
    path = native_state_files(tmp_path) / "metadata.json"
    metadata = json.loads(path.read_text())
    assert diagnostic._native_layout_matches(metadata)
    if wrong == "pitch":
        metadata["descriptors"][0]["strides"][1] -= 64
    elif wrong == "alias":
        metadata["descriptors"][25]["address"] = metadata["descriptors"][24]["address"]
    elif wrong == "missing":
        metadata["descriptors"].pop()
    else:
        metadata["rows"][88]["layout_rechecked"] = False
    assert metadata["layout_admitted"] and not diagnostic._native_layout_matches(metadata)


@pytest.mark.parametrize(
    "changed_row,corrupt_shape", [(False, False), (True, False), (False, True)]
)
def test_native_state_requires_full_same_module_row_anchor_and_transport(
    monkeypatch, tmp_path, changed_row, corrupt_shape
):
    directory = native_state_files(tmp_path, corrupt_shape=corrupt_shape)
    baseline = np.zeros(128000, dtype=np.float32)
    observed = baseline.copy()
    if changed_row:
        observed[127999] = np.float32(0.125)
    output = tmp_path / "observed.f32"
    observed.tofile(output)
    recorded = []
    monkeypatch.setattr(
        diagnostic, "record_evidence", lambda name, value: recorded.append((name, value))
    )
    positions = [{"position": 16, "vocab_size": 128000}, {"position": 104, "vocab_size": 128000}]
    paired = {"status": "admitted"}
    diagnostic._record_native_states(
        directory, output, baseline, [{"ready": True}, *positions, *positions], paired
    )
    assert paired == {"status": "admitted"}
    assert recorded[0][0] == "native_state_rows"
    expected = "capture_error" if corrupt_shape else "not_admitted" if changed_row else "admitted"
    assert recorded[0][1]["status"] == expected
    if not corrupt_shape:
        assert (
            recorded[0][1]["row88_full_words_identical_to_unobserved_same_module"]
            is not changed_row
        )


def test_partial_native_state_survives_missing_observed_row88(monkeypatch, tmp_path):
    directory = native_state_files(tmp_path)
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        rows_captured=8,
        kv_compacted_length=23,
        kv_history_shape=[6, 2, 8, 23, 64],
        conv_compacted_taps=10,
        conv_shape=[18, 2048, 10],
    )
    for key in ("rows", "positions", "prefix_checks", "conv_prefix_checks"):
        metadata[key] = metadata[key][:8]
    metadata_path.write_text(json.dumps(metadata))
    kv = np.full(metadata["kv_history_shape"], 0x8000, dtype=np.uint16)
    conv = np.full(metadata["conv_shape"], 0x3F80, dtype=np.uint16)
    kv.tofile(directory / "kv.u16")
    conv.tofile(directory / "conv.u16")
    recorded = []
    monkeypatch.setattr(
        diagnostic, "record_evidence", lambda name, value: recorded.append((name, value))
    )
    paired = {"status": "baseline_captured_observer_failed", "native_probe_returncode": 1}
    diagnostic._record_native_states(
        directory, tmp_path / "missing-row88.f32", np.zeros(128000, dtype=np.float32), [], paired
    )
    report = recorded[0][1]
    assert report["status"] == "not_admitted"
    assert report["observed_row88_status"] == "unavailable"
    assert report["observed_row88_error_type"] == "FileNotFoundError"
    assert report["row88_full_words_identical_to_unobserved_same_module"] is False
    assert not report["kv_reconstruction_complete"] and not report["conv_reconstruction_complete"]
    np.testing.assert_array_equal(report["kv_history_bits"], kv)
    np.testing.assert_array_equal(report["conv_history_bits"], conv)
    assert paired == {
        "status": "baseline_captured_observer_failed",
        "native_probe_returncode": 1,
    }


def test_state_payloads_fit_existing_serializer_bounds_without_float_widening(tmp_path):
    from tools import e2e_evidence

    state = diagnostic.StateRows(state_config())
    arrays = {"kv": state.kv, "conv": state.conv_values}
    recorder = e2e_evidence.Evidence(
        tmp_path, family="lfm2_moe", case="state", source_revision="fixture"
    )
    recorder.record("state", arrays)
    assert sum(value.nbytes for value in arrays.values()) == 7987200
    assert len(recorder.data["artifacts"]) == 2
    assert all(item["size_bytes"] < 8 * 1024 * 1024 for item in recorder.data["artifacts"])
    for key, value in arrays.items():
        descriptor = recorder.data["state"][key]
        assert descriptor["dtype"] == "uint16"
        actual = np.load(tmp_path / descriptor["artifact"], allow_pickle=False)
        np.testing.assert_array_equal(actual, value)
    # Bound added arrays, including the latest possible first anomalies. The
    # complete report's other cases and JSON still need actual export admission.
    kv_with_anomaly = 6 * 2 * 8 * (103 + 104) * 64 * 2
    conv_with_anomaly = 18 * 2048 * (90 + 3) * 2
    max_added = (
        2 * (kv_with_anomaly + conv_with_anomaly) + 18 * 2048 * 3 * 2 + 128000 * 4 + 10 * 128
    )
    assert max_added == 19535104 and max_added < 20 * 1024 * 1024


def prefix_modules(torch):
    """Small callable fixture; no checkpoint or alternate reference arithmetic."""

    class Embedding(torch.nn.Module):
        def forward(self, value):
            return value

    class Norm(torch.nn.Module):
        def forward(self, value):
            return value.clone()

    class Projection(torch.nn.Module):
        def forward(self, value):
            return torch.cat((value, value, value), dim=-1)

    class Conv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = Projection()
            self.calls = 0

        def slow_forward(self, value, past_key_values):
            projected = self.in_proj(value)
            # The diagnostic must read this existing state, not derive it from
            # the projection (the fixture deliberately stores a distinct value).
            past_key_values.conv_cache[0][..., -1] = value[:, 0, :]
            return projected[..., :2048]

        def forward(self, value, past_key_values):
            self.calls += 1
            return self.slow_forward(value, past_key_values)

    layer = torch.nn.Module()
    layer.operator_norm, layer.conv = Norm(), Conv()
    inner = torch.nn.Module()
    inner.embed_tokens, inner.layers = Embedding(), torch.nn.ModuleList([layer])
    model = torch.nn.Module()
    model.model = inner
    return model


def test_layer0_prefix_observes_existing_values_once_without_changing_forward():
    torch = pytest.importorskip("torch")
    model = prefix_modules(torch)
    layer = model.model.layers[0]
    cache = SimpleNamespace(conv_cache=[torch.zeros((1, 2048, 3), dtype=torch.bfloat16)])
    value = torch.arange(2048, dtype=torch.float32).reshape(1, 1, 2048).to(torch.bfloat16)
    value[0, 0, 0] = -0.0

    def forward():
        embedded = model.model.embed_tokens(value)
        normalized = layer.operator_norm(embedded)
        return layer.conv(normalized, past_key_values=cache)

    expected = forward().clone()
    expected_cache = cache.conv_cache[0].clone()
    state = SimpleNamespace(
        calls=[
            {},
            {
                "input_ids": [[target()["reference_prefix"][0]]],
                "cache_position": [16],
                "cache_length_before": 16,
            },
        ],
        target=target(),
        rows=[{}, {}],
        conv_taps=4,
        conv_values=np.zeros((18, 2048, 4), dtype=np.uint16),
    )
    owner = SimpleNamespace(calls=0, state=state)
    prefix = diagnostic.Layer0Prefix(owner)
    prefix.attach(model)
    rng = torch.get_rng_state().clone()
    for row in (0, 1, 2):
        owner.calls = row
        actual = forward()
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        assert torch.equal(cache.conv_cache[0].view(torch.int16), expected_cache.view(torch.int16))
    assert layer.conv.calls == 4 and torch.equal(rng, torch.get_rng_state())
    state.conv_values[0, :, 3] = diagnostic._bits(cache.conv_cache[0])[0, :, -1]
    report = prefix.finish()
    assert report["status"] == "captured" and report["endpoint_matches_root_state"]
    assert report["input_ids"] == [[124901]]
    assert report["calls"] == dict.fromkeys(
        ["embedding", "operator_norm", "in_proj", "newest_conv_tap"], 1
    )
    assert report["tensors"]["embedding"][0, 0, 0] == 0x8000
    assert all(not array.flags.writeable for array in report["tensors"].values())
    before = report["tensors"]["newest_conv_tap"].copy()
    cache.conv_cache[0].fill_(1)
    np.testing.assert_array_equal(report["tensors"]["newest_conv_tap"], before)
    assert all(not module._forward_hooks for module in model.modules())
    assert set(report["sources"]) == {"embedding", "operator_norm", "in_proj", "conv", "slow_conv"}
    assert all(
        record["source_text"] and record["first_line"] > 0 for record in report["sources"].values()
    )


@pytest.mark.parametrize("failure", ["dtype", "shape", "duplicate", "history", "endpoint"])
def test_layer0_prefix_errors_are_separate_from_original_outputs(failure):
    torch = pytest.importorskip("torch")
    state = SimpleNamespace(
        calls=[{}, {"input_ids": [[124901]], "cache_position": [16], "cache_length_before": 16}],
        target=target(),
        rows=[{}, {}],
        conv_taps=4,
        conv_values=np.zeros((18, 2048, 4), dtype=np.uint16),
    )
    owner = SimpleNamespace(calls=1, state=state)
    prefix = diagnostic.Layer0Prefix(owner)
    for name, shape in (
        ("embedding", [1, 1, 2048]),
        ("operator_norm", [1, 1, 2048]),
        ("in_proj", [1, 1, 6144]),
    ):
        value = torch.zeros(shape, dtype=torch.bfloat16)
        if name == "in_proj" and failure == "dtype":
            value = value.float()
        if name == "in_proj" and failure == "shape":
            value = value[..., :1]
        hook = prefix._output(name, shape)
        assert hook(None, (), value) is None
        if name == "in_proj" and failure == "duplicate":
            assert hook(None, (), value) is None
    cache = SimpleNamespace(conv_cache=[torch.zeros((1, 2048, 3), dtype=torch.bfloat16)])
    if failure == "endpoint":
        cache.conv_cache[0][0, 0, -1] = -0.0
    sentinel = object()
    assert prefix._conv(None, (), {"past_key_values": cache}, sentinel) is None
    if failure == "history":
        state.calls[1]["input_ids"] = [[1]]
    report = prefix.finish()
    assert report["status"] in {"capture_error", "unavailable"}
    if failure == "endpoint":
        assert not report["endpoint_matches_root_state"]
        assert report["tensors"]["newest_conv_tap"][0, 0, 0] == 0x8000


def test_prefix_recording_failure_does_not_replace_original_failure(monkeypatch):
    torch = pytest.importorskip("torch")
    model = prefix_modules(torch)
    probe = diagnostic.Row88Capture()
    probe.attach(model)
    records = []

    def record(name, value):
        if name == "reference_layer0_row1":
            raise OSError("diagnostic export failed")
        records.append(name)

    monkeypatch.setattr(diagnostic, "record_evidence", record)
    with pytest.raises(AssertionError, match="unchanged original failure"):
        try:
            raise AssertionError("unchanged original failure")
        finally:
            probe.finish(target()["prompt_ids"])
    assert records == ["reference_row88", "reference_state_rows"]
    assert all(
        not module._forward_hooks and not module._forward_pre_hooks for module in model.modules()
    )


def test_prefix_arrays_use_bounded_lossless_trusted_payloads(tmp_path):
    from tools.e2e_evidence import Evidence

    arrays = {
        name: np.zeros(shape, np.uint16)
        for name, shape in (
            ("embedding", (1, 1, 2048)),
            ("operator_norm", (1, 1, 2048)),
            ("in_proj", (1, 1, 6144)),
            ("newest_conv_tap", (1, 2048, 1)),
        )
    }
    arrays["newest_conv_tap"][0, 0, 0] = 0x8000
    recorder = Evidence(tmp_path, family="lfm2_moe", case="prefix", source_revision="fixture")
    recorder.record("reference_layer0_row1", {"tensors": arrays})
    assert len(recorder.data["artifacts"]) == 4
    added = sum(item["size_bytes"] for item in recorder.data["artifacts"])
    assert added == 24576 + 4 * 128
    for name, value in arrays.items():
        descriptor = recorder.data["reference_layer0_row1"]["tensors"][name]
        assert descriptor["dtype"] == "uint16"
        np.testing.assert_array_equal(
            np.load(tmp_path / descriptor["artifact"], allow_pickle=False), value
        )
    assert 19535104 + added == 19560192 < 20 * 1024 * 1024
    # This bounds these arrays only; actual whole-report export is still required.


def test_prefix_hooks_accept_actual_original_meta_modules_without_forward():
    torch = pytest.importorskip("torch")
    original = pytest.importorskip("transformers.models.lfm2_moe.modeling_lfm2_moe")
    rng = torch.get_rng_state().clone()
    with torch.device("meta"):
        config = original.Lfm2MoeConfig(hidden_size=2048, conv_L_cache=3, conv_bias=False)
        layer = SimpleNamespace(
            operator_norm=original.Lfm2MoeRMSNorm(2048, eps=config.norm_eps),
            conv=original.Lfm2MoeShortConv(config, layer_idx=0),
        )
        model = SimpleNamespace(
            model=SimpleNamespace(embed_tokens=torch.nn.Embedding(1, 2048), layers=[layer])
        )
    prefix = diagnostic.Layer0Prefix(SimpleNamespace(calls=0, state=None))
    prefix.attach(model)
    assert prefix.error is None and len(prefix.handles) == 4
    assert prefix.sources["operator_norm"]["filename"] == "modeling_lfm2_moe.py"
    assert not prefix.values and torch.equal(rng, torch.get_rng_state())
    report = prefix.finish()
    assert report["status"] == "unavailable" and report["hook_removed"]
    assert all(
        not module._forward_hooks
        for module in (
            model.model.embed_tokens,
            layer.operator_norm,
            layer.conv.in_proj,
            layer.conv,
        )
    )
