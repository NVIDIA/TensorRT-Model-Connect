# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real checkpoint -> bundle -> native semantic Tasks -> pinned reference output."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

from tools.e2e_evidence import evidence_stage, record_evidence

from families.lfm2_audio.tests.compare_semantics import compare

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "manifests/lfm25-audio-1.5b.json").read_text())


def selected(config, case_name):
    models = {
        part
        for name in config.getoption("--e2e-model", default=[]) or []
        for part in name.split(",")
    }
    cases = {
        part
        for name in config.getoption("--e2e-testcase", default=[]) or []
        for part in name.split(",")
    }
    if models and not models.intersection({"lfm2_audio", MANIFEST["name"]}):
        return False
    if cases and case_name not in cases:
        return False
    return os.environ.get("TRTMC_E2E") == "1" or bool(models or cases)


def native_probe(runtime, directory, *, public_api=False):
    name = "lfm2_audio_api_probe" if public_api else "lfm2_audio_probe"
    probe = runtime / name
    if probe.is_file():
        return probe
    # Runtime-only builds omit test executables. Build this family-owned
    # validation client against the already-built native DSOs in that case.
    repo = ROOT.parents[2]
    probe = directory / name
    command = [
        "c++",
        "-O2",
        "-std=c++17",
        "-I",
        str(repo),
        "-I",
        str(repo / "core/runtime/include"),
        "-I",
        str(repo / "core/api/include"),
        str(ROOT / ("cpp/api_probe.cpp" if public_api else "cpp/audio_probe.cpp")),
        "-L",
        str(runtime),
        *(
            ["-ltrtmc_c"]
            if public_api
            else ["-ltrtmc_model_lfm2_audio", "-ltrtmc_runtime", "-ltrtmc_core"]
        ),
        "-Wl,-rpath," + str(runtime),
        "-o",
        str(probe),
    ]
    subprocess.run(command, check=True)
    record_evidence("commands", {"argv": command})
    return probe


def audio_sources(directory):
    import soundfile as sf

    staged = os.environ.get("TRTMC_LFM2_AUDIO_MODEL_DIR")
    sources = {}
    for asset in MANIFEST["external_files"]:
        fixture = directory / asset["path"]
        if staged:
            source = Path(staged) / asset["path"]
            if not source.is_file():
                raise FileNotFoundError(f"staged LFM audio input is missing: {source}")
            # The evidence recorder can snapshot only files owned by this test.
            fixture.write_bytes(source.read_bytes())
        else:
            urllib.request.urlretrieve(asset["url"], fixture)
        samples, rate = sf.read(fixture, dtype="float32")
        assert samples.ndim == 1
        source = directory / (fixture.stem + ".f32")
        samples.tofile(source)
        sources[fixture.stem] = (source, int(rate))
        record_evidence(
            "inputs",
            {
                fixture.stem: {
                    "declared_source_url": asset["url"],
                    "acquisition": "staged" if staged else "downloaded",
                    "file": fixture,
                }
            },
        )
    return sources


def checkpoint_directory():
    from huggingface_hub import constants, snapshot_download

    return Path(
        snapshot_download(
            MANIFEST["hf_id"],
            revision=MANIFEST["hf_revision"],
            local_files_only=constants.HF_HUB_OFFLINE,
        )
    )


def run_model_card_matrix(tmp_path):
    import torch
    from tensorrt_model_connect.bundle_writer import BundleWriter
    from families.lfm2_audio.model import BuildRequest, build

    with evidence_stage("input_preparation"):
        sources = audio_sources(tmp_path)
        assert torch.cuda.is_available()
        assert MANIFEST["tensor_parallel_size"] == 1, "LFM2.5 audio requires a single device"
        runtime = Path(os.environ["TRTMC_RUNTIME_ROOT"])
        probe = native_probe(runtime, tmp_path)
        api_probe = native_probe(runtime, tmp_path, public_api=True)
        checkpoint = checkpoint_directory()
    bundle = tmp_path / MANIFEST["bundle"]
    with evidence_stage("build"):
        writer = BundleWriter(bundle)
        try:
            build(
                BuildRequest(
                    model_dir=checkpoint,
                    task=MANIFEST["task"],
                    precision=MANIFEST["precision"],
                    max_sequence_length=MANIFEST["max_sequence_length"],
                ),
                writer,
            )
            writer.finish()
        except BaseException:
            writer.abort()
            raise
    cases = []
    for original in MANIFEST["testcases"]:
        case = dict(original)
        case.setdefault("audio_temperature", 0.0)
        case.setdefault("audio_top_k", 1)
        if case["mode"] in {"tts", "tts_stream"}:
            case.setdefault("voice", "us_male")
        source, rate = sources["asr" if case["mode"] == "asr" else "question"]
        case["sample_rate"] = rate
        if case["mode"] in {"asr", "mixed"}:
            case["audio"] = str(source)
        elif case["mode"] == "text":
            if "context_tokens" in case:
                from tokenizers import Tokenizer

                tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
                prefix = "<|startoftext|><|im_start|>system\nRespond with text.<|im_end|>\n<|im_start|>user\n"
                suffix = "<|im_end|>\n<|im_start|>assistant\n"
                reserved = len(tokenizer.encode(prefix + suffix, add_special_tokens=False).ids)
                case["text"] = " hello" * (
                    case["context_tokens"] - case["max_new_tokens"] - reserved
                )
                assert (
                    len(
                        tokenizer.encode(
                            prefix + case["text"] + suffix, add_special_tokens=False
                        ).ids
                    )
                    + case["max_new_tokens"]
                    == case["context_tokens"]
                )
            from liquid_audio import ChatState, LFM2AudioProcessor

            chat = ChatState(LFM2AudioProcessor.from_pretrained(checkpoint, device="cpu"))
            chat.new_turn("system")
            chat.add_text("Respond with text.")
            chat.end_turn()
            chat.new_turn("user")
            chat.add_text(case["text"])
            chat.end_turn()
            chat.new_turn("assistant")
            case["prompt_text_tokens"] = chat.text[0].tolist()
        elif case["mode"] == "speech_response":
            case["audio"] = str(source)
            case["output"] = str(tmp_path / (case["name"] + ".native.f32"))
        elif case["mode"] == "chat":
            case["turns"] = [
                {"audio": str(source)},
                {
                    "text": "My business specialized in chairs, can you give me something related to that?"
                },
            ]
            case["output_prefix"] = str(tmp_path / (case["name"] + "-native-"))
        else:
            case.setdefault("text", "Hello there. How can I help you today?")
            case["output"] = str(tmp_path / (case["name"] + ".native.f32"))
        cases.append(case)
    cases_file = tmp_path / "cases.json"
    cases_file.write_text(json.dumps(cases))
    reference_file = tmp_path / "reference.json"
    with evidence_stage("reference"):
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "reference_generation.py"),
                "--checkpoint",
                str(checkpoint),
                "--cases",
                str(cases_file),
                "--output",
                str(reference_file),
            ],
            check=True,
        )
    with evidence_stage("native"):
        native = subprocess.run(
            [str(probe), str(bundle), str(runtime), str(cases_file)],
            capture_output=True,
            text=True,
        )
        (tmp_path / "native.jsonl").write_text(native.stdout)
        (tmp_path / "native.log").write_text(native.stderr)
        native.check_returncode()
    actual = {item["name"]: item for item in map(json.loads, native.stdout.splitlines())}
    with evidence_stage("public_api"):
        public = subprocess.run(
            [str(api_probe), str(bundle), str(runtime), str(cases_file)],
            capture_output=True,
            text=True,
        )
        (tmp_path / "public-api.jsonl").write_text(public.stdout)
        (tmp_path / "public-api.log").write_text(public.stderr)
        public.check_returncode()
    api_results = {item["name"]: item for item in map(json.loads, public.stdout.splitlines())}
    expected = {item["name"]: item for item in json.loads(reference_file.read_text())}
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    jobs = []
    for case in cases:
        name = case["name"]
        a = actual[name]
        e = expected[name]["turns"]
        if case["mode"] in {"asr", "text", "mixed"}:
            continue
        turns = a["turns"] if case["mode"] == "chat" else [a]
        for i, (turn, ref) in enumerate(zip(turns, e, strict=True)):
            native_text = (
                case["text"]
                if case["mode"] in {"tts", "tts_stream"}
                else tokenizer.decode(turn["trace"]["tokens"])
            )
            reference_text = case["text"] if case["mode"] in {"tts", "tts_stream"} else ref["text"]
            native_wave = (
                case["output_prefix"] + str(i) + ".f32"
                if case["mode"] == "chat"
                else case["output"]
            )
            for producer, waveform, intended in (
                ("native", native_wave, native_text),
                ("reference", ref["waveform"], reference_text),
            ):
                jobs.append(
                    {
                        "name": name,
                        "turn": i,
                        "producer": producer,
                        "waveform": waveform,
                        "intended": intended,
                    }
                )
    jobs_file = tmp_path / "speech-jobs.json"
    jobs_file.write_text(json.dumps(jobs, indent=2) + "\n")
    transcripts_file = tmp_path / "speech-transcripts.json"
    with evidence_stage("speech_content"):
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "speech_reference.py"),
                "--checkpoint",
                str(checkpoint),
                "--jobs",
                str(jobs_file),
                "--output",
                str(transcripts_file),
            ],
            check=True,
        )
    transcripts = json.loads(transcripts_file.read_text())
    with evidence_stage("comparison"):
        receipt = compare(
            cases,
            list(actual.values()),
            list(expected.values()),
            list(api_results.values()),
            transcripts,
        )
        (tmp_path / "semantic-comparison.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return {
        "receipt": receipt,
        "cases": {case["name"]: case for case in cases},
        "native": actual,
        "reference": expected,
        "public": api_results,
        "transcripts": transcripts,
        "speech_jobs": jobs,
    }


@pytest.fixture(scope="module")
def tmp_path(tmp_path_factory):
    # All selected nodes report one real matrix execution. Keep its artifacts
    # alive until every node has recorded its own evidence, including when the
    # runner removes ordinary passed-test temporary directories.
    return tmp_path_factory.mktemp("lfm2-audio-model-card")


@pytest.fixture(scope="module")
def matrix_execution():
    # Only state is module-scoped. GPU work happens inside test_e2e, after the
    # function-scoped evidence recorder is active for the selected case.
    return {}


def record_matrix_execution(matrix, directory, case_name):
    # Keep the complete proof once even when premerge selects just one node.
    # Subsequent nodes record only their own media and reference this origin.
    record_evidence(
        "matrix_execution",
        {
            "executed_by_case": case_name,
            "cases": directory / "cases.json",
            "native_records": directory / "native.jsonl",
            "public_records": directory / "public-api.jsonl",
            "reference_records": directory / "reference.json",
            "speech_transcripts": directory / "speech-transcripts.json",
            "waveforms": [Path(item["waveform"]) for item in matrix["speech_jobs"]],
            "asr_inputs": [Path(item["asr_input"]) for item in matrix["transcripts"]],
        },
    )


def record_case_result(matrix, case_name, directory, execution_case):
    receipt = matrix["receipt"]
    rows = [row for row in receipt["cases"] if row["name"] == case_name]
    assert len(rows) == 1, f"matrix has no unique result for {case_name}"
    row = rows[0]
    transcripts = [item for item in matrix["transcripts"] if item["name"] == case_name]
    records = {
        "input": matrix["cases"][case_name],
        "native": matrix["native"][case_name],
        "reference": matrix["reference"][case_name],
        "public": matrix["public"].get(case_name),
        "transcripts": transcripts,
    }
    files = {}
    for kind, value in records.items():
        path = directory / f"{case_name}.{kind}.json"
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        files[kind] = path
    record_evidence("inputs", {"case": files["input"]})
    record_evidence(
        "checkpoint",
        {
            "repository": MANIFEST["hf_id"],
            "checkpoint_revision": MANIFEST["hf_revision"],
            "reference_repository": MANIFEST["reference_repository"],
            "reference_revision": MANIFEST["reference_revision"],
        },
    )
    record_evidence("reference", {"records": files["reference"]})
    record_evidence(
        "native",
        {
            "comparison": row,
            "records": files["native"],
            "public_records": files["public"],
            "speech_transcripts": files["transcripts"],
            "waveforms": [
                Path(job["waveform"]) for job in matrix["speech_jobs"] if job["name"] == case_name
            ],
            "asr_inputs": [Path(item["asr_input"]) for item in transcripts],
        },
    )
    record_evidence(
        "model_card_matrix",
        {
            "executed_by_case": execution_case,
            "reused_execution": case_name != execution_case,
            "full_suite_passed": receipt["full_suite_passed"],
            "scope": receipt["scope"],
            "suite_coverage": receipt["suite_coverage"],
            "all_case_results": {item["name"]: item["passed"] for item in receipt["cases"]},
            "receipt": directory / "semantic-comparison.json",
        },
    )
    return row


@pytest.mark.parametrize("case_name", [case["name"] for case in MANIFEST["testcases"]])
def test_e2e(case_name, request, tmp_path, matrix_execution):
    if not selected(request.config, case_name):
        pytest.skip("select this LFM2.5 audio testcase or family to run native GPU validation")
    if not matrix_execution:
        matrix_execution["case"] = case_name
        try:
            matrix_execution["result"] = run_model_card_matrix(tmp_path)
            record_matrix_execution(matrix_execution["result"], tmp_path, case_name)
        except Exception as error:
            matrix_execution["error"] = error
            raise
    if "error" in matrix_execution:
        record_evidence("model_card_matrix", {"failed_execution_case": matrix_execution["case"]})
        with evidence_stage("shared_matrix_failure"):
            raise matrix_execution["error"]
    matrix = matrix_execution["result"]
    with evidence_stage("semantic_validation"):
        row = record_case_result(matrix, case_name, tmp_path, matrix_execution["case"])
        receipt = matrix["receipt"]
        names = [item["name"] for item in receipt["cases"]]
        assert sorted(names) == sorted(case["name"] for case in MANIFEST["testcases"]), receipt
        assert receipt["scope"] == "model-card" and receipt["full_suite_passed"], receipt
        assert receipt["passed"], receipt
        assert row["passed"], row
