# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only stand-in for expensive matrix execution in routing regressions.

These deliberately synthetic records test pytest, evidence, and JUnit routing;
they are not model output or a substitute for the real semantic comparator.
"""

import json
import os
from pathlib import Path
from functools import partial

from families.lfm2_audio.tests.compare_semantics import suite_coverage


def matrix(manifest, directory):
    from tools.e2e_evidence import evidence_enabled, evidence_stage, record_evidence

    assert evidence_enabled(), "matrix execution began before evidence capture"
    with Path(os.environ["LFM_AUDIO_ROUTING_CALLS"]).open("a") as output:
        output.write("executed\n")
    for stage in ("build", "reference", "native", "public_api", "speech_content"):
        with evidence_stage(stage):
            record_evidence("routing_stage", stage)
            if os.environ.get("LFM_AUDIO_ROUTING_EXCEPTION") == stage:
                raise RuntimeError("injected " + stage + " failure")
    cases = {case["name"]: dict(case) for case in manifest["testcases"]}
    rows, native, reference, public, transcripts, jobs = [], {}, {}, {}, [], []
    for name, case in cases.items():
        passed = name != os.environ.get("LFM_AUDIO_ROUTING_FAILED_CASE")
        rows.append({"name": name, "passed": passed, "checks": {"routing_control": passed}})
        native[name] = {"name": name, "producer": "synthetic-native"}
        reference[name] = {"name": name, "producer": "synthetic-reference"}
        public[name] = {"name": name, "producer": "synthetic-public"}
        if case["mode"] in {"asr", "mixed", "text"}:
            continue
        for producer in ("native", "reference"):
            path = directory / f"{name}.{producer}.f32"
            path.write_bytes(b"\0\0\0\0")
            snapshot = directory / f"{name}.{producer}.asr.f32"
            snapshot.write_bytes(path.read_bytes())
            jobs.append({"name": name, "producer": producer, "waveform": str(path)})
            transcripts.append({"name": name, "producer": producer, "asr_input": str(snapshot)})
    passed = all(row["passed"] for row in rows)
    receipt = {
        "scope": "model-card",
        "passed": passed,
        "full_suite_passed": passed,
        "suite_coverage": suite_coverage(cases.values()),
        "cases": rows,
    }
    (directory / "semantic-comparison.json").write_text(json.dumps(receipt))
    for filename, payload in (
        ("cases.json", list(cases.values())),
        ("reference.json", list(reference.values())),
        ("speech-transcripts.json", transcripts),
    ):
        (directory / filename).write_text(json.dumps(payload))
    for filename, payload in (("native.jsonl", native), ("public-api.jsonl", public)):
        (directory / filename).write_text("\n".join(json.dumps(item) for item in payload.values()))
    return {
        "receipt": receipt,
        "cases": cases,
        "native": native,
        "reference": reference,
        "public": public,
        "transcripts": transcripts,
        "speech_jobs": jobs,
    }


def pytest_collection_modifyitems(items):
    for item in items:
        if item.module.__name__ == "families.lfm2_audio.tests.test_e2e":
            item.module.run_model_card_matrix = partial(matrix, item.module.MANIFEST)
