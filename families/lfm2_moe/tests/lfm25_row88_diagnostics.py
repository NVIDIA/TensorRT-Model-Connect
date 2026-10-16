# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded observations of the retained public full-card divergence, not a verdict."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from tools.e2e_evidence import record_evidence


_FIXTURE = Path(__file__).parent / "fixtures/lfm25-row88-public-prefix.json"


class Row88Capture:
    """A passive root-forward hook; no generation argument, output or RNG change."""

    def __init__(self):
        self.calls = 0
        self.row = None
        self.handle = None
        self.error = None
        self.report = {"status": "not_captured", "row": 88}
        self.prompt_ids = []

    def attach(self, model):
        try:
            self.handle = model.register_forward_hook(self._observe)
        except Exception as error:
            self.error = type(error).__name__

    def _observe(self, module, args, output):
        index = self.calls
        self.calls += 1
        if index == 88:
            try:
                self.row = output.logits[0, -1, :].detach().clone()
                self.report.update(
                    output_shape=list(output.logits.shape), dtype=str(self.row.dtype)
                )
            except Exception as error:
                self.error = type(error).__name__
        return None

    def finish(self, prompt_ids):
        self.prompt_ids = list(prompt_ids)
        try:
            if self.handle is not None:
                self.handle.remove()
            if self.error is not None:
                self.report.update(status="capture_error", error_type=self.error)
            elif self.row is not None:
                self.report.update(status="captured", logits=self.row.float().cpu().numpy())
        except Exception as error:
            self.report.update(status="capture_error", error_type=type(error).__name__)
        finally:
            self.row = None
        self.report.update(
            forward_calls=self.calls,
            prompt_ids=self.prompt_ids,
            scope="Root forward logits before generation processors; hook returns None.",
        )
        record_evidence("reference_row88", self.report)


def capture_for(manifest, case):
    if (
        manifest.get("hf_id") == "LiquidAI/LFM2.5-8B-A1B"
        and manifest.get("hf_revision") == "5dd22602c2e9f6a097b1de4c4efe0658b605015c"
        and case.get("name") == "lfm25-8b-a1b-model-card-greedy-full"
    ):
        return Row88Capture()
    return None


def collect_native(capture, bundle, prompt, actual_ids, reference_ids, tmp_path):
    """Record missing/failed observations, then let the unchanged semantic check run."""
    if capture is None:
        return
    report = {"status": "unavailable", "row": 88, "acceptance_changed": False}
    try:
        _collect_native(report, capture, bundle, prompt, actual_ids, reference_ids, tmp_path)
    except Exception as error:
        report.update(status="diagnostic_error", error_type=type(error).__name__)
    record_evidence("paired_row88", report)


def _collect_native(report, capture, bundle, prompt, actual_ids, reference_ids, tmp_path):
    import numpy as np

    target = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    report.update(
        public_run=target["public_run"],
        expected_reference_id=target["reference_prefix"][88],
        expected_native_id=target["native_prefix"][88],
        reference_prefix=reference_ids[:89],
        actual_native_prefix=actual_ids[:89],
    )
    checks = {
        "reference_prefix": reference_ids[:89] == target["reference_prefix"],
        "native_prefix": actual_ids[:89] == target["native_prefix"],
        "shared_history": reference_ids[:88] == actual_ids[:88],
        "reference_prompt": capture.prompt_ids == target["prompt_ids"],
        "reference_capture": capture.report["status"] == "captured",
        "reference_call_count": capture.calls == len(reference_ids),
    }
    report["admission"] = checks
    if not all(checks.values()):
        report["status"] = "target_not_reproduced"
        return
    reference_logits = np.asarray(capture.report["logits"])
    checks["reference_raw_argmax"] = (
        reference_logits.shape == (128000,)
        and bool(np.isfinite(reference_logits).all())
        and int(reference_logits.argmax()) == target["reference_prefix"][88]
    )
    if not checks["reference_raw_argmax"]:
        report["status"] = "target_not_reproduced"
        return
    build = Path(os.environ["TRTMC_NATIVE_BUILD_DIR"])
    built = subprocess.run(
        ["cmake", "--build", str(build), "--target", "lfm2_moe_logits_probe", "--parallel", "2"],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    report["probe_build_returncode"] = built.returncode
    build_log = tmp_path / "row88-probe-build.log"
    build_log.write_text(built.stdout + built.stderr)
    report["probe_build_log"] = build_log
    if built.returncode:
        report["status"] = "probe_build_failed"
        return
    probe = build / "families/lfm2_moe/lfm2_moe_logits_probe"
    encoded = subprocess.run(
        [str(probe), "--encode-prompt", str(bundle), prompt],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    report["prompt_probe_returncode"] = encoded.returncode
    prompt_log = tmp_path / "row88-prompt-probe.log"
    prompt_log.write_text(encoded.stderr)
    report["prompt_probe_log"] = prompt_log
    if encoded.returncode:
        report["status"] = "prompt_probe_failed"
        return
    native_prompt = json.loads(encoded.stdout)["prompt_ids"]
    report["native_prompt_ids"] = native_prompt
    report["prompt_scope"] = (
        "Reconstructed using production renderer/tokenizer helpers, not intercepted from the earlier CLI."
    )
    checks["native_prompt"] = native_prompt == capture.prompt_ids == target["prompt_ids"]
    if not checks["native_prompt"]:
        report["status"] = "target_not_reproduced"
        return
    prefill, row = tmp_path / "row88-native-prefill.f32", tmp_path / "row88-native.f32"
    requests = [
        {"reset": True, "tokens": native_prompt, "decode_phase": False, "output": str(prefill)},
        {"tokens": reference_ids[:88], "decode_phase": True, "output": str(row)},
    ]
    completed = subprocess.run(
        [str(probe), str(bundle)],
        input="".join(json.dumps(item) + "\n" for item in requests),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    report["native_probe_returncode"] = completed.returncode
    native_log = tmp_path / "row88-native-probe.log"
    native_log.write_text(completed.stdout + completed.stderr)
    report["native_probe_log"] = native_log
    if completed.returncode:
        report["status"] = "native_probe_failed"
        return
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    checks["native_positions"] = (
        len(responses) == 3
        and responses[0] == {"ready": True}
        and responses[1] == {"position": 16, "vocab_size": 128000}
        and responses[2] == {"position": 104, "vocab_size": 128000}
    )
    native_logits = np.fromfile(row, dtype=np.float32)
    report.update(
        native_logits=native_logits, shared_history_ids=native_prompt + reference_ids[:88]
    )
    checks["native_raw_argmax"] = (
        native_logits.shape == (128000,)
        and bool(np.isfinite(native_logits).all())
        and int(native_logits.argmax()) == target["native_prefix"][88]
    )
    if native_logits.shape == (128000,) and bool(np.isfinite(native_logits).all()):
        for name, values in (("reference", reference_logits), ("native", native_logits)):
            indices = np.argsort(-values, kind="stable")[:5]
            report[name + "_top_logits"] = [
                {"token_id": int(index), "value": float(values[index])} for index in indices
            ]
            report[name + "_top_margin"] = float(values[indices[0]]) - float(values[indices[1]])
    report["status"] = "admitted" if all(checks.values()) else "target_not_reproduced"
