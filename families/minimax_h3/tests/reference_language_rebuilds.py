# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rebuild the same language gate three times before/after the RoPE correction.

The baseline must match the retained source exactly. Its numerical failures
remain failures in their original JUnit files; this runner records the A/B
experiment and requires all candidate rebuilds to pass the unchanged gate.
No checkpoint, trained-language parity or complete Ref2VA is qualified here.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


BASELINE_REVISION = "851e60eb7e975122fbeb8ef13f57481cf50abd42"
BASELINE_SHA256 = {
    "families/minimax_h3/text_encoder_builder.py": "9a4b04d10b84fe4082d465068332dcdad9f31d0afbce9b9d3782af6ae3c59174",
    "families/minimax_h3/graph_ops.py": "03dcf95b70efaa318cc9707a38be18f930d7c5c9dfca544a808b9e75f0b3c441",
    "families/minimax_h3/config.py": "7cffaec86c47e192b593d9302bd895b93675efb1860d1a55da6315b1412d170c",
    "families/minimax_h3/checkpoint.py": "8d1d7e1d0b5c197f2a2c5400cbcc7916fe7f8fb1012e2ed3c3eb0df0ecbbe176",
    "families/minimax_h3/tests/reference_language_case.py": "6d6b5cfe4236925093d90737f5e2ba397a6fc64638cec43a628b5465f8a8d8c3",
    "families/minimax_h3/tests/test_reference_conditioner.py": "393bd32a656c3a5d696cae2c28f168d3a7cea075a6f2cf7137399edf38f751c7",
    "conftest.py": "84ba49e6c1f927dbce779c0696da270674d91f581590dfe9630fe6fce5bbd757",
}
CASE = (
    "families/minimax_h3/tests/test_reference_conditioner.py::"
    "test_reference_language_injects_visual_features_and_deepstack_before_hidden_state_boundary"
)


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    candidate = Path(__file__).resolve().parents[3]
    baseline = args.baseline_repository.resolve(strict=True)
    assert baseline != candidate
    baseline_hashes = {name: sha(baseline / name) for name in BASELINE_SHA256}
    assert baseline_hashes == BASELINE_SHA256, "baseline source does not match the retained failure"
    # The seed, row counts, reference construction and numerical criterion are
    # one identical source file, not configurable independently for the sides.
    case_file = "families/minimax_h3/tests/reference_language_case.py"
    assert sha(candidate / case_file) == BASELINE_SHA256[case_file]
    args.output.mkdir(mode=0o700)
    results = []
    for side, repo in (("baseline", baseline), ("candidate", candidate)):
        for repeat in range(1, 4):
            name = f"{side}-{repeat}"
            junit = args.output / (name + ".xml")
            command = [
                sys.executable,
                "-m",
                "pytest",
                str(repo / CASE),
                "-q",
                "-s",
                "-p",
                "no:cacheprovider",
                "--rootdir=" + str(repo),
                "--basetemp=" + str(args.output / (name + "-tmp")),
                "--junitxml=" + str(junit),
            ]
            paths = [str(repo / "core/builder"), str(repo / "apps/benchmark"), str(repo)]
            # Keep the already-installed pinned package overlay, but replace
            # every source path; never import candidate code in the baseline.
            source_roots = {str(candidate), str(baseline)}
            packages = [
                p
                for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
                if p and not any(p == root or p.startswith(root + "/") for root in source_roots)
            ]
            env = dict(
                os.environ,
                PYTHONPATH=os.pathsep.join(packages + paths),
                PYTHONDONTWRITEBYTECODE="1",
            )
            with (args.output / (name + ".log")).open("x") as log:
                result = subprocess.run(
                    command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180
                )
            assert result.returncode in (0, 1), (name, result.returncode)
            log_path = args.output / (name + ".log")
            assert log_path.read_text().count("building native Qwen3-VL text stack:") == 1
            suites = list(ET.parse(junit).getroot().iter("testsuite"))
            counts = {
                key: sum(int(s.attrib.get(key, 0)) for s in suites)
                for key in ("tests", "failures", "errors", "skipped")
            }
            assert counts["tests"] == 1 and counts["errors"] == counts["skipped"] == 0, counts
            assert counts["failures"] == (1 if result.returncode else 0), counts
            results.append(
                dict(
                    side=side,
                    repeat=repeat,
                    returncode=result.returncode,
                    counts=counts,
                    junit=junit.name,
                    junit_sha256=sha(junit),
                    build_log_sha256=sha(log_path),
                )
            )
            with (args.output / "progress.json").open("w") as file:
                json.dump(results, file, indent=2, allow_nan=False)
    baseline_rows = [r for r in results if r["side"] == "baseline"]
    candidate_rows = [r for r in results if r["side"] == "candidate"]
    report = dict(
        baseline_revision=BASELINE_REVISION,
        baseline_source_sha256=baseline_hashes,
        candidate_builder_sha256=sha(candidate / "families/minimax_h3/text_encoder_builder.py"),
        results=results,
        original_gate_unchanged=True,
        baseline_failure_reproduced=any(r["returncode"] for r in baseline_rows),
        all_candidate_rebuilds_passed=all(r["returncode"] == 0 for r in candidate_rows),
        trained_language_qualified=False,
        native_ref2va_video_qualified=False,
    )
    with (args.output / "report.json").open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
    print(json.dumps(report, allow_nan=False))
    assert report["baseline_failure_reproduced"], "the retained baseline failure was not reproduced"
    assert report["all_candidate_rebuilds_passed"], "a candidate rebuild failed the unchanged gate"


if __name__ == "__main__":
    main()
