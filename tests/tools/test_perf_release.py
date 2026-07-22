# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import runpy
import subprocess
import sys

import yaml

from tools import perf_release


REPOSITORY = Path(__file__).resolve().parents[2]
SUITE = REPOSITORY / "benchmarks/performance/release.yaml"


def _write_fake_trtmc(path: Path) -> None:
    manifest = REPOSITORY / "tests/e2e/models/gpt2/manifests/distilgpt2.json"
    path.write_text(
        f"""#!/usr/bin/env python3
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('command')
p.add_argument('--model')
p.add_argument('--warmup', type=int)
p.add_argument('--iterations', type=int)
p.add_argument('--telemetry')
p.add_argument('--output', type=Path)
p.add_argument('--dry-run', action='store_true')
p.add_argument('--bundle-cache')
p.add_argument('--worker')
p.add_argument('--bundle-root', action='append')
p.add_argument('--runtime-dir', action='append')
p.add_argument('--set', action='append')
a=p.parse_args()
resolved={{
 'schema_version':'trtmc.benchmark-case/v1',
 'name':'default', 'testcase':'distilgpt2', 'operation':'generate',
 'bundle_name':'distilgpt2.trtfb', 'bundle_path':'/tmp/distilgpt2.trtfb',
 'resolved_case_digest':'candidate-digest', 'sources':{{}},
 'request':{{'batch_size':1,'prompt':\"Hello, I'm a language model\",'max_new_tokens':2,
            'temperature':0.0,'top_k':1,'top_p':1.0,'min_p':0.0,'seed':-1,
            'use_chat_template':False,'enable_thinking':True}},
 'runtime':{{'cuda_graphs':False}},
 'measurement':{{'warmup':a.warmup,'iterations':a.iterations,'telemetry':'off','telemetry_interval_ms':1000}},
 'model':{{'name':'distilgpt2','hf_id':'distilbert/distilgpt2','family':'gpt2',
          'task_strategy':'text_generation_causal','runtime_strategy':'gpt2_decoder_kv_cache',
          'precision':'fp16','manifest':'gpt2/manifests/distilgpt2.json',
          'manifest_path':{str(manifest)!r},'manifest_sha256':'fake','bundle_name':'distilgpt2.trtfb',
          'build':{{'max_cache_length':256,'trust_remote_code':False}}}}
}}
if a.dry_run:
 print(json.dumps([resolved])); raise SystemExit(0)
a.output.mkdir(parents=True)
artifact=a.output/'001-distilgpt2-default'; artifact.mkdir()
observations=[{{'iteration':i,'runtime_e2e_wall_ms':10.0+i/10,'output_tokens':2}} for i in range(a.iterations)]
(artifact/'observations.jsonl').write_text(''.join(json.dumps(v)+'\\n' for v in observations))
result={{'schema_version':'trtmc.benchmark-run/v1','run_id':'fake','status':'completed',
 'measurement_policy':{{'timing_scope':'public_pipeline_call_wall'}},'environment':{{'gpu':'fake'}},
 'cells':[{{'status':'completed','name':'default','model':'distilgpt2','operation':'generate',
           'case_digest':'candidate-digest','artifact_dir':artifact.name,
           'metrics':{{'sample_count':a.iterations,'latency_ms':{{'p50':10.5}}}},
           'output_summary':{{'text':'ok','token_ids':[7,8]}}}}]}}
(a.output/'result.json').write_text(json.dumps(result))
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_fake_baseline(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--model'); p.add_argument('--task'); p.add_argument('--request-json')
p.add_argument('--precision'); p.add_argument('--max-length'); p.add_argument('--padding'); p.add_argument('--mode')
p.add_argument('--warmup'); p.add_argument('--iterations', type=int)
p.add_argument('--workload-digest'); p.add_argument('--output', type=Path)
p.add_argument('--output-token-policy')
p.add_argument('--experts-implementation')
p.add_argument('--revision'); p.add_argument('--compile-mode'); p.add_argument('--compile-dynamic', action='store_true')
p.add_argument('--compile-fullgraph', action='store_true'); p.add_argument('--trust-remote-code', action='store_true')
p.add_argument('--local-files-only', action='store_true')
a=p.parse_args()
compiled=a.mode == 'torch-compile'
value={'schema_version':'trtmc.perf-baseline/v1','status':'completed','backend':'hf-transformers',
 'mode':a.mode,'precision':a.precision,'padding':a.padding,
 'experts_implementation':a.experts_implementation,
 'compile_scope':'model.forward' if compiled else None,
 'compile_evidence':{'applied':True,'timed_callable_uses_compiled_target':True} if compiled else None,
 'workload_digest':a.workload_digest,'samples_ms':[20.0+i/10 for i in range(a.iterations)],
 'output_summary':{'text':'ok','token_ids':[7,8],'output_tokens':2},'environment':{'gpu':'fake'}}
a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(value))
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_release_suite_covers_every_ready_family_operation() -> None:
    suite = perf_release._read_yaml(SUITE)
    cases = perf_release._cases(suite)

    perf_release._validate_coverage(cases)

    assert len(cases) == 78
    assert len({(case["family"], case["operation"]) for case in cases}) == 78
    by_id = {case["id"]: case for case in cases}
    assert by_id["deberta.encode"]["baseline"]["precision"] == "fp32"
    assert by_id["fnet.encode"]["baseline"]["padding"] == "max-length"
    assert by_id["mixtral.generate"]["baseline"]["experts_implementation"] == "batched_mm"
    assert by_id["phi_moe.generate"]["baseline"]["experts_implementation"] == "batched_mm"
    assert by_id["phi_moe.generate"]["baseline"]["output_contract"] == "exact-text"
    assert by_id["opt.generate"]["request"]["max_new_tokens"] == 10


def test_compile_contract_cannot_silently_fall_back_to_eager() -> None:
    case = {
        "operation": "generate",
        "baseline": {"mode": "torch-compile"},
        "equivalence_margin_percent": 5.0,
    }
    candidate = {
        "workload_digest": "same",
        "samples_ms": [10.0],
        "output_summary": {"token_ids": [1]},
    }
    baseline = {
        "workload_digest": "same",
        "mode": "hf-eager",
        "samples_ms": [20.0],
        "output_summary": {"token_ids": [1]},
    }

    status, comparison = perf_release._classify(case, candidate, baseline)

    assert status == "contract-mismatch"
    assert "mode" in comparison["reason"]


def test_exact_text_contract_is_explicit_and_still_strict() -> None:
    case = {
        "operation": "generate",
        "baseline": {"mode": "hf-eager", "output_contract": "exact-text"},
        "equivalence_margin_percent": 5.0,
    }
    candidate = {
        "precision": "fp16",
        "workload_digest": "same",
        "samples_ms": [10.0],
        "output_summary": {"text": "same", "token_ids": [1]},
    }
    baseline = {
        "mode": "hf-eager",
        "precision": "fp16",
        "padding": "longest",
        "experts_implementation": None,
        "workload_digest": "same",
        "samples_ms": [20.0],
        "output_summary": {"text": "same", "token_ids": [2]},
    }

    status, _ = perf_release._classify(case, candidate, baseline)

    assert status == "green"


def test_run_consolidates_results_and_records_replayable_commands(tmp_path: Path) -> None:
    fake_trtmc = tmp_path / "trtmc-bench"
    fake_baseline = tmp_path / "hf_transformers.py"
    output = tmp_path / "results"
    _write_fake_trtmc(fake_trtmc)
    _write_fake_baseline(fake_baseline)

    exit_code = perf_release.main(
        [
            str(SUITE),
            "--case",
            "gpt2.generate",
            "--trtmc-bench",
            str(fake_trtmc),
            "--hf-transformers-runner",
            str(fake_baseline),
            "--output",
            str(output),
            "--ci",
        ]
    )

    assert exit_code == 0
    assert sorted(path.name for path in output.iterdir()) == [
        "report.html",
        "reproduce.py",
        "results.json",
    ]
    results = json.loads((output / "results.json").read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in results["cases"]}
    assert len(rows) == 78
    assert rows["gpt2.generate"]["status"] == "green"
    assert rows["gpt2.generate"]["candidate"]["backend"] == "trtmc-bench"
    assert rows["gpt2.generate"]["baseline"]["mode"] == "torch-compile"
    assert rows["mamba.generate"]["baseline_contract"]["mode"] == "hf-eager"
    report = (output / "report.html").read_text(encoding="utf-8")
    assert "gpt2.generate" in report
    assert "HF eager" in report
    assert "10.5" not in report
    assert "20.0" not in report

    replay = subprocess.run(
        [sys.executable, output / "reproduce.py", "gpt2.generate", "baseline", "--print"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert replay.returncode == 0
    assert str(fake_baseline) in replay.stdout
    assert "tools/perf_release.py" not in replay.stdout


def test_suite_has_explicit_eager_rows_and_unsupported_reasons() -> None:
    raw = yaml.safe_load(SUITE.read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in raw["cases"]}

    assert rows["mamba.generate"]["baseline"]["mode"] == "hf-eager"
    assert rows["rwkv.generate"]["baseline"]["mode"] == "hf-eager"
    assert rows["gemma.generate"]["baseline"]["output_contract"] == "exact-text"
    assert rows["phi.generate"]["baseline"]["output_contract"] == "exact-text"
    assert rows["phi_moe.generate"]["baseline"]["output_contract"] == "exact-text"
    assert rows["flux.generate_image"]["baseline"]["runner"] == "unsupported"
    assert rows["flux.generate_image"]["baseline"]["reason"]


def test_resolution_failure_is_recorded_per_case(tmp_path: Path, monkeypatch) -> None:
    case = perf_release._cases(perf_release._read_yaml(SUITE))[0]
    options = perf_release.RunOptions(
        output=tmp_path,
        trtmc_bench="trtmc-bench",
        trtmc_worker=None,
        hf_transformers_runner=tmp_path / "baseline.py",
        bundle_cache=None,
        bundle_roots=(),
        runtime_dirs=(),
        only="both",
        dry_run=False,
        ci=False,
        resume=False,
        rerun_failed=False,
        local_files_only=False,
        timeout_seconds=1,
    )

    def fail_resolution(*_args, **_kwargs):
        raise perf_release.PerfReleaseError("profile unavailable")

    monkeypatch.setattr(perf_release, "_resolve_candidate", fail_resolution)

    row = perf_release._run_one(case, options, tmp_path)

    assert row["status"] == "failed"
    assert row["reason"] == "profile unavailable"
    assert row["commands"]["resolve"]["argv"][-1] == "--dry-run"


def test_explicit_case_takes_precedence_over_priority() -> None:
    cases = perf_release._cases(perf_release._read_yaml(SUITE))

    selected = perf_release._selected_cases(
        cases,
        requested=["flux.generate_image"],
        priority="fast",
        max_cases=None,
    )

    assert [case["id"] for case in selected] == ["flux.generate_image"]


def test_seq2seq_token_framing_is_explicit_and_exact() -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/hf_transformers.py"))
    normalize = runner["_normalize_seq2seq_tokens"]

    assert normalize([2, 0, 11, 2], 2, 2, "strip-start-and-eos") == [0, 11]
    assert normalize([0, 11, 1], 0, 1, "strip-start") == [11, 1]
    assert normalize([0, 11, 1], 0, 1, "new-tokens") == [0, 11, 1]


def test_hf_runner_bridges_dynamic_cache_method_rename() -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/hf_transformers.py"))

    class LegacyDynamicCache:
        def get_max_length(self) -> int:
            return 17

    runner["_ensure_dynamic_cache_api"](LegacyDynamicCache)

    assert LegacyDynamicCache().get_max_cache_shape() == 17


def test_source_revision_can_be_injected_without_git(monkeypatch) -> None:
    monkeypatch.setenv("TRTMC_PERF_SOURCE_REVISION", "tested-commit")

    assert perf_release._git_commit() == "tested-commit"
