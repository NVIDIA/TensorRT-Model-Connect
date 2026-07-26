# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import Counter
import json
from argparse import Namespace
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType

import pytest
import yaml

from tools import perf_matrix


REPOSITORY = Path(__file__).resolve().parents[2]
SUITE = REPOSITORY / "benchmarks/performance/release.yaml"
GB300_ENVIRONMENT = REPOSITORY / "benchmarks/performance/environments/gb300.yaml"
PERFORMANCE_WORKFLOW = REPOSITORY / ".github/workflows/performance.yml"
TASK_ADAPTERS = {
    "bark.generate_audio": "hf-transformers-tts",
    "canary.transcribe": "nemo-asr",
    "chronos_bolt.solve": "pytorch-timeseries",
    "deepseek_ocr.generate": "hf-transformers-vlm",
    "eagle_vlm.embed": "hf-transformers-embedding",
    "eagle_vlm.rerank": "hf-transformers-reranking",
    "flux.generate_image": "hf-diffusers",
    "internvl.generate": "hf-transformers-vlm",
    "lance.generate": "upstream-lance",
    "locateanything.generate": "hf-transformers-vlm",
    "magpie_tts.generate_audio": "nemo-tts",
    "nemotron_speech_streaming.transcribe": "nemo-asr",
    "patchtsmixer.solve": "pytorch-timeseries",
    "patchtst.solve": "pytorch-timeseries",
    "personaplex.speak": "pytorch-personaplex",
    "phi4_multimodal.generate": "hf-transformers-vlm",
    "pixart.generate_image": "hf-diffusers",
    "qwen3_omni.generate_audio": "hf-qwen3-omni",
    "qwen_image.generate_image": "hf-diffusers",
    "qwen_vl.generate": "hf-transformers-vlm",
    "sam.segment_prompted": "hf-transformers-vision",
    "sam3.segment_prompted": "hf-transformers-vision",
    "sana_wm.generate_image": "upstream-sana-wm",
    "segformer.segment": "hf-transformers-vision",
    "timesfm.solve": "pytorch-timeseries",
    "timm_vit.classify": "hf-transformers-vision",
    "wan_t2v.generate_image": "hf-diffusers",
    "wan2_2_ti2v.generate_image": "hf-diffusers",
    "whisper.transcribe": "hf-transformers-asr",
    "z_image.generate_image": "hf-diffusers",
}


def _write_fake_trtmc(path: Path) -> None:
    manifest = REPOSITORY / "tests/e2e/models/gpt2/manifests/distilgpt2.json"
    path.write_text(
        f"""#!/usr/bin/env python3
import argparse, json, subprocess
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('command')
p.add_argument('--model')
p.add_argument('--case')
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
overrides={{value.split('=',1)[0]:json.loads(value.split('=',1)[1]) for value in a.set or []}}
timing_scope=overrides.get('measurement.timing_scope','public_pipeline_call_wall')
asset_loading=overrides.get('measurement.asset_loading_included',False)
resolved={{
 'schema_version':'trtmc.benchmark-case/v1',
 'name':a.case, 'testcase':a.case, 'operation':'generate',
 'bundle_name':'distilgpt2.trtfb', 'bundle_path':'/tmp/distilgpt2.trtfb',
 'resolved_case_digest':'candidate-digest', 'sources':{{}},
 'request':{{'batch_size':1,'prompt':\"Hello, I'm a language model\",'max_new_tokens':2,
            'temperature':0.0,'top_k':1,'top_p':1.0,'min_p':0.0,'seed':-1,
            'use_chat_template':False,'enable_thinking':True}},
 'runtime':{{'cuda_graphs':False}},
 'measurement':{{'warmup':a.warmup,'iterations':a.iterations,'telemetry':'off','telemetry_interval_ms':1000,
                'timing_scope':timing_scope,'asset_loading_included':asset_loading}},
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
 'measurement_policy':{{'timing_scope':timing_scope,
                       'input_preparation_included':timing_scope=='public_pipeline_call_wall',
                       'asset_loading_included':asset_loading}},
 'preparation':{{'included_in_performance_metrics':False,
                 'bundles':[{{'model':'distilgpt2','status':'built',
                             'build_time_s':83.125,
                             'included_in_performance_metrics':False}}]}},
 'environment':{{'gpu':'fake',
                'worker_build':json.loads(subprocess.run(
                    [a.worker, '--metadata'], check=True, capture_output=True, text=True
                ).stdout)['build']}},
 'cells':[{{'status':'completed','name':'default','model':'distilgpt2','operation':'generate',
           'case_digest':'candidate-digest','artifact_dir':artifact.name,
           'metrics':{{'sample_count':a.iterations,'latency_ms':{{'p50':10.5}}}},
           'output_summary':{{'text':'ok','token_ids':[7,8]}}}}]}}
(a.output/'result.json').write_text(json.dumps(result))
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_fake_worker(path: Path, revision: str) -> None:
    path.write_text(
        f"""#!/usr/bin/env python3
import json, sys
if sys.argv[1:] != ['--metadata']:
    raise SystemExit('expected --metadata')
print(json.dumps({{
    'schema_version': 'trtmc.benchmark-worker-metadata/v1',
    'build': {{
        'configuration': 'Release',
        'source_revision': {revision!r},
    }},
}}))
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
p.add_argument('--model-class', default='task'); p.add_argument('--generation-method', default='generate')
p.add_argument('--revision'); p.add_argument('--compile-mode'); p.add_argument('--compile-dynamic', action='store_true')
p.add_argument('--compile-fullgraph', action='store_true'); p.add_argument('--trust-remote-code', action='store_true')
p.add_argument('--local-files-only', action='store_true')
a=p.parse_args()
compiled=a.mode == 'torch-compile'
value={'schema_version':'trtmc.perf-baseline/v1','status':'completed','backend':'hf-transformers',
 'mode':a.mode,'precision':a.precision,'padding':a.padding,
 'model_class':a.model_class,'generation_method':a.generation_method,
 'experts_implementation':a.experts_implementation,
 'compile_scope':'model.forward' if compiled else None,
 'compile_evidence':{'applied':True,'timed_callable_uses_compiled_target':True} if compiled else None,
 'measurement_policy':{'timing_scope':'public_operation_call_wall',
                       'input_preparation_included':True,'asset_loading_included':False,
                       'model_load_excluded':True,'warmup_excluded':True,
                       'tokenization_included':True},
 'workload_digest':a.workload_digest,'samples_ms':[20.0+i/10 for i in range(a.iterations)],
 'output_summary':{'text':'ok','token_ids':[7,8],'output_tokens':2},'environment':{'gpu':'fake'}}
a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(value))
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_environment(
    path: Path,
    *,
    results_root: Path,
    scratch_root: Path,
    trtmc_bench: Path,
    trtmc_worker: Path,
    hf_transformers_runner: Path,
) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "trtmc.perf-environment/v1",
                "name": "test-gb300",
                "tools": {
                    "trtmc_bench": str(trtmc_bench),
                    "trtmc_worker": str(trtmc_worker),
                    "hf_transformers_runner": str(hf_transformers_runner),
                    "task_reference_runner": str(
                        REPOSITORY
                        / "benchmarks/performance/baselines/task_reference.py"
                    ),
                },
                "storage": {
                    "results_root": str(results_root),
                    "scratch_root": str(scratch_root),
                    "bundle_cache": None,
                    "bundle_roots": [],
                    "runtime_dirs": [],
                    "minimum_free_space_gib": 0,
                },
                "execution": {
                    "local_files_only": False,
                    "minimum_gpu_free_fraction": 0.0,
                    "timeout_seconds": 30,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_release_suite_covers_every_non_l0_ready_model_profile() -> None:
    from tensorrt_model_connect.families.wan2_2_ti2v.model_config import (
        OFFICIAL_NEGATIVE_PROMPT,
    )

    suite = perf_matrix._read_yaml(SUITE)
    cases = perf_matrix._cases(suite)
    raw_suite = yaml.safe_load(SUITE.read_text(encoding="utf-8"))
    raw_entries = raw_suite["entries"]
    raw_additional = raw_suite["additional_profiles"]
    ready_profiles = {
        entry.name
        for entry in perf_matrix.ManifestCatalog().entries()
        if entry.status == "ready" and not perf_matrix._is_l0_profile(entry.name)
    }

    perf_matrix._validate_coverage(cases)

    assert len(cases) == 105
    assert len(raw_entries) == 77
    assert len(raw_additional) == 28
    assert all(set(entry["workload"]) <= {"testcase", "request"} for entry in raw_entries)
    assert all(entry["workload"].get("testcase") for entry in raw_entries)
    assert all(entry.get("model") and entry.get("inherit") for entry in raw_additional)
    assert not any("priority" in entry for entry in raw_entries)
    assert {case["model"] for case in cases} == ready_profiles
    assert not any(perf_matrix._is_l0_profile(case["model"]) for case in cases)
    assert len({(case["family"], case["operation"]) for case in cases}) == 77
    assert len({case["family"] for case in cases}) == 76
    assert [case["operation"] for case in cases if case["family"] == "eagle_vlm"] == [
        "embed",
        "rerank",
    ]
    assert Counter(perf_matrix._candidate_timing_scope(case) for case in cases) == {
        "model_call_wall": 22,
        "public_pipeline_call_wall": 83,
    }
    assert {
        case["id"]
        for case in cases
        if case["baseline"]["asset_loading_included"]
    } == {
        "canary.transcribe",
        "deepseek_ocr.generate",
        "lance.generate",
        "nemotron_speech_streaming.transcribe",
        "nemotron_speech_streaming.transcribe@nemotron-speech-streaming-en-0.6b",
    }
    by_id = {case["id"]: case for case in cases}
    assert by_id["deberta.encode"]["baseline"]["precision"] == "fp32"
    assert by_id["fnet.encode"]["baseline"]["padding"] == "max-length"
    assert by_id["lance.generate"]["baseline"]["python_profile"] == "lance_reference"
    assert by_id["sana_wm.generate_image"]["baseline"]["adapter_options"] == {
        "reference_commit": "59629fdf790850797cb657bad014fce432bd713d",
        "intrinsics": "assets/demo_0_intrinsics.npy",
    }
    assert (
        by_id["sana_wm.generate_image"]["baseline"]["python_profile"]
        == "sana_wm_reference"
    )
    assert (
        by_id["locateanything.generate"]["baseline"]["output_contract"]
        == "exact-token-ids"
    )
    assert by_id["mixtral.generate"]["baseline"]["experts_implementation"] == "batched_mm"
    assert by_id["phi_moe.generate"]["baseline"]["experts_implementation"] == "batched_mm"
    assert by_id["qwen_moe.generate"]["baseline"]["experts_implementation"] == "batched_mm"
    assert by_id["phi_moe.generate"]["baseline"]["output_contract"] == "exact-text"
    assert by_id["opt.generate"]["workload"]["request"]["max_new_tokens"] == 10
    assert by_id["deepseek_ocr.generate"]["baseline"]["precision"] == "bf16"
    assert by_id["nemotron_h.generate"]["baseline"]["mode"] == "hf-eager"
    nemotron_baseline = by_id["nemotron_speech_streaming.transcribe"]["baseline"]
    assert {
        key: nemotron_baseline[key] for key in ("runner", "adapter", "mode", "reference_backend")
    } == {
        "runner": "task-reference",
        "adapter": "nemo-asr",
        "mode": "pytorch-eager",
        "reference_backend": "nemo_reference",
    }
    assert by_id["magpie_tts.generate_audio"]["baseline"]["adapter_options"] == {
        "speaker_encoder_revision": "e9124b5364a2c3e9b4f78da429a33cbca8f8c22b"
    }
    assert by_id["personaplex.speak"]["baseline"]["adapter_options"] == {
        "reference_commit": "3428dfd95309a7f3c84fd93259ded0f810d1ff91"
    }
    assert by_id["wan2_2_ti2v.generate_image"]["baseline"]["adapter_options"] == {
        "model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "model_revision": "b8fff7315c768468a5333511427288870b2e9635",
    }
    assert (
        by_id["wan2_2_ti2v.generate_image"]["workload"]["request"]["negative_prompt"]
        == OFFICIAL_NEGATIVE_PROMPT
    )
    diffusion_baseline = by_id["nemotron_labs_diffusion.generate"]["baseline"]
    assert diffusion_baseline["mode"] == "hf-eager"
    assert diffusion_baseline["model_class"] == "auto"
    assert diffusion_baseline["generation_method"] == "ar-generate"


def test_checked_in_gb300_environment_is_ci_runnable() -> None:
    raw = yaml.safe_load(GB300_ENVIRONMENT.read_text(encoding="utf-8"))

    assert raw["schema_version"] == "trtmc.perf-environment/v1"
    assert raw["tools"]["trtmc_bench"] == "scripts/trtmc-bench"
    assert raw["tools"]["trtmc_worker"] == "${TRTMC_PERF_WORKER}"
    assert raw["storage"]["results_root"] == "artifacts/perf"
    assert raw["storage"]["bundle_cache"] == "${TRTMC_PERF_BUNDLE_CACHE}"
    assert raw["storage"]["bundle_roots"] == "${TRTMC_PERF_BUNDLE_ROOTS}"
    assert raw["storage"]["runtime_dirs"] == "${TRTMC_PERF_RUNTIME_DIRS}"
    assert raw["execution"]["minimum_gpu_free_fraction"] == 0.0
    assert raw["execution"]["timeout_seconds"] == 7200


def test_performance_workflow_uses_the_matrix_cli_and_reference_checkouts() -> None:
    workflow = PERFORMANCE_WORKFLOW.read_text(encoding="utf-8")

    assert "python3 tools/perf_matrix.py run" in workflow
    assert "benchmarks/performance/environments/gb300.yaml" in workflow
    assert '--entry "${{ inputs.entry }}"' in workflow
    assert "tools/perf_release.py" not in workflow
    for name in (
        "TRTMC_ELF_REFERENCE_REPO",
        "TRTMC_LANCE_REFERENCE_REPO",
        "TRTMC_SANA_WM_REFERENCE_REPO",
        "PERSONAPLEX_OFFICIAL_REPO",
    ):
        assert f"{name}: ${{{{ vars.{name} }}}}" in workflow


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

    status, comparison = perf_matrix._classify(case, candidate, baseline)

    assert status == "contract-mismatch"
    assert "mode" in comparison["reason"]


def test_suite_timing_contract_drift_is_rejected_before_execution() -> None:
    case = next(
        value
        for value in perf_matrix._cases(perf_matrix._read_yaml(SUITE))
        if value["id"] == "bark.generate_audio"
    )
    drifted = {
        **case,
        "baseline": {
            **case["baseline"],
            "timing_scope": "task-pipeline-call-wall",
        },
    }

    with pytest.raises(
        perf_matrix.PerfMatrixError,
        match=r"baseline\.timing_scope must be 'task-model-call-wall'",
    ):
        perf_matrix._validate_baseline(drifted)


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

    status, _ = perf_matrix._classify(case, candidate, baseline)

    assert status == "green"


def test_generated_token_count_contract_allows_stochastic_token_content() -> None:
    case = {
        "operation": "generate",
        "baseline": {"output_contract": "generated-token-count"},
    }
    candidate = {
        "output_summary": {
            "text": "sampled candidate",
            "token_ids": [1, 2, 3],
        }
    }
    reference = {
        "output_summary": {
            "output_tokens": 3,
            "text": "different sampled reference",
            "token_ids": [4, 5, 6],
        }
    }

    assert perf_matrix._output_contract(case, candidate, reference) == (True, "")

    case["baseline"] = {}
    assert perf_matrix._output_contract(
        case,
        candidate,
        reference,
        request={"temperature": 0.7, "top_p": 0.9},
    ) == (True, "")

    case["baseline"] = {"output_contract": "generated-token-count"}
    reference["output_summary"]["output_tokens"] = 2
    reference["output_summary"]["token_ids"] = [4, 5]
    assert perf_matrix._output_contract(case, candidate, reference) == (
        False,
        "generated token count differs",
    )


def test_gpu_memory_headroom_waits_for_reclaimable_capacity(monkeypatch) -> None:
    snapshots = iter(
        [
            [(249_291, 256_703)],
            [(135_401, 256_703)],
        ]
    )
    sleeps = []
    monkeypatch.setattr(
        perf_matrix,
        "_gpu_memory_usage_mib",
        lambda: next(snapshots),
    )
    monkeypatch.setattr(perf_matrix.time, "sleep", sleeps.append)

    perf_matrix._wait_for_gpu_memory_headroom(timeout_seconds=10.0)

    assert sleeps == [1.0]


def test_backend_waits_for_gpu_headroom_before_each_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    monkeypatch.setattr(perf_matrix, "_command_environment", lambda: {})
    monkeypatch.setattr(perf_matrix, "_workload_digest", lambda _resolved: "digest")
    monkeypatch.setattr(
        perf_matrix,
        "_candidate_base_argv",
        lambda _case, _options: ["candidate"],
    )
    monkeypatch.setattr(
        perf_matrix,
        "_baseline_argv",
        lambda _case, _resolved, _output, _options: (["baseline"], "base"),
    )
    monkeypatch.setattr(
        perf_matrix,
        "_wait_for_gpu_memory_headroom",
        lambda **kwargs: events.append(
            ("wait", kwargs["minimum_free_fraction"])
        ),
    )
    monkeypatch.setattr(
        perf_matrix,
        "_run_command",
        lambda argv, _environment, _timeout: events.append(("run", argv[0]))
        or {"exit_code": 0, "stdout": ""},
    )
    monkeypatch.setattr(
        perf_matrix,
        "_candidate_result",
        lambda _directory, _digest: {},
    )
    monkeypatch.setattr(perf_matrix, "_read_baseline", lambda _path: {})
    monkeypatch.setattr(
        perf_matrix,
        "_classify",
        lambda *_args, **_kwargs: ("green", {}),
    )
    monkeypatch.setattr(perf_matrix, "_stable_even", lambda _value: True)

    row = {"resolved_settings": {}}
    perf_matrix._run_supported_case(
        {"id": "example"},
        {"model": {"precision": "fp16"}, "request": {}},
        Namespace(timeout_seconds=30, minimum_gpu_free_fraction=0.25),
        tmp_path,
        row,
    )

    assert events == [
        ("wait", 0.25),
        ("run", "candidate"),
        ("wait", 0.25),
        ("run", "baseline"),
    ]
    assert row["status"] == "green"


@pytest.mark.parametrize(
    "status",
    ["green", "yellow", "red", "contract-mismatch"],
)
def test_resume_keeps_terminal_comparison_results(status: str) -> None:
    assert perf_matrix._should_skip({"status": status})


def test_timing_scope_details_state_measured_included_and_excluded_work() -> None:
    candidate = {
        "measurement_policy": {
            "timing_scope": "model_call_wall",
            "load_excluded": True,
            "warmup_excluded": True,
            "asset_loading_included": False,
            "telemetry_in_timed_path": False,
        }
    }
    model_only_baseline = {
        "timing_scope": "task-model-call-wall",
        "input_preparation_included": False,
        "model_load_included": False,
    }

    assert perf_matrix._timing_scope_details(candidate, "candidate") == {
        "measured": "first TensorRT module call through returned output",
        "included": "module input transfer, model execution, inter-module work, output materialization",
        "excluded": "bundle/model load, warmup, pipeline preprocessing, asset loading, telemetry",
    }
    assert perf_matrix._timing_scope_details(model_only_baseline, "baseline") == {
        "measured": "task model call",
        "included": "prepared model invocation through returned summary",
        "excluded": "model load, warmup, input preparation, asset loading",
    }


def test_ocr_text_contract_preserves_required_content_and_allows_format_variation() -> None:
    baseline = {
        "output_contract": "ocr-text",
        "max_normalized_edit_distance": 0.5,
        "required_substrings": ["Architecture", "Attention:Standard Q/K/V/O"],
    }
    case = {"operation": "generate", "baseline": baseline}
    candidate = {
        "output_summary": {
            "text": "OCR title\nArchitecture:\nAttention: Standard Q/K/V/O (no biases)"
        }
    }
    reference = {
        "output_summary": {"text": "Architecture:\nAttention:Standard Q/K/V/O"}
    }

    assert perf_matrix._output_contract(case, candidate, reference) == (True, "")

    candidate["output_summary"]["text"] = "OCR title only"
    matched, reason = perf_matrix._output_contract(case, candidate, reference)
    assert not matched
    assert reason == "TRTMC OCR text misses required content"


def test_normalized_text_contract_allows_only_case_and_whitespace_variation() -> None:
    case = {
        "operation": "generate",
        "baseline": {"output_contract": "normalized-text"},
    }
    candidate = {"output_summary": {"text": "Paris\n</think>\n"}}
    reference = {"output_summary": {"text": "paris  </think>"}}

    assert perf_matrix._output_contract(case, candidate, reference) == (True, "")

    reference["output_summary"]["text"] = "London </think>"
    assert perf_matrix._output_contract(case, candidate, reference) == (
        False,
        "normalized generated text differs",
    )


def test_token_agreement_contract_bounds_cross_precision_drift() -> None:
    case = {
        "operation": "generate",
        "baseline": {
            "output_contract": "token-agreement",
            "min_positional_token_agreement": 0.85,
            "max_normalized_edit_distance": 0.2,
        },
    }
    candidate = {
        "output_summary": {
            "text": "used Iran force as words is used",
            "token_ids": [261, 7449, 2054, 38, 1234, 19, 261],
        }
    }
    reference = {
        "output_summary": {
            "text": "expression Iran force as words are used",
            "token_ids": [3893, 7449, 2054, 38, 1234, 33, 261],
        }
    }

    assert perf_matrix._output_contract(case, candidate, reference) == (
        False,
        "positional token agreement is below the configured contract",
    )

    case["baseline"]["min_positional_token_agreement"] = 0.7
    assert perf_matrix._output_contract(case, candidate, reference) == (
        False,
        "normalized text distance exceeds the configured contract",
    )

    case["baseline"]["max_normalized_edit_distance"] = 0.4
    assert perf_matrix._output_contract(case, candidate, reference) == (True, "")


def test_segmentation_contract_rejects_raw_masks_against_postprocessed_masks() -> None:
    case = {
        "operation": "segment_prompted",
        "baseline": {"output_contract": "segmentation-shape"},
    }
    candidate = {
        "output_summary": {
            "num_masks": 3,
            "height": 382,
            "width": 640,
        }
    }
    raw_reference = {
        "output_summary": {
            "shape": [1, 1, 3, 256, 256],
            "element_count": 196_608,
        }
    }

    assert perf_matrix._output_contract(case, candidate, raw_reference) == (
        False,
        "segmentation output shape differs",
    )

    aligned_reference = {
        "output_summary": {
            "num_masks": 3,
            "height": 382,
            "width": 640,
        }
    }
    assert perf_matrix._output_contract(case, candidate, aligned_reference) == (True, "")


def test_audio_contract_rejects_different_generated_sample_counts() -> None:
    case = {
        "operation": "generate_audio",
        "baseline": {"output_contract": "audio-shape"},
    }
    candidate = {
        "output_summary": {
            "num_samples": 58_965,
            "sample_rate": 24_000,
        }
    }
    reference = {
        "output_summary": {
            "audio_samples": 37_845,
            "sample_rate": 24_000,
        }
    }

    assert perf_matrix._output_contract(case, candidate, reference) == (
        False,
        "audio output shape differs",
    )

    reference["output_summary"]["audio_samples"] = 58_965
    assert perf_matrix._output_contract(case, candidate, reference) == (True, "")


def test_media_contract_compares_materialized_frame_geometry() -> None:
    case = {
        "operation": "generate_image",
        "baseline": {"output_contract": "media-shape"},
    }
    candidate = {
        "output_summary": {
            "num_frames": 5,
            "height": 384,
            "width": 672,
            "channels": 3,
        }
    }
    reference = {
        "output_summary": {
            "media_count": 5,
            "media_type": "video",
            "height": 384,
            "width": 672,
            "channels": 3,
        }
    }

    assert perf_matrix._output_contract(case, candidate, reference) == (True, "")
    reference["output_summary"]["height"] = 704
    assert perf_matrix._output_contract(case, candidate, reference) == (
        False,
        "media output shape differs",
    )


def test_media_contract_compares_image_batch_size_to_media_count() -> None:
    case = {
        "operation": "generate_image",
        "baseline": {"output_contract": "media-shape"},
    }
    candidate = {
        "output_summary": {
            "batch_size": 2,
            "num_frames": 1,
            "height": 384,
            "width": 384,
            "channels": 3,
        }
    }
    reference = {
        "output_summary": {
            "media_count": 2,
            "media_type": "image",
            "height": 384,
            "width": 384,
            "channels": 3,
        }
    }

    assert perf_matrix._output_contract(case, candidate, reference) == (True, "")


@pytest.mark.parametrize("configuration", ["", "Debug", "RelWithDebInfo"])
def test_worker_preflight_rejects_non_release_builds(configuration: str) -> None:
    metadata = {
        "schema_version": "trtmc.benchmark-worker-metadata/v1",
        "build": {
            "configuration": configuration,
            "source_revision": "abc123",
        },
    }

    with pytest.raises(
        perf_matrix.PerfMatrixError,
        match="worker build configuration must be Release",
    ):
        perf_matrix._validate_worker_metadata(metadata, "abc123")


def test_worker_preflight_rejects_stale_source_revision() -> None:
    metadata = {
        "schema_version": "trtmc.benchmark-worker-metadata/v1",
        "build": {
            "configuration": "Release",
            "source_revision": "old-revision",
        },
    }

    with pytest.raises(
        perf_matrix.PerfMatrixError,
        match="worker source revision",
    ):
        perf_matrix._validate_worker_metadata(metadata, "current-revision")


def test_run_consolidates_results_and_records_replayable_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_trtmc = tmp_path / "trtmc-bench"
    fake_worker = tmp_path / "trtmc_benchmark_worker"
    fake_baseline = tmp_path / "hf_transformers.py"
    results_root = tmp_path / "results"
    scratch_root = tmp_path / "scratch"
    environment = tmp_path / "gb300.yaml"
    monkeypatch.setenv("TRTMC_PERF_SOURCE_REVISION", "tested-commit")
    _write_fake_trtmc(fake_trtmc)
    _write_fake_worker(fake_worker, "tested-commit")
    _write_fake_baseline(fake_baseline)
    _write_environment(
        environment,
        results_root=results_root,
        scratch_root=scratch_root,
        trtmc_bench=fake_trtmc,
        trtmc_worker=fake_worker,
        hf_transformers_runner=fake_baseline,
    )

    exit_code = perf_matrix.main(
        [
            "run",
            str(SUITE),
            "--environment",
            str(environment),
            "--entry",
            "gpt2.generate",
        ]
    )

    assert exit_code == 0
    run_directories = [path for path in results_root.iterdir() if path.is_dir()]
    assert len(run_directories) == 1
    output = run_directories[0]
    assert sorted(path.name for path in output.iterdir()) == [
        "report.html",
        "results.json",
    ]
    assert not scratch_root.exists()
    results = json.loads((output / "results.json").read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in results["cases"]}
    assert len(rows) == 105
    assert results["environment_config"]["name"] == "test-gb300"
    assert (
        results["environment_config"]["execution"]["minimum_gpu_free_fraction"]
        == 0.0
    )
    assert results["environment_config"]["source"] == str(environment.resolve())
    catalog_entries = perf_matrix.ManifestCatalog().entries()
    catalog_counts = Counter(entry.status for entry in catalog_entries)
    excluded_l0_profiles = sum(
        entry.status == "ready" and perf_matrix._is_l0_profile(entry.name)
        for entry in catalog_entries
    )
    expected_catalog_coverage = {
        "total_profiles": len(catalog_entries),
        "ready_profiles": catalog_counts["ready"],
        "release_profiles": catalog_counts["ready"] - excluded_l0_profiles,
        "excluded_l0_profiles": excluded_l0_profiles,
        "distributed_profiles": catalog_counts["distributed"],
        "other_profiles": sum(
            count
            for status, count in catalog_counts.items()
            if status not in {"ready", "distributed"}
        ),
    }
    assert results["catalog_coverage"] == expected_catalog_coverage
    assert results["timing_preflight"]["status"] == "aligned"
    assert results["timing_preflight"]["case_count"] == 1
    assert results["reference_preflight"]["status"] == "ready"
    assert results["reference_preflight"]["entry_count"] == 1
    assert results["candidate_worker_preflight"]["build"] == {
        "configuration": "Release",
        "source_revision": "tested-commit",
    }
    assert results["candidate_worker_preflight"]["validated_against"] == "tested-commit"
    assert rows["gpt2.generate"]["status"] == "green"
    assert rows["gpt2.generate"]["candidate"]["backend"] == "trtmc-bench"
    assert rows["gpt2.generate"]["candidate"]["preparation"] == {
        "included_in_performance_metrics": False,
        "bundles": [
            {
                "model": "distilgpt2",
                "status": "built",
                "build_time_s": 83.125,
                "included_in_performance_metrics": False,
            }
        ],
    }
    assert rows["gpt2.generate"]["baseline"]["mode"] == "torch-compile"
    assert rows["mamba.generate"]["baseline_contract"]["mode"] == "hf-eager"
    report = (output / "report.html").read_text(encoding="utf-8")
    assert ">gpt2<" in report
    assert "HF eager" in report
    assert "76 families" in report
    assert "105 model-profile comparisons" in report
    assert "105 single-process profiles" in report
    assert (
        f"{expected_catalog_coverage['excluded_l0_profiles']} duplicate L0 profiles are excluded"
    ) in report
    assert (f"{expected_catalog_coverage['ready_profiles']} ready catalog profiles") in report
    assert (f"{expected_catalog_coverage['distributed_profiles']} distributed profiles") in report
    assert (
        f"{expected_catalog_coverage['other_profiles']} other or unsupported profiles"
    ) in report
    assert "<th>Model profile</th>" in report
    assert "TRTMC infer p50 (ms)" in report
    assert "Baseline infer p50 (ms)" in report
    assert "TRTMC bundle preparation" in report
    assert "Built · 1m 23.1s" in report
    assert "83.125 s" in report
    assert "1 built in this run (1m 23.1s total)" in report
    assert "excluded from the infer-time traffic-light comparison" in report
    assert ">10.450<" in report
    assert ">20.450<" in report
    assert "<th>Status</th>" not in report
    assert "<td>green</td>" not in report
    assert "Needs alignment" not in report
    assert "Measured scope" in report
    assert "Timing contracts were validated before execution for 1 comparisons" in report
    assert "Measured: public pipeline call" in report
    assert "Measured: public operation call" in report
    assert "Includes: pipeline-internal preprocessing, model execution, returned output" in report
    assert "Excludes: model load, compile setup, warmup" in report
    assert "Show raw commands" in report
    assert str(fake_trtmc) in report
    assert str(fake_baseline) in report
    assert "reproduce.py" not in report
    assert str(REPOSITORY) in report
    assert "PYTHONPATH" in report

    baseline_argv = rows["gpt2.generate"]["commands"]["baseline"]["argv"]
    request = baseline_argv[baseline_argv.index("--request-json") + 1]
    assert json.loads(request)["prompt"] == "Hello, I'm a language model"
    assert rows["gpt2.generate"]["resolved_settings"]["workload"] == {
        "source": "testcase",
        "testcase": "distilgpt2",
        "request": rows["gpt2.generate"]["resolved_settings"]["request"],
    }
    assert rows["gpt2.generate"]["commands"]["trtmc"]["cwd"] == str(REPOSITORY)
    assert rows["gpt2.generate"]["commands"]["baseline"]["cwd"] == str(REPOSITORY)

    rows["gpt2.generate"]["status"] = "failed"
    (output / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    assert perf_matrix.main(["resume", str(output)]) == 0
    resumed = json.loads((output / "results.json").read_text(encoding="utf-8"))
    resumed_rows = {row["id"]: row for row in resumed["cases"]}
    assert resumed_rows["gpt2.generate"]["status"] == "green"
    assert sorted(path.name for path in output.iterdir()) == [
        "report.html",
        "results.json",
    ]
    assert not scratch_root.exists()


def test_check_runs_preflight_without_creating_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_trtmc = tmp_path / "trtmc-bench"
    fake_worker = tmp_path / "trtmc_benchmark_worker"
    fake_baseline = tmp_path / "hf_transformers.py"
    results_root = tmp_path / "results"
    scratch_root = tmp_path / "scratch"
    environment = tmp_path / "gb300.yaml"
    monkeypatch.setenv("TRTMC_PERF_SOURCE_REVISION", "tested-commit")
    _write_fake_trtmc(fake_trtmc)
    _write_fake_worker(fake_worker, "tested-commit")
    _write_fake_baseline(fake_baseline)
    _write_environment(
        environment,
        results_root=results_root,
        scratch_root=scratch_root,
        trtmc_bench=fake_trtmc,
        trtmc_worker=fake_worker,
        hf_transformers_runner=fake_baseline,
    )

    exit_code = perf_matrix.main(
        [
            "check",
            str(SUITE),
            "--environment",
            str(environment),
            "--entry",
            "gpt2.generate",
        ]
    )

    assert exit_code == 0
    assert not results_root.exists()
    assert not scratch_root.exists()


def test_run_records_preflight_failure_and_finishes_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_trtmc = tmp_path / "trtmc-bench"
    fake_worker = tmp_path / "trtmc_benchmark_worker"
    fake_baseline = tmp_path / "hf_transformers.py"
    results_root = tmp_path / "results"
    scratch_root = tmp_path / "scratch"
    environment = tmp_path / "gb300.yaml"
    monkeypatch.setenv("TRTMC_PERF_SOURCE_REVISION", "tested-commit")
    _write_fake_trtmc(fake_trtmc)
    _write_fake_worker(fake_worker, "tested-commit")
    _write_fake_baseline(fake_baseline)
    _write_environment(
        environment,
        results_root=results_root,
        scratch_root=scratch_root,
        trtmc_bench=fake_trtmc,
        trtmc_worker=fake_worker,
        hf_transformers_runner=fake_baseline,
    )

    def fail_reference(cases, _options):
        case_id = str(cases[0]["id"])
        return (
            {},
            {
                "status": "partial",
                "entry_count": 0,
                "failed_entry_count": 1,
                "entries": [],
                "failures": [],
            },
            {
                case_id: {
                    "stage": "reference-preflight",
                    "reason": "profile unavailable",
                    "argv": [str(fake_trtmc), "run", "--dry-run"],
                }
            },
        )

    monkeypatch.setattr(perf_matrix, "_preflight_selected", fail_reference)

    exit_code = perf_matrix.main(
        [
            "run",
            str(SUITE),
            "--environment",
            str(environment),
            "--entry",
            "gpt2.generate",
        ]
    )

    assert exit_code == 1
    output = next(path for path in results_root.iterdir() if path.is_dir())
    results = json.loads((output / "results.json").read_text(encoding="utf-8"))
    row = next(row for row in results["cases"] if row["id"] == "gpt2.generate")
    assert results["status"] == "completed-with-errors"
    assert results["timing_preflight"]["status"] == "partial"
    assert row["status"] == "failed"
    assert row["failure_stage"] == "reference-preflight"
    assert row["reason"] == "profile unavailable"
    assert row["commands"]["resolve"]["rendered"].endswith("run --dry-run")
    assert sorted(path.name for path in output.iterdir()) == [
        "report.html",
        "results.json",
    ]
    assert not scratch_root.exists()


def test_task_reference_commands_record_external_checkout_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRTMC_ELF_REFERENCE_REPO", "/references/ELF")
    monkeypatch.setenv("TRTMC_LANCE_REFERENCE_REPO", "/references/Lance")
    monkeypatch.setenv("TRTMC_SANA_WM_REFERENCE_REPO", "/references/Sana")
    monkeypatch.setenv("PERSONAPLEX_OFFICIAL_REPO", "/references/PersonaPlex")

    assert perf_matrix._resolved_adapter_options({"adapter": "upstream-elf"}) == {
        "reference_repo": "/references/ELF"
    }
    assert perf_matrix._resolved_adapter_options({"adapter": "upstream-lance"}) == {
        "reference_repo": "/references/Lance"
    }
    assert perf_matrix._resolved_adapter_options({"adapter": "upstream-sana-wm"}) == {
        "reference_repo": "/references/Sana"
    }
    assert perf_matrix._resolved_adapter_options({"adapter": "pytorch-personaplex"}) == {
        "official_repo": "/references/PersonaPlex"
    }
    assert perf_matrix._resolved_adapter_options(
        {
            "adapter": "upstream-elf",
            "adapter_options": {"reference_repo": "/explicit/ELF"},
        }
    ) == {"reference_repo": "/explicit/ELF"}


def test_external_reference_adapter_rejects_a_missing_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTMC_ELF_REFERENCE_REPO", raising=False)

    with pytest.raises(
        perf_matrix.PerfMatrixError,
        match="requires adapter_options.reference_repo or TRTMC_ELF_REFERENCE_REPO",
    ):
        perf_matrix._resolved_adapter_options({"adapter": "upstream-elf"})


def test_suite_has_explicit_eager_and_task_reference_rows() -> None:
    raw = yaml.safe_load(SUITE.read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in raw["entries"]}

    assert rows["mamba.generate"]["baseline"]["mode"] == "hf-eager"
    assert rows["rwkv.generate"]["baseline"]["mode"] == "hf-eager"
    assert rows["deepseek_v2.generate"]["baseline"] == {
        "runner": "hf-transformers",
        "mode": "hf-eager",
        "experts_implementation": "batched_mm",
    }
    assert rows["nemotron_labs_diffusion.generate"]["baseline"] == {
        "runner": "hf-transformers",
        "mode": "hf-eager",
        "model_class": "auto",
        "generation_method": "ar-generate",
        "output_contract": "normalized-text",
    }
    assert rows["gemma.generate"]["baseline"]["output_contract"] == "exact-text"
    assert rows["phi.generate"]["baseline"]["output_contract"] == "exact-text"
    assert rows["phi_moe.generate"]["baseline"]["output_contract"] == "exact-text"
    assert rows["deepseek_ocr.generate"]["baseline"]["output_contract"] == "ocr-text"
    assert rows["qwen_vl.generate"]["baseline"]["output_contract"] == "normalized-text"
    assert not any(row["baseline"]["runner"] == "unsupported" for row in rows.values())
    assert {
        case_id: row["baseline"].get("adapter")
        for case_id, row in rows.items()
        if row["baseline"]["runner"] == "task-reference"
    } == TASK_ADAPTERS
    for case_id in TASK_ADAPTERS:
        assert rows[case_id]["baseline"]["reference_backend"]
        assert rows[case_id]["baseline"]["mode"] in {"hf-eager", "pytorch-eager"}


def test_resolution_failure_is_recorded_without_stopping_other_entries(
    tmp_path: Path, monkeypatch
) -> None:
    case = perf_matrix._cases(perf_matrix._read_yaml(SUITE))[0]
    options = perf_matrix.RunOptions(
        output=tmp_path,
        scratch_root=tmp_path / "scratch",
        trtmc_bench="trtmc-bench",
        trtmc_worker=None,
        hf_transformers_runner=tmp_path / "baseline.py",
        task_reference_runner=tmp_path / "task_reference.py",
        bundle_cache=None,
        bundle_roots=(),
        runtime_dirs=(),
        local_files_only=False,
        minimum_free_space_gib=0,
        minimum_gpu_free_fraction=0.45,
        timeout_seconds=1,
    )

    def fail_resolution(*_args, **_kwargs):
        raise perf_matrix.PerfMatrixError("profile unavailable")

    monkeypatch.setattr(perf_matrix, "_resolve_candidate", fail_resolution)

    preflight, failures = perf_matrix._preflight_candidates([case], options)

    assert preflight == {}
    assert failures[case["id"]]["stage"] == "candidate-preflight"
    assert failures[case["id"]]["reason"] == "profile unavailable"
    assert failures[case["id"]]["argv"][-1] == "--dry-run"


def test_entry_is_the_only_run_selection() -> None:
    cases = perf_matrix._cases(perf_matrix._read_yaml(SUITE))

    selected = perf_matrix._selected_cases(
        cases,
        requested=["flux.generate_image"],
    )

    assert [case["id"] for case in selected] == ["flux.generate_image"]


def test_candidate_preflight_resolves_the_build_python_profile(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        perf_matrix,
        "default_execution_profiles",
        lambda **_kwargs: {
            "build": "family-build",
            "runtime": "base",
            "reference": "base",
        },
    )
    monkeypatch.setattr(
        perf_matrix,
        "resolve_profile_python",
        lambda profile, python: calls.append((profile, python)) or "/profile/python",
    )

    profile, python = perf_matrix._candidate_build_python_profile({"model": {"family": "example"}})

    assert profile == "family-build"
    assert python == "/profile/python"
    assert calls == [("family-build", sys.executable)]


def test_candidate_preflight_rejects_an_unavailable_build_python_profile(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        perf_matrix,
        "default_execution_profiles",
        lambda **_kwargs: {
            "build": "family-build",
            "runtime": "base",
            "reference": "base",
        },
    )

    def reject_profile(*_args):
        raise RuntimeError("profile is not prebuilt")

    monkeypatch.setattr(perf_matrix, "resolve_profile_python", reject_profile)

    with pytest.raises(
        perf_matrix.PerfMatrixError,
        match="candidate build Python profile 'family-build' is unavailable",
    ):
        perf_matrix._candidate_build_python_profile({"model": {"family": "example"}})


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


def test_hf_runner_bridges_removed_input_check_decorator() -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/hf_transformers.py"))

    class GenericModule:
        pass

    def forward() -> str:
        return "ok"

    runner["_ensure_transformers_generic_api"](GenericModule)

    assert GenericModule.check_model_inputs(forward) is forward


def test_hf_runner_closes_ignored_disabled_thinking_prompt() -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/hf_transformers.py"))
    captured: dict[str, object] = {}

    class FakeTokenizer:
        def apply_chat_template(self, _messages, **kwargs):
            captured["template_kwargs"] = kwargs
            return "<SPECIAL_11>Assistant\n<think>\n"

        def __call__(self, text, **kwargs):
            captured.update(text=text, tokenizer_kwargs=kwargs)
            return {"input_ids": "encoded"}

    encoded = runner["_chat_prompt_inputs"](
        FakeTokenizer(), "hello", enable_thinking=False
    )

    assert encoded == {"input_ids": "encoded"}
    assert captured["text"] == "<SPECIAL_11>Assistant\n<think></think>"
    assert captured["template_kwargs"] == {
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": False,
    }
    assert captured["tokenizer_kwargs"] == {
        "return_tensors": "pt",
        "add_special_tokens": False,
    }


def test_source_revision_can_be_injected_without_git(monkeypatch) -> None:
    monkeypatch.setenv("TRTMC_PERF_SOURCE_REVISION", "tested-commit")

    assert perf_matrix._git_commit() == "tested-commit"


def test_task_reference_runner_measures_loaded_public_operation(
    tmp_path: Path, monkeypatch
) -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    calls: list[int] = []

    def load_session(*_args):
        def invoke():
            calls.append(1)
            return {"text": "ok", "output_tokens": 1}

        return runner["Session"](
            invoke,
            "revision",
            "fake-framework",
            reference_source={"repository": "official", "revision": "source-revision"},
        )

    runner["LOADERS"]["hf-transformers-asr"] = load_session
    runner["_synchronize"] = lambda: None
    runner["_environment"] = lambda: {"gpu": "fake"}
    output = tmp_path / "baseline.json"
    arguments = runner["build_parser"]().parse_args(
        [
            "--adapter",
            "hf-transformers-asr",
            "--family",
            "whisper",
            "--operation",
            "transcribe",
            "--model",
            "openai/whisper-tiny",
            "--manifest",
            str(SUITE),
            "--request-json",
            '{"audio_path":"sample.wav"}',
            "--precision",
            "fp16",
            "--mode",
            "hf-eager",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--workload-digest",
            "digest",
            "--output",
            str(output),
        ]
    )

    assert runner["run"](arguments) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert len(calls) == 3
    assert result["adapter"] == "hf-transformers-asr"
    assert result["timing_scope"] == "task-model-call-wall"
    assert result["input_preparation_included"] is False
    assert result["asset_loading_included"] is False
    assert result["model_load_included"] is False
    assert result["measurement_policy"] == {
        "timing_scope": "task-model-call-wall",
        "input_preparation_included": False,
        "asset_loading_included": False,
        "model_load_excluded": True,
        "warmup_excluded": True,
        "output_materialization_included": True,
    }
    assert result["measurement"] == {"warmup": 1, "iterations": 2}
    assert len(result["samples_ms"]) == 2
    assert result["reference_source"] == {
        "repository": "official",
        "revision": "source-revision",
    }


def test_vlm_adapter_routes_non_generic_families_to_owned_loaders() -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    deepseek = object()
    locateanything = object()
    globals_ = runner["_load_vlm"].__globals__
    globals_["_load_deepseek_ocr"] = lambda *_args: deepseek
    globals_["_load_locateanything"] = lambda *_args: locateanything

    assert runner["_load_vlm"](Namespace(family="deepseek_ocr"), {}, {}) is deepseek
    assert runner["_load_vlm"](Namespace(family="locateanything"), {}, {}) is locateanything


def test_locateanything_fallback_tokenizer_supports_batch_decode(
    tmp_path: Path, monkeypatch
) -> None:
    import tokenizers
    import transformers

    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        '{"model_max_length": 2048}', encoding="utf-8"
    )

    def fail_auto_tokenizer(*_args, **_kwargs):
        raise OSError("unsupported tokenizer class")

    class FakeRawTokenizer:
        def decode(self, token_ids, *, skip_special_tokens):
            suffix = "clean" if skip_special_tokens else "raw"
            return f"{','.join(str(token) for token in token_ids)}:{suffix}"

    auto_tokenizer = transformers.AutoTokenizer
    monkeypatch.setattr(
        tokenizers.Tokenizer,
        "from_file",
        lambda _path: FakeRawTokenizer(),
    )
    monkeypatch.setattr(auto_tokenizer, "from_pretrained", fail_auto_tokenizer)
    torch_module = Namespace(is_tensor=lambda _value: False)
    arguments = Namespace(
        local_files_only=True,
        model=str(tmp_path),
        revision=None,
        trust_remote_code=True,
    )

    tokenizer = runner["_locateanything_tokenizer"](arguments, torch_module)

    assert tokenizer.model_max_length == 2048
    assert tokenizer.batch_decode([[1, 2], [3]], skip_special_tokens=False) == [
        "1,2:raw",
        "3:raw",
    ]


def test_locateanything_tokenizer_retries_slow_backend_before_raw_json_fallback(
    monkeypatch,
) -> None:
    import transformers

    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    slow_tokenizer = object()
    calls: list[dict[str, object]] = []

    def load_tokenizer(_model: str, **kwargs):
        calls.append(kwargs)
        if kwargs.get("use_fast") is False:
            return slow_tokenizer
        raise OSError("tokenizer.json is not available")

    auto_tokenizer = transformers.AutoTokenizer
    fake_hub = ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = lambda **_kwargs: pytest.fail(
        "raw tokenizer.json fallback should not run when the slow tokenizer loads"
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setattr(auto_tokenizer, "from_pretrained", load_tokenizer)
    arguments = Namespace(
        local_files_only=True,
        model="nvidia/LocateAnything-3B",
        revision="model-revision",
        trust_remote_code=True,
    )

    tokenizer = runner["_locateanything_tokenizer"](
        arguments, Namespace(is_tensor=lambda _value: False)
    )

    assert tokenizer is slow_tokenizer
    assert len(calls) == 2
    assert calls[0].get("use_fast") is None
    assert calls[1]["use_fast"] is False


def test_locateanything_tokenizer_builds_qwen_bpe_when_tokenizer_json_is_absent(
    tmp_path: Path, monkeypatch
) -> None:
    import transformers

    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    (tmp_path / "vocab.json").write_text(
        json.dumps({"F": 0, "i": 1, "n": 2, "d": 3}), encoding="utf-8"
    )
    (tmp_path / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (tmp_path / "added_tokens.json").write_text(
        json.dumps({"<special>": 4}), encoding="utf-8"
    )
    (tmp_path / "tokenizer_config.json").write_text(
        '{"model_max_length": 512}', encoding="utf-8"
    )

    def fail_auto_tokenizer(*_args, **_kwargs):
        raise OSError("tokenizer.json is not available")

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", fail_auto_tokenizer
    )
    arguments = Namespace(
        local_files_only=True,
        model=str(tmp_path),
        revision=None,
        trust_remote_code=True,
    )

    tokenizer = runner["_locateanything_tokenizer"](
        arguments, Namespace(is_tensor=lambda _value: False)
    )

    assert tokenizer.model_max_length == 512
    assert tokenizer.encode("<special>Find") == [4, 0, 1, 2, 3]
    assert tokenizer.decode([0, 1, 2, 3]) == "Find"


def test_task_reference_resolves_revision_from_hugging_face_cache(monkeypatch) -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    revision = Namespace(commit_hash="abc123", refs=frozenset({"main"}))
    repository = Namespace(repo_id="nvidia/canary-1b-v2", revisions=[revision])
    cache = Namespace(repos=[repository])
    fake_hub = Namespace(scan_cache_dir=lambda: cache)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    assert runner["_cached_snapshot_revision"]("nvidia/canary-1b-v2", None) == "abc123"


def test_snapshot_revision_keeps_revision_from_symlink_path(tmp_path: Path) -> None:
    runner = runpy.run_path(
        str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py")
    )
    blob = tmp_path / "blobs" / "weights"
    blob.parent.mkdir()
    blob.write_bytes(b"weights")
    snapshot = tmp_path / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    checkpoint = snapshot / "model.safetensors"
    checkpoint.symlink_to(blob)

    assert runner["_snapshot_revision"](checkpoint) == "abc123"


def test_task_reference_pinned_checkout_scopes_safe_directory(
    tmp_path: Path, monkeypatch
) -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    captured: list[str] = []

    def fake_run(command, **_kwargs):
        captured.extend(str(value) for value in command)
        return subprocess.CompletedProcess(command, 0, "abc123\n", "")

    monkeypatch.setattr(runner["subprocess"], "run", fake_run)

    assert (
        runner["_pinned_checkout_revision"](
            str(tmp_path), "abc123", repository="official reference"
        )
        == "abc123"
    )
    assert captured[:4] == [
        "git",
        "-c",
        f"safe.directory={tmp_path.resolve()}",
        "-C",
    ]


def test_diffusers_local_mode_loads_resolved_snapshot_path(tmp_path: Path, monkeypatch) -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    snapshot = tmp_path / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            captured.update(model=model, kwargs=kwargs)
            return cls()

    monkeypatch.setitem(sys.modules, "diffusers", Namespace(FluxPipeline=FakePipeline))
    globals_ = runner["_diffusion_pipeline"].__globals__
    globals_["_cached_snapshot_path"] = lambda *_args: snapshot
    arguments = Namespace(
        family="flux",
        local_files_only=True,
        model="black-forest-labs/FLUX.1-schnell",
        precision="fp16",
        revision=None,
        trust_remote_code=False,
    )
    torch_module = Namespace(float16="fp16", float32="fp32", bfloat16="bf16")

    runner["_diffusion_pipeline"](arguments, torch_module, {})

    assert captured["model"] == snapshot
    assert captured["kwargs"]["local_files_only"] is True


def test_diffusers_adapter_uses_configured_pipeline_classes(monkeypatch) -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    selected = []

    class FluxPipeline:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            selected.append(cls.__name__)
            return cls()

    class Flux2Pipeline:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            selected.append(cls.__name__)
            return cls()

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        Namespace(FluxPipeline=FluxPipeline, Flux2Pipeline=Flux2Pipeline),
    )
    arguments = Namespace(
        family="flux",
        local_files_only=False,
        model="black-forest-labs/FLUX.2-dev",
        precision="fp16",
        revision=None,
        trust_remote_code=False,
    )
    torch_module = Namespace(float16="fp16", float32="fp32", bfloat16="bf16")

    runner["_diffusion_pipeline"](
        arguments,
        torch_module,
        {"pipeline_classes": ["Flux2Pipeline"]},
    )

    assert selected == ["Flux2Pipeline"]


def test_diffusers_adapter_selects_flux2_pipeline_from_model_id(monkeypatch) -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    selected = []

    class FluxPipeline:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            selected.append(cls.__name__)
            return cls()

    class Flux2Pipeline:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            selected.append(cls.__name__)
            return cls()

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        Namespace(FluxPipeline=FluxPipeline, Flux2Pipeline=Flux2Pipeline),
    )
    arguments = Namespace(
        family="flux",
        local_files_only=False,
        model="black-forest-labs/FLUX.2-dev",
        precision="fp16",
        revision=None,
        trust_remote_code=False,
    )
    torch_module = Namespace(float16="fp16", float32="fp32", bfloat16="bf16")

    runner["_diffusion_pipeline"](arguments, torch_module, {})

    assert selected == ["Flux2Pipeline"]


def test_wan22_diffusers_adapter_uses_pinned_conversion_and_fp32_vae(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    captured: dict[str, object] = {}

    class FakeVae:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            captured.update(vae_model=model, vae_kwargs=kwargs)
            return cls()

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            captured.update(pipeline_model=model, pipeline_kwargs=kwargs)
            return cls()

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        Namespace(
            AutoencoderKLWan=FakeVae,
            WanPipeline=FakePipeline,
            DiffusionPipeline=FakePipeline,
        ),
    )
    arguments = Namespace(
        family="wan2_2_ti2v",
        local_files_only=False,
        model="Wan-AI/Wan2.2-TI2V-5B",
        precision="bf16",
        revision="native-revision",
        trust_remote_code=False,
    )
    torch_module = Namespace(float16="fp16", float32="fp32", bfloat16="bf16")
    options = {
        "model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "model_revision": "diffusers-revision",
    }

    runner["_diffusion_pipeline"](arguments, torch_module, options)

    assert captured["vae_model"] == options["model_id"]
    assert captured["vae_kwargs"] == {
        "subfolder": "vae",
        "torch_dtype": "fp32",
        "revision": "diffusers-revision",
        "local_files_only": False,
    }
    assert captured["pipeline_model"] == options["model_id"]
    assert captured["pipeline_kwargs"]["torch_dtype"] == "bf16"
    assert captured["pipeline_kwargs"]["revision"] == "diffusers-revision"
    assert isinstance(captured["pipeline_kwargs"]["vae"], FakeVae)


def test_cached_snapshot_path_keeps_snapshot_parent_for_symlinked_marker(
    tmp_path: Path, monkeypatch
) -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    blob = tmp_path / "blobs" / "model-index"
    blob.parent.mkdir()
    blob.write_text("{}", encoding="utf-8")
    snapshot = tmp_path / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    marker = snapshot / "model_index.json"
    marker.symlink_to(blob)
    fake_hub = Namespace(try_to_load_from_cache=lambda **_kwargs: str(marker))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    assert runner["_cached_snapshot_path"]("org/model", None, "model_index.json") == snapshot


def test_diffusers_adapter_uses_resolved_sana_runtime_controls(tmp_path: Path) -> None:
    from PIL import Image

    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    image = tmp_path / "input.png"
    Image.new("RGB", (4, 4)).save(image)
    captured: dict[str, object] = {}

    class FakePipeline:
        def to(self, _device):
            return self

        def __call__(
            self,
            *,
            prompt,
            image,
            action,
            intrinsics,
            translation_speed,
            rotation_speed_deg,
            num_frames,
            fps,
            flow_shift,
            step,
            cfg_scale,
        ):
            captured.update(locals())
            return Namespace(frames=[[object(), object()]])

    def fake_pipeline(_arguments, _torch, options):
        assert options["trust_remote_code"] is True
        return FakePipeline()

    globals_ = runner["_load_diffusers"].__globals__
    globals_["_diffusion_pipeline"] = fake_pipeline
    globals_["_resolved_revision"] = lambda *_args: "snapshot"
    arguments = Namespace(
        family="sana_wm",
        precision="bf16",
        resolved_runtime={
            "config": {
                "sana_wm.action": "w-80,jw-40",
                "sana_wm.intrinsics": "1,2,3,4",
                "sana_wm.translation_speed": 0.055,
                "sana_wm.rotation_speed_deg": 1.2,
                "sana_wm.num_frames": 321,
                "sana_wm.fps": 16,
                "sana_wm.flow_shift": 9.8,
            }
        },
    )
    request = {
        "prompt": "A stationary camera.",
        "image_path": str(image),
        "media_type": "video",
        "num_inference_steps": 60,
        "cfg_scale": 5.0,
        "video_num_frames": 321,
    }

    session = runner["_load_diffusers"](
        arguments,
        request,
        {
            "trust_remote_code": True,
            "required_call_arguments": [
                "image",
                "action",
                "intrinsics",
                "translation_speed",
                "rotation_speed_deg",
                "num_frames",
            ],
        },
    )
    summary = session.invoke()

    assert captured["action"] == "w-80,jw-40"
    assert captured["intrinsics"] == "1,2,3,4"
    assert captured["translation_speed"] == 0.055
    assert captured["rotation_speed_deg"] == 1.2
    assert captured["num_frames"] == 321
    assert captured["step"] == 60
    assert summary == {
        "media_type": "video",
        "media_count": 2,
        "height": None,
        "width": None,
        "channels": None,
    }


def test_diffusers_adapter_preserves_batched_prompts_and_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    captured = []

    class FakeGenerator:
        def __init__(self, device):
            assert device == "cuda"
            self.seed = None

        def manual_seed(self, seed):
            self.seed = seed
            return self

    class FakePipeline:
        def to(self, device):
            assert device == "cuda"
            return self

        def __call__(self, *, prompt, generator):
            captured.append(
                {
                    "prompt": prompt,
                    "seeds": [value.seed for value in generator],
                }
            )
            return Namespace(images=[object(), object()])

    globals_ = runner["_load_diffusers"].__globals__
    globals_["_diffusion_pipeline"] = lambda *_args: FakePipeline()
    globals_["_resolved_revision"] = lambda *_args: "snapshot"
    monkeypatch.setitem(sys.modules, "torch", Namespace(Generator=FakeGenerator))
    arguments = Namespace(
        family="flux",
        precision="fp16",
        model="black-forest-labs/FLUX.1-schnell",
        revision=None,
    )

    session = runner["_load_diffusers"](
        arguments,
        {
            "batch_size": 2,
            "prompt": "unused",
            "prompts": ["red cube", "blue sphere"],
            "seed": 0,
            "seeds": [41, 42],
            "media_type": "image",
        },
        {},
    )

    assert session.invoke()["media_count"] == 2
    assert session.invoke()["media_count"] == 2
    assert captured == [
        {"prompt": ["red cube", "blue sphere"], "seeds": [41, 42]},
        {"prompt": ["red cube", "blue sphere"], "seeds": [41, 42]},
    ]


def test_diffusers_media_count_accepts_array_like_video_frames() -> None:
    runner = runpy.run_path(
        str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py")
    )

    class Frames:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return [object()] * 5

        def __bool__(self):
            raise ValueError("array truth value is ambiguous")

    assert runner["_media_count"](Frames(), "video") == 5
    assert runner["_media_count"](Frames(), "image") == 1


def test_personaplex_loader_adds_vendored_moshi_package_root() -> None:
    source = (REPOSITORY / "benchmarks/performance/baselines/task_reference.py").read_text()

    assert 'str(Path(official_repo) / "moshi")' in source


@pytest.mark.parametrize(
    ("architecture", "output_name"),
    [
        ("PatchTSTForRegression", "regression_outputs"),
        ("PatchTSTForPrediction", "prediction_outputs"),
        ("PatchTSTForClassification", "prediction_logits"),
    ],
)
def test_patchtst_reference_runs_model_under_precision_autocast(
    monkeypatch,
    architecture: str,
    output_name: str,
) -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    selected = []

    class FakeTensor:
        shape = (1, 1)

        def reshape(self, *_shape):
            return self

        def gt(self, _value):
            return self

        def numel(self):
            return 1

        def isfinite(self):
            return self

        def all(self):
            return self

        def item(self):
            return True

    class Context:
        def __init__(self, torch_module, *, autocast=False):
            self.torch_module = torch_module
            self.autocast = autocast

        def __enter__(self):
            if self.autocast:
                self.torch_module.autocast_active = True

        def __exit__(self, *_args):
            if self.autocast:
                self.torch_module.autocast_active = False

    fake_torch = ModuleType("torch")
    fake_torch.float16 = "fp16"
    fake_torch.float32 = "fp32"
    fake_torch.bfloat16 = "bf16"
    fake_torch.autocast_active = False
    fake_torch.device = lambda value: value
    fake_torch.tensor = lambda *_args, **_kwargs: FakeTensor()
    fake_torch.inference_mode = lambda: Context(fake_torch)
    fake_torch.autocast = lambda *_args, **_kwargs: Context(fake_torch, autocast=True)
    fake_torch.stack = lambda values, dim: FakeTensor()

    class FakePatchTST:
        config = Namespace(_commit_hash="snapshot")
        architecture = ""

        @classmethod
        def from_pretrained(cls, _model, **kwargs):
            assert kwargs["torch_dtype"] == "fp16"
            selected.append(cls.architecture)
            return cls()

        def eval(self):
            return self

        def to(self, _device):
            return self

        def __call__(self, **_kwargs):
            assert fake_torch.autocast_active
            return Namespace(**{output_name: FakeTensor()})

    class FakePatchTSTForRegression(FakePatchTST):
        architecture = "PatchTSTForRegression"

    class FakePatchTSTForPrediction(FakePatchTST):
        architecture = "PatchTSTForPrediction"

    class FakePatchTSTForClassification(FakePatchTST):
        architecture = "PatchTSTForClassification"

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoConfig = Namespace(
        from_pretrained=lambda *_args, **_kwargs: Namespace(
            architectures=[architecture],
            context_length=2,
            num_input_channels=1,
        )
    )
    fake_transformers.PatchTSTForRegression = FakePatchTSTForRegression
    fake_transformers.PatchTSTForPrediction = FakePatchTSTForPrediction
    fake_transformers.PatchTSTForClassification = FakePatchTSTForClassification
    fake_transformers.PatchTSMixerForPrediction = object
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    session = runner["_load_timeseries"](
        Namespace(
            family="patchtst",
            local_files_only=True,
            model="ibm/patchtst",
            precision="fp16",
            revision=None,
            trust_remote_code=False,
        ),
        {"field_input": [1.0, 2.0]},
        {},
    )

    assert session.invoke() == {"shape": [1, 1], "element_count": 1, "finite": True}
    assert selected == [architecture]


def test_qwen3_omni_supplies_text_chat_template_when_snapshot_omits_it() -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    captured: dict[str, object] = {}

    class FakeProcessor:
        chat_template = None
        tokenizer = Namespace(chat_template=None)

        def apply_chat_template(self, conversation, **kwargs):
            captured.update(conversation=conversation, kwargs=kwargs)
            return "inputs"

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": "system"}]},
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
    ]

    assert runner["_qwen3_omni_chat_inputs"](FakeProcessor(), conversation) == "inputs"
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    template = kwargs["chat_template"]
    assert isinstance(template, str)
    assert "<|im_start|>" in template
    assert "<|im_end|>" in template
    assert kwargs["add_generation_prompt"] is True
    assert kwargs["tokenize"] is True
    assert kwargs["return_tensors"] == "pt"


def test_qwen3_omni_uses_installed_generation_speaker_argument() -> None:
    source = (
        REPOSITORY / "benchmarks/performance/baselines/task_reference.py"
    ).read_text(encoding="utf-8")

    assert 'speaker=str(options.get("speaker", "Ethan"))' in source
    assert 'spk=str(options.get("speaker", "Ethan"))' not in source


def test_qwen3_omni_uses_visible_single_gpu_placement(monkeypatch) -> None:
    runner = runpy.run_path(
        str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py")
    )
    captured: dict[str, object] = {}

    class FakeInputs:
        def to(self, device):
            captured["input_device"] = device
            return self

    class FakeProcessor:
        chat_template = "{{ messages }}"
        tokenizer = Namespace(chat_template=None)

        @classmethod
        def from_pretrained(cls, _model, **_kwargs):
            return cls()

        def apply_chat_template(self, _conversation, **_kwargs):
            return FakeInputs()

    class FakeModel:
        config = Namespace(_commit_hash="snapshot")
        device = "cuda:0"

        @classmethod
        def from_pretrained(cls, _model, **kwargs):
            captured["load_options"] = kwargs
            return cls()

        def eval(self):
            return self

    fake_torch = ModuleType("torch")
    fake_torch.float16 = "fp16"
    fake_torch.float32 = "fp32"
    fake_torch.bfloat16 = "bf16"
    fake_transformers = ModuleType("transformers")
    fake_transformers.Qwen3OmniMoeForConditionalGeneration = FakeModel
    fake_transformers.Qwen3OmniMoeProcessor = FakeProcessor
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    cases = perf_matrix._cases(perf_matrix._read_yaml(SUITE))
    case = next(case for case in cases if case["id"] == "qwen3_omni.generate_audio")
    options = case["baseline"]["adapter_options"]
    runner["_load_qwen3_omni"](
        Namespace(
            local_files_only=True,
            model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
            precision="bf16",
            revision=None,
            trust_remote_code=True,
        ),
        {"prompt": "hello", "max_new_tokens": 16},
        options,
    )

    assert options["device_map"] == "cuda:0"
    assert captured["load_options"]["device_map"] == "cuda:0"
    assert captured["input_device"] == "cuda:0"


def test_lance_reference_builds_repeated_official_x2t_dataset(tmp_path: Path) -> None:
    runner = runpy.run_path(str(REPOSITORY / "tools/lance_reference.py"))
    image = tmp_path / "input.png"
    image.write_bytes(b"image")

    payload = runner["_dataset_payload"](
        image=image,
        prompt="What color is the vehicle?",
        instruction="Inspect the image.",
        count=3,
    )

    assert list(payload) == ["0000", "0001", "0002"]
    assert payload["0000"] == {
        "interleave_array": [
            str(image.resolve()),
            ["Inspect the image.", "What color is the vehicle?", ""],
        ],
        "element_dtype_array": ["image", "text"],
        "istarget_in_interleave": [0, 1],
    }


def test_lance_image_only_decord_stub_rejects_video_use() -> None:
    runner = runpy.run_path(str(REPOSITORY / "tools/lance_reference.py"))
    modules = runner["_decord_image_only_stub"]()

    assert modules["decord"].VideoReader is modules["decord.video_reader"].VideoReader
    assert modules["decord"].__spec__.name == "decord"
    with pytest.raises(RuntimeError, match="video workloads require"):
        modules["decord"].VideoReader("video.mp4")


def test_lance_git_revision_scopes_safe_directory(tmp_path: Path, monkeypatch) -> None:
    runner = runpy.run_path(str(REPOSITORY / "tools/lance_reference.py"))
    captured: list[str] = []

    def fake_run(command, **_kwargs):
        captured.extend(str(value) for value in command)
        return subprocess.CompletedProcess(command, 0, "abc123\n", "")

    monkeypatch.setattr(runner["subprocess"], "run", fake_run)

    assert runner["_git_revision"](tmp_path) == "abc123"
    assert captured[:4] == [
        "git",
        "-c",
        f"safe.directory={tmp_path.resolve()}",
        "-C",
    ]


def test_lance_reference_loads_once_then_measures_each_dataset_row(
    tmp_path: Path, monkeypatch
) -> None:
    import torch

    runner = runpy.run_path(str(REPOSITORY / "tools/lance_reference.py"))
    reference_repo = tmp_path / "Lance"
    reference_repo.mkdir()
    (reference_repo / "inference_lance.py").write_text(
        """
import argparse
import json
from types import SimpleNamespace

MAX_GENERATION_LENGTH = 256

def normalize_understanding_answer(value):
    return value.replace("<|im_end|>", "").strip()

def validate_on_fixed_batch(*, inference_args, sample_id):
    inference_args.prompt_data_dict[sample_id] = "blue<|im_end|>"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dataset_config_file")
    arguments, _ = parser.parse_known_args()
    rows = json.loads(open(arguments.val_dataset_config_file, encoding="utf-8").read())
    state = SimpleNamespace(prompt_data_dict={})
    for sample_id in rows:
        validate_on_fixed_batch(inference_args=state, sample_id=sample_id)
""",
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset.json"
    dataset.write_text('{"0000":{},"0001":{},"0002":{}}', encoding="utf-8")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    configured = []
    runner["_run_upstream"].__globals__["_configure_upstream_vae"] = lambda repo, vae: (
        configured.append((repo, vae))
    )
    arguments = Namespace(
        reference_repo=reference_repo,
        max_new_tokens=10,
        warmup=1,
        iterations=2,
        height=768,
        width=768,
        resolution="image_768res",
    )

    samples, answers = runner["_run_upstream"](
        arguments,
        tmp_path / "Lance_3B",
        tmp_path / "Qwen2.5-VL-ViT",
        tmp_path / "Wan2.2_VAE.pth",
        dataset,
        tmp_path / "results",
    )

    assert len(samples) == 2
    assert all(value > 0 for value in samples)
    assert answers == ["blue", "blue"]
    assert configured == [(reference_repo, tmp_path / "Wan2.2_VAE.pth")]


def test_lance_reference_requires_the_upstream_vae(tmp_path: Path) -> None:
    runner = runpy.run_path(str(REPOSITORY / "tools/lance_reference.py"))
    root = tmp_path / "Lance"
    model = root / "Lance_3B"
    vit = root / "Qwen2.5-VL-ViT"
    model.mkdir(parents=True)
    vit.mkdir()
    (model / "llm_config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"model")
    (vit / "vit.safetensors").write_bytes(b"vit")
    arguments = Namespace(
        model=str(root),
        revision=None,
        local_files_only=True,
        model_subdir="Lance_3B",
        vit_subdir="Qwen2.5-VL-ViT",
    )

    with pytest.raises(FileNotFoundError, match="Wan2.2_VAE.pth"):
        runner["_model_paths"](arguments)

    vae = root / "Wan2.2_VAE.pth"
    vae.write_bytes(b"vae")

    assert runner["_model_paths"](arguments) == (
        model,
        vit,
        vae,
        "local-path",
    )


def test_lance_adapter_records_pinned_upstream_and_model_revisions(
    tmp_path: Path, monkeypatch
) -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    image = tmp_path / "input.png"
    image.write_bytes(b"image")
    reference_repo = tmp_path / "Lance"
    reference_repo.mkdir()
    captured: list[list[str]] = []

    def fake_run(command, **_kwargs):
        command = [str(value) for value in command]
        captured.append(command)
        output = Path(command[command.index("--output") + 1])
        output.write_text(
            json.dumps(
                {
                    "samples_ms": [11.0, 12.0],
                    "text": "blue",
                    "model_revision": "hf-snapshot",
                    "reference_revision": "upstream-commit",
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner["subprocess"], "run", fake_run)
    arguments = Namespace(
        precision="bf16",
        model="bytedance-research/Lance",
        manifest=SUITE,
        family="lance",
        warmup=1,
        iterations=2,
        revision=None,
        local_files_only=False,
    )

    result = runner["_run_lance"](
        arguments,
        {
            "image_path": str(image),
            "prompt": "What color is the vehicle?",
            "max_new_tokens": 10,
        },
        {
            "reference_repo": str(reference_repo),
            "reference_commit": "upstream-commit",
        },
    )

    assert result[:6] == (
        [11.0, 12.0],
        {"text": "blue", "output_tokens": None},
        "hf-snapshot",
        "lance-pytorch",
        "task-pipeline-call-wall",
        True,
    )
    assert result[6]["revision"] == "upstream-commit"
    assert str(REPOSITORY / "tools/lance_reference.py") in captured[0]
    assert captured[0][captured[0].index("--max-new-tokens") + 1] == "10"


def test_sana_wm_adapter_runs_pinned_official_pipeline_with_resolved_inputs(
    tmp_path: Path, monkeypatch
) -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    model_root = tmp_path / "sana_wm"
    manifest = model_root / "manifests" / "model.json"
    assets = model_root / "assets"
    manifest.parent.mkdir(parents=True)
    assets.mkdir()
    manifest.write_text("{}", encoding="utf-8")
    (assets / "image.png").write_bytes(b"image")
    (assets / "prompt.txt").write_text("prompt", encoding="utf-8")
    (assets / "intrinsics.npy").write_bytes(b"intrinsics")
    reference_repo = tmp_path / "Sana"
    reference_repo.mkdir()
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        command = [str(value) for value in command]
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0, "pinned-commit\n", "")
        captured.update(command=command, kwargs=kwargs)
        output = Path(kwargs["env"]["TRTMC_SANA_WM_BENCHMARK_OUTPUT"])
        output.write_text(
            json.dumps(
                {
                    "samples_ms": [101.0, 102.0],
                    "output_summary": {"frame_count": 321, "shape": [321, 704, 1280, 3]},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner["subprocess"], "run", fake_run)
    arguments = Namespace(
        manifest=manifest,
        resolved_runtime={
            "config": {
                "sana_wm.action": "w-80,jw-40",
                "sana_wm.translation_speed": 0.055,
                "sana_wm.rotation_speed_deg": 1.2,
            }
        },
        warmup=1,
        iterations=2,
    )

    result = runner["_run_sana_wm"](
        arguments,
        {
            "image_path": "assets/image.png",
            "prompt_path": "assets/prompt.txt",
            "video_num_frames": 321,
            "fps": 16,
            "num_inference_steps": 60,
            "cfg_scale": 5.0,
            "flow_shift": 9.8,
            "seed": 42,
        },
        {
            "reference_repo": str(reference_repo),
            "reference_commit": "pinned-commit",
            "intrinsics": "assets/intrinsics.npy",
        },
    )

    assert result[:6] == (
        [101.0, 102.0],
        {
            "frame_count": 321,
            "shape": [321, 704, 1280, 3],
            "media_count": 321,
            "height": 704,
            "width": 1280,
            "channels": 3,
        },
        "pinned-commit",
        "sana-wm-pytorch",
        "task-pipeline-call-wall",
        True,
    )
    command = captured["command"]
    assert command[command.index("--num_frames") + 1] == "321"
    assert command[command.index("--step") + 1] == "60"
    assert captured["kwargs"]["cwd"] == str(reference_repo)
    assert captured["kwargs"]["env"]["TRTMC_SANA_WM_BENCHMARK_WARMUP"] == "1"
    assert captured["kwargs"]["env"]["TRTMC_SANA_WM_BENCHMARK_ITERATIONS"] == "2"
