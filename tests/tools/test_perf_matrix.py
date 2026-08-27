# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import json
import os
from argparse import Namespace
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

from tools import perf_matrix
from tools.performance import catalog as performance_catalog


REPOSITORY = Path(__file__).resolve().parents[2]
SUITE = REPOSITORY / "benchmarks/performance/release.yaml"
GB300_ENVIRONMENT = REPOSITORY / "benchmarks/performance/environments/gb300.yaml"
L4T_THOR_ENVIRONMENT = REPOSITORY / "benchmarks/performance/environments/l4t-thor.yaml"
AUTO_THOR_ENVIRONMENT = REPOSITORY / "benchmarks/performance/environments/auto-thor.yaml"


def _suite_for_cases(cases, *, exclusions=None):
    return performance_catalog.PerformanceSuite(
        source=SUITE,
        definition={},
        cases=tuple(cases),
        excluded_profiles=dict(exclusions or {}),
    )


MINIMAX_H3_EXCLUSION_REASON = (
    "The pinned Diffusers reference for MiniMax-H3 has not yet been integrated "
    "into the release performance runner."
)
LFM2_EXCLUSION_REASON = (
    "Dense LFM2 functional and reference-parity qualification is present, but "
    "this change does not add a matching release-performance workload or receipt."
)
TASK_ADAPTERS = {
    "bark.generate_audio": "hf-transformers-tts",
    "canary.transcribe": "nemo-asr",
    "chronos_bolt.solve": "pytorch-timeseries",
    "deepseek_ocr.generate": "hf-transformers-vlm",
    "dinov3.extract_features": "hf-transformers-vision",
    "eagle_vlm.embed": "hf-transformers-embedding",
    "eagle_vlm.rerank": "hf-transformers-reranking",
    "fast_foundation_stereo.disparity": "upstream-fast-foundation-stereo",
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
 'bundle_name':'distilgpt2.bundle', 'bundle_path':'/tmp/distilgpt2.bundle',
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
          'manifest_path':{str(manifest)!r},'manifest_sha256':'fake','bundle_name':'distilgpt2.bundle',
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
                        REPOSITORY / "benchmarks/performance/baselines/task_reference.py"
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

    suite = performance_catalog.load_suite(SUITE)
    cases = list(suite.cases)
    raw_suite = yaml.safe_load(SUITE.read_text(encoding="utf-8"))
    raw_entries = raw_suite["entries"]
    raw_additional = raw_suite["additional_profiles"]
    excluded_profiles = suite.excluded_profiles
    ready_profiles = {
        entry.name
        for entry in perf_matrix.ManifestCatalog().entries()
        if entry.status == "ready" and not performance_catalog.is_l0_profile(entry.name)
    }

    performance_catalog.validate_release_coverage(cases, excluded_profiles)

    assert len(cases) == 108
    assert len(raw_entries) == 79
    assert len(raw_additional) == 29
    assert excluded_profiles == {
        "lfm2-1.2b": LFM2_EXCLUSION_REASON,
        "lfm2-2.6b": LFM2_EXCLUSION_REASON,
        "lfm2-350m-bf16-model-card": LFM2_EXCLUSION_REASON,
        "lfm2-350m-fp16": LFM2_EXCLUSION_REASON,
        "lfm2-700m": LFM2_EXCLUSION_REASON,
        "minimax-h3-768p": MINIMAX_H3_EXCLUSION_REASON,
    }
    assert all(
        set(entry["workload"]) <= {"testcase", "request", "runtime"} for entry in raw_entries
    )
    assert all(entry["workload"].get("testcase") for entry in raw_entries)
    assert all(entry.get("model") and entry.get("inherit") for entry in raw_additional)
    assert not any("priority" in entry for entry in raw_entries)
    assert {case["model"] for case in cases} == ready_profiles - set(excluded_profiles)
    assert not any(performance_catalog.is_l0_profile(case["model"]) for case in cases)
    assert len({(case["family"], case["operation"]) for case in cases}) == 79
    assert len({case["family"] for case in cases}) == 78
    assert [case["operation"] for case in cases if case["family"] == "eagle_vlm"] == [
        "embed",
        "rerank",
    ]
    assert Counter(perf_matrix._candidate_timing_scope(case) for case in cases) == {
        "model_call_wall": 24,
        "public_pipeline_call_wall": 84,
    }
    assert {case["id"] for case in cases if case["baseline"]["asset_loading_included"]} == {
        "canary.transcribe",
        "deepseek_ocr.generate",
        "lance.generate",
        "nemotron_speech_streaming.transcribe",
        "nemotron_speech_streaming.transcribe@nemotron-speech-streaming-en-0.6b",
    }
    by_id = {case["id"]: case for case in cases}
    assert by_id["deberta.encode"]["baseline"]["precision"] == "fp32"
    assert by_id["eagle_vlm.rerank"]["baseline"]["precision"] == "fp16"
    assert by_id["eagle_vlm.rerank"]["baseline"]["local_files_only"] is True
    assert by_id["eagle_vlm.rerank"]["baseline"]["output_contract"] == "reranking-order"
    assert by_id["fnet.encode"]["baseline"]["padding"] == "max-length"
    assert by_id["lance.generate"]["baseline"]["python_profile"] == "lance_reference"
    assert by_id["sana_wm.generate_image"]["baseline"]["adapter_options"] == {
        "reference_commit": "59629fdf790850797cb657bad014fce432bd713d",
        "intrinsics": "assets/demo_0_intrinsics.npy",
    }
    assert by_id["sana_wm.generate_image"]["baseline"]["python_profile"] == "sana_wm_reference"
    locateanything_baseline = by_id["locateanything.generate"]["baseline"]
    assert locateanything_baseline["output_contract"] == "localization"
    assert locateanything_baseline["min_localization_box_iou"] == 0.9
    assert locateanything_baseline["max_localization_point_distance"] == 10.0
    assert locateanything_baseline["max_normalized_edit_distance"] == 0.5
    assert by_id["mixtral.generate"]["baseline"]["experts_implementation"] == "batched_mm"
    assert by_id["phi_moe.generate"]["baseline"]["experts_implementation"] == "batched_mm"
    assert by_id["qwen_moe.generate"]["model"] == "qwen3-moe-30b-a3b"
    assert by_id["qwen_moe.generate"]["baseline"]["experts_implementation"] == "batched_mm"
    assert by_id["sam3.segment_prompted"]["baseline"]["local_files_only"] is True
    assert by_id["phi_moe.generate"]["baseline"]["output_contract"] == "exact-text"
    assert by_id["opt.generate"]["workload"]["request"]["max_new_tokens"] == 10
    assert by_id["deepseek_ocr.generate"]["baseline"]["precision"] == "bf16"
    assert by_id["llama.generate@minitron-4b-width"]["baseline"]["precision"] == "fp16"
    assert by_id["nemotron.generate@nemotron-mini-4b"]["workload"]["runtime"] == {
        "prefer_gpu_greedy": True
    }
    assert by_id["nemotron.generate@nemotron-mini-4b"]["baseline"]["precision"] == "fp16"
    assert by_id["nemotron_h.generate"]["baseline"]["mode"] == "hf-eager"
    assert by_id["nemotron_h.generate"]["workload"]["runtime"] == {"cuda_graphs": True}
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
    assert by_id["bark.generate_audio"]["workload"]["request"]["max_new_tokens"] == 128
    assert by_id["magpie_tts.generate_audio"]["workload"]["request"]["max_new_tokens"] == 256
    for case_id in (
        "bark.generate_audio",
        "magpie_tts.generate_audio",
        "personaplex.speak",
        "qwen3_omni.generate_audio",
    ):
        assert by_id[case_id]["baseline"]["output_contract"] == "audio-shape"
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


def test_release_suite_rejects_unknown_explicit_exclusion() -> None:
    suite = performance_catalog.load_suite(SUITE)
    cases = list(suite.cases)
    exclusions = {
        **suite.excluded_profiles,
        "unknown-profile": "Invalid test exclusion.",
    }

    with pytest.raises(
        performance_catalog.PerformanceSuiteError,
        match="invalid-exclusion=unknown-profile",
    ):
        performance_catalog.validate_release_coverage(cases, exclusions)


def test_release_suite_rejects_a_configured_explicit_exclusion() -> None:
    suite = performance_catalog.load_suite(SUITE)
    cases = list(suite.cases)

    with pytest.raises(
        performance_catalog.PerformanceSuiteError,
        match="excluded-and-configured=qwen3-moe-30b-a3b",
    ):
        performance_catalog.validate_release_coverage(
            cases,
            {"qwen3-moe-30b-a3b": "Temporary test exclusion."},
        )


def test_checked_in_gb300_environment_is_ci_runnable() -> None:
    raw = yaml.safe_load(GB300_ENVIRONMENT.read_text(encoding="utf-8"))

    assert raw["schema_version"] == "trtmc.perf-environment/v1"
    assert raw["tools"]["trtmc_bench"] == "scripts/trtmc-bench"
    assert raw["tools"]["trtmc_worker"] == "${TRTMC_PERF_WORKER}"
    assert raw["storage"]["results_root"] == "artifacts/perf"
    assert raw["storage"]["bundle_cache"] == "${TRTMC_PERF_BUNDLE_CACHE}"
    assert raw["storage"]["bundle_roots"] == "${TRTMC_PERF_BUNDLE_ROOTS}"
    assert raw["storage"]["runtime_dirs"] == "${TRTMC_PERF_RUNTIME_DIRS}"
    assert raw["storage"]["bundle_retention"] == "delete_always"
    assert "minimum_free_space_gib" not in raw["storage"]
    assert raw["execution"]["hf_cache_mode"] == "shared"
    assert raw["execution"]["hf_cache_retention"] == "retain"
    assert raw["execution"]["minimum_gpu_free_fraction"] == 0.0
    assert raw["execution"]["timeout_seconds"] == 7200


def test_checked_in_l4t_environment_bounds_storage_and_cleanup() -> None:
    raw = yaml.safe_load(L4T_THOR_ENVIRONMENT.read_text(encoding="utf-8"))

    assert raw["storage"]["storage_root"] == "${TRTMC_CHECK_STORAGE_ROOT}"
    assert raw["storage"]["bundle_retention"] == "delete_always"
    assert "minimum_free_space_gib" not in raw["storage"]
    assert raw["execution"]["hf_cache_mode"] == "per_entry"
    assert raw["execution"]["hf_cache_retention"] == "delete_always"


def test_checked_in_auto_thor_environment_deletes_managed_bundles() -> None:
    raw = yaml.safe_load(AUTO_THOR_ENVIRONMENT.read_text(encoding="utf-8"))

    assert raw["storage"]["bundle_retention"] == "delete_always"
    assert "minimum_free_space_gib" not in raw["storage"]
    assert raw["execution"]["hf_cache_mode"] == "shared"
    assert raw["execution"]["hf_cache_retention"] == "retain"


def test_environment_rejects_deleting_a_shared_hf_cache(tmp_path: Path) -> None:
    environment_path = tmp_path / "environment.yaml"
    _write_environment(
        environment_path,
        results_root=tmp_path / "results",
        scratch_root=tmp_path / "scratch",
        trtmc_bench=tmp_path / "trtmc-bench",
        trtmc_worker=tmp_path / "worker",
        hf_transformers_runner=tmp_path / "hf.py",
    )
    raw = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    raw["execution"]["hf_cache_mode"] = "shared"
    raw["execution"]["hf_cache_retention"] = "delete_always"
    environment_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(perf_matrix.PerfMatrixError, match="shared Hugging Face"):
        perf_matrix._read_environment(environment_path)


@pytest.mark.parametrize(
    ("policy", "passed"),
    (("delete_on_pass", True), ("delete_always", False)),
)
def test_perf_bundle_cleanup_deletes_only_engine_and_retains_build_artifacts(
    tmp_path: Path,
    policy: str,
    passed: bool,
) -> None:
    cache = tmp_path / "bundles"
    bundle = cache / "model-a" / "fingerprint" / "model-a.bundle"
    bundle.parent.mkdir(parents=True)
    bundle.write_text("engine", encoding="utf-8")
    stdout_log = bundle.parent / "build.stdout.log"
    stderr_log = bundle.parent / "build.stderr.log"
    timing = bundle.parent / "build-timing.json"
    stdout_log.write_text("stdout evidence", encoding="utf-8")
    stderr_log.write_text("stderr evidence", encoding="utf-8")
    timing.write_text('{"build_seconds": 1}', encoding="utf-8")
    options = perf_matrix.RunOptions(
        output=tmp_path / "results",
        scratch_root=tmp_path / "scratch",
        trtmc_bench="trtmc-bench",
        trtmc_worker=None,
        hf_transformers_runner=tmp_path / "hf.py",
        task_reference_runner=tmp_path / "task.py",
        bundle_cache=cache,
        bundle_roots=(),
        runtime_dirs=(),
        local_files_only=False,
        minimum_free_space_gib=0,
        minimum_gpu_free_fraction=0.0,
        timeout_seconds=1,
        bundle_retention=policy,
    )

    deleted = perf_matrix._cleanup_managed_bundle(
        {"bundle_path": str(bundle)},
        options,
        passed=passed,
    )
    assert deleted["status"] == "deleted"
    assert deleted["scope"] == "bundle_only"
    assert not bundle.exists()
    assert bundle.parent.is_dir()
    assert stdout_log.read_text(encoding="utf-8") == "stdout evidence"
    assert stderr_log.read_text(encoding="utf-8") == "stderr evidence"
    assert timing.read_text(encoding="utf-8") == '{"build_seconds": 1}'


def test_perf_bundle_delete_on_pass_retains_failure(tmp_path: Path) -> None:
    cache = tmp_path / "bundles"
    bundle = cache / "model-a" / "fingerprint" / "model-a.bundle"
    bundle.parent.mkdir(parents=True)
    bundle.write_text("engine", encoding="utf-8")
    options = perf_matrix.RunOptions(
        output=tmp_path / "results",
        scratch_root=tmp_path / "scratch",
        trtmc_bench="trtmc-bench",
        trtmc_worker=None,
        hf_transformers_runner=tmp_path / "hf.py",
        task_reference_runner=tmp_path / "task.py",
        bundle_cache=cache,
        bundle_roots=(),
        runtime_dirs=(),
        local_files_only=False,
        minimum_free_space_gib=0,
        minimum_gpu_free_fraction=0.0,
        timeout_seconds=1,
        bundle_retention="delete_on_pass",
    )

    retained = perf_matrix._cleanup_managed_bundle(
        {"bundle_path": str(bundle)},
        options,
        passed=False,
    )
    assert retained["status"] == "retained"
    assert bundle.is_file()


def test_perf_per_entry_hf_cache_can_retain_failure_and_delete_success(
    tmp_path: Path,
) -> None:
    options = perf_matrix.RunOptions(
        output=tmp_path / "results",
        scratch_root=tmp_path / "scratch",
        trtmc_bench="trtmc-bench",
        trtmc_worker=None,
        hf_transformers_runner=tmp_path / "hf.py",
        task_reference_runner=tmp_path / "task.py",
        bundle_cache=None,
        bundle_roots=(),
        runtime_dirs=(),
        local_files_only=False,
        minimum_free_space_gib=0,
        minimum_gpu_free_fraction=0.0,
        timeout_seconds=1,
        hf_cache_mode="per_entry",
        hf_cache_retention="delete_on_pass",
    )
    case_work = tmp_path / "scratch/case"
    (case_work / "hf-cache").mkdir(parents=True)

    retained = perf_matrix._cleanup_entry_work(case_work, options, passed=False)
    assert retained["status"] == "retained"
    assert case_work.is_dir()

    deleted = perf_matrix._cleanup_entry_work(case_work, options, passed=True)
    assert deleted["status"] == "deleted"
    assert not case_work.exists()


def test_perf_shared_hf_cache_retains_failed_entry_work(tmp_path: Path) -> None:
    options = perf_matrix.RunOptions(
        output=tmp_path / "results",
        scratch_root=tmp_path / "scratch",
        trtmc_bench="trtmc-bench",
        trtmc_worker=None,
        hf_transformers_runner=tmp_path / "hf.py",
        task_reference_runner=tmp_path / "task.py",
        bundle_cache=None,
        bundle_roots=(),
        runtime_dirs=(),
        local_files_only=False,
        minimum_free_space_gib=0,
        minimum_gpu_free_fraction=0.0,
        timeout_seconds=1,
        hf_cache_mode="shared",
        hf_cache_retention="retain",
    )
    case_work = tmp_path / "scratch/case"
    case_work.mkdir(parents=True)

    retained = perf_matrix._cleanup_entry_work(case_work, options, passed=False)
    assert retained["status"] == "retained"
    assert case_work.is_dir()

    deleted = perf_matrix._cleanup_entry_work(case_work, options, passed=True)
    assert deleted["status"] == "deleted"
    assert not case_work.exists()


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


def test_timing_stability_accepts_a_settled_measurement() -> None:
    stability = perf_matrix._timing_stability(
        [100.0, 101.0, 99.0, 100.0, 100.0, 101.0, 100.0, 99.0, 100.0, 100.0]
    )

    assert stability["status"] == "stable"
    assert stability["sample_count"] == 10
    assert stability["samples_within_band"] == 10


def test_timing_stability_flags_a_measurement_that_is_still_falling() -> None:
    stability = perf_matrix._timing_stability([3.7, 3.4, 3.0, 2.7, 2.3, 1.9, 1.6, 1.4, 1.2, 1.0])

    assert stability["status"] == "unstable"
    assert stability["half_median_change_percent"] > 5.0


def test_timing_stability_rejects_scattered_samples_even_without_drift() -> None:
    stability = perf_matrix._timing_stability(
        [100.0, 80.0, 120.0, 100.0, 100.0, 100.0, 80.0, 120.0, 100.0, 100.0]
    )

    assert stability["half_median_change_percent"] == 0.0
    assert stability["samples_within_band"] == 6
    assert stability["status"] == "unstable"


def test_timing_stability_accepts_exactly_eight_samples_in_band() -> None:
    stability = perf_matrix._timing_stability(
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 120.0, 120.0]
    )

    assert stability["samples_within_band"] == 8
    assert stability["status"] == "stable"


def test_timing_stability_does_not_judge_a_nonstandard_sample_count() -> None:
    stability = perf_matrix._timing_stability([10.0, 11.0])

    assert stability == {
        "status": "not_evaluated",
        "sample_count": 2,
        "reason": "requires_10_samples",
    }


def test_suite_timing_contract_drift_is_rejected_before_execution() -> None:
    case = next(
        value
        for value in performance_catalog.load_suite(SUITE).cases
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
        performance_catalog.PerformanceSuiteError,
        match=r"baseline\.timing_scope must be 'task-model-call-wall'",
    ):
        performance_catalog.validate_case(drifted)


def test_suite_rejects_non_boolean_baseline_local_files_only() -> None:
    case = next(
        value
        for value in performance_catalog.load_suite(SUITE).cases
        if value["id"] == "sam3.segment_prompted"
    )
    drifted = {
        **case,
        "baseline": {
            **case["baseline"],
            "local_files_only": "true",
        },
    }

    with pytest.raises(
        performance_catalog.PerformanceSuiteError,
        match="baseline local_files_only must be boolean",
    ):
        performance_catalog.validate_case(drifted)


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


def test_reranking_order_contract_checks_all_document_scores() -> None:
    case = {
        "operation": "rerank",
        "baseline": {"output_contract": "reranking-order"},
    }
    candidate = {"output_summary": {"documents": 3, "scores": [0.8, 0.1, 0.4]}}
    reference = {"output_summary": {"document_count": 3, "scores": [12.0, -4.0, 2.0]}}

    assert perf_matrix._output_contract(case, candidate, reference) == (True, "")

    candidate["output_summary"]["scores"] = [0.1, 0.8, 0.4]
    assert perf_matrix._output_contract(case, candidate, reference) == (
        False,
        "reranking order differs",
    )

    candidate["output_summary"]["scores"] = [0.8, float("nan"), 0.4]
    assert perf_matrix._output_contract(case, candidate, reference) == (
        False,
        "reranking scores are missing or invalid",
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
    capsys,
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
        lambda _case, _resolved, _output, _options: (
            ["baseline", "--precision", "fp16"],
            "base",
        ),
    )
    monkeypatch.setattr(
        perf_matrix,
        "_wait_for_gpu_memory_headroom",
        lambda **kwargs: events.append(("wait", kwargs["minimum_free_fraction"])),
    )
    monkeypatch.setattr(
        perf_matrix,
        "_run_command",
        lambda argv, _environment, _timeout: (
            events.append(("run", argv[0])) or {"exit_code": 0, "stdout": ""}
        ),
    )
    monkeypatch.setattr(
        perf_matrix,
        "_candidate_result",
        lambda _directory, _digest: {"samples_ms": [10.0] * 10},
    )
    monkeypatch.setattr(
        perf_matrix,
        "_read_baseline",
        lambda _path: {"samples_ms": [10.0] * 10},
    )
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
        Namespace(timeout_seconds=30, minimum_gpu_free_fraction=0.25, verbose=False),
        tmp_path,
        row,
        progress=lambda stage, evidence: events.append(
            ("progress", stage, tuple(evidence["commands"]))
        ),
    )

    assert events == [
        ("progress", "candidate", ("trtmc", "baseline")),
        ("wait", 0.25),
        ("run", "candidate"),
        ("progress", "reference", ("trtmc", "baseline")),
        ("wait", 0.25),
        ("run", "baseline"),
    ]
    assert row["status"] == "green"
    assert "TRTMC:" not in capsys.readouterr().out

    events.clear()
    row = {"resolved_settings": {}}
    perf_matrix._run_supported_case(
        {"id": "example"},
        {"model": {"precision": "fp16"}, "request": {}},
        Namespace(timeout_seconds=30, minimum_gpu_free_fraction=0.25, verbose=True),
        tmp_path,
        row,
    )

    output = capsys.readouterr().out
    assert "[example] TRTMC: candidate" in output
    assert "[example] baseline: baseline --precision fp16" in output


def test_unsettled_measurement_is_retried_once_in_fresh_processes(tmp_path, monkeypatch) -> None:
    falling = [3.7, 3.4, 3.0, 2.7, 2.3, 1.9, 1.6, 1.4, 1.2, 1.0]
    settled = [10.0] * 10
    commands_run = []
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
        lambda _case, _resolved, output, _options: (
            ["baseline", "--precision", "fp16", "--output", str(output)],
            "base",
        ),
    )
    monkeypatch.setattr(perf_matrix, "_wait_for_gpu_memory_headroom", lambda **_kwargs: None)
    monkeypatch.setattr(
        perf_matrix,
        "_run_command",
        lambda argv, _environment, _timeout: (
            commands_run.append(argv[0]) or {"exit_code": 0, "stdout": "ok", "stderr": ""}
        ),
    )
    monkeypatch.setattr(
        perf_matrix,
        "_candidate_result",
        lambda directory, _digest: {
            "samples_ms": settled if "measurement-2" in str(directory) else falling
        },
    )
    monkeypatch.setattr(
        perf_matrix,
        "_read_baseline",
        lambda _path: {"samples_ms": settled},
    )
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
        Namespace(timeout_seconds=30, minimum_gpu_free_fraction=0, verbose=False),
        tmp_path / "work",
        row,
    )

    assert commands_run == ["candidate", "baseline", "candidate", "baseline"]
    assert row["status"] == "green"
    assert row["measurement_stability"]["status"] == "stable_after_retry"
    assert [attempt["attempt"] for attempt in row["measurement_stability"]["attempts"]] == [1, 2]
    assert set(row["commands"]) == {
        "trtmc",
        "baseline",
        "trtmc_measurement_2",
        "baseline_measurement_2",
    }
    assert len(row["logs"]) == 8
    assert any("measurement-2" in record["href"] for record in row["logs"])


def test_measurement_that_remains_unsettled_has_no_performance_light(tmp_path, monkeypatch) -> None:
    falling = [3.7, 3.4, 3.0, 2.7, 2.3, 1.9, 1.6, 1.4, 1.2, 1.0]
    monkeypatch.setattr(perf_matrix, "_command_environment", lambda: {})
    monkeypatch.setattr(perf_matrix, "_workload_digest", lambda _resolved: "digest")
    monkeypatch.setattr(perf_matrix, "_candidate_base_argv", lambda *_args: ["candidate"])
    monkeypatch.setattr(
        perf_matrix,
        "_baseline_argv",
        lambda _case, _resolved, _output, _options: (
            ["baseline", "--precision", "fp16"],
            "base",
        ),
    )
    monkeypatch.setattr(perf_matrix, "_wait_for_gpu_memory_headroom", lambda **_kwargs: None)
    monkeypatch.setattr(
        perf_matrix,
        "_run_command",
        lambda *_args: {"exit_code": 0, "stdout": "", "stderr": ""},
    )
    monkeypatch.setattr(
        perf_matrix,
        "_candidate_result",
        lambda *_args: {"samples_ms": falling},
    )
    monkeypatch.setattr(
        perf_matrix,
        "_read_baseline",
        lambda _path: {"samples_ms": [10.0] * 10},
    )
    monkeypatch.setattr(perf_matrix, "_classify", lambda *_args, **_kwargs: ("yellow", {}))
    monkeypatch.setattr(perf_matrix, "_stable_even", lambda _value: True)

    row = {"resolved_settings": {}}
    perf_matrix._run_supported_case(
        {"id": "example"},
        {"model": {"precision": "fp16"}, "request": {}},
        Namespace(timeout_seconds=30, minimum_gpu_free_fraction=0, verbose=False),
        tmp_path / "work",
        row,
    )
    public = perf_matrix._public_perf_result(row)

    assert row["status"] == "measurement-inconclusive"
    assert public["result"] == "white"
    assert public["issue"]["code"] == "measurement_inconclusive"
    assert public["output_validation"]["status"] == "Pass"
    assert public["latency"] == {"reference_ms": None, "candidate_ms": None}
    assert public["measurement_stability"]["status"] == "measurement_inconclusive"


@pytest.mark.parametrize(
    "status",
    ["green", "yellow", "red", "contract-mismatch"],
)
def test_resume_keeps_terminal_comparison_results(status: str) -> None:
    assert perf_matrix._should_skip({"status": status})


def test_performance_projection_is_rebuilt_from_ordered_live_receipts(tmp_path) -> None:
    base_rows = [
        {
            "id": "model-a.generate",
            "family": "family-a",
            "operation": "generate",
            "model": "model-a",
            "status": "pending",
        },
        {
            "id": "model-b.generate",
            "family": "family-b",
            "operation": "generate",
            "model": "model-b",
            "status": "pending",
        },
    ]
    ledger = perf_matrix.ExecutionLedger.open(
        tmp_path,
        campaign_id="run-1",
        task_kind="performance",
        fingerprint="revision-1",
        cases=[{"id": row["id"], "report": row} for row in base_rows],
    )
    terminal = {
        **base_rows[0],
        "status": "contract-mismatch",
        "reason": "outputs differ",
    }
    ledger.begin("model-a.generate", stage="candidate")
    ledger.finish("model-a.generate", result="white", payload=terminal)
    results = {
        "selected_entry_ids": ["model-a.generate", "model-b.generate"],
        "cases": [
            {**base_rows[0], "status": "green"},
            {**base_rows[1], "status": "red"},
        ],
    }

    perf_matrix._sync_perf_results_from_ledger(results, ledger)

    assert results["cases"] == [
        {**terminal, "progress": {"stage": "candidate", "attempt": 1}},
        {
            **base_rows[1],
            "commands": {},
            "logs": [],
            "progress": {"stage": None, "attempt": 0},
        },
    ]


def test_performance_projection_rejects_receipt_classification_drift(tmp_path) -> None:
    row = {
        "id": "model-a.generate",
        "family": "family-a",
        "operation": "generate",
        "model": "model-a",
        "status": "pending",
    }
    ledger = perf_matrix.ExecutionLedger.open(
        tmp_path,
        campaign_id="run-1",
        task_kind="performance",
        fingerprint="revision-1",
        cases=[{"id": row["id"], "report": row}],
    )
    ledger.begin(row["id"], stage="candidate")
    ledger.finish(row["id"], result="green", payload={**row, "status": "failed"})

    with pytest.raises(perf_matrix.PerfMatrixError, match="ledger result mismatch"):
        perf_matrix._sync_perf_results_from_ledger(
            {"selected_entry_ids": [row["id"]], "cases": [row]},
            ledger,
        )


def test_performance_adapter_resumes_an_interrupted_case_as_a_new_attempt(
    tmp_path, monkeypatch
) -> None:
    case = next(
        row for row in performance_catalog.load_suite(SUITE).cases if row["id"] == "gpt2.generate"
    )
    results = {
        "run_id": "run-1",
        "git_commit": "revision-1",
        "suite_sha256": "suite-1",
        "environment_config": {"sha256": "environment-1"},
        "selected_entry_ids": [case["id"]],
        "cases": [perf_matrix._pending_perf_row(case)],
    }
    options = perf_matrix.RunOptions(
        output=tmp_path / "output",
        scratch_root=tmp_path / "scratch",
        trtmc_bench="trtmc-bench",
        trtmc_worker=None,
        hf_transformers_runner=tmp_path / "reference.py",
        task_reference_runner=tmp_path / "task_reference.py",
        bundle_cache=None,
        bundle_roots=(),
        runtime_dirs=(),
        local_files_only=True,
        minimum_free_space_gib=0,
        minimum_gpu_free_fraction=0,
        timeout_seconds=1,
    )
    contract = perf_matrix.timing_contract(runner=case["baseline"]["runner"], family=case["family"])
    preflight = {
        case["id"]: (
            {
                "measurement": {"timing_scope": contract["candidate_timing_scope"]},
                "_candidate_build_python_profile": "test-build",
            },
            [],
            {},
        )
    }
    arguments = {
        "selected": [case],
        "options": options,
        "results": results,
        "preflight": preflight,
        "preflight_failures": {},
        "reference_preflight": {},
        "worker": {},
        "storage_preflight": {},
    }

    def interrupt_with_leaf_commands(*_args, **kwargs):
        kwargs["progress"](
            "reference",
            {
                "commands": {
                    "resolve": {},
                    "trtmc": {"argv": ["trtmc-bench"], "cwd": str(REPOSITORY)},
                    "baseline": {"argv": ["reference"], "cwd": str(REPOSITORY)},
                }
            },
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(perf_matrix, "_run_one", interrupt_with_leaf_commands)

    with pytest.raises(KeyboardInterrupt):
        perf_matrix._execute_campaign(**arguments)
    ledger = perf_matrix.ExecutionLedger.load(options.output, task_kind="performance")
    running = ledger.receipt(case["id"])
    assert running["state"] == "running"
    evidence = running["attempts"][0]["evidence"]
    assert set(evidence["commands"]) == {"resolve", "trtmc", "baseline"}
    assert evidence["commands"]["trtmc"]["argv"] == ["trtmc-bench"]
    assert len(evidence["logs"]) == 4
    assert all((options.output / log["href"]).is_file() for log in evidence["logs"])
    live = json.loads((options.output / "report.json").read_text(encoding="utf-8"))
    assert live["results"][0]["progress"] == {"stage": "reference", "attempt": 1}
    assert len(live["results"][0]["debug"]["logs"]) == 4

    monkeypatch.setattr(
        perf_matrix,
        "_run_one",
        lambda *_args, **_kwargs: {
            **perf_matrix._pending_perf_row(case),
            "status": "failed",
            "reason": "candidate failed",
            "commands": {},
        },
    )

    assert perf_matrix._execute_campaign(**arguments) == 1
    receipt = ledger.receipt(case["id"])
    assert receipt["state"] == "terminal"
    assert [(attempt["attempt"], attempt["state"]) for attempt in receipt["attempts"]] == [
        (1, "interrupted"),
        (2, "failed"),
    ]


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
    reference = {"output_summary": {"text": "Architecture:\nAttention:Standard Q/K/V/O"}}

    assert perf_matrix._output_contract(case, candidate, reference) == (True, "")

    candidate["output_summary"]["text"] = "OCR title only"
    matched, reason = perf_matrix._output_contract(case, candidate, reference)
    assert not matched
    assert reason == "TRTMC OCR text misses required content"


def test_localization_contract_accepts_structurally_equivalent_boxes() -> None:
    case = {
        "operation": "generate",
        "baseline": {
            "output_contract": "localization",
            "min_localization_box_iou": 0.9,
            "max_localization_point_distance": 10.0,
            "max_normalized_edit_distance": 0.5,
        },
    }
    candidate = {
        "output_summary": {
            "text": "<ref>white vehicle</ref><box><302><266><830><710></box>",
            "token_ids": [1, 2, 3],
        }
    }
    reference = {
        "output_summary": {
            "text": "<ref>white vehicle</ref><box><304><267><828><708></box>",
            "token_ids": [9, 8, 7, 6],
        }
    }

    assert perf_matrix._output_contract(case, candidate, reference) == (True, "")


def test_localization_contract_checks_point_type_and_distance() -> None:
    case = {
        "operation": "generate",
        "baseline": {
            "output_contract": "localization",
            "min_localization_box_iou": 0.9,
            "max_localization_point_distance": 10.0,
            "max_normalized_edit_distance": 0.5,
        },
    }
    candidate = {"output_summary": {"text": "<ref>button</ref><box><504><252></box>"}}
    reference = {"output_summary": {"text": "<ref>button</ref><box><500><250></box>"}}

    assert perf_matrix._output_contract(case, candidate, reference) == (True, "")

    candidate["output_summary"]["text"] = "<ref>button</ref><box><450><200><550><300></box>"
    assert perf_matrix._output_contract(case, candidate, reference) == (
        False,
        "localization output type differs",
    )

    candidate["output_summary"]["text"] = "<ref>button</ref><box><700><600></box>"
    assert perf_matrix._output_contract(case, candidate, reference) == (
        False,
        "localization point distance exceeds the configured contract",
    )


def test_localization_contract_rejects_invalid_thresholds() -> None:
    case = next(
        value
        for value in performance_catalog.load_suite(SUITE).cases
        if value["id"] == "locateanything.generate"
    )
    drifted = {
        **case,
        "baseline": {
            **case["baseline"],
            "max_localization_point_distance": -1.0,
        },
    }

    with pytest.raises(
        performance_catalog.PerformanceSuiteError,
        match="localization contract has invalid max_localization_point_distance",
    ):
        performance_catalog.validate_case(drifted)


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
    assert perf_matrix._output_contract(case, candidate, aligned_reference) == (
        True,
        "",
    )


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


def test_disparity_contract_compares_every_fp32_pixel(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.f32"
    reference_path = tmp_path / "reference.f32"
    reference = np.full((700, 700), 10.0, dtype=np.float32)
    candidate = reference.copy()
    candidate.reshape(-1)[:100] += np.float32(2.1)
    candidate.tofile(candidate_path)
    reference.tofile(reference_path)
    case = {
        "operation": "disparity",
        "baseline": {
            "output_contract": "disparity-parity",
            "min_disparity_cosine": 0.999,
            "max_disparity_mean_abs_error": 0.5,
            "max_disparity_bad_2px_fraction": 0.03,
        },
    }
    candidate_result = {
        "output_summary": {
            "height": 700,
            "width": 700,
            "element_count": 490_000,
            "disparity_artifact": str(candidate_path),
        }
    }
    reference_result = {
        "output_summary": {
            "height": 700,
            "width": 700,
            "element_count": 490_000,
            "disparity_artifact": str(reference_path),
        }
    }

    assert perf_matrix._output_contract(case, candidate_result, reference_result) == (True, "")

    candidate.reshape(-1)[:20_000] += np.float32(2.1)
    candidate.tofile(candidate_path)
    assert perf_matrix._output_contract(case, candidate_result, reference_result) == (
        False,
        "disparity output parity is outside the configured contract",
    )


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


def test_image_feature_contract_compares_both_public_output_shapes() -> None:
    case = {
        "operation": "extract_features",
        "baseline": {"output_contract": "image-features-shape"},
    }
    candidate = {
        "output_summary": {
            "last_hidden_state_shape": [1, 201, 384],
            "pooler_output_shape": [1, 384],
        }
    }
    reference = {
        "output_summary": {
            "last_hidden_state_shape": [1, 201, 384],
            "pooler_output_shape": [1, 384],
        }
    }

    assert perf_matrix._output_contract(case, candidate, reference) == (True, "")
    reference["output_summary"]["pooler_output_shape"] = [1, 768]
    assert perf_matrix._output_contract(case, candidate, reference) == (
        False,
        "image feature output shape differs",
    )


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


def test_report_displays_campaign_and_model_profile_wall_times() -> None:
    results = {
        "started_at": "2026-07-29T07:16:00+00:00",
        "finished_at": "2026-07-29T08:18:03+00:00",
        "repository_root": "/workspace/repository",
        "cases": [
            {
                "id": "example.generate",
                "family": "example",
                "operation": "generate",
                "model": "example-model",
                "task_type": "Text → Text",
                "task_strategy": "text_generation_causal",
                "status": "green",
                "started_at": "2026-07-29T07:20:00+00:00",
                "finished_at": "2026-07-29T07:21:30+00:00",
                "baseline_contract": {},
            }
        ],
    }

    report = perf_matrix._report_html(results)

    assert "Total campaign wall time:</strong> 1h 2m 3s" in report
    assert "3,723.000 s" in report
    assert "<th>Model wall time</th>" in report
    assert "1m 30.0s" in report
    assert "90.000 s" in report
    assert "including bundle preparation, GPU headroom waits" in report
    assert "not used for traffic-light classification" in report


def test_write_report_rebuilds_an_existing_run(tmp_path: Path) -> None:
    output = tmp_path / "performance-run"
    output.mkdir()
    results = {
        "schema_version": perf_matrix.RESULT_SCHEMA,
        "status": "completed",
        "suite_name": "example",
        "selected_entry_ids": [],
        "cases": [],
    }
    (output / "results.json").write_text(
        json.dumps(results),
        encoding="utf-8",
    )

    report_json, report_html, report = perf_matrix.write_report(output)

    assert report_json == output / "report.json"
    assert report_html == output / "report.html"
    assert report["report_kind"] == "performance"
    assert report_json.is_file()
    assert report_html.is_file()


def test_report_prefers_test_task_bundle_preparation_receipt() -> None:
    bundle = "/shared/engines/example/cache-key/example.bundle"
    results = {
        "git_commit": "tested-commit",
        "cases": [
            {
                "id": "example.generate",
                "family": "example",
                "operation": "generate",
                "model": "example-model",
                "task_type": "Text → Text",
                "task_strategy": "text_generation_causal",
                "status": "green",
                "baseline_contract": {},
                "candidate": {
                    "preparation": {
                        "included_in_performance_metrics": False,
                        "bundles": [
                            {
                                "model": "example-model",
                                "bundle": bundle,
                                "status": "reused",
                                "build_time_s": None,
                                "included_in_performance_metrics": False,
                            }
                        ],
                    }
                },
            }
        ],
    }
    receipt = {
        "schema_version": "trtmc.perf-bundle-preparation/v1",
        "scope": "test_task",
        "git_commit": "tested-commit",
        "included_in_performance_metrics": False,
        "bundles": [
            {
                "model": "example-model",
                "bundle": bundle,
                "status": "built",
                "build_time_s": 83.125,
                "included_in_performance_metrics": False,
            }
        ],
    }

    perf_matrix._apply_bundle_preparation_receipt(results, receipt)
    report = perf_matrix._report_html(results)

    assert "Built · 1m 23.1s" in report
    assert "data-filter-preparation='reused'" not in report
    assert "1 built in this test task (1m 23.1s total)" in report


def test_report_includes_client_side_row_filters() -> None:
    results = {
        "cases": [
            {
                "id": "example.generate",
                "family": "example",
                "operation": "generate",
                "model": "example-model",
                "task_type": "Text → Text",
                "task_strategy": "text_generation_causal",
                "status": "green",
                "baseline_contract": {},
                "candidate": {
                    "preparation": {
                        "bundles": [
                            {
                                "model": "example-model",
                                "bundle": "/shared/example.bundle",
                                "status": "built",
                                "build_time_s": 1.0,
                            }
                        ]
                    }
                },
            },
            {
                "id": "other.generate",
                "family": "other",
                "operation": "generate",
                "model": "other-model",
                "task_type": "Image + Text → Text",
                "task_strategy": "vision_language_generation",
                "status": "red",
                "baseline_contract": {},
                "candidate": {
                    "preparation": {
                        "bundles": [
                            {
                                "model": "other-model",
                                "bundle": "/shared/other.bundle",
                                "status": "reused",
                            }
                        ]
                    }
                },
            },
        ],
    }

    report = perf_matrix._report_html(results)

    assert 'id="report-filter-search"' in report
    assert 'id="report-filter-model-type"' in report
    assert 'id="report-filter-operation"' in report
    assert 'id="report-filter-task-type"' in report
    assert 'id="report-filter-status"' in report
    assert 'id="report-filter-preparation"' in report
    assert 'id="report-filter-reset"' in report
    assert 'id="report-filter-count">Showing 2 of 2 rows<' in report
    assert "data-filter-search='example generate example-model example.generate'" not in report
    assert "example generate example-model text → text" in report
    assert "data-filter-model-type='example'" in report
    assert "data-filter-operation='generate'" in report
    assert "data-filter-task-type='Text → Text'" in report
    assert "data-filter-status='green'" in report
    assert "data-filter-preparation='built'" in report
    assert "data-filter-status='red'" in report
    assert "data-filter-preparation='reused'" in report
    assert "const matches = controls.every((control)" in report
    assert "row.hidden = !matches;" in report


def test_report_recovers_task_type_from_an_existing_result_manifest() -> None:
    task_type, user_contract, task_strategy = perf_matrix._report_task_metadata(
        {
            "model": "codegen-350m",
            "operation": "generate",
            "resolved_settings": {
                "testcase": "codegen-350m",
                "model": {
                    "manifest": "codegen/manifests/codegen-350m.json",
                    "task_strategy": "text_generation_causal",
                },
            },
        }
    )

    assert task_type == "Text → Code"
    assert user_contract == "code_completion"
    assert task_strategy == "text_generation_causal"


def test_apply_bundle_preparation_receipt_rejects_unmatched_bundle() -> None:
    results = {
        "git_commit": "tested-commit",
        "cases": [
            {
                "id": "example.generate",
                "model": "example-model",
                "candidate": {
                    "preparation": {
                        "bundles": [
                            {
                                "model": "example-model",
                                "bundle": "/shared/engines/example.bundle",
                            }
                        ]
                    }
                },
            }
        ],
    }
    receipt = {
        "schema_version": "trtmc.perf-bundle-preparation/v1",
        "scope": "test_task",
        "git_commit": "tested-commit",
        "included_in_performance_metrics": False,
        "bundles": [
            {
                "model": "example-model",
                "bundle": "/different/example.bundle",
                "status": "built",
                "build_time_s": 1.0,
                "included_in_performance_metrics": False,
            }
        ],
    }

    with pytest.raises(
        perf_matrix.PerfMatrixError,
        match="does not match a bundle used by the performance run",
    ):
        perf_matrix._apply_bundle_preparation_receipt(results, receipt)


@pytest.mark.parametrize(
    "record",
    [
        {},
        {
            "started_at": "invalid",
            "finished_at": "2026-07-29T08:18:03+00:00",
        },
        {
            "started_at": "2026-07-29T08:18:03+00:00",
            "finished_at": "2026-07-29T07:16:00+00:00",
        },
    ],
)
def test_wall_time_rejects_incomplete_or_invalid_timestamps(
    record: dict[str, str],
) -> None:
    assert perf_matrix._wall_time_seconds(record) is None
    assert perf_matrix._wall_time_html(record) == "—"


def test_public_perf_result_with_unknown_precision_has_no_performance_light() -> None:
    row = {
        "id": "model.generate",
        "status": "green",
        "resolved_settings": {"model": {"precision": "fp16"}},
        "candidate": {"precision": "fp16", "samples_ms": [10.0]},
        "baseline": {"samples_ms": [20.0]},
        "comparison": {},
    }

    public = perf_matrix._public_perf_result(row)

    assert public["state"] == "terminal"
    assert public["result"] == "white"
    assert public["issue"]["code"] == "comparison_contract"
    assert public["output_validation"]["status"] == "Pass"
    assert public["latency"] == {"reference_ms": None, "candidate_ms": None}


def test_public_perf_result_surfaces_quantized_candidate_precision() -> None:
    row = {
        "id": "qwen.generate@qwen3-0.6b-fp8",
        "status": "green",
        "resolved_settings": {
            "baseline_precision": "fp16",
            "model": {
                "precision": "fp16",
                "build": {"quantization": {"format": "fp8"}},
            },
            "output_contract": "exact-token-ids",
        },
        "candidate": {"precision": "fp16", "samples_ms": [5.0]},
        "baseline": {"precision": "fp16", "samples_ms": [16.0]},
        "comparison": {},
    }

    public = perf_matrix._public_perf_result(row)

    assert public["precision"] == {
        "reference": "fp16",
        "candidate": "fp8 (fp16 base)",
    }


def test_public_perf_result_publishes_timing_stability_as_shadow_evidence() -> None:
    row = {
        "id": "model.generate",
        "status": "yellow",
        "resolved_settings": {
            "baseline_precision": "fp16",
            "model": {"precision": "fp16"},
            "output_contract": "exact-token-ids",
        },
        "candidate": {
            "precision": "fp16",
            "samples_ms": [
                3.7,
                3.4,
                3.0,
                2.7,
                2.3,
                1.9,
                1.6,
                1.4,
                1.2,
                1.0,
            ],
        },
        "baseline": {
            "precision": "fp16",
            "samples_ms": [
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
            ],
        },
        "comparison": {},
    }

    public = perf_matrix._public_perf_result(row)

    assert public["result"] == "yellow"
    assert public["measurement_stability"]["mode"] == "shadow"
    assert public["measurement_stability"]["status"] == "retry_recommended"
    assert public["measurement_stability"]["reference"]["status"] == "stable"
    assert public["measurement_stability"]["trtmc"]["status"] == "unstable"


def test_public_perf_result_does_not_evaluate_stability_without_valid_comparison() -> None:
    row = {
        "id": "model.generate",
        "status": "contract-mismatch",
        "resolved_settings": {
            "baseline_precision": "fp16",
            "model": {"precision": "fp16"},
            "output_contract": "exact-token-ids",
        },
        "candidate": {"precision": "fp16", "samples_ms": [10.0] * 10},
        "baseline": {"precision": "fp16", "samples_ms": [10.0] * 10},
        "comparison": {"reason": "outputs differ"},
    }

    public = perf_matrix._public_perf_result(row)

    assert public["result"] == "white"
    assert public["measurement_stability"] is None


def test_command_diagnostic_materializes_nested_build_logs(tmp_path: Path) -> None:
    build_stderr = tmp_path / "bundle-cache" / "build.stderr.log"
    build_stderr.parent.mkdir()
    build_stderr.write_text("builder crashed\n", encoding="utf-8")
    diagnostic = {
        "schema_version": "trtmc.command-diagnostic/v1",
        "stage": "build",
        "domain": "harness/unknown",
        "code": "bundle_build_failed",
        "artifacts": [{"label": "Bundle build stderr", "path": str(build_stderr)}],
    }
    command = {
        "stdout": "",
        "stderr": (
            "bundle build failed\n"
            f"TRTMC_DIAGNOSTIC_JSON={json.dumps(diagnostic, separators=(',', ':'))}\n"
        ),
    }

    parsed = perf_matrix._command_diagnostic(command["stderr"])
    command["diagnostic"] = parsed
    links = perf_matrix._materialize_command_logs(
        tmp_path / "report",
        "gpt2.generate",
        "trtmc",
        command,
    )

    nested = next(item for item in links if item["label"] == "Bundle build stderr")
    published = tmp_path / "report" / nested["href"]
    assert published.read_text(encoding="utf-8") == "builder crashed\n"
    assert not published.is_symlink()
    assert command["diagnostic"] == {
        "schema_version": "trtmc.command-diagnostic/v1",
        "stage": "build",
        "domain": "harness/unknown",
        "code": "bundle_build_failed",
        "artifacts": [nested],
    }


def test_perf_issue_uses_structured_command_failure() -> None:
    issue = perf_matrix._perf_issue(
        {
            "status": "failed",
            "reason": "trtmc command failed with rc=2",
            "failure_stage": "build",
            "failure_domain": "harness/unknown",
            "failure_code": "bundle_build_failed",
        },
        "white",
    )

    assert issue == {
        "priority": "P1",
        "stage": "build",
        "domain": "harness/unknown",
        "code": "bundle_build_failed",
        "message": "trtmc command failed with rc=2",
    }


def test_cleanup_warning_does_not_replace_a_valid_performance_result() -> None:
    row = {
        "status": "green",
        "resolved_settings": {
            "baseline_precision": "fp16",
            "model": {"precision": "fp16"},
        },
        "candidate": {"precision": "fp16"},
        "warnings": [
            {
                "stage": "cleanup",
                "code": "resource_cleanup_failed",
                "message": "scratch directory retained",
            }
        ],
    }

    assert perf_matrix._final_status([row]) == "completed-with-warnings"
    assert perf_matrix._perf_state_and_result(row) == ("terminal", "green")


@pytest.mark.parametrize("status", ["pending", "running"])
def test_public_perf_progress_has_no_traffic_light(status: str) -> None:
    public = perf_matrix._public_perf_result({"id": "model.generate", "status": status})

    assert public["state"] == status
    assert public["result"] is None


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
    original_preflight = perf_matrix._preflight_selected
    preflight_calls = 0

    def preflight_after_pending_report(cases, options):
        nonlocal preflight_calls
        preflight_calls += 1
        live = json.loads((options.output / "report.json").read_text(encoding="utf-8"))
        selected = [row for row in live["results"] if row["id"] == "gpt2.generate"]
        if preflight_calls == 1:
            assert [(row["state"], row["result"]) for row in selected] == [("pending", None)]
        return original_preflight(cases, options)

    monkeypatch.setattr(
        perf_matrix,
        "_preflight_selected",
        preflight_after_pending_report,
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
        "artifacts",
        "assets",
        "ledger",
        "report.html",
        "report.json",
        "results.json",
    ]
    assert not scratch_root.exists()
    results = json.loads((output / "results.json").read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in results["cases"]}
    assert len(rows) == 108
    assert results["environment_config"]["name"] == "test-gb300"
    assert results["environment_config"]["execution"]["minimum_gpu_free_fraction"] == 0.0
    assert results["environment_config"]["source"] == str(environment.resolve())
    catalog_entries = perf_matrix.ManifestCatalog().entries()
    catalog_counts = Counter(entry.status for entry in catalog_entries)
    excluded_l0_profiles = sum(
        entry.status == "ready" and performance_catalog.is_l0_profile(entry.name)
        for entry in catalog_entries
    )
    expected_catalog_coverage = {
        "total_profiles": len(catalog_entries),
        "ready_profiles": catalog_counts["ready"],
        "release_profiles": catalog_counts["ready"] - excluded_l0_profiles - 6,
        "explicitly_excluded_profiles": 6,
        "explicit_exclusions": [
            {
                "model": "lfm2-1.2b",
                "reason": LFM2_EXCLUSION_REASON,
            },
            {
                "model": "lfm2-2.6b",
                "reason": LFM2_EXCLUSION_REASON,
            },
            {
                "model": "lfm2-350m-bf16-model-card",
                "reason": LFM2_EXCLUSION_REASON,
            },
            {
                "model": "lfm2-350m-fp16",
                "reason": LFM2_EXCLUSION_REASON,
            },
            {
                "model": "lfm2-700m",
                "reason": LFM2_EXCLUSION_REASON,
            },
            {
                "model": "minimax-h3-768p",
                "reason": MINIMAX_H3_EXCLUSION_REASON,
            },
        ],
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
    public_report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert public_report["schema_version"] == "trtmc.qualification-report/v1"
    assert public_report["report_kind"] == "performance"
    assert public_report["accounting"] == {
        "selected": 1,
        "comparable": 1,
        "operational_coverage_percent": 100.0,
        "progress": {"pending": 0, "running": 0, "terminal": 1},
        "outcomes": {"green": 1, "yellow": 0, "red": 0, "white": 0},
        "invariants": {
            "selected": "pending + running + terminal",
            "terminal": "green + yellow + red + white",
            "comparable": "green + yellow + red",
        },
        "definitions": {
            "green": {
                "class": "valid-comparison",
                "label": "Meets the target",
                "denominator": "comparable",
            },
            "yellow": {
                "class": "valid-comparison",
                "label": "Valid comparison in the review band",
                "denominator": "comparable",
            },
            "red": {
                "class": "valid-comparison",
                "label": "Valid comparison that misses the target",
                "denominator": "comparable",
            },
            "white": {
                "class": "coverage-gap",
                "label": "No valid comparison",
                "denominator": "selected",
            },
        },
    }
    assert len(public_report["results"]) == 1
    public_row = public_report["results"][0]
    assert public_row["id"] == "gpt2.generate"
    assert public_row["state"] == "terminal"
    assert public_row["result"] == "green"
    assert public_row["precision"] == {
        "reference": "fp32",
        "candidate": "fp16",
    }
    assert public_row["output_validation"]["status"] == "Pass"
    assert public_row["latency"] == {
        "reference_ms": 20.45,
        "candidate_ms": 10.45,
    }
    assert public_row["measurement_stability"]["mode"] == "enforced"
    assert public_row["measurement_stability"]["status"] == "stable"
    assert public_row["issue"] is None
    assert all(
        "stdout_tail" not in command and "stderr_tail" not in command
        for command in public_row["commands"].values()
    )
    assert "minimax-h3-768p" not in json.dumps(public_report)
    assert MINIMAX_H3_EXCLUSION_REASON not in json.dumps(public_report)
    log_records = public_row["debug"]["logs"]
    assert {record["label"] for record in log_records} == {
        "TRTMC stdout",
        "TRTMC stderr",
        "Reference stdout",
        "Reference stderr",
    }
    for record in log_records:
        assert not Path(record["href"]).is_absolute()
        assert (output / record["href"]).is_file()

    report = (output / "report.html").read_text(encoding="utf-8")
    assert 'data-report="report.json"' in report
    assert "gpt2.generate" not in report
    assert "minimax-h3-768p" not in report
    frontend = (output / "assets/qualification-report.js").read_text(encoding="utf-8")
    assert "Comparable results" in frontend
    assert "Operational coverage" in frontend
    assert "Failures" in frontend
    assert "Reference latency" in frontend
    assert "TRTMC latency" in frontend
    assert "Timing stability" in frontend

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
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    receipt_path = next((output / "ledger" / "cases").glob("*/receipt.json"))
    receipt_mtime = receipt_path.stat().st_mtime_ns

    assert perf_matrix.main(["report", str(output)]) == 0
    rebuilt = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert rebuilt["results"][0]["result"] == "green"
    rows["gpt2.generate"]["status"] = "failed"
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    assert perf_matrix.main(["resume", str(output)]) == 0
    assert receipt_path.stat().st_mtime_ns == receipt_mtime
    resumed = json.loads((output / "results.json").read_text(encoding="utf-8"))
    resumed_rows = {row["id"]: row for row in resumed["cases"]}
    assert resumed_rows["gpt2.generate"]["status"] == "green"
    assert sorted(path.name for path in output.iterdir()) == [
        "artifacts",
        "assets",
        "ledger",
        "report.html",
        "report.json",
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
        "artifacts",
        "assets",
        "ledger",
        "report.html",
        "report.json",
        "results.json",
    ]
    public_report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert public_report["accounting"]["selected"] == 1
    assert public_report["accounting"]["comparable"] == 0
    assert public_report["accounting"]["outcomes"]["white"] == 1
    public_row = public_report["results"][0]
    assert public_row["state"] == "terminal"
    assert public_row["result"] == "white"
    assert public_row["issue"]["stage"] == "reference-preflight"
    assert public_row["issue"]["code"] == "execution_failure"
    assert public_row["output_validation"]["status"] == "Not completed"
    assert public_row["debug"]["logs"] == [
        {
            "label": "Reference Preflight diagnostic",
            "href": "artifacts/gpt2.generate/logs/reference-preflight.log",
        }
    ]
    diagnostic = output / public_row["debug"]["logs"][0]["href"]
    assert diagnostic.is_file()
    assert "profile unavailable" in diagnostic.read_text(encoding="utf-8")
    assert "minimax-h3-768p" not in json.dumps(public_report)
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
    assert rows["gemma.generate"]["baseline"]["precision"] == "fp16"
    assert rows["phi.generate"]["baseline"]["output_contract"] == "exact-text"
    assert rows["phi_moe.generate"]["baseline"]["output_contract"] == "exact-text"
    assert rows["deepseek_ocr.generate"]["baseline"]["output_contract"] == "ocr-text"
    assert rows["locateanything.generate"]["baseline"]["output_contract"] == "localization"
    assert rows["qwen_vl.generate"]["baseline"]["output_contract"] == "normalized-text"
    assert rows["pixart.generate_image"]["baseline"]["adapter_options"] == {
        "component_precision_contract": "pixart_fp16_dit_fp32_t5"
    }
    assert "adapter_options" not in rows["flux.generate_image"]["baseline"]
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
    case = performance_catalog.load_suite(SUITE).cases[0]
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


def test_candidate_command_forwards_workload_runtime_overrides(tmp_path: Path) -> None:
    case = {
        "id": "nemotron_h.generate",
        "family": "nemotron_h",
        "model": "nemotron-h-nano-9b",
        "workload": {
            "testcase": "nemotron-h-nano-9b",
            "runtime": {"cuda_graphs": True},
        },
        "measurement": {"warmup": 2, "iterations": 10},
        "baseline": {
            "runner": "hf-transformers",
            "asset_loading_included": False,
        },
    }
    options = Namespace(
        trtmc_bench="trtmc-bench",
        trtmc_worker=None,
        bundle_cache=None,
        bundle_roots=(),
        runtime_dirs=(),
    )

    argv = perf_matrix._candidate_base_argv(case, options)

    assert "runtime.cuda_graphs=true" in argv


def test_task_reference_can_require_local_model_files_per_case(tmp_path: Path) -> None:
    case = {
        "baseline": {
            "adapter": "hf-transformers-vision",
            "local_files_only": True,
            "timing_scope": "task-model-call-wall",
            "input_preparation_included": False,
            "asset_loading_included": False,
        },
        "measurement": {"warmup": 1, "iterations": 2},
    }
    resolved = {
        "operation": "segment_prompted",
        "request": {"image_path": "input.png", "prompt": "object"},
        "runtime": {},
        "model": {
            "family": "sam3",
            "hf_id": "facebook/sam3",
            "manifest_path": str(tmp_path / "manifest.json"),
            "precision": "fp32",
            "task_strategy": "vision_segmentation",
        },
    }
    options = Namespace(
        local_files_only=False,
        task_reference_runner=tmp_path / "task_reference.py",
    )

    argv = perf_matrix._task_reference_argv(
        case=case,
        resolved=resolved,
        manifest={},
        output=tmp_path / "baseline.json",
        options=options,
        profile="default",
        python="python",
        mode="hf-eager",
    )

    assert "--local-files-only" in argv


def test_task_reference_uses_manifest_reference_precision(tmp_path: Path) -> None:
    case = {
        "baseline": {
            "adapter": "hf-diffusers",
            "timing_scope": "task-pipeline-call-wall",
            "input_preparation_included": True,
            "asset_loading_included": False,
        },
        "measurement": {"warmup": 1, "iterations": 2},
        "workload": {"testcase": "z-image-turbo"},
    }
    resolved = {
        "testcase": "z-image-turbo",
        "operation": "generate_image",
        "request": {"prompt": "cat"},
        "runtime": {},
        "model": {
            "family": "z_image",
            "hf_id": "Tongyi-MAI/Z-Image-Turbo",
            "manifest_path": str(tmp_path / "manifest.json"),
            "precision": "fp16",
            "task_strategy": "diffusion_media_generation",
        },
    }
    manifest = {
        "testcases": [
            {
                "name": "z-image-turbo",
                "reference_precision": "bf16",
            }
        ]
    }
    options = Namespace(
        local_files_only=False,
        task_reference_runner=tmp_path / "task_reference.py",
    )

    argv = perf_matrix._task_reference_argv(
        case=case,
        resolved=resolved,
        manifest=manifest,
        output=tmp_path / "baseline.json",
        options=options,
        profile="default",
        python="python",
        mode="hf-eager",
    )

    assert argv[argv.index("--precision") + 1] == "bf16"


def test_entry_is_the_only_run_selection() -> None:
    suite = performance_catalog.load_suite(SUITE)

    selected = suite.select(entries=["flux.generate_image"])

    assert [case["id"] for case in selected] == ["flux.generate_image"]


def test_model_selection_expands_every_matching_perf_entry_in_model_order() -> None:
    cases = [
        {"id": "model-a.long", "family": "family-a", "model": "model-a"},
        {"id": "model-b.default", "family": "family-b", "model": "model-b"},
        {"id": "model-a.short", "family": "family-a", "model": "model-a"},
    ]

    selected = _suite_for_cases(cases).select(models=["model-b", "model-a"])

    assert [case["id"] for case in selected] == [
        "model-b.default",
        "model-a.long",
        "model-a.short",
    ]


def test_model_ci_family_selection_expands_owned_perf_profiles():
    cases = [
        {"id": "a.one", "family": "family-a", "model": "model-a1"},
        {"id": "b.one", "family": "family-b", "model": "model-b1"},
        {"id": "a.two", "family": "family-a", "model": "model-a2"},
    ]

    selected = _suite_for_cases(cases).select(families=["family-a"])

    assert [case["id"] for case in selected] == ["a.one", "a.two"]


def test_model_selection_rejects_models_without_perf_entries() -> None:
    cases = [{"id": "model-a.default", "family": "family-a", "model": "model-a"}]

    with pytest.raises(
        performance_catalog.PerformanceSuiteError,
        match="models have no performance entries: model-b",
    ):
        _suite_for_cases(cases).select(models=["model-b"])


def test_model_selection_reports_explicit_perf_exclusion() -> None:
    cases = [{"id": "model-a.default", "family": "family-a", "model": "model-a"}]

    with pytest.raises(
        performance_catalog.PerformanceSuiteError,
        match="excluded performance models: model-b: baseline unavailable",
    ):
        _suite_for_cases(
            cases,
            exclusions={"model-b": "baseline unavailable"},
        ).select(models=["model-b"])


def test_perf_selection_modes_are_mutually_exclusive() -> None:
    arguments = perf_matrix.build_parser().parse_args(
        [
            "check",
            str(SUITE),
            "--environment",
            str(GB300_ENVIRONMENT),
            "--entry",
            "flux.generate_image",
            "--model",
            "flux-schnell",
        ]
    )

    with pytest.raises(perf_matrix.PerfMatrixError, match="choose exactly one"):
        perf_matrix._load_suite_request(arguments)


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


def test_candidate_preflight_requires_modelopt_for_auto_fp8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        perf_matrix,
        "_run_command",
        lambda argv, _environment, timeout: (
            calls.append((argv, timeout))
            or {
                "exit_code": 1,
                "stderr_tail": "ModuleNotFoundError: No module named 'modelopt'",
            }
        ),
    )

    with pytest.raises(
        perf_matrix.PerfMatrixError,
        match="candidate build Python profile 'base' is missing nvidia-modelopt",
    ):
        perf_matrix._candidate_build_dependency_preflight(
            {
                "model": {
                    "build": {
                        "quantization": {
                            "format": "fp8",
                            "scale_source": "modelopt",
                        }
                    }
                }
            },
            profile="base",
            python="/profile/python",
            timeout_seconds=30,
        )

    assert calls == [
        (
            [
                "/profile/python",
                "-c",
                "import modelopt.torch.quantization",
            ],
            30,
        )
    ]


def test_candidate_preflight_skips_modelopt_for_non_calibrated_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        perf_matrix,
        "_run_command",
        lambda *_args, **_kwargs: pytest.fail("dependency probe must not run"),
    )

    perf_matrix._candidate_build_dependency_preflight(
        {
            "model": {
                "build": {
                    "quantization": {
                        "format": "fp8",
                        "scale_source": "precomputed",
                    }
                }
            }
        },
        profile="base",
        python="/profile/python",
        timeout_seconds=30,
    )


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


def test_hf_runner_loads_model_directly_on_cuda() -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/hf_transformers.py"))
    captured: dict[str, object] = {}

    class FakeTorch:
        float16 = "float16"
        float32 = "float32"
        bfloat16 = "bfloat16"

    class FakeModel:
        def eval(self):
            captured["eval"] = True
            return self

        def to(self, _device):
            pytest.fail("direct CUDA loading must not copy a complete CPU model")

    class FakeModelClass:
        @staticmethod
        def from_pretrained(model, **options):
            captured.update(model=model, options=options)
            return FakeModel()

    arguments = Namespace(
        model="example/model",
        precision="fp16",
        experts_implementation=None,
    )
    model = runner["_load_model"](
        FakeModelClass,
        arguments,
        FakeTorch,
        {"local_files_only": True},
    )

    assert isinstance(model, FakeModel)
    assert captured == {
        "model": "example/model",
        "options": {
            "torch_dtype": "float16",
            "low_cpu_mem_usage": True,
            "device_map": "cuda",
            "local_files_only": True,
        },
        "eval": True,
    }


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

    encoded = runner["_chat_prompt_inputs"](FakeTokenizer(), "hello", enable_thinking=False)

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


def test_task_reference_request_seed_is_explicit_and_strict() -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))

    assert runner["_request_seed"]({"seed": 0}) == 0
    assert runner["_request_seed"]({}, 42) == 42
    for invalid in (True, 0.5, "42"):
        with pytest.raises(ValueError, match="request seed must be an integer"):
            runner["_request_seed"]({"seed": invalid})


def test_bark_reference_maps_public_token_cap_to_semantic_stage() -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))

    assert runner["_bark_generation_options"]({"max_new_tokens": 128}) == {
        "semantic_max_new_tokens": 128
    }
    assert runner["_bark_generation_options"]({"max_new_tokens": 0}) == {}


def test_task_reference_local_only_cli_disables_huggingface_network(monkeypatch) -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    captured: dict[str, str | None] = {}

    def fake_run(_arguments):
        captured["hub"] = os.environ.get("HF_HUB_OFFLINE")
        captured["transformers"] = os.environ.get("TRANSFORMERS_OFFLINE")
        return 0

    parser = SimpleNamespace(parse_args=lambda _argv: Namespace(local_files_only=True))
    monkeypatch.setitem(runner["main"].__globals__, "build_parser", lambda: parser)
    monkeypatch.setitem(runner["main"].__globals__, "run", fake_run)

    assert runner["main"]([]) == 0
    assert captured == {"hub": "1", "transformers": "1"}


def test_magpie_reference_maps_public_token_cap_to_decoder_steps() -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    inference_parameters = SimpleNamespace(max_decoder_steps=750)
    model = SimpleNamespace(inference_parameters=inference_parameters)

    runner["_apply_magpie_generation_options"](model, {"max_new_tokens": 256})

    assert inference_parameters.max_decoder_steps == 256


def test_task_reference_runner_measures_loaded_public_operation(
    tmp_path: Path, monkeypatch
) -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    calls: list[int] = []
    environment_calls: list[int] = []

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

    def fake_environment():
        environment_calls.append(1)
        return {"gpu": "fake"}

    run_globals = runner["run"].__globals__
    monkeypatch.setitem(run_globals["LOADERS"], "hf-transformers-asr", load_session)
    monkeypatch.setitem(run_globals, "_synchronize", lambda: None)
    monkeypatch.setitem(run_globals, "_environment", fake_environment)
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
    assert environment_calls == [1]
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


def test_nemotron35_perf_reference_uses_archive_compatible_loader(monkeypatch) -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    from tools.reference import speech

    expected = object()
    captured: dict[str, object] = {}

    def load_compatible(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        speech,
        "load_nemotron35_asr_model",
        load_compatible,
        raising=False,
    )
    arguments = Namespace(
        family="nemotron_speech_streaming",
        model="nvidia/nemotron-3.5-asr-streaming-0.6b",
        revision="model-revision",
        local_files_only=True,
    )

    model = runner["_load_nemo_asr_reference_model"](
        arguments,
        device="cuda",
    )

    assert model is expected
    assert captured == {
        "model": "nvidia/nemotron-3.5-asr-streaming-0.6b",
        "revision": "model-revision",
        "local_files_only": True,
        "device": "cuda",
    }


def test_nemo_asr_perf_reference_disables_rnnt_cuda_graphs() -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    calls = 0

    class GreedyDecoder:
        def disable_cuda_graphs(self) -> bool:
            nonlocal calls
            calls += 1
            return True

    model = Namespace(decoding=Namespace(decoding=GreedyDecoder()))

    assert runner["_disable_nemo_asr_cuda_graphs"](model) is True
    assert calls == 1


def test_vlm_adapter_routes_non_generic_families_to_owned_loaders() -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    deepseek = object()
    locateanything = object()
    globals_ = runner["_load_vlm"].__globals__
    globals_["_load_deepseek_ocr"] = lambda *_args: deepseek
    globals_["_load_locateanything"] = lambda *_args: locateanything

    assert runner["_load_vlm"](Namespace(family="deepseek_ocr"), {}, {}) is deepseek
    assert runner["_load_vlm"](Namespace(family="locateanything"), {}, {}) is locateanything


def test_sam3_reference_reports_source_image_dimensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image

    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    image = tmp_path / "input.png"
    Image.new("RGB", (6, 4)).save(image)

    class FakeTensor:
        def __init__(self, value=None):
            self.value = value

        def is_floating_point(self):
            return False

        def to(self, *_args, **_kwargs):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.value

    class FakeMasks:
        shape = (2, 4, 6)
        ndim = 3

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def __call__(self, **_kwargs):
            return {"original_sizes": FakeTensor([[4, 6]])}

        def post_process_instance_segmentation(self, *_args, **_kwargs):
            return [{"masks": FakeMasks()}]

    class FakeModel:
        config = Namespace(_commit_hash="model-revision")

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def eval(self):
            return self

        def to(self, *_args, **_kwargs):
            return self

        def parameters(self):
            return iter([Namespace(dtype="fp32")])

        def __call__(self, **_kwargs):
            return Namespace()

    fake_torch = ModuleType("torch")
    fake_torch.device = lambda value: value
    fake_torch.float16 = "fp16"
    fake_torch.float32 = "fp32"
    fake_torch.bfloat16 = "bf16"
    fake_torch.inference_mode = nullcontext
    fake_transformers = ModuleType("transformers")
    fake_transformers.Sam3Processor = FakeProcessor
    fake_transformers.Sam3Model = FakeModel
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    arguments = Namespace(
        family="sam3",
        model="facebook/sam3",
        precision="fp32",
        trust_remote_code=False,
        local_files_only=True,
        revision=None,
        manifest=tmp_path / "manifest.json",
    )

    session = runner["_load_vision"](
        arguments,
        {"image_path": str(image), "prompt": "the object"},
        {},
    )

    assert session.invoke() == {"num_masks": 2, "height": 4, "width": 6}


def test_sam3_reference_falls_back_to_processor_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    processor_config = snapshot / "processor_config.json"
    processor_config.write_text(
        json.dumps(
            {
                "target_size": 1008,
                "image_processor": {
                    "image_processor_type": "Sam3ImageProcessorFast",
                    "size": {"height": 1008, "width": 1008},
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeProcessor:
        def __init__(self, image_processor, tokenizer, *, target_size):
            self.image_processor = image_processor
            self.tokenizer = tokenizer
            self.target_size = target_size

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise OSError("preprocessor_config.json is absent")

    class FakeImageProcessor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return "tokenizer"

    fake_transformers = Namespace(
        AutoTokenizer=FakeTokenizer,
        Sam3ImageProcessorFast=FakeImageProcessor,
        Sam3Processor=FakeProcessor,
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        Namespace(try_to_load_from_cache=lambda *_args, **_kwargs: str(processor_config)),
    )

    processor = runner["_load_sam3_processor"](
        fake_transformers,
        "facebook/sam3",
        {"local_files_only": True, "revision": "pinned"},
    )

    assert processor.target_size == 1008
    assert processor.tokenizer == "tokenizer"
    assert processor.image_processor.kwargs == {"size": {"height": 1008, "width": 1008}}


def test_locateanything_fallback_tokenizer_supports_batch_decode(
    tmp_path: Path, monkeypatch
) -> None:
    import tokenizers
    import transformers

    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text('{"model_max_length": 2048}', encoding="utf-8")

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
    (tmp_path / "added_tokens.json").write_text(json.dumps({"<special>": 4}), encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text('{"model_max_length": 512}', encoding="utf-8")

    def fail_auto_tokenizer(*_args, **_kwargs):
        raise OSError("tokenizer.json is not available")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fail_auto_tokenizer)
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
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    blob = tmp_path / "blobs" / "weights"
    blob.parent.mkdir()
    blob.write_bytes(b"weights")
    snapshot = tmp_path / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    checkpoint = snapshot / "model.safetensors"
    checkpoint.symlink_to(blob)

    assert runner["_snapshot_revision"](checkpoint) == "abc123"


def test_task_reference_pinned_checkout_scopes_safe_directory(tmp_path: Path, monkeypatch) -> None:
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


def test_pixart_diffusers_adapter_matches_trtmc_component_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    captured: dict[str, object] = {}

    class FakeTextEncoder:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            captured.update(text_encoder_model=model, text_encoder_kwargs=kwargs)
            return cls()

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            captured.update(pipeline_model=model, pipeline_kwargs=kwargs)
            return cls()

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        Namespace(PixArtSigmaPipeline=FakePipeline, DiffusionPipeline=FakePipeline),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        Namespace(T5EncoderModel=FakeTextEncoder),
    )
    arguments = Namespace(
        family="pixart",
        local_files_only=False,
        model="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        precision="fp16",
        revision="snapshot",
        trust_remote_code=False,
    )
    torch_module = Namespace(float16="fp16", float32="fp32", bfloat16="bf16")
    options = {"component_precision_contract": "pixart_fp16_dit_fp32_t5"}

    runner["_diffusion_pipeline"](arguments, torch_module, options)

    assert captured["text_encoder_model"] == arguments.model
    assert captured["text_encoder_kwargs"] == {
        "subfolder": "text_encoder",
        "torch_dtype": "fp32",
        "revision": "snapshot",
        "local_files_only": False,
    }
    assert captured["pipeline_model"] == arguments.model
    assert captured["pipeline_kwargs"]["torch_dtype"] == "fp16"
    assert isinstance(captured["pipeline_kwargs"]["text_encoder"], FakeTextEncoder)


def test_pixart_component_precision_registers_fp16_input_and_fp32_output_hooks() -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))

    class FakeTransformer:
        def __init__(self) -> None:
            self.pre_hook = None
            self.post_hook = None
            self.with_kwargs = False

        def register_forward_pre_hook(self, hook, *, with_kwargs=False):
            self.pre_hook = hook
            self.with_kwargs = with_kwargs

        def register_forward_hook(self, hook):
            self.post_hook = hook

    transformer = FakeTransformer()
    pipeline = Namespace(transformer=transformer)
    arguments = Namespace(family="pixart", precision="fp16")
    torch_module = Namespace(float16="fp16", float32="fp32", Tensor=object)

    runner["_configure_diffusion_component_precision"](
        pipeline,
        arguments,
        {"component_precision_contract": "pixart_fp16_dit_fp32_t5"},
        torch_module,
    )

    assert transformer.with_kwargs is True
    assert callable(transformer.pre_hook)
    assert callable(transformer.post_hook)


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
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))

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


def test_diffusers_adapter_requests_numeric_output_before_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    captured: dict[str, object] = {}

    class FakePipeline:
        def to(self, _device):
            return self

        def __call__(self, *, prompt, output_type):
            captured.update(prompt=prompt, output_type=output_type)
            return Namespace(images=np.zeros((1, 4, 6, 3), dtype=np.float32))

    globals_ = runner["_load_diffusers"].__globals__
    globals_["_diffusion_pipeline"] = lambda *_args: FakePipeline()
    globals_["_resolved_revision"] = lambda *_args: "snapshot"
    arguments = Namespace(
        family="z_image",
        precision="bf16",
        model="Tongyi-MAI/Z-Image-Turbo",
        revision=None,
    )

    session = runner["_load_diffusers"](
        arguments,
        {"prompt": "cat", "media_type": "image"},
        {},
    )

    assert session.invoke()["finite"] is True
    assert captured == {"prompt": "cat", "output_type": "np"}


def test_diffusers_media_summary_rejects_non_finite_pixels() -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    finite = np.zeros((1, 4, 6, 3), dtype=np.float32)

    assert runner["_media_summary"](finite, "image") == {
        "media_type": "image",
        "media_count": 1,
        "height": 4,
        "width": 6,
        "channels": 3,
        "finite": True,
    }

    invalid = finite.copy()
    invalid[0, 0, 0, 0] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        runner["_media_summary"](invalid, "image")


def test_personaplex_loader_adds_vendored_moshi_package_root() -> None:
    source = (REPOSITORY / "benchmarks/performance/baselines/task_reference.py").read_text()

    assert 'str(Path(official_repo) / "moshi")' in source
    assert "personaplex_audio_compat" in source


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
    source = (REPOSITORY / "benchmarks/performance/baselines/task_reference.py").read_text(
        encoding="utf-8"
    )

    assert 'speaker=str(options.get("speaker", "Ethan"))' in source
    assert 'spk=str(options.get("speaker", "Ethan"))' not in source


def test_qwen3_omni_uses_visible_single_gpu_placement(monkeypatch) -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
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

    cases = performance_catalog.load_suite(SUITE).cases
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


def test_lance_reference_loads_sdpa_attention_compat(tmp_path: Path, monkeypatch) -> None:
    runner = runpy.run_path(str(REPOSITORY / "tools/lance_reference.py"))
    reference_repo = tmp_path / "Lance"
    reference_repo.mkdir()
    (reference_repo / "inference_lance.py").write_text(
        "import flash_attn\nATTENTION = flash_attn.flash_attn_varlen_func.__module__\n",
        encoding="utf-8",
    )
    monkeypatch.delitem(sys.modules, "flash_attn", raising=False)

    upstream = runner["_load_upstream"](reference_repo)

    assert upstream.ATTENTION.endswith("flash_attn")
    assert "lance_image_attention_compat" in sys.modules["flash_attn"].__file__


def test_lance_reference_selects_sdpa_in_a_run_owned_vit_view(tmp_path: Path) -> None:
    runner = runpy.run_path(str(REPOSITORY / "tools/lance_reference.py"))
    vit_path = tmp_path / "Qwen2.5-VL-ViT"
    vit_path.mkdir()
    config = vit_path / "config.json"
    config.write_text('{"_attn_implementation": "flash_attention_2"}\n')
    weights = vit_path / "vit.safetensors"
    weights.write_bytes(b"weights")

    overlay = runner["_sdpa_vit_path"](tmp_path / "run", vit_path)

    assert json.loads((overlay / "config.json").read_text())["_attn_implementation"] == "sdpa"
    assert (overlay / "vit.safetensors").resolve() == weights.resolve()
    assert json.loads(config.read_text())["_attn_implementation"] == "flash_attention_2"


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
                    "output_summary": {
                        "frame_count": 321,
                        "shape": [321, 704, 1280, 3],
                    },
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
