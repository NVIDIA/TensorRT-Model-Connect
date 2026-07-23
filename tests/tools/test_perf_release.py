# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType

import yaml

from tools import perf_release


REPOSITORY = Path(__file__).resolve().parents[2]
SUITE = REPOSITORY / "benchmarks/performance/release.yaml"
TASK_ADAPTERS = {
    "bark.generate_audio": "hf-transformers-tts",
    "canary.transcribe": "nemo-asr",
    "chronos_bolt.solve": "pytorch-timeseries",
    "deepseek_ocr.generate": "hf-transformers-vlm",
    "eagle_vlm.embed": "hf-transformers-embedding",
    "eagle_vlm.rerank": "hf-transformers-reranking",
    "elf_flow.generate": "upstream-elf",
    "flux.generate_image": "hf-diffusers",
    "internvl.generate": "hf-transformers-vlm",
    "lance.generate": "upstream-lance",
    "locateanything.generate": "hf-transformers-vlm",
    "ltx_video.generate_image": "hf-diffusers",
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
    "sana_wm.generate_image": "hf-diffusers",
    "segformer.segment": "hf-transformers-vision",
    "timesfm.solve": "pytorch-timeseries",
    "timm_vit.classify": "hf-transformers-vision",
    "wan_t2v.generate_image": "hf-diffusers",
    "whisper.transcribe": "hf-transformers-asr",
    "z_image.generate_image": "hf-diffusers",
}


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
    diffusion_baseline = by_id["nemotron_labs_diffusion.generate"]["baseline"]
    assert diffusion_baseline["mode"] == "hf-eager"
    assert diffusion_baseline["model_class"] == "auto"
    assert diffusion_baseline["generation_method"] == "ar-generate"


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

    assert perf_release._output_contract(case, candidate, reference) == (True, "")

    candidate["output_summary"]["text"] = "OCR title only"
    matched, reason = perf_release._output_contract(case, candidate, reference)
    assert not matched
    assert reason == "TRTMC OCR text misses required content"


def test_normalized_text_contract_allows_only_case_and_whitespace_variation() -> None:
    case = {
        "operation": "generate",
        "baseline": {"output_contract": "normalized-text"},
    }
    candidate = {"output_summary": {"text": "Paris\n</think>\n"}}
    reference = {"output_summary": {"text": "paris  </think>"}}

    assert perf_release._output_contract(case, candidate, reference) == (True, "")

    reference["output_summary"]["text"] = "London </think>"
    assert perf_release._output_contract(case, candidate, reference) == (
        False,
        "normalized generated text differs",
    )


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
    assert ">gpt2<" in report
    assert "HF eager" in report
    assert ">10.5<" not in report
    assert ">20.0<" not in report
    assert "Show raw commands" in report
    assert str(fake_trtmc) in report
    assert str(fake_baseline) in report
    assert "reproduce.py" not in report

    baseline_argv = rows["gpt2.generate"]["commands"]["baseline"]["argv"]
    request = baseline_argv[baseline_argv.index("--request-json") + 1]
    assert json.loads(request)["prompt"] == "Hello, I'm a language model"


def test_suite_has_explicit_eager_and_task_reference_rows() -> None:
    raw = yaml.safe_load(SUITE.read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in raw["cases"]}

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
    assert not any(row["baseline"]["runner"] == "unsupported" for row in rows.values())
    assert {
        case_id: row["baseline"].get("adapter")
        for case_id, row in rows.items()
        if row["baseline"]["runner"] == "task-reference"
    } == TASK_ADAPTERS
    for case_id in TASK_ADAPTERS:
        assert rows[case_id]["baseline"]["reference_backend"]
        assert rows[case_id]["baseline"]["mode"] in {"hf-eager", "pytorch-eager"}


def test_resolution_failure_is_recorded_per_case(tmp_path: Path, monkeypatch) -> None:
    case = perf_release._cases(perf_release._read_yaml(SUITE))[0]
    options = perf_release.RunOptions(
        output=tmp_path,
        trtmc_bench="trtmc-bench",
        trtmc_worker=None,
        hf_transformers_runner=tmp_path / "baseline.py",
        task_reference_runner=tmp_path / "task_reference.py",
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


def test_hf_runner_bridges_removed_input_check_decorator() -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/hf_transformers.py"))

    class GenericModule:
        pass

    def forward() -> str:
        return "ok"

    runner["_ensure_transformers_generic_api"](GenericModule)

    assert GenericModule.check_model_inputs(forward) is forward


def test_source_revision_can_be_injected_without_git(monkeypatch) -> None:
    monkeypatch.setenv("TRTMC_PERF_SOURCE_REVISION", "tested-commit")

    assert perf_release._git_commit() == "tested-commit"


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
    assert result["model_load_included"] is False
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
    fake_tokenizers = ModuleType("tokenizers")
    fake_tokenizers.Tokenizer = Namespace(from_file=lambda _path: FakeRawTokenizer())
    monkeypatch.setitem(sys.modules, "tokenizers", fake_tokenizers)
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


def test_task_reference_resolves_revision_from_hugging_face_cache(monkeypatch) -> None:
    runner = runpy.run_path(str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py"))
    revision = Namespace(commit_hash="abc123", refs=frozenset({"main"}))
    repository = Namespace(repo_id="nvidia/canary-1b-v2", revisions=[revision])
    cache = Namespace(repos=[repository])
    fake_hub = Namespace(scan_cache_dir=lambda: cache)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    assert runner["_cached_snapshot_revision"]("nvidia/canary-1b-v2", None) == "abc123"


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
    assert summary == {"media_type": "video", "media_count": 2}


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
        dataset,
        tmp_path / "results",
    )

    assert len(samples) == 2
    assert all(value > 0 for value in samples)
    assert answers == ["blue", "blue"]


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
