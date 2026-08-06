# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tests.e2e_harness.contracts import (
    E2ECase,
    RunContext,
    StageOutput,
    StageSpec,
    ThresholdProfile,
)
from tests.e2e_harness.manifest_loader import find_manifest_path, load_manifest
from tests.e2e_harness.plugins import find_plugin
from tests.e2e_harness.registry import (
    activate_model_plugins,
    get_comparator,
    get_runner,
    reset,
)
from tests.e2e.models.elf_flow.e2e_plugins.runners import diffusion_text_generation


REPO_ROOT = Path(__file__).resolve().parents[4]
ELF_MODEL_DIR = REPO_ROOT / "tests" / "e2e" / "models" / "elf_flow"


@pytest.fixture(autouse=True)
def _activate_elf_plugins():
    reset()
    activate_model_plugins(ELF_MODEL_DIR)
    yield
    reset()


def _case(reference_family: str, inputs: dict | None = None) -> E2ECase:
    return E2ECase(
        name="elf-contract",
        hf_id="github.com/lillian039/ELF",
        family="elf_flow",
        runtime_strategy="elf_flow",
        reference_backend="invariant_only",
        reference_family=reference_family,
        user_contract="diffusion_text_generation",
        inputs=inputs or {},
    )


def test_elf_contract_plugin_discovers_for_unconditional_and_conditional() -> None:
    uncond = find_plugin("elf_unconditional_text")
    cond = find_plugin("elf_conditional_text")

    assert uncond is not None
    assert cond is uncond
    assert uncond.user_contract == "diffusion_text_generation"
    assert (
        uncond.configure_reference(_case("elf_unconditional_text"))["output_schema"]
        == "jsonl_id_generated_token_ids"
    )


def test_diffusion_text_generation_runner_and_comparator_are_registered() -> None:
    runner = get_runner("diffusion_text_generation")
    comparator = get_comparator("diffusion_text_generation")

    assert runner is not None
    assert comparator is not None


def test_elf_unconditional_contract_accepts_jsonl_samples_and_metrics() -> None:
    plugin = find_plugin("elf_unconditional_text")
    assert plugin is not None

    trt = StageOutput(
        stage_name="decoded_text",
        data={
            "generated_jsonl": '{"id": 0, "generated": "A complete generated sentence."}\n',
            "gen_ppl": 24.0,
            "unigram_entropy": 5.16,
        },
    )
    threshold = ThresholdProfile(
        task_strategy="neural_operator",
        metrics={
            "contract_min_samples": 1,
            "contract_max_gen_ppl": 24.5,
            "contract_min_unigram_entropy": 5.0,
        },
    )

    result = plugin.verify(
        trt,
        StageOutput("decoded_text"),
        _case("elf_unconditional_text", {"generation_mode": "unconditional"}),
        threshold,
    )

    assert result.passed
    assert result.metrics["num_generated_samples"].passed
    assert result.metrics["gen_ppl"].passed
    assert result.metrics["unigram_entropy"].passed


def test_elf_unconditional_contract_enforces_expected_sample_count() -> None:
    plugin = find_plugin("elf_unconditional_text")
    assert plugin is not None

    case = _case("elf_unconditional_text", {"generation_mode": "unconditional"})
    threshold = ThresholdProfile(
        task_strategy="neural_operator",
        metrics={"contract_min_samples": 1, "contract_expected_samples": 2},
    )

    passing = plugin.verify(
        StageOutput(
            stage_name="decoded_text",
            data={
                "generated_samples": [
                    {"id": 0, "generated": "First generated sentence."},
                    {"id": 1, "generated": "Second generated sentence."},
                ]
            },
        ),
        StageOutput("decoded_text"),
        case,
        threshold,
    )
    failing = plugin.verify(
        StageOutput(
            stage_name="decoded_text",
            data={
                "generated_samples": [
                    {"id": 0, "generated": "Only one generated sentence."}
                ]
            },
        ),
        StageOutput("decoded_text"),
        case,
        threshold,
    )

    assert passing.passed
    assert passing.metrics["expected_generated_sample_count"].passed
    assert not failing.passed
    assert not failing.metrics["expected_generated_sample_count"].passed


def test_elf_conditional_contract_requires_condition_and_generated_text() -> None:
    plugin = find_plugin("elf_conditional_text")
    assert plugin is not None

    trt = StageOutput(
        stage_name="decoded_text",
        data={
            "generated_samples": [
                {
                    "id": 0,
                    "source": "Ein kurzer deutscher Satz.",
                    "generated": "A short German sentence.",
                }
            ],
            "bleu": 32.0,
            "rougeL": 0.41,
        },
    )
    threshold = ThresholdProfile(
        task_strategy="neural_operator",
        metrics={
            "contract_min_samples": 1,
            "contract_min_bleu": 1.0,
            "contract_min_rouge_l": 0.1,
        },
    )

    result = plugin.verify(
        trt,
        StageOutput("decoded_text"),
        _case(
            "elf_conditional_text",
            {
                "generation_mode": "conditional",
                "condition_latents_path": "/tmp/cond.f32",
                "condition_mask_path": "/tmp/mask.f32",
            },
        ),
        threshold,
    )

    assert result.passed
    assert result.metrics["condition_available"].passed
    assert result.metrics["bleu"].passed
    assert result.metrics["rouge_l"].passed


def test_elf_conditional_contract_accepts_source_text_prompt_condition() -> None:
    plugin = find_plugin("elf_conditional_text")
    assert plugin is not None

    result = plugin.verify(
        StageOutput(
            stage_name="decoded_text",
            data={
                "generated_samples": [
                    {"id": 0, "source": "Ein kurzer deutscher Satz.", "generated": "A sentence."}
                ]
            },
        ),
        StageOutput("decoded_text"),
        _case(
            "elf_conditional_text",
            {"generation_mode": "conditional", "source_text": "Ein kurzer deutscher Satz."},
        ),
        ThresholdProfile(task_strategy="neural_operator", metrics={"contract_min_samples": 1}),
    )

    assert result.passed
    assert result.metrics["condition_available"].passed


def test_elf_contract_compares_upstream_replay_expected_text() -> None:
    plugin = find_plugin("elf_unconditional_text")
    assert plugin is not None

    result = plugin.verify(
        StageOutput(
            stage_name="decoded_text",
            data={
                "generated_samples": [
                    {"id": 0, "generated": "A replayed sentence.", "token_ids": [7, 8, 9]}
                ],
                "expected_generated_samples": [
                    {"id": 0, "generated": "A replayed sentence.", "token_ids": [7, 8, 9]}
                ],
            },
        ),
        StageOutput("decoded_text"),
        _case("elf_unconditional_text", {"generation_mode": "unconditional"}),
        ThresholdProfile(task_strategy="neural_operator", metrics={"contract_min_samples": 1}),
    )

    assert result.passed
    assert result.metrics["upstream_text_match_rate"].passed
    assert result.metrics["upstream_token_id_match_rate"].passed


def test_elf_contract_rejects_upstream_replay_text_mismatch() -> None:
    plugin = find_plugin("elf_unconditional_text")
    assert plugin is not None

    result = plugin.verify(
        StageOutput(
            stage_name="decoded_text",
            data={
                "generated_samples": [{"id": 0, "generated": "A C++ sentence."}],
                "expected_generated_samples": [{"id": 0, "generated": "An upstream sentence."}],
            },
        ),
        StageOutput("decoded_text"),
        _case("elf_unconditional_text", {"generation_mode": "unconditional"}),
        ThresholdProfile(task_strategy="neural_operator", metrics={"contract_min_samples": 1}),
    )

    assert not result.passed
    assert not result.metrics["upstream_text_match_rate"].passed


def test_elf_contract_rejects_upstream_replay_token_mismatch() -> None:
    plugin = find_plugin("elf_unconditional_text")
    assert plugin is not None

    result = plugin.verify(
        StageOutput(
            stage_name="decoded_text",
            data={
                "generated_samples": [
                    {"id": 0, "generated": "Same text.", "token_ids": [1, 2, 3]}
                ],
                "expected_generated_samples": [
                    {"id": 0, "generated": "Same text.", "token_ids": [1, 2, 4]}
                ],
            },
        ),
        StageOutput("decoded_text"),
        _case("elf_unconditional_text", {"generation_mode": "unconditional"}),
        ThresholdProfile(task_strategy="neural_operator", metrics={"contract_min_samples": 1}),
    )

    assert not result.passed
    assert not result.metrics["upstream_token_id_match_rate"].passed


def test_elf_contract_applies_bounded_upstream_replay_thresholds() -> None:
    plugin = find_plugin("elf_unconditional_text")
    assert plugin is not None

    result = plugin.verify(
        StageOutput(
            stage_name="decoded_text",
            data={
                "generated_samples": [
                    {
                        "id": 0,
                        "generated": "Same texts.",
                        "token_ids": [1, 2, 9, 4],
                    }
                ],
            },
        ),
        StageOutput(
            stage_name="decoded_text",
            data={
                "expected_generated_samples": [
                    {
                        "id": 0,
                        "generated": "Same text.",
                        "token_ids": [1, 2, 3, 4],
                    }
                ],
            },
        ),
        _case("elf_unconditional_text", {"generation_mode": "unconditional"}),
        ThresholdProfile(
            task_strategy="neural_operator",
            metrics={
                "contract_min_samples": 1,
                "contract_max_upstream_text_ned": 0.1,
                "contract_min_upstream_token_agreement_rate": 0.75,
            },
        ),
    )

    assert result.passed
    assert result.metrics["upstream_text_ned"].passed
    assert result.metrics["upstream_token_id_agreement_rate"].passed


def test_elf_contract_ignores_only_repeated_terminal_tokens() -> None:
    plugin = find_plugin("elf_conditional_text")
    assert plugin is not None

    result = plugin.verify(
        StageOutput(
            stage_name="decoded_text",
            data={
                "generated_samples": [
                    {"id": 0, "generated": "Same text.", "token_ids": [7, 8, 9]}
                ],
            },
        ),
        StageOutput(
            stage_name="decoded_text",
            data={
                "expected_generated_samples": [
                    {
                        "id": 0,
                        "generated": "Same text.",
                        "token_ids": [7, 8, 9, 1, 1],
                    }
                ],
                "terminal_token_ids": [0, 1],
            },
        ),
        _case(
            "elf_conditional_text",
            {
                "generation_mode": "conditional",
                "source_text": "A source sentence.",
            },
        ),
        ThresholdProfile(
            task_strategy="neural_operator",
            metrics={"contract_min_samples": 1},
        ),
    )

    assert result.passed
    assert result.metrics["upstream_token_id_match_rate"].passed


def test_diffusion_text_runner_requires_prompt_or_api_condition_for_conditional(
    tmp_path: Path,
) -> None:
    runner = get_runner("diffusion_text_generation")
    assert runner is not None
    case = _case(
        "elf_conditional_text",
        {"generation_mode": "conditional"},
    )

    with pytest.raises(RuntimeError, match="requires either prompt/source_text"):
        runner.run_stage(
            case,
            StageSpec(name="decoded_text"),
            RunContext(
                case=case,
                artifacts_dir=str(tmp_path),
                binary_path="/bin/false",
                engine_dir=str(tmp_path),
            ),
        )


def test_diffusion_text_runner_uses_source_text_prompt_condition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(
        "elf_conditional_text",
        {
            "generation_mode": "conditional",
            "source_text": "Ein kurzer deutscher Satz.",
            "num_sampling_steps": 64,
        },
    )
    captured: dict[str, list[str]] = {}

    def _fake_run(cmd, **kwargs):
        del kwargs
        captured["cmd"] = cmd
        out_path = Path(cmd[cmd.index("--output") + 1])
        out_path.write_text(
            '{"id": 0, "generated": "A short German sentence.", "token_ids": [7]}\n',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(diffusion_text_generation.subprocess, "run", _fake_run)
    runner = diffusion_text_generation.DiffusionTextGenerationRunner()
    out = runner.run_stage(
        case,
        StageSpec(name="decoded_text"),
        RunContext(
            case=case,
            artifacts_dir=str(tmp_path / "artifacts"),
            binary_path="/bin/trtmc",
            engine_dir=str(tmp_path),
        ),
    )

    cmd = captured["cmd"]
    assert "--prompt" in cmd
    assert cmd[cmd.index("--prompt") + 1] == "Ein kurzer deutscher Satz."
    assert "--condition-latents-raw" not in cmd
    assert "--condition-mask-raw" not in cmd
    assert out.data["generated_samples"][0]["generated"] == "A short German sentence."


def test_diffusion_text_runner_consumes_upstream_replay_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "initial.f32",
        "steps.f32",
        "sde.f32",
        "cond.f32",
        "mask.f32",
    ):
        (tmp_path / name).write_bytes(b"\0\0\0\0")
    (tmp_path / "expected.jsonl").write_text(
        '{"id": 0, "generated": "A replayed sentence.", "token_ids": [7, 8, 9]}\n',
        encoding="utf-8",
    )
    artifact = tmp_path / "elf_replay.json"
    artifact.write_text(
        json.dumps(
            {
                "generation_mode": "conditional",
                "num_sampling_steps": 64,
                "self_cond_cfg_scale": 1.0,
                "cfg_scale": 2.0,
                "sde_gamma": 0.0,
                "files": {
                    "initial_latents_raw": "initial.f32",
                    "sampling_steps_raw": "steps.f32",
                    "sde_noise_raw": "sde.f32",
                    "condition_latents_raw": "cond.f32",
                    "condition_mask_raw": "mask.f32",
                    "expected_generated_jsonl_path": "expected.jsonl",
                },
            }
        ),
        encoding="utf-8",
    )
    case = _case(
        "elf_conditional_text",
        {"elf_replay_artifact": str(artifact)},
    )
    captured: dict[str, list[str]] = {}

    def _fake_run(cmd, **kwargs):
        del kwargs
        captured["cmd"] = cmd
        out_path = Path(cmd[cmd.index("--output") + 1])
        out_path.write_text(
            '{"id": 0, "generated": "A replayed sentence.", "token_ids": [7, 8, 9]}\n',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(diffusion_text_generation.subprocess, "run", _fake_run)
    runner = diffusion_text_generation.DiffusionTextGenerationRunner()
    out = runner.run_stage(
        case,
        StageSpec(name="decoded_text"),
        RunContext(
            case=case,
            artifacts_dir=str(tmp_path / "artifacts"),
            binary_path="/bin/trtmc",
            engine_dir=str(tmp_path),
        ),
    )

    cmd = captured["cmd"]
    assert "--initial-latents-raw" in cmd
    assert str(tmp_path / "initial.f32") in cmd
    assert "--sampling-steps-raw" in cmd
    assert str(tmp_path / "steps.f32") in cmd
    assert "--sde-noise-raw" in cmd
    assert str(tmp_path / "sde.f32") in cmd
    assert "--condition-latents-raw" in cmd
    assert str(tmp_path / "cond.f32") in cmd
    assert "--condition-mask-raw" in cmd
    assert str(tmp_path / "mask.f32") in cmd
    assert out.data["expected_generated_samples"][0]["generated"] == "A replayed sentence."

    plugin = find_plugin("elf_conditional_text")
    assert plugin is not None
    result = plugin.verify(
        out,
        StageOutput("decoded_text"),
        case,
        ThresholdProfile(task_strategy="neural_operator", metrics={"contract_min_samples": 1}),
    )
    assert result.passed
    assert result.metrics["condition_available"].passed
    assert result.metrics["upstream_text_match_rate"].passed
    assert result.metrics["upstream_token_id_match_rate"].passed


def test_diffusion_text_runner_consumes_multi_sample_replay_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "initial0.f32",
        "initial1.f32",
        "steps.f32",
        "sde0.f32",
        "sde1.f32",
    ):
        (tmp_path / name).write_bytes(b"\0\0\0\0")
    (tmp_path / "expected.jsonl").write_text(
        "\n".join(
            [
                '{"id": 0, "generated": "Replay zero.", "token_ids": [1]}',
                '{"id": 1, "generated": "Replay one.", "token_ids": [2]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    artifact = tmp_path / "elf_replay.json"
    artifact.write_text(
        json.dumps(
            {
                "generation_mode": "unconditional",
                "num_samples": 2,
                "num_sampling_steps": 32,
                "files": {
                    "sampling_steps_raw": "steps.f32",
                    "expected_generated_jsonl_path": "expected.jsonl",
                },
                "samples": [
                    {
                        "id": 0,
                        "files": {
                            "initial_latents_raw": "initial0.f32",
                            "sde_noise_raw": "sde0.f32",
                        },
                    },
                    {
                        "id": 1,
                        "files": {
                            "initial_latents_raw": "initial1.f32",
                            "sde_noise_raw": "sde1.f32",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    case = _case("elf_unconditional_text", {"elf_replay_artifact": str(artifact)})
    captured: dict[str, list[list[str]]] = {"cmds": []}

    def _fake_run(cmd, **kwargs):
        del kwargs
        sample_idx = len(captured["cmds"])
        captured["cmds"].append(cmd)
        out_path = Path(cmd[cmd.index("--output") + 1])
        out_path.write_text(
            json.dumps(
                {
                    "id": 0,
                    "generated": f"Replay {'zero' if sample_idx == 0 else 'one'}.",
                    "token_ids": [sample_idx + 1],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(diffusion_text_generation.subprocess, "run", _fake_run)
    runner = diffusion_text_generation.DiffusionTextGenerationRunner()
    out = runner.run_stage(
        case,
        StageSpec(name="decoded_text"),
        RunContext(
            case=case,
            artifacts_dir=str(tmp_path / "artifacts"),
            binary_path="/bin/trtmc",
            engine_dir=str(tmp_path),
        ),
    )

    assert len(captured["cmds"]) == 2
    assert all("--num-samples" not in cmd for cmd in captured["cmds"])
    assert str(tmp_path / "initial0.f32") in captured["cmds"][0]
    assert str(tmp_path / "initial1.f32") in captured["cmds"][1]
    assert [sample["id"] for sample in out.data["generated_samples"]] == [0, 1]

    plugin = find_plugin("elf_unconditional_text")
    assert plugin is not None
    result = plugin.verify(
        out,
        StageOutput("decoded_text"),
        case,
        ThresholdProfile(
            task_strategy="neural_operator",
            metrics={"contract_min_samples": 1, "contract_expected_samples": 2},
        ),
    )
    assert result.passed
    assert result.metrics["expected_generated_sample_count"].passed
    assert result.metrics["upstream_text_match_rate"].passed
    assert result.metrics["upstream_token_id_match_rate"].passed


def test_elf_contract_rejects_empty_generation() -> None:
    plugin = find_plugin("elf_unconditional_text")
    assert plugin is not None

    result = plugin.verify(
        StageOutput(stage_name="decoded_text", data={"generated_samples": [{"generated": ""}]}),
        StageOutput("decoded_text"),
        _case("elf_unconditional_text"),
        ThresholdProfile(task_strategy="neural_operator", metrics={"contract_min_samples": 1}),
    )

    assert not result.passed
    assert not result.metrics["non_empty_generated_text"].passed


def test_elf_l0_manifests_use_upstream_replay_contract() -> None:
    for name, family in (
        ("elf-b-owt-l0", "elf_unconditional_text"),
        ("elf-b-xsum-l0", "elf_conditional_text"),
        ("elf-b-de-en-l0", "elf_conditional_text"),
    ):
        manifest_path = find_manifest_path(
            name,
            REPO_ROOT / "tests" / "e2e" / "models",
        )
        assert manifest_path is not None
        case = load_manifest(manifest_path)

        assert case.family == "elf_flow"
        assert case.runtime_strategy == "elf_flow"
        assert case.task_strategy == "diffusion_text_generation"
        assert case.reference_family == family
        assert case.user_contract == "diffusion_text_generation"
        assert case.oracle_level == "L1_external_reference"
        assert case.reference_backend == "upstream_replay"
        assert "token_ids" in case.inputs["output_schema"]
        assert "skip_reason" not in case.metadata
        assert Path(case.inputs["elf_replay_artifact"]).is_file()
        assert any(stage.artifact_type == "text_samples" for stage in case.stages)
        if name == "elf-b-owt-l0":
            assert case.metadata["e2e_parallel_resource"] == "exclusive_gpu"
            build_env = case.metadata["build_env"]
            assert build_env["TRTMC_BUILDER_OPTIMIZATION_LEVEL"] == "1"
            assert build_env["TRTMC_MAX_NUM_TACTICS"] == ""
            assert build_env["TRTMC_AVG_TIMING_ITERATIONS"] == ""
            assert build_env["TRTMC_TRT_TIMING_CACHE_PATH"] == ""
            assert build_env["TRTMC_TRT_TIMING_CACHE_DIR"] == ""
            assert build_env["TRTMC_ELF_TIMING_CACHE_GENERATE"] == "0"
            assert build_env["TRTMC_ELF_TIMING_CACHE_PATH"] == ""
            assert build_env["TRTMC_ELF_TIMING_CACHE_METADATA_PATH"] == ""
