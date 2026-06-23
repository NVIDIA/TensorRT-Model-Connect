from __future__ import annotations

from tests.e2e_harness.contracts import E2ECase, StageOutput, StageStatus, ThresholdProfile
from tests.e2e.models.nemotron_labs_diffusion.e2e_plugins.contract import plugin


def _case() -> E2ECase:
    return E2ECase(
        name="nemotron-labs-diffusion-8b-linear-spec",
        hf_id="nvidia/Nemotron-Labs-Diffusion-8B",
        family="nemotron_labs_diffusion",
        runtime_strategy="nemotron_labs_diffusion",
        task_strategy="text_generation_causal",
        reference_family="nemotron_labs_diffusion_model_card",
        user_contract="model_card_generation_parity",
        metadata={
            "contract_config": {
                "token_parity_eos_token_ids": [11],
                "token_parity_ignore_terminal_token_ids": [1010],
                "forbidden_token_ids": [0],
            }
        },
    )


def test_nemotron_labs_diffusion_contract_accepts_terminal_newline_drift() -> None:
    trt = StageOutput(
        stage_name="full_generation",
        data={"token_ids": [42572, 1010, 13, 1010, 11]},
        text="Paris\n</think>\n",
    )
    ref = StageOutput(
        stage_name="full_generation",
        data={"token_ids": [42572, 1010, 13, 11], "eos_token_id": 11},
        text="Paris\n</think>",
    )

    result = plugin.verify(
        trt,
        ref,
        _case(),
        ThresholdProfile(
            task_strategy="text_generation_causal",
            metrics={"canonical_token_agreement_rate": 1.0},
        ),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["generated_token_raw_exact"].value == 0.0
    assert result.metrics["generated_token_raw_exact"].passed
    assert result.metrics["generated_token_canonical_exact"].passed


def test_nemotron_labs_diffusion_contract_rejects_forbidden_unknown_token() -> None:
    trt = StageOutput(
        stage_name="full_generation",
        data={"token_ids": [0, 1010, 42572, 11]},
        text="\nParis",
    )
    ref = StageOutput(
        stage_name="full_generation",
        data={"token_ids": [42572, 11], "eos_token_id": 11},
        text="Paris",
    )

    result = plugin.verify(
        trt,
        ref,
        _case(),
        ThresholdProfile(task_strategy="text_generation_causal"),
    )

    assert result.status == StageStatus.FAILED.value
    assert not result.metrics["forbidden_token_count"].passed
