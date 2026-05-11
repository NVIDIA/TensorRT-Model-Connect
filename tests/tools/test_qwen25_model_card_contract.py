from tests.e2e_harness.contracts import E2ECase, StageOutput, ThresholdProfile
from tests.e2e_harness.manifest_loader import load_manifest
from tests.e2e_harness.plugins.qwen25_model_card import Qwen25ModelCardPlugin


def _case() -> E2ECase:
    return E2ECase(
        name="qwen2.5-0.5b-torchtrt",
        hf_id="Qwen/Qwen2.5-0.5B",
        family="qwen",
        runtime_strategy="decoder_kv_cache",
        task_strategy="text_generation_causal",
        reference_family="qwen25_base_model_card",
        user_contract="chat_response",
        bundle="qwen2.5-0.5b-torchtrt.trtfb",
        inputs={"prompt": "Who are you?", "max_new_tokens": 40},
    )


def _out(text: str, command: list[str]) -> StageOutput:
    return StageOutput(
        stage_name="full_generation",
        text=text,
        metadata={"cpp": {"command": command}},
    )


def test_manifest_uses_model_card_chat_template_contract() -> None:
    case = load_manifest("tests/e2e/models/qwen2.5-0.5b-torchtrt.json")
    assert case.inputs["prompt"] == "Who are you?"
    assert case.inputs["max_new_tokens"] == 40
    assert case.reference_family == "qwen25_base_model_card"
    assert case.user_contract == "chat_response"
    assert case.metadata["build_args"]["backend"] == "torchtrt"


def test_configure_reference_matches_model_card() -> None:
    config = Qwen25ModelCardPlugin().configure_reference(_case())
    assert config == {"use_chat_template": True, "enable_thinking": True}


def test_verify_requires_chat_template_without_no_thinking() -> None:
    plugin = Qwen25ModelCardPlugin()
    case = _case()
    threshold = ThresholdProfile(task_strategy="text_generation_causal")
    ref = _out("I am Qwen.", [])

    good = _out("I am Qwen.", ["trtmc", "run", "--chat-template"])
    assert plugin.verify(good, ref, case, threshold).status == "passed"

    missing_template = _out("I am Qwen.", ["trtmc", "run"])
    assert plugin.verify(missing_template, ref, case, threshold).status == "failed"

    no_thinking = _out("I am Qwen.", ["trtmc", "run", "--chat-template", "--no-thinking"])
    assert plugin.verify(no_thinking, ref, case, threshold).status == "failed"


def test_verify_rejects_empty_or_divergent_output() -> None:
    plugin = Qwen25ModelCardPlugin()
    case = _case()
    threshold = ThresholdProfile(task_strategy="text_generation_causal")
    command = ["trtmc", "run", "--chat-template"]
    ref = _out("I am Qwen, an AI language model.", [])

    empty = _out("", command)
    assert plugin.verify(empty, ref, case, threshold).status == "failed"

    divergent = _out("The largest ocean is the Pacific.", command)
    assert plugin.verify(divergent, ref, case, threshold).status == "failed"
