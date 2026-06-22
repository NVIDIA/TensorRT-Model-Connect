from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_qwen3_omni_thinker_marks_hidden_state_output() -> None:
    source = (
        ROOT
        / "python/tensorrt_model_connect/families/qwen3_omni/plugin.py"
    ).read_text()

    assert 'hidden_out.name = "hidden_state"' in source
    assert "network.mark_output(hidden_out)" in source


def test_qwen3_omni_runtime_feeds_generated_hidden_states_to_talker() -> None:
    source = (ROOT / "src/runtime/models/omni/pipeline.cpp").read_text()

    assert 'outputs.find("hidden_state")' in source
    assert "hidden_states_out.insert" in source
    assert "run_thinker_step(token, logits, &hidden_state)" in source


def test_qwen3_omni_detects_real_talker_checkpoint_keys() -> None:
    source = (
        ROOT
        / "python/tensorrt_model_connect/families/qwen3_omni/plugin.py"
    ).read_text()

    assert "talker.model.codec_embedding.weight" in source
    assert "talker.code_predictor.lm_head" in source
    assert "num_code_groups" in source


def test_qwen3_omni_builds_stateless_talker_projection() -> None:
    source = (
        ROOT
        / "python/tensorrt_model_connect/families/qwen3_omni/plugin.py"
    ).read_text()

    assert "talker.hidden_projection.linear_fc1.weight" in source
    assert "talker.hidden_projection.linear_fc2.weight" in source
    assert "talker.codec_head.weight" in source
    assert 'input_embed", trt.float32' in source
    assert "network.add_concatenation(head_parts)" in source


def test_qwen3_omni_runtime_allows_stateless_talker_engine() -> None:
    source = (ROOT / "src/runtime/models/omni/plugin.cpp").read_text()

    assert 'talker_module->has_input("cache_k_0")' in source
    assert "std::make_unique<RecurrentState>" in source
