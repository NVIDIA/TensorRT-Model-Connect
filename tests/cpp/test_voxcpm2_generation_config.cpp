// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-VOXCPM2-CFG-01
// Architecture:   ARCH-AUD-001
// Unit Design:    UD-AUD-VOXCPM2-01
// Intent:         Validate native VoxCPM2 generation defaults and execution plan metadata.
// Preconditions:  In-memory config.json fragments.
// Postconditions: VoxCPM2 model-card generation parameters resolve before runtime execution.
// =============================================================================

#include "runtime/domains/audio/voxcpm2_config.h"
#include "runtime/domains/audio/voxcpm2_generation_plan.h"

#include <cmath>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

void test_model_card_defaults() {
    const auto cfg = trtmc::make_voxcpm2_config_from_json("{}");

    check(cfg.sample_rate == 48000, "default output sample rate is 48 kHz");
    check(cfg.reference_sample_rate == 16000, "default reference sample rate is 16 kHz");
    check(std::fabs(cfg.cfg_value - 2.0F) < 1e-6F, "default cfg_value matches model card");
    check(cfg.inference_timesteps == 10, "default inference_timesteps matches model card");
    check(cfg.normalize, "default normalize enabled");
    check(cfg.denoise, "default denoise enabled");
    check(cfg.retry_badcase, "default retry_badcase enabled");
    check(cfg.retry_badcase_max_times == 3, "default retry_badcase_max_times");
    check(std::fabs(cfg.retry_badcase_ratio_threshold - 6.0F) < 1e-6F,
          "default retry_badcase_ratio_threshold");
    check(cfg.seed == -1, "default seed disabled");
}

void test_bundle_audio_metadata_overrides_defaults() {
    const std::string config_json = R"json({
      "sample_rate": 44100,
      "reference_sample_rate": 22050,
      "voxcpm2_cfg_value": 3.5,
      "voxcpm2_inference_timesteps": 12
    })json";

    const auto cfg = trtmc::make_voxcpm2_config_from_json(config_json);

    check(cfg.sample_rate == 44100, "bundle sample_rate overrides default");
    check(cfg.reference_sample_rate == 22050, "bundle reference_sample_rate overrides default");
    check(std::fabs(cfg.cfg_value - 3.5F) < 1e-6F, "bundle cfg_value overrides default");
    check(cfg.inference_timesteps == 12, "bundle inference_timesteps overrides default");
}

void test_config_description_includes_model_card_fields() {
    const auto cfg = trtmc::make_voxcpm2_config_from_json("{}");
    const auto description = trtmc::describe_voxcpm2_config(cfg);

    check(description.find("sample_rate=48000") != std::string::npos,
          "description includes sample_rate");
    check(description.find("cfg_value=2") != std::string::npos, "description includes cfg_value");
    check(description.find("inference_timesteps=10") != std::string::npos,
          "description includes inference_timesteps");
}

void test_generation_plan_matches_component_contract() {
    check(trtmc::runtime::builders::audio::voxcpm2_generation_plan_matches_component_contract(),
          "generation plan follows component contract order and sections");
}

void test_generation_plan_preserves_acceptance_artifact_name() {
    const auto cfg = trtmc::make_voxcpm2_config_from_json(R"json({
      "sample_rate": 48000,
      "voxcpm2_cfg_value": 2.0,
      "voxcpm2_inference_timesteps": 10
    })json");
    const auto plan = trtmc::runtime::builders::audio::make_voxcpm2_generation_plan(cfg);

    check(std::string(plan.stages[0].name) == "locenc", "first VoxCPM2 stage is LocEnc");
    check(std::string(plan.stages[4].name) == "audiovae", "last VoxCPM2 stage is AudioVAE");
    check(std::string(plan.stages[4].output_artifact) == "waveform_f32",
          "AudioVAE produces float waveform");
    check(std::string(plan.stages[0].input_tensor.name) == "text_utf8",
          "LocEnc consumes UTF-8 byte tensor");
    check(plan.stages[0].input_tensor.rank == 1, "LocEnc text input is rank 1");
    check(std::string(trtmc::runtime::builders::audio::voxcpm2_dtype_contract_name(
              plan.stages[0].input_tensor.dtype_contract)) == "int8",
          "LocEnc text input is int8 bytes");
    check(std::string(plan.stages[0].output_tensor.name) == "local_text_features",
          "LocEnc produces local text feature tensor");
    check(plan.stages[0].output_tensor.rank == 2, "LocEnc features are rank 2");
    check(plan.stages[2].required_side_input_count == 1, "RALM requires one preserved side tensor");
    check(std::string(plan.stages[2].required_side_inputs[0]) == "local_text_features",
          "RALM consumes preserved local text features");
    check(plan.stages[3].required_side_input_count == 2,
          "LocDiT requires semantic and residual hidden side tensors");
    check(std::string(plan.stages[3].required_side_inputs[0]) == "lm_hidden",
          "LocDiT consumes TSLM hidden state");
    check(std::string(plan.stages[3].required_side_inputs[1]) == "residual_hidden",
          "LocDiT consumes RALM residual hidden state");
    check(plan.stages[3].required_control_input_count == 2,
          "LocDiT requires generation control bindings");
    check(std::string(plan.stages[3].required_control_inputs[0]) == "cfg_value",
          "LocDiT consumes cfg_value control");
    check(std::string(plan.stages[3].required_control_inputs[1]) == "inference_timesteps",
          "LocDiT consumes inference_timesteps control");
    check(std::string(trtmc::runtime::builders::audio::voxcpm2_dtype_contract_name(
              plan.stages[4].output_tensor.dtype_contract)) == "float32",
          "AudioVAE waveform output is float32");
    check(std::string(plan.output_wav_artifact) == "trt_output.wav",
          "TRT output WAV artifact name is stable");
    check(plan.config.sample_rate == 48000, "plan carries output sample rate");
}

void test_generation_plan_description_includes_stage_order_and_artifact() {
    const auto plan = trtmc::runtime::builders::audio::make_voxcpm2_generation_plan(
        trtmc::make_voxcpm2_config_from_json("{}"));
    const auto description =
        trtmc::runtime::builders::audio::describe_voxcpm2_generation_plan(plan);

    check(description.find("locenc(text_utf8=>local_text_features") != std::string::npos,
          "plan description includes LocEnc input/output");
    check(description.find("input=int8[utf8_bytes]") != std::string::npos,
          "plan description includes LocEnc input tensor contract");
    check(description.find("output=float32|bfloat16[text_steps,feat_dim]") != std::string::npos,
          "plan description includes LocEnc output tensor contract");
    check(description.find("-> tslm(") != std::string::npos,
          "plan description includes TSLM order");
    check(description.find("-> ralm(") != std::string::npos,
          "plan description includes RALM order");
    check(description.find("-> locdit(") != std::string::npos,
          "plan description includes LocDiT order");
    check(description.find("ralm(semantic_lm_states=>acoustic_residual_states") !=
              std::string::npos,
          "plan description includes RALM stage");
    check(description.find("side_inputs=local_text_features") != std::string::npos,
          "plan description includes RALM side input");
    check(description.find("side_inputs=lm_hidden,residual_hidden") != std::string::npos,
          "plan description includes LocDiT side inputs");
    check(description.find("controls=cfg_value,inference_timesteps") != std::string::npos,
          "plan description includes LocDiT controls");
    check(description.find("-> audiovae(audio_vae_latents=>waveform_f32") != std::string::npos,
          "plan description includes AudioVAE waveform output");
    check(description.find("output=float32[audio_samples]") != std::string::npos,
          "plan description includes waveform tensor contract");
    check(description.find("output_wav_artifact=trt_output.wav") != std::string::npos,
          "plan description includes output WAV artifact");
}

} // namespace

int main() {
    test_model_card_defaults();
    test_bundle_audio_metadata_overrides_defaults();
    test_config_description_includes_model_card_fields();
    test_generation_plan_matches_component_contract();
    test_generation_plan_preserves_acceptance_artifact_name();
    test_generation_plan_description_includes_stage_order_and_artifact();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }

    std::cerr << "All VoxCPM2 generation config tests passed.\n";
    return 0;
}
