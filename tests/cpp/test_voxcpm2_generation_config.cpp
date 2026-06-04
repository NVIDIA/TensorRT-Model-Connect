// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-VOXCPM2-CFG-01
// Architecture:   ARCH-AUD-001
// Unit Design:    UD-AUD-VOXCPM2-01
// Intent:         Validate native VoxCPM2 generation defaults carried by bundle metadata.
// Preconditions:  In-memory config.json fragments.
// Postconditions: VoxCPM2 model-card generation parameters resolve before runtime execution.
// =============================================================================

#include "runtime/domains/audio/voxcpm2_config.h"

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

} // namespace

int main() {
    test_model_card_defaults();
    test_bundle_audio_metadata_overrides_defaults();
    test_config_description_includes_model_card_fields();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }

    std::cerr << "All VoxCPM2 generation config tests passed.\n";
    return 0;
}
