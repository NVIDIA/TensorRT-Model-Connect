#pragma once

#include "trtmc/pipeline.h"
#include "trtmc/runtime/domains/audio/subprocess_runner.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct SanaWmRuntimeConfig {
    std::string hf_id{"Efficient-Large-Model/SANA-WM_bidirectional"};
    std::string script_path;
    std::string default_image;
    std::string action{"w-80,jw-40,w-40,lw-60,w-100"};
    float translation_speed{0.055F};
    float rotation_speed_deg{1.2F};
    int32_t num_frames{321};
    int32_t height{704};
    int32_t width{1280};
    bool require_official_script{false};
};

SanaWmRuntimeConfig parse_sana_wm_config(const std::string& config_json);

struct SanaWmPose {
    // Row-major 4x4 camera-to-world matrix, matching the upstream .npy layout.
    std::array<float, 16> c2w{};
};

std::vector<SanaWmPose> sana_wm_action_to_c2w(const std::string& action, float translation_speed,
                                              float rotation_speed_deg);

class SanaWmPipeline final : public IPipeline {
  public:
    SanaWmPipeline(SanaWmRuntimeConfig config, std::string hf_python,
                   std::shared_ptr<ISubprocessRunner> subprocess_runner = nullptr);

    const char* model_id() const override { return config_.hf_id.c_str(); }
    const char* pipeline_type() const override { return "SanaWmPipeline"; }

    ImageResult generate_image(const std::string& prompt, const GenerateConfig& cfg = {}) override;

  private:
    SanaWmRuntimeConfig config_;
    std::string hf_python_;
    std::shared_ptr<ISubprocessRunner> subprocess_runner_;
};

} // namespace trtmc
