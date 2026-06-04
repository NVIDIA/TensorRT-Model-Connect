#pragma once

// VoxCPM2Pipeline: native runtime boundary for openbmb/VoxCPM2.
// The pipeline owns the five component TRT modules and the model-card
// generation contract. The current native execution boundary chains
// component engines through the explicit VoxCPM2 artifact tensor names; the
// remaining acceptance proof is producing the TRT WAV and comparing it against
// the saved Hugging Face model-card WAV.

#include "runtime/domains/audio/voxcpm2_component_loader.h"
#include "runtime/domains/audio/voxcpm2_generation_plan.h"
#include "trtmc/pipeline.h"
#include "trtmc/tokenizer.h"

#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class VoxCPM2Pipeline final : public IPipeline {
  public:
    VoxCPM2Pipeline(std::vector<runtime::builders::audio::VoxCPM2LoadedComponent> components,
                    runtime::builders::audio::VoxCPM2GenerationPlan plan,
                    std::string model_id_str = "",
                    std::shared_ptr<ITokenizer> tokenizer = nullptr);

    AudioResult generate_audio(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "VoxCPM2Pipeline"; }

  private:
    void validate_components() const;

    std::vector<runtime::builders::audio::VoxCPM2LoadedComponent> components_;
    runtime::builders::audio::VoxCPM2GenerationPlan plan_;
    std::string model_id_;
    std::shared_ptr<ITokenizer> tokenizer_;
};

} // namespace trtmc
