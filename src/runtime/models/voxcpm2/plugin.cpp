// VoxCPM2Plugin: explicit runtime boundary for "text_to_audio_voxcpm2".
// Full support requires native LocEnc, TSLM, RALM, LocDiT, and AudioVAE engines.

#include "trtmc/runtime/pipeline_registry.h"

#include <memory>
#include <stdexcept>

namespace trtmc {

class VoxCPM2Plugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        (void)ctx;
        throw std::runtime_error(
            "VoxCPM2 TRT runtime is not implemented yet. Full openbmb/VoxCPM2 "
            "text-to-audio support requires native LocEnc, TSLM, RALM, LocDiT, "
            "and AudioVAE TensorRT engines plus waveform generation that preserves "
            "the TRT WAV artifact for comparison against the Hugging Face VoxCPM "
            "reference WAV.");
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_voxcpm2_plugin, VoxCPM2Plugin,
                                       "text_to_audio_voxcpm2");

} // namespace trtmc
