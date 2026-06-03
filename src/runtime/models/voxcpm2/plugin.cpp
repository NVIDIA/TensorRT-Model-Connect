// VoxCPM2Plugin: registers "text_to_audio_voxcpm2" with an explicit runtime
// limitation until dedicated LocEnc/TSLM/RALM/LocDiT/AudioVAE builders exist.

#include "trtmc/runtime/pipeline_registry.h"

#include <memory>
#include <stdexcept>

namespace trtmc {

namespace {
constexpr const char* kVoxCPM2RuntimeLimitation =
    "VoxCPM2 runtime is not implemented yet. The upstream model is a tokenizer-free "
    "diffusion-autoregressive TTS stack (LocEnc -> TSLM -> RALM -> LocDiT -> AudioVAE "
    "V2) served through the external voxcpm library; TensorRT-Model-Connect needs "
    "dedicated builders and a runtime pipeline for those stages before "
    "text_to_audio_voxcpm2 bundles can generate audio.";
}

class VoxCPM2Plugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        (void)ctx;
        throw std::runtime_error(kVoxCPM2RuntimeLimitation);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_voxcpm2_plugin, VoxCPM2Plugin,
                                       "text_to_audio_voxcpm2");

} // namespace trtmc
