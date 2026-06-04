// VoxCPM2Plugin: explicit runtime boundary for "text_to_audio_voxcpm2".
// Full support requires native LocEnc, TSLM, RALM, LocDiT, and AudioVAE engines.

#include "runtime/domains/audio/audio_bundle_validation.h"
#include "trtmc/runtime/pipeline_registry.h"

#include <memory>
#include <stdexcept>

namespace trtmc {

class VoxCPM2Plugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        runtime::builders::audio::validate_text_to_audio_bundle_sections(
            runtime::builders::audio::TextToAudioBundleKind::kVoxCpm2, ctx.bundle,
            ctx.bundle_path);
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
