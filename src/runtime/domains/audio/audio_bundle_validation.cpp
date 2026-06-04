#include "runtime/domains/audio/audio_bundle_validation.h"

#include "runtime/domains/audio/voxcpm2_component_contract.h"

#include <stdexcept>

namespace trtmc::runtime::builders::audio {

namespace {

void require_section(const trtmc::BundleFile& bundle, const std::string& name,
                     const std::string& bundle_path) {
    auto* data = trtmc::find_section(bundle, name);
    if (!data || data->empty())
        throw std::runtime_error(bundle_path + ": missing " + name + " section");
}

} // namespace

static void validate_bark(const trtmc::BundleFile& bundle, const std::string& bundle_path) {
    require_section(bundle, "semantic_embed", bundle_path);
    require_section(bundle, "coarse_embed", bundle_path);
    require_section(bundle, "coarse_engine_plan", bundle_path);
}

static void validate_magpie(const trtmc::BundleFile& bundle, const std::string& bundle_path) {
    require_section(bundle, "magpie_audio_embed", bundle_path);
    require_section(bundle, "magpie_text_embed", bundle_path);
    require_section(bundle, "magpie_context_embed", bundle_path);
    require_section(bundle, "magpie_ipa_phoneme_dict", bundle_path);
    require_section(bundle, "magpie_ipa_vocab", bundle_path);
}

static void validate_voxcpm2(const trtmc::BundleFile& bundle, const std::string& bundle_path) {
    for (const auto& component : kVoxCPM2ComponentSpecs)
        require_section(bundle, component.engine_section, bundle_path);
}

void validate_text_to_audio_bundle_sections(TextToAudioBundleKind kind,
                                            const trtmc::BundleFile& bundle,
                                            const std::string& bundle_path) {
    switch (kind) {
    case TextToAudioBundleKind::kBark:
        validate_bark(bundle, bundle_path);
        break;
    case TextToAudioBundleKind::kMagpieTts:
        validate_magpie(bundle, bundle_path);
        break;
    case TextToAudioBundleKind::kVoxCpm2:
        validate_voxcpm2(bundle, bundle_path);
        break;
    }
}

} // namespace trtmc::runtime::builders::audio
