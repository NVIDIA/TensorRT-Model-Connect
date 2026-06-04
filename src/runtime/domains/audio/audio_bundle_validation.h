#pragma once

#include "bundle/bundle_format.h"
#include "bundle/bundle_view.h"

#include <string>

namespace trtmc::runtime::builders::audio {

enum class TextToAudioBundleKind {
    kBark,
    kMagpieTts,
    kVoxCpm2,
};

void validate_text_to_audio_bundle_sections(
    TextToAudioBundleKind kind,
    const trtmc::BundleFile& bundle,
    const std::string& bundle_path);

} // namespace trtmc::runtime::builders::audio
