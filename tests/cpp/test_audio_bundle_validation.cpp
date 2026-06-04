// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-01
// Architecture:   ARCH-BDL-001
// Unit Design:    UD-BDL-02
// Intent:         Validate bundle section requirements for audio pipelines (Bark, Magpie, VoxCPM2)
// Preconditions:  BundleFile with audio-specific sections constructed in memory
// Postconditions: Validation accepts complete bundles and rejects incomplete ones
// =============================================================================

#include "runtime/domains/audio/audio_bundle_validation.h"
#include "runtime/domains/audio/voxcpm2_component_contract.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

std::vector<char> bytes_from_text(const std::string& text) {
    return std::vector<char>(text.begin(), text.end());
}

// Helper to add a named section to a BundleFile.
void add_section(trtmc::BundleFile& bundle, const std::string& name,
                 const std::vector<char>& data) {
    trtmc::BundleSection sec;
    sec.name = name;
    sec.data = data;
    bundle.sections.push_back(std::move(sec));
}

void test_bark_validation_requires_semantic_and_coarse_assets() {
    // Empty bundle — missing all required sections.
    trtmc::BundleFile bundle;

    try {
        trtmc::runtime::builders::audio::validate_text_to_audio_bundle_sections(
            trtmc::runtime::builders::audio::TextToAudioBundleKind::kBark, bundle, "bark.trtfb");
        check(false, "bark validation rejects missing semantic/coarse sections");
    } catch (const std::runtime_error& error) {
        check(std::string(error.what()).find("semantic_embed") != std::string::npos,
              "bark validation reports semantic_embed");
    }
}

void test_magpie_validation_requires_ipa_tokenizer_sections() {
    trtmc::BundleFile bundle;
    const auto audio = bytes_from_text("audio");
    const auto text = bytes_from_text("text");
    const auto context = bytes_from_text("context");

    // Provide the embed sections but NOT the IPA tokenizer sections.
    add_section(bundle, "magpie_audio_embed", audio);
    add_section(bundle, "magpie_text_embed", text);
    add_section(bundle, "magpie_context_embed", context);

    try {
        trtmc::runtime::builders::audio::validate_text_to_audio_bundle_sections(
            trtmc::runtime::builders::audio::TextToAudioBundleKind::kMagpieTts, bundle,
            "magpie.trtfb");
        check(false, "magpie validation rejects missing IPA tokenizer sections");
    } catch (const std::runtime_error& error) {
        check(std::string(error.what()).find("magpie_ipa") != std::string::npos,
              "magpie validation reports IPA tokenizer sections");
    }
}

void test_magpie_validation_accepts_complete_required_sections() {
    trtmc::BundleFile bundle;
    const auto audio = bytes_from_text("audio");
    const auto text = bytes_from_text("text");
    const auto context = bytes_from_text("context");
    const auto phoneme_dict = bytes_from_text("dict");
    const auto vocab = bytes_from_text("vocab");

    add_section(bundle, "magpie_audio_embed", audio);
    add_section(bundle, "magpie_text_embed", text);
    add_section(bundle, "magpie_context_embed", context);
    add_section(bundle, "magpie_ipa_phoneme_dict", phoneme_dict);
    add_section(bundle, "magpie_ipa_vocab", vocab);

    try {
        trtmc::runtime::builders::audio::validate_text_to_audio_bundle_sections(
            trtmc::runtime::builders::audio::TextToAudioBundleKind::kMagpieTts, bundle,
            "magpie.trtfb");
        check(true, "magpie validation accepts complete section set");
    } catch (const std::exception&) {
        check(false, "magpie validation accepts complete section set");
    }
}

void test_voxcpm2_validation_requires_native_component_engines() {
    trtmc::BundleFile bundle;

    try {
        trtmc::runtime::builders::audio::validate_text_to_audio_bundle_sections(
            trtmc::runtime::builders::audio::TextToAudioBundleKind::kVoxCpm2, bundle,
            "voxcpm2.trtfb");
        check(false, "voxcpm2 validation rejects missing component engines");
    } catch (const std::runtime_error& error) {
        check(std::string(error.what()).find("locenc_engine_plan") != std::string::npos,
              "voxcpm2 validation reports first missing component engine");
    }
}

void test_voxcpm2_component_contract_lists_native_engine_sections() {
    const auto& specs = trtmc::runtime::builders::audio::kVoxCPM2ComponentSpecs;

    check(specs.size() == 5, "voxcpm2 component contract lists five engines");
    check(std::string(specs[0].name) == "locenc", "voxcpm2 component 0 is locenc");
    check(std::string(specs[0].engine_section) == "locenc_engine_plan",
          "voxcpm2 locenc section name");
    check(std::string(specs[0].input_artifact) == "audio_feats",
          "voxcpm2 locenc input artifact name");
    check(std::string(specs[0].input_tensor.name) == "audio_feats",
          "voxcpm2 locenc input tensor name");
    check(specs[0].input_tensor.rank == 3, "voxcpm2 locenc input tensor rank");
    check(std::string(trtmc::runtime::builders::audio::voxcpm2_dtype_contract_name(
              specs[0].input_tensor.dtype_contract)) == "float32|bfloat16",
          "voxcpm2 locenc input tensor dtype");
    check(std::string(specs[0].output_artifact) == "local_text_features",
          "voxcpm2 locenc output artifact name");
    check(specs[0].output_tensor.rank == 2, "voxcpm2 locenc output tensor rank");
    check(std::string(specs[4].name) == "audiovae", "voxcpm2 component 4 is audiovae");
    check(std::string(specs[4].engine_section) == "audiovae_engine_plan",
          "voxcpm2 audiovae section name");
    check(std::string(specs[4].input_artifact) == "audio_vae_latents",
          "voxcpm2 audiovae input artifact name");
    check(std::string(specs[4].output_artifact) == "waveform_f32",
          "voxcpm2 audiovae output artifact name");
    check(std::string(trtmc::runtime::builders::audio::voxcpm2_dtype_contract_name(
              specs[4].output_tensor.dtype_contract)) == "float32",
          "voxcpm2 audiovae output tensor dtype");
    check(specs[4].output_tensor.rank == 1, "voxcpm2 audiovae output tensor rank");
}

void test_voxcpm2_validation_accepts_complete_required_sections() {
    trtmc::BundleFile bundle;
    const auto plan = bytes_from_text("plan");

    add_section(bundle, "locenc_engine_plan", plan);
    add_section(bundle, "tslm_engine_plan", plan);
    add_section(bundle, "ralm_engine_plan", plan);
    add_section(bundle, "locdit_engine_plan", plan);
    add_section(bundle, "audiovae_engine_plan", plan);

    try {
        trtmc::runtime::builders::audio::validate_text_to_audio_bundle_sections(
            trtmc::runtime::builders::audio::TextToAudioBundleKind::kVoxCpm2, bundle,
            "voxcpm2.trtfb");
        check(true, "voxcpm2 validation accepts complete component engine set");
    } catch (const std::exception&) {
        check(false, "voxcpm2 validation accepts complete component engine set");
    }
}

} // namespace

int main() {
    test_bark_validation_requires_semantic_and_coarse_assets();
    test_magpie_validation_requires_ipa_tokenizer_sections();
    test_magpie_validation_accepts_complete_required_sections();
    test_voxcpm2_validation_requires_native_component_engines();
    test_voxcpm2_component_contract_lists_native_engine_sections();
    test_voxcpm2_validation_accepts_complete_required_sections();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }

    std::cerr << "All audio bundle validation tests passed.\n";
    return 0;
}
