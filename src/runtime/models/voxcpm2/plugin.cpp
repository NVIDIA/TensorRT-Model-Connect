// VoxCPM2Plugin: explicit runtime boundary for "text_to_audio_voxcpm2".
// Full acceptance requires valid component plans plus the HF/TRT WAV parity run.

#include "bundle/bundle_view.h"
#include "runtime/domains/audio/audio_bundle_validation.h"
#include "runtime/domains/audio/voxcpm2_component_loader.h"
#include "runtime/domains/audio/voxcpm2_config.h"
#include "runtime/domains/audio/voxcpm2_generation_plan.h"
#include "runtime/models/voxcpm2/pipeline.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtmc/config/config_bundle.h"
#include "trtmc/runtime/pipeline_registry.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

constexpr char kVoxCPM2ZeroPrefillFeaturesSection[] =
    "voxcpm2_zero_prefill_local_text_features_bf16";
constexpr std::uint32_t kVoxCPM2ZeroPrefillFeaturesVersion = 1;

std::uint32_t read_u32_le(const std::vector<char>& data, std::size_t& offset,
                          const std::string& section_name) {
    if (offset > data.size() || data.size() - offset < sizeof(std::uint32_t)) {
        throw std::runtime_error("VoxCPM2 zero-prefill section '" + section_name +
                                 "' is truncated");
    }
    std::uint32_t value = 0;
    std::memcpy(&value, data.data() + offset, sizeof(value));
    offset += sizeof(value);
    return value;
}

VoxCPM2ZeroPrefillFeatureTable
load_zero_prefill_feature_table(const BundleFile& bundle) {
    VoxCPM2ZeroPrefillFeatureTable table;
    const auto* section = find_section(bundle, kVoxCPM2ZeroPrefillFeaturesSection);
    if (section == nullptr || section->empty())
        return table;

    std::size_t offset = 0;
    const auto version = read_u32_le(*section, offset, kVoxCPM2ZeroPrefillFeaturesSection);
    if (version != kVoxCPM2ZeroPrefillFeaturesVersion) {
        throw std::runtime_error(
            "VoxCPM2 zero-prefill section has unsupported version " +
            std::to_string(version));
    }
    const auto entry_count = read_u32_le(*section, offset, kVoxCPM2ZeroPrefillFeaturesSection);
    const auto hidden_size = read_u32_le(*section, offset, kVoxCPM2ZeroPrefillFeaturesSection);
    if (hidden_size == 0) {
        throw std::runtime_error("VoxCPM2 zero-prefill section has zero hidden size");
    }

    table.hidden_size = static_cast<int32_t>(hidden_size);
    const std::size_t row_bytes = static_cast<std::size_t>(hidden_size) * sizeof(std::uint16_t);
    table.rows.reserve(entry_count);
    for (std::uint32_t i = 0; i < entry_count; ++i) {
        VoxCPM2ZeroPrefillFeatureRow row;
        row.text_steps = static_cast<int32_t>(
            read_u32_le(*section, offset, kVoxCPM2ZeroPrefillFeaturesSection));
        if (row.text_steps <= 0) {
            throw std::runtime_error(
                "VoxCPM2 zero-prefill section contains non-positive text step count");
        }
        if (offset > section->size() || section->size() - offset < row_bytes) {
            throw std::runtime_error("VoxCPM2 zero-prefill section '" +
                                     std::string(kVoxCPM2ZeroPrefillFeaturesSection) +
                                     "' is truncated in row payload");
        }
        row.local_text_features_bf16.resize(row_bytes);
        std::memcpy(row.local_text_features_bf16.data(), section->data() + offset, row_bytes);
        offset += row_bytes;
        table.rows.push_back(std::move(row));
    }
    if (offset != section->size()) {
        throw std::runtime_error(
            "VoxCPM2 zero-prefill section contains trailing bytes");
    }
    return table;
}

} // namespace

class VoxCPM2Plugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        runtime::builders::audio::validate_text_to_audio_bundle_sections(
            runtime::builders::audio::TextToAudioBundleKind::kVoxCpm2, ctx.bundle, ctx.bundle_path);

        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        auto components =
            runtime::builders::audio::load_voxcpm2_component_modules(ctx.backend, ctx.bundle, opts);
        auto tokenizer =
            try_create_native_tokenizer(ctx.bundle, /*add_special_tokens=*/false);
        const auto generation_cfg =
            make_voxcpm2_config(ctx.config_json, ctx.runtime_config, ctx.config.max_cache_length);
        const auto generation_plan =
            runtime::builders::audio::make_voxcpm2_generation_plan(generation_cfg);
        auto zero_prefill_features = load_zero_prefill_feature_table(ctx.bundle);

        return std::make_unique<VoxCPM2Pipeline>(std::move(components), generation_plan,
                                                 ctx.bundle.info.model_id, std::move(tokenizer),
                                                 std::move(zero_prefill_features));
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_voxcpm2_plugin, VoxCPM2Plugin,
                                       "text_to_audio_voxcpm2");

} // namespace trtmc
