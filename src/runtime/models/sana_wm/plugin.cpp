/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// SANA-WM plugin: loads native TensorRT modules from bundle sections.

#include "bundle/bundle_view.h"
#include "runtime/models/sana_wm/native_ops.h"
#include "runtime/models/sana_wm/pipeline.h"
#include "runtime/models/sana_wm/plugin_helpers.h"
#include "runtime/models/sana_wm/sana_wm_bpe_tokenizer.h"
#include "trtmc/config/config_bundle.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "trtmc/tokenizer.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace trtmc {

namespace {

std::string native_plugin_cache_path(const std::vector<char>& bytes) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (unsigned char value : bytes) {
        hash ^= static_cast<std::uint64_t>(value);
        hash *= 1099511628211ULL;
    }
    const char* configured = std::getenv("TRTMC_SANA_WM_NATIVE_PLUGIN_CACHE_DIR");
    const auto directory = configured != nullptr && configured[0] != '\0'
                               ? std::filesystem::path(configured)
                               : std::filesystem::temp_directory_path() / "trtmc-sana-wm";
    std::ostringstream name;
    name << "libtrtmc_sana_wm_native_plugin_" << std::hex << hash << ".so";
    return (directory / name.str()).string();
}

void write_native_plugin_cache_file(const std::filesystem::path& output,
                                    const std::vector<char>& bytes) {
    std::filesystem::create_directories(output.parent_path());
    if (std::filesystem::is_regular_file(output) &&
        std::filesystem::file_size(output) == bytes.size()) {
        return;
    }

    const auto temporary = output.string() + ".tmp";
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream)
        throw std::runtime_error("Unable to create SANA-WM native plugin cache file");
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    stream.close();
    if (!stream)
        throw std::runtime_error("Unable to write SANA-WM native plugin cache file");
    std::filesystem::rename(temporary, output);
}

std::string resolve_sana_wm_native_plugin_path(const PipelineContext& ctx) {
    const auto* bytes = find_section(ctx.bundle, "sana_wm_native_plugin_so");
    if (bytes != nullptr && !bytes->empty()) {
        const auto path = native_plugin_cache_path(*bytes);
        write_native_plugin_cache_file(path, *bytes);
        return path;
    }

    const char* configured = std::getenv("TRTMC_SANA_WM_NATIVE_PLUGIN_LIBRARY");
    return configured != nullptr ? configured : std::string{};
}

void load_sana_wm_native_plugin(const PipelineContext& ctx) {
    const auto path = resolve_sana_wm_native_plugin_path(ctx);
    if (path.empty()) {
        throw std::runtime_error(
            "SANA-WM bundle is missing the model-owned native TensorRT plugin section");
    }

    dlerror();
    if (dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL) == nullptr) {
        const char* message = dlerror();
        throw std::runtime_error(std::string("Unable to load SANA-WM native plugin: ") +
                                 (message != nullptr ? message : path));
    }
    require_sana_wm_native_ops();
}

bool runtime_override(const config::ConfigBundle& bundle, const char* field) {
    return bundle.source_of("sana_wm", field) != config::Layer::SchemaDefault;
}

int32_t checked_int32(std::int64_t value, const char* field) {
    if (value < std::numeric_limits<int32_t>::min() ||
        value > std::numeric_limits<int32_t>::max()) {
        throw std::out_of_range(std::string("sana_wm.") + field + " is outside int32 range");
    }
    return static_cast<int32_t>(value);
}

std::vector<float> parse_intrinsics_csv(const std::string& text) {
    std::vector<float> values;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (!token.empty())
            values.push_back(std::stof(token));
    }
    if (values.size() != 4U && values.size() != 9U) {
        throw std::invalid_argument(
            "sana_wm.intrinsics must contain fx,fy,cx,cy or one row-major 3x3 matrix");
    }
    return values;
}

template <typename T>
void apply_runtime_value(T& destination, const config::ConfigBundle& runtime, const char* field) {
    if (runtime_override(runtime, field))
        destination = runtime.get<T>("sana_wm", field);
}

void apply_runtime_int32(int32_t& destination, const config::ConfigBundle& runtime,
                         const char* field) {
    if (runtime_override(runtime, field))
        destination = checked_int32(runtime.get<std::int64_t>("sana_wm", field), field);
}

void apply_runtime_intrinsics(std::vector<float>& destination,
                              const config::ConfigBundle& runtime) {
    if (runtime_override(runtime, "intrinsics")) {
        destination = parse_intrinsics_csv(runtime.get<std::string>("sana_wm", "intrinsics"));
    }
}

void apply_sana_wm_runtime_config(SanaWmRuntimeConfig& config,
                                  const config::ConfigBundle* runtime) {
    if (runtime == nullptr)
        return;
    try {
        apply_runtime_value(config.default_image, *runtime, "image_path");
        apply_runtime_value(config.action, *runtime, "action");
        apply_runtime_value(config.translation_speed, *runtime, "translation_speed");
        apply_runtime_value(config.rotation_speed_deg, *runtime, "rotation_speed_deg");
        apply_runtime_int32(config.num_frames, *runtime, "num_frames");
        apply_runtime_int32(config.fps, *runtime, "fps");
        apply_runtime_value(config.flow_shift, *runtime, "flow_shift");
        apply_runtime_intrinsics(config.default_intrinsics, *runtime);
        apply_runtime_value(config.no_refiner, *runtime, "no_refiner");
    } catch (const std::exception& exc) {
        throw std::runtime_error(std::string("Invalid SANA-WM runtime config: ") + exc.what());
    }
}

std::unique_ptr<ITrtModule> load_optional_sana_wm_module(const PipelineContext& ctx,
                                                         const char* section,
                                                         const ModuleCreateOptions& opts) {
    const auto* plan = find_section(ctx.bundle, section);
    if (!plan || plan->empty())
        return nullptr;
    auto loaded = load_trt_module_from_plan(ctx.backend, plan, section, opts);
    return std::move(loaded.module);
}

std::vector<std::unique_ptr<ITrtModule>>
load_sana_wm_stage1_denoiser_segments(const PipelineContext& ctx, const ModuleCreateOptions& opts) {
    constexpr std::array<const char*, 5> sections = {
        "sana_wm_stage1_denoiser_block0_3_plan",      "sana_wm_stage1_denoiser_block4_7_plan",
        "sana_wm_stage1_denoiser_block8_11_plan",     "sana_wm_stage1_denoiser_block12_15_plan",
        "sana_wm_stage1_denoiser_block16_final_plan",
    };
    std::vector<std::unique_ptr<ITrtModule>> modules;
    modules.reserve(sections.size());
    std::vector<std::string> missing;
    bool saw_any = false;
    for (const char* section : sections) {
        auto module = load_optional_sana_wm_module(ctx, section, opts);
        if (module) {
            saw_any = true;
            modules.push_back(std::move(module));
        } else {
            missing.emplace_back(section);
        }
    }
    if (!missing.empty()) {
        if (!saw_any)
            return {};
        std::string message =
            "SANA-WM segmented Stage-1 denoiser requires all segment plans; missing";
        for (const auto& section : missing)
            message += " " + section;
        throw std::runtime_error(message);
    }
    return modules;
}

std::string vae_tile_section_name(const std::string& prefix, int32_t frames, int32_t height,
                                  int32_t width) {
    return prefix + "_tile_t" + std::to_string(frames) + "_h" + std::to_string(height) + "_w" +
           std::to_string(width) + "_plan";
}

std::vector<std::pair<int32_t, int32_t>>
sana_wm_expected_temporal_tiles(const SanaWmRuntimeConfig& config, int32_t latent_frames) {
    const int32_t tile_latent_min_frames =
        std::max(1, config.vae_tile_sample_min_num_frames / config.vae_time_stride);
    const int32_t tile_latent_stride_frames =
        std::max(1, config.vae_tile_sample_stride_num_frames / config.vae_time_stride);
    std::vector<std::pair<int32_t, int32_t>> tiles;
    if (!config.vae_use_framewise_decoding || latent_frames <= tile_latent_min_frames) {
        tiles.push_back({0, latent_frames});
        return tiles;
    }
    for (int32_t t0 = 0; t0 < latent_frames; t0 += tile_latent_stride_frames) {
        const int32_t frames = std::min(tile_latent_min_frames + 1, latent_frames - t0);
        if (t0 > 0 && frames <= 1)
            continue;
        tiles.push_back({t0, frames});
    }
    return tiles;
}

std::vector<std::pair<int32_t, int32_t>> sana_wm_expected_spatial_tiles(int32_t latent_size,
                                                                        int32_t tile_min_size,
                                                                        int32_t tile_stride,
                                                                        bool use_tiling) {
    std::vector<std::pair<int32_t, int32_t>> tiles;
    if (!use_tiling || latent_size <= tile_min_size) {
        tiles.push_back({0, latent_size});
        return tiles;
    }
    for (int32_t start = 0; start < latent_size; start += tile_stride)
        tiles.push_back({start, std::min(tile_min_size, latent_size - start)});
    return tiles;
}

std::set<std::tuple<int32_t, int32_t, int32_t>>
sana_wm_expected_vae_tile_shapes(const SanaWmRuntimeConfig& config) {
    const int32_t latent_frames = (config.num_frames - 1) / config.vae_time_stride + 1;
    const int32_t latent_height = config.height / config.vae_spatial_stride;
    const int32_t latent_width = config.width / config.vae_spatial_stride;
    const int32_t tile_latent_min_height =
        std::max(1, config.vae_tile_sample_min_height / config.vae_spatial_stride);
    const int32_t tile_latent_min_width =
        std::max(1, config.vae_tile_sample_min_width / config.vae_spatial_stride);
    const int32_t tile_latent_stride_height =
        std::max(1, config.vae_tile_sample_stride_height / config.vae_spatial_stride);
    const int32_t tile_latent_stride_width =
        std::max(1, config.vae_tile_sample_stride_width / config.vae_spatial_stride);

    const auto temporal_tiles = sana_wm_expected_temporal_tiles(config, latent_frames);
    const bool spatial_tiled =
        config.vae_use_spatial_tiling &&
        (latent_height > tile_latent_min_height || latent_width > tile_latent_min_width);
    const auto height_tiles = sana_wm_expected_spatial_tiles(
        latent_height, tile_latent_min_height, tile_latent_stride_height, spatial_tiled);
    const auto width_tiles = sana_wm_expected_spatial_tiles(
        latent_width, tile_latent_min_width, tile_latent_stride_width, spatial_tiled);

    std::set<std::tuple<int32_t, int32_t, int32_t>> shapes;
    for (const auto& [_t0, frames] : temporal_tiles)
        for (const auto& [_y0, height] : height_tiles)
            for (const auto& [_x0, width] : width_tiles)
                shapes.emplace(frames, height, width);
    return shapes;
}

std::vector<SanaWmVaeDecoderTile> load_sana_wm_vae_tile_modules(const PipelineContext& ctx,
                                                                const SanaWmRuntimeConfig& config,
                                                                const std::string& prefix,
                                                                const ModuleCreateOptions& opts) {
    std::vector<SanaWmVaeDecoderTile> tiles;
    const auto expected_shapes = sana_wm_expected_vae_tile_shapes(config);
    for (const auto& [frames, height, width] : expected_shapes) {
        const auto section = vae_tile_section_name(prefix, frames, height, width);
        auto module = load_optional_sana_wm_module(ctx, section.c_str(), opts);
        if (module) {
            tiles.push_back({frames, height, width, std::move(module)});
        }
    }
    if (!tiles.empty() && tiles.size() != expected_shapes.size()) {
        std::cerr << "[trtmc] Ignoring incomplete SANA-WM VAE tile plan set for " << prefix << " ("
                  << tiles.size() << "/" << expected_shapes.size() << " present)" << std::endl;
        tiles.clear();
    }
    return tiles;
}

SanaWmNativeModules load_sana_wm_native_modules(const PipelineContext& ctx,
                                                const SanaWmRuntimeConfig& config) {
    ModuleCreateOptions opts;
    opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
    opts.cuda_graphs = ctx.cuda_graphs;

    SanaWmNativeModules modules;
    modules.text_encoder = load_optional_sana_wm_module(ctx, "text_encoder_0_plan", opts);
    modules.stage1_denoiser = load_optional_sana_wm_module(ctx, "denoiser_plan", opts);
    if (!modules.stage1_denoiser)
        modules.stage1_denoiser_segments = load_sana_wm_stage1_denoiser_segments(ctx, opts);
    modules.vae_encoder = load_optional_sana_wm_module(ctx, "sana_wm_vae_encoder_plan", opts);
    modules.vae_decoder_tiles =
        load_sana_wm_vae_tile_modules(ctx, config, "sana_wm_vae_decoder", opts);
    if (modules.vae_decoder_tiles.empty())
        modules.vae_decoder = load_optional_sana_wm_module(ctx, "vae_decoder_plan", opts);
    modules.refiner_text_encoder =
        load_optional_sana_wm_module(ctx, "sana_wm_refiner_text_encoder_plan", opts);
    modules.refiner_text_connector =
        load_optional_sana_wm_module(ctx, "sana_wm_refiner_text_connector_plan", opts);
    modules.refiner_denoiser =
        load_optional_sana_wm_module(ctx, "sana_wm_refiner_denoiser_plan", opts);
    modules.refiner_vae_decoder_tiles =
        load_sana_wm_vae_tile_modules(ctx, config, "sana_wm_refiner_vae_decoder", opts);
    if (modules.refiner_vae_decoder_tiles.empty()) {
        modules.refiner_vae_decoder_tiles =
            load_sana_wm_vae_tile_modules(ctx, config, "sana_wm_vae_decoder", opts);
    }
    if (modules.refiner_vae_decoder_tiles.empty()) {
        modules.refiner_vae_decoder =
            load_optional_sana_wm_module(ctx, "sana_wm_refiner_vae_decoder_plan", opts);
    }
    return modules;
}

std::shared_ptr<ITokenizer> create_sana_wm_tokenizer_from_section(const BundleFile& bundle,
                                                                  const char* section,
                                                                  const char* label) {
    const auto* tok_data = find_section(bundle, section);
    if (!tok_data || tok_data->empty())
        return nullptr;

    const bool add_special = detect_add_special_tokens(bundle);
    const char* data = tok_data->data();
    const std::size_t size = tok_data->size();

    auto wrap = [&](std::unique_ptr<ITokenizer> tok,
                    const char* kind) -> std::shared_ptr<ITokenizer> {
        if (!tok)
            return nullptr;
        std::cerr << "[trtmc] Using native " << kind << " tokenizer for " << label << std::endl;
        return std::shared_ptr<ITokenizer>(std::move(tok));
    };

    try {
        if (auto tok = wrap(CreateSanaWmBpeTokenizer(data, size, add_special), "BPE"))
            return tok;
    } catch (...) {
    }
    try {
        if (auto tok = wrap(CreateWordPieceTokenizer(data, size, add_special), "WordPiece"))
            return tok;
    } catch (...) {
    }
    try {
        if (auto tok = wrap(CreateUnigramTokenizer(data, size, add_special), "Unigram"))
            return tok;
    } catch (...) {
    }
    throw std::runtime_error(std::string("SANA-WM failed to create native tokenizer from ") +
                             section);
}

} // namespace

class SanaWmPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        auto config = parse_sana_wm_config(ctx.config_json);
        apply_sana_wm_runtime_config(config, ctx.runtime_config);
        load_sana_wm_native_plugin(ctx);
        auto native_modules = load_sana_wm_native_modules(ctx, config);
        auto fallback_tokenizer = create_tokenizer_from_bundle(ctx.bundle);
        auto stage1_tokenizer = create_sana_wm_tokenizer_from_section(
            ctx.bundle, "sana_wm_stage1_tokenizer.json", "SANA-WM Stage-1");
        if (!stage1_tokenizer)
            stage1_tokenizer = fallback_tokenizer;
        auto refiner_tokenizer = create_sana_wm_tokenizer_from_section(
            ctx.bundle, "sana_wm_refiner_tokenizer.json", "SANA-WM refiner");
        if (!refiner_tokenizer)
            refiner_tokenizer = fallback_tokenizer;
        return std::make_unique<SanaWmPipeline>(std::move(config), std::move(native_modules),
                                                std::move(stage1_tokenizer),
                                                std::move(refiner_tokenizer));
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_sana_wm_plugin, SanaWmPlugin, "diffusion_sana_wm");

} // namespace trtmc
