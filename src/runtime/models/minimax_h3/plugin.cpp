/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_format.h"
#include "bundle/bundle_view.h"
#include "runtime/backend/prebound_backend.h"
#include "runtime/models/minimax_h3/hot_engine_policy.h"
#include "runtime/models/minimax_h3/pipeline.h"
#include "trtmc/config/config_bundle.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/tokenizer.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

using SectionMap = std::unordered_map<std::string, BundleSectionInfo>;
using PlanSha256Map = std::unordered_map<std::string, std::string>;

std::string plan_filename(std::string_view section) {
    if (section == "text_encoder_plan")
        return "text_encoder.plan";
    if (section == "vision_encoder_plan")
        return "vision_encoder.plan";
    if (section == "fl2va_keyframe_vae_encoder_plan")
        return "fl2va_keyframe_vae_encoder.plan";
    if (section == "ref2va_denoiser_plan")
        return "ref2va_denoiser.plan";
    if (section == "ref2va_adaln_precompute_plan")
        return "ref2va_adaln_precompute.plan";
    if (section == "ref2va_video_vae_encoder_plan")
        return "ref2va_video_vae_encoder.plan";
    if (section == "ref2va_audio_vae_encoder_plan")
        return "ref2va_audio_vae_encoder.plan";
    if (section == "adaln_precompute_plan")
        return "adaln_precompute.plan";
    if (section == "denoiser_plan")
        return "denoiser.plan";
    if (section == "denoiser_head_plan")
        return "denoiser_head.plan";
    if (section == "denoiser_tail_plan")
        return "denoiser_tail.plan";
    if (section == "denoiser_finish_plan")
        return "denoiser_finish.plan";
    if (section == "denoiser_entry_plan")
        return "denoiser_entry.plan";
    if (minimax_h3::is_fast_h3_transition_plan(section))
        return std::string(minimax_h3::kDenoiserTransitionPrefix) +
               std::string(section.substr(minimax_h3::kDenoiserTransitionPrefix.size(), 2)) +
               ".plan";
    if (section == "vae_tile_decoder_plan")
        return "vae_tile_decoder.plan";
    if (section == "audio_vae_decoder_plan")
        return "audio_vae_decoder.plan";
    throw std::runtime_error("Unknown MiniMax-H3 plan section: " + std::string(section));
}

bool is_sha256(std::string_view value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char character) {
               return (character >= '0' && character <= '9') ||
                      (character >= 'a' && character <= 'f');
           });
}

struct RuntimeMemoryConfig {
    bool staged{false};
    std::int64_t weight_streaming_budget_bytes{-1};
};

void validate_rtx_runtime_context(const PipelineContext& ctx, std::string_view recorded_backend) {
    if (ctx.backend == nullptr || std::string_view(ctx.backend->name()) != recorded_backend)
        throw std::runtime_error(
            "MiniMax-H3 bundle backend does not match the loaded runtime backend");
    if (ctx.cuda_graphs)
        throw std::runtime_error(
            "MiniMax-H3 TensorRT-RTX weight streaming does not support CUDA graphs");
}

const nlohmann::json& require_runtime_memory_object(const nlohmann::json& root) {
    if (!root.is_object() || root.value("engine_backend", std::string{}) != "trt_rtx" ||
        !root.contains("runtime_memory") || !root.at("runtime_memory").is_object()) {
        throw std::runtime_error(
            "MiniMax-H3 TensorRT-RTX bundle is missing staged runtime metadata");
    }
    return root.at("runtime_memory");
}

std::int64_t parse_weight_streaming_budget(const nlohmann::json& memory) {
    if (memory.value("mode", std::string{}) != "staged" ||
        !memory.contains("weight_streaming_budget_bytes") ||
        !memory.at("weight_streaming_budget_bytes").is_number_integer()) {
        throw std::runtime_error(
            "MiniMax-H3 TensorRT-RTX bundle has invalid staged runtime metadata");
    }
    const auto budget = memory.at("weight_streaming_budget_bytes").get<std::int64_t>();
    if (budget < 0)
        throw std::runtime_error(
            "MiniMax-H3 TensorRT-RTX weight-streaming budget must be nonnegative");
    return budget;
}

RuntimeMemoryConfig load_runtime_memory_config(const PipelineContext& ctx) {
    const std::string recorded_backend =
        extract_json_string(ctx.config_json, "engine_backend", "trt");
    if (recorded_backend != "trt_rtx")
        return {};
    validate_rtx_runtime_context(ctx, recorded_backend);
    try {
        const auto root = nlohmann::json::parse(ctx.config_json);
        return RuntimeMemoryConfig{
            true, parse_weight_streaming_budget(require_runtime_memory_object(root))};
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(std::string("MiniMax-H3 invalid runtime-memory JSON: ") +
                                 error.what());
    }
}

PlanSha256Map load_plan_sha256(const std::string& config_json, const SectionMap& sections) {
    try {
        const auto root = nlohmann::json::parse(config_json);
        if (!root.is_object() || !root.contains("plan_sha256") ||
            !root.at("plan_sha256").is_object()) {
            throw std::runtime_error("MiniMax-H3 bundle is missing plan SHA-256 records");
        }
        const auto& records = root.at("plan_sha256");
        if (records.size() != sections.size()) {
            throw std::runtime_error(
                "MiniMax-H3 bundle plan SHA-256 records do not exactly cover its plan sections");
        }
        PlanSha256Map result;
        for (const auto& [section, _] : sections) {
            const std::string filename = plan_filename(section);
            if (!records.contains(filename) || !records.at(filename).is_string())
                throw std::runtime_error("MiniMax-H3 bundle is missing plan SHA-256 for " +
                                         section);
            const std::string digest = records.at(filename).get<std::string>();
            if (!is_sha256(digest))
                throw std::runtime_error("MiniMax-H3 bundle has invalid plan SHA-256 for " +
                                         section);
            result.emplace(section, digest);
        }
        return result;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(std::string("MiniMax-H3 invalid plan SHA-256 JSON: ") +
                                 error.what());
    }
}

void validate_plan_section_attestations(const std::string& bundle_path, const SectionMap& sections,
                                        const PlanSha256Map& plan_sha256) {
    std::vector<const BundleSectionInfo*> ordered;
    ordered.reserve(sections.size());
    for (const auto& [_, section] : sections)
        ordered.push_back(&section);
    std::sort(ordered.begin(), ordered.end(), [](const auto* left, const auto* right) {
        return left->offset < right->offset ||
               (left->offset == right->offset && left->name < right->name);
    });
    for (const auto* section : ordered) {
        const auto expected = plan_sha256.find(section->name);
        if (expected == plan_sha256.end())
            throw std::runtime_error("MiniMax-H3 plan SHA-256 is missing: " + section->name);
        try {
            ValidateBundleSectionSha256(bundle_path, *section, expected->second);
        } catch (const std::runtime_error& error) {
            throw std::runtime_error("MiniMax-H3 plan section '" + section->name +
                                     "' failed runtime validation: " + error.what());
        }
    }
}

void validate_plan_section_bounds(const std::string& bundle_path, const SectionMap& sections) {
    std::vector<BundleSectionInfo> required;
    required.reserve(sections.size());
    for (const auto& [_, section] : sections)
        required.push_back(section);
    ValidateBundleSectionBounds(bundle_path, required);
}

struct CacheConfig {
    bool enabled{false};
    float threshold{0.08F};
};

struct HotEngineConfig {
    bool retain_engines{false};
    std::int64_t tail_weight_budget_bytes{24LL << 30};
};

SectionMap index_sections(const BundleInfo& info, bool first_block_cache, bool native_vsa,
                          bool ref2va) {
    constexpr std::array<const char*, 5> monolithic_names = {
        "text_encoder_plan", "adaln_precompute_plan", "denoiser_plan", "vae_tile_decoder_plan",
        "audio_vae_decoder_plan"};
    constexpr std::array<const char*, 7> first_block_cache_names = {
        "text_encoder_plan",     "adaln_precompute_plan", "denoiser_head_plan",
        "denoiser_tail_plan",    "denoiser_finish_plan",  "vae_tile_decoder_plan",
        "audio_vae_decoder_plan"};
    SectionMap sections;
    const auto add_section = [&](const char* name) {
        const auto it =
            std::find_if(info.sections.begin(), info.sections.end(),
                         [name](const BundleSectionInfo& item) { return item.name == name; });
        if (it == info.sections.end() || it->size == 0)
            throw std::runtime_error(std::string("MiniMax-H3 bundle is missing ") + name);
        sections.emplace(name, *it);
    };
    const auto add_optional_section = [&](const char* name) {
        const auto it =
            std::find_if(info.sections.begin(), info.sections.end(),
                         [name](const BundleSectionInfo& item) { return item.name == name; });
        if (it != info.sections.end() && it->size != 0)
            sections.emplace(name, *it);
    };
    if (native_vsa) {
        for (const char* name :
             {"text_encoder_plan", "adaln_precompute_plan", "denoiser_entry_plan",
              "denoiser_finish_plan", "vae_tile_decoder_plan", "audio_vae_decoder_plan"}) {
            add_section(name);
        }
        for (int32_t index = 0; index < 49; ++index) {
            std::string name = "denoiser_transition_";
            if (index < 10)
                name += '0';
            name += std::to_string(index);
            name += "_plan";
            add_section(name.c_str());
        }
    } else if (first_block_cache) {
        for (const char* name : first_block_cache_names)
            add_section(name);
    } else {
        for (const char* name : monolithic_names)
            add_section(name);
    }
    // Older T2VA-only bundles remain loadable. Complete H3-Base bundles add
    // these two sections and the structured FL2VA request path fails closed if
    // either is absent when first/last-frame conditioning is actually used.
    add_optional_section("vision_encoder_plan");
    add_optional_section("fl2va_keyframe_vae_encoder_plan");
    if (ref2va) {
        for (const char* name :
             {"vision_encoder_plan", "fl2va_keyframe_vae_encoder_plan", "ref2va_denoiser_plan",
              "ref2va_adaln_precompute_plan", "ref2va_video_vae_encoder_plan",
              "ref2va_audio_vae_encoder_plan"}) {
            add_section(name);
        }
    }
    return sections;
}

std::unique_ptr<ITokenizer> load_tokenizer(const BundleFile& bundle) {
    const auto* data = find_section(bundle, "tokenizer.json");
    if (data == nullptr || data->empty())
        throw std::runtime_error("MiniMax-H3 bundle is missing tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(data->data(), data->size(), false);
    if (!tokenizer)
        throw std::runtime_error("MiniMax-H3 could not create the native Qwen BPE tokenizer");
    return tokenizer;
}

bool int_array_equals(const nlohmann::json& object, const char* name,
                      std::initializer_list<int32_t> expected) {
    if (!object.contains(name) || !object.at(name).is_array() ||
        object.at(name).size() != expected.size()) {
        return false;
    }
    std::size_t index = 0;
    for (const int32_t value : expected) {
        const auto& item = object.at(name).at(index++);
        if (!item.is_number_integer() || item.get<int32_t>() != value)
            return false;
    }
    return true;
}

bool json_int_array_equals(const nlohmann::json& value, std::initializer_list<int32_t> expected) {
    if (!value.is_array() || value.size() != expected.size())
        return false;
    std::size_t index = 0;
    for (const int32_t expected_value : expected) {
        const auto& item = value.at(index++);
        if (!item.is_number_integer() || item.get<int32_t>() != expected_value)
            return false;
    }
    return true;
}

bool explicit_canvas_sizes_are_exact(const nlohmann::json& root) {
    if (!root.contains("explicit_canvas_sizes") || !root.at("explicit_canvas_sizes").is_array() ||
        root.at("explicit_canvas_sizes").size() != 2) {
        return false;
    }
    const auto& sizes = root.at("explicit_canvas_sizes");
    return json_int_array_equals(sizes.at(0), {544, 960}) &&
           json_int_array_equals(sizes.at(1), {960, 544});
}

bool declares_public_workflow(const nlohmann::json& root, std::string_view name) {
    if (!root.contains("public_workflows") || !root.at("public_workflows").is_array())
        return false;
    return std::any_of(
        root.at("public_workflows").begin(), root.at("public_workflows").end(),
        [name](const auto& item) { return item.is_string() && item.get<std::string>() == name; });
}

bool public_workflows_are_exact(const nlohmann::json& root,
                                std::initializer_list<std::string_view> expected) {
    if (!root.contains("public_workflows") || !root.at("public_workflows").is_array() ||
        root.at("public_workflows").size() != expected.size()) {
        return false;
    }
    std::size_t index = 0;
    for (const std::string_view name : expected) {
        const auto& item = root.at("public_workflows").at(index++);
        if (!item.is_string() || item.get<std::string>() != name)
            return false;
    }
    return true;
}

bool public_workflows_are_supported_prefix(const nlohmann::json& root) {
    if (!root.contains("public_workflows"))
        return true; // Legacy T2VA bundles predate the explicit declaration.
    const auto& workflows = root.at("public_workflows");
    constexpr std::array<std::string_view, 3> supported = {"t2va", "fl2va", "ref2va"};
    if (!workflows.is_array() || workflows.empty() || workflows.size() > supported.size())
        return false;
    for (std::size_t index = 0; index < workflows.size(); ++index) {
        if (!workflows.at(index).is_string() ||
            workflows.at(index).get<std::string>() != supported.at(index)) {
            return false;
        }
    }
    return true;
}

bool tensor_metadata_equals(const nlohmann::json& tensor, std::string_view name,
                            std::string_view dtype, std::initializer_list<int32_t> minimum,
                            std::initializer_list<int32_t> optimum,
                            std::initializer_list<int32_t> maximum) {
    return tensor.is_object() && tensor.value("name", std::string{}) == name &&
           tensor.value("dtype", std::string{}) == dtype && tensor.contains("min_shape") &&
           json_int_array_equals(tensor.at("min_shape"), minimum) && tensor.contains("opt_shape") &&
           json_int_array_equals(tensor.at("opt_shape"), optimum) && tensor.contains("max_shape") &&
           json_int_array_equals(tensor.at("max_shape"), maximum);
}

bool plan_metadata_header(const nlohmann::json& plan, std::string_view filename,
                          std::size_t input_count, std::size_t output_count) {
    return plan.is_object() && plan.value("filename", std::string{}) == filename &&
           plan.contains("inputs") && plan.at("inputs").is_array() &&
           plan.at("inputs").size() == input_count && plan.contains("outputs") &&
           plan.at("outputs").is_array() && plan.at("outputs").size() == output_count;
}

bool ref2va_plan_abis_are_exact(const nlohmann::json& root) {
    if (!root.contains("ref2va_plan_abis") || !root.at("ref2va_plan_abis").is_object())
        return false;
    const auto& plans = root.at("ref2va_plan_abis");
    if (plans.size() != 4U)
        return false;
    for (const char* name : {"ref2va_denoiser_plan", "ref2va_adaln_precompute_plan",
                             "ref2va_video_vae_encoder_plan", "ref2va_audio_vae_encoder_plan"}) {
        if (!plans.contains(name))
            return false;
    }

    const auto& denoiser = plans.at("ref2va_denoiser_plan");
    if (!plan_metadata_header(denoiser, "ref2va_denoiser.plan", 60, 2))
        return false;
    const auto& inputs = denoiser.at("inputs");
    const auto dynamic = [&](std::size_t index, std::string_view name, std::string_view dtype,
                             std::initializer_list<int32_t> minimum,
                             std::initializer_list<int32_t> optimum,
                             std::initializer_list<int32_t> maximum) {
        return tensor_metadata_equals(inputs.at(index), name, dtype, minimum, optimum, maximum);
    };
    if (!dynamic(0, "video_hidden_states", "float32", {18870, 96}, {44592, 96}, {364608, 96}) ||
        !dynamic(1, "audio_hidden_states", "float32", {414, 32}, {414, 32}, {3558, 32}) ||
        !dynamic(2, "encoder_hidden_states", "float32", {1, 5120}, {7433, 5120}, {262144, 5120}) ||
        !dynamic(3, "position_ids", "float32", {19285, 3}, {52439, 3}, {630310, 3}) ||
        !dynamic(4, "video_indices", "int32", {18870}, {44592}, {364608}) ||
        !dynamic(5, "audio_indices", "int32", {414}, {414}, {3558}) ||
        !dynamic(6, "text_indices", "int32", {1}, {7433}, {262144}) ||
        !dynamic(7, "adaln_indices", "int32", {19285}, {52439}, {630310}) ||
        !dynamic(8, "timestep_indices", "int32", {19285}, {52439}, {630310})) {
        return false;
    }
    for (int32_t layer = 0; layer < 50; ++layer) {
        if (!dynamic(static_cast<std::size_t>(9 + layer),
                     "block_modulation_" + std::to_string(layer), "bfloat16", {12, 6, 5376},
                     {12, 6, 5376}, {12, 6, 5376})) {
            return false;
        }
    }
    if (!dynamic(59, "final_modulation", "bfloat16", {4, 2, 5376}, {4, 2, 5376}, {4, 2, 5376}) ||
        !tensor_metadata_equals(denoiser.at("outputs").at(0), "video_velocity", "float32",
                                {18870, 96}, {44592, 96}, {364608, 96}) ||
        !tensor_metadata_equals(denoiser.at("outputs").at(1), "audio_velocity", "float32",
                                {414, 32}, {414, 32}, {3558, 32})) {
        return false;
    }

    const auto& adaln = plans.at("ref2va_adaln_precompute_plan");
    if (!plan_metadata_header(adaln, "ref2va_adaln_precompute.plan", 1, 51) ||
        !tensor_metadata_equals(adaln.at("inputs").at(0), "timestep_features", "float32", {4, 256},
                                {4, 256}, {4, 256})) {
        return false;
    }
    for (int32_t layer = 0; layer < 50; ++layer) {
        if (!tensor_metadata_equals(adaln.at("outputs").at(static_cast<std::size_t>(layer)),
                                    "block_modulation_" + std::to_string(layer), "bfloat16",
                                    {12, 6, 5376}, {12, 6, 5376}, {12, 6, 5376})) {
            return false;
        }
    }
    if (!tensor_metadata_equals(adaln.at("outputs").at(50), "final_modulation", "bfloat16",
                                {4, 2, 5376}, {4, 2, 5376}, {4, 2, 5376})) {
        return false;
    }

    const auto& video = plans.at("ref2va_video_vae_encoder_plan");
    if (!plan_metadata_header(video, "ref2va_video_vae_encoder.plan", 1, 1) ||
        !tensor_metadata_equals(video.at("inputs").at(0), "pixel_tile_clip", "float32",
                                {1, 3, 17, 256, 256}, {1, 3, 17, 256, 256}, {1, 3, 17, 256, 256}) ||
        !tensor_metadata_equals(video.at("outputs").at(0), "posterior_parameter_tile_clip",
                                "float32", {1, 48, 5, 16, 16}, {1, 48, 5, 16, 16},
                                {1, 48, 5, 16, 16})) {
        return false;
    }
    const auto& audio = plans.at("ref2va_audio_vae_encoder_plan");
    return plan_metadata_header(audio, "ref2va_audio_vae_encoder.plan", 1, 1) &&
           tensor_metadata_equals(audio.at("inputs").at(0), "audio_samples", "float32",
                                  {2, 1, 64000}, {2, 1, 165600}, {2, 1, 480000}) &&
           tensor_metadata_equals(audio.at("outputs").at(0), "posterior_mean", "float32",
                                  {2, 32, 80}, {2, 32, 207}, {2, 32, 600});
}

bool exact_string_map(const nlohmann::json& value,
                      std::initializer_list<std::pair<std::string_view, std::string_view>> map) {
    if (!value.is_object() || value.size() != map.size())
        return false;
    return std::all_of(map.begin(), map.end(), [&](const auto& expected) {
        return value.contains(std::string(expected.first)) &&
               value.at(std::string(expected.first)).is_string() &&
               value.at(std::string(expected.first)).get<std::string>() == expected.second;
    });
}

std::array<float, 32> load_ref2va_audio_array(const nlohmann::json& root, const char* name,
                                              bool positive) {
    if (!root.contains(name) || !root.at(name).is_array() || root.at(name).size() != 32U)
        throw std::runtime_error(std::string("MiniMax-H3 Ref2VA bundle has invalid ") + name);
    std::array<float, 32> result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        const auto& item = root.at(name).at(index);
        if (!item.is_number())
            throw std::runtime_error(std::string("MiniMax-H3 Ref2VA bundle has invalid ") + name);
        result[index] = item.get<float>();
        if (!std::isfinite(result[index]) || (positive && result[index] <= 0.0F))
            throw std::runtime_error(std::string("MiniMax-H3 Ref2VA bundle has invalid ") + name);
    }
    return result;
}

bool ref2va_provenance_files_are_exact(const nlohmann::json& files) {
    struct ExpectedFile {
        const char* name;
        std::int64_t bytes;
        const char* sha256;
    };
    constexpr std::array<ExpectedFile, 16> expected = {{
        {"config.json", 546LL, "74c11bff524336576096993cbfcdcdc2ef4fa2fa4409df693bdcbc6c666282ae"},
        {"diffusion_pytorch_model.safetensors.index.json", 64488LL,
         "ac30a3b58963f2e735d493475fbb81853a5735ec947619648b3e045acda6783e"},
        {"diffusion_pytorch_model-00001-of-00014.safetensors", 4825958704LL,
         "7a3fcad885f51560e550b2e84c9a8d8b35e62996cfd9076937e992bd23478df9"},
        {"diffusion_pytorch_model-00002-of-00014.safetensors", 4702158032LL,
         "1638ae1dc8ae26c4ba43ad28a6d851ad8983847324bb2b468719c7c81f219706"},
        {"diffusion_pytorch_model-00003-of-00014.safetensors", 4933368192LL,
         "1ef3c4954ffe5a664c2e3028e2a3241190d9c159dce6ba1136002c6af1db5353"},
        {"diffusion_pytorch_model-00004-of-00014.safetensors", 4567069608LL,
         "12d92f2975cfd5c5b786126385c52e5bf64884d4b4d6e60c3ef5d857c3f7469f"},
        {"diffusion_pytorch_model-00005-of-00014.safetensors", 4702158080LL,
         "304d41ce03d59ac94bceb055935bf4e034df0badf8b0df4ded327c08a288a4cc"},
        {"diffusion_pytorch_model-00006-of-00014.safetensors", 4933368232LL,
         "12a134b7c76d86edbe8fa2dc315f6cdaf4e1aca1b6ea4dfe4cad92df03d42eeb"},
        {"diffusion_pytorch_model-00007-of-00014.safetensors", 4567069608LL,
         "b96395261359937c00fb42f4eb29306dc59b1a3368eeba52af4fb66e3e142c69"},
        {"diffusion_pytorch_model-00008-of-00014.safetensors", 4702158080LL,
         "1897a6bf3b4fc834bb82d73ca02a7afc7d38c07f50ec5382cd54cd2f91b604d1"},
        {"diffusion_pytorch_model-00009-of-00014.safetensors", 4933368232LL,
         "edfb38235adc96b99f55a401849befce59075a745e99c2d8c63ff358dd36443d"},
        {"diffusion_pytorch_model-00010-of-00014.safetensors", 4567069608LL,
         "f8710775cf3413670edd7e23861b650a3431a71a6cc14cb1080623ab6b052385"},
        {"diffusion_pytorch_model-00011-of-00014.safetensors", 4702158080LL,
         "9e18acc09f84edb5b34df9628efa15cfcab8bb76e8e20c1c2e979a107a0f7215"},
        {"diffusion_pytorch_model-00012-of-00014.safetensors", 4933368232LL,
         "ea2e18228f8bdba1a4e0f32b155e4586df055997c45356213d05b971ba13e2f4"},
        {"diffusion_pytorch_model-00013-of-00014.safetensors", 4567069608LL,
         "1e12083b1875678f7414ff55b09cd8bb1c30b861243f9bb7ff1e75b6ad3f1bdc"},
        {"diffusion_pytorch_model-00014-of-00014.safetensors", 4644161920LL,
         "b340f44b5690cc745d48ae399381ec15b26a4fe25d483f677ccb4960dadb50d4"},
    }};
    if (!files.is_object() || files.size() != expected.size())
        return false;
    return std::all_of(expected.begin(), expected.end(), [&](const ExpectedFile& file) {
        if (!files.contains(file.name) || !files.at(file.name).is_object())
            return false;
        const auto& record = files.at(file.name);
        return record.size() == 2U && record.value("bytes", std::int64_t{0}) == file.bytes &&
               record.value("sha256", std::string{}) == file.sha256;
    });
}

MiniMaxH3Ref2VAConfig load_ref2va_config(const PipelineContext& ctx) {
    try {
        const auto root = nlohmann::json::parse(ctx.config_json);
        if (!root.is_object())
            throw std::runtime_error("MiniMax-H3 bundle config must be a JSON object");
        if (!public_workflows_are_supported_prefix(root))
            throw std::runtime_error(
                "MiniMax-H3 public workflow declaration must be an ordered prefix of "
                "[t2va, fl2va, ref2va]");
        const bool contains_ref2va_plan = std::any_of(
            ctx.bundle.info.sections.begin(), ctx.bundle.info.sections.end(),
            [](const BundleSectionInfo& section) {
                return section.size != 0 && (section.name == "ref2va_denoiser_plan" ||
                                             section.name == "ref2va_adaln_precompute_plan" ||
                                             section.name == "ref2va_video_vae_encoder_plan" ||
                                             section.name == "ref2va_audio_vae_encoder_plan");
            });
        const bool advertised = declares_public_workflow(root, "ref2va");
        if (!advertised) {
            if (contains_ref2va_plan || root.value("ref2va_supported", false))
                throw std::runtime_error(
                    "MiniMax-H3 bundle has Ref2VA plans or metadata without the exact public "
                    "workflow declaration");
            return {};
        }
        if (root.value("ref2va_schema_version", 0) != 2 || !root.value("ref2va_supported", false) ||
            !public_workflows_are_exact(root, {"t2va", "fl2va", "ref2va"}) ||
            root.value("engine_backend", std::string{}) != "trt_rtx" ||
            root.value("ref2va_context_ir_supported", true) ||
            root.value("ref2va_regenerate_2k_supported", true) ||
            root.value("ref2va_runtime_language", std::string{}) != "c++/cuda" ||
            root.value("ref2va_transformer_fallback_allowed", true) ||
            !root.contains("ref2va_runtime_dependencies") ||
            !root.at("ref2va_runtime_dependencies").is_array() ||
            root.at("ref2va_runtime_dependencies").size() != 1U ||
            !root.at("ref2va_runtime_dependencies").at(0).is_string() ||
            root.at("ref2va_runtime_dependencies").at(0).get<std::string>() != "TensorRT-RTX") {
            throw std::runtime_error("MiniMax-H3 Ref2VA native runtime declaration is invalid");
        }
        if (!root.contains("ref2va_scheduler") || !root.at("ref2va_scheduler").is_object()) {
            throw std::runtime_error("MiniMax-H3 Ref2VA scheduler metadata is missing");
        }
        const auto& scheduler = root.at("ref2va_scheduler");
        MiniMaxH3Ref2VAConfig result;
        result.enabled = true;
        result.scheduler_grid_points = scheduler.value("sigma_grid_points", 0);
        result.transformer_forwards = scheduler.value("transformer_forwards", 0);
        result.video_shift = scheduler.value("video_shift", 0.0F);
        result.audio_shift = scheduler.value("audio_shift", 0.0F);
        result.guidance_scale = scheduler.value("guidance_scale", -1.0F);
        result.guidance_distilled = scheduler.value("guidance_distilled", false);
        if (scheduler.size() != 6U || result.scheduler_grid_points != 50 ||
            result.transformer_forwards != 49 || result.video_shift != 12.0F ||
            result.audio_shift != 3.0F || result.guidance_scale != 1.0F ||
            !result.guidance_distilled) {
            throw std::runtime_error("MiniMax-H3 Ref2VA scheduler metadata is invalid");
        }
        if (!root.contains("ref2va_plan_sections") ||
            !exact_string_map(root.at("ref2va_plan_sections"),
                              {{"ref2va_denoiser", "ref2va_denoiser_plan"},
                               {"ref2va_adaln_precompute", "ref2va_adaln_precompute_plan"},
                               {"ref2va_video_vae_encoder", "ref2va_video_vae_encoder_plan"},
                               {"ref2va_audio_vae_encoder", "ref2va_audio_vae_encoder_plan"}}) ||
            !root.contains("ref2va_shared_sections") ||
            !exact_string_map(root.at("ref2va_shared_sections"),
                              {{"text_encoder", "text_encoder_plan"},
                               {"vision_encoder", "vision_encoder_plan"},
                               {"image_vae_encoder", "fl2va_keyframe_vae_encoder_plan"},
                               {"video_vae_decoder", "vae_tile_decoder_plan"},
                               {"audio_vae_decoder", "audio_vae_decoder_plan"}}) ||
            !ref2va_plan_abis_are_exact(root)) {
            throw std::runtime_error("MiniMax-H3 Ref2VA plan/section ABI metadata is invalid");
        }

        if (!root.contains("ref2va_shared_qwen_profiles") ||
            !root.at("ref2va_shared_qwen_profiles").is_object()) {
            throw std::runtime_error("MiniMax-H3 Ref2VA shared Qwen profile is missing");
        }
        const auto& qwen = root.at("ref2va_shared_qwen_profiles");
        if (qwen.size() != 2U || !qwen.contains("vision_encoder_plan") ||
            !qwen.contains("text_encoder_plan")) {
            throw std::runtime_error("MiniMax-H3 Ref2VA shared Qwen profile is invalid");
        }
        const auto& vision = qwen.at("vision_encoder_plan");
        const auto& text = qwen.at("text_encoder_plan");
        if (!int_array_equals(vision, "patch_rows_per_call", {2040, 4032, 65536}) ||
            vision.value("invocation_unit", std::string{}) !=
                "one_image_or_one_two_frame_video_temporal_block" ||
            vision.value("spatial_chunking_allowed", true) ||
            !vision.value("concatenate_outputs_in_reference_timestamp_order", false) ||
            !int_array_equals(text, "sequence_rows", {1, 1144, 262144}) ||
            !int_array_equals(text, "compact_vision_rows", {1, 1008, 262144}) ||
            text.value("sequence_chunking_allowed", true)) {
            throw std::runtime_error("MiniMax-H3 Ref2VA shared Qwen profile is invalid");
        }

        if (!root.contains("ref2va_limits") || !root.at("ref2va_limits").is_object() ||
            !root.contains("ref2va_capacity") || !root.at("ref2va_capacity").is_object()) {
            throw std::runtime_error("MiniMax-H3 Ref2VA limits/capacity metadata is missing");
        }
        const auto& limits = root.at("ref2va_limits");
        const auto& capacity = root.at("ref2va_capacity");
        const bool exact_limits =
            limits.size() == 10U && limits.value("max_images", 0) == 9 &&
            limits.value("max_videos", 0) == 3 && limits.value("max_explicit_audios", 0) == 3 &&
            limits.value("max_reference_files", 0) == 12 &&
            limits.value("min_seconds_each_video_or_audio", 0.0) == 2.0 &&
            limits.value("max_seconds_each_video_or_audio", 0.0) == 15.0 &&
            limits.value("max_total_video_seconds", 0.0) == 15.0 &&
            limits.value("max_total_explicit_audio_seconds", 0.0) == 15.0 &&
            limits.value("requires_image_or_video", false) &&
            limits.value("video_soundtrack_stays_attached", false) &&
            int_array_equals(capacity, "video_rows", {18870, 44592, 364608}) &&
            int_array_equals(capacity, "audio_rows", {414, 414, 3558}) &&
            int_array_equals(capacity, "text_rows", {1, 7433, 262144}) &&
            int_array_equals(capacity, "packed_rows", {19285, 52439, 630310});
        if (!exact_limits)
            throw std::runtime_error("MiniMax-H3 Ref2VA limits/capacity metadata is invalid");

        if (!root.contains("ref2va_transformer_ref") ||
            !root.at("ref2va_transformer_ref").is_object()) {
            throw std::runtime_error("MiniMax-H3 Ref2VA transformer provenance is missing");
        }
        const auto& provenance = root.at("ref2va_transformer_ref");
        if (provenance.value("schema_version", 0) != 1 ||
            provenance.value("model_id", std::string{}) != "MiniMaxAI/MiniMax-H3" ||
            provenance.value("revision", std::string{}) !=
                "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc" ||
            provenance.value("component", std::string{}) != "transformer_ref" ||
            provenance.value("tensor_bytes", std::int64_t{0}) != 66280430080LL ||
            provenance.value("tensor_count", 0) != 638 ||
            !provenance.contains("inventory_sha256") ||
            !provenance.at("inventory_sha256").is_string() ||
            provenance.at("inventory_sha256").get<std::string>() !=
                "ee55ebab7503e89d1eeab8cd788fc58402a9f2e5379986d5c78345dbefd0e980" ||
            !provenance.contains("files") || !provenance.at("files").is_object() ||
            !ref2va_provenance_files_are_exact(provenance.at("files")) ||
            !provenance.contains("runtime_framework") ||
            !provenance.at("runtime_framework").is_null()) {
            throw std::runtime_error("MiniMax-H3 Ref2VA transformer provenance is invalid");
        }

        result.audio_latent_mean = load_ref2va_audio_array(root, "audio_latents_mean", false);
        result.audio_latent_std = load_ref2va_audio_array(root, "audio_latents_std", true);
        return result;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(std::string("MiniMax-H3 invalid Ref2VA JSON: ") + error.what());
    }
}

bool vsa_tensor_abi_is_exact(const nlohmann::json& vsa) {
    if (!vsa.contains("tensor_abi") || !vsa.at("tensor_abi").is_object())
        return false;
    const auto& abi = vsa.at("tensor_abi");
    const auto exact_shape = [&](const char* name, int32_t first, const char* middle,
                                 int32_t last) {
        if (!abi.contains(name) || !abi.at(name).is_array() || abi.at(name).size() != 3U) {
            return false;
        }
        const auto& shape = abi.at(name);
        return shape.at(0).is_number_integer() && shape.at(0).get<int32_t>() == first &&
               shape.at(1).is_string() && shape.at(1).get<std::string>() == middle &&
               shape.at(2).is_number_integer() && shape.at(2).get<int32_t>() == last;
    };
    const bool residual_shape =
        abi.contains("residual_shape") && abi.at("residual_shape").is_array() &&
        abi.at("residual_shape").size() == 2U && abi.at("residual_shape").at(0).is_string() &&
        abi.at("residual_shape").at(0).get<std::string>() == "S" &&
        abi.at("residual_shape").at(1).is_number_integer() &&
        abi.at("residual_shape").at(1).get<int32_t>() == 5376;
    return abi.value("dtype", std::string{}) == "bf16" &&
           abi.value("residual_input", std::string{}) == "residual_hidden" &&
           abi.value("residual_output", std::string{}) == "next_residual_hidden" &&
           abi.value("attention_input", std::string{}) == "vsa_attention_output" &&
           abi.value("query_output", std::string{}) == "vsa_query" &&
           abi.value("key_output", std::string{}) == "vsa_key" &&
           abi.value("value_output", std::string{}) == "vsa_value" &&
           abi.value("gate_output", std::string{}) == "vsa_gate" && residual_shape &&
           exact_shape("attention_shape", 56, "S", 128) &&
           int_array_equals(abi, "packed_rows_profile", {19285, 37838, 112367});
}

bool vsa_segments_are_exact(const nlohmann::json& vsa) {
    if (!vsa.contains("segments") || !vsa.at("segments").is_array() ||
        vsa.at("segments").size() != 51U) {
        return false;
    }
    const auto segment_matches = [&](std::size_t offset, const std::string& component,
                                     const std::string& filename, const std::string& section) {
        const auto& item = vsa.at("segments").at(offset);
        return item.is_object() && item.value("component", std::string{}) == component &&
               item.value("filename", std::string{}) == filename &&
               item.value("section", std::string{}) == section;
    };
    if (!segment_matches(0, "denoiser_entry", "denoiser_entry.plan", "denoiser_entry_plan")) {
        return false;
    }
    for (int32_t index = 0; index < 49; ++index) {
        std::string suffix = index < 10 ? "0" : "";
        suffix += std::to_string(index);
        const std::string component = "denoiser_transition_" + suffix;
        if (!segment_matches(static_cast<std::size_t>(index + 1), component, component + ".plan",
                             component + "_plan")) {
            return false;
        }
    }
    return segment_matches(50, "denoiser_finish", "denoiser_finish.plan", "denoiser_finish_plan");
}

void validate_fl2va_conditioning_contract(const PipelineContext& ctx, const SectionMap& sections) {
    try {
        const auto root = nlohmann::json::parse(ctx.config_json);
        if (!root.contains("public_workflows") || !root.at("public_workflows").is_array())
            return; // Legacy T2VA-only bundle.
        const bool declares_fl2va = declares_public_workflow(root, "fl2va");
        const bool declares_ref2va = declares_public_workflow(root, "ref2va");
        if (!declares_fl2va)
            return;
        if (sections.count("vision_encoder_plan") == 0 ||
            sections.count("fl2va_keyframe_vae_encoder_plan") == 0 ||
            !root.contains("conditioning") || !root.at("conditioning").is_object()) {
            throw std::runtime_error(
                "MiniMax-H3 FL2VA bundle is missing its native conditioning plans");
        }
        const auto& conditioning = root.at("conditioning");
        const bool exact_qwen_profile =
            declares_ref2va
                ? int_array_equals(conditioning, "text_sequence_profile", {1, 1144, 262144}) &&
                      int_array_equals(conditioning, "vision_patch_profile", {2040, 4032, 65536}) &&
                      int_array_equals(conditioning, "vision_row_profile", {1, 1008, 262144})
                : int_array_equals(conditioning, "text_sequence_profile", {1, 1144, 2641}) &&
                      int_array_equals(conditioning, "vision_patch_profile", {2040, 4032, 4176}) &&
                      int_array_equals(conditioning, "vision_row_profile", {1, 1008, 2088});
        const bool exact =
            conditioning.value("implementation", std::string{}) == "shared_native_qwen3_vl" &&
            conditioning.value("text_encoder_section", std::string{}) == "text_encoder_plan" &&
            conditioning.value("vision_encoder_section", std::string{}) == "vision_encoder_plan" &&
            conditioning.value("keyframe_vae_encoder_section", std::string{}) ==
                "fl2va_keyframe_vae_encoder_plan" &&
            exact_qwen_profile &&
            int_array_equals(conditioning, "keyframe_vae_tile_batch_profile", {1, 28, 33}) &&
            conditioning.value("t2va_dummy_vision_rows", 0) == 1 &&
            conditioning.value("t2va_vision_count", -1) == 0 &&
            conditioning.value("t2va_vision_mask_nonzero", -1) == 0 &&
            conditioning.value("reachable_canvas_count", 0) == 95 &&
            int_array_equals(conditioning, "max_rounded_canvas", {576, 1856}) &&
            conditioning.value("max_condition_video_rows", 0) == 2088 &&
            conditioning.value("mode_coupled_profile_required", false);
        if (!exact)
            throw std::runtime_error(
                "MiniMax-H3 FL2VA bundle has an invalid native conditioning ABI");
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(std::string("MiniMax-H3 invalid FL2VA conditioning JSON: ") +
                                 error.what());
    }
}

void validate_profile(const PipelineContext& ctx, const MiniMaxH3DenoiserConfig& denoiser) {
    if (ctx.backend == nullptr)
        throw std::runtime_error("MiniMax-H3 requires the TensorRT backend");
    if (extract_json_int(ctx.config_json, "context_parallel_size", 1) != 1)
        throw std::runtime_error("MiniMax-H3 requires context_parallel_size=1");
    if (extract_json_int(ctx.config_json, "vae_tile_batch", 28) != 28)
        throw std::runtime_error("MiniMax-H3 requires vae_tile_batch=28");
    const int32_t packed = extract_json_int(ctx.config_json, "padded_sequence_length", 38247);
    if (!denoiser.native_vsa) {
        if (packed != 38247 && packed != 108175 && packed != 112367)
            throw std::runtime_error("MiniMax-H3 dense bundle has an invalid sequence profile");
        const int32_t expected_max_text_rows = packed == 112367 ? 2641 : 537;
        if (denoiser.max_text_rows != expected_max_text_rows)
            throw std::runtime_error("MiniMax-H3 dense bundle has an invalid text profile");
        return;
    }

    try {
        const auto root = nlohmann::json::parse(ctx.config_json);
        const bool exact =
            packed == 112367 && root.value("canvas_multiple", 0) == 32 &&
            root.value("canvas_short_edge", 0) == 768 &&
            root.value("canvas_max_pixels", 0) == 768 * 1344 &&
            explicit_canvas_sizes_are_exact(root) && root.value("num_frames_min", 0) == 124 &&
            root.value("num_frames_opt", 0) == 124 && root.value("num_frames_max", 0) == 345 &&
            denoiser.max_text_rows == 2641 &&
            root.value("text_rows_min", 0) == 1 && root.value("text_rows_opt", 0) == 128 &&
            root.value("text_rows_max", 0) == 2641 && root.value("audio_rows_min", 0) == 414 &&
            root.value("audio_rows_opt", 0) == 414 && root.value("audio_rows_max", 0) == 1150 &&
            root.value("video_rows_min", 0) == 18870 && root.value("video_rows_opt", 0) == 37296 &&
            root.value("video_rows_max", 0) == 108576 &&
            root.value("packed_sequence_length_min", 0) == 19285 &&
            root.value("packed_sequence_length_opt", 0) == 37838 &&
            root.value("packed_sequence_length_max", 0) == 112367 &&
            root.value("vae_tile_batch_min", 0) == 15 &&
            root.value("vae_tile_batch_opt", 0) == 28 && root.value("vae_tile_batch_max", 0) == 33;
        if (!exact)
            throw std::runtime_error(
                "MiniMax-H3 native VSA bundle does not declare the public dynamic profile");
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(std::string("MiniMax-H3 invalid native VSA profile JSON: ") +
                                 error.what());
    }
}

CacheConfig load_cache_config(const PipelineContext& ctx) {
    CacheConfig result;
    const std::string mode =
        extract_json_string(ctx.config_json, "denoiser_cache_mode", "monolithic");
    if (mode != "monolithic" && mode != "first_block" && mode != "segmented_vsa")
        throw std::runtime_error("MiniMax-H3 bundle has an invalid denoiser_cache_mode");
    result.enabled = extract_json_bool(ctx.config_json, "first_block_cache", false);
    if (result.enabled != (mode == "first_block"))
        throw std::runtime_error("MiniMax-H3 bundle cache mode and profile flag disagree");
    result.threshold =
        extract_json_float(ctx.config_json, "first_block_cache_threshold", result.threshold);
    if (ctx.runtime_config != nullptr &&
        ctx.runtime_config->source_of("minimax_h3", "first_block_cache_threshold") !=
            config::Layer::SchemaDefault) {
        result.threshold = static_cast<float>(
            ctx.runtime_config->get<double>("minimax_h3", "first_block_cache_threshold"));
    }
    if (!std::isfinite(result.threshold) || result.threshold <= 0.0F)
        throw std::runtime_error(
            "MiniMax-H3 first_block_cache_threshold must be finite and positive");
    return result;
}

MiniMaxH3DenoiserConfig load_denoiser_config(const PipelineContext& ctx, const CacheConfig& cache) {
    try {
        const auto root = nlohmann::json::parse(ctx.config_json);
        if (!root.is_object())
            throw std::runtime_error("MiniMax-H3 bundle config must be a JSON object");
        const std::string attention = root.value("attention_mode", std::string("dense"));
        MiniMaxH3DenoiserConfig result;
        result.native_vsa = attention == "native_vsa";
        if (!result.native_vsa && attention != "dense")
            throw std::runtime_error("MiniMax-H3 bundle has an invalid attention_mode");
        result.scheduler_grid_points = root.value("scheduler_grid_points", 50);
        result.transformer_forwards = root.value("transformer_forwards", 49);
        result.guidance_scale = root.value("guidance_scale", 1.0F);
        const int32_t packed = root.value("padded_sequence_length", 38247);
        result.max_text_rows =
            root.value("text_rows_max", packed == 112367 ? 2641 : 537);

        if (!result.native_vsa) {
            if (result.scheduler_grid_points != 50 || result.transformer_forwards != 49 ||
                result.guidance_scale != 1.0F) {
                throw std::runtime_error(
                    "MiniMax-H3 dense bundle has an invalid scheduler contract");
            }
            return result;
        }

        if (cache.enabled || root.value("denoiser_cache_mode", std::string{}) != "segmented_vsa" ||
            root.value("num_inference_steps", 0) != 4 || result.scheduler_grid_points != 5 ||
            result.transformer_forwards != 4 || result.guidance_scale != 1.0F ||
            !root.contains("fast_h3") || !root.at("fast_h3").is_object() || !root.contains("vsa") ||
            !root.at("vsa").is_object()) {
            throw std::runtime_error(
                "MiniMax-H3 native VSA requires the authenticated segmented four-forward "
                "FastH3 contract");
        }
        const auto& vsa = root.at("vsa");
        const bool exact_vsa =
            vsa.value("implementation", std::string{}) == "native_cuda_segmented" &&
            vsa.value("segment_count", 0) == 51 && vsa.value("transition_count", 0) == 49 &&
            vsa.value("attention_calls_per_forward", 0) == 50 &&
            !vsa.value("fbc_composable", true) && vsa.value("tile_size", 0) == 64 &&
            int_array_equals(vsa, "video_tile_shape", {4, 4, 4}) &&
            vsa.value("video_keep_numerator", 0) == 1 &&
            vsa.value("video_keep_denominator", 0) == 10 &&
            vsa.value("max_video_tiles", 0) == 2080 && vsa.value("max_total_tiles", 0) == 2140 &&
            int_array_equals(vsa, "packed_row_to_tile_slot_profile", {19285, 37838, 112367}) &&
            int_array_equals(vsa, "prefix_valid_sizes_profile", {8, 9, 60}) &&
            int_array_equals(vsa, "video_valid_sizes_profile", {360, 660, 2080}) &&
            vsa_tensor_abi_is_exact(vsa) && vsa_segments_are_exact(vsa);
        if (!exact_vsa)
            throw std::runtime_error("MiniMax-H3 bundle has an invalid native VSA tensor ABI");
        return result;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(std::string("MiniMax-H3 invalid denoiser JSON: ") + error.what());
    }
}

HotEngineConfig load_hot_engine_config(const PipelineContext& ctx) {
    HotEngineConfig result;
    if (ctx.runtime_config == nullptr)
        return result;
    result.retain_engines = ctx.runtime_config->get<bool>("minimax_h3", "retain_engines");
    const auto budget_gib =
        ctx.runtime_config->get<std::int64_t>("minimax_h3", "retained_tail_weight_budget_gib");
    if (budget_gib <= 0 || budget_gib > (std::numeric_limits<std::int64_t>::max() >> 30)) {
        throw std::runtime_error("MiniMax-H3 retained_tail_weight_budget_gib must be positive");
    }
    result.tail_weight_budget_bytes = budget_gib << 30;
    return result;
}

const BundleSectionInfo& require_plan_section(const SectionMap& sections, const std::string& name) {
    const auto it = sections.find(name);
    if (it == sections.end())
        throw std::runtime_error("Unknown MiniMax-H3 plan section: " + name);
    return it->second;
}

bool retain_hot_engine(std::string_view name, const HotEngineConfig& hot) {
    return minimax_h3::should_retain_hot_engine(name, hot.retain_engines);
}

ModuleCreateOptions module_options(cudaStream_t stream, const std::string& runtime_cache,
                                   bool cuda_graphs, bool verify_plan_sha256) {
    ModuleCreateOptions options;
    options.stream = stream;
    options.runtime_cache_path = runtime_cache.c_str();
    options.cuda_graphs = cuda_graphs;
    options.verify_plan_sha256 = verify_plan_sha256;
    return options;
}

std::int64_t staged_plan_budget(const std::string& name, const RuntimeMemoryConfig& memory,
                                const HotEngineConfig& hot) {
    return minimax_h3::staged_plan_weight_streaming_budget(
        name, memory.weight_streaming_budget_bytes, hot.retain_engines,
        hot.tail_weight_budget_bytes);
}

std::unique_ptr<ITrtModule> load_staged_module(
    const std::string& name, const BundleSectionInfo& section, const std::string& bundle_path,
    const PlanSha256Map& plan_sha256, IFileBackedBackend* file_backed_backend,
    const ModuleCreateOptions& options, const std::vector<ModuleExternalBinding>& external_bindings,
    const RuntimeMemoryConfig& memory, const HotEngineConfig& hot, bool serial_execution_context) {
    if (file_backed_backend == nullptr)
        throw std::runtime_error("MiniMax-H3 TensorRT-RTX backend lacks file-backed plan support");
    const auto digest = plan_sha256.find(name);
    if (digest == plan_sha256.end())
        throw std::runtime_error("MiniMax-H3 plan SHA-256 is missing: " + name);
    const auto range = ResolveBundleSectionFileRange(bundle_path, section);
    std::unique_ptr<ITrtModule> module;
    if (serial_execution_context) {
        auto* serial_backend = dynamic_cast<ISerialFileBackedBackend*>(file_backed_backend);
        if (serial_backend == nullptr) {
            throw std::runtime_error(
                "MiniMax-H3 TensorRT-RTX backend lacks serial activation arena support");
        }
        module = serial_backend->create_serial_module_from_file(
            bundle_path.c_str(), range.offset, range.size, digest->second.c_str(), options,
            external_bindings, staged_plan_budget(name, memory, hot), retain_hot_engine(name, hot));
    } else {
        module = file_backed_backend->create_module_from_file(
            bundle_path.c_str(), range.offset, range.size, digest->second.c_str(), options,
            external_bindings, staged_plan_budget(name, memory, hot), retain_hot_engine(name, hot));
    }
    if (!module)
        throw std::runtime_error("MiniMax-H3 backend rejected file-backed plan deserialization");
    return module;
}

std::unique_ptr<ITrtModule>
load_in_memory_module(const BundleSectionInfo& section, const std::string& bundle_path,
                      IBackend* backend, IPreboundBackend* prebound_backend,
                      const ModuleCreateOptions& options,
                      const std::vector<ModuleExternalBinding>& external_bindings) {
    auto plan = ReadBundleSection(bundle_path, section);
    if (external_bindings.empty())
        return backend->create_module(plan.data(), plan.size(), options);
    if (prebound_backend == nullptr)
        throw std::runtime_error("MiniMax-H3 backend lacks external I/O prebinding support");
    return prebound_backend->create_module_prebound(plan.data(), plan.size(), options,
                                                    external_bindings);
}

std::unique_ptr<ITrtModule> load_module(const std::string& name, cudaStream_t stream,
                                        const std::vector<ModuleExternalBinding>& external_bindings,
                                        const SectionMap& sections, const std::string& bundle_path,
                                        const std::string& runtime_cache, IBackend* backend,
                                        bool cuda_graphs, const RuntimeMemoryConfig& memory,
                                        const PlanSha256Map& plan_sha256,
                                        const HotEngineConfig& hot, bool verify_plan_sha256) {
    const auto& section = require_plan_section(sections, name);
    const auto options = module_options(stream, runtime_cache, cuda_graphs, verify_plan_sha256);
    auto* prebound_backend = dynamic_cast<IPreboundBackend*>(backend);
    if (memory.staged) {
        auto* file_backed_backend = dynamic_cast<IFileBackedBackend*>(backend);
        const bool segmented_bundle = sections.count("denoiser_entry_plan") != 0;
        const bool serial_execution_context =
            minimax_h3::uses_serial_execution_context(name, segmented_bundle);
        return load_staged_module(name, section, bundle_path, plan_sha256, file_backed_backend,
                                  options, external_bindings, memory, hot,
                                  serial_execution_context);
    }
    return load_in_memory_module(section, bundle_path, backend, prebound_backend, options,
                                 external_bindings);
}

class RuntimeCacheLeaseState final {
  public:
    RuntimeCacheLeaseState(IRuntimeCacheBackend& backend, const std::string& path)
        : backend_(&backend), lease_(backend.acquire_runtime_cache_lease(path.c_str())) {}

    ~RuntimeCacheLeaseState() {
        if (lease_ == 0)
            return;
        try {
            finalize();
        } catch (const std::exception& error) {
            std::cerr << "[trtmc] Failed to persist RTX runtime cache during lease cleanup: "
                      << error.what() << '\n';
        } catch (...) {
            std::cerr << "[trtmc] Failed to persist RTX runtime cache during lease cleanup\n";
        }
    }

    void require_active() const {
        if (lease_ == 0)
            throw std::runtime_error("MiniMax-H3 runtime cache lease is already finalized");
    }

    void finalize() {
        if (lease_ == 0)
            return;
        // The backend intentionally leaves the lease active when persistence
        // fails. Clear our token only after a successful release so explicit
        // finalization and the destructor can retry the same lease.
        backend_->release_runtime_cache_lease(lease_);
        lease_ = 0;
    }

  private:
    IRuntimeCacheBackend* backend_;
    std::uint64_t lease_;
};

std::shared_ptr<RuntimeCacheLeaseState> make_runtime_cache_lease(const PipelineContext& ctx) {
    if (ctx.runtime_cache_path.empty())
        return {};
    auto* cache_backend = dynamic_cast<IRuntimeCacheBackend*>(ctx.backend);
    if (cache_backend == nullptr) {
        throw std::runtime_error(
            "MiniMax-H3 runtime cache requires a backend with explicit persistence support");
    }
    return std::make_shared<RuntimeCacheLeaseState>(*cache_backend, ctx.runtime_cache_path);
}

MiniMaxH3ModuleLoader make_module_loader(const PipelineContext& ctx, SectionMap sections,
                                         RuntimeMemoryConfig memory, HotEngineConfig hot,
                                         std::shared_ptr<RuntimeCacheLeaseState> cache_lease) {
    if (hot.retain_engines && !memory.staged) {
        throw std::runtime_error(
            "MiniMax-H3 retained engines require a staged TensorRT-RTX bundle");
    }
    const std::string bundle_path = ctx.bundle_path;
    const std::string runtime_cache = ctx.runtime_cache_path;
    IBackend* const backend = ctx.backend;
    const bool cuda_graphs = ctx.cuda_graphs;
    const bool verify_plan_sha256 = ctx.validate_bundle_payloads;
    PlanSha256Map plan_sha256;
    // Segmented plans are always indexed, even if a forged config tries to
    // suppress the staged-runtime flag. Full digests can be deferred until
    // TensorRT opens each section so large video bundles start promptly.
    if (memory.staged || sections.count("denoiser_entry_plan") != 0) {
        plan_sha256 = load_plan_sha256(ctx.config_json, sections);
        validate_plan_section_bounds(ctx.bundle_path, sections);
        if (ctx.validate_bundle_payloads)
            validate_plan_section_attestations(ctx.bundle_path, sections, plan_sha256);
    }
    return [sections = std::move(sections), bundle_path, runtime_cache, backend, cuda_graphs,
            memory, hot, verify_plan_sha256, plan_sha256 = std::move(plan_sha256),
            cache_lease = std::move(cache_lease)](
            const std::string& name, cudaStream_t stream,
            const std::vector<ModuleExternalBinding>& external_bindings) {
        if (cache_lease)
            cache_lease->require_active();
        return load_module(name, stream, external_bindings, sections, bundle_path, runtime_cache,
                           backend, cuda_graphs, memory, plan_sha256, hot, verify_plan_sha256);
    };
}

} // namespace

class MiniMaxH3Plugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        const CacheConfig cache = load_cache_config(ctx);
        const MiniMaxH3DenoiserConfig denoiser = load_denoiser_config(ctx, cache);
        const MiniMaxH3Ref2VAConfig ref2va = load_ref2va_config(ctx);
        validate_profile(ctx, denoiser);
        auto sections =
            index_sections(ctx.bundle.info, cache.enabled, denoiser.native_vsa, ref2va.enabled);
        validate_fl2va_conditioning_contract(ctx, sections);
        auto runtime_cache_lease = make_runtime_cache_lease(ctx);
        auto loader = make_module_loader(ctx, std::move(sections), load_runtime_memory_config(ctx),
                                         load_hot_engine_config(ctx), runtime_cache_lease);
        std::function<void()> runtime_cache_finalizer;
        if (runtime_cache_lease) {
            runtime_cache_finalizer = [runtime_cache_lease = std::move(runtime_cache_lease)] {
                runtime_cache_lease->finalize();
            };
        }
        return std::make_unique<MiniMaxH3Pipeline>(
            std::move(loader), load_tokenizer(ctx.bundle), ctx.bundle.info.model_id, cache.enabled,
            cache.threshold, denoiser, ref2va, std::move(runtime_cache_finalizer));
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_minimax_h3_plugin, MiniMaxH3Plugin,
                                       "diffusion_minimax_h3");

} // namespace trtmc
