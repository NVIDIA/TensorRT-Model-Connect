/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/tipsv2/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::tipsv2 {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0) {
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    }
    return bundle.read_section(name);
}

// `get<std::int32_t>()` truncates 384.5 and wraps a large integer, so the JSON type and the range
// are checked before the conversion.
std::int32_t positive_int(const nlohmann::json& json, const char* name) {
    const auto& value = json.at(name);
    if (!value.is_number_unsigned()) {
        throw std::runtime_error(std::string("tipsv2 runtime.json ") + name +
                                 " must be a non-negative integer");
    }
    const auto raw = value.get<std::uint64_t>();
    if (raw == 0 || raw > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error(std::string("tipsv2 runtime.json ") + name + " is out of range");
    }
    return static_cast<std::int32_t>(raw);
}

std::int32_t non_negative_int(const nlohmann::json& json, const char* name) {
    const auto& value = json.at(name);
    if (!value.is_number_unsigned()) {
        throw std::runtime_error(std::string("tipsv2 runtime.json ") + name +
                                 " must be a non-negative integer");
    }
    const auto raw = value.get<std::uint64_t>();
    if (raw > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error(std::string("tipsv2 runtime.json ") + name + " is out of range");
    }
    return static_cast<std::int32_t>(raw);
}

float positive_float(const nlohmann::json& json, const char* name) {
    const auto& value = json.at(name);
    if (!value.is_number() || value.is_boolean()) {
        throw std::runtime_error(std::string("tipsv2 runtime.json ") + name + " must be a number");
    }
    const auto raw = value.get<double>();
    if (!std::isfinite(raw) || raw <= 0.0 ||
        raw > static_cast<double>(std::numeric_limits<float>::max())) {
        throw std::runtime_error(std::string("tipsv2 runtime.json ") + name +
                                 " must be positive and finite");
    }
    return static_cast<float>(raw);
}

// nlohmann keeps the last of two identical keys, so a `runtime.json` making two contradictory
// claims would be accepted on whichever came second. This refuses both, as the builder does.
nlohmann::json parse_without_duplicates(const std::vector<char>& data) {
    std::vector<std::set<std::string>> keys_by_depth;
    const auto callback = [&keys_by_depth](int depth, nlohmann::json::parse_event_t event,
                                           nlohmann::json& parsed) {
        const auto level = static_cast<std::size_t>(depth);
        if (event == nlohmann::json::parse_event_t::object_start) {
            if (keys_by_depth.size() <= level) {
                keys_by_depth.resize(level + 1);
            }
            keys_by_depth[level].clear();
        } else if (event == nlohmann::json::parse_event_t::key) {
            // The key event arrives one level inside its object.
            const std::size_t owner = level == 0 ? 0 : level - 1;
            if (keys_by_depth.size() <= owner) {
                keys_by_depth.resize(owner + 1);
            }
            if (!keys_by_depth[owner].insert(parsed.get<std::string>()).second) {
                throw std::runtime_error("tipsv2 runtime.json states " + parsed.get<std::string>() +
                                         " more than once");
            }
        }
        return true;
    };
    return nlohmann::json::parse(data.begin(), data.end(), callback);
}

RuntimeConfig parse_config(const std::vector<char>& data) {
    const auto json = parse_without_duplicates(data);
    RuntimeConfig config;
    config.preprocess.input_image_h = positive_int(json, "input_image_h");
    config.preprocess.input_image_w = positive_int(json, "input_image_w");
    config.preprocess.rescale_factor = positive_float(json, "rescale_factor");
    config.resample = positive_int(json, "resample");
    config.max_text_len = positive_int(json, "max_text_len");
    config.embedding_dim = positive_int(json, "embedding_dim");
    config.pad_token_id = non_negative_int(json, "pad_token_id");
    config.vocab_size = positive_int(json, "vocab_size");
    config.temperature = positive_float(json, "temperature");
    if (config.preprocess.input_image_h != config.preprocess.input_image_w) {
        // The position table has one row per patch of a square grid.
        throw std::runtime_error("tipsv2 runtime.json must describe a square input");
    }
    // The same bounds the builder checks, in the type that is stored here: being a positive,
    // finite float is not enough when the value is divided by, or multiplied into a sample. A
    // bundle that builds must load, and one that does not must say why here rather than divide
    // by zero later.
    const float brightest = 255.0F * config.preprocess.rescale_factor;
    if (!std::isfinite(brightest) || brightest <= 0.0F || brightest > 1.0F) {
        throw std::runtime_error(
            "tipsv2 runtime.json rescale_factor does not send an 8-bit sample into (0, 1]");
    }
    if (!std::isfinite(1.0F / config.temperature)) {
        throw std::runtime_error("tipsv2 runtime.json temperature overflows when inverted");
    }
    if (config.pad_token_id >= config.vocab_size) {
        throw std::runtime_error("tipsv2 runtime.json pads with an id the embedding table has "
                                 "no row for");
    }
    return config;
}

// What each engine must declare, checked at load rather than at the first call: a bundle whose
// engine does not match `runtime.json` is a build mistake, and finding it here names the
// mismatch instead of failing somewhere inside the first inference.
struct BindingContract {
    const char* name;
    DType dtype;
    std::vector<std::int64_t> shape;
};

void require_bindings(const ITrtModule& engine, const char* what,
                      const std::vector<BindingContract>& inputs,
                      const std::vector<BindingContract>& outputs) {
    const auto check = [what](const std::vector<TensorInfo>& actual,
                              const std::vector<BindingContract>& expected, const char* side) {
        if (actual.size() != expected.size()) {
            throw std::runtime_error(std::string("tipsv2 ") + what + " engine declares " +
                                     std::to_string(actual.size()) + " " + side + "s, expected " +
                                     std::to_string(expected.size()));
        }
        for (const auto& contract : expected) {
            const auto found =
                std::find_if(actual.begin(), actual.end(), [&contract](const TensorInfo& info) {
                    return info.name == contract.name;
                });
            if (found == actual.end()) {
                throw std::runtime_error(std::string("tipsv2 ") + what + " engine has no " + side +
                                         " named " + contract.name);
            }
            if (found->dtype != contract.dtype) {
                throw std::runtime_error(std::string("tipsv2 ") + what + " engine " + side + " " +
                                         contract.name + " has the wrong dtype");
            }
            if (found->shape != contract.shape) {
                throw std::runtime_error(std::string("tipsv2 ") + what + " engine " + side + " " +
                                         contract.name + " has the wrong shape");
            }
        }
    };
    check(engine.input_info(), inputs, "input");
    check(engine.output_info(), outputs, "output");
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan,
                                        const char* what,
                                        const std::vector<BindingContract>& inputs,
                                        const std::vector<BindingContract>& outputs) {
    ModuleCreateOptions options{};
    auto engine = backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok()) {
        throw std::runtime_error(std::string("tipsv2 ") + what + " engine failed to load");
    }
    require_bindings(*engine, what, inputs, outputs);
    return engine;
}

std::string require_task(const BundleReader& bundle) {
    const auto& task = bundle.info().task;
    if (task != internal::IImageToEmbedding::kTask && task != internal::ITextToEmbedding::kTask &&
        task != internal::ITextImageToRelevance::kTask) {
        throw std::runtime_error("tipsv2 bundle declares an unsupported task: " + task);
    }
    return task;
}

} // namespace
} // namespace trtmc::tipsv2

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0) {
        throw std::invalid_argument("tipsv2 does not support --kv-cache-size");
    }
    const auto config_data = trtmc::tipsv2::require_section(context.reader, "runtime.json");
    const auto vision_plan = trtmc::tipsv2::require_section(context.reader, "vision.plan");
    const auto text_plan = trtmc::tipsv2::require_section(context.reader, "text.plan");
    const auto tokenizer_data = trtmc::tipsv2::require_section(context.reader, "tokenizer.spm");

    auto config = trtmc::tipsv2::parse_config(config_data);
    auto primary_task = trtmc::tipsv2::require_task(context.reader);
    auto tokenizer =
        std::make_shared<trtmc::tipsv2::Tokenizer>(trtmc::tipsv2::parse_tokenizer_asset(
            std::string_view(tokenizer_data.data(), tokenizer_data.size())));
    // Both engines take and return float32 whatever precision they were built at, and every
    // dimension is fixed, so the whole contract is checkable here.
    const std::int64_t size = config.preprocess.input_image_h;
    const std::int64_t width = config.embedding_dim;
    auto vision =
        trtmc::tipsv2::load_engine(context.backend, vision_plan, "vision",
                                   {{"pixel_values", trtmc::DType::kFloat32, {1, 3, size, size}}},
                                   {{"image_embeds", trtmc::DType::kFloat32, {width}}});
    auto text =
        trtmc::tipsv2::load_engine(context.backend, text_plan, "text",
                                   {{"input_ids", trtmc::DType::kInt32, {config.max_text_len}}},
                                   {{"text_embeds", trtmc::DType::kFloat32, {width}}});
    return new trtmc::tipsv2::Pipeline(std::move(vision), std::move(text), std::move(tokenizer),
                                       std::move(config), std::move(primary_task));
}
