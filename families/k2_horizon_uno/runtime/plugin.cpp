/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/k2_horizon_uno/runtime/chat_template.h"
#include "families/k2_horizon_uno/runtime/kv_cache.h"
#include "families/k2_horizon_uno/runtime/pipeline.h"
#include "families/k2_horizon_uno/runtime/runtime_config.h"
#include "families/k2_horizon_uno/runtime/tokenizer.h"
#include "trtmc/bundle.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trtmc::k2_horizon_uno {
namespace {

constexpr std::int32_t kMaximumBlockLength = 8;
constexpr std::int32_t kVocabSize = 250624;
constexpr std::int32_t kNumLayers = 36;
constexpr std::int32_t kMaxPositionEmbeddings = 524288;

std::vector<char> require_section(const BundleReader& bundle, std::string_view name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("K2-Horizon-Uno bundle section is missing or empty: " +
                                 std::string(name));
    return bundle.read_section(name);
}

std::string require_text_section(const BundleReader& bundle, std::string_view name) {
    const auto data = require_section(bundle, name);
    return {data.begin(), data.end()};
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan) {
    const auto start = std::chrono::steady_clock::now();
    auto module = backend.create_module(plan.data(), plan.size(), {});
    const auto elapsed =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::ostringstream receipt;
    receipt << std::fixed << std::setprecision(6)
            << "[trtmc.load_timing] label=\"k2_horizon_uno.engine.plan\" load_deserialize_ms="
            << elapsed << " plan_bytes=" << plan.size() << '\n';
    std::cerr << receipt.str();
    if (module == nullptr || !module->ok())
        throw std::runtime_error("K2-Horizon-Uno failed to load engine.plan");
    module->set_timing_label("k2_horizon_uno.engine.plan");
    return module;
}

std::shared_ptr<ITokenizer> create_tokenizer(const BundleReader& bundle) {
    const auto data = require_section(bundle, "tokenizer.json");
    auto tokenizer = CreateK2HorizonUnoBpeTokenizer(data.data(), data.size(), true);
    if (tokenizer == nullptr)
        throw std::runtime_error("K2-Horizon-Uno native BPE tokenizer construction failed");
    return std::shared_ptr<ITokenizer>(std::move(tokenizer));
}

std::int32_t parse_max_cache_length(std::string_view text) {
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("K2-Horizon-Uno invalid runtime.json: " +
                                 std::string(error.what()));
    }
    if (!json.is_object() || json.size() != 1)
        throw std::runtime_error("K2-Horizon-Uno runtime.json must contain only max_cache_length");
    std::int64_t max_cache_length;
    try {
        const auto& value = json.at("max_cache_length");
        if (!value.is_number_integer())
            throw std::runtime_error("K2-Horizon-Uno runtime.json has invalid max_cache_length");
        max_cache_length = value.get<std::int64_t>();
    } catch (const nlohmann::json::exception&) {
        throw std::runtime_error("K2-Horizon-Uno runtime.json has invalid max_cache_length");
    }
    if (max_cache_length < kMaximumBlockLength || max_cache_length > kMaxPositionEmbeddings)
        throw std::runtime_error("K2-Horizon-Uno runtime.json has invalid max_cache_length");
    return static_cast<std::int32_t>(max_cache_length);
}

K2HorizonUnoKvCacheNames make_cache_names() {
    K2HorizonUnoKvCacheNames names;
    for (std::int32_t layer = 0; layer < kNumLayers; ++layer) {
        const auto suffix = std::to_string(layer);
        names.cache_k.push_back("cache_k_" + suffix);
        names.cache_v.push_back("cache_v_" + suffix);
        names.present_k.push_back("present_k_" + suffix);
        names.present_v.push_back("present_v_" + suffix);
    }
    return names;
}

void require_dynamic_vector_input(const ITrtModule& module, const std::string& name, DType dtype) {
    if (!module.has_input(name) || module.tensor_dtype(name) != dtype ||
        !module.input_is_dynamic(name) ||
        module.input_profile_shape(name, 0, ProfileShapeSelector::kMin) !=
            std::vector<std::int64_t>{1} ||
        module.input_profile_shape(name, 0, ProfileShapeSelector::kMax) !=
            std::vector<std::int64_t>{kMaximumBlockLength}) {
        throw std::runtime_error("K2-Horizon-Uno dynamic input contract mismatch for '" + name +
                                 "'");
    }
    const auto optimum = module.input_profile_shape(name, 0, ProfileShapeSelector::kOpt);
    if (optimum.size() != 1 || optimum.front() < 1 || optimum.front() > kMaximumBlockLength) {
        throw std::runtime_error("K2-Horizon-Uno input profile optimum is invalid for '" + name +
                                 "'");
    }
}

std::set<std::string> tensor_names(const std::vector<TensorInfo>& tensors) {
    std::set<std::string> result;
    for (const auto& tensor : tensors)
        result.insert(tensor.name);
    return result;
}

void validate_engine(const ITrtModule& module, const K2HorizonUnoKvCacheNames& cache_names) {
    if (module.optimization_profile_count() != 1)
        throw std::runtime_error("K2-Horizon-Uno requires exactly one optimization profile");
    require_dynamic_vector_input(module, "token_id", DType::kInt32);
    require_dynamic_vector_input(module, "position_id", DType::kInt32);
    require_dynamic_vector_input(module, "lora_mask", DType::kFloat32);
    if (!module.has_output("logits") || module.tensor_dtype("logits") != DType::kFloat32 ||
        module.tensor_shape("logits") !=
            std::vector<std::int64_t>{kMaximumBlockLength, kVocabSize}) {
        throw std::runtime_error("K2-Horizon-Uno logits must be dynamic float32 [S,vocab_size]");
    }

    std::set<std::string> expected_inputs{"token_id", "position_id", "cache_write_indices",
                                          "key_value_lengths", "lora_mask"};
    expected_inputs.insert(cache_names.cache_k.begin(), cache_names.cache_k.end());
    expected_inputs.insert(cache_names.cache_v.begin(), cache_names.cache_v.end());
    std::set<std::string> expected_outputs{"logits"};
    expected_outputs.insert(cache_names.present_k.begin(), cache_names.present_k.end());
    expected_outputs.insert(cache_names.present_v.begin(), cache_names.present_v.end());
    if (tensor_names(module.input_info()) != expected_inputs ||
        tensor_names(module.output_info()) != expected_outputs) {
        throw std::runtime_error("K2-Horizon-Uno engine I/O inventory is not exact");
    }
}

void validate_chat_tokenizer(const ITokenizer& tokenizer) {
    constexpr std::int32_t im_start = 250018;
    constexpr std::int32_t im_end = 250019;
    constexpr std::int32_t think = 250029;
    if (tokenizer.id_for_token("<|ifm|begin_of_text|>") != 0 ||
        tokenizer.id_for_token("<|ifm|im_start|>") != im_start ||
        tokenizer.id_for_token("<|ifm|im_end|>") != im_end ||
        tokenizer.id_for_token("<ifm|think>") != think) {
        throw std::runtime_error("K2-Horizon-Uno chat protocol token IDs are inconsistent");
    }
    const auto probe = tokenizer.encode(
        k2_horizon_uno_apply_chat_template(kK2HorizonUnoPublisherChatTemplateFormat, "", "high"));
    const std::vector<std::int32_t> expected{0,        im_start, 2672, 200,   im_end,
                                             im_start, 142036,   200,  think, 200};
    if (probe != expected)
        throw std::runtime_error("K2-Horizon-Uno tokenizer chat framing is inconsistent");

    const std::vector<std::pair<std::string, std::vector<std::int32_t>>> ascii_probes{
        {"IT'S", {0, 1938, 17456}},
        {"'z", {0, 180302}},
        {"\t.A", {0, 199, 6197}},
    };
    for (const auto& [text, token_ids] : ascii_probes) {
        if (tokenizer.encode(text) != token_ids) {
            throw std::runtime_error(
                "K2-Horizon-Uno tokenizer ASCII pre-tokenization is inconsistent");
        }
    }
}

} // namespace

void validate_runtime_config_json(std::string_view json) {
    (void)parse_max_cache_length(json);
}

ITask* create(const FamilyContext& context) {
    const char* backend_name = context.backend.name();
    if (backend_name == nullptr || std::string(backend_name) != "trt")
        throw std::runtime_error("K2-Horizon-Uno requires the TensorRT backend");

    const auto max_cache_length =
        parse_max_cache_length(require_text_section(context.reader, "runtime.json"));
    (void)require_section(context.reader, "chat_template.jinja");
    auto tokenizer = create_tokenizer(context.reader);
    validate_chat_tokenizer(*tokenizer);

    auto decoder = load_engine(context.backend, require_section(context.reader, "engine.plan"));
    auto cache_names = make_cache_names();
    validate_engine(*decoder, cache_names);
    auto cache = std::make_unique<K2HorizonUnoKvCache>(max_cache_length, decoder->stream(),
                                                       std::move(cache_names));
    if (!cache->ok())
        throw std::runtime_error("K2-Horizon-Uno KV allocation failed");
    cache->bind_to(*decoder);

    return new K2HorizonUnoTextGenerationPipeline(std::move(decoder), std::move(cache),
                                                  std::move(tokenizer));
}

} // namespace trtmc::k2_horizon_uno

TRTMC_DEFINE_FAMILY_PLUGIN_V1("k2_horizon_uno")

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("k2_horizon_uno does not support --kv-cache-size");
    return trtmc::k2_horizon_uno::create(context);
}
