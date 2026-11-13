/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/layoutlm/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::layoutlm {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("layoutlm bundle section is missing or empty: " +
                                 std::string(name));
    return bundle.read_section(name);
}

struct BundleConfig {
    RuntimeConfig runtime;
    std::int32_t cls_token_id{0};
    std::int32_t sep_token_id{2};
    std::int32_t vocab_size{0};
};

BundleConfig parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    BundleConfig config;
    config.runtime.max_sequence_length = json.at("max_sequence_length").get<std::int32_t>();
    config.runtime.doc_stride = json.at("doc_stride").get<std::int32_t>();
    config.runtime.max_answer_length = json.at("max_answer_length").get<std::int32_t>();
    config.runtime.max_2d_position = json.at("max_2d_position").get<std::int32_t>();
    config.runtime.ocr_language = json.at("ocr_language").get<std::string>();
    config.runtime.ocr_page_segmentation_mode =
        json.at("ocr_page_segmentation_mode").get<std::int32_t>();
    config.cls_token_id = json.at("cls_token_id").get<std::int32_t>();
    config.sep_token_id = json.at("sep_token_id").get<std::int32_t>();
    config.vocab_size = json.at("vocab_size").get<std::int32_t>();
    const auto& runtime = config.runtime;
    if (runtime.max_sequence_length < 8 || runtime.doc_stride < 0 ||
        runtime.doc_stride >= runtime.max_sequence_length || runtime.max_answer_length < 1 ||
        runtime.max_2d_position < 1000 || runtime.ocr_language.empty() ||
        runtime.ocr_page_segmentation_mode < 0 || config.vocab_size <= 0 ||
        config.cls_token_id < 0 || config.sep_token_id < 0)
        throw std::runtime_error("layoutlm runtime.json does not match its runtime contract");
    return config;
}

std::string require_task(const BundleReader& bundle) {
    const auto& task = bundle.info().task;
    if (task != internal::IImagesTextToText::kTask)
        throw std::runtime_error("layoutlm bundle declares an unsupported task: " + task);
    return task;
}

void require_engine_io(const ITrtModule& encoder) {
    for (const auto* name : {"input_ids", "bbox"}) {
        if (!encoder.has_input(name) || encoder.tensor_dtype(name) != DType::kInt32)
            throw std::runtime_error(std::string("layoutlm engine lacks the int32 input ") + name);
    }
    for (const auto* name : {"start_logits", "end_logits"}) {
        if (!encoder.has_output(name) || encoder.tensor_dtype(name) != DType::kFloat32)
            throw std::runtime_error(std::string("layoutlm engine lacks the float32 output ") +
                                     name);
    }
}

} // namespace
} // namespace trtmc::layoutlm

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc::layoutlm;
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("layoutlm does not support --kv-cache-size");
    const auto config = parse_config(require_section(context.reader, "runtime.json"));
    const auto tokenizer_json = require_section(context.reader, "tokenizer.json");
    const auto plan = require_section(context.reader, "model.plan");
    auto primary_task = require_task(context.reader);

    auto tokenizer =
        WordTokenizer::create(std::string_view(tokenizer_json.data(), tokenizer_json.size()));
    if (tokenizer->cls_id() != config.cls_token_id || tokenizer->sep_id() != config.sep_token_id ||
        tokenizer->vocab_size() > static_cast<std::size_t>(config.vocab_size))
        throw std::runtime_error("layoutlm tokenizer.json does not match the engine's vocabulary");
    trtmc::ModuleCreateOptions options{};
    auto encoder = context.backend.create_module(plan.data(), plan.size(), options);
    if (!encoder || !encoder->ok())
        throw std::runtime_error("layoutlm engine failed to load");
    require_engine_io(*encoder);
    return new DocumentQaPipeline(std::move(encoder), std::move(tokenizer), config.runtime,
                                  std::move(primary_task));
}
