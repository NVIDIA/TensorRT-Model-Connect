/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/deepseek_ocr/runtime/cuda_stream.h"
#include "families/deepseek_ocr/runtime/pipeline.h"
#include "families/deepseek_ocr/runtime/plugin_helpers.h"
#include "families/deepseek_ocr/runtime/runtime_config.h"
#include "families/deepseek_ocr/runtime/tensor_names.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstdlib>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::deepseek_ocr_factory {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::string section_text(const BundleReader& bundle, const char* name) {
    const auto& data = require_section(bundle, name);
    return std::string(data.begin(), data.end());
}

void require_nccl(std::int32_t tensor_parallel_size) {
    if (tensor_parallel_size <= 1)
        return;
    static void* const handle = dlopen("libnccl.so.2", RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        const char* error = dlerror();
        throw std::runtime_error("tensor-parallel runtime requires NCCL: " +
                                 std::string(error == nullptr ? "unknown loader error" : error));
    }
}

std::int32_t require_rank(std::int32_t tp_size) {
    require_nccl(tp_size);
    if (tp_size == 1)
        return 0;
    const char* text = std::getenv("OMPI_COMM_WORLD_RANK");
    if (text == nullptr || *text == '\0')
        throw std::runtime_error("DeepSeek-OCR TP runtime requires OMPI_COMM_WORLD_RANK");
    char* end = nullptr;
    const long rank = std::strtol(text, &end, 10);
    if (*end != '\0' || rank < 0 || rank >= tp_size)
        throw std::runtime_error("DeepSeek-OCR RANK is outside tensor_parallel_size");
    return static_cast<std::int32_t>(rank);
}

DeepseekOcrKvCacheNames build_kv_names(const DeepseekOcrRuntimeConfig& config) {
    DeepseekOcrKvCacheNames names;
    for (std::int32_t layer = 0; layer < config.model.num_layers; ++layer) {
        names.cache_k.push_back(deepseek_ocr_expand_layer_name(config.cache_k_pattern, layer));
        names.cache_v.push_back(deepseek_ocr_expand_layer_name(config.cache_v_pattern, layer));
        names.present_k.push_back(
            deepseek_ocr_expand_layer_name(config.model.present_k_pattern, layer));
        names.present_v.push_back(
            deepseek_ocr_expand_layer_name(config.model.present_v_pattern, layer));
    }
    return names;
}

} // namespace
} // namespace trtmc::deepseek_ocr_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    const std::string runtime_text =
        deepseek_ocr_factory::section_text(context.reader, "runtime.json");
    auto config = deepseek_ocr_parse_runtime_config(runtime_text);

    const auto tp_size = config.tensor_parallel_size;
    const auto rank = deepseek_ocr_factory::require_rank(tp_size);
    const std::string decode_section =
        tp_size == 1 ? "engine.plan" : "engine.rank" + std::to_string(rank) + ".plan";
    const auto& decode_plan =
        deepseek_ocr_factory::require_section(context.reader, decode_section.c_str());
    const auto& prefill_plan =
        deepseek_ocr_factory::require_section(context.reader, "prefill.plan");
    const auto& vision_plan = deepseek_ocr_factory::require_section(context.reader, "vision.plan");

    auto stream_owner = std::make_shared<DeepseekOcrCudaStream>();
    if (!stream_owner->ok())
        throw std::runtime_error("DeepSeek-OCR failed to create its CUDA stream");
    ModuleCreateOptions options{};
    options.stream = stream_owner->get();
    auto decode =
        load_trt_module_from_plan(&context.backend, &decode_plan, decode_section.c_str(), options);
    auto prefill =
        load_trt_module_from_plan(&context.backend, &prefill_plan, "prefill.plan", options);
    auto vision = load_trt_module_from_plan(&context.backend, &vision_plan, "vision.plan", options);
    decode.module->keep_alive(stream_owner);
    prefill.module->keep_alive(stream_owner);
    vision.module->keep_alive(stream_owner);

    const auto num_layers = config.model.num_layers;
    const auto max_cache_length = config.max_cache_length;
    auto kv_names = deepseek_ocr_factory::build_kv_names(config);
    const auto cache_k_name = kv_names.cache_k.front();
    const auto cache_shape = decode.module->tensor_shape(cache_k_name);
    if (cache_shape.size() < 2 || cache_shape[1] <= 0)
        throw std::runtime_error("DeepSeek-OCR decode engine has invalid KV-cache shape");
    const auto kv_dim = static_cast<std::int32_t>(cache_shape[1]);
    const auto cache_dtype = decode.module->tensor_dtype(cache_k_name);
    std::unique_ptr<DeepseekOcrInferenceState> state = std::make_unique<DeepseekOcrKvCache>(
        num_layers, max_cache_length, kv_dim, decode.module->stream(), cache_dtype,
        std::move(kv_names));

    auto tokenizer = create_tokenizer_from_bundle(context.reader);
    if (!tokenizer)
        throw std::runtime_error("DeepSeek-OCR bundle does not contain its required tokenizer");

    auto model = std::move(config.model);
    model.has_position_input = decode.module->has_input("position_id");
    auto preprocess = deepseek_ocr_parse_preprocess_config(runtime_text);
    return new DeepseekOcrPipeline(
        std::move(decode.module), std::move(vision.module), std::move(state), std::move(model),
        std::move(preprocess), options.stream, std::move(tokenizer), "", std::move(prefill.module));
}
