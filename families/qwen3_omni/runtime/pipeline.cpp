/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_omni/runtime/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace trtmc {
namespace {

constexpr const char* kSystemPrompt =
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
    "perceiving auditory and visual inputs, as well as generating text and speech.";

std::string shape_text(const std::vector<std::int64_t>& shape) {
    std::string result{"["};
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0)
            result += ',';
        result += std::to_string(shape[index]);
    }
    result += ']';
    return result;
}

const char* dtype_text(DType dtype) {
    switch (dtype) {
    case DType::kFloat32:
        return "float32";
    case DType::kFloat16:
        return "float16";
    case DType::kBFloat16:
        return "bfloat16";
    case DType::kInt32:
        return "int32";
    case DType::kInt8:
        return "int8";
    }
    return "unknown";
}

std::string thinker_prompt(const std::string& prompt) {
    return std::string("<|im_start|>system\n") + kSystemPrompt + "<|im_end|>\n<|im_start|>user\n" +
           prompt + "<|im_end|>\n<|im_start|>assistant\n";
}

std::vector<float> copy_logits(const TensorMap& outputs, std::int32_t vocab_size) {
    const auto found = outputs.find("logits");
    if (found == outputs.end())
        throw std::runtime_error("Qwen3-Omni Thinker has no 'logits' output");
    const Tensor& tensor = found->second;
    if (tensor.dtype != DType::kFloat32 || tensor.data == nullptr ||
        tensor.shape != std::vector<std::int64_t>{1, vocab_size}) {
        throw std::runtime_error("Qwen3-Omni Thinker logits contract is invalid");
    }
    std::vector<float> result(tensor.numel());
    std::memcpy(result.data(), tensor.data, tensor.nbytes());
    if (!std::all_of(result.begin(), result.end(),
                     [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("Qwen3-Omni Thinker logits are not finite");
    }
    return result;
}

void collect_prefill_kv(ITrtModule& module, const TensorMap& outputs, std::int32_t layers,
                        std::int32_t sequence, std::vector<const void*>& keys,
                        std::vector<const void*>& values) {
    keys.reserve(static_cast<std::size_t>(layers));
    values.reserve(static_cast<std::size_t>(layers));
    for (std::int32_t layer = 0; layer < layers; ++layer) {
        const std::string suffix = "_" + std::to_string(layer);
        const std::string cache_k_name = "cache_k" + suffix;
        const std::string cache_v_name = "cache_v" + suffix;
        const std::string present_k_name = "present_k" + suffix;
        const std::string present_v_name = "present_v" + suffix;
        const auto cache_k_shape = module.tensor_shape(cache_k_name);
        const auto cache_v_shape = module.tensor_shape(cache_v_name);
        const auto cache_k_dtype = module.tensor_dtype(cache_k_name);
        const auto cache_v_dtype = module.tensor_dtype(cache_v_name);
        const auto present_k = outputs.find(present_k_name);
        const auto present_v = outputs.find(present_v_name);
        if (present_k == outputs.end() || present_v == outputs.end()) {
            throw std::runtime_error("Qwen3-Omni Thinker prefill is missing KV output for layer " +
                                     std::to_string(layer));
        }
        const bool valid =
            cache_k_shape.size() == 2 && cache_k_shape[1] > 0 && cache_v_shape == cache_k_shape &&
            present_k->second.shape == std::vector<std::int64_t>{sequence, cache_k_shape[1]} &&
            present_v->second.shape == present_k->second.shape && cache_v_dtype == cache_k_dtype &&
            present_k->second.dtype == cache_k_dtype && present_v->second.dtype == cache_k_dtype;
        if (!valid) {
            throw std::runtime_error(
                "Qwen3-Omni Thinker prefill KV contract mismatch at layer " +
                std::to_string(layer) + ": sequence=" + std::to_string(sequence) +
                ", cache_k=" + shape_text(cache_k_shape) + "/" + dtype_text(cache_k_dtype) +
                ", cache_v=" + shape_text(cache_v_shape) + "/" + dtype_text(cache_v_dtype) +
                ", runtime_present_k=" + shape_text(present_k->second.shape) + "/" +
                dtype_text(present_k->second.dtype) + ", runtime_present_v=" +
                shape_text(present_v->second.shape) + "/" + dtype_text(present_v->second.dtype));
        }
        const void* key = module.device_ptr(present_k_name);
        const void* value = module.device_ptr(present_v_name);
        if (key == nullptr || value == nullptr) {
            throw std::runtime_error("Qwen3-Omni Thinker prefill KV output is null for layer " +
                                     std::to_string(layer));
        }
        keys.push_back(key);
        values.push_back(value);
    }
}

std::int32_t argmax(const std::vector<float>& logits) {
    if (logits.empty())
        throw std::runtime_error("Qwen3-Omni cannot select from empty logits");
    return static_cast<std::int32_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

std::string clean_assistant_text(std::string text) {
    for (const std::string marker : {"<|im_end|>", "<|endoftext|>"}) {
        const auto position = text.find(marker);
        if (position != std::string::npos)
            text.erase(position);
    }
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return {};
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

} // namespace

Qwen3OmniTextPipeline::Qwen3OmniTextPipeline(std::unique_ptr<ITrtModule> thinker_prefill,
                                             std::unique_ptr<ITrtModule> thinker_decode,
                                             std::unique_ptr<Qwen3OmniKvCache> thinker_state,
                                             Qwen3OmniRuntimeConfig config,
                                             std::shared_ptr<ITokenizer> tokenizer)
    : thinker_prefill_(std::move(thinker_prefill)), thinker_decode_(std::move(thinker_decode)),
      thinker_state_(std::move(thinker_state)), config_(std::move(config)),
      tokenizer_(std::move(tokenizer)) {
    if (!thinker_prefill_ || !thinker_decode_ || !thinker_state_ || !tokenizer_)
        throw std::invalid_argument("Qwen3-Omni text pipeline is missing a required component");
    if (!thinker_prefill_->ok() || !thinker_decode_->ok() || !thinker_state_->ok())
        throw std::invalid_argument("Qwen3-Omni text pipeline has an invalid component");
}

std::vector<float>
Qwen3OmniTextPipeline::run_token_prefill(const std::vector<std::int32_t>& token_ids) {
    if (token_ids.empty() ||
        token_ids.size() > static_cast<std::size_t>(thinker_state_->max_length())) {
        throw std::runtime_error("Qwen3-Omni Thinker prompt exceeds its prefill profile");
    }
    thinker_state_->reset();
    thinker_state_->bind_to(*thinker_decode_);
    thinker_state_->bind_cache_inputs(*thinker_prefill_);
    const auto sequence = static_cast<std::int32_t>(token_ids.size());
    TensorMap inputs;
    inputs["token_id"] =
        Tensor{const_cast<std::int32_t*>(token_ids.data()), {sequence}, DType::kInt32};
    thinker_state_->prepare_step(inputs, sequence);
    const TensorMap outputs = thinker_prefill_->forward(inputs);
    std::vector<const void*> keys;
    std::vector<const void*> values;
    collect_prefill_kv(*thinker_prefill_, outputs, thinker_state_->num_layers(), sequence, keys,
                       values);
    thinker_state_->write_prefill_kv(keys, values, sequence);
    thinker_state_->bind_to(*thinker_decode_);
    return copy_logits(outputs, config_.thinker_vocab_size);
}

std::vector<float> Qwen3OmniTextPipeline::run_token_step(std::int32_t token_id) {
    if (thinker_state_->position() >= thinker_state_->max_length())
        throw std::runtime_error("Qwen3-Omni Thinker exhausted its KV cache");
    TensorMap inputs;
    inputs["token_id"] = Tensor{&token_id, {1}, DType::kInt32};
    thinker_state_->prepare_step(inputs);
    const TensorMap outputs = thinker_decode_->forward(inputs);
    thinker_state_->advance();
    return copy_logits(outputs, config_.thinker_vocab_size);
}

std::vector<std::int32_t> Qwen3OmniTextPipeline::run_thinker(const std::string& prompt,
                                                             std::int32_t max_new_tokens) {
    const auto prompt_ids = tokenizer_->encode(thinker_prompt(prompt));
    std::vector<float> logits = run_token_prefill(prompt_ids);
    const std::int32_t endoftext_token = tokenizer_->id_for_token("<|endoftext|>");
    if (endoftext_token < 0)
        throw std::runtime_error("Qwen3-Omni tokenizer has no <|endoftext|> token");
    std::vector<std::int32_t> generated;
    generated.reserve(static_cast<std::size_t>(max_new_tokens));
    for (std::int32_t step = 0; step < max_new_tokens; ++step) {
        const std::int32_t token = argmax(logits);
        if (token == config_.thinker_eos_token_id || token == endoftext_token)
            break;
        generated.push_back(token);
        if (step + 1 < max_new_tokens)
            logits = run_token_step(token);
    }
    return generated;
}

TextResult Qwen3OmniTextPipeline::generate(const std::string& prompt,
                                           const TextGenerationConfig& config) {
    if (config.max_new_tokens <= 0 || config.temperature != 1.0F || config.top_k != 1 ||
        config.top_p != 1.0F || config.min_p != 0.0F || config.repetition_penalty != 1.0F ||
        config.use_chat_template || !config.enable_thinking || !config.lora_adapter_id.empty()) {
        throw std::invalid_argument(
            "Qwen3-Omni text generation supports only its fixed greedy decoding contract");
    }
    std::vector<std::int32_t> generated = run_thinker(prompt, config.max_new_tokens);
    if (generated.empty())
        throw std::runtime_error("Qwen3-Omni Thinker produced no text");
    std::string text = clean_assistant_text(tokenizer_->decode(generated));
    if (text.empty())
        throw std::runtime_error("Qwen3-Omni Thinker decoded to empty text");
    return TextResult{std::move(text), std::move(generated)};
}

} // namespace trtmc
