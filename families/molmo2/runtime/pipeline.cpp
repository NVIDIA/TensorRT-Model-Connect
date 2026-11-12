/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/molmo2/runtime/preprocess.h"
#include "families/molmo2/runtime/tokenizer.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::molmo2 {
namespace {
void checked(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Molmo2 CUDA: ") + cudaGetErrorString(status));
}
class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t bytes) { checked(cudaMalloc(&ptr_, bytes)); }
    ~DeviceBuffer() {
        if (ptr_)
            cudaFree(ptr_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void* data() const { return ptr_; }

  private:
    void* ptr_{nullptr};
};
std::vector<char> section(const BundleReader& reader, const char* name) {
    const auto* entry = reader.find_section(name);
    if (!entry || !entry->length)
        throw std::runtime_error(std::string("Molmo2 bundle is missing ") + name);
    return reader.read_section(name);
}
std::unique_ptr<ITrtModule> module(const FamilyContext& context, const char* name) {
    const auto plan = section(context.reader, name);
    auto result = context.backend.create_module(plan.data(), plan.size(), {});
    if (!result || !result->ok())
        throw std::runtime_error(std::string("Molmo2 failed to load ") + name);
    return result;
}
double milliseconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}

class Pipeline final : public ITextGeneration, public IVisionLanguageGeneration {
  public:
    explicit Pipeline(const FamilyContext& context) {
        const auto bytes = section(context.reader, "runtime.json");
        const auto config = nlohmann::json::parse(bytes.begin(), bytes.end());
        layers_ = positive(config, "num_layers");
        hidden_ = positive(config, "hidden_size");
        kv_dim_ = positive(config, "kv_dim");
        max_length_ = positive(config, "max_sequence_length");
        vocab_ = positive(config, "vocab_size");
        embedding_rows_ = positive(config, "embedding_rows");
        bos_ = config.at("bos_token_id").get<int>();
        eos_ = config.at("eos_token_id").get<int>();
        const auto precision = config.at("precision").get<std::string>();
        if (precision != "bf16" && precision != "fp32")
            throw std::runtime_error("Molmo2 invalid precision");
        if (bos_ < 0 || bos_ >= embedding_rows_ || eos_ < 0 || eos_ >= vocab_ ||
            config.at("max_crops") != 8)
            throw std::runtime_error("Molmo2 invalid tokenizer or crop contract");
        cache_dtype_ = precision == "bf16" ? DType::kBFloat16 : DType::kFloat32;
        element_bytes_ = dtype_size(cache_dtype_);
        auto embedding = section(context.reader, "embedding.f32");
        if (embedding.size() != static_cast<std::size_t>(embedding_rows_) * hidden_ * sizeof(float))
            throw std::runtime_error("Molmo2 embedding table size mismatch");
        embedding_.resize(embedding.size() / sizeof(float));
        std::memcpy(embedding_.data(), embedding.data(), embedding.size());
        embedding.clear();
        embedding.shrink_to_fit();
        const auto tokenizer = section(context.reader, "tokenizer.json");
        tokenizer_ = CreateBpeTokenizer(tokenizer.data(), tokenizer.size(), false);
        decoder_ = module(context, "engine.plan");
        vision_ = module(context, "vision.plan");
        for (int i = 0; i < layers_; ++i)
            for (int which = 0; which < 2; ++which) {
                auto buffer = std::make_unique<DeviceBuffer>(cache_capacity());
                caches_.push_back(std::move(buffer));
                const auto name = (which == 0 ? "past_k_" : "past_v_") + std::to_string(i);
                if (!decoder_->has_input(name) || decoder_->tensor_dtype(name) != cache_dtype_)
                    throw std::runtime_error("Molmo2 decoder KV contract mismatch");
            }
    }
    const char* task() const noexcept override { return IVisionLanguageGeneration::kTask; }
    std::int32_t default_max_new_tokens() const override { return 128; }

    TextResult generate(const std::string& prompt, const TextGenerationConfig& cfg) override {
        std::lock_guard<std::mutex> lock(mutex_);
        validate_generation(cfg);
        if (prompt.empty())
            throw std::invalid_argument("Molmo2 requires a nonempty prompt");
        auto ids = tokenizer_->encode(cfg.use_chat_template ? chat(prompt) : prompt);
        if (ids.empty() || ids.front() != bos_)
            ids.insert(ids.begin(), bos_);
        return generate_ids(ids, {}, cfg);
    }

    TextResult generate(const std::string& prompt, const float* pixels, int height, int width,
                        const TextGenerationConfig& cfg) override {
        std::lock_guard<std::mutex> lock(mutex_);
        validate_generation(cfg);
        if (prompt.empty())
            throw std::invalid_argument("Molmo2 requires a nonempty prompt");
        const auto started = std::chrono::steady_clock::now();
        auto image = preprocess(pixels, height, width);
        auto ids = tokenizer_->encode(image_prompt(image) + chat(prompt));
        ids.insert(ids.begin(), bos_);
        validate_length(ids.size(), cfg.max_new_tokens);
        const int pooled = static_cast<int>(image.pooling.size() / 4);
        TensorMap inputs{
            {"pixels", {image.pixels.data(), {image.crops, 729, 588}, DType::kFloat32}},
            {"pooling", {image.pooling.data(), {pooled, 4}, DType::kInt32}},
        };
        vision_->forward_async(inputs);
        std::vector<float> features(static_cast<std::size_t>(pooled) * hidden_);
        checked(cudaMemcpyAsync(features.data(), vision_->device_ptr("features"),
                                features.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                vision_->stream()));
        vision_->sync();
        auto result = generate_ids(ids, features, cfg);
        result.setup_ms = milliseconds(started) - result.prefill_ms - result.decode_ms;
        return result;
    }

  private:
    static int positive(const nlohmann::json& config, const char* name) {
        const auto& value = config.at(name);
        if (!value.is_number_integer() || value.get<int>() <= 0)
            throw std::runtime_error(std::string("Molmo2 invalid runtime dimension: ") + name);
        return value.get<int>();
    }
    static std::string chat(const std::string& prompt) {
        return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    }
    static void validate_generation(const TextGenerationConfig& cfg) {
        if (cfg.max_new_tokens < 1 || cfg.top_k != 1 || cfg.top_p != 1 || cfg.min_p != 0 ||
            cfg.temperature < 0 || cfg.repetition_penalty != 1 || !cfg.system_prompt.empty() ||
            !cfg.lora_adapter_id.empty())
            throw std::invalid_argument(
                "Molmo2 currently supports greedy generation without system or LoRA overrides");
    }
    void validate_length(std::size_t size, int generated) const {
        if (size == 0 ||
            size + static_cast<std::size_t>(generated) > static_cast<std::size_t>(max_length_))
            throw std::invalid_argument(
                "Molmo2 prompt plus max_new_tokens exceeds the bundle sequence limit");
    }
    std::size_t cache_capacity() const {
        return static_cast<std::size_t>(max_length_ + 1) * kv_dim_ * element_bytes_;
    }
    std::vector<float> step(const std::vector<std::int32_t>& ids,
                            const std::vector<float>& features, int past) {
        const int n = static_cast<int>(ids.size());
        std::vector<float> embeds(static_cast<std::size_t>(n) * hidden_);
        std::size_t feature = 0;
        for (int i = 0; i < n; ++i) {
            if (ids[i] < 0 || ids[i] >= embedding_rows_)
                throw std::invalid_argument("Molmo2 token ID outside embedding table");
            auto* destination = embeds.data() + static_cast<std::size_t>(i) * hidden_;
            std::copy_n(embedding_.data() + static_cast<std::size_t>(ids[i]) * hidden_, hidden_,
                        destination);
            if (ids[i] == 151938 && !features.empty()) {
                if (feature + hidden_ > features.size())
                    throw std::runtime_error("Molmo2 missing image features");
                for (int j = 0; j < hidden_; ++j)
                    destination[j] += features[feature++];
            }
        }
        if (feature != features.size())
            throw std::runtime_error("Molmo2 unused image features");
        std::vector<std::int32_t> positions(n);
        std::iota(positions.begin(), positions.end(), past);
        auto mask = attention_mask(ids, past);
        for (int layer = 0; layer < layers_; ++layer)
            for (int which = 0; which < 2; ++which) {
                const auto name = (which == 0 ? "past_k_" : "past_v_") + std::to_string(layer);
                decoder_->bind_external(name, caches_[2 * layer + which]->data(),
                                        {past + 1, kv_dim_});
            }
        TensorMap inputs{
            {"embeddings", {embeds.data(), {n, hidden_}, DType::kFloat32}},
            {"positions", {positions.data(), {n}, DType::kInt32}},
            {"attention_mask", {mask.data(), {1, n, past + 1 + n}, DType::kFloat32}},
        };
        decoder_->forward_async(inputs);
        for (int layer = 0; layer < layers_; ++layer)
            for (int which = 0; which < 2; ++which) {
                const auto name = (which == 0 ? "new_k_" : "new_v_") + std::to_string(layer);
                auto* destination = static_cast<char*>(caches_[2 * layer + which]->data()) +
                                    static_cast<std::size_t>(past + 1) * kv_dim_ * element_bytes_;
                checked(cudaMemcpyAsync(destination, decoder_->device_ptr(name),
                                        static_cast<std::size_t>(n) * kv_dim_ * element_bytes_,
                                        cudaMemcpyDeviceToDevice, decoder_->stream()));
            }
        std::vector<float> logits(vocab_);
        checked(cudaMemcpyAsync(logits.data(), decoder_->device_ptr("logits"),
                                logits.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                decoder_->stream()));
        decoder_->sync();
        if (!std::all_of(logits.begin(), logits.end(), [](float x) { return std::isfinite(x); }))
            throw std::runtime_error("Molmo2 produced nonfinite logits");
        return logits;
    }
    TextResult generate_ids(const std::vector<std::int32_t>& ids,
                            const std::vector<float>& features, const TextGenerationConfig& cfg) {
        validate_length(ids.size(), cfg.max_new_tokens);
        decoder_->reset_execution_context();
        for (const auto& buffer : caches_)
            checked(cudaMemsetAsync(buffer->data(), 0, cache_capacity(), decoder_->stream()));
        auto started = std::chrono::steady_clock::now();
        auto logits = step(ids, features, 0);
        TextResult result;
        result.prefill_ms = milliseconds(started);
        int past = static_cast<int>(ids.size());
        const int eos = cfg.eos_token_id >= 0 ? cfg.eos_token_id : eos_;
        started = std::chrono::steady_clock::now();
        for (int i = 0; i < cfg.max_new_tokens; ++i) {
            const auto token = static_cast<std::int32_t>(
                std::max_element(logits.begin(), logits.end()) - logits.begin());
            result.token_ids.push_back(token);
            if (token == eos || i + 1 == cfg.max_new_tokens)
                break;
            logits = step({token}, {}, past++);
        }
        result.decode_ms = milliseconds(started);
        auto decoded = result.token_ids;
        if (!decoded.empty() && decoded.back() == eos)
            decoded.pop_back();
        result.text = tokenizer_->decode(decoded);
        return result;
    }
    int layers_{0}, hidden_{0}, kv_dim_{0}, max_length_{0}, vocab_{0}, embedding_rows_{0}, bos_{0},
        eos_{0};
    std::size_t element_bytes_{0};
    DType cache_dtype_{DType::kFloat32};
    std::vector<float> embedding_;
    std::unique_ptr<ITokenizer> tokenizer_;
    std::unique_ptr<ITrtModule> decoder_, vision_;
    std::vector<std::unique_ptr<DeviceBuffer>> caches_;
    std::mutex mutex_;
};
} // namespace
ITask* create(const FamilyContext& context) {
    return new Pipeline(context);
}
} // namespace trtmc::molmo2

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("Molmo2 does not support --kv-cache-size");
    return trtmc::molmo2::create(context);
}
