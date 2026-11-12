/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/paligemma/runtime/image_preprocess.h"
#include "families/paligemma/runtime/paligemma_bpe.h"
#include "families/paligemma/runtime/prompt.h"
#include "trtmc/internal/language.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/text.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unordered_set>

namespace trtmc::paligemma {
namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using namespace internal;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void cuda_check(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}
std::string read_text(const BundleReader& reader, const char* name) {
    auto bytes = reader.read_section(name);
    return {bytes.begin(), bytes.end()};
}
std::unique_ptr<ITrtModule> load(const FamilyContext& context, const char* name) {
    auto plan = context.reader.read_section(name);
    auto module = context.backend.create_module(plan.data(), plan.size(), {});
    if (!module || !module->ok())
        throw std::runtime_error(std::string("PaliGemma failed to load ") + name);
    module->set_timing_label(name);
    return module;
}
std::vector<uint8_t> pack(const float* values, size_t count, DType dtype) {
    std::vector<uint8_t> result(count * dtype_size(dtype));
    if (dtype == DType::kFloat32) {
        std::memcpy(result.data(), values, result.size());
    } else if (dtype == DType::kBFloat16) {
        auto* output = reinterpret_cast<__nv_bfloat16*>(result.data());
        for (size_t i = 0; i < count; ++i)
            output[i] = __float2bfloat16_rn(values[i]);
    } else {
        throw std::runtime_error("Invalid PaliGemma engine precision");
    }
    return result;
}
std::vector<uint8_t> rgb_bytes(const ImageView& image) {
    if (image.data == nullptr || image.channels != 3 || image.height == 0 || image.width == 0 ||
        image.height > uint32_t(std::numeric_limits<int32_t>::max()) ||
        image.width > uint32_t(std::numeric_limits<int32_t>::max()))
        throw std::invalid_argument("PaliGemma requires non-empty RGB images");
    const size_t values = size_t(image.height) * image.width * 3;
    std::vector<uint8_t> rgb(values);
    if (image.format == ImageFormat::UInt8) {
        if (image.byte_size != values)
            throw std::invalid_argument("PaliGemma RGB image byte size does not match its shape");
        std::memcpy(rgb.data(), image.data, values);
    } else {
        if (image.byte_size != values * sizeof(float))
            throw std::invalid_argument("PaliGemma RGB image byte size does not match its shape");
        // Float pixels are the [0, 1] scaling of 8-bit values; the reference
        // resamples the 8-bit image.
        const auto* source = static_cast<const float*>(image.data);
        for (size_t i = 0; i < values; ++i)
            rgb[i] = uint8_t(std::lround(std::clamp(source[i], 0.0F, 1.0F) * 255.0F));
    }
    return rgb;
}

const std::vector<ConfigField>& fields() {
    static const std::vector<ConfigField> result{
        {"max_new_tokens", ConfigKind::I64, int64_t{64},
         "Maximum generated tokens; decoding is greedy and stops at EOS"},
    };
    return result;
}
Span<const ConfigField> field_view() {
    return {fields().data(), fields().size()};
}

class Pipeline final : public IModel, public IImagesTextToText {
  public:
    explicit Pipeline(const FamilyContext& context)
        : task_(context.reader.info().task),
          metadata_(Json::parse(read_text(context.reader, "runtime.json"))),
          tokenizer_(parse_paligemma_bpe_asset(read_text(context.reader, "tokenizer.bpe"))),
          embeddings_(context.reader.read_section("embeddings.f32")) {
        if (context.kv_cache_size_bytes != 0)
            throw std::invalid_argument("paligemma does not support --kv-cache-size");
        if (metadata_.at("format") != 1)
            throw std::invalid_argument("Unsupported PaliGemma bundle format");
        hidden_ = metadata_.at("hidden_size");
        vocab_ = metadata_.at("vocab_size");
        layers_ = metadata_.at("num_hidden_layers");
        kv_heads_ = metadata_.at("num_key_value_heads");
        head_dim_ = metadata_.at("head_dim");
        capacity_ = metadata_.at("max_sequence_length");
        theta_ = metadata_.at("rope_theta");
        tokens_.image_token_id = metadata_.at("image_token_id");
        tokens_.image_tokens_per_image = metadata_.at("num_image_tokens");
        tokens_.bos_token_id = metadata_.at("bos_token_id");
        image_.size = metadata_.at("image_size");
        image_.rescale_factor = metadata_.at("rescale_factor");
        image_.mean = metadata_.at("image_mean").get<std::array<float, 3>>();
        image_.std = metadata_.at("image_std").get<std::array<float, 3>>();
        for (const int32_t id : metadata_.at("eos_token_ids").get<std::vector<int32_t>>())
            eos_.insert(id);
        for (const int32_t id : metadata_.at("special_token_ids").get<std::vector<int32_t>>())
            skipped_.insert(id);
        tokens_.reserved_text = reserved_text(
            tokenizer_, metadata_.at("reserved_token_ids").get<std::vector<int32_t>>(),
            tokens_.image_token_id);
        if (hidden_ <= 0 || vocab_ <= 0 || layers_ <= 0 || kv_heads_ <= 0 || head_dim_ <= 0 ||
            head_dim_ % 2 || capacity_ <= 0 || !(theta_ > 0) || eos_.empty() ||
            tokens_.image_token_id < 0 || tokens_.image_token_id >= vocab_ ||
            embeddings_.size() != size_t(hidden_) * vocab_ * sizeof(float))
            throw std::invalid_argument("Invalid PaliGemma runtime dimensions or embedding table");
        vision_ = load(context, "vision.plan");
        decoder_ = load(context, "decoder.plan");
        dtype_ = decoder_->tensor_dtype("inputs_embeds");
        if (dtype_ != vision_->tensor_dtype("pixel_values") ||
            dtype_ != vision_->tensor_dtype("image_features"))
            throw std::invalid_argument("PaliGemma engine precisions differ");
        for (int layer = 0; layer < layers_; ++layer)
            for (const char* kind : {"k", "v"}) {
                cache_.emplace_back(DeviceTensor::zeros({kv_heads_, capacity_, head_dim_}, dtype_,
                                                        decoder_->stream()));
                if (!cache_.back().ok())
                    throw std::runtime_error("PaliGemma cache allocation failed");
                decoder_->bind_external(std::string("cache_") + kind + "_" + std::to_string(layer),
                                        cache_.back().data());
            }
        inverse_frequency_.resize(size_t(head_dim_) / 2);
        // The reference computes the RoPE frequencies in float32.
        for (int j = 0; j < head_dim_ / 2; ++j)
            inverse_frequency_[size_t(j)] =
                1.0F / std::pow(float(theta_), float(2 * j) / float(head_dim_));
    }

    const char* task() const noexcept override { return task_.c_str(); }
    std::vector<TaskInstance> task_bindings() override {
        return {bind<IImagesTextToText>(*this, field_view())};
    }

    TextResult run(const ImagesTextToTextRequest& input, ConfigView supplied) override {
        validate_config(field_view(), supplied);
        const auto requested = *config_get<int64_t>(supplied, field_view(), "max_new_tokens");
        if (requested < 1 || requested > capacity_)
            throw std::invalid_argument("PaliGemma max_new_tokens must be between 1 and the "
                                        "built sequence capacity");
        // Prompt assembly reads only immutable state, so an oversized request is
        // rejected before it waits for the model. The last generated token is
        // never fed back, so it needs no cache row.
        const auto tokenize = Clock::now();
        const auto prompt = collect_prompt(input.messages, input.tools);
        const auto ids =
            prompt_ids(prompt, tokenizer_, tokens_, size_t(capacity_) - size_t(requested - 1));
        const double tokenize_ms = elapsed(tokenize);
        std::lock_guard<std::mutex> lock(mutex_);
        const auto setup = Clock::now();
        auto embeddings = embed(ids);
        for (size_t i = 0; i < prompt.images.size(); ++i)
            insert_image(prompt.images[i], i * size_t(tokens_.image_tokens_per_image), embeddings);
        for (auto& tensor : cache_)
            cuda_check(cudaMemsetAsync(tensor.data(), 0, tensor.nbytes(), decoder_->stream()));
        TextResult result;
        result.setup_ms = tokenize_ms + elapsed(setup);
        const auto prefill = Clock::now();
        auto logits = step(embeddings, int(ids.size()), 0);
        result.prefill_ms = elapsed(prefill);
        const auto decode = Clock::now();
        for (int64_t i = 0; i < requested; ++i) {
            // Greedy selection keeps the first maximum, like the reference argmax.
            const auto token =
                int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
            result.token_ids.push_back(token);
            if (eos_.count(token) != 0 || i + 1 == requested)
                break;
            embeddings = embed({token});
            logits = step(embeddings, 1, int(ids.size() + size_t(i)));
        }
        result.decode_ms = elapsed(decode);
        result.text = tokenizer_.decode(result.token_ids, skipped_);
        return result;
    }

  private:
    std::string task_;
    Json metadata_;
    PaligemmaBpeTokenizer tokenizer_;
    std::vector<char> embeddings_;
    int hidden_{}, vocab_{}, layers_{}, kv_heads_{}, head_dim_{}, capacity_{};
    double theta_{};
    PromptTokens tokens_;
    ImageNormalization image_;
    std::unordered_set<int32_t> eos_, skipped_;
    std::vector<float> inverse_frequency_;
    DType dtype_{};
    // Destroy execution contexts before their external cache storage.
    std::vector<DeviceTensor> cache_;
    std::unique_ptr<ITrtModule> vision_, decoder_;
    std::mutex mutex_;

    std::vector<uint8_t> embed(const std::vector<int32_t>& ids) {
        std::vector<float> values(ids.size() * size_t(hidden_));
        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] < 0 || ids[i] >= vocab_)
                throw std::invalid_argument("PaliGemma token id is outside the embedding table");
            std::memcpy(values.data() + i * size_t(hidden_),
                        embeddings_.data() + size_t(ids[i]) * size_t(hidden_) * sizeof(float),
                        size_t(hidden_) * sizeof(float));
        }
        return pack(values.data(), values.size(), dtype_);
    }

    void insert_image(const ImageView& image, size_t row, std::vector<uint8_t>& embeddings) {
        const auto rgb = rgb_bytes(image);
        const auto pixels =
            preprocess_image(rgb.data(), int32_t(image.height), int32_t(image.width), image_);
        auto packed = pack(pixels.data(), pixels.size(), dtype_);
        TensorMap input{
            {"pixel_values", {packed.data(), {1, 3, image_.size, image_.size}, dtype_}}};
        const auto output = vision_->forward(input).at("image_features");
        const size_t bytes =
            size_t(tokens_.image_tokens_per_image) * size_t(hidden_) * dtype_size(dtype_);
        if (output.dtype != dtype_ || output.nbytes() != bytes)
            throw std::runtime_error("PaliGemma vision output shape mismatch");
        std::memcpy(embeddings.data() + row * size_t(hidden_) * dtype_size(dtype_), output.data,
                    bytes);
    }

    // Runs `count` rows that start at sequence index `offset`. The rows of one
    // call see each other in full: the prompt is one bidirectional block and a
    // decode call carries a single token. Earlier tokens come from the cache.
    std::vector<float> step(std::vector<uint8_t>& embeddings, int count, int offset) {
        const size_t width = size_t(head_dim_);
        std::vector<float> cosine(size_t(count) * width), sine(cosine.size());
        for (int i = 0; i < count; ++i) {
            // PaliGemma positions are 1-based.
            const auto position = float(offset + i + 1);
            for (size_t j = 0; j < width / 2; ++j) {
                const float phase = position * inverse_frequency_[j];
                for (size_t half = 0; half < 2; ++half) {
                    cosine[size_t(i) * width + j + half * width / 2] = std::cos(phase);
                    sine[size_t(i) * width + j + half * width / 2] = std::sin(phase);
                }
            }
        }
        const int keys = capacity_ + count;
        std::vector<float> mask(size_t(count) * size_t(keys), std::numeric_limits<float>::lowest());
        for (int i = 0; i < count; ++i) {
            std::fill_n(mask.data() + size_t(i) * size_t(keys), offset, 0.0F);
            std::fill_n(mask.data() + size_t(i) * size_t(keys) + capacity_, count, 0.0F);
        }
        TensorMap input{
            {"inputs_embeds", {embeddings.data(), {count, hidden_}, dtype_}},
            {"rope_cos", {cosine.data(), {count, head_dim_}, DType::kFloat32}},
            {"rope_sin", {sine.data(), {count, head_dim_}, DType::kFloat32}},
            {"attention_mask", {mask.data(), {1, count, keys}, DType::kFloat32}},
        };
        decoder_->forward_async(input);
        std::vector<float> logits(static_cast<size_t>(vocab_));
        cuda_check(cudaMemcpyAsync(logits.data(), decoder_->device_ptr("logits"),
                                   logits.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                   decoder_->stream()));
        const size_t row = width * dtype_size(dtype_);
        for (int layer = 0; layer < layers_; ++layer)
            for (int kind = 0; kind < 2; ++kind) {
                const auto name =
                    std::string(kind ? "present_v_" : "present_k_") + std::to_string(layer);
                cuda_check(cudaMemcpy2DAsync(
                    static_cast<uint8_t*>(cache_[size_t(2 * layer + kind)].data()) +
                        size_t(offset) * row,
                    size_t(capacity_) * row, decoder_->device_ptr(name), size_t(count) * row,
                    size_t(count) * row, size_t(kv_heads_), cudaMemcpyDeviceToDevice,
                    decoder_->stream()));
            }
        decoder_->sync();
        if (!std::all_of(logits.begin(), logits.end(), [](float x) { return std::isfinite(x); }))
            throw std::runtime_error("PaliGemma decoder produced nonfinite logits");
        return logits;
    }
};
} // namespace
} // namespace trtmc::paligemma

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    return new trtmc::paligemma::Pipeline(context);
}
