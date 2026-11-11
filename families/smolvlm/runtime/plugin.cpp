/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "media.h"
#include "prompt.h"
#include "request.h"
#include "sampler.h"
#include "tokenizer.h"
#include "trtmc/internal/language.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::smolvlm {
namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using namespace internal;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void cuda_check(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("SmolVLM CUDA error: ") + cudaGetErrorString(error));
}

std::string read_text(const BundleReader& reader, const char* name) {
    const auto bytes = reader.read_section(name);
    return {bytes.begin(), bytes.end()};
}

std::unique_ptr<ITrtModule> load(const FamilyContext& context, const char* name) {
    const auto plan = context.reader.read_section(name);
    auto module = context.backend.create_module(plan.data(), plan.size(), {});
    if (!module || !module->ok())
        throw std::runtime_error(std::string("SmolVLM failed to load ") + name);
    module->set_timing_label(name);
    return module;
}

// Host float32 values in the engines' activation precision.
std::vector<uint8_t> pack(const float* values, size_t count, DType dtype) {
    std::vector<uint8_t> result(count * dtype_size(dtype));
    if (dtype == DType::kFloat32) {
        std::memcpy(result.data(), values, result.size());
    } else if (dtype == DType::kFloat16) {
        auto* output = reinterpret_cast<__half*>(result.data());
        for (size_t i = 0; i < count; ++i)
            output[i] = __float2half_rn(values[i]);
    } else if (dtype == DType::kBFloat16) {
        auto* output = reinterpret_cast<__nv_bfloat16*>(result.data());
        for (size_t i = 0; i < count; ++i)
            output[i] = __float2bfloat16_rn(values[i]);
    } else {
        throw std::runtime_error("SmolVLM engines must run in fp32, fp16 or bf16");
    }
    return result;
}

int positive(const Json& document, const char* name) {
    const auto value = document.at(name).get<int64_t>();
    if (value <= 0 || value > std::numeric_limits<int>::max())
        throw std::invalid_argument(std::string("Invalid SmolVLM runtime value ") + name);
    return static_cast<int>(value);
}

class Pipeline final : public IModel, public IImagesTextToText {
  public:
    explicit Pipeline(const FamilyContext& context)
        : task_(context.reader.info().task),
          runtime_(Json::parse(read_text(context.reader, "runtime.json"))),
          tokenizer_(read_text(context.reader, "tokenizer.json")),
          embeddings_(context.reader.read_section("embeddings.f32")) {
        if (context.kv_cache_size_bytes != 0)
            throw std::invalid_argument("smolvlm does not support --kv-cache-size");
        if (task_ != IImagesTextToText::kTask || runtime_.at("format") != 1)
            throw std::invalid_argument("Invalid SmolVLM bundle");
        hidden_ = positive(runtime_, "hidden_size");
        layers_ = positive(runtime_, "num_hidden_layers");
        kv_heads_ = positive(runtime_, "num_key_value_heads");
        head_dim_ = positive(runtime_, "head_dim");
        vocab_ = positive(runtime_, "vocab_size");
        capacity_ = positive(runtime_, "max_sequence_length");
        chunk_ = positive(runtime_, "max_prefill_chunk");
        max_tiles_ = positive(runtime_, "max_tiles_per_image");
        image_size_ = positive(runtime_, "image_size");
        image_token_ = runtime_.at("image_token_id");
        eos_ = runtime_.at("eos_token_id");
        const auto& processing = runtime_.at("image_processing");
        processing_.longest_edge = positive(processing, "longest_edge");
        processing_.tile = positive(processing, "tile_size");
        processing_.rescale_factor = processing.at("rescale_factor");
        processing_.mean = processing.at("image_mean").get<std::array<float, 3>>();
        processing_.std = processing.at("image_std").get<std::array<float, 3>>();
        processing_.image_seq_len = positive(processing, "image_seq_len");
        if (head_dim_ % 2 || chunk_ > capacity_ || processing_.tile != image_size_ ||
            embeddings_.size() != size_t(vocab_) * hidden_ * sizeof(float) ||
            tokenizer_.added_token_id(
                runtime_.at("special_tokens").at("image_token").get<std::string>()) !=
                image_token_ ||
            tokenizer_.added_token_id(
                runtime_.at("special_tokens").at("end_of_utterance_token").get<std::string>()) !=
                eos_)
            throw std::invalid_argument("Inconsistent SmolVLM bundle tables");
        const double theta = runtime_.at("rope_theta");
        // Llama default RoPE, evaluated in float32 like the reference.
        for (int j = 0; j < head_dim_ / 2; ++j)
            inverse_frequency_.push_back(
                1.0F / std::pow(static_cast<float>(theta),
                                static_cast<float>(2 * j) / static_cast<float>(head_dim_)));
        decoder_ = load(context, "decoder.plan");
        vision_ = load(context, "vision.plan");
        dtype_ = decoder_->tensor_dtype("inputs_embeds");
        if (dtype_ != vision_->tensor_dtype("pixel_values") ||
            dtype_ != vision_->tensor_dtype("image_features"))
            throw std::invalid_argument("SmolVLM vision and decoder precisions differ");
        for (int layer = 0; layer < layers_; ++layer)
            for (const char* kind : {"k", "v"}) {
                cache_.push_back(DeviceTensor::zeros({kv_heads_, capacity_, head_dim_}, dtype_,
                                                     decoder_->stream()));
                if (!cache_.back().ok())
                    throw std::runtime_error("SmolVLM cache allocation failed");
                decoder_->bind_external(std::string("cache_") + kind + "_" + std::to_string(layer),
                                        cache_.back().data());
            }
    }

    const char* task() const noexcept override { return task_.c_str(); }

    std::vector<TaskInstance> task_bindings() override {
        return {bind<IImagesTextToText>(*this, config_view())};
    }

    TextResult run(const ImagesTextToTextRequest& request, ConfigView supplied) override {
        if (!request.tools.empty())
            throw std::invalid_argument("SmolVLM does not take tool definitions");
        const auto config = read_config(supplied);
        Sampler sampler(config.sampling);
        std::lock_guard<std::mutex> lock(mutex_);
        const auto setup = Clock::now();
        std::vector<TiledImage> images;
        const auto text = render_chat(request.messages, [&](const ImageView& view) {
            images.push_back(tile_image(to_rgb8(view), processing_));
            if (images.back().frames > max_tiles_)
                throw std::logic_error("SmolVLM image produced more tiles than the vision profile");
            return image_prompt(images.back().rows, images.back().cols, processing_.image_seq_len);
        });
        const auto ids = tokenizer_.encode(text);
        if (ids.empty() || ids.size() + size_t(config.max_new_tokens) - 1 > size_t(capacity_))
            throw std::invalid_argument("SmolVLM prompt (" + std::to_string(ids.size()) +
                                        " tokens) plus max_new_tokens exceeds the built context (" +
                                        std::to_string(capacity_) + ")");
        auto embeddings = embed(ids);
        insert_images(ids, images, embeddings);
        for (auto& tensor : cache_)
            cuda_check(cudaMemsetAsync(tensor.data(), 0, tensor.nbytes(), decoder_->stream()));
        TextResult result;
        result.setup_ms = elapsed(setup);
        const auto prefill = Clock::now();
        const size_t row = size_t(hidden_) * dtype_size(dtype_);
        std::vector<float> logits;
        for (size_t offset = 0; offset < ids.size(); offset += size_t(chunk_)) {
            const int count = int(std::min(ids.size() - offset, size_t(chunk_)));
            logits = step(embeddings.data() + offset * row, count, int(offset));
        }
        result.prefill_ms = elapsed(prefill);
        const auto decode = Clock::now();
        for (int i = 0; i < config.max_new_tokens; ++i) {
            const int32_t token = sampler.next(logits.data(), vocab_);
            result.token_ids.push_back(token);
            if (token == eos_ || i + 1 == config.max_new_tokens)
                break;
            const auto next = embed({token});
            logits = step(next.data(), 1, int(ids.size()) + i);
        }
        result.decode_ms = elapsed(decode);
        result.text = tokenizer_.decode(result.token_ids);
        return result;
    }

  private:
    std::string task_;
    Json runtime_;
    Tokenizer tokenizer_;
    std::vector<char> embeddings_;
    ImageProcessing processing_;
    int hidden_{}, layers_{}, kv_heads_{}, head_dim_{}, vocab_{}, capacity_{}, chunk_{};
    int max_tiles_{}, image_size_{};
    int32_t image_token_{}, eos_{};
    std::vector<float> inverse_frequency_;
    DType dtype_{};
    // Destroy the execution contexts before the external cache storage.
    std::vector<DeviceTensor> cache_;
    std::unique_ptr<ITrtModule> decoder_, vision_;
    std::mutex mutex_;

    std::vector<uint8_t> embed(const std::vector<int32_t>& ids) const {
        std::vector<float> values(ids.size() * size_t(hidden_));
        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] < 0 || ids[i] >= vocab_)
                throw std::invalid_argument("SmolVLM token id is outside the vocabulary");
            std::memcpy(values.data() + i * hidden_,
                        embeddings_.data() + size_t(ids[i]) * hidden_ * sizeof(float),
                        size_t(hidden_) * sizeof(float));
        }
        return pack(values.data(), values.size(), dtype_);
    }

    // Image features replace the <image> placeholders in prompt order:
    // tiles of the first image, its global view, then the next image.
    void insert_images(const std::vector<int32_t>& ids, const std::vector<TiledImage>& images,
                       std::vector<uint8_t>& embeddings) {
        std::vector<size_t> slots;
        for (size_t i = 0; i < ids.size(); ++i)
            if (ids[i] == image_token_)
                slots.push_back(i);
        size_t expected = 0;
        for (const auto& image : images)
            expected += size_t(image.frames) * processing_.image_seq_len;
        if (slots.size() != expected)
            throw std::invalid_argument("SmolVLM prompt image slots do not match the images");
        const size_t row = size_t(hidden_) * dtype_size(dtype_);
        size_t next = 0;
        for (const auto& image : images) {
            const auto pixels = pack(image.pixels.data(), image.pixels.size(), dtype_);
            TensorMap input{{"pixel_values",
                             {const_cast<uint8_t*>(pixels.data()),
                              {image.frames, 3, image_size_, image_size_},
                              dtype_}}};
            const auto output = vision_->forward(input).at("image_features");
            const size_t rows = size_t(image.frames) * processing_.image_seq_len;
            if (output.dtype != dtype_ || output.nbytes() != rows * row)
                throw std::runtime_error("SmolVLM vision output shape mismatch");
            const auto* features = static_cast<const uint8_t*>(output.data);
            for (size_t r = 0; r < rows; ++r)
                std::memcpy(embeddings.data() + slots[next++] * row, features + r * row, row);
        }
    }

    // One decoder call over `count` new rows at cache offset `offset`.
    std::vector<float> step(const uint8_t* embeddings, int count, int offset) {
        const int dim = head_dim_;
        std::vector<float> cosine(size_t(count) * dim), sine(cosine.size());
        for (int i = 0; i < count; ++i)
            for (int j = 0; j < dim / 2; ++j) {
                const float phase = inverse_frequency_[size_t(j)] * float(offset + i);
                for (int half = 0; half < 2; ++half) {
                    cosine[size_t(i) * dim + j + half * (dim / 2)] = std::cos(phase);
                    sine[size_t(i) * dim + j + half * (dim / 2)] = std::sin(phase);
                }
            }
        // Row i sees cache slots [0, offset) and the new rows [0, i].
        const int keys = capacity_ + count;
        std::vector<float> mask(size_t(count) * keys, std::numeric_limits<float>::lowest());
        for (int i = 0; i < count; ++i) {
            std::fill_n(mask.data() + size_t(i) * keys, offset, 0.0F);
            std::fill_n(mask.data() + size_t(i) * keys + capacity_, i + 1, 0.0F);
        }
        TensorMap input{
            {"inputs_embeds", {const_cast<uint8_t*>(embeddings), {count, hidden_}, dtype_}},
            {"rope_cos", {cosine.data(), {count, dim}, DType::kFloat32}},
            {"rope_sin", {sine.data(), {count, dim}, DType::kFloat32}},
            {"attention_mask", {mask.data(), {1, count, keys}, DType::kFloat32}},
        };
        decoder_->forward_async(input);
        std::vector<float> logits(static_cast<size_t>(vocab_));
        cuda_check(cudaMemcpyAsync(logits.data(), decoder_->device_ptr("logits"),
                                   logits.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                   decoder_->stream()));
        const size_t row = size_t(dim) * dtype_size(dtype_);
        for (int layer = 0; layer < layers_; ++layer)
            for (int kind = 0; kind < 2; ++kind) {
                const auto name =
                    std::string("present_") + (kind ? "v_" : "k_") + std::to_string(layer);
                cuda_check(cudaMemcpy2DAsync(
                    static_cast<uint8_t*>(cache_[size_t(2 * layer + kind)].data()) + offset * row,
                    size_t(capacity_) * row, decoder_->device_ptr(name), size_t(count) * row,
                    size_t(count) * row, size_t(kv_heads_), cudaMemcpyDeviceToDevice,
                    decoder_->stream()));
            }
        decoder_->sync();
        if (!std::all_of(logits.begin(), logits.end(), [](float x) { return std::isfinite(x); }))
            throw std::runtime_error("SmolVLM decoder produced non-finite logits");
        return logits;
    }
};

} // namespace
} // namespace trtmc::smolvlm

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    return new trtmc::smolvlm::Pipeline(context);
}
