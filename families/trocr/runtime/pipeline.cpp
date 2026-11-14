/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/trocr/runtime/pipeline.h"

#include <chrono>
#include <cmath>
#include <cuda_runtime_api.h>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace trtmc::trocr {
namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

const internal::ConfigField kConfigFields[] = {
    {"max_new_tokens", internal::ConfigKind::I64, std::nullopt,
     "Most tokens to generate, the end token included. Default: until the end token or the "
     "bundle's decode capacity (its max_sequence_length)."},
};

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("trocr ") + what + ": " + cudaGetErrorString(status));
}

DeviceTensor device_zeros(std::vector<int64_t> shape, DType dtype) {
    auto tensor = DeviceTensor::zeros(std::move(shape), dtype, nullptr);
    if (!tensor.ok())
        throw std::runtime_error("trocr could not allocate its decode buffers");
    return tensor;
}

} // namespace

DecodeBuffers allocate_buffers(const RuntimeConfig& config) {
    const std::int64_t layers = config.layers, heads = config.heads, dim = config.head_dim;
    const auto dtype = config.work_dtype;
    DecodeBuffers buffers;
    buffers.cross_key = device_zeros({layers, heads, config.encoder_sequence_length, dim}, dtype);
    buffers.cross_value = device_zeros({layers, heads, config.encoder_sequence_length, dim}, dtype);
    // Masked rows are multiplied by an exact zero weight; zero-filled storage
    // keeps them finite before the first request writes them.
    buffers.cache_key = device_zeros({layers, heads, config.max_sequence_length, dim}, dtype);
    buffers.cache_value = device_zeros({layers, heads, config.max_sequence_length, dim}, dtype);
    buffers.new_key = device_zeros({layers, heads, 1, dim}, dtype);
    buffers.new_value = device_zeros({layers, heads, 1, dim}, dtype);
    check_cuda(cudaStreamSynchronize(nullptr), "zero-filling the decode buffers");
    return buffers;
}

const internal::ImageView& single_image(const internal::ImagesTextToTextRequest& request) {
    if (!request.tools.empty())
        throw std::invalid_argument("trocr takes no tools");
    const internal::ImageView* image = nullptr;
    for (const auto& message : request.messages) {
        if (message.role != internal::MessageRole::User)
            throw std::invalid_argument("trocr reads one user message; other roles are not used");
        for (const auto& part : message.parts) {
            if (const auto* text = std::get_if<internal::TextPartView>(&part)) {
                if (!text->text.empty())
                    throw std::invalid_argument(
                        "trocr transcribes the text in the image and takes no instruction; "
                        "the text part must be empty");
            } else if (const auto* view = std::get_if<internal::ImageView>(&part)) {
                if (image != nullptr)
                    throw std::invalid_argument("trocr reads exactly one image per request");
                image = view;
            } else {
                throw std::invalid_argument("trocr accepts only an image and an empty text part");
            }
        }
    }
    if (image == nullptr)
        throw std::invalid_argument("trocr needs one image");
    return *image;
}

Span<const internal::ConfigField> config_fields() noexcept {
    return {kConfigFields, sizeof(kConfigFields) / sizeof(kConfigFields[0])};
}

std::int32_t resolve_max_new_tokens(internal::ConfigView config, std::int32_t capacity) {
    const auto requested =
        internal::config_get<std::int64_t>(config, config_fields(), "max_new_tokens");
    if (!requested)
        return capacity;
    if (*requested < 1 || *requested > capacity)
        throw internal::ConfigError("trocr max_new_tokens must be in [1, " +
                                    std::to_string(capacity) + "] for this bundle");
    return static_cast<std::int32_t>(*requested);
}

std::int32_t greedy_token(const float* logits, std::int32_t vocab_size) {
    std::int32_t best = 0;
    for (std::int32_t index = 0; index < vocab_size; ++index) {
        if (!std::isfinite(logits[index]))
            throw std::runtime_error("trocr decoder produced a non-finite logit");
        if (logits[index] > logits[best])
            best = index;
    }
    return best;
}

TrocrPipeline::TrocrPipeline(RuntimeConfig config, std::unique_ptr<DecodeBuffers> buffers,
                             std::unique_ptr<ITrtModule> encoder,
                             std::unique_ptr<ITrtModule> decoder, ByteLevelDetokenizer detokenizer)
    : config_(std::move(config)), buffers_(std::move(buffers)), encoder_(std::move(encoder)),
      decoder_(std::move(decoder)), detokenizer_(std::move(detokenizer)) {
    if (!buffers_ || !encoder_ || !encoder_->ok() || !decoder_ || !decoder_->ok())
        throw std::runtime_error("trocr pipeline needs loaded engines and their buffers");
    if (detokenizer_.size() != static_cast<std::size_t>(config_.vocab_size))
        throw std::runtime_error("trocr vocabulary does not match the decoder's logits");
}

const char* TrocrPipeline::task() const noexcept {
    return internal::IImagesTextToText::kTask.data();
}

std::vector<internal::TaskInstance> TrocrPipeline::task_bindings() {
    return {internal::bind<internal::IImagesTextToText>(*this, config_fields())};
}

std::int32_t TrocrPipeline::decode_step(std::int32_t token, std::int32_t position) {
    std::int32_t token_value = token, position_value = position;
    TensorMap inputs;
    inputs["token_id"] = Tensor{&token_value, {1}, DType::kInt32};
    inputs["position"] = Tensor{&position_value, {1}, DType::kInt32};
    const auto outputs = decoder_->forward(inputs);
    const auto found = outputs.find("logits");
    if (found == outputs.end() || found->second.data == nullptr ||
        found->second.dtype != DType::kFloat32 ||
        found->second.numel() != static_cast<std::size_t>(config_.vocab_size))
        throw std::runtime_error("trocr decoder returned unusable logits");

    // The token's own key/value rows become cache row `position` for the
    // steps after it: [layers * heads] rows of head_dim, strided by capacity.
    const auto row = static_cast<std::size_t>(config_.head_dim) * dtype_size(config_.work_dtype);
    const auto rows = static_cast<std::size_t>(config_.layers) * config_.heads;
    const auto pitch = row * static_cast<std::size_t>(config_.max_sequence_length);
    const auto offset = row * static_cast<std::size_t>(position);
    for (auto [source, target] : {std::pair{&buffers_->new_key, &buffers_->cache_key},
                                  std::pair{&buffers_->new_value, &buffers_->cache_value}}) {
        check_cuda(cudaMemcpy2DAsync(static_cast<char*>(target->data()) + offset, pitch,
                                     source->data(), row, row, rows, cudaMemcpyDeviceToDevice,
                                     decoder_->stream()),
                   "writing the key/value cache");
    }
    return greedy_token(static_cast<const float*>(found->second.data), config_.vocab_size);
}

TextResult TrocrPipeline::run(const internal::ImagesTextToTextRequest& request,
                              internal::ConfigView config) {
    const auto started = Clock::now();
    const auto& image = single_image(request);
    const auto max_new_tokens = resolve_max_new_tokens(config, config_.max_sequence_length);
    auto pixels = preprocess(image, config_.preprocess);
    const auto preprocessed = Clock::now();

    std::lock_guard<std::mutex> lock(mutex_);
    const auto prepared = Clock::now();
    // Each request starts from an empty cache: masked rows meet an exact zero
    // weight, and zeros keep anything an earlier request left out of reach.
    for (auto* cache : {&buffers_->cache_key, &buffers_->cache_value})
        check_cuda(cudaMemsetAsync(cache->data(), 0, cache->nbytes(), decoder_->stream()),
                   "clearing the key/value cache");
    TensorMap encoder_inputs;
    encoder_inputs["pixel_values"] =
        Tensor{pixels.data(),
               {1, 3, config_.preprocess.height, config_.preprocess.width},
               DType::kFloat32};
    encoder_->forward(encoder_inputs); // writes cross_key / cross_value

    // Greedy decoding from the start token. Position p feeds the token at p
    // and yields the token at p + 1; the end token is kept in token_ids.
    std::vector<std::int32_t> generated;
    std::int32_t token = config_.decoder_start_token_id;
    auto first_token = prepared;
    for (std::int32_t position = 0; position < max_new_tokens; ++position) {
        token = decode_step(token, position);
        generated.push_back(token);
        if (position == 0)
            first_token = Clock::now();
        if (token == config_.eos_token_id)
            break;
    }
    check_cuda(cudaStreamSynchronize(decoder_->stream()), "finishing the decode");
    const auto finished = Clock::now();

    // Decode before the ids move into the result: argument order is unspecified.
    auto text = detokenizer_.decode({generated.data(), generated.size()});
    TextResult result(std::move(text), std::move(generated), milliseconds(prepared, first_token),
                      milliseconds(first_token, finished));
    result.setup_ms = milliseconds(started, preprocessed);
    return result;
}

} // namespace trtmc::trocr
