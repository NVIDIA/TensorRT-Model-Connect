/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/diffusion_gemma/runtime/pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace trtmc::diffusion_gemma {
namespace {

const internal::ConfigField kFields[] = {
    {"max_new_tokens", internal::ConfigKind::I64, std::nullopt,
     "Generation budget; diffusion emits complete canvases, as in the reference model."},
    {"seed", internal::ConfigKind::I64, std::int64_t{0}, "Native diffusion sampler random seed."},
};

const Tensor& require_output(const TensorMap& outputs, const std::string& name) {
    const auto found = outputs.find(name);
    if (found == outputs.end() || (found->second.nbytes() != 0 && found->second.data == nullptr))
        throw std::runtime_error("diffusion_gemma engine omitted output " + name);
    return found->second;
}

OwnedTensor empty_cache(std::int32_t heads, std::int32_t dimension, DType dtype) {
    OwnedTensor result;
    result.dtype = dtype;
    result.shape = {heads, 0, dimension};
    return result;
}

OwnedTensor tail_cache(const Tensor& tensor, std::int64_t maximum) {
    if (tensor.shape.size() != 3)
        throw std::runtime_error("diffusion_gemma cache must have [heads,length,dimension] shape");
    if (tensor.shape[1] <= maximum)
        return OwnedTensor(tensor);
    OwnedTensor result;
    result.shape = {tensor.shape[0], maximum, tensor.shape[2]};
    result.dtype = tensor.dtype;
    const auto item = dtype_size(tensor.dtype);
    const auto source_stride = static_cast<std::size_t>(tensor.shape[1] * tensor.shape[2]) * item;
    const auto stride = static_cast<std::size_t>(maximum * tensor.shape[2]) * item;
    result.bytes.resize(static_cast<std::size_t>(tensor.shape[0]) * stride);
    const auto offset = source_stride - stride;
    for (std::int64_t head = 0; head < tensor.shape[0]; ++head)
        std::memcpy(result.bytes.data() + static_cast<std::size_t>(head) * stride,
                    static_cast<const std::uint8_t*>(tensor.data) +
                        static_cast<std::size_t>(head) * source_stride + offset,
                    stride);
    return result;
}

} // namespace

OwnedTensor::OwnedTensor(const Tensor& tensor) : shape(tensor.shape), dtype(tensor.dtype) {
    bytes.resize(tensor.nbytes());
    if (!bytes.empty()) {
        if (tensor.data == nullptr)
            throw std::runtime_error("diffusion_gemma tensor data is missing");
        std::memcpy(bytes.data(), tensor.data, bytes.size());
    }
}

Tensor OwnedTensor::view() const {
    return {const_cast<std::uint8_t*>(bytes.data()), shape, dtype};
}

Pipeline::Pipeline(BundleReader reader, IBackend& backend, RuntimeConfig config,
                   std::unique_ptr<ITokenizer> tokenizer)
    : reader_(std::move(reader)), backend_(backend), config_(std::move(config)),
      tokenizer_(std::move(tokenizer)) {
    if (!tokenizer_)
        throw std::runtime_error("diffusion_gemma tokenizer could not be loaded");
    layers_.resize(static_cast<std::size_t>(config_.num_layers));
    reset();
}

std::unique_ptr<ITrtModule> Pipeline::load(const std::string& name) {
    const auto* section = reader_.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("diffusion_gemma bundle is missing " + name);
    const auto plan = reader_.read_section(name);
    auto module = backend_.create_module(plan.data(), plan.size(), {});
    if (!module || !module->ok())
        throw std::runtime_error("diffusion_gemma failed to load " + name);
    module->set_timing_label("diffusion_gemma/" + name);
    return module;
}

ITrtModule& Pipeline::layer(std::size_t index) {
    auto& module = layers_.at(index);
    if (!module)
        module = load("layer_" + std::to_string(index) + ".plan");
    return *module;
}

void Pipeline::reset() {
    prefix_length_ = 0;
    keys_.clear();
    values_.clear();
    for (const auto& type : config_.layer_types) {
        const bool full = type == "full_attention";
        const auto heads = full ? config_.num_global_key_value_heads : config_.num_key_value_heads;
        const auto dimension = full ? config_.global_head_dim : config_.head_dim;
        keys_.push_back(empty_cache(heads, dimension, config_.dtype));
        values_.push_back(empty_cache(heads, dimension, config_.dtype));
    }
}

std::vector<internal::TaskInstance> Pipeline::task_bindings() {
    return {internal::bind<internal::ITextContinuation>(*this, kFields)};
}

void Pipeline::validate_ids(const std::vector<std::int32_t>& ids, bool external_prompt) const {
    validate_token_ids(ids, config_, external_prompt);
}

void validate_token_ids(const std::vector<std::int32_t>& ids, const RuntimeConfig& config,
                        bool external_prompt) {
    if (ids.empty())
        throw std::invalid_argument("diffusion_gemma requires a nonempty token sequence");
    for (const auto id : ids) {
        if (id < 0 || id >= config.vocab_size)
            throw std::invalid_argument("diffusion_gemma token ID is outside the vocabulary");
        if (external_prompt &&
            std::find(config.unsupported_token_ids.begin(), config.unsupported_token_ids.end(),
                      id) != config.unsupported_token_ids.end())
            throw std::invalid_argument(
                "diffusion_gemma text bundle does not accept image operands");
    }
}

OwnedTensor Pipeline::zero_conditioning(std::size_t length) const {
    OwnedTensor result;
    result.dtype = config_.dtype;
    result.shape = {static_cast<std::int64_t>(length), config_.hidden_size};
    result.bytes.resize(length * static_cast<std::size_t>(config_.hidden_size) *
                        dtype_size(result.dtype));
    return result;
}

OwnedTensor Pipeline::embed(const std::vector<std::int32_t>& ids,
                            const OwnedTensor& self_conditioning, bool decoder) {
    if (!embedding_)
        embedding_ = load("embedding.plan");
    if (self_conditioning.shape !=
            std::vector<std::int64_t>{static_cast<std::int64_t>(ids.size()), config_.hidden_size} ||
        self_conditioning.dtype != config_.dtype)
        throw std::invalid_argument(
            "diffusion_gemma self-conditioning tensor has an invalid layout");
    Tensor input{const_cast<std::int32_t*>(ids.data()),
                 {static_cast<std::int64_t>(ids.size())},
                 DType::kInt32};
    const auto outputs = embedding_->forward(
        {{"input_ids", input}, {"self_conditioning", self_conditioning.view()}});
    return OwnedTensor(
        require_output(outputs, decoder ? "decoder_embeddings" : "encoder_embeddings"));
}

OwnedTensor Pipeline::transform(OwnedTensor hidden, bool decoder, const TraceCallback& trace) {
    const auto length = hidden.shape.at(0);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(length));
    std::iota(positions.begin(), positions.end(), static_cast<std::int32_t>(prefix_length_));
    Tensor position_tensor{positions.data(), {length}, DType::kInt32};
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        const auto past = keys_[index].shape.at(1);
        const bool sliding = config_.layer_types[index] == "sliding_attention";
        auto mask = attention_mask(length, past, sliding ? config_.sliding_window : 0, decoder);
        Tensor mask_tensor{mask.data(), {length, past + length}, DType::kFloat32};
        auto outputs = layer(index).forward({{"hidden_states", hidden.view()},
                                             {"position_ids", position_tensor},
                                             {"attention_mask", mask_tensor},
                                             {"past_key", keys_[index].view()},
                                             {"past_value", values_[index].view()}});
        hidden = OwnedTensor(require_output(outputs, "hidden_states_out"));
        if (trace)
            trace("layer_" + std::to_string(index) + "_hidden_states", hidden.view());
        if (!decoder) {
            const auto limit = sliding ? config_.sliding_window - 1 : config_.max_sequence_length;
            keys_[index] = tail_cache(require_output(outputs, "present_key"), limit);
            values_[index] = tail_cache(require_output(outputs, "present_value"), limit);
        }
        // Plan weights and profile-maximum activation buffers are large. Each
        // model-owned layer can be released once its outputs have been copied;
        // cache and sequence state belong to this pipeline, not the module.
        layers_[index].reset();
    }
    return hidden;
}

void Pipeline::encode(const std::vector<std::int32_t>& input_ids, const TraceCallback& trace) {
    encode_tokens(input_ids, true, trace);
}

void Pipeline::encode_tokens(const std::vector<std::int32_t>& input_ids, bool external_prompt,
                             const TraceCallback& trace) {
    validate_ids(input_ids, external_prompt);
    if (prefix_length_ + static_cast<std::int64_t>(input_ids.size()) > config_.max_sequence_length)
        throw std::invalid_argument(
            "diffusion_gemma encoder input exceeds the built sequence capacity");
    auto hidden = embed(input_ids, zero_conditioning(input_ids.size()), false);
    if (trace)
        trace("embeddings", hidden.view());
    (void)transform(std::move(hidden), false, trace);
    prefix_length_ += static_cast<std::int64_t>(input_ids.size());
}

DecoderObservation Pipeline::denoise(const std::vector<std::int32_t>& canvas,
                                     const OwnedTensor& self_conditioning, float temperature,
                                     const TraceCallback& trace) {
    // Uniformly initialized canvases may contain every vocabulary ID, including
    // image sentinels: they are noise token IDs rather than image operands.
    if (canvas.size() != static_cast<std::size_t>(config_.canvas_length))
        throw std::invalid_argument("diffusion_gemma requires its complete configured canvas");
    if (prefix_length_ == 0 ||
        prefix_length_ + static_cast<std::int64_t>(canvas.size()) > config_.max_sequence_length)
        throw std::invalid_argument("diffusion_gemma decoder prefix/canvas exceeds built capacity");
    if (!(temperature > 0.0F) || !std::isfinite(temperature))
        throw std::invalid_argument("diffusion_gemma requires a finite positive temperature");
    for (const auto id : canvas)
        if (id < 0 || id >= config_.vocab_size)
            throw std::invalid_argument("diffusion_gemma canvas token is outside vocabulary");
    auto hidden = embed(canvas, self_conditioning, true);
    if (trace)
        trace("decoder_embeddings", hidden.view());
    hidden = transform(std::move(hidden), true, trace);
    if (!output_)
        output_ = load("output.plan");
    Tensor temperature_tensor{&temperature, {1, 1}, DType::kFloat32};
    const auto outputs =
        output_->forward({{"hidden_states", hidden.view()}, {"temperature", temperature_tensor}});
    DecoderObservation result{OwnedTensor(require_output(outputs, "raw_logits")),
                              OwnedTensor(require_output(outputs, "self_conditioning"))};
    if (result.logits.dtype != DType::kFloat32 ||
        result.logits.shape != std::vector<std::int64_t>{config_.canvas_length, config_.vocab_size})
        throw std::runtime_error("diffusion_gemma output logits violate the declared contract");
    return result;
}

} // namespace trtmc::diffusion_gemma
