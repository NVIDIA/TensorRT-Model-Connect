/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// BartPlugin: handles "bart_seq2seq_encoder_decoder" strategy.
// Encoder-decoder text-to-text pipeline for BART models.

#include "families/bart/runtime/decode_runtime.h"
#include "families/bart/runtime/distributed_runtime.h"
#include "families/bart/runtime/kv_cache.h"
#include "families/bart/runtime/plugin_helpers.h"
#include "families/bart/runtime/runtime_config.h"
#include "trtmc/runtime/family_factory.h"

#include <algorithm>
#include <cstring>
#include <cuda_runtime_api.h>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc {

namespace {

struct BartRuntimeConfig {
    int32_t hidden_size;
    int32_t decoder_layers;
    int32_t decoder_attention_heads;
    int32_t max_source_length;
    int32_t vocab_size;
    int32_t decoder_start_token_id;
    int32_t eos_token_id;
    int32_t bos_token_id;
    int32_t pad_token_id;
    int32_t max_cache_length;
    int32_t tensor_parallel_size;
    std::string precision;
    std::string tensor_parallel_mode;
    std::string decoder_engine_layout;
};

template <typename T>
T require_value(const nlohmann::json& json, const char* name) {
    if (!json.contains(name))
        throw std::runtime_error(std::string("bart runtime.json missing '") + name + "'");
    try {
        return json.at(name).get<T>();
    } catch (const nlohmann::json::exception&) {
        throw std::runtime_error(std::string("bart runtime.json has invalid '") + name + "'");
    }
}

BartRuntimeConfig parse_runtime_config(std::string_view text) {
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("bart invalid runtime.json: " + std::string(error.what()));
    }
    if (!json.is_object())
        throw std::runtime_error("bart runtime.json must be an object");
    if (json.size() != 22)
        throw std::runtime_error("bart runtime.json has an unexpected field set");
    const int32_t encoder_layers = require_value<int32_t>(json, "encoder_layers");
    const int32_t encoder_heads = require_value<int32_t>(json, "encoder_attention_heads");
    const int32_t encoder_ffn = require_value<int32_t>(json, "encoder_ffn_dim");
    const int32_t decoder_ffn = require_value<int32_t>(json, "decoder_ffn_dim");
    const int32_t max_positions = require_value<int32_t>(json, "max_position_embeddings");
    (void)require_value<int32_t>(json, "forced_bos_token_id");
    const int32_t position_offset = require_value<int32_t>(json, "position_embedding_offset");
    const bool has_encoder = require_value<bool>(json, "has_vision_engine");
    const bool is_encoder_decoder = require_value<bool>(json, "is_encoder_decoder");
    BartRuntimeConfig config{
        require_value<int32_t>(json, "hidden_size"),
        require_value<int32_t>(json, "decoder_layers"),
        require_value<int32_t>(json, "decoder_attention_heads"),
        require_value<int32_t>(json, "max_cache_length"),
        require_value<int32_t>(json, "vocab_size"),
        require_value<int32_t>(json, "decoder_start_token_id"),
        require_value<int32_t>(json, "eos_token_id"),
        require_value<int32_t>(json, "bos_token_id"),
        require_value<int32_t>(json, "pad_token_id"),
        require_value<int32_t>(json, "max_cache_length"),
        require_value<int32_t>(json, "tensor_parallel_size"),
        require_value<std::string>(json, "precision"),
        require_value<std::string>(json, "tensor_parallel_mode"),
        require_value<std::string>(json, "decoder_engine_layout"),
    };
    if (config.hidden_size <= 0 || config.decoder_layers <= 0 ||
        config.decoder_attention_heads <= 0 ||
        config.hidden_size % config.decoder_attention_heads != 0 || config.max_source_length <= 0 ||
        config.vocab_size <= 0 || config.max_cache_length <= 0 ||
        config.tensor_parallel_size <= 0 || encoder_layers <= 0 || encoder_heads <= 0 ||
        encoder_ffn <= 0 || decoder_ffn <= 0 || max_positions <= 0 || !has_encoder ||
        !is_encoder_decoder || position_offset != 2 ||
        config.tensor_parallel_mode !=
            (config.tensor_parallel_size > 1 ? "tensor_parallel" : "single") ||
        config.decoder_engine_layout !=
            (config.tensor_parallel_size > 1 ? "dual_profile" : "single") ||
        (config.precision != "fp16" && config.precision != "fp32")) {
        throw std::runtime_error("bart runtime.json contains invalid geometry");
    }
    return config;
}

std::string tp_engine_section_name(int32_t rank) {
    return "engine.rank" + std::to_string(rank) + ".plan";
}

int32_t dim_at(const std::vector<int64_t>& shape, int32_t dim) {
    if (dim < 0 || static_cast<std::size_t>(dim) >= shape.size())
        return -1;
    const auto value = shape[static_cast<std::size_t>(dim)];
    if (value <= 0 || value > std::numeric_limits<int32_t>::max())
        return -1;
    return static_cast<int32_t>(value);
}

int32_t decoder_cache_row_width(const ITrtModule& module, const BartRuntimeConfig& config) {
    const int32_t from_engine = dim_at(module.tensor_shape("cache_k_0"), 1);
    return from_engine > 0 ? from_engine : config.hidden_size;
}

} // namespace

namespace bart {

void validate_runtime_config_json(std::string_view json) {
    (void)parse_runtime_config(json);
}

} // namespace bart

class BartPipeline final : public ITextGeneration {
  public:
    BartPipeline(std::unique_ptr<ITrtModule> encoder, std::unique_ptr<ITrtModule> decoder,
                 std::unique_ptr<BartInferenceState> state, int32_t hidden_size,
                 int32_t num_decoder_layers, int32_t max_source_length,
                 int32_t decoder_start_token_id, int32_t eos_token_id, int32_t bos_token_id,
                 int32_t pad_token_id, cudaStream_t stream, std::shared_ptr<ITokenizer> tokenizer,
                 std::string model_id_str)
        : encoder_(std::move(encoder)), decoder_(std::move(decoder)), state_(std::move(state)),
          hidden_size_(hidden_size), num_decoder_layers_(num_decoder_layers),
          max_source_length_(max_source_length), decoder_start_token_id_(decoder_start_token_id),
          eos_token_id_(eos_token_id), bos_token_id_(bos_token_id), pad_token_id_(pad_token_id),
          stream_(stream), tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id_str)) {
        if (!encoder_ || !encoder_->ok())
            throw std::runtime_error("BartPipeline: invalid encoder");
        if (!decoder_ || !decoder_->ok())
            throw std::runtime_error("BartPipeline: invalid decoder");
        if (!state_ || !state_->ok())
            throw std::runtime_error("BartPipeline: invalid state");

        cross_kv_bytes_ = static_cast<std::size_t>(max_source_length_) *
                          static_cast<std::size_t>(hidden_size_) * sizeof(float);
        cross_k_ptrs_.resize(static_cast<std::size_t>(num_decoder_layers_), nullptr);
        cross_v_ptrs_.resize(static_cast<std::size_t>(num_decoder_layers_), nullptr);
        for (int32_t i = 0; i < num_decoder_layers_; ++i) {
            cudaMalloc(&cross_k_ptrs_[static_cast<std::size_t>(i)], cross_kv_bytes_);
            cudaMalloc(&cross_v_ptrs_[static_cast<std::size_t>(i)], cross_kv_bytes_);
        }
    }

    ~BartPipeline() override {
        for (auto* ptr : cross_k_ptrs_) {
            if (ptr)
                cudaFree(ptr);
        }
        for (auto* ptr : cross_v_ptrs_) {
            if (ptr)
                cudaFree(ptr);
        }
    }

    TextResult generate(const std::string& prompt, const TextGenerationConfig& cfg) override {
        auto [padded, copy_len] = prepare_encoder_input(prompt);
        if (copy_len == 0)
            return TextResult{"[empty input]", {}};

        run_encoder(padded, copy_len);
        setup_cross_attention();

        int32_t max_tokens = cfg.max_new_tokens > 0 ? cfg.max_new_tokens : 128;
        auto output_ids = run_decoder(max_tokens);

        TextResult out;
        out.token_ids = std::move(output_ids);
        if (tokenizer_ && !out.token_ids.empty())
            out.text = tokenizer_->decode(out.token_ids);
        return out;
    }

    int32_t default_max_new_tokens() const override { return 128; }

  private:
    std::pair<std::vector<int32_t>, int32_t> prepare_encoder_input(const std::string& prompt) {
        std::vector<int32_t> ids;
        if (tokenizer_)
            ids = tokenizer_->encode(prompt);
        if (ids.empty())
            return {{}, 0};

        // Add BOS/EOS if the native tokenizer didn't
        if (bos_token_id_ >= 0 && (ids.empty() || ids.front() != bos_token_id_))
            ids.insert(ids.begin(), bos_token_id_);
        if (eos_token_id_ >= 0 && (ids.empty() || ids.back() != eos_token_id_))
            ids.push_back(eos_token_id_);

        int32_t copy_len = std::min(static_cast<int32_t>(ids.size()), max_source_length_);
        std::vector<int32_t> padded(static_cast<std::size_t>(max_source_length_), pad_token_id_);
        std::copy_n(ids.begin(), copy_len, padded.begin());
        return {std::move(padded), copy_len};
    }

    void run_encoder(const std::vector<int32_t>& padded_ids, int32_t actual_len) {
        actual_enc_len_ = actual_len;

        // Build attention mask: 0.0 for valid positions, -1e9 for padding
        encoder_attention_mask_.assign(static_cast<std::size_t>(max_source_length_), -1e9f);
        for (int32_t i = 0; i < actual_len; ++i)
            encoder_attention_mask_[static_cast<std::size_t>(i)] = 0.0f;

        TensorMap inputs;
        Tensor ids_tensor;
        ids_tensor.data = const_cast<int32_t*>(padded_ids.data());
        ids_tensor.shape = {max_source_length_};
        ids_tensor.dtype = DType::kInt32;
        inputs["input_ids"] = ids_tensor;

        // Provide attention mask if the encoder expects it
        Tensor mask_tensor;
        if (encoder_->has_input("attention_mask")) {
            mask_tensor.data = encoder_attention_mask_.data();
            mask_tensor.shape = {max_source_length_};
            mask_tensor.dtype = DType::kFloat32;
            inputs["attention_mask"] = mask_tensor;
        }

        encoder_->forward_async(inputs);
        encoder_->sync();
    }

    void setup_cross_attention() {
        void* enc_out = encoder_->device_ptr("encoder_output");
        if (!enc_out)
            throw std::runtime_error("BartPipeline: no encoder_output");
        for (int32_t i = 0; i < num_decoder_layers_; ++i) {
            auto idx = static_cast<std::size_t>(i);
            cudaMemcpy(cross_k_ptrs_[idx], enc_out, cross_kv_bytes_, cudaMemcpyDeviceToDevice);
            cudaMemcpy(cross_v_ptrs_[idx], enc_out, cross_kv_bytes_, cudaMemcpyDeviceToDevice);
        }
        for (int32_t i = 0; i < num_decoder_layers_; ++i) {
            std::string s = "_" + std::to_string(i);
            decoder_->bind_external("cross_k" + s, cross_k_ptrs_[static_cast<std::size_t>(i)]);
            decoder_->bind_external("cross_v" + s, cross_v_ptrs_[static_cast<std::size_t>(i)]);
        }
    }

    std::vector<int32_t> run_decoder(int32_t max_new_tokens) {
        state_->reset();
        state_->bind_to(*decoder_);
        std::vector<int32_t> output_ids;
        std::vector<float> logits;
        int32_t current_token = decoder_start_token_id_;
        for (int32_t step = 0; step < max_new_tokens; ++step) {
            run_decoder_step(current_token, logits);
            int32_t next = bart_select_argmax_token(logits);
            if (next == eos_token_id_)
                break;
            output_ids.push_back(next);
            current_token = next;
        }
        return output_ids;
    }

    void run_decoder_step(int32_t token_id, std::vector<float>& logits) {
        Tensor token_tensor;
        token_tensor.data = &token_id;
        token_tensor.shape = {1};
        token_tensor.dtype = DType::kInt32;
        TensorMap inputs;
        inputs["token_id"] = token_tensor;
        Tensor cross_mask_tensor;
        if (decoder_->has_input("cross_attention_mask")) {
            cross_mask_tensor.data = encoder_attention_mask_.data();
            cross_mask_tensor.shape = {max_source_length_};
            cross_mask_tensor.dtype = DType::kFloat32;
            inputs["cross_attention_mask"] = cross_mask_tensor;
        }
        state_->prepare_step(inputs);
        TensorMap outputs = decoder_->forward(inputs);
        auto it = outputs.find("logits");
        if (it == outputs.end())
            throw std::runtime_error("BartPipeline: no logits output");
        auto num = it->second.numel();
        logits.resize(static_cast<std::size_t>(num));
        std::memcpy(logits.data(), it->second.data, num * sizeof(float));
        state_->advance();
    }

    std::unique_ptr<ITrtModule> encoder_;
    std::unique_ptr<ITrtModule> decoder_;
    std::unique_ptr<BartInferenceState> state_;
    int32_t hidden_size_;
    int32_t num_decoder_layers_;
    int32_t max_source_length_;
    int32_t decoder_start_token_id_;
    int32_t eos_token_id_;
    int32_t bos_token_id_;
    int32_t pad_token_id_;
    int32_t actual_enc_len_{0};
    cudaStream_t stream_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    std::vector<void*> cross_k_ptrs_;
    std::vector<void*> cross_v_ptrs_;
    std::vector<float> encoder_attention_mask_;
    std::size_t cross_kv_bytes_{0};
};

ITask* create_bart(const FamilyContext& context) {
    const BartRuntimeConfig config =
        parse_runtime_config(bart::require_text_section(context.reader, "runtime.json"));
    bart::DistributedRuntimeGroup group =
        bart::initialize_tensor_parallel_group(config.tensor_parallel_size);
    auto encoder = bart::load_engine(
        context.backend, bart::require_section(context.reader, "encoder.plan"), "bart encoder");

    ModuleCreateOptions options;
    if (config.tensor_parallel_size > 1) {
        options.distributed_communicator = group.communicator;
        options.distributed_owner = group.owner;
    }
    const std::string decoder_section = config.tensor_parallel_size > 1
                                            ? tp_engine_section_name(group.rank)
                                            : std::string("engine.plan");
    const auto& plan = bart::require_section(context.reader, decoder_section);
    auto decoder = context.backend.create_module(plan.data(), plan.size(), options);
    if (decoder == nullptr || !decoder->ok())
        throw std::runtime_error("bart failed to load decoder");
    decoder->set_timing_label("bart decoder");
    const cudaStream_t stream = decoder->stream();
    const int32_t kv_dim = decoder_cache_row_width(*decoder, config);
    auto state = std::make_unique<BartKvCache>(config.decoder_layers, config.max_cache_length,
                                               kv_dim, stream);
    if (!state->ok())
        throw std::runtime_error("bart failed to create KV cache");

    return new BartPipeline(std::move(encoder), std::move(decoder), std::move(state),
                            config.hidden_size, config.decoder_layers, config.max_source_length,
                            config.decoder_start_token_id, config.eos_token_id, config.bos_token_id,
                            config.pad_token_id, stream, bart::create_tokenizer(context.reader),
                            std::string{});
}

} // namespace trtmc

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    return trtmc::create_bart(context);
}
