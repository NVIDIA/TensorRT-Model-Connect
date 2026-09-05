/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// M2M100Plugin: handles "m2m_100_seq2seq_encoder_decoder" strategy.
// Encoder-decoder text-to-text pipeline for M2M-100 and NLLB models.

#include "families/m2m_100/runtime/decode_runtime.h"
#include "families/m2m_100/runtime/kv_cache.h"
#include "families/m2m_100/runtime/plugin_helpers.h"
#include "families/m2m_100/runtime/request_tokens.h"
#include "families/m2m_100/runtime/runtime_config.h"
#include "families/m2m_100/runtime/tokenizer.h"
#include "trtmc/runtime/family_factory.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <cuda_runtime_api.h>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc {

namespace {

struct M2M100RuntimeConfig {
    int32_t hidden_size;
    int32_t encoder_layers;
    int32_t decoder_layers;
    int32_t max_cache_length;
    int32_t num_attention_heads;
    int32_t num_key_value_heads;
    int32_t head_dim;
    int32_t max_source_length;
    int32_t vocab_size;
    int32_t decoder_start_token_id;
    int32_t eos_token_id;
    int32_t bos_token_id;
    int32_t pad_token_id;
    bool scale_embedding;
    bool has_vision_engine;
    bool is_encoder_decoder;
    std::string precision;
    std::string decoder_engine_layout;
};

constexpr std::array<std::string_view, 17> kRuntimeFields = {
    "vocab_size",
    "hidden_size",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "num_attention_heads",
    "num_key_value_heads",
    "encoder_layers",
    "decoder_layers",
    "max_source_length",
    "decoder_start_token_id",
    "scale_embedding",
    "has_vision_engine",
    "is_encoder_decoder",
    "precision",
    "max_cache_length",
    "decoder_engine_layout",
};

M2M100RuntimeConfig parse_runtime_config(const std::string& text) {
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("m2m_100 invalid runtime.json: " + std::string(error.what()));
    }
    if (!json.is_object() || json.size() != kRuntimeFields.size())
        throw std::runtime_error("m2m_100 runtime.json has an unexpected field set");
    for (const auto field : kRuntimeFields) {
        if (!json.contains(std::string(field)))
            throw std::runtime_error("m2m_100 runtime.json missing '" + std::string(field) + "'");
    }
    const int32_t hidden_size = json.at("hidden_size").get<int32_t>();
    const int32_t num_attention_heads = json.at("num_attention_heads").get<int32_t>();
    if (hidden_size <= 0 || num_attention_heads <= 0 || hidden_size % num_attention_heads != 0)
        throw std::runtime_error("m2m_100 runtime.json has invalid attention geometry");
    M2M100RuntimeConfig config{
        hidden_size,
        json.at("encoder_layers").get<int32_t>(),
        json.at("decoder_layers").get<int32_t>(),
        json.at("max_cache_length").get<int32_t>(),
        num_attention_heads,
        json.at("num_key_value_heads").get<int32_t>(),
        hidden_size / num_attention_heads,
        json.at("max_source_length").get<int32_t>(),
        json.at("vocab_size").get<int32_t>(),
        json.at("decoder_start_token_id").get<int32_t>(),
        json.at("eos_token_id").get<int32_t>(),
        json.at("bos_token_id").get<int32_t>(),
        json.at("pad_token_id").get<int32_t>(),
        json.at("scale_embedding").get<bool>(),
        json.at("has_vision_engine").get<bool>(),
        json.at("is_encoder_decoder").get<bool>(),
        json.at("precision").get<std::string>(),
        json.at("decoder_engine_layout").get<std::string>(),
    };
    if (config.hidden_size <= 0 || config.encoder_layers <= 0 || config.decoder_layers <= 0 ||
        config.max_cache_length <= 0 || config.hidden_size % config.num_attention_heads != 0 ||
        config.num_attention_heads <= 0 || config.num_key_value_heads <= 0 ||
        config.head_dim <= 0 || config.max_source_length <= 0 || config.vocab_size <= 0 ||
        !config.has_vision_engine || !config.is_encoder_decoder ||
        config.decoder_engine_layout != "single" ||
        (config.precision != "fp16" && config.precision != "fp32")) {
        throw std::runtime_error("m2m_100 runtime.json contains invalid dimensions");
    }
    return config;
}

int32_t dim_at(const std::vector<int64_t>& shape, int32_t dim) {
    if (dim < 0 || static_cast<std::size_t>(dim) >= shape.size())
        return -1;
    const auto value = shape[static_cast<std::size_t>(dim)];
    if (value <= 0 || value > std::numeric_limits<int32_t>::max())
        return -1;
    return static_cast<int32_t>(value);
}

int32_t decoder_cache_row_width(const ITrtModule& module, const M2M100RuntimeConfig& config) {
    const int32_t from_engine = dim_at(module.tensor_shape("cache_k_0"), 1);
    return from_engine > 0 ? from_engine : config.num_key_value_heads * config.head_dim;
}

} // namespace

namespace m2m_100 {

void validate_runtime_config_json(std::string_view json) {
    (void)parse_runtime_config(std::string(json));
}

} // namespace m2m_100

class M2M100Pipeline final : public ITextGeneration {
  public:
    M2M100Pipeline(std::unique_ptr<ITrtModule> encoder, std::unique_ptr<ITrtModule> decoder,
                   std::unique_ptr<M2m100InferenceState> state, int32_t hidden_size,
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
            throw std::runtime_error("M2M100Pipeline: invalid encoder");
        if (!decoder_ || !decoder_->ok())
            throw std::runtime_error("M2M100Pipeline: invalid decoder");
        if (!state_ || !state_->ok())
            throw std::runtime_error("M2M100Pipeline: invalid state");

        cross_kv_bytes_ = static_cast<std::size_t>(max_source_length_) *
                          static_cast<std::size_t>(hidden_size_) * sizeof(float);
        cross_k_ptrs_.resize(static_cast<std::size_t>(num_decoder_layers_), nullptr);
        cross_v_ptrs_.resize(static_cast<std::size_t>(num_decoder_layers_), nullptr);
        for (int32_t i = 0; i < num_decoder_layers_; ++i) {
            cudaMalloc(&cross_k_ptrs_[static_cast<std::size_t>(i)], cross_kv_bytes_);
            cudaMalloc(&cross_v_ptrs_[static_cast<std::size_t>(i)], cross_kv_bytes_);
        }
    }

    ~M2M100Pipeline() override {
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
        auto [padded, copy_len] = prepare_encoder_input(prompt, cfg.source_language_token_id);
        if (copy_len == 0)
            return TextResult{"[empty input]", {}};

        run_encoder(padded, copy_len);
        setup_cross_attention();

        int32_t max_tokens = cfg.max_new_tokens > 0 ? cfg.max_new_tokens : 128;
        auto output_ids = run_decoder(max_tokens, cfg.forced_bos_token_id);

        TextResult out;
        out.token_ids = std::move(output_ids);
        if (tokenizer_ && !out.token_ids.empty())
            out.text = tokenizer_->decode(out.token_ids);
        return out;
    }

    int32_t default_max_new_tokens() const override { return 128; }

  private:
    std::pair<std::vector<int32_t>, int32_t>
    prepare_encoder_input(const std::string& prompt, int32_t source_language_token_id) {
        std::vector<int32_t> ids;
        if (tokenizer_)
            ids = tokenizer_->encode(prompt);
        if (ids.empty())
            return {{}, 0};

        if (source_language_token_id >= 0) {
            // NLLB tokenizers frame source input as: text, EOS, language.
            // Replace an unset/unknown language suffix instead of adding a
            // second special-token frame.
            m2m_100_apply_source_language_token(ids, eos_token_id_, source_language_token_id);
        } else {
            // Use M2M-100 BOS/EOS framing when no request-level language token
            // is supplied.
            if (bos_token_id_ >= 0 && (ids.empty() || ids.front() != bos_token_id_))
                ids.insert(ids.begin(), bos_token_id_);
            if (eos_token_id_ >= 0 && (ids.empty() || ids.back() != eos_token_id_))
                ids.push_back(eos_token_id_);
        }

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
            throw std::runtime_error("M2M100Pipeline: no encoder_output");
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

    std::vector<int32_t> run_decoder(int32_t max_new_tokens, int32_t forced_bos_token_id) {
        state_->reset();
        state_->bind_to(*decoder_);
        std::vector<int32_t> output_ids;
        std::vector<float> logits;
        int32_t current_token = decoder_start_token_id_;
        for (int32_t step = 0; step < max_new_tokens; ++step) {
            run_decoder_step(current_token, logits);
            int32_t next = m2m_100_apply_forced_bos_token(m2m_100_select_argmax_token(logits), step,
                                                          forced_bos_token_id);
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
            throw std::runtime_error("M2M100Pipeline: no logits output");
        auto num = it->second.numel();
        logits.resize(static_cast<std::size_t>(num));
        std::memcpy(logits.data(), it->second.data, num * sizeof(float));
        state_->advance();
    }

    std::unique_ptr<ITrtModule> encoder_;
    std::unique_ptr<ITrtModule> decoder_;
    std::unique_ptr<M2m100InferenceState> state_;
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

ITask* create_m2m_100(const FamilyContext& context) {
    const M2M100RuntimeConfig config =
        parse_runtime_config(m2m_100::require_text_section(context.reader, "runtime.json"));
    auto encoder = m2m_100::load_engine(context.backend,
                                        m2m_100::require_section(context.reader, "encoder.plan"),
                                        "m2m_100 encoder");
    auto decoder = m2m_100::load_engine(context.backend,
                                        m2m_100::require_section(context.reader, "engine.plan"),
                                        "m2m_100 decoder");

    cudaStream_t stream = decoder->stream();
    const int32_t kv_dim = decoder_cache_row_width(*decoder, config);
    auto state = std::make_unique<M2m100KvCache>(config.decoder_layers, config.max_cache_length,
                                                 kv_dim, stream);
    if (!state->ok())
        throw std::runtime_error("m2m_100 failed to create KV cache");
    auto tokenizer = m2m_100::create_tokenizer(context.reader);

    return new M2M100Pipeline(std::move(encoder), std::move(decoder), std::move(state),
                              config.hidden_size, config.decoder_layers, config.max_source_length,
                              config.decoder_start_token_id, config.eos_token_id,
                              config.bos_token_id, config.pad_token_id, stream,
                              std::move(tokenizer), std::string{});
}

} // namespace trtmc

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("m2m_100 does not support --kv-cache-size");
    return trtmc::create_m2m_100(context);
}
