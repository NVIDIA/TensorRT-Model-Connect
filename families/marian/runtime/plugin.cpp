/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// MarianPlugin: handles "marian_translation" strategy for Marian MT models.
// Encoder-decoder pipeline for machine translation.
//
// Pipeline:
//   1. Tokenize input text
//   2. Run encoder on input tokens -> encoder_hidden_states
//   3. Run decoder autoregressively with cross-attention to encoder output
//   4. Detokenize output

#include "families/marian/runtime/distributed_runtime.h"
#include "families/marian/runtime/kv_cache.h"
#include "families/marian/runtime/plugin_helpers.h"
#include "families/marian/runtime/runtime_config.h"
#include "families/marian/runtime/tokenizer.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

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

struct MarianRuntimeConfig {
    int32_t hidden_size;
    int32_t encoder_layers;
    int32_t decoder_layers;
    int32_t max_cache_length;
    int32_t encoder_attention_heads;
    int32_t decoder_attention_heads;
    int32_t head_dim;
    int32_t max_source_positions;
    int32_t encoder_ffn_dim;
    int32_t decoder_ffn_dim;
    int32_t vocab_size;
    int32_t decoder_start_token_id;
    int32_t eos_token_id;
    int32_t pad_token_id;
    bool has_vision_engine;
    bool scale_embedding;
    std::string activation_function;
    std::string precision;
    int32_t tensor_parallel_size;
    std::string tensor_parallel_mode;
    std::string decoder_engine_layout;
};

constexpr std::array<std::string_view, 21> kRuntimeFields = {
    "vocab_size",
    "hidden_size",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "max_source_positions",
    "encoder_layers",
    "decoder_layers",
    "encoder_ffn_dim",
    "decoder_ffn_dim",
    "encoder_attention_heads",
    "decoder_attention_heads",
    "has_vision_engine",
    "decoder_start_token_id",
    "scale_embedding",
    "activation_function",
    "precision",
    "max_cache_length",
    "decoder_engine_layout",
    "tensor_parallel_size",
    "tensor_parallel_mode",
};

MarianRuntimeConfig parse_runtime_config(const std::string& text) {
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("marian invalid runtime.json: " + std::string(error.what()));
    }
    if (!json.is_object() || json.size() != kRuntimeFields.size())
        throw std::runtime_error("marian runtime.json has an unexpected field set");
    for (const auto field : kRuntimeFields) {
        if (!json.contains(std::string(field)))
            throw std::runtime_error("marian runtime.json missing '" + std::string(field) + "'");
    }
    const int32_t hidden_size = json.at("hidden_size").get<int32_t>();
    const int32_t decoder_heads = json.at("decoder_attention_heads").get<int32_t>();
    if (hidden_size <= 0 || decoder_heads <= 0 || hidden_size % decoder_heads != 0)
        throw std::runtime_error("marian runtime.json has invalid attention geometry");
    MarianRuntimeConfig config{
        hidden_size,
        json.at("encoder_layers").get<int32_t>(),
        json.at("decoder_layers").get<int32_t>(),
        json.at("max_cache_length").get<int32_t>(),
        json.at("encoder_attention_heads").get<int32_t>(),
        decoder_heads,
        hidden_size / decoder_heads,
        json.at("max_source_positions").get<int32_t>(),
        json.at("encoder_ffn_dim").get<int32_t>(),
        json.at("decoder_ffn_dim").get<int32_t>(),
        json.at("vocab_size").get<int32_t>(),
        json.at("decoder_start_token_id").get<int32_t>(),
        json.at("eos_token_id").get<int32_t>(),
        json.at("pad_token_id").get<int32_t>(),
        json.at("has_vision_engine").get<bool>(),
        json.at("scale_embedding").get<bool>(),
        json.at("activation_function").get<std::string>(),
        json.at("precision").get<std::string>(),
        json.at("tensor_parallel_size").get<int32_t>(),
        json.at("tensor_parallel_mode").get<std::string>(),
        json.at("decoder_engine_layout").get<std::string>(),
    };
    if (config.hidden_size <= 0 || config.encoder_layers <= 0 || config.decoder_layers <= 0 ||
        config.max_cache_length <= 0 || config.encoder_attention_heads <= 0 ||
        config.decoder_attention_heads <= 0 || config.head_dim <= 0 ||
        config.max_source_positions <= 0 || config.vocab_size <= 0 || config.encoder_ffn_dim <= 0 ||
        config.decoder_ffn_dim <= 0 || config.tensor_parallel_size <= 0 ||
        !config.has_vision_engine || config.decoder_engine_layout != "single" ||
        config.tensor_parallel_mode !=
            (config.tensor_parallel_size > 1 ? "tensor_parallel" : "single") ||
        (config.precision != "fp16" && config.precision != "fp32")) {
        throw std::runtime_error("marian runtime.json contains invalid dimensions");
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

int32_t decoder_cache_row_width(const ITrtModule& module, const MarianRuntimeConfig& config) {
    const int32_t from_engine = dim_at(module.tensor_shape("cache_k_0"), 1);
    return from_engine > 0 ? from_engine : config.hidden_size;
}

} // namespace

namespace marian {

void validate_runtime_config_json(std::string_view json) {
    (void)parse_runtime_config(std::string(json));
}

} // namespace marian

// ---------------------------------------------------------------------------
// MarianPipeline: encoder-decoder machine translation
// ---------------------------------------------------------------------------

class MarianPipeline final : public ITextGeneration {
  public:
    MarianPipeline(std::unique_ptr<ITrtModule> encoder, std::unique_ptr<ITrtModule> decoder,
                   std::unique_ptr<MarianKvCache> cache, int32_t hidden_size,
                   int32_t num_decoder_layers, int32_t max_enc_seq_len, int32_t vocab_size,
                   int32_t decoder_start_token_id, int32_t eos_token_id, int32_t pad_token_id,
                   cudaStream_t stream, std::shared_ptr<ITokenizer> tokenizer,
                   std::string model_id_str)
        : encoder_(std::move(encoder)), decoder_(std::move(decoder)), cache_(std::move(cache)),
          hidden_size_(hidden_size), num_decoder_layers_(num_decoder_layers),
          max_enc_seq_len_(max_enc_seq_len), vocab_size_(vocab_size),
          decoder_start_token_id_(decoder_start_token_id), eos_token_id_(eos_token_id),
          pad_token_id_(pad_token_id), stream_(stream), tokenizer_(std::move(tokenizer)),
          model_id_(std::move(model_id_str)) {
        cross_kv_bytes_ = static_cast<size_t>(max_enc_seq_len_) *
                          static_cast<size_t>(hidden_size_) * sizeof(float);
        for (int32_t i = 0; i < num_decoder_layers_; ++i) {
            void* dk = nullptr;
            void* dv = nullptr;
            cudaMalloc(&dk, cross_kv_bytes_);
            cudaMalloc(&dv, cross_kv_bytes_);
            cross_k_ptrs_.push_back(dk);
            cross_v_ptrs_.push_back(dv);
        }
    }

    ~MarianPipeline() override {
        for (auto* p : cross_k_ptrs_)
            cudaFree(p);
        for (auto* p : cross_v_ptrs_)
            cudaFree(p);
        if (enc_mask_device_)
            cudaFree(enc_mask_device_);
    }

    TextResult generate(const std::string& prompt, const TextGenerationConfig& cfg) override {
        if (!tokenizer_)
            throw std::runtime_error("MarianPipeline: no tokenizer configured");

        auto input_ids = tokenizer_->encode(prompt);
        // Append EOS token if not already present (Marian convention)
        if (input_ids.empty() || input_ids.back() != eos_token_id_)
            input_ids.push_back(eos_token_id_);

        run_encoder(input_ids);
        setup_cross_attention();

        int32_t max_new = (cfg.max_new_tokens > 0) ? cfg.max_new_tokens : 128;
        int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : eos_token_id_;
        auto output_ids = run_decoder(max_new, eos);

        std::string text = tokenizer_->decode(output_ids);
        return TextResult{std::move(text), std::move(output_ids)};
    }

    int32_t default_max_new_tokens() const override { return 128; }

  private:
    void run_encoder(const std::vector<int32_t>& input_ids) {
        std::vector<int32_t> padded(static_cast<size_t>(max_enc_seq_len_), pad_token_id_);
        size_t copy_len = std::min(input_ids.size(), static_cast<size_t>(max_enc_seq_len_));
        std::memcpy(padded.data(), input_ids.data(), copy_len * sizeof(int32_t));
        actual_enc_len_ = static_cast<int32_t>(copy_len);

        std::vector<float> enc_mask(static_cast<size_t>(max_enc_seq_len_), -1e9f);
        for (int32_t i = 0; i < actual_enc_len_; ++i)
            enc_mask[static_cast<size_t>(i)] = 0.0f;

        Tensor ids_tensor;
        ids_tensor.data = padded.data();
        ids_tensor.shape = {max_enc_seq_len_};
        ids_tensor.dtype = DType::kInt32;

        Tensor mask_tensor;
        mask_tensor.data = enc_mask.data();
        mask_tensor.shape = {max_enc_seq_len_};
        mask_tensor.dtype = DType::kFloat32;

        TensorMap inputs;
        inputs["input_ids"] = ids_tensor;
        inputs["attention_mask"] = mask_tensor;

        auto outputs = encoder_->forward(inputs);

        auto it = outputs.find("encoder_output");
        if (it == outputs.end())
            throw std::runtime_error("MarianPipeline: encoder has no 'encoder_output'");

        auto& enc_out = it->second;
        size_t enc_bytes = static_cast<size_t>(enc_out.numel()) * sizeof(float);
        encoder_output_host_.resize(enc_out.numel());
        std::memcpy(encoder_output_host_.data(), enc_out.data, enc_bytes);
    }

    void setup_cross_attention() {
        for (int32_t i = 0; i < num_decoder_layers_; ++i) {
            cudaMemcpyAsync(cross_k_ptrs_[static_cast<size_t>(i)], encoder_output_host_.data(),
                            cross_kv_bytes_, cudaMemcpyHostToDevice, stream_);
            cudaMemcpyAsync(cross_v_ptrs_[static_cast<size_t>(i)], encoder_output_host_.data(),
                            cross_kv_bytes_, cudaMemcpyHostToDevice, stream_);
        }
        cudaStreamSynchronize(stream_);

        std::vector<float> enc_mask_host(static_cast<size_t>(max_enc_seq_len_), -1e9f);
        for (int32_t i = 0; i < actual_enc_len_; ++i)
            enc_mask_host[static_cast<size_t>(i)] = 0.0f;
        size_t mask_bytes = static_cast<size_t>(max_enc_seq_len_) * sizeof(float);
        if (!enc_mask_device_)
            cudaMalloc(&enc_mask_device_, mask_bytes);
        cudaMemcpyAsync(enc_mask_device_, enc_mask_host.data(), mask_bytes, cudaMemcpyHostToDevice,
                        stream_);
        cudaStreamSynchronize(stream_);

        for (int32_t i = 0; i < num_decoder_layers_; ++i) {
            std::string ck_name = "cross_k_" + std::to_string(i);
            std::string cv_name = "cross_v_" + std::to_string(i);
            decoder_->bind_external(ck_name, cross_k_ptrs_[static_cast<size_t>(i)]);
            decoder_->bind_external(cv_name, cross_v_ptrs_[static_cast<size_t>(i)]);
        }
        decoder_->bind_external("encoder_mask", enc_mask_device_);
    }

    std::vector<int32_t> run_decoder(int32_t max_new_tokens, int32_t eos_id) {
        cache_->reset();
        cache_->bind_to(*decoder_);

        std::vector<float> logits;
        std::vector<int32_t> output_ids;

        int32_t current_token = decoder_start_token_id_;
        run_decoder_step(current_token, logits);

        for (int32_t step = 0; step < max_new_tokens; ++step) {
            int32_t next_token = argmax(logits);
            output_ids.push_back(next_token);

            if (next_token == eos_id)
                break;

            run_decoder_step(next_token, logits);
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
        cache_->prepare_step(inputs);

        TensorMap outputs = decoder_->forward(inputs);

        auto it = outputs.find("logits");
        if (it == outputs.end())
            throw std::runtime_error("MarianPipeline: no 'logits' output");

        const auto& logits_tensor = it->second;
        auto num_logits = logits_tensor.numel();
        logits.resize(static_cast<size_t>(num_logits));
        std::memcpy(logits.data(), logits_tensor.data,
                    static_cast<size_t>(num_logits) * sizeof(float));

        cache_->advance();
    }

    static int32_t argmax(const std::vector<float>& logits) {
        if (logits.empty())
            return 0;
        return static_cast<int32_t>(
            std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    }

    std::unique_ptr<ITrtModule> encoder_;
    std::unique_ptr<ITrtModule> decoder_;
    std::unique_ptr<MarianKvCache> cache_;
    int32_t hidden_size_;
    int32_t num_decoder_layers_;
    int32_t max_enc_seq_len_;
    int32_t vocab_size_;
    int32_t decoder_start_token_id_;
    int32_t eos_token_id_;
    int32_t pad_token_id_;
    cudaStream_t stream_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;

    std::vector<void*> cross_k_ptrs_;
    std::vector<void*> cross_v_ptrs_;
    size_t cross_kv_bytes_{0};

    std::vector<float> encoder_output_host_;
    int32_t actual_enc_len_{0};
    void* enc_mask_device_{nullptr};
};

ITask* create_marian(const FamilyContext& context) {
    const MarianRuntimeConfig config =
        parse_runtime_config(marian::require_text_section(context.reader, "runtime.json"));
    marian::DistributedRuntimeGroup group =
        marian::initialize_tensor_parallel_group(config.tensor_parallel_size);

    auto encoder = marian::load_engine(
        context.backend, marian::require_section(context.reader, "encoder.plan"), "marian encoder");
    ModuleCreateOptions decoder_options;
    if (config.tensor_parallel_size > 1) {
        decoder_options.distributed_communicator = group.communicator;
        decoder_options.distributed_owner = group.owner;
    }
    const std::string decoder_section = config.tensor_parallel_size > 1
                                            ? tp_engine_section_name(group.rank)
                                            : std::string("engine.plan");
    const auto& decoder_plan = marian::require_section(context.reader, decoder_section);
    auto decoder =
        context.backend.create_module(decoder_plan.data(), decoder_plan.size(), decoder_options);
    if (decoder == nullptr || !decoder->ok())
        throw std::runtime_error("marian failed to load decoder");
    decoder->set_timing_label("marian decoder");

    cudaStream_t stream = decoder->stream();
    const int32_t kv_dim = decoder_cache_row_width(*decoder, config);
    auto cache = std::make_unique<MarianKvCache>(config.decoder_layers, config.max_cache_length,
                                                 kv_dim, stream);
    if (!cache->ok())
        throw std::runtime_error("marian failed to create KV cache");
    auto tokenizer = marian::create_tokenizer(context.reader);

    return new MarianPipeline(std::move(encoder), std::move(decoder), std::move(cache),
                              config.hidden_size, config.decoder_layers,
                              config.max_source_positions, config.vocab_size,
                              config.decoder_start_token_id, config.eos_token_id,
                              config.pad_token_id, stream, std::move(tokenizer), std::string{});
}

} // namespace trtmc

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("marian does not support --kv-cache-size");
    return trtmc::create_marian(context);
}
