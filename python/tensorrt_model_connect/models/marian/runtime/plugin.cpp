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

#include "plugin_helpers.h"
#include "kv_cache.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc {

namespace {

struct TensorParallelRuntimeConfig {
    bool enabled{false};
    int32_t tp_size{1};
};

TensorParallelRuntimeConfig parse_tensor_parallel_runtime_config(const std::string& config_json) {
    TensorParallelRuntimeConfig cfg;
    cfg.tp_size = extract_json_int(config_json, "tensor_parallel_size", 1);
    const auto mode = extract_json_string(config_json, "tensor_parallel_mode", "single");
    cfg.enabled = (mode == "tensor_parallel" && cfg.tp_size > 1);
    return cfg;
}

std::string tp_engine_section_name(int32_t rank) {
    return "engine_plan_tp_rank" + std::to_string(rank);
}

int32_t dim_at(const std::vector<int64_t>& shape, int32_t dim) {
    if (dim < 0 || static_cast<std::size_t>(dim) >= shape.size())
        return -1;
    const auto value = shape[static_cast<std::size_t>(dim)];
    if (value <= 0 || value > std::numeric_limits<int32_t>::max())
        return -1;
    return static_cast<int32_t>(value);
}

int32_t decoder_cache_row_width(const TrtModule& module, const BaseConfig& config) {
    const int32_t from_engine = dim_at(module.tensor_shape("cache_k_0"), 1);
    return from_engine > 0 ? from_engine : compute_kv_dim(config);
}

} // namespace

// ---------------------------------------------------------------------------
// MarianPipeline: encoder-decoder machine translation
// ---------------------------------------------------------------------------

class MarianPipeline final : public IPipeline {
  public:
    MarianPipeline(std::unique_ptr<TrtModule> encoder, std::unique_ptr<TrtModule> decoder,
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

    TextResult generate(const std::string& prompt, const GenerateConfig& cfg) override {
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

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "MarianPipeline"; }

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
        std::vector<float> mask;
        cache_->build_attention_mask(mask);
        int32_t position = cache_->position();

        Tensor token_tensor;
        token_tensor.data = &token_id;
        token_tensor.shape = {1};
        token_tensor.dtype = DType::kInt32;

        Tensor position_tensor;
        position_tensor.data = &position;
        position_tensor.shape = {1};
        position_tensor.dtype = DType::kInt32;

        Tensor mask_tensor;
        mask_tensor.data = mask.data();
        mask_tensor.shape = {static_cast<int64_t>(mask.size())};
        mask_tensor.dtype = DType::kFloat32;

        TensorMap inputs;
        inputs["token_id"] = token_tensor;
        if (decoder_->has_input("position_id"))
            inputs["position_id"] = position_tensor;
        inputs["attention_mask"] = mask_tensor;

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

    std::unique_ptr<TrtModule> encoder_;
    std::unique_ptr<TrtModule> decoder_;
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

// ---------------------------------------------------------------------------
// MarianPlugin
// ---------------------------------------------------------------------------

class MarianPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        const auto& json = ctx.config_json;
        const auto tp_config = parse_tensor_parallel_runtime_config(json);
        DistributedRuntimeGroup tp_group;
        if (tp_config.enabled)
            tp_group = initialize_tensor_parallel_group(tp_config.tp_size);

        const auto* enc_plan = find_section(ctx.bundle, "vision_engine_plan");
        if (!enc_plan || enc_plan->empty())
            throw std::runtime_error("MarianPlugin: no encoder engine in bundle");
        auto enc_loaded = load_trt_module_from_plan(ctx.backend, enc_plan, "marian encoder", opts);

        ModuleCreateOptions decoder_opts = opts;
        if (tp_config.enabled) {
            decoder_opts.distributed_communicator = tp_group.communicator;
            decoder_opts.distributed_owner = tp_group.owner;
        }

        const std::string decoder_section =
            tp_config.enabled ? tp_engine_section_name(tp_group.rank) : std::string("engine_plan");
        auto dec_loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, decoder_section), "marian decoder", decoder_opts);

        int32_t decoder_layers =
            extract_json_int(json, "decoder_layers",
                             extract_json_int(json, "num_decoder_layers", ctx.config.num_layers));
        int32_t dl = (decoder_layers > 0) ? decoder_layers : ctx.config.num_layers;
        int32_t max_enc_seq_len =
            extract_json_int(json, "max_source_positions", ctx.config.max_cache_length);
        int32_t decoder_start_token_id = extract_json_int(json, "decoder_start_token_id", 0);
        int32_t eos_token_id = extract_json_int(json, "eos_token_id", 0);
        int32_t pad_token_id = extract_json_int(json, "pad_token_id", eos_token_id);

        cudaStream_t stream = dec_loaded.module->stream();
        int32_t kv_dim = decoder_cache_row_width(*dec_loaded.module, ctx.config);
        auto cache =
            std::make_unique<MarianKvCache>(dl, ctx.config.max_cache_length, kv_dim, stream);
        if (!cache->ok())
            throw std::runtime_error("MarianPlugin: failed to create MarianKvCache");

        auto tok = create_tokenizer_from_bundle(ctx.bundle);
        if (!tok) {
            std::cerr << "[trtmc] MarianPlugin: tokenizer creation failed" << std::endl;
        }

        return std::make_unique<MarianPipeline>(
            std::move(enc_loaded.module), std::move(dec_loaded.module), std::move(cache),
            ctx.config.hidden_size, dl, max_enc_seq_len, ctx.config.vocab_size,
            decoder_start_token_id, eos_token_id, pad_token_id, stream, std::move(tok),
            ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_marian_plugin, MarianPlugin, "marian_translation");

} // namespace trtmc
