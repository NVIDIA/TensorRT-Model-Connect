/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// M2M100Plugin: handles "m2m_100_seq2seq_encoder_decoder" strategy.
// Encoder-decoder text-to-text pipeline for M2M-100 and NLLB models.

#include "plugin_helpers.h"
#include "decode_runtime.h"
#include "kv_cache.h"
#include "request_tokens.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/runtime/pipeline_plugin.h"
#include "trtmc/runtime/pipeline_registry.h"
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

class M2M100Pipeline final : public IPipeline {
  public:
    M2M100Pipeline(std::unique_ptr<TrtModule> encoder, std::unique_ptr<TrtModule> decoder,
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

    TextResult generate(const std::string& prompt, const GenerateConfig& cfg) override {
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

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "M2M100Pipeline"; }

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
            // Preserve the legacy M2M-100 framing when no request-level
            // language contract is supplied.
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

    std::unique_ptr<TrtModule> encoder_;
    std::unique_ptr<TrtModule> decoder_;
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

class M2M100Plugin final : public IPipelinePlugin {
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

        auto enc_loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "vision_engine_plan"), "m2m_100 encoder", opts);

        ModuleCreateOptions decoder_opts = opts;
        if (tp_config.enabled) {
            decoder_opts.distributed_communicator = tp_group.communicator;
            decoder_opts.distributed_owner = tp_group.owner;
        }

        const std::string decoder_section =
            tp_config.enabled ? tp_engine_section_name(tp_group.rank) : std::string("engine_plan");
        auto dec_loaded =
            load_trt_module_from_plan(ctx.backend, find_section(ctx.bundle, decoder_section),
                                      "m2m_100 decoder", decoder_opts);
        int32_t decoder_layers = extract_json_int(json, "decoder_layers", ctx.config.num_layers);
        int32_t dl = (decoder_layers > 0) ? decoder_layers : ctx.config.num_layers;
        int32_t max_source_length = extract_json_int(
            json, "max_source_length",
            extract_json_int(json, "max_source_positions", ctx.config.max_cache_length));
        int32_t decoder_start_token_id = extract_json_int(json, "decoder_start_token_id", 2);
        int32_t eos_token_id = (ctx.config.id_eos >= 0) ? ctx.config.id_eos : 2;
        int32_t bos_token_id = (ctx.config.id_bos >= 0) ? ctx.config.id_bos : -1;
        int32_t pad_token_id = extract_json_int(json, "pad_token_id", 1);
        cudaStream_t stream = dec_loaded.module->stream();
        int32_t kv_dim = decoder_cache_row_width(*dec_loaded.module, ctx.config);
        int32_t max_cache = ctx.config.max_cache_length;
        auto state = std::make_unique<M2m100KvCache>(dl, max_cache, kv_dim, stream);
        if (!state->ok())
            throw std::runtime_error("Failed to create M2m100KvCache for M2M-100 decoder");
        auto tok = create_tokenizer_from_bundle(ctx.bundle);
        return std::make_unique<M2M100Pipeline>(
            std::move(enc_loaded.module), std::move(dec_loaded.module), std::move(state),
            ctx.config.hidden_size, dl, max_source_length, decoder_start_token_id, eos_token_id,
            bos_token_id, pad_token_id, stream, std::move(tok), ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_m2m_100_plugin, M2M100Plugin,
                                       "m2m_100_seq2seq_encoder_decoder");

} // namespace trtmc
