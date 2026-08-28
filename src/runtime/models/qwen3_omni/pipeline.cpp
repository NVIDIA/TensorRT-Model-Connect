/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/qwen3_omni/pipeline.h"

#include "runtime/models/qwen3_omni/argmax_kernel.h"
#include "runtime/models/qwen3_omni/omni_audio_plan.h"
#include "runtime/models/qwen3_omni/omni_thinker_plan.h"
#include "trtmc/tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace trtmc {

namespace {

using OmniClock = std::chrono::steady_clock;

double elapsed_ms(OmniClock::time_point start, OmniClock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace

// ─── OmniPipeline (TrtModule-based) ───

OmniPipeline::OmniPipeline(std::unique_ptr<TrtModule> thinker,
                           std::unique_ptr<Qwen3OmniInferenceState> thinker_state,
                           std::unique_ptr<TrtModule> code2wav, OmniConfig config,
                           cudaStream_t stream, std::shared_ptr<ITokenizer> tokenizer,
                           std::string model_id_str, std::unique_ptr<TrtModule> thinker_prefill)
    : thinker_(std::move(thinker)), thinker_prefill_(std::move(thinker_prefill)),
      thinker_state_(std::move(thinker_state)), code2wav_(std::move(code2wav)),
      config_(std::make_unique<OmniConfig>(std::move(config))), stream_(stream),
      tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id_str)),
      thinker_token_id_({1}, DType::kInt32, stream) {
    if (!thinker_ || !thinker_->ok())
        throw std::runtime_error("OmniPipeline: invalid thinker module");
    if (!thinker_state_ || !thinker_state_->ok())
        throw std::runtime_error("OmniPipeline: invalid thinker cache");
}

OmniPipeline::~OmniPipeline() = default;

namespace {

int32_t host_argmax(const TensorMap& outputs, int32_t& d2h_count) {
    auto it = outputs.find("logits");
    if (it == outputs.end())
        throw std::runtime_error("OmniPipeline thinker: no 'logits' output");
    const auto& logits = it->second;
    const auto count = static_cast<std::size_t>(logits.numel());
    if (count == 0)
        return 0;
    ++d2h_count;
    const auto* begin = static_cast<const float*>(logits.data);
    return static_cast<int32_t>(std::distance(begin, std::max_element(begin, begin + count)));
}

bool gather_prefill_kv(TrtModule& prefill, int32_t num_layers, std::vector<const void*>& present_k,
                       std::vector<const void*>& present_v) {
    present_k.resize(static_cast<std::size_t>(num_layers));
    present_v.resize(static_cast<std::size_t>(num_layers));
    for (int32_t layer = 0; layer < num_layers; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        present_k[index] = prefill.device_ptr("present_k_" + std::to_string(layer));
        present_v[index] = prefill.device_ptr("present_v_" + std::to_string(layer));
        if (present_k[index] == nullptr || present_v[index] == nullptr)
            return false;
    }
    return true;
}

Qwen3OmniKvCache* eligible_prefill_cache(TrtModule* prefill, Qwen3OmniInferenceState* state,
                                         int32_t sequence_length) {
    auto* cache = dynamic_cast<Qwen3OmniKvCache*>(state);
    if (prefill == nullptr || cache == nullptr || sequence_length <= 0 ||
        sequence_length > cache->max_length())
        return nullptr;
    return cache;
}

bool can_use_device_argmax(const TrtModule& module, const DeviceTensor& token_id,
                           int32_t vocab_size) {
    return module.device_ptr("logits") != nullptr &&
           module.tensor_dtype("logits") == DType::kFloat32 && vocab_size > 0 && token_id.ok();
}

} // namespace

int32_t OmniPipeline::run_thinker_step(int32_t token_id) {
    Tensor token_tensor;
    token_tensor.data = &token_id;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;

    TensorMap inputs;
    inputs["token_id"] = token_tensor;
    thinker_state_->prepare_step(inputs);

    ++thinker_stats_.decode_launches;
    const bool gpu_argmax =
        can_use_device_argmax(*thinker_, thinker_token_id_, config_->thinker_vocab_size);
    if (!gpu_argmax) {
        TensorMap outputs = thinker_->forward(inputs);
        thinker_state_->advance();
        return host_argmax(outputs, thinker_stats_.full_logits_d2h);
    }

    thinker_->forward_async(inputs);
    const auto* logits = static_cast<const float*>(thinker_->device_ptr("logits"));
    qwen3_omni_gpu_argmax(logits, config_->thinker_vocab_size,
                          static_cast<int32_t*>(thinker_token_id_.data()), stream_);
    thinker_state_->advance();
    cudaMemcpyAsync(&thinker_token_host_, thinker_token_id_.data(), sizeof(thinker_token_host_),
                    cudaMemcpyDeviceToHost, stream_);
    thinker_->sync();
    return thinker_token_host_;
}

bool OmniPipeline::run_thinker_prefill(const std::vector<int32_t>& input_ids, int32_t& next_token) {
    const auto sequence_length = static_cast<int32_t>(input_ids.size());
    auto* cache =
        eligible_prefill_cache(thinker_prefill_.get(), thinker_state_.get(), sequence_length);
    if (cache == nullptr)
        return false;

    std::vector<const void*> present_k;
    std::vector<const void*> present_v;
    if (!gather_prefill_kv(*thinker_prefill_, cache->num_layers(), present_k, present_v))
        return false;

    TensorMap inputs;
    inputs["token_id"] =
        Tensor{const_cast<int32_t*>(input_ids.data()), {sequence_length}, DType::kInt32};
    cache->bind_cache_inputs(*thinker_prefill_);
    thinker_state_->prepare_step(inputs, sequence_length);

    const bool gpu_argmax =
        can_use_device_argmax(*thinker_prefill_, thinker_token_id_, config_->thinker_vocab_size);
    ++thinker_stats_.prefill_launches;
    if (gpu_argmax) {
        thinker_prefill_->forward_async(inputs);
        const auto* logits = static_cast<const float*>(thinker_prefill_->device_ptr("logits"));
        qwen3_omni_gpu_argmax(logits, config_->thinker_vocab_size,
                              static_cast<int32_t*>(thinker_token_id_.data()), stream_);
        cache->write_prefill_kv(present_k, present_v, sequence_length);
        cudaMemcpyAsync(&thinker_token_host_, thinker_token_id_.data(), sizeof(thinker_token_host_),
                        cudaMemcpyDeviceToHost, stream_);
        thinker_prefill_->sync();
        next_token = thinker_token_host_;
        return true;
    }

    TensorMap outputs = thinker_prefill_->forward(inputs);
    cache->write_prefill_kv(present_k, present_v, sequence_length);
    next_token = host_argmax(outputs, thinker_stats_.full_logits_d2h);
    return true;
}

std::vector<int32_t> OmniPipeline::run_thinker(const std::vector<int32_t>& input_ids,
                                               int32_t max_tokens, int32_t eos_token_id) {
    thinker_stats_ = {};
    thinker_stats_.prompt_tokens = static_cast<int32_t>(input_ids.size());
    if (input_ids.empty() || max_tokens <= 0)
        return {};

    const int32_t cache_capacity = thinker_state_->max_length();
    if (!omni_thinker_request_fits_cache(input_ids.size(), max_tokens, cache_capacity)) {
        throw std::runtime_error(
            "OmniPipeline: prompt and generation exceed the native Thinker KV cache capacity");
    }

    thinker_state_->reset();
    thinker_state_->set_prompt_length(static_cast<int32_t>(input_ids.size()));
    thinker_state_->bind_to(*thinker_);

    int32_t next_token = 0;
    if (!run_thinker_prefill(input_ids, next_token)) {
        for (int32_t token : input_ids)
            next_token = run_thinker_step(token);
    }
    thinker_state_->mark_prefill_complete();

    std::vector<int32_t> output_ids;
    output_ids.reserve(static_cast<std::size_t>(max_tokens));

    for (int32_t step = 0; step < max_tokens; ++step) {
        const int32_t token = next_token;
        if (omni_thinker_should_stop(token, eos_token_id))
            break;
        output_ids.push_back(token);
        if (step + 1 < max_tokens)
            next_token = run_thinker_step(token);
    }

    std::cerr << "[trtmc] Omni Thinker: generated " << output_ids.size() << " text tokens"
              << std::endl;
    return output_ids;
}

std::vector<int32_t> OmniPipeline::generate_thinker_ids(const std::vector<int32_t>& input_ids,
                                                        int32_t max_tokens) {
    return run_thinker(input_ids, max_tokens, config_->thinker_eos_token_id);
}

TextResult OmniPipeline::generate(const std::string& prompt, const GenerateConfig& cfg) {
    if (!tokenizer_)
        throw std::runtime_error("OmniPipeline: native tokenizer is required for text generation");

    const auto input_ids = tokenizer_->encode(format_omni_chat_prompt(prompt));
    const int32_t max_tokens = cfg.max_new_tokens > 0 ? cfg.max_new_tokens : 128;
    const int32_t eos_token_id =
        cfg.eos_token_id >= 0 ? cfg.eos_token_id : config_->thinker_eos_token_id;
    auto output_ids = run_thinker(input_ids, max_tokens, eos_token_id);
    auto text = tokenizer_->decode(output_ids);
    return TextResult{std::move(text), std::move(output_ids)};
}

std::vector<float> OmniPipeline::run_code2wav(const std::vector<int32_t>& codec_tokens,
                                              int32_t n_codebooks, int32_t n_frames,
                                              double& code2wav_and_transfer_ms,
                                              double& output_materialization_ms) {
    if (!code2wav_) {
        throw std::runtime_error("OmniPipeline: required Code2Wav engine is unavailable");
    }

    const int32_t max_frames = config_->code2wav_max_frames;
    const int32_t actual_frames = std::min(n_frames, max_frames);

    std::vector<int32_t> input_codes =
        build_omni_code2wav_input_codes(codec_tokens, n_codebooks, max_frames, actual_frames);

    Tensor codes_tensor;
    codes_tensor.data = input_codes.data();
    codes_tensor.shape = {1, static_cast<int64_t>(n_codebooks), static_cast<int64_t>(max_frames)};
    codes_tensor.dtype = DType::kInt32;

    TensorMap inputs;
    inputs["codec_tokens"] = codes_tensor;

    const auto code2wav_start = OmniClock::now();
    TensorMap outputs = code2wav_->forward(inputs);
    code2wav_and_transfer_ms = elapsed_ms(code2wav_start, OmniClock::now());

    auto it = outputs.find("waveform");
    if (it == outputs.end()) {
        std::cerr << "[trtmc] Omni Code2Wav: no 'waveform' output" << std::endl;
        return {};
    }

    const auto& wt = it->second;
    const auto total_out = wt.numel();
    const auto copy_n =
        code2wav_output_samples(*config_, actual_frames, static_cast<std::size_t>(total_out));

    const auto materialize_start = OmniClock::now();
    std::vector<float> waveform(static_cast<std::size_t>(copy_n));
    std::memcpy(waveform.data(), wt.data, copy_n * sizeof(float));
    output_materialization_ms = elapsed_ms(materialize_start, OmniClock::now());

    std::cerr << "[trtmc] Omni Code2Wav: " << actual_frames << " frames -> " << waveform.size()
              << " samples" << std::endl;
    return waveform;
}

AudioResult OmniPipeline::generate_audio(const std::string& prompt, const GenerateConfig& cfg) {
    (void)prompt;
    (void)cfg;
    throw std::runtime_error(
        "OmniPipeline: native Qwen3-Omni Talker is unavailable; audio generation is disabled");
}

} // namespace trtmc
