/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/nemotron_h/pipeline.h"

#include "runtime/models/nemotron_h/chat_templates.h"
#include "runtime/models/nemotron_h/recurrent_generation_plan.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {
using SteadyClock = std::chrono::steady_clock;
using TimePoint = SteadyClock::time_point;
inline double elapsed_ms(TimePoint start, TimePoint end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}
} // namespace

namespace trtmc {

namespace {

RecurrentGenConfig normalize_eos_token_ids(RecurrentGenConfig config) {
    if (config.id_eos_ids.empty() && config.id_eos >= 0)
        config.id_eos_ids.push_back(config.id_eos);
    if (!config.id_eos_ids.empty())
        config.id_eos = config.id_eos_ids.front();
    return config;
}

} // namespace

RecurrentPipeline::RecurrentPipeline(std::unique_ptr<TrtModule> decoder,
                                     std::unique_ptr<NemotronHInferenceState> state,
                                     RecurrentGenConfig config, cudaStream_t stream,
                                     const char* name, std::shared_ptr<ITokenizer> tokenizer,
                                     std::string model_id_str,
                                     std::unique_ptr<NemotronHISampler> sampler)
    : decoder_(std::move(decoder)), state_(std::move(state)),
      config_(normalize_eos_token_ids(std::move(config))), stream_(stream), name_(name),
      tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id_str)),
      sampler_(std::move(sampler)) {
    if (!decoder_ || !decoder_->ok())
        throw std::runtime_error(std::string(name_) + ": invalid decoder module");
}

static std::vector<int32_t> encode_prompt(const ITokenizer& tokenizer,
                                          const RecurrentGenConfig& config,
                                          const std::string& prompt, const GenerateConfig& cfg) {
    std::string effective = prompt;
    bool templated = false;
    if (cfg.use_chat_template && !config.chat_template_format.empty()) {
        effective = nemotron_h_apply_chat_template(config.chat_template_format, prompt,
                                                   cfg.enable_thinking);
        templated = true;
    }

    auto ids = tokenizer.encode(effective);
    if (templated && ids.size() >= 2 && config.id_bos >= 0 && ids[0] == config.id_bos &&
        ids[1] == config.id_bos) {
        ids.erase(ids.begin());
    }
    return ids;
}

TextResult RecurrentPipeline::generate(const std::string& prompt, const GenerateConfig& cfg) {
    if (!tokenizer_)
        throw std::runtime_error(std::string(name_) + ": no tokenizer configured");

    auto input_ids = encode_prompt(*tokenizer_, config_, prompt, cfg);
    int32_t max_new = (cfg.max_new_tokens > 0) ? cfg.max_new_tokens : 128;

    auto sp = nemotron_h_sampling_params_from_config(cfg, config_.id_eos_ids);
    auto output_ids = generate_from_ids(input_ids, max_new, sp);

    std::vector<int32_t> new_tokens(
        output_ids.begin() + static_cast<std::ptrdiff_t>(input_ids.size()), output_ids.end());
    std::string text = tokenizer_->decode(new_tokens);

    return TextResult{std::move(text), std::move(new_tokens)};
}

RecurrentPipeline::GenerationResult
RecurrentPipeline::generate_ids(const std::vector<int32_t>& input_ids, const GenerateConfig& cfg) {
    int32_t max_new = cfg.max_new_tokens;
    auto sp = nemotron_h_sampling_params_from_config(cfg, config_.id_eos_ids);
    return GenerationResult{generate_from_ids(input_ids, max_new, sp)};
}

std::vector<int32_t> RecurrentPipeline::generate_from_ids(const std::vector<int32_t>& input_ids,
                                                          int32_t max_new_tokens,
                                                          const NemotronHSamplingParams& params) {
    if (max_new_tokens == 0 || input_ids.empty())
        return input_ids;

    // Create a per-call sampler if none was injected at construction time.
    NemotronHISampler* active_sampler = sampler_.get();
    std::unique_ptr<NemotronHISampler> local_sampler;
    if (!active_sampler) {
        local_sampler = create_nemotron_h_sampler(params);
        active_sampler = local_sampler.get();
    }
    active_sampler->reset();

    state_->reset();
    state_->bind_to(*decoder_);

    prof_prepare_ms_ = prof_forward_ms_ = prof_logits_copy_ms_ = prof_advance_ms_ = 0;
    prof_logits_copy_bytes_ = 0;
    prof_logits_copies_ = 0;
    prof_steps_ = 0;

    std::vector<float> logits;

    // ── Prefill phase ──
    auto t_prefill_start = SteadyClock::now();
    for (std::size_t i = 0; i < input_ids.size(); ++i)
        run_step(input_ids[i], logits,
                 nemotron_h_recurrent::prefill_step_needs_logits(i, input_ids.size()));
    auto t_prefill_end = SteadyClock::now();

    // ── Decode phase ──
    std::vector<int32_t> output = input_ids;
    const int32_t vocab_size = static_cast<int32_t>(logits.size());
    int32_t decode_steps = 0;

    auto t_decode_start = SteadyClock::now();
    for (int32_t step = 0; step < max_new_tokens; ++step) {
        NemotronHSampleResult result = active_sampler->sample(logits.data(), vocab_size, params);
        output.push_back(result.token_id);
        if (result.is_eos)
            break;
        run_step(result.token_id, logits, true);
        ++decode_steps;
    }
    auto t_decode_end = SteadyClock::now();

    report_timing(t_prefill_start, t_prefill_end, t_decode_start, t_decode_end,
                  static_cast<int>(input_ids.size()), decode_steps);

    return output;
}

void RecurrentPipeline::report_timing(SteadyClock::time_point t_prefill_start,
                                      SteadyClock::time_point t_prefill_end,
                                      SteadyClock::time_point t_decode_start,
                                      SteadyClock::time_point t_decode_end, int prefill_tokens,
                                      int decode_steps) {
    double prefill_ms = elapsed_ms(t_prefill_start, t_prefill_end);
    double decode_ms = elapsed_ms(t_decode_start, t_decode_end);
    double total_ms = elapsed_ms(t_prefill_start, t_decode_end);

    std::cerr << std::fixed << std::setprecision(1);
    std::cerr << "[trtmc-perf] Prefill: " << prefill_tokens << " tokens, " << prefill_ms << " ms";
    if (prefill_tokens > 0)
        std::cerr << " (" << std::setprecision(1) << (prefill_tokens / (prefill_ms / 1000.0))
                  << " tok/s)";
    std::cerr << "\n";

    std::cerr << "[trtmc-perf] Decode:  " << decode_steps << " steps, " << decode_ms << " ms";
    if (decode_steps > 0)
        std::cerr << " (" << std::setprecision(1) << (decode_steps / (decode_ms / 1000.0))
                  << " tok/s, " << std::setprecision(2) << (decode_ms / decode_steps) << " ms/tok)";
    std::cerr << "\n";

    std::cerr << "[trtmc-perf] Total generation: " << total_ms << " ms"
              << " (" << (prefill_tokens + decode_steps) << " tokens)\n";

    if (prof_steps_ > 0) {
        std::cerr << std::setprecision(2);
        std::cerr << "[trtmc-perf] Per-step breakdown (avg over " << prof_steps_ << " steps):\n";
        std::cerr << "[trtmc-perf]   prepare_step:  " << (prof_prepare_ms_ / prof_steps_)
                  << " ms\n";
        std::cerr << "[trtmc-perf]   forward (TRT): " << (prof_forward_ms_ / prof_steps_)
                  << " ms\n";
        std::cerr << "[trtmc-perf]   logits copy:   " << (prof_logits_copy_ms_ / prof_steps_)
                  << " ms\n";
        std::cerr << "[trtmc-perf]   state advance: " << (prof_advance_ms_ / prof_steps_)
                  << " ms\n";

        const auto host_outputs =
            nemotron_h_recurrent::host_visible_output_summary(decoder_->output_info());
        std::cerr << "[trtmc-perf]   host-visible output: " << std::setprecision(1)
                  << (prof_logits_copy_bytes_ / (1024.0 * 1024.0)) << " MB total ("
                  << host_outputs.tensor_count << " tensor/copy, " << prof_logits_copies_
                  << " copies)\n";
    }
}

void RecurrentPipeline::run_step(int32_t token_id, std::vector<float>& logits, bool copy_logits) {
    auto t0 = SteadyClock::now();

    TensorMap inputs;

    Tensor token_t;
    token_t.data = &token_id;
    token_t.shape = {1};
    token_t.dtype = DType::kInt32;
    inputs["token_id"] = token_t;

    state_->prepare_step(inputs);

    auto t1 = SteadyClock::now();

    // Use forward_async instead of forward() to avoid downloading
    // all 63 output tensors (140+ MB of state) to CPU every step.
    // Only the logits tensor (~512 KB) needs to reach CPU for sampling.
    decoder_->forward_async(inputs);

    // Wait for TRT kernel to finish before reading logits or advancing state.
    decoder_->sync();

    // Resolve logits device pointer + size once on first call.
    if (!logits_device_ptr_) {
        logits_device_ptr_ = decoder_->device_ptr("logits");
        if (!logits_device_ptr_)
            throw std::runtime_error(std::string(name_) + ": no 'logits' output");
        for (const auto& info : decoder_->output_info()) {
            if (info.name == "logits") {
                logits_numel_ = 1;
                for (auto d : info.shape)
                    logits_numel_ *= static_cast<std::size_t>(d);
                break;
            }
        }
        if (logits_numel_ == 0)
            throw std::runtime_error(std::string(name_) + ": logits tensor has zero size");
    }

    auto t2 = SteadyClock::now();

    if (copy_logits) {
        // D2H logits only (~512 KB). Intermediate prefill logits are not sampled.
        logits.resize(logits_numel_);
        const auto copy_bytes = logits_numel_ * sizeof(float);
        cudaMemcpy(logits.data(), logits_device_ptr_, copy_bytes, cudaMemcpyDeviceToHost);
        prof_logits_copy_bytes_ += copy_bytes;
        ++prof_logits_copies_;
    }

    auto t3 = SteadyClock::now();

    // Advance the logical position; recurrent buffers update in place.
    state_->advance();

    auto t4 = SteadyClock::now();

    prof_prepare_ms_ += elapsed_ms(t0, t1);
    prof_forward_ms_ += elapsed_ms(t1, t2);
    prof_logits_copy_ms_ += elapsed_ms(t2, t3);
    prof_advance_ms_ += elapsed_ms(t3, t4);
    ++prof_steps_;
}

int32_t RecurrentPipeline::argmax(const std::vector<float>& logits) {
    if (logits.empty())
        return 0;
    return static_cast<int32_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

} // namespace trtmc
