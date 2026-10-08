/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_8/runtime/pipeline.h"

#include "families/qwen3_8/runtime/chat_templates.h"

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

RecurrentPipeline::RecurrentPipeline(std::unique_ptr<ITrtModule> decoder,
                                     std::unique_ptr<Qwen38InferenceState> state,
                                     RecurrentGenConfig config, cudaStream_t stream,
                                     const char* name, std::shared_ptr<ITokenizer> tokenizer,
                                     std::string model_id_str,
                                     std::unique_ptr<Qwen38ISampler> sampler,
                                     std::unique_ptr<Qwen38MtpScheduler> mtp_scheduler)
    : decoder_(std::move(decoder)), state_(std::move(state)), config_(config), stream_(stream),
      name_(name), tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id_str)),
      sampler_(std::move(sampler)), mtp_scheduler_(std::move(mtp_scheduler)) {
    if (!decoder_ || !decoder_->ok())
        throw std::runtime_error(std::string(name_) + ": invalid decoder module");
}

static std::vector<int32_t> encode_prompt(const ITokenizer& tokenizer,
                                          const RecurrentGenConfig& config,
                                          const std::string& prompt,
                                          const TextGenerationConfig& cfg) {
    std::string effective = prompt;
    bool templated = false;
    if (cfg.use_chat_template && !config.chat_template_format.empty()) {
        effective =
            qwen3_8_apply_chat_template(config.chat_template_format, prompt, cfg.enable_thinking);
        templated = true;
    }

    auto ids = tokenizer.encode(effective);
    if (templated && ids.size() >= 2 && config.id_bos >= 0 && ids[0] == config.id_bos &&
        ids[1] == config.id_bos) {
        ids.erase(ids.begin());
    }
    return ids;
}

TextResult RecurrentPipeline::generate(const std::string& prompt, const TextGenerationConfig& cfg) {
    if (!tokenizer_)
        throw std::runtime_error(std::string(name_) + ": no tokenizer configured");

    auto input_ids = encode_prompt(*tokenizer_, config_, prompt, cfg);
    int32_t max_new = (cfg.max_new_tokens > 0) ? cfg.max_new_tokens : 128;
    auto sp = qwen38_sampling_params_from_config(cfg, config_.id_eos_ids);
    auto output_ids = generate_from_ids(input_ids, max_new, sp);

    std::vector<int32_t> new_tokens(
        output_ids.begin() + static_cast<std::ptrdiff_t>(input_ids.size()), output_ids.end());
    std::string text = tokenizer_->decode(new_tokens);

    return TextResult{std::move(text), std::move(new_tokens)};
}

RecurrentPipeline::GenerationResult
RecurrentPipeline::generate_ids(const std::vector<int32_t>& input_ids,
                                const TextGenerationConfig& cfg) {
    int32_t max_new = cfg.max_new_tokens;
    auto sp = qwen38_sampling_params_from_config(cfg, config_.id_eos_ids);
    return GenerationResult{generate_from_ids(input_ids, max_new, sp)};
}

std::vector<int32_t> RecurrentPipeline::generate_from_ids(const std::vector<int32_t>& input_ids,
                                                          int32_t max_new_tokens,
                                                          const Qwen38SamplingParams& params) {
    if (max_new_tokens == 0 || input_ids.empty())
        return input_ids;

    if (mtp_scheduler_)
        return generate_from_ids_speculative(input_ids, max_new_tokens);

    // Create a per-call sampler if none was injected at construction time.
    Qwen38ISampler* active_sampler = sampler_.get();
    std::unique_ptr<Qwen38ISampler> local_sampler;
    if (!active_sampler) {
        local_sampler = create_qwen38_sampler(params);
        active_sampler = local_sampler.get();
    }
    active_sampler->reset();

    state_->reset();
    state_->bind_to(*decoder_);

    prof_prepare_ms_ = prof_forward_ms_ = prof_logits_copy_ms_ = prof_advance_ms_ = 0;
    prof_steps_ = 0;

    std::vector<float> logits;

    // ── Prefill phase ──
    auto t_prefill_start = SteadyClock::now();
    for (std::size_t i = 0; i + 1 < input_ids.size(); ++i)
        run_step(input_ids[i], logits);

    run_step(input_ids.back(), logits);
    auto t_prefill_end = SteadyClock::now();

    // ── Decode phase ──
    std::vector<int32_t> output = input_ids;
    const int32_t vocab_size = static_cast<int32_t>(logits.size());
    int32_t decode_steps = 0;

    auto t_decode_start = SteadyClock::now();
    for (int32_t step = 0; step < max_new_tokens; ++step) {
        qwen38_apply_repetition_penalty(logits, params.repetition_penalty, output);
        Qwen38SampleResult result = active_sampler->sample(logits.data(), vocab_size, params);
        output.push_back(result.token_id);
        if (result.is_eos)
            break;
        run_step(result.token_id, logits);
        ++decode_steps;
    }
    auto t_decode_end = SteadyClock::now();

    report_timing(t_prefill_start, t_prefill_end, t_decode_start, t_decode_end,
                  static_cast<int>(input_ids.size()), decode_steps);

    return output;
}

bool RecurrentPipeline::is_eos(int32_t token) const {
    for (auto id : config_.id_eos_ids) {
        if (id == token)
            return true;
    }
    return false;
}

std::vector<int32_t>
RecurrentPipeline::generate_from_ids_speculative(const std::vector<int32_t>& input_ids,
                                                 int32_t max_new_tokens) {
    state_->reset();
    state_->bind_to(*decoder_);
    mtp_scheduler_->reset();
    mtp_scheduler_->bind_state();

    prof_prepare_ms_ = prof_forward_ms_ = prof_logits_copy_ms_ = prof_advance_ms_ = 0;
    prof_steps_ = 0;

    std::vector<float> logits;
    std::vector<float> hidden;

    // -- Prefill: process position 0, then every subsequent prompt
    // position, warming up MTP's cache along the way using the hidden
    // state from the PREVIOUS main-engine call paired with the real
    // (known) token at the current position -- mirrors the decode loop
    // below exactly, just with known tokens instead of sampled ones. --
    auto t_prefill_start = SteadyClock::now();
    run_step(input_ids[0], logits, &hidden);
    for (std::size_t i = 1; i < input_ids.size(); ++i) {
        mtp_scheduler_->draft(input_ids[i], static_cast<int32_t>(i), hidden.data());
        run_step(input_ids[i], logits, &hidden);
    }
    auto t_prefill_end = SteadyClock::now();

    // Committed through the LAST prompt position. Its logits predict the
    // first GENERATED token (not a re-prediction of a known prompt token).
    int32_t real_next = argmax(logits);
    int32_t step = static_cast<int32_t>(input_ids.size()) - 1; // last committed index
    int32_t draft = mtp_scheduler_->draft(real_next, step + 1, hidden.data());

    std::vector<int32_t> output = input_ids;
    int32_t decode_steps = 0;

    auto t_decode_start = SteadyClock::now();
    while (decode_steps < max_new_tokens) {
        auto verify = mtp_scheduler_->verify_and_maybe_commit(real_next, draft, step);

        if (verify.accepted) {
            output.push_back(real_next);
            ++decode_steps;
            ++prof_steps_;
            bool stop = is_eos(real_next) || decode_steps >= max_new_tokens;
            if (!stop) {
                output.push_back(draft);
                ++decode_steps;
                ++prof_steps_;
                stop = is_eos(draft) || decode_steps >= max_new_tokens;
            }
            if (stop)
                break;

            // Catch-up call: MTP must still process `draft` (now a
            // confirmed real token) to keep its own cache in sync with the
            // main token stream -- its own logits are discarded here.
            mtp_scheduler_->draft(draft, step + 2, verify.hidden_row0.data());
            const int32_t new_real_next = verify.next_real_candidate;
            const int32_t new_draft =
                mtp_scheduler_->draft(new_real_next, step + 3, verify.hidden_row1.data());
            step += 2;
            real_next = new_real_next;
            draft = new_draft;
        } else {
            // Reject: re-run the plain single-token main engine on the
            // confirmed-real token from the pre-round committed state --
            // bit-identical to what the verify call's row 0 already
            // computed (both are the same greedy argmax), but this is what
            // actually advances state_ for the real main decoder path.
            run_step(real_next, logits, &hidden);
            output.push_back(real_next);
            ++decode_steps;
            ++prof_steps_;
            if (is_eos(real_next) || decode_steps >= max_new_tokens)
                break;
            ++step;
            const int32_t new_draft =
                mtp_scheduler_->draft(verify.verified_token, step + 1, hidden.data());
            real_next = verify.verified_token;
            draft = new_draft;
        }
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

        std::size_t output_bytes = 0;
        for (const auto& info : decoder_->output_info()) {
            std::size_t n = 1;
            for (auto d : info.shape)
                n *= static_cast<std::size_t>(d);
            n *= dtype_size(info.dtype);
            output_bytes += n;
        }
        std::cerr << "[trtmc-perf]   D2H output size: " << std::setprecision(1)
                  << (output_bytes / (1024.0 * 1024.0)) << " MB (" << decoder_->output_info().size()
                  << " tensors)\n";
    }
}

void RecurrentPipeline::run_step(int32_t token_id, std::vector<float>& logits,
                                 std::vector<float>* hidden_state_out) {
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

    // Resolve logits device pointer + size once on first call.
    if (!logits_device_ptr_) {
        decoder_->sync(); // must sync before first device_ptr query
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

    // Wait for TRT kernel to finish.
    decoder_->sync();

    auto t2 = SteadyClock::now();

    // D2H logits only (~512 KB). Synchronous cudaMemcpy is faster than
    // cudaMemcpyAsync+sync here because it bypasses stream ordering overhead.
    logits.resize(logits_numel_);
    // A failed copy would leave the previous step's values in `logits` and the
    // sampler would silently emit a token from stale data, so surface it here.
    const cudaError_t logits_copy = cudaMemcpy(
        logits.data(), logits_device_ptr_, logits_numel_ * sizeof(float), cudaMemcpyDeviceToHost);
    if (logits_copy != cudaSuccess)
        throw std::runtime_error(std::string(name_) + ": failed to copy logits to host: " +
                                 cudaGetErrorString(logits_copy));

    if (hidden_state_out != nullptr) {
        if (!hidden_state_device_ptr_) {
            hidden_state_device_ptr_ = decoder_->device_ptr("hidden_state");
            if (!hidden_state_device_ptr_)
                throw std::runtime_error(std::string(name_) + ": no 'hidden_state' output");
            for (const auto& info : decoder_->output_info()) {
                if (info.name == "hidden_state") {
                    hidden_state_numel_ = 1;
                    for (auto d : info.shape)
                        hidden_state_numel_ *= static_cast<std::size_t>(d);
                    break;
                }
            }
            if (hidden_state_numel_ == 0)
                throw std::runtime_error(std::string(name_) + ": hidden_state tensor has zero size");
        }
        hidden_state_out->resize(hidden_state_numel_);
        const cudaError_t hidden_copy =
            cudaMemcpy(hidden_state_out->data(), hidden_state_device_ptr_,
                      hidden_state_numel_ * sizeof(float), cudaMemcpyDeviceToHost);
        if (hidden_copy != cudaSuccess) {
            throw std::runtime_error(std::string(name_) + ": failed to copy hidden_state to host: " +
                                     cudaGetErrorString(hidden_copy));
        }
    }

    auto t3 = SteadyClock::now();

    // D2D state copies (present -> state) — async on the CUDA stream.
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
