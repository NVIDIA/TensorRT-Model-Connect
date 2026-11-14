/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_5_moe/runtime/pipeline.h"

#include "families/qwen3_5_moe/runtime/chat_templates.h"

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
                                     std::unique_ptr<Qwen35MoeInferenceState> state,
                                     RecurrentGenConfig config, cudaStream_t stream,
                                     const char* name, std::shared_ptr<ITokenizer> tokenizer,
                                     std::string model_id_str,
                                     std::unique_ptr<Qwen35MoeMtpScheduler> mtp_scheduler,
                                     std::unique_ptr<Qwen35MoePairedDraft> paired_draft)
    : decoder_(std::move(decoder)), state_(std::move(state)), config_(config), stream_(stream),
      name_(name), tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id_str)),
      mtp_scheduler_(std::move(mtp_scheduler)), paired_draft_(std::move(paired_draft)) {
    if (mtp_scheduler_ && paired_draft_)
        throw std::runtime_error("Qwen3.5-MoE cannot combine MTP and external draft schedulers");
    if (!decoder_ || !decoder_->ok())
        throw std::runtime_error(std::string(name_) + ": invalid decoder module");
    // Inspect declared inputs rather than optional Edge metadata. Older text
    // bundles remain loadable and never receive media-only bindings.
    int media_bindings = 0;
    for (const auto& input : decoder_->input_info()) {
        if (input.name == "visual_embedding") {
            if (input.shape.size() != 2 || input.shape[0] != 1 || input.shape[1] <= 0 ||
                input.shape[1] > INT32_MAX ||
                (input.dtype != DType::kFloat16 && input.dtype != DType::kBFloat16 &&
                 input.dtype != DType::kFloat32))
                throw std::runtime_error("Qwen native media embedding has invalid geometry/dtype");
            media_hidden_size_ = static_cast<int32_t>(input.shape[1]);
            media_dtype_ = input.dtype;
            ++media_bindings;
        } else if (input.name == "use_visual_embedding" ||
                   input.name == "rotary_position_offsets") {
            const auto shape = input.name == "use_visual_embedding" ? std::vector<int64_t>{1, 1}
                                                                    : std::vector<int64_t>{3};
            if (input.shape != shape || input.dtype != DType::kInt32)
                throw std::runtime_error(
                    "Qwen native media selector/coordinates have invalid metadata");
            ++media_bindings;
        }
    }
    if (media_bindings != 0 && media_bindings != 3)
        throw std::runtime_error("Qwen native media decoder has an incomplete input contract");
    media_inputs_ = media_bindings == 3;
    if (media_inputs_) {
        zero_visual_.resize(static_cast<std::size_t>(media_hidden_size_) *
                            dtype_size(media_dtype_));
    }
    if (mtp_scheduler_ &&
        (mtp_scheduler_->has_media_inputs() != media_inputs_ ||
         (media_inputs_ && (mtp_scheduler_->media_dtype() != media_dtype_ ||
                            mtp_scheduler_->hidden_size() != media_hidden_size_))))
        throw std::runtime_error("Qwen3.5-MoE MTP and target media inputs disagree");
}

static std::vector<int32_t> encode_prompt(const ITokenizer& tokenizer,
                                          const RecurrentGenConfig& config,
                                          const std::string& prompt,
                                          const TextGenerationConfig& cfg) {
    std::string effective = prompt;
    bool templated = false;
    if (cfg.use_chat_template && !config.chat_template_format.empty()) {
        effective = qwen3_5_moe_apply_chat_template(config.chat_template_format, prompt,
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

TextResult RecurrentPipeline::generate(const std::string& prompt, const TextGenerationConfig& cfg) {
    if (!tokenizer_)
        throw std::runtime_error(std::string(name_) + ": no tokenizer configured");

    auto input_ids = encode_prompt(*tokenizer_, config_, prompt, cfg);
    int32_t max_new = (cfg.max_new_tokens > 0) ? cfg.max_new_tokens : 128;
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;

    auto sp = qwen35_sampling_params_from_config(cfg, eos, config_.id_eos_ids);
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
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;
    auto sp = qwen35_sampling_params_from_config(cfg, eos, config_.id_eos_ids);
    return GenerationResult{generate_from_ids(input_ids, max_new, sp)};
}

RecurrentPipeline::GenerationResult
RecurrentPipeline::generate_media_ids(const std::vector<int32_t>& input_ids,
                                      const Qwen35MoeMediaPrompt& media,
                                      const TextGenerationConfig& cfg) {
    if (!media_inputs_)
        throw std::runtime_error("Qwen bundle has no native media decoder inputs");
    const int32_t capacity = state_->max_length();
    const auto count = input_ids.size();
    if (capacity <= 0 || count == 0 || cfg.max_new_tokens <= 0 ||
        count > static_cast<std::size_t>(capacity) ||
        static_cast<std::size_t>(cfg.max_new_tokens) > static_cast<std::size_t>(capacity) - count)
        throw std::invalid_argument("Qwen media prompt and output exceed decoder capacity");
    if (media.positions.size() != count || media.feature_rows.size() != count ||
        media.hidden_size != media_hidden_size_ || media.dtype != media_dtype_ ||
        media.features.size() % zero_visual_.size() != 0)
        throw std::invalid_argument("Qwen media features/positions disagree with decoder geometry");
    const auto rows = media.features.size() / zero_visual_.size();
    int32_t maximum = -1;
    for (std::size_t i = 0; i < count; ++i) {
        if (input_ids[i] < 0 || input_ids[i] >= config_.vocab_size || media.feature_rows[i] < -1 ||
            (media.feature_rows[i] >= 0 && static_cast<std::size_t>(media.feature_rows[i]) >= rows))
            throw std::invalid_argument("Qwen media prompt has an invalid token or feature row");
        for (const auto position : media.positions[i]) {
            if (position < 0 || position >= capacity)
                throw std::invalid_argument(
                    "Qwen media rotary coordinate exceeds decoder capacity");
            maximum = std::max(maximum, position);
        }
    }
    if (media.decode_position != maximum + 1 ||
        static_cast<int64_t>(media.decode_position) + cfg.max_new_tokens > capacity)
        throw std::invalid_argument("Qwen media decode rotary positions exceed decoder capacity");
    media_prompt_ = &media;
    if (mtp_scheduler_)
        mtp_scheduler_->set_media_prompt(&media);
    try {
        auto result = generate_ids(input_ids, cfg);
        if (mtp_scheduler_)
            mtp_scheduler_->set_media_prompt(nullptr);
        media_prompt_ = nullptr;
        return result;
    } catch (...) {
        // Drain before releasing request-owned feature storage. A later request
        // resets state and must not inherit this request's positions or pixels.
        try {
            decoder_->sync();
            if (mtp_scheduler_)
                mtp_scheduler_->sync();
        } catch (...) {
        }
        if (mtp_scheduler_)
            mtp_scheduler_->set_media_prompt(nullptr);
        media_prompt_ = nullptr;
        throw;
    }
}

std::vector<int32_t> RecurrentPipeline::generate_from_ids(const std::vector<int32_t>& input_ids,
                                                          int32_t max_new_tokens,
                                                          const Qwen35MoeSamplingParams& params) {
    if (max_new_tokens == 0 || input_ids.empty())
        return input_ids;

    if (paired_draft_)
        return generate_paired(input_ids, max_new_tokens, params);

    auto local_sampler = create_qwen35_sampler(params);
    auto* active_sampler = local_sampler.get();
    active_sampler->reset();
    if (mtp_scheduler_)
        return generate_from_ids_speculative(input_ids, max_new_tokens, params, *active_sampler);

    const int32_t capacity = state_->max_length();
    // The final sampled token is returned without another decoder/cache write.
    // Recurrent-only state advertises -1 because it has no bounded KV cache.
    if (max_new_tokens < 0 ||
        (capacity != -1 &&
         (capacity <= 0 || input_ids.size() > static_cast<std::size_t>(capacity) ||
          static_cast<std::size_t>(max_new_tokens - 1) >
              static_cast<std::size_t>(capacity) - input_ids.size())))
        throw std::invalid_argument("Qwen3.5-MoE prompt and output exceed decoder capacity");

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
        Qwen35MoeSampleResult result = active_sampler->sample(logits.data(), vocab_size, params);
        output.push_back(result.token_id);
        if (result.is_eos || step + 1 == max_new_tokens)
            break;
        run_step(result.token_id, logits);
        ++decode_steps;
    }
    auto t_decode_end = SteadyClock::now();

    report_timing(t_prefill_start, t_prefill_end, t_decode_start, t_decode_end,
                  static_cast<int>(input_ids.size()), decode_steps);

    return output;
}

std::vector<int32_t> RecurrentPipeline::generate_from_ids_speculative(
    const std::vector<int32_t>& input_ids, int32_t max_new_tokens,
    const Qwen35MoeSamplingParams& params, Qwen35MoeISampler& sampler) {
    const int32_t capacity = mtp_scheduler_->max_sequence_length();
    if (max_new_tokens < 0 || input_ids.size() > static_cast<std::size_t>(capacity) ||
        static_cast<std::size_t>(max_new_tokens) >
            static_cast<std::size_t>(capacity) - input_ids.size())
        throw std::runtime_error(std::string(name_) + ": prompt and output exceed MTP capacity");

    state_->reset();
    state_->bind_to(*decoder_);
    mtp_scheduler_->reset();
    mtp_scheduler_->bind_state();
    prof_prepare_ms_ = prof_forward_ms_ = prof_logits_copy_ms_ = prof_advance_ms_ = 0;
    prof_steps_ = 0;
    std::vector<float> logits, hidden;
    auto t_prefill_start = SteadyClock::now();
    run_step(input_ids[0], logits, &hidden);
    for (std::size_t i = 1; i < input_ids.size(); ++i) {
        // MTP pairs this token with the PREVIOUS target hidden state.
        mtp_scheduler_->draft(input_ids[i], static_cast<int32_t>(i), hidden.data());
        run_step(input_ids[i], logits, &hidden);
    }
    auto t_prefill_end = SteadyClock::now();

    std::vector<int32_t> output = input_ids;
    std::vector<float> sample_logits(static_cast<std::size_t>(config_.vocab_size));
    auto sample_next = [&](const float* row) {
        std::copy_n(row, config_.vocab_size, sample_logits.data());
        auto result = sampler.sample(sample_logits.data(), config_.vocab_size, params);
        return result;
    };
    int32_t real_next = sample_next(logits.data()).token_id;
    int32_t step = static_cast<int32_t>(input_ids.size()) - 1;
    int32_t decode_steps = 0;
    const int32_t hidden_size = mtp_scheduler_->hidden_size();
    int32_t round_num = 0;
    auto t_decode_start = SteadyClock::now();
    while (decode_steps < max_new_tokens) {
        const bool eos = params.is_eos(real_next);
        const int32_t remaining = max_new_tokens - decode_steps;
        if (eos || remaining == 1) {
            // No future token can be speculated after a terminal sample.
            output.push_back(real_next);
            ++decode_steps;
            break;
        }
        auto drafts = mtp_scheduler_->draft_chain(real_next, step + 1, hidden.data());
        mtp_scheduler_->draft(real_next, step + 1, hidden.data());
        auto verify = mtp_scheduler_->verify_and_maybe_commit(real_next, drafts.token_ids, step,
                                                              sample_next, remaining);
        ++round_num;
        std::cerr << "[trtmc-mtp-round] round=" << round_num << " base_step=" << step
                  << " real_next=" << real_next << " drafts=" << drafts.token_ids.size()
                  << " accepted_length=" << verify.accepted_length
                  << " full_accept=" << (verify.full_accept ? "true" : "false") << "\n";
        output.insert(output.end(), verify.accepted_tokens.begin(), verify.accepted_tokens.end());
        decode_steps += verify.accepted_length;
        if (verify.stopped)
            break;

        if (verify.full_accept) {
            for (int32_t i = 0; i < mtp_scheduler_->num_draft_tokens(); ++i)
                mtp_scheduler_->draft(
                    verify.accepted_tokens[static_cast<std::size_t>(i) + 1], step + 2 + i,
                    verify.hidden_states.data() + static_cast<std::size_t>(i) * hidden_size);
            hidden.assign(verify.hidden_states.end() - hidden_size, verify.hidden_states.end());
        } else {
            // The verifier exposes only its final recurrent state, including
            // rejected tokens. Rebuild just the confirmed prefix.
            for (int32_t i = 0; i < verify.accepted_length; ++i) {
                const int32_t token = verify.accepted_tokens[static_cast<std::size_t>(i)];
                if (i >= 1)
                    mtp_scheduler_->draft(token, step + 1 + i, hidden.data());
                run_step(token, logits, &hidden);
            }
        }
        step += verify.accepted_length;
        real_next = verify.next_real_candidate;
    }
    auto t_decode_end = SteadyClock::now();
    report_timing(t_prefill_start, t_prefill_end, t_decode_start, t_decode_end,
                  static_cast<int>(input_ids.size()), decode_steps);
    return output;
}

std::vector<int32_t> RecurrentPipeline::generate_paired(const std::vector<int32_t>& input_ids,
                                                        int32_t max_new_tokens,
                                                        const Qwen35MoeSamplingParams& params) {
    for (const auto token : input_ids)
        if (token < 0 || token >= config_.vocab_size)
            throw std::invalid_argument("Qwen3.5-MoE paired prompt token is outside vocabulary");
    paired_draft_->reset(*decoder_, input_ids.size(), max_new_tokens, params);
    state_->reset();
    state_->bind_to(*decoder_);
    prof_prepare_ms_ = prof_forward_ms_ = prof_logits_copy_ms_ = prof_advance_ms_ = 0;
    prof_steps_ = 0;
    std::vector<float> logits;
    const auto consume = [&](int32_t token) {
        run_step(token, logits);
        paired_draft_->observe(*decoder_, state_->position() - 1);
    };
    for (const auto token : input_ids)
        consume(token);
    std::vector<int32_t> output = input_ids;
    uint64_t rng = params.seed >= 0 ? static_cast<uint64_t>(params.seed) : 42ULL;
    const auto distribution = [&]() {
        return qwen35_probabilities(logits.data(), static_cast<int32_t>(logits.size()), params);
    };
    const auto ended = [&]() {
        return output.size() - input_ids.size() >= static_cast<std::size_t>(max_new_tokens) ||
               params.is_eos(output.back());
    };
    output.push_back(qwen35_draw_probabilities(distribution(), rng));
    int32_t rounds = 0, proposed = 0, accepted_count = 0;
    while (!ended()) {
        const auto budget = max_new_tokens - static_cast<int32_t>(output.size() - input_ids.size());
        const auto proposals = paired_draft_->propose(output.back(), params, budget, rng);
        ++rounds;
        proposed += static_cast<int32_t>(proposals.size()) - 1;
        if (paired_draft_->tree()) {
            int32_t parent = 0;
            while (!ended()) {
                // Traverse the real proposal tree. Only the confirmed path is
                // executed, so siblings/rejected tokens cannot affect DeltaNet.
                consume(output.back());
                const auto token = qwen35_draw_probabilities(distribution(), rng);
                const auto child =
                    std::find_if(proposals.begin() + 1, proposals.end(), [&](const auto& node) {
                        return node.parent == parent && node.token == token;
                    });
                output.push_back(token);
                if (child == proposals.end())
                    break; // Correction, or the bonus following a leaf.
                ++accepted_count;
                parent = static_cast<int32_t>(child - proposals.begin());
            }
        } else {
            bool rejected = false;
            for (std::size_t i = 1; i < proposals.size(); ++i) {
                consume(output.back());
                bool accepted = false;
                const auto token = qwen35_verify_proposal(
                    proposals[i].token, distribution(), proposals[i].probabilities, rng, accepted);
                output.push_back(token);
                accepted_count += accepted;
                if (ended())
                    break;
                if (!accepted) {
                    rejected = true;
                    break;
                }
            }
            if (!ended() && !rejected) {
                consume(output.back());
                output.push_back(qwen35_draw_probabilities(distribution(), rng));
            }
        }
    }
    std::cerr << "Qwen3.5-MoE " << (paired_draft_->tree() ? "dflash_ddtree" : "dflash")
              << " rounds=" << rounds << " proposals=" << proposed << " accepted=" << accepted_count
              << "\n";
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

    std::cerr << "[trtmc-perf] Total generation: " << total_ms << " ms" << " ("
              << (prefill_tokens + decode_steps) << " tokens)\n";

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

    int32_t use_visual = 0;
    std::array<int32_t, 3> offsets{0, 0, 0};
    if (media_inputs_) {
        const uint8_t* features = zero_visual_.data();
        if (media_prompt_) {
            const auto& prompt = *media_prompt_;
            const auto position = static_cast<std::size_t>(state_->position());
            if (position < prompt.positions.size()) {
                for (int axis = 0; axis < 3; ++axis)
                    offsets[axis] = prompt.positions[position][axis] - state_->position();
                const auto row = prompt.feature_rows[position];
                if (row >= 0) {
                    use_visual = 1;
                    features = prompt.features.data() +
                               static_cast<std::size_t>(row) * zero_visual_.size();
                }
            } else {
                offsets.fill(prompt.decode_position -
                             static_cast<int32_t>(prompt.positions.size()));
            }
        }
        inputs["visual_embedding"] =
            Tensor{const_cast<uint8_t*>(features), {1, media_hidden_size_}, media_dtype_};
        inputs["use_visual_embedding"] = Tensor{&use_visual, {1, 1}, DType::kInt32};
        inputs["rotary_position_offsets"] = Tensor{offsets.data(), {3}, DType::kInt32};
    }

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
    const cudaError_t logits_copy = cudaMemcpy(
        logits.data(), logits_device_ptr_, logits_numel_ * sizeof(float), cudaMemcpyDeviceToHost);
    if (logits_copy != cudaSuccess) {
        throw std::runtime_error(std::string(name_) + ": failed to copy logits to host: " +
                                 cudaGetErrorString(logits_copy));
    }

    auto t3 = SteadyClock::now();

    // D2D state copies (present -> state) — async on the CUDA stream.
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
                throw std::runtime_error(std::string(name_) +
                                         ": hidden_state tensor has zero size");
        }
        hidden_state_out->resize(hidden_state_numel_);
        const cudaError_t hidden_copy =
            cudaMemcpy(hidden_state_out->data(), hidden_state_device_ptr_,
                       hidden_state_numel_ * sizeof(float), cudaMemcpyDeviceToHost);
        if (hidden_copy != cudaSuccess) {
            throw std::runtime_error(
                std::string(name_) +
                ": failed to copy hidden_state to host: " + cudaGetErrorString(hidden_copy));
        }
    }

    state_->advance();

    auto t4 = SteadyClock::now();

    prof_prepare_ms_ += elapsed_ms(t0, t1);
    prof_forward_ms_ += elapsed_ms(t1, t2);
    prof_logits_copy_ms_ += elapsed_ms(t2, t3);
    prof_advance_ms_ += elapsed_ms(t3, t4);
    ++prof_steps_;
}

} // namespace trtmc
