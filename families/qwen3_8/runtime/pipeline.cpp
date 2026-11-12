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
                                     std::unique_ptr<Qwen38MtpScheduler> mtp_scheduler,
                                     std::unique_ptr<Qwen38PrefillRunner> prefill_runner,
                                     std::unique_ptr<Qwen38ExternalDraft> external_draft)
    : decoder_(std::move(decoder)), state_(std::move(state)), config_(config), stream_(stream),
      name_(name), tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id_str)),
      sampler_(std::move(sampler)), mtp_scheduler_(std::move(mtp_scheduler)),
      prefill_runner_(std::move(prefill_runner)), external_draft_(std::move(external_draft)) {
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
        if (external_draft_)
            throw std::runtime_error("Qwen native media and external drafts cannot be combined");
        zero_visual_.resize(static_cast<std::size_t>(media_hidden_size_) *
                            dtype_size(media_dtype_));
    }
    if (mtp_scheduler_ &&
        (mtp_scheduler_->has_media_inputs() != media_inputs_ ||
         (media_inputs_ && (mtp_scheduler_->media_dtype() != media_dtype_ ||
                            mtp_scheduler_->hidden_size() != media_hidden_size_))))
        throw std::runtime_error("Qwen target and MTP media contracts disagree");
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

RecurrentPipeline::GenerationResult
RecurrentPipeline::generate_media_ids(const std::vector<int32_t>& input_ids,
                                      const Qwen38MediaPrompt& media,
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
    if (mtp_scheduler_)
        mtp_scheduler_->set_media_prompt(&media);
    media_prompt_ = &media;
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
        } catch (...) {
        }
        if (mtp_scheduler_) {
            try {
                mtp_scheduler_->sync();
            } catch (...) {
            }
            mtp_scheduler_->set_media_prompt(nullptr);
        }
        media_prompt_ = nullptr;
        throw;
    }
}

std::vector<int32_t> RecurrentPipeline::generate_from_ids(const std::vector<int32_t>& input_ids,
                                                          int32_t max_new_tokens,
                                                          const Qwen38SamplingParams& params) {
    if (max_new_tokens == 0 || input_ids.empty())
        return input_ids;

    // Create a per-call sampler if none was injected at construction time.
    Qwen38ISampler* active_sampler = sampler_.get();
    std::unique_ptr<Qwen38ISampler> local_sampler;
    if (!active_sampler) {
        local_sampler = create_qwen38_sampler(params);
        active_sampler = local_sampler.get();
    }
    active_sampler->reset();
    if (external_draft_)
        return generate_external_draft(input_ids, max_new_tokens, params);
    if (mtp_scheduler_)
        return generate_from_ids_speculative(input_ids, max_new_tokens, params, *active_sampler);

    state_->reset();
    state_->bind_to(*decoder_);

    prof_prepare_ms_ = prof_forward_ms_ = prof_logits_copy_ms_ = prof_advance_ms_ = 0;
    prof_steps_ = 0;

    std::vector<float> logits;

    // ── Prefill phase ──
    auto t_prefill_start = SteadyClock::now();
    // Multi-token engine first; whatever it could not take (prompt longer than
    // the KV cache) goes through the single-token decoder, which shifts the
    // cache as it fills.
    std::size_t consumed = prefill_chunks(input_ids, logits, nullptr);
    for (; consumed < input_ids.size(); ++consumed)
        run_step(input_ids[consumed], logits);
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

std::size_t RecurrentPipeline::prefill_chunks(const std::vector<int32_t>& input_ids,
                                              std::vector<float>& logits,
                                              std::vector<float>* hidden) {
    if (!prefill_runner_)
        return 0;
    prefill_runner_->bind_state();

    const int32_t chunk = prefill_runner_->chunk_len();
    std::vector<float> rows;
    std::vector<float> warm_hidden;
    std::vector<float> prev_hidden; // hidden state of the token just before the chunk
    std::size_t consumed = 0;
    while (consumed < input_ids.size()) {
        const int32_t n = static_cast<int32_t>(
            std::min<std::size_t>(static_cast<std::size_t>(chunk), input_ids.size() - consumed));
        if (!prefill_runner_->fits(n))
            break;
        prefill_runner_->run_chunk(input_ids.data() + consumed, n, logits,
                                   hidden != nullptr ? &rows : nullptr);
        if (hidden != nullptr) {
            const std::size_t hs = rows.size() / static_cast<std::size_t>(n);
            // MTP warm-up pairs each token with the hidden state of the one before it;
            // position 0 has no predecessor and gets no entry.
            const int32_t first = (consumed == 0) ? 1 : 0;
            if (mtp_scheduler_->has_prefill()) {
                const int32_t m = n - first;
                if (m > 0) {
                    warm_hidden.resize(static_cast<std::size_t>(m) * hs);
                    for (int32_t r = 0; r < m; ++r) {
                        const int32_t j = first + r;
                        const float* prev =
                            (j == 0) ? prev_hidden.data()
                                     : rows.data() + static_cast<std::size_t>(j - 1) * hs;
                        std::copy(prev, prev + hs,
                                  warm_hidden.begin() + static_cast<std::ptrdiff_t>(r * hs));
                    }
                    const std::size_t pos0 = consumed + static_cast<std::size_t>(first);
                    mtp_scheduler_->prefill_warmup(
                        input_ids.data() + pos0, static_cast<int32_t>(pos0), warm_hidden.data(), m);
                }
            } else {
                for (int32_t j = first; j < n; ++j) {
                    const std::size_t pos = consumed + static_cast<std::size_t>(j);
                    const float* prev = (j == 0)
                                            ? prev_hidden.data()
                                            : rows.data() + static_cast<std::size_t>(j - 1) * hs;
                    mtp_scheduler_->draft(input_ids[pos], static_cast<int32_t>(pos), prev);
                }
            }
            prev_hidden.assign(rows.end() - static_cast<std::ptrdiff_t>(hs), rows.end());
        }
        consumed += static_cast<std::size_t>(n);
    }
    if (hidden != nullptr && consumed > 0)
        *hidden = std::move(prev_hidden);
    return consumed;
}

std::vector<int32_t> RecurrentPipeline::generate_from_ids_speculative(
    const std::vector<int32_t>& input_ids, int32_t max_new_tokens,
    const Qwen38SamplingParams& params, Qwen38ISampler& sampler) {
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
    std::size_t consumed = prefill_chunks(input_ids, logits, &hidden);
    if (consumed == 0) {
        run_step(input_ids[0], logits, &hidden);
        consumed = 1;
    }
    for (std::size_t i = consumed; i < input_ids.size(); ++i) {
        // MTP pairs this token with the PREVIOUS target hidden state.
        mtp_scheduler_->draft(input_ids[i], static_cast<int32_t>(i), hidden.data());
        run_step(input_ids[i], logits, &hidden);
    }
    auto t_prefill_end = SteadyClock::now();

    std::vector<int32_t> output = input_ids;
    std::vector<int32_t> history = input_ids;
    std::vector<float> sample_logits(static_cast<std::size_t>(config_.vocab_size));
    auto sample_next = [&](const float* row) {
        std::copy_n(row, config_.vocab_size, sample_logits.data());
        qwen38_apply_repetition_penalty(sample_logits, params.repetition_penalty, history);
        auto result = sampler.sample(sample_logits.data(), config_.vocab_size, params);
        history.push_back(result.token_id);
        return result;
    };
    int32_t real_next = sample_next(logits.data()).token_id;
    int32_t step = static_cast<int32_t>(input_ids.size()) - 1;
    int32_t decode_steps = 0;
    const int32_t hidden_size = mtp_scheduler_->hidden_size();
    int32_t round_num = 0;
    auto t_decode_start = SteadyClock::now();
    while (decode_steps < max_new_tokens) {
        const bool eos = std::find(params.eos_token_ids.begin(), params.eos_token_ids.end(),
                                   real_next) != params.eos_token_ids.end();
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

std::vector<int32_t>
RecurrentPipeline::generate_external_draft(const std::vector<int32_t>& input_ids,
                                           int32_t max_new_tokens,
                                           const Qwen38SamplingParams& params) {
    external_draft_->reset(*decoder_, input_ids.size(), max_new_tokens);
    state_->reset();
    state_->bind_to(*decoder_);
    std::vector<float> logits;
    for (const auto token : input_ids) {
        run_step(token, logits);
        external_draft_->observe(*decoder_);
    }
    std::vector<int32_t> output = input_ids;
    uint64_t rng = params.seed >= 0 ? static_cast<uint64_t>(params.seed) : 42ULL;
    const auto ended = [&]() {
        return output.size() - input_ids.size() >= static_cast<std::size_t>(max_new_tokens) ||
               std::find(params.eos_token_ids.begin(), params.eos_token_ids.end(), output.back()) !=
                   params.eos_token_ids.end();
    };
    const auto distribution = [&]() {
        auto adjusted = logits;
        qwen38_apply_repetition_penalty(adjusted, params.repetition_penalty, output);
        return qwen38_probabilities(adjusted.data(), static_cast<int32_t>(adjusted.size()), params);
    };
    output.push_back(qwen38_draw_probabilities(distribution(), rng));
    int32_t rounds = 0, proposed = 0, accepted_count = 0;
    while (!ended()) {
        const int32_t remaining =
            max_new_tokens - static_cast<int32_t>(output.size() - input_ids.size());
        auto proposals = external_draft_->propose(output.back(), params, output, remaining, rng);
        ++rounds;
        proposed += static_cast<int32_t>(proposals.size());
        bool rejected = false;
        for (const auto& proposal : proposals) {
            // Consume only the already confirmed anchor/previous proposal. The
            // recurrent state never executes a rejected draft token.
            run_step(output.back(), logits);
            external_draft_->observe(*decoder_);
            bool accepted = false;
            const int32_t token = qwen38_verify_proposal(proposal.token, distribution(),
                                                         proposal.probabilities, rng, accepted);
            output.push_back(token);
            accepted_count += accepted;
            if (ended())
                break;
            if (!accepted) {
                rejected = true;
                break;
            }
        }
        if (ended())
            break;
        if (!rejected) {
            run_step(output.back(), logits);
            external_draft_->observe(*decoder_);
            output.push_back(qwen38_draw_probabilities(distribution(), rng));
        }
    }
    std::cerr << "Qwen3.8 " << external_draft_->name() << " rounds=" << rounds
              << " proposals=" << proposed << " accepted=" << accepted_count << "\n";
    return output;
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
