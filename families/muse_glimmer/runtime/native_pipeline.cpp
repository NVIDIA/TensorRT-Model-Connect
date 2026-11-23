/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/muse_glimmer/runtime/native_pipeline.h"

#include "families/muse_glimmer/runtime/sampler.h"
#include "families/muse_glimmer/runtime/tokenizer.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::muse_glimmer {
namespace {
void checked(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("Muse CUDA runtime error: ") +
                                 cudaGetErrorString(error));
}
std::string section_text(const BundleReader& reader, const char* name) {
    const auto bytes = reader.read_section(name);
    return {bytes.begin(), bytes.end()};
}
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
class NativePipeline final : public ITextGeneration {
  public:
    explicit NativePipeline(const FamilyContext& context)
        : config_(nlohmann::json::parse(section_text(context.reader, "runtime.json"))),
          tokenizer_(section_text(context.reader, "tokenizer.json")) {
        const auto variant = config_.at("execution_variant").get<std::string>();
        if (config_.at("version") != 1 || (variant != "autoregressive" && variant != "dflash") ||
            config_.at("precision") != "fp16" || config_.at("weight_format") != "nvfp4_mxfp8")
            throw std::invalid_argument("Muse native bundle has an incompatible runtime contract");
        capacity_ = config_.at("max_cache_length");
        input_limit_ = config_.at("max_input_length");
        layers_ = config_.at("num_hidden_layers");
        kv_width_ =
            config_.at("num_key_value_heads").get<int>() * config_.at("head_dim").get<int>();
        vocab_ = config_.at("vocab_size");
        bos_ = config_.at("bos_token_id");
        eos_ = config_.at("eos_token_ids").get<std::vector<std::int32_t>>();
        if (capacity_ <= 0 || input_limit_ <= 0 || input_limit_ > capacity_ || layers_ <= 0 ||
            kv_width_ <= 0 || vocab_ <= 0 || bos_ < 0 || bos_ >= vocab_ || eos_.empty() ||
            std::any_of(eos_.begin(), eos_.end(), [&](int id) { return id < 0 || id >= vocab_; }))
            throw std::invalid_argument(
                "Muse native bundle contains invalid dimensions or token IDs");
        const auto tokenizer_config =
            nlohmann::json::parse(section_text(context.reader, "tokenizer_config.json"));
        const auto& bos = tokenizer_config.at("bos_token");
        bos_text_ = bos.is_string() ? bos.get<std::string>() : bos.at("content").get<std::string>();
        load_target(context);
        if (decoder_->tensor_shape("attention_mask") != std::vector<int64_t>{1, capacity_} ||
            decoder_->tensor_dtype("logits") != DType::kFloat32 ||
            decoder_->tensor_shape("logits") != std::vector<int64_t>{1, vocab_})
            throw std::invalid_argument("Muse native decoder shapes differ from runtime metadata");
        for (int i = 0; i < layers_; ++i)
            for (const char* stem : {"k", "v"}) {
                const auto name = "cache_" + std::string(stem) + "_" + std::to_string(i);
                if (decoder_->tensor_dtype(name) != DType::kFloat16 ||
                    decoder_->tensor_shape(name) != std::vector<int64_t>{capacity_, kv_width_} ||
                    decoder_->device_ptr(name) == nullptr)
                    throw std::invalid_argument("Muse native decoder has an incompatible cache: " +
                                                name);
            }
        bind_prefill_caches();
        decoder_->set_timing_label("muse_glimmer_native_target");
        if (variant == "dflash") {
            draft_version_ = config_.at("dflash_version");
            block_ = config_.at("dflash_block_size");
            feature_width_ = config_.at("feature_width");
            mask_token_ = config_.at("mask_token_id");
            draft_window_ = config_.at("draft_window");
            rank_ = config_.at("selector_rank");
            candidates_ = config_.at("selector_top_k");
            if ((draft_version_ != 1 && draft_version_ != 2) || block_ != 16 ||
                capacity_ < block_ || feature_width_ <= 0 || mask_token_ < 0 ||
                mask_token_ >= vocab_ ||
                decoder_->tensor_shape("target_features") !=
                    std::vector<int64_t>{1, feature_width_})
                throw std::invalid_argument(
                    "Muse native draft metadata does not match target features");
            const auto draft_plan = context.reader.read_section("draft.plan");
            draft_ = context.backend.create_module(draft_plan.data(), draft_plan.size(), {});
            if (!draft_ || !draft_->ok() ||
                draft_->tensor_shape("target_hidden") !=
                    std::vector<int64_t>{capacity_, feature_width_} ||
                draft_->tensor_dtype("target_hidden") != DType::kFloat16 ||
                draft_->tensor_shape("draft_tokens") != std::vector<int64_t>{block_})
                throw std::invalid_argument(
                    "Muse native draft engine has an incompatible contract");
            auto expect = [](const ITrtModule& module, const std::string& name, DType dtype,
                             const std::vector<int64_t>& shape) {
                if (module.tensor_dtype(name) != dtype || module.tensor_shape(name) != shape)
                    throw std::invalid_argument("Muse native paired binding mismatch: " + name);
            };
            expect(*decoder_, "target_features", DType::kFloat16, {1, feature_width_});
            expect(*draft_, "draft_tokens", DType::kInt32, {block_});
            expect(*draft_, "draft_positions", DType::kInt32, {block_});
            expect(*draft_, "draft_attention_mask", DType::kFloat32, {block_, capacity_ + block_});
            draft_->set_timing_label("muse_glimmer_native_draft");
            if (draft_version_ == 1) {
                expect(*draft_, "draft_logits", DType::kFloat32, {block_, vocab_});
                if (draft_->tensor_shape("draft_logits") != std::vector<int64_t>{block_, vocab_})
                    throw std::invalid_argument(
                        "Muse native V1 draft logits have incompatible dimensions");
            } else {
                if (rank_ <= 0 || candidates_ != 16 || draft_window_ <= 0 ||
                    draft_->tensor_shape("projected_hidden") !=
                        std::vector<int64_t>{block_ - 1, rank_} ||
                    draft_->tensor_shape("candidate_ids") !=
                        std::vector<int64_t>{block_ - 1, candidates_})
                    throw std::invalid_argument("Muse native V2 proposal geometry is invalid");
                const auto selector_plan = context.reader.read_section("selector.plan");
                selector_ =
                    context.backend.create_module(selector_plan.data(), selector_plan.size(), {});
                if (!selector_ || !selector_->ok() ||
                    selector_->tensor_shape("selector_scores") !=
                        std::vector<int64_t>{1, candidates_})
                    throw std::invalid_argument("Muse native V2 selector contract is invalid");
                expect(*draft_, "candidate_ids", DType::kInt32, {block_ - 1, candidates_});
                expect(*draft_, "unary_values", DType::kFloat32, {block_ - 1, candidates_});
                expect(*draft_, "projected_hidden", DType::kFloat16, {block_ - 1, rank_});
                expect(*selector_, "previous_token", DType::kInt32, {1});
                expect(*selector_, "candidate_ids", DType::kInt32, {candidates_});
                expect(*selector_, "unary_values", DType::kFloat32, {1, candidates_});
                expect(*selector_, "projected_hidden", DType::kFloat16, {1, rank_});
                expect(*selector_, "selector_scores", DType::kFloat32, {1, candidates_});
                selector_->set_timing_label("muse_glimmer_native_selector");
            }
        }
    }

    int32_t default_max_new_tokens() const override { return 128; }

    TextResult generate(const std::string& prompt, const TextGenerationConfig& cfg) override {
        const std::lock_guard<std::mutex> guard(mutex_);
        validate(cfg);
        const auto params = muse_sampling_params_from_config(cfg, eos_.front());
        if (draft_version_ == 1 && !muse_is_greedy(params))
            throw std::invalid_argument("Muse-Glimmer DFlash V1 requires greedy generation");
        if (cfg.eos_token_id < -1 || cfg.eos_token_id >= vocab_ || cfg.seed < -1)
            throw std::invalid_argument("Muse native EOS override or seed is out of range");
        const auto setup = Clock::now();
        const auto text =
            cfg.use_chat_template ? render_atem(prompt, cfg.system_prompt, bos_text_) : prompt;
        auto input = tokenizer_.encode(text);
        if (input.empty())
            input.push_back(bos_);
        if (input.size() > static_cast<std::size_t>(input_limit_) ||
            input.size() + static_cast<std::size_t>(cfg.max_new_tokens) >
                static_cast<std::size_t>(capacity_))
            throw std::invalid_argument(
                "Muse prompt and output budget exceed the native bundle capacity");
        reset();
        auto sampler = create_muse_sampler(params);
        sampler->reset();
        const double setup_ms = milliseconds(setup);
        auto start = Clock::now();
        std::vector<float> logits(vocab_);
        prefill(input, logits);
        int position = static_cast<int>(input.size());
        const double prefill_ms = milliseconds(start);
        start = Clock::now();
        std::vector<std::int32_t> output;
        if (draft_) {
            uint64_t rng = cfg.seed >= 0 ? static_cast<uint64_t>(cfg.seed) : 42ULL;
            output.push_back(
                muse_draw_probabilities(muse_probabilities(logits.data(), vocab_, params), rng));
            int rounds = 0, proposed = 0, accepted_count = 0;
            while (static_cast<int>(output.size()) < cfg.max_new_tokens &&
                   !is_eos(output.back(), cfg)) {
                const auto proposals = propose(output.back(), position, params, rng);
                ++rounds;
                proposed += static_cast<int>(proposals.size());
                bool all_accepted = true;
                for (const auto& proposal : proposals) {
                    // Verify only the confirmed prefix. A rejected proposal is
                    // never executed by the target or committed to its caches.
                    step(output.back(), position++, logits);
                    bool accepted = false;
                    const int token = muse_verify_proposal(
                        proposal.token, muse_probabilities(logits.data(), vocab_, params),
                        proposal.probabilities, rng, accepted);
                    output.push_back(token);
                    accepted_count += accepted;
                    if (!accepted)
                        all_accepted = false;
                    if (!accepted || is_eos(token, cfg) ||
                        static_cast<int>(output.size()) == cfg.max_new_tokens)
                        break;
                }
                if (all_accepted && static_cast<int>(output.size()) < cfg.max_new_tokens &&
                    !is_eos(output.back(), cfg)) {
                    step(output.back(), position++, logits);
                    output.push_back(muse_draw_probabilities(
                        muse_probabilities(logits.data(), vocab_, params), rng));
                }
            }
            std::clog << "[muse_glimmer] native DFlash v" << draft_version_ << " rounds=" << rounds
                      << " proposed=" << proposed << " accepted=" << accepted_count << '\n';
        } else {
            for (int i = 0; i < cfg.max_new_tokens; ++i) {
                const auto sampled = sampler->sample(logits.data(), vocab_, params);
                output.push_back(sampled.token_id);
                const bool stop = cfg.eos_token_id >= 0 ? sampled.token_id == cfg.eos_token_id
                                                        : std::find(eos_.begin(), eos_.end(),
                                                                    sampled.token_id) != eos_.end();
                if (stop || i + 1 == cfg.max_new_tokens)
                    break;
                step(sampled.token_id, position++, logits);
            }
        }
        TextResult result(tokenizer_.decode(output), output, prefill_ms, milliseconds(start));
        result.setup_ms = setup_ms;
        return result;
    }

  private:
    struct Proposal {
        int32_t token;
        std::vector<float> probabilities;
    };
    bool is_eos(int32_t token, const TextGenerationConfig& cfg) const {
        return cfg.eos_token_id >= 0 ? token == cfg.eos_token_id
                                     : std::find(eos_.begin(), eos_.end(), token) != eos_.end();
    }

    std::vector<Proposal> propose(int32_t anchor, int32_t position,
                                  const MuseSamplingParams& params, uint64_t& rng) {
        std::vector<int32_t> tokens(block_, mask_token_), positions(block_);
        tokens.front() = anchor;
        const int width = capacity_ + block_;
        std::vector<float> mask(static_cast<std::size_t>(block_) * width,
                                -std::numeric_limits<float>::infinity());
        for (int row = 0; row < block_; ++row) {
            positions[row] = position + row;
            for (int col = 0; col < position; ++col)
                if (draft_window_ == 0 || positions[row] - col < draft_window_)
                    mask[static_cast<std::size_t>(row) * width + col] = 0;
            // Published Muse drafts use bidirectional proposal attention.
            for (int col = 0; col < block_; ++col)
                if (draft_window_ == 0 || row - col < draft_window_)
                    mask[static_cast<std::size_t>(row) * width + capacity_ + col] = 0;
        }
        TensorMap inputs{{"draft_tokens", {tokens.data(), {block_}, DType::kInt32}},
                         {"draft_positions", {positions.data(), {block_}, DType::kInt32}},
                         {"draft_attention_mask", {mask.data(), {block_, width}, DType::kFloat32}}};
        draft_->forward_async(inputs);
        std::vector<Proposal> proposals;
        if (draft_version_ == 1) {
            std::vector<float> logits(static_cast<std::size_t>(block_) * vocab_);
            checked(cudaMemcpyAsync(logits.data(), draft_->device_ptr("draft_logits"),
                                    logits.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                    draft_->stream()));
            draft_->sync();
            if (std::any_of(logits.begin(), logits.end(),
                            [](float x) { return !std::isfinite(x); }))
                throw std::runtime_error("Muse native V1 draft produced nonfinite logits");
            for (int row = 1; row < block_; ++row) {
                const auto begin = logits.begin() + static_cast<std::size_t>(row) * vocab_;
                const auto id =
                    static_cast<int32_t>(std::max_element(begin, begin + vocab_) - begin);
                std::vector<float> q(vocab_, 0);
                q[id] = 1;
                proposals.push_back({id, std::move(q)});
            }
            return proposals;
        }
        const int rows = block_ - 1;
        std::vector<int32_t> ids(rows * candidates_);
        std::vector<float> values(rows * candidates_);
        std::vector<uint16_t> projected(rows * rank_);
        checked(cudaMemcpyAsync(ids.data(), draft_->device_ptr("candidate_ids"),
                                ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost,
                                draft_->stream()));
        checked(cudaMemcpyAsync(values.data(), draft_->device_ptr("unary_values"),
                                values.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                draft_->stream()));
        checked(cudaMemcpyAsync(projected.data(), draft_->device_ptr("projected_hidden"),
                                projected.size() * 2, cudaMemcpyDeviceToHost, draft_->stream()));
        draft_->sync();
        const bool greedy = muse_is_greedy(params);
        MuseSamplingParams draft_params = params;
        // V2 proposes over its conditional top-16 support using temperature;
        // target top-k/top-p/min-p filtering belongs to rejection verification.
        draft_params.temperature = greedy ? 0.0F : params.temperature;
        draft_params.top_k = candidates_;
        draft_params.top_p = 1;
        draft_params.min_p = 0;
        draft_params.seed = 0;
        int32_t previous = anchor;
        for (int row = 0; row < rows; ++row) {
            int32_t* row_ids = ids.data() + row * candidates_;
            if (std::any_of(row_ids, row_ids + candidates_,
                            [&](int id) { return id < 0 || id >= vocab_; }))
                throw std::runtime_error("Muse native V2 returned an invalid proposal token");
            TensorMap selector_inputs{
                {"previous_token", {&previous, {1}, DType::kInt32}},
                {"candidate_ids", {row_ids, {candidates_}, DType::kInt32}},
                {"unary_values",
                 {values.data() + row * candidates_, {1, candidates_}, DType::kFloat32}},
                {"projected_hidden",
                 {projected.data() + row * rank_, {1, rank_}, DType::kFloat16}}};
            selector_->forward_async(selector_inputs);
            std::vector<float> scores(candidates_);
            checked(cudaMemcpyAsync(scores.data(), selector_->device_ptr("selector_scores"),
                                    scores.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                    selector_->stream()));
            selector_->sync();
            if (std::any_of(scores.begin(), scores.end(),
                            [](float x) { return !std::isfinite(x); }))
                throw std::runtime_error("Muse native V2 selector produced nonfinite scores");
            std::vector<float> q(vocab_, 0);
            if (greedy) {
                int best = 0;
                for (int col = 1; col < candidates_; ++col)
                    if (scores[col] > scores[best] ||
                        (scores[col] == scores[best] && row_ids[col] < row_ids[best]))
                        best = col;
                q[row_ids[best]] = 1;
            } else {
                const auto supported = muse_probabilities(scores.data(), candidates_, draft_params);
                for (int col = 0; col < candidates_; ++col)
                    q[row_ids[col]] = supported[col];
            }
            previous = muse_draw_probabilities(q, rng);
            proposals.push_back({previous, std::move(q)});
        }
        return proposals;
    }

    static void validate(const TextGenerationConfig& c) {
        if (c.max_new_tokens <= 0 || !std::isfinite(c.temperature) || !std::isfinite(c.top_p) ||
            !std::isfinite(c.min_p) || c.temperature < 0 || c.top_p < 0 || c.top_p > 1 ||
            c.min_p < 0 || c.min_p > 1)
            throw std::invalid_argument("Muse native sampling parameters are invalid");
        if (c.source_language_token_id != -1 || c.forced_bos_token_id != -1 ||
            c.guidance_scale != -1 || c.cfg_scale != -1 || c.num_steps != -1 || c.sde_gamma != -1 ||
            !c.initial_latents.empty() || !c.condition_latents.empty() ||
            !c.condition_mask.empty() || !c.sampling_steps.empty() || !c.sde_noises.empty() ||
            c.block_length != 0 || c.confidence_threshold != -1 || c.stop_on_boxed_answer ||
            !c.lora_adapter_id.empty() || c.repetition_penalty != 1 ||
            (c.text_generation_mode != "auto" && c.text_generation_mode != "autoregressive"))
            throw std::invalid_argument(
                "Requested generation controls are unsupported by native Muse");
        if (!c.use_chat_template && !c.system_prompt.empty())
            throw std::invalid_argument("Muse system_prompt requires use_chat_template");
    }

    void reset() {
        decoder_->reset_execution_context();
        if (draft_) {
            draft_->reset_execution_context();
            checked(cudaMemsetAsync(draft_->device_ptr("target_hidden"), 0,
                                    static_cast<std::size_t>(capacity_) * feature_width_ * 2,
                                    draft_->stream()));
            draft_->sync();
            if (selector_)
                selector_->reset_execution_context();
        }
        for (int i = 0; i < layers_; ++i)
            for (const char* stem : {"k", "v"})
                checked(cudaMemsetAsync(
                    decoder_->device_ptr("cache_" + std::string(stem) + "_" + std::to_string(i)), 0,
                    static_cast<std::size_t>(capacity_) * kv_width_ * 2, decoder_->stream()));
        decoder_->sync();
    }

    // Loads the target engine. A "dual_profile" bundle has one engine with a
    // multi-row prefill profile and a single-row decode profile; older
    // "single" bundles take one token per launch.
    void load_target(const FamilyContext& context) {
        const auto layout = config_.value("decoder_engine_layout", std::string("single"));
        const auto plan = context.reader.read_section("engine.plan");
        if (layout == "single") {
            decoder_ = context.backend.create_module(plan.data(), plan.size(), {});
            if (!decoder_ || !decoder_->ok())
                throw std::runtime_error("Muse native decoder deserialization failed");
            return;
        }
        if (layout != "dual_profile")
            throw std::invalid_argument("Muse native bundle has an unknown decoder_engine_layout");
        auto modules = context.backend.create_dual_profile_modules(plan.data(), plan.size(), {});
        if (!modules.decode || !modules.decode->ok() || !modules.prefill || !modules.prefill->ok())
            throw std::runtime_error("Muse native dual-profile decoder deserialization failed");
        decoder_ = std::move(modules.decode);
        prefill_ = std::move(modules.prefill);
        const auto rows = prefill_->input_profile_shape("token_id", prefill_->profile_idx(),
                                                        ProfileShapeSelector::kMax);
        if (rows.size() != 1 || rows[0] < 1 ||
            prefill_->tensor_shape("logits") != std::vector<int64_t>{1, vocab_})
            throw std::invalid_argument("Muse native prefill profile is invalid");
        prefill_rows_ = static_cast<int>(std::min<int64_t>(rows[0], capacity_));
        prefill_->set_timing_label("muse_glimmer_native_prefill");
    }

    // Both execution contexts read the decode context's KV cache buffers.
    void bind_prefill_caches() {
        if (!prefill_)
            return;
        for (int i = 0; i < layers_; ++i)
            for (const char* stem : {"k", "v"}) {
                const auto name = "cache_" + std::string(stem) + "_" + std::to_string(i);
                if (prefill_->tensor_dtype(name) != DType::kFloat16 ||
                    prefill_->tensor_shape(name) != std::vector<int64_t>{capacity_, kv_width_})
                    throw std::invalid_argument("Muse native prefill has an incompatible cache: " +
                                                name);
                prefill_->bind_external(name, decoder_->device_ptr(name));
            }
    }

    void prefill(const std::vector<std::int32_t>& input, std::vector<float>& logits) {
        const int total = static_cast<int>(input.size());
        if (!prefill_) {
            for (int position = 0; position < total; ++position)
                run_rows(*decoder_, &input[position], 1, position, logits);
            return;
        }
        for (int start = 0; start < total; start += prefill_rows_)
            run_rows(*prefill_, &input[start], std::min(prefill_rows_, total - start), start,
                     logits);
    }

    void step(int32_t token, int32_t position, std::vector<float>& logits) {
        run_rows(*decoder_, &token, 1, position, logits);
    }

    // Runs `rows` tokens at positions [position, position + rows) on `module`,
    // commits their K/V (and draft features) to the shared caches, and returns
    // the logits of the last row.
    void run_rows(ITrtModule& module, const int32_t* tokens, int rows, int32_t position,
                  std::vector<float>& logits) {
        if (rows < 1 || position < 0 || position + rows > capacity_ ||
            std::any_of(tokens, tokens + rows, [&](int32_t t) { return t < 0 || t >= vocab_; }))
            throw std::out_of_range("Muse token or cache position is out of range");
        std::vector<int32_t> positions(static_cast<std::size_t>(rows));
        std::vector<float> mask(static_cast<std::size_t>(rows) * capacity_,
                                -std::numeric_limits<float>::infinity());
        for (int row = 0; row < rows; ++row) {
            positions[row] = position + row;
            const auto begin = mask.begin() + static_cast<std::ptrdiff_t>(row) * capacity_;
            std::fill(begin, begin + position + row + 1, 0.0F);
        }
        TensorMap inputs{{"token_id", {const_cast<int32_t*>(tokens), {rows}, DType::kInt32}},
                         {"position_id", {positions.data(), {rows}, DType::kInt32}},
                         {"attention_mask", {mask.data(), {rows, capacity_}, DType::kFloat32}}};
        module.forward_async(inputs);
        for (int i = 0; i < layers_; ++i)
            for (const char* stem : {"k", "v"}) {
                const auto suffix = std::string(stem) + "_" + std::to_string(i);
                auto* destination = static_cast<char*>(decoder_->device_ptr("cache_" + suffix)) +
                                    static_cast<std::size_t>(position) * kv_width_ * 2;
                checked(cudaMemcpyAsync(destination, module.device_ptr("present_" + suffix),
                                        static_cast<std::size_t>(rows) * kv_width_ * 2,
                                        cudaMemcpyDeviceToDevice, module.stream()));
            }
        if (draft_) {
            auto* destination = static_cast<char*>(draft_->device_ptr("target_hidden")) +
                                static_cast<std::size_t>(position) * feature_width_ * 2;
            checked(cudaMemcpyAsync(destination, module.device_ptr("target_features"),
                                    static_cast<std::size_t>(rows) * feature_width_ * 2,
                                    cudaMemcpyDeviceToDevice, module.stream()));
        }
        checked(cudaMemcpyAsync(logits.data(), module.device_ptr("logits"),
                                logits.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                module.stream()));
        module.sync();
        if (std::any_of(logits.begin(), logits.end(),
                        [](float value) { return !std::isfinite(value); }))
            throw std::runtime_error("Muse native target produced nonfinite logits at position " +
                                     std::to_string(position + rows - 1));
    }

    nlohmann::json config_;
    Tokenizer tokenizer_;
    std::unique_ptr<ITrtModule> decoder_, prefill_, draft_, selector_;
    int prefill_rows_{1};
    int draft_version_{}, block_{}, feature_width_{}, mask_token_{}, draft_window_{}, rank_{},
        candidates_{};
    std::mutex mutex_;
    int capacity_{}, input_limit_{}, layers_{}, kv_width_{}, vocab_{}, bos_{};
    std::vector<std::int32_t> eos_;
    std::string bos_text_;
};
} // namespace
ITask* create_native(const FamilyContext& context) {
    return new NativePipeline(context);
}
} // namespace trtmc::muse_glimmer
