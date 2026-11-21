/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Batched prefill already supplies the first generated token. A one-token
// request must therefore finish without launching the decoder for discarded
// CUDA-graph priming or unused next-token logits.

#include "families/gemma/runtime/chat_templates.h"
#include "families/gemma/runtime/kv_cache.h"
#include "families/gemma/runtime/pipeline.h"
#include "trtmc/runtime/trt_module.h"

#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

class TestTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<std::int32_t> encode(const std::string&) const override { return {1, 1}; }
    std::string decode(const std::vector<std::int32_t>&) const override { return {}; }
    std::int32_t id_for_token(std::string_view) const override { return -1; }
    std::string token_for_id(std::int32_t) const override { return {}; }
};

struct ModuleStats {
    int32_t calls{0};
    std::unordered_map<std::string, std::vector<int64_t>> shapes;
};

class CountingModule : public trtmc::ITrtModule {
  public:
    CountingModule(std::shared_ptr<ModuleStats> stats, bool prefill, cudaStream_t stream)
        : stats_(std::move(stats)), prefill_(prefill), stream_(stream),
          present_k_(prefill ? trtmc::DeviceTensor::zeros({704, 4}, trtmc::DType::kFloat32, stream)
                             : trtmc::DeviceTensor{}),
          present_v_(prefill ? trtmc::DeviceTensor::zeros({704, 4}, trtmc::DType::kFloat32, stream)
                             : trtmc::DeviceTensor{}) {}

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        record(inputs);
        return {{"logits", trtmc::Tensor{logits_.data(), {1, 4}, trtmc::DType::kFloat32}}};
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap& inputs) override { record(inputs); }
    void sync() override { cudaStreamSynchronize(stream_); }
    cudaStream_t stream() const override { return stream_; }
    void enable_cuda_graph() override { cuda_graph_active_ = true; }
    bool cuda_graph_active() const override { return cuda_graph_active_; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return prefill_ ? 0 : 1; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override {
        return name == "token_id" || name == "position_id" || name == "attention_mask" ||
               name == "cache_k_0" || name == "cache_v_0";
    }
    bool has_output(const std::string& name) const override {
        return name == "logits" || name == "present_k_0" || name == "present_v_0";
    }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (name == "cache_k_0" || name == "cache_v_0")
            return {704, 4};
        return {};
    }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return prefill_ ? 2 : 1; }
    void* device_ptr(const std::string& name) const override {
        if (name == "present_k_0")
            return const_cast<void*>(present_k_.data());
        if (name == "present_v_0")
            return const_cast<void*>(present_v_.data());
        return nullptr;
    }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string& name, void* pointer,
                       const std::vector<int64_t>&) override {
        bind_external(name, pointer);
    }
    int32_t input_rank(const std::string& name) const override {
        return name == "token_id" || name == "position_id" ? 1 : 2;
    }
    bool input_is_dynamic(const std::string&) const override { return prefill_; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return !prefill_ || (present_k_.ok() && present_v_.ok()); }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    void record(const trtmc::TensorMap& inputs) {
        ++stats_->calls;
        stats_->shapes.clear();
        for (const auto& [name, tensor] : inputs)
            stats_->shapes[name] = tensor.shape;
    }

    std::shared_ptr<ModuleStats> stats_;
    bool prefill_{false};
    bool cuda_graph_active_{false};
    cudaStream_t stream_{nullptr};
    mutable trtmc::DeviceTensor present_k_;
    mutable trtmc::DeviceTensor present_v_;
    std::vector<float> logits_{0.1F, 0.2F, 0.9F, 0.3F};
};

// Exercise the real family verifier with deterministic model outputs. The
// successor law makes accidental proposal publication and cache rollback visible.
class MtpModule final : public CountingModule {
  public:
    MtpModule(std::shared_ptr<ModuleStats> stats, cudaStream_t stream, bool assistant,
              bool reject = false)
        : CountingModule(std::move(stats), !assistant, stream), assistant_(assistant),
          reject_(reject) {}
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        CountingModule::forward(inputs);
        const auto& ids = inputs.at("token_id");
        const auto* tokens = static_cast<const int32_t*>(ids.data);
        std::vector<int32_t> indices{0};
        if (!assistant_) {
            const auto& selected = inputs.at("logits_indices");
            const auto* rows = static_cast<const int32_t*>(selected.data);
            indices.assign(rows, rows + selected.numel());
            check(indices.size() == 1,
                  "MTP verification preserves the ordinary one-row target shape");
        }
        logits_.assign(indices.size() * 4, -10.0F);
        hidden_.resize(indices.size() * 2);
        for (std::size_t row = 0; row < indices.size(); ++row) {
            const int32_t token = tokens[indices[row]];
            const int32_t next = assistant_ && reject_ ? 0 : (token + 1) % 4;
            logits_[row * 4 + next] = 10.0F;
            hidden_[row * 2] = static_cast<float>(token);
            hidden_[row * 2 + 1] = 1.0F;
        }
        return {
            {"logits",
             {logits_.data(), {static_cast<int64_t>(indices.size()), 4}, trtmc::DType::kFloat32}},
            {"target_hidden",
             {hidden_.data(), {static_cast<int64_t>(indices.size()), 2}, trtmc::DType::kFloat32}}};
    }
    bool has_input(const std::string& name) const override {
        return name == "logits_indices" || name == "shared_k_sliding_attention" ||
               name == "shared_v_sliding_attention" || CountingModule::has_input(name);
    }
    bool has_output(const std::string& name) const override {
        return name == "target_hidden" || CountingModule::has_output(name);
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (name == "shared_k_sliding_attention" || name == "shared_v_sliding_attention")
            return {704, 4};
        return CountingModule::tensor_shape(name);
    }

  private:
    bool assistant_, reject_;
    std::vector<float> logits_, hidden_;
};

class BlockModule final : public CountingModule {
  public:
    BlockModule(std::shared_ptr<ModuleStats> stats, bool draft, bool reject = false,
                bool markov = false, bool eagle = false, bool skip_first = false,
                bool sampling = false)
        : CountingModule(std::move(stats), !draft, nullptr), draft_(draft), reject_(reject),
          markov_(markov), eagle_(eagle), skip_first_(skip_first), sampling_(sampling) {}
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        CountingModule::forward(inputs);
        const auto& ids = inputs.at(draft_ ? "draft_tokens" : "token_id");
        const auto* tokens = static_cast<const int32_t*>(ids.data);
        const int32_t input_count = static_cast<int32_t>(ids.numel());
        int32_t count = input_count;
        if (draft_) {
            const auto& context = inputs.at("target_features");
            const int32_t past = static_cast<int32_t>(context.shape[0]);
            const auto* features = static_cast<const float*>(context.data);
            for (int32_t i = 0; i < past; ++i)
                check((sampling_ ? (features[2 * i] >= 0 && features[2 * i] < 4)
                                 : features[2 * i] == i % 4) &&
                          features[2 * i + 1] == 1,
                      "block draft sees only the committed target feature prefix");
            const auto* positions = static_cast<const int32_t*>(inputs.at("draft_positions").data);
            if (eagle_) {
                check(input_count == past, "EAGLE3 inputs cover every committed feature");
                for (int32_t i = 0; i < past; ++i)
                    check(positions[i] == i &&
                              (sampling_ ? (i + 1 == past || tokens[i] == features[2 * (i + 1)])
                                         : tokens[i] == (i + 1) % 4),
                          "EAGLE3 feature i pairs with next token i+1 at position i");
                count = 7; // Static EAGLE3 outputs; runtime must trim at capacity.
            } else {
                for (int32_t i = 0; i < count; ++i)
                    check(positions[i] == past + i,
                          "block proposal positions follow committed context");
            }
            const int32_t pending = tokens[eagle_ ? input_count - 1 : 0];
            logits_.assign(count * 4, -10.0F);
            for (int32_t i = 0; i < count; ++i) {
                const int32_t successor =
                    (pending + i + (skip_first_ ? 0 : 1) + (reject_ && i == 2 ? 1 : 0)) % 4;
                logits_[i * 4 + successor] = 10.0F;
            }
            trtmc::TensorMap outputs{
                {"draft_logits", {logits_.data(), {count, 4}, trtmc::DType::kFloat32}}};
            if (markov_ || eagle_) {
                proposals_.resize(count);
                confidence_.assign(count, -100.0F); // Scheduler off must retain the whole block.
                for (int32_t i = 0; i < count; ++i)
                    proposals_[i] =
                        (pending + i + (skip_first_ ? 0 : 1) + (reject_ && i == 2 ? 1 : 0)) % 4;
                // Deliberately different: the runtime must use sequential Markov IDs.
                std::fill(logits_.begin(), logits_.end(), 0.0F);
                outputs["proposal_ids"] = {proposals_.data(), {count}, trtmc::DType::kInt32};
                if (markov_)
                    outputs["proposal_confidence"] = {
                        confidence_.data(), {count}, trtmc::DType::kFloat32};
            }
            return outputs;
        }
        const auto& selected = inputs.at("logits_indices");
        const auto* rows = static_cast<const int32_t*>(selected.data);
        logits_.assign(selected.numel() * 4, -10.0F);
        for (std::size_t i = 0; i < selected.numel(); ++i) {
            if (sampling_)
                for (int32_t j = 0; j < 4; ++j)
                    logits_[i * 4 + j] = -0.2F * j;
            logits_[i * 4 + (tokens[rows[i]] + 1) % 4] = sampling_ ? 0.7F : 10.0F;
        }
        features_.resize(count * 2);
        for (int32_t i = 0; i < count; ++i) {
            features_[i * 2] = static_cast<float>(tokens[i]);
            features_[i * 2 + 1] = 1.0F;
        }
        return {
            {"logits",
             {logits_.data(), {static_cast<int64_t>(selected.numel()), 4}, trtmc::DType::kFloat32}},
            {"target_features", {features_.data(), {count, 2}, trtmc::DType::kFloat32}}};
    }
    bool has_input(const std::string& name) const override {
        return name == "logits_indices" || name == "target_features" ||
               CountingModule::has_input(name);
    }
    bool has_output(const std::string& name) const override {
        return name == "target_features" || name == "draft_logits" ||
               ((markov_ || eagle_) && name == "proposal_ids") ||
               (markov_ && name == "proposal_confidence") || CountingModule::has_output(name);
    }

  private:
    bool draft_, reject_, markov_, eagle_, skip_first_, sampling_;
    std::vector<int32_t> proposals_;
    std::vector<float> logits_, features_, confidence_;
};

void test_gemma4_block_draft_verification() {
    for (int variant : {0, 1, 2, 3})
        for (bool reject : {false, true}) {
            const bool markov = variant == 1, eagle = variant == 2, skip_first = variant == 3;
            const int32_t maximum = skip_first ? 20 : 12;
            auto decode_stats = std::make_shared<ModuleStats>();
            auto prefill_stats = std::make_shared<ModuleStats>();
            auto draft_stats = std::make_shared<ModuleStats>();
            trtmc::GemmaTextGenConfig config;
            config.vocab_size = 4;
            config.id_eos = -1;
            config.num_layers = 1;
            config.prefill_max_length = 704;
            config.draft_block_size = skip_first ? 16 : 7;
            config.draft_mask_token_id = 0;
            config.draft_feature_width = 2;
            config.draft_markov = markov;
            config.draft_eagle = eagle;
            config.draft_skip_first = skip_first;
            auto cache = std::make_unique<trtmc::GemmaKvCache>(1, 704, 4, nullptr);
            auto* view = cache.get();
            trtmc::GemmaTextGenerationPipeline pipeline(
                std::make_unique<CountingModule>(decode_stats, false, nullptr), std::move(cache),
                config, std::make_shared<TestTokenizer>(),
                std::make_unique<BlockModule>(prefill_stats, false), nullptr, nullptr, nullptr,
                std::make_unique<BlockModule>(draft_stats, true, reject, markov, eagle,
                                              skip_first));
            trtmc::TextGenerationConfig generation;
            generation.max_new_tokens = maximum;
            generation.top_k = 1;
            generation.seed = -1;
            generation.eos_token_id = -1;
            for (int repeat = 0; repeat < 2; ++repeat) {
                const auto result = pipeline.generate_ids({0}, generation);
                std::vector<int32_t> expected;
                for (int32_t i = 0; i <= maximum; ++i)
                    expected.push_back(i % 4);
                check(
                    result.token_ids == expected,
                    "block verification preserves target tokens through acceptance and rejection");
                const auto stats = pipeline.block_draft_stats();
                check(stats.proposed > 0 && stats.accepted > 0 && stats.verified >= 2,
                      "block runtime actually proposes, verifies and accepts draft tokens");
                check(!reject || stats.accepted < stats.proposed,
                      "partial block rejection does not count later draft tokens as accepted");
                check(view->position() == maximum + 1,
                      "block cache excludes every rejected row after reset");
            }
            generation.eos_token_id = 2;
            auto result = pipeline.generate_ids({0}, generation);
            check(result.token_ids == std::vector<int32_t>({0, 1, 2}) && view->position() == 3,
                  "block accepts verified EOS without committing later proposal rows");
            generation.eos_token_id = 1;
            result = pipeline.generate_ids({0}, generation);
            check(result.token_ids == std::vector<int32_t>({0, 1}) &&
                      pipeline.block_draft_stats().proposed == 0,
                  "block never executes draft after target EOS");
            generation.eos_token_id = -1;
            generation.max_new_tokens = 2;
            std::vector<int32_t> prompt;
            for (int32_t i = 0; i < 703; ++i)
                prompt.push_back(i % 4);
            result = pipeline.generate_ids(prompt, generation);
            check(result.token_ids.size() == 705 && result.token_ids[703] == 3 &&
                      result.token_ids[704] == 0 && view->position() == 704,
                  "block capacity boundary emits final target token without cache overflow");
            generation.max_new_tokens = 1;
            pipeline.generate_ids({0}, generation);
            check(pipeline.block_draft_stats().proposed == 0,
                  "one-token block request never performs unused draft work");
            // Non-degenerate logits exercise real sampling, not greedy disguised
            // by a seed. Check each target draw across block acceptance/rejection,
            // nucleus filtering, request reset, EOS and the capacity boundary.
            auto sampled_cache = std::make_unique<trtmc::GemmaKvCache>(1, 704, 4, nullptr);
            trtmc::GemmaTextGenerationPipeline sampled(
                std::make_unique<CountingModule>(decode_stats, false, nullptr),
                std::move(sampled_cache), config, std::make_shared<TestTokenizer>(),
                std::make_unique<BlockModule>(prefill_stats, false, false, false, false, false,
                                              true),
                nullptr, nullptr, nullptr,
                std::make_unique<BlockModule>(draft_stats, true, reject, markov, eagle, skip_first,
                                              true));
            std::uint64_t accepted = 0, proposed = 0;
            bool differs_from_greedy = false;
            for (int seed : {7, 19, 42})
                for (int policy : {0, 1, 2})
                    for (int boundary : {0, 1, 2}) {
                        generation.max_new_tokens = boundary == 2 ? 2 : 32;
                        generation.temperature = policy == 0 ? 1.0F : 0.8F;
                        generation.top_k = policy == 1 ? 0 : 4;
                        generation.top_p = policy == 0 ? 1.0F : 0.85F;
                        generation.min_p = policy == 2 ? 0.3F : 0.0F;
                        generation.seed = seed;
                        generation.eos_token_id = boundary == 1 ? 2 : -1;
                        const auto input = boundary == 2 ? prompt : std::vector<int32_t>{0};
                        const auto params = trtmc::gemma_sampling_params_from_config(generation);
                        auto reference = trtmc::create_gemma_sampler(params);
                        auto expected = input;
                        for (int i = 0; i < generation.max_new_tokens; ++i) {
                            float values[]{0.0F, -0.2F, -0.4F, -0.6F};
                            const int32_t greedy = (expected.back() + 1) % 4;
                            values[greedy] = 0.7F;
                            const auto next = reference->sample(values, 4, params);
                            differs_from_greedy |= next.token_id != greedy;
                            expected.push_back(next.token_id);
                            if (next.is_eos)
                                break;
                        }
                        for (int repeat = 0; repeat < 2; ++repeat) {
                            check(sampled.generate_ids(input, generation).token_ids == expected,
                                  "sampled blocks preserve every seeded ordinary-target draw");
                            const auto stats = sampled.block_draft_stats();
                            accepted += stats.accepted;
                            proposed += stats.proposed;
                        }
                    }
            check(differs_from_greedy && accepted > 0 && proposed > accepted,
                  "sampled coverage includes non-greedy tokens, acceptance and rejection");
        }
}

void test_one_token_batched_prefill_does_not_touch_decoder() {
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::cerr << "FAIL: create CUDA stream\n";
        ++failures;
        return;
    }

    auto decode_stats = std::make_shared<ModuleStats>();
    auto prefill_stats = std::make_shared<ModuleStats>();
    auto decoder = std::make_unique<CountingModule>(decode_stats, false, stream);
    auto prefill = std::make_unique<CountingModule>(prefill_stats, true, stream);
    auto cache = std::make_unique<trtmc::GemmaKvCache>(1, 704, 4, stream, trtmc::DType::kFloat32);

    trtmc::GemmaTextGenConfig config;
    config.vocab_size = 4;
    config.id_eos = 2;
    config.num_layers = 1;
    config.prefill_max_length = 704;

    trtmc::GemmaTextGenerationPipeline pipeline(std::move(decoder), std::move(cache), config,
                                                std::make_shared<TestTokenizer>(),
                                                std::move(prefill));

    const auto bindings = pipeline.task_bindings();
    check(bindings.size() == 1 && bindings[0].key.id == "text_continuation",
          "native pipeline binds admitted continuation contract");
    check(bindings[0].implementation == static_cast<trtmc::internal::ITextContinuation*>(&pipeline),
          "binding preserves the correct multiple-inheritance interface pointer");
    const trtmc::internal::ConfigEntry options[]{{"max_new_tokens", std::int64_t{1}},
                                                 {"temperature", 0.0}};
    const auto continued = pipeline.run(
        trtmc::internal::TextContinuationRequest{std::string_view{"prefix"}}, {options, 2});
    check(continued.token_ids == std::vector<std::int32_t>{2},
          "continuation uses the native generation pipeline");
    check(decode_stats->calls == 0, "continuation one-token prefill does not decode");
    const trtmc::internal::ConfigEntry unknown[]{{"unsupported", true}};
    try {
        pipeline.run({std::string_view{"prefix"}}, {unknown, 1});
        check(false, "continuation rejects unsupported controls");
    } catch (const trtmc::internal::ConfigError&) {
    }
    prefill_stats->calls = 0;

    trtmc::TextGenerationConfig request;
    request.max_new_tokens = 1;
    request.temperature = 0.0F;
    request.top_k = 1;
    request.eos_token_id = 3;

    std::vector<int32_t> prompt(700, 1);
    const auto result = pipeline.generate_ids(prompt, request);
    check(result.token_ids.size() == 701 && result.token_ids.back() == 2,
          "one non-EOS token comes from prefill logits");
    check(prefill_stats->calls == 1, "long batched prefill launches once");
    check(decode_stats->calls == 0, "one-token request avoids discarded decoder launches");
    check(prefill_stats->shapes["token_id"] == std::vector<int64_t>({700}),
          "prefill receives the complete prompt");

    cudaStreamDestroy(stream);
    request.max_new_tokens = 6;
    try {
        (void)pipeline.generate_ids(prompt, request);
        check(false, "generation beyond engine capacity is rejected before execution");
    } catch (const std::invalid_argument&) {
    }
}

void test_gemma4_cache_geometry_and_chat() {
    const auto exercise_mtp = [](bool single_context, bool reject) {
        auto decode_stats = std::make_shared<ModuleStats>();
        auto prefill_stats = std::make_shared<ModuleStats>();
        auto assistant_stats = std::make_shared<ModuleStats>();
        trtmc::GemmaTextGenConfig config;
        config.single_context = single_context;
        config.vocab_size = 4;
        config.id_eos = -1;
        config.num_layers = 1;
        config.prefill_max_length = 704;
        config.mtp_hidden_size = 2;
        config.mtp_target_layers = {{"sliding_attention", 0}};
        auto cache = std::make_unique<trtmc::GemmaKvCache>(1, 704, 4, nullptr);
        auto* cache_view = cache.get();
        std::unique_ptr<trtmc::ITrtModule> target =
            std::make_unique<MtpModule>(prefill_stats, nullptr, false);
        std::unique_ptr<trtmc::ITrtModule> decoder;
        if (single_context)
            decoder = std::move(target);
        else
            decoder = std::make_unique<CountingModule>(decode_stats, false, nullptr);
        trtmc::GemmaTextGenerationPipeline pipeline(
            std::move(decoder), std::move(cache), config, std::make_shared<TestTokenizer>(),
            std::move(target), nullptr, nullptr, nullptr,
            std::make_unique<MtpModule>(assistant_stats, nullptr, true, reject));
        trtmc::TextGenerationConfig generation;
        generation.max_new_tokens = 4;
        generation.top_k = 1;
        generation.seed = -1;
        generation.eos_token_id = -1;
        for (int repeat = 0; repeat < 2; ++repeat) {
            const auto output = pipeline.generate_ids({0}, generation);
            check(output.token_ids == std::vector<int32_t>({0, 1, 2, 3, 0}),
                  "MTP accepts only the target sequence and resets between requests");
            const auto stats = pipeline.mtp_stats();
            check(stats.proposed == (reject ? 3 : 2) && stats.accepted == (reject ? 1 : 2) &&
                      stats.verified == stats.proposed,
                  "MTP counters distinguish real accepted and rejected proposals");
            check(cache_view->position() == 5, "MTP commits only accepted causal rows");
        }
        generation.top_k = 0;
        const auto default_greedy = pipeline.generate_ids({0}, generation);
        check(default_greedy.token_ids == std::vector<int32_t>({0, 1, 2, 3, 0}),
              "unseeded top-k zero retains the existing greedy MTP contract");
        generation.top_k = 1;
        generation.eos_token_id = 2;
        const auto eos = pipeline.generate_ids({0}, generation);
        check(eos.token_ids == std::vector<int32_t>({0, 1, 2}),
              "MTP stops at verified EOS, including rejected-proposal correction");
        generation.eos_token_id = 1;
        const auto early = pipeline.generate_ids({0}, generation);
        check(early.token_ids == std::vector<int32_t>({0, 1}) && pipeline.mtp_stats().proposed == 0,
              "MTP never proposes beyond target EOS");
        generation.eos_token_id = -1;
        generation.max_new_tokens = 2;
        std::vector<int32_t> prompt(703, 0);
        const auto boundary = pipeline.generate_ids(prompt, generation);
        check(boundary.token_ids.size() == 705 && boundary.token_ids[703] == 1 &&
                  boundary.token_ids[704] == 2 && cache_view->position() == 704,
              "MTP verifies the final output without writing beyond capacity");
        generation.max_new_tokens = 1;
        pipeline.generate_ids({0}, generation);
        check(pipeline.mtp_stats().proposed == 0,
              "one-token MTP request needs no unused assistant execution");
        generation.max_new_tokens = 4;
        generation.top_k = 4;
        generation.seed = 7;
        bool rejected = false;
        try {
            pipeline.generate_ids({0}, generation);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "MTP refuses unimplemented stochastic verification");
        check(decode_stats->calls == 0, "MTP uses the target verification module");
    };
    for (const bool single_context : {false, true}) {
        for (const bool reject : {false, true})
            exercise_mtp(single_context, reject);
    }

    const std::string closed = "{{ '<|turn>model\\n' }}{% if add_generation_prompt %}"
                               "{{ '<|channel>thought\\n<channel|>' }}{% endif %}";
    check(trtmc::gemma_detect_chat_template_format(closed) == "gemma4_closed_thought",
          "unified template retains the explicit empty thought channel");
    check(trtmc::gemma_apply_chat_template("gemma4_closed_thought", "Hello", false) ==
              "<bos><|turn>user\nHello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
          "unified non-thinking generation uses the provider suffix");
    check(trtmc::gemma4_generation_suffix("gemma4_closed_thought", true).empty(),
          "unified thinking does not force an empty thought channel");

    check(trtmc::gemma_detect_chat_template_format("{{ '<|turn>user' }}") == "gemma4",
          "Gemma4 template has its own framing");
    check(trtmc::gemma_apply_chat_template("gemma4", "Hello", false) ==
              "<bos><|turn>user\nHello<turn|>\n<|turn>model\n",
          "Gemma4 non-thinking prompt matches checkpoint");
    check(trtmc::gemma_apply_chat_template("gemma4", "Hello", true) ==
              "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHello<turn|>\n<|turn>model\n",
          "Gemma4 thinking prompt retains system control");
    check(trtmc::gemma_apply_chat_template("gemma4",
                                           "\xE3\x80\x80"
                                           "Hello"
                                           "\xC2\xA0",
                                           false,
                                           "\xC2\xA0"
                                           "Be concise."
                                           "\xE3\x80\x80") ==
              "<bos><|turn>system\nBe concise.<turn|>\n"
              "<|turn>user\nHello<turn|>\n<|turn>model\n",
          "Gemma4 keeps system message and provider Unicode trim semantics");
    check(trtmc::gemma_apply_chat_template("gemma4", "Hello", true, "Be concise.") ==
              "<bos><|turn>system\n<|think|>\nBe concise.<turn|>\n"
              "<|turn>user\nHello<turn|>\n<|turn>model\n",
          "Gemma4 thinking marker precedes the explicit system message");
    trtmc::GemmaKvCache cache(2, 8, 4, nullptr, trtmc::DType::kFloat32, {}, {4, 8});
    check(cache.num_layers() == 2, "only owning layers allocate cache");
    check(cache.cache_k(0).nbytes() == 8 * 4 * sizeof(float), "local cache width");
    check(cache.cache_k(1).nbytes() == 8 * 8 * sizeof(float), "global cache width");
    trtmc::DeviceTensor local({2, 4}, trtmc::DType::kFloat32, nullptr);
    trtmc::DeviceTensor global({2, 8}, trtmc::DType::kFloat32, nullptr);
    std::vector<float> first(8, 1.0F), second(16, 2.0F), received(16);
    cudaMemcpy(local.data(), first.data(), local.nbytes(), cudaMemcpyHostToDevice);
    cudaMemcpy(global.data(), second.data(), global.nbytes(), cudaMemcpyHostToDevice);
    const std::vector<const void*> sources{local.data(), global.data()};
    cache.write_prefill_kv(sources, sources, 2);
    check(cache.position() == 2, "mixed-width prefill advances shared position");
    cache.append_prefill_kv(sources, sources, 2);
    check(cache.position() == 4, "mixed-width append advances shared position");
    cudaMemcpy(received.data(), static_cast<float*>(cache.cache_v(1).data()) + 16, global.nbytes(),
               cudaMemcpyDeviceToHost);
    check(received == second, "global cache append uses its own row byte stride");
    received.resize(8);
    cudaMemcpy(received.data(), static_cast<float*>(cache.cache_k(0).data()) + 8, local.nbytes(),
               cudaMemcpyDeviceToHost);
    check(received == first, "local cache append uses its own row byte stride");
}

void test_single_dynamic_context_reuses_prefill_and_decode() {
    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream) == cudaSuccess, "single-context stream");
    {
        auto stats = std::make_shared<ModuleStats>();
        auto module = std::make_unique<CountingModule>(stats, true, stream);
        trtmc::GemmaTextGenConfig config;
        config.vocab_size = 4;
        config.id_bos = 1;
        config.id_eos = 3;
        config.prefill_max_length = 704;
        config.num_layers = 1;
        config.single_context = true;
        auto cache = std::make_unique<trtmc::GemmaKvCache>(1, 704, 4, stream);
        const auto* cache_state = cache.get();
        trtmc::GemmaTextGenerationPipeline pipeline(std::move(module), std::move(cache), config,
                                                    std::make_shared<TestTokenizer>(), nullptr);
        trtmc::TextGenerationConfig generation;
        generation.max_new_tokens = 3;
        generation.top_k = 1;
        generation.temperature = 0.0F;
        for (int repeat = 0; repeat < 2; ++repeat) {
            stats->calls = 0;
            const auto output = pipeline.generate_ids({1, 1, 1, 1}, generation);
            check(output.token_ids == std::vector<int32_t>({1, 1, 1, 1, 2, 2, 2}),
                  "single dynamic context preserves generated token sequence");
            check(stats->calls == 3, "shared context executes one prefill and two decode steps");
            check(cache_state->position() == 6,
                  "shared context appends and resets actual KV state");
            check(stats->shapes["token_id"] == std::vector<int64_t>({1}),
                  "shared dynamic context returns to a one-token shape");
        }
    }
    cudaStreamDestroy(stream);
}

void test_gemma4_image_contract() {
    using namespace trtmc::internal;
    std::vector<uint8_t> pixels(4 * 4 * 3);
    for (std::size_t i = 0; i < pixels.size(); ++i)
        pixels[i] = static_cast<uint8_t>(i);
    trtmc::GemmaImageConfig config{1, 2, 4, 8, 20, 21, 22};
    check(config.valid_token_ids(32), "image markers are inside the target vocabulary");
    for (auto field : {&trtmc::GemmaImageConfig::image_token, &trtmc::GemmaImageConfig::begin_token,
                       &trtmc::GemmaImageConfig::end_token}) {
        auto invalid = config;
        for (const int token : {-1, 32}) {
            invalid.*field = token;
            check(!invalid.valid_token_ids(32), "each out-of-vocabulary image marker is rejected");
        }
        auto boundary = config;
        for (const int token : {0, 31}) {
            boundary.*field = token;
            check(boundary.valid_token_ids(32), "valid image marker boundaries are retained");
        }
    }
    ImageView image{pixels.data(), pixels.size(), 4, 4, 3, ImageFormat::UInt8};
    const auto patches = trtmc::gemma4_image_patches(image, config);
    check(patches.soft_tokens == 4, "image soft-token count follows pooled spatial size");
    check(patches.pixels.size() == pixels.size(), "image patch capacity preserves RGB rows");
    for (std::size_t i = 0; i < pixels.size(); ++i)
        check(patches.pixels[i] == static_cast<float>(pixels[i] / 255.0),
              "image patch values preserve provider rescaling");
    for (int i = 0; i < 16; ++i)
        check(patches.positions[2 * i] == i % 4 && patches.positions[2 * i + 1] == i / 4,
              "image patch coordinates are row-major x,y");
    config.merged_patches = true;
    const auto merged = trtmc::gemma4_image_patches(image, config);
    check(merged.soft_tokens == 4 && merged.positions.size() == 8,
          "unified coordinates belong to merged RGB patches, not tower subpatches");
    for (int patch = 0; patch < 4; ++patch) {
        check(merged.positions[2 * patch] == patch % 2 &&
                  merged.positions[2 * patch + 1] == patch / 2,
              "unified merged patch coordinates are row-major x,y");
        for (int row = 0; row < 2; ++row)
            for (int column = 0; column < 2; ++column)
                for (int channel = 0; channel < 3; ++channel) {
                    const int source =
                        (((patch / 2) * 2 + row) * 4 + (patch % 2) * 2 + column) * 3 + channel;
                    const int target = patch * 12 + (row * 2 + column) * 3 + channel;
                    check(merged.pixels[target] == pixels[source] * (1.0F / 255.0F),
                          "unified merged patches preserve every RGB sample");
                }
    }
    image.byte_size -= 1;
    try {
        trtmc::gemma4_image_patches(image, config);
        check(false, "image storage mismatch must be rejected");
    } catch (const std::invalid_argument&) {
    }

    const ImagesTextPart parts[]{TextPartView{"  Before "}, ImageView{}, TextPartView{" After  "}};
    const ImagesTextMessage message{MessageRole::User, {parts, 3}};
    trtmc::TextGenerationConfig generation;
    generation.use_chat_template = true;
    generation.enable_thinking = false;
    check(trtmc::gemma4_image_prompt({{&message, 1}}, generation) ==
              "<bos><|turn>user\nBefore<|image|>After<turn|>\n<|turn>model\n",
          "image conversation retains ordered media parts without invented whitespace");
    generation.use_chat_template = false;
    try {
        trtmc::gemma4_image_prompt({{&message, 1}}, generation);
        check(false, "media requests require the provider template");
    } catch (const std::invalid_argument&) {
    }
}

} // namespace

void test_gemma4_audio_contract() {
    trtmc::GemmaAudioConfig config;
    config.frame_length = 4;
    config.hop_length = 2;
    config.fft_length = 4;
    config.mel_bins = 1;
    config.max_frames = 4;
    config.window = {1, 1, 1, 1};
    config.mel_filters = {1, 1, 1};
    config.mean = {0};
    config.stddev = {1};
    std::vector<float> pcm(9, 0);
    trtmc::internal::AudioView audio{{pcm.data(), pcm.size()}, 16000, 1};
    const auto features = trtmc::gemma4_audio_features(audio, config);
    check(features.mask == std::vector<int32_t>({1, 1, 1, 1}) && features.soft_tokens == 1,
          "audio frontend preserves frame and soft-token counts");
    for (const float value : features.mel)
        check(std::abs(value - std::log(0.001F)) < 1e-6F,
              "silent audio uses magnitude mel floor without invented energy");
    pcm.resize(11);
    audio.samples = {pcm.data(), pcm.size()};
    bool rejected = false;
    try {
        trtmc::gemma4_audio_features(audio, config);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "audio rejects capacity overflow rather than truncating the recording");
    audio.samples = {pcm.data(), 9};
    audio.sample_rate = 48000;
    rejected = false;
    try {
        trtmc::gemma4_audio_features(audio, config);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "audio rejects an incompatible rate rather than relabeling PCM");
    audio.sample_rate = 16000;
    audio.channels = 2;
    rejected = false;
    try {
        trtmc::gemma4_audio_features(audio, config);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "audio rejects unsupported interleaved channels");

    config.waveform = true;
    config.mel_bins = 4;
    config.max_frames = 4;
    pcm = {0.5F, -0.25F, 1.0F, 0.0F, -1.0F};
    audio = {{pcm.data(), pcm.size()}, 16000, 1};
    const auto waveform = trtmc::gemma4_audio_features(audio, config);
    check(waveform.soft_tokens == 2 && waveform.mask == std::vector<int32_t>({1, 1, 0, 0}),
          "unified waveform includes a final partial frame without fourfold subsampling");
    for (std::size_t i = 0; i < waveform.mel.size(); ++i)
        check(waveform.mel[i] == (i < pcm.size() ? pcm[i] : 0),
              "unified waveform preserves samples and zero-pads only the final frame");
    pcm.resize(17, 0.0F);
    audio.samples = {pcm.data(), pcm.size()};
    rejected = false;
    try {
        trtmc::gemma4_audio_features(audio, config);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "unified waveform rejects excess frames without truncation");

    using namespace trtmc::internal;
    const ImageAudioTextPart parts[]{TextPartView{" Listen "}, AudioView{},
                                     TextPartView{" and look "}, ImageView{}};
    const ImageAudioTextMessage message{MessageRole::User, {parts, 4}};
    trtmc::TextGenerationConfig generation;
    generation.use_chat_template = true;
    generation.enable_thinking = false;
    check(trtmc::gemma4_media_prompt({{&message, 1}}, generation) ==
              "<bos><|turn>user\nListen<|audio|>and look<|image|><turn|>\n<|turn>model\n",
          "combined request preserves the relative audio, text and image order");
}

int main() {
    if (!std::filesystem::exists("/dev/nvidiactl")) {
        std::cout << "SKIP: CUDA device is unavailable\n";
        return 77;
    }
    test_one_token_batched_prefill_does_not_touch_decoder();
    test_single_dynamic_context_reuses_prefill_and_decode();
    test_gemma4_cache_geometry_and_chat();
    test_gemma4_block_draft_verification();
    test_gemma4_image_contract();
    test_gemma4_audio_contract();
    return failures;
}
