/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-REC-CPP-02
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-REC-01
// Intent:         RecurrentPipeline with Mamba, RWKV, and Hybrid state managers
// Preconditions:  TRT + CUDA GPU available, mock engines
// Postconditions: Pipeline generates tokens with correct state management per backend type
// =============================================================================

// =============================================================================
// Test suite: RecurrentPipeline (Mamba, RWKV, Hybrid)
// =============================================================================
//
// Tests the RecurrentPipeline with mock engines and Qwen3.8-owned recurrent
// and hybrid state implementations.
// =============================================================================

#include "families/qwen3_8/runtime/dflash2.h"
#include "families/qwen3_8/runtime/hybrid_state.h"
#include "families/qwen3_8/runtime/kv_cache.h"
#include "families/qwen3_8/runtime/mtp_scheduler.h"
#include "families/qwen3_8/runtime/native_visual.h"
#include "families/qwen3_8/runtime/pipeline.h"
#include "families/qwen3_8/runtime/recurrent_state.h"
#include "trtmc/runtime/trt_module.h"
// TextGenerationConfig is part of the public Task API.
// (already included transitively via recurrent_pipeline.h)

#include "runtime/primitives/trt_common.h"
#include "runtime/tensorrt/trt_module_impl.h"
#include "trtmc/bundle.h"
#include "trtmc/language.hpp"

#include <NvInfer.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <vector>

static int failures = 0;
static bool skipped = false;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

// A test body that bails out has verified nothing, so it is a skip and not a
// pass. ctest reads 77 through this target's SKIP_RETURN_CODE.
static void skip(const char* what) {
    std::cerr << "SKIP: " << what << '\n';
    skipped = true;
}

static trtmc::TrtLogger g_logger;

class RecordingTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string& text) const override {
        last_text = text;
        return {1, 0};
    }

    std::string decode(const std::vector<int32_t>& ids) const override {
        std::string out;
        for (int32_t id : ids)
            out += token_for_id(id);
        return out;
    }

    int32_t id_for_token(std::string_view token) const override {
        if (token == "<bos>")
            return 1;
        if (token == "Paris")
            return 2;
        return 0;
    }

    std::string token_for_id(int32_t id) const override {
        if (id == 2)
            return "Paris";
        return "";
    }

    mutable std::string last_text;
};

// The scheduler contract is independent of a compiled graph. Device outputs
// and external cache bindings are real CUDA storage, while this module records
// the exact request sequence and allows a deterministic failed draft step.
class RecordingMtpModule final : public trtmc::ITrtModule {
  public:
    explicit RecordingMtpModule(cudaStream_t stream, bool chain = false, bool verifier = false,
                                bool media = false)
        : stream_(stream), chain_(chain), verifier_(verifier), media_(media),
          logits_({16}, trtmc::DType::kFloat32, stream),
          hidden_({8}, trtmc::DType::kFloat32, stream), ids_({3}, trtmc::DType::kInt32, stream) {}
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        forward_async(inputs);
        return {};
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override {
        throw std::logic_error("unused");
    }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {
        throw std::logic_error("unused");
    }
    void forward_async(const trtmc::TensorMap& inputs) override {
        ++calls;
        if (calls == throw_on_call)
            throw std::runtime_error("injected draft failure");
        if (media_) {
            const auto* offsets =
                static_cast<const int32_t*>(inputs.at("rotary_position_offsets").data);
            rotary_offsets.push_back({offsets[0], offsets[1], offsets[2]});
            if (!verifier_) {
                const auto* row = static_cast<const float*>(inputs.at("visual_embedding").data);
                visual_rows.push_back({row[0], row[1]});
                visual_selectors.push_back(
                    *static_cast<const int32_t*>(inputs.at("use_visual_embedding").data));
            }
        }
        if (inputs.count("token_ids") || inputs.count("token_id")) {
            const bool verify = inputs.count("token_ids") != 0;
            const auto* tokens =
                static_cast<const int32_t*>(inputs.at(verify ? "token_ids" : "token_id").data);
            const int rows = verify ? 4 : 1;
            float logits[16], hidden[8], kv[8];
            for (int row = 0; row < rows; ++row) {
                const float values[4] = {1, 2, 4, 3};
                std::copy_n(values, 4, logits + 4 * row);
                hidden[2 * row] = static_cast<float>(tokens[row]);
                hidden[2 * row + 1] = static_cast<float>(tokens[row] + 10);
                kv[2 * row] = kv[2 * row + 1] = static_cast<float>(tokens[row]);
            }
            cudaMemcpy(logits_.data(), logits, rows * 4 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(hidden_.data(), hidden, rows * 2 * sizeof(float), cudaMemcpyHostToDevice);
            for (const auto* name : {"present_k_0", "present_v_0"})
                cudaMemcpy(bindings.at(name), kv, rows * 2 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemset(bindings.at("present_conv_0"), 0, 2 * sizeof(float));
            cudaMemset(bindings.at("present_ssm_0"), 0, 4 * sizeof(float));
            return;
        }
        const int32_t token = *static_cast<const int32_t*>(inputs.at("next_token_id").data);
        const int32_t position = *static_cast<const int32_t*>(inputs.at("position_id").data);
        const float* hidden = static_cast<const float*>(inputs.at("hidden_state").data);
        positions.push_back(position);
        hidden_inputs.push_back(hidden[0]);
        std::vector<float> values(4, -10.0F);
        values[static_cast<std::size_t>((token + 1) % 4)] = 10.0F;
        cudaMemcpy(logits_.data(), values.data(), 4 * sizeof(float), cudaMemcpyHostToDevice);
        float hidden_values[6];
        int32_t tokens[3];
        for (int32_t i = 0; i < 3; ++i) {
            tokens[i] = (token + i + 1) % 4;
            hidden_values[2 * i] = hidden[0] + i + 1;
            hidden_values[2 * i + 1] = hidden[1] + i + 1;
        }
        cudaMemcpy(hidden_.data(), hidden_values, sizeof(hidden_values), cudaMemcpyHostToDevice);
        cudaMemcpy(ids_.data(), tokens, sizeof(tokens), cudaMemcpyHostToDevice);
        if (!chain_) {
            const float kv[2] = {static_cast<float>(token), static_cast<float>(position)};
            cudaMemcpy(bindings.at("mtp_present_k"), kv, sizeof(kv), cudaMemcpyHostToDevice);
            cudaMemcpy(bindings.at("mtp_present_v"), kv, sizeof(kv), cudaMemcpyHostToDevice);
        }
    }
    void sync() override { cudaStreamSynchronize(stream_); }
    cudaStream_t stream() const override { return stream_; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override {
        if (!media_)
            return {};
        std::vector<trtmc::TensorInfo> inputs{
            {"rotary_position_offsets", {3}, trtmc::DType::kInt32}};
        if (!verifier_) {
            inputs.push_back({"visual_embedding", {1, 2}, trtmc::DType::kFloat32});
            inputs.push_back({"use_visual_embedding", {1, 1}, trtmc::DType::kInt32});
        }
        return inputs;
    }
    std::vector<trtmc::TensorInfo> output_info() const override {
        return {{"logits", {1, 4}, trtmc::DType::kFloat32},
                {"hidden_state", {1, 2}, trtmc::DType::kFloat32}};
    }
    bool has_input(const std::string& name) const override {
        return name != "position_id" || !verifier_;
    }
    bool has_output(const std::string&) const override { return true; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {1, 2}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {1, 2};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string& name) const override {
        if (name == "logits" || name == "mtp_logits")
            return const_cast<void*>(logits_.data());
        if (name == "hidden_state" || name == "hidden_states" || name == "mtp_hidden_state" ||
            name == "mtp_draft_hidden_states")
            return const_cast<void*>(hidden_.data());
        if (name == "mtp_draft_token_ids")
            return const_cast<void*>(ids_.data());
        return nullptr;
    }
    void bind_external(const std::string& name, void* ptr) override { bindings[name] = ptr; }
    void bind_external(const std::string& name, void* ptr, const std::vector<int64_t>&) override {
        bind_external(name, ptr);
    }
    int32_t input_rank(const std::string& name) const override {
        return name == "position_id" ? 1 : 2;
    }
    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return logits_.ok() && hidden_.ok() && ids_.ok(); }
    void keep_alive(std::shared_ptr<void>) override {}
    std::map<std::string, void*> bindings;
    std::vector<int32_t> positions;
    std::vector<float> hidden_inputs;
    std::vector<std::array<int32_t, 3>> rotary_offsets;
    std::vector<std::array<float, 2>> visual_rows;
    std::vector<int32_t> visual_selectors;
    int calls{0};
    int throw_on_call{-1};

  private:
    cudaStream_t stream_;
    bool chain_;
    bool verifier_;
    bool media_;
    trtmc::DeviceTensor logits_, hidden_, ids_;
};

static void test_mtp_draft_cache_isolation() {
    cudaStream_t stream, draft_stream;
    cudaStreamCreate(&stream);
    cudaStreamCreate(&draft_stream);
    {
        trtmc::Qwen38KvCache cache(1, 8, 2, stream, trtmc::DType::kFloat8E4M3);
        RecordingMtpModule module(stream);
        cache.bind_to(module);
        const uint8_t k[2] = {0x38, 0xb8}, v[2] = {0x7e, 0xfe};
        check(cache.cache_k(0).nbytes() == 16 && cache.cache_v(0).nbytes() == 16,
              "Qwen E4M3 cache allocates one byte per element");
        cudaMemcpy(module.bindings.at("present_k_0"), k, sizeof(k), cudaMemcpyHostToDevice);
        cudaMemcpy(module.bindings.at("present_v_0"), v, sizeof(v), cudaMemcpyHostToDevice);
        cache.advance();
        cudaStreamSynchronize(stream);
        uint8_t actual_k[2]{}, actual_v[2]{};
        cudaMemcpy(actual_k, cache.cache_k(0).data(), 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(actual_v, cache.cache_v(0).data(), 2, cudaMemcpyDeviceToHost);
        check(std::equal(k, k + 2, actual_k) && std::equal(v, v + 2, actual_v),
              "Qwen KV advance preserves exact E4M3 bytes");
        cache.reset();
        check(cache.position() == 0, "FP8 request reset discards the logical prefix");
    }
    {
        auto kv = std::make_unique<trtmc::Qwen38KvCache>(1, 8, 2, stream);
        auto recurrent = std::make_unique<trtmc::Qwen38RecurrentState>(
            1,
            std::vector<trtmc::Qwen38RecurrentState::TensorSpec>{
                {"conv_state", {2}, "present_conv"}, {"ssm_state", {4}, "present_ssm"}},
            stream);
        trtmc::Qwen38HybridState state(std::move(kv), std::move(recurrent));
        auto step = std::make_unique<RecordingMtpModule>(draft_stream);
        auto* observed = step.get();
        trtmc::Qwen38MtpScheduler scheduler(std::move(step), nullptr,
                                            std::make_unique<RecordingMtpModule>(stream), state, 2,
                                            4, 3, stream);
        scheduler.bind_state();
        const float hidden[2] = {5.0F, 6.0F};
        scheduler.draft(0, 1, hidden);
        void* persistent = observed->bindings.at("mtp_cache_k");
        const auto first = scheduler.draft_chain(1, 2, hidden);
        check(first.token_ids == std::vector<int32_t>({2, 3, 0}),
              "MTP creates all three sequential proposals");
        check(first.hidden_states == std::vector<float>({6, 7, 7, 8, 8, 9}),
              "MTP self-chains draft hidden states");
        check(observed->positions == std::vector<int32_t>({1, 2, 3, 4}),
              "MTP preserves absolute draft positions");
        check(observed->bindings.at("mtp_cache_k") == persistent,
              "MTP restores persistent cache binding");
        float prefix[2]{};
        cudaMemcpy(prefix, persistent, sizeof(prefix), cudaMemcpyDeviceToHost);
        check(prefix[0] == 0 && prefix[1] == 1,
              "MTP speculative writes do not change confirmed prefix");
        scheduler.draft(1, 2, hidden); // fails if draft_chain advanced the confirmed cache
        observed->throw_on_call = observed->calls + 2;
        bool threw = false;
        try {
            scheduler.draft_chain(2, 3, hidden);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check(threw, "MTP propagates failed draft execution");
        check(observed->bindings.at("mtp_cache_k") == persistent,
              "MTP restores cache after failed draft");
        observed->throw_on_call = -1;
        scheduler.draft(2, 3, hidden);
        scheduler.reset();
        for (int32_t position = 1; position < 8; ++position)
            scheduler.draft(0, position, hidden);
        check(scheduler.draft_chain(1, 8, hidden).token_ids.size() == 3,
              "MTP scratch suffix fits at the target capacity boundary");
        scheduler.draft(1, 8, hidden);
    }
    {
        auto kv = std::make_unique<trtmc::Qwen38KvCache>(1, 8, 2, stream);
        auto recurrent = std::make_unique<trtmc::Qwen38RecurrentState>(
            1,
            std::vector<trtmc::Qwen38RecurrentState::TensorSpec>{
                {"conv_state", {2}, "present_conv"}, {"ssm_state", {4}, "present_ssm"}},
            stream);
        trtmc::Qwen38HybridState state(std::move(kv), std::move(recurrent));
        auto chain = std::make_unique<RecordingMtpModule>(stream, true);
        auto* observed_chain = chain.get();
        trtmc::Qwen38MtpScheduler scheduler(
            std::make_unique<RecordingMtpModule>(stream), std::move(chain),
            std::make_unique<RecordingMtpModule>(stream), state, 2, 4, 3, stream);
        scheduler.bind_state();
        const float hidden[2] = {5, 6};
        check(scheduler.draft_chain(1, 1, hidden).token_ids == std::vector<int32_t>({2, 3, 0}),
              "MTP retains legacy composed-engine bundle behavior");
        check(observed_chain->calls == 1, "MTP legacy chain still executes its bundled engine");
    }
    cudaStreamDestroy(draft_stream);
    cudaStreamDestroy(stream);
}

class ScriptedSampler final : public trtmc::Qwen38ISampler {
  public:
    explicit ScriptedSampler(std::vector<int32_t> tokens) : tokens_(std::move(tokens)) {}
    trtmc::Qwen38SampleResult sample(const float* logits, int32_t vocab,
                                     const trtmc::Qwen38SamplingParams& params) override {
        check(vocab == 4 && params.temperature == 0.7F && params.top_k == 3 &&
                  params.top_p == 0.9F && params.repetition_penalty == 1.3F,
              "MTP forwards the complete requested sampler policy");
        observed_logits.emplace_back(logits, logits + vocab);
        const int32_t token = tokens_.at(calls++);
        return {token, 0,
                std::find(params.eos_token_ids.begin(), params.eos_token_ids.end(), token) !=
                    params.eos_token_ids.end()};
    }
    trtmc::Qwen38LogitsLocation logits_location() const override {
        return trtmc::Qwen38LogitsLocation::HOST;
    }
    const char* sampler_type() const override { return "scripted"; }
    void reset() override {
        calls = 0;
        observed_logits.clear();
    }
    std::size_t calls{0};
    std::vector<std::vector<float>> observed_logits;

  private:
    std::vector<int32_t> tokens_;
};

static void test_mtp_sampling_lifecycle() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    auto make_state = [&]() {
        return std::make_unique<trtmc::Qwen38HybridState>(
            std::make_unique<trtmc::Qwen38KvCache>(1, 8, 2, stream),
            std::make_unique<trtmc::Qwen38RecurrentState>(
                1,
                std::vector<trtmc::Qwen38RecurrentState::TensorSpec>{
                    {"conv_state", {2}, "present_conv"}, {"ssm_state", {4}, "present_ssm"}},
                stream));
    };
    {
        auto state = make_state();
        RecordingMtpModule main(stream), verify(stream, false, true);
        state->bind_to(main);
        state->kv_cache()->set_position(3);
        state->kv_cache()->bind_cache_inputs(verify);
        trtmc::TensorMap inputs;
        state->prepare_step(inputs);
        check(inputs.count("position_id") == 1,
              "Read-only verifier binding preserves the main position input");
        if (inputs.count("position_id"))
            check(*static_cast<const int32_t*>(inputs.at("position_id").data) == 3,
                  "Main decoder retains its absolute position after verifier binding");
    }
    trtmc::RecurrentGenConfig config;
    config.vocab_size = 4;
    config.id_eos_ids = {99};
    trtmc::TextGenerationConfig request;
    request.max_new_tokens = 7;
    request.temperature = 0.7F;
    request.top_k = 3;
    request.top_p = 0.9F;
    request.repetition_penalty = 1.3F;
    request.seed = 38;
    for (const auto& tokens :
         {std::vector<int32_t>{1, 2, 3, 0, 1, 2, 3}, std::vector<int32_t>{1, 2, 0, 3, 2, 1, 0}}) {
        auto state = make_state();
        auto step = std::make_unique<RecordingMtpModule>(stream);
        auto* observed_step = step.get();
        auto verifier = std::make_unique<RecordingMtpModule>(stream, false, true);
        auto* observed_verify = verifier.get();
        auto scheduler = std::make_unique<trtmc::Qwen38MtpScheduler>(
            std::move(step), nullptr, std::move(verifier), *state, 2, 4, 3, stream);
        auto sampler = std::make_unique<ScriptedSampler>(tokens);
        auto* observed = sampler.get();
        trtmc::RecurrentPipeline pipeline(std::make_unique<RecordingMtpModule>(stream),
                                          std::move(state), config, stream, "MtpSampling", nullptr,
                                          "", std::move(sampler), std::move(scheduler));
        auto expected = std::vector<int32_t>{0};
        expected.insert(expected.end(), tokens.begin(), tokens.end());
        check(pipeline.generate_ids({0}, request).token_ids == expected,
              "MTP preserves sampled full and partial acceptance tokens");
        check(observed->calls == tokens.size(), "MTP consumes one RNG draw per emitted token");
        check(observed_verify->calls > 0, "MTP sampling executes the verification engine");
        check(std::abs(observed->observed_logits[1][1] - 2.0F / 1.3F) < 1e-6F,
              "MTP repetition penalty includes the confirmed anchor");
        check(std::abs(observed->observed_logits[2][2] - 4.0F / 1.3F) < 1e-6F,
              "MTP repetition penalty includes accepted drafts");
        if (tokens[2] == 0) {
            check(observed_step->positions.at(4) == 2 && observed_step->hidden_inputs.at(4) == 1,
                  "MTP partial recovery uses the previous target hidden state");
        }
        check(pipeline.generate_ids({0}, request).token_ids == expected,
              "MTP request resets sampler and recurrent state");
        request.max_new_tokens = 8;
        bool rejected = false;
        try {
            pipeline.generate_ids({0}, request);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected, "MTP rejects output beyond the admitted context capacity");
        request.max_new_tokens = 7;
    }
    // The actual family sampler must match ordinary generation for identical
    // target logits, including seed, filtering, history and request-local EOS.
    for (const int count : {1, 2, 4, 7}) {
        for (const auto& eos : {std::vector<int32_t>{99}, std::vector<int32_t>{2}}) {
            request.max_new_tokens = count;
            request.eos_token_id = eos.front();
            auto state = make_state();
            auto scheduler = std::make_unique<trtmc::Qwen38MtpScheduler>(
                std::make_unique<RecordingMtpModule>(stream), nullptr,
                std::make_unique<RecordingMtpModule>(stream, false, true), *state, 2, 4, 3, stream);
            trtmc::RecurrentPipeline mtp(std::make_unique<RecordingMtpModule>(stream),
                                         std::move(state), config, stream, "Mtp", nullptr, "",
                                         nullptr, std::move(scheduler));
            trtmc::RecurrentPipeline ordinary(std::make_unique<RecordingMtpModule>(stream),
                                              make_state(), config, stream, "Ordinary");
            check(mtp.generate_ids({0}, request).token_ids ==
                      ordinary.generate_ids({0}, request).token_ids,
                  "MTP seeded sampling equals ordinary target through EOS and capacity boundary");
        }
    }
    {
        auto state = make_state();
        auto step = std::make_unique<RecordingMtpModule>(stream, false, false, true);
        auto* observed_step = step.get();
        auto verifier = std::make_unique<RecordingMtpModule>(stream, false, true, true);
        auto* observed_verify = verifier.get();
        auto scheduler = std::make_unique<trtmc::Qwen38MtpScheduler>(
            std::move(step), nullptr, std::move(verifier), *state, 2, 4, 3, stream);
        trtmc::RecurrentPipeline pipeline(
            std::make_unique<RecordingMtpModule>(stream, false, false, true), std::move(state),
            config, stream, "MediaMtp", nullptr, "", nullptr, std::move(scheduler));
        trtmc::Qwen38MediaPrompt media;
        media.positions = {{0, 0, 0}, {1, 0, 1}, {1, 1, 1}};
        media.feature_rows = {-1, 0, 1};
        media.hidden_size = 2;
        media.dtype = trtmc::DType::kFloat32;
        media.decode_position = 2;
        const std::array<float, 4> pixels{0.125F, -0.25F, 0.5F, 1.0F};
        media.features.resize(sizeof(pixels));
        std::memcpy(media.features.data(), pixels.data(), sizeof(pixels));
        trtmc::TextGenerationConfig media_request;
        media_request.max_new_tokens = 4;
        media_request.temperature = 0;
        media_request.top_k = 1;
        const auto generated = pipeline.generate_media_ids({0, 1, 1}, media, media_request);
        check(generated.token_ids.size() == 7 && observed_verify->calls > 0,
              "Media MTP executes speculative verification after visual prefill");
        check(observed_step->visual_selectors.at(0) == 1 &&
                  observed_step->visual_selectors.at(1) == 1 &&
                  observed_step->visual_selectors.at(2) == 0 &&
                  observed_step->visual_rows.at(0) == std::array<float, 2>{0.125F, -0.25F} &&
                  observed_step->visual_rows.at(1) == std::array<float, 2>{0.5F, 1.0F},
              "MTP warmup receives the exact target visual rows before switching to text drafts");
        check(observed_step->positions.at(0) == 1 && observed_step->positions.at(1) == 2 &&
                  observed_step->rotary_offsets.at(0) == std::array<int32_t, 3>{0, -1, 0} &&
                  observed_step->rotary_offsets.at(1) == std::array<int32_t, 3>{-1, -1, -1} &&
                  observed_verify->rotary_offsets.at(0) == std::array<int32_t, 3>{-1, -1, -1},
              "MTP keeps physical cache positions and provider prefill/decode rotary coordinates "
              "distinct");
        pipeline.generate_ids({0}, media_request);
        check(observed_step->visual_selectors.back() == 0 &&
                  observed_step->rotary_offsets.back() == std::array<int32_t, 3>{0, 0, 0} &&
                  observed_verify->rotary_offsets.back() == std::array<int32_t, 3>{0, 0, 0},
              "Text after media MTP does not inherit visual rows or rotary offsets");
        observed_step->throw_on_call = observed_step->calls + 1;
        bool threw = false;
        try {
            pipeline.generate_media_ids({0, 1, 1}, media, media_request);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check(threw,
              "Media draft failures propagate instead of silently running ordinary decoding");
        observed_step->throw_on_call = -1;
        check(pipeline.generate_media_ids({0, 1, 1}, media, media_request).token_ids ==
                  generated.token_ids,
              "Media MTP resets state and borrowed feature metadata after a failed draft");
    }
    cudaStreamDestroy(stream);
}

// Mock decoder: token_id[1] → logits[4] = constant [0.1, 0.2, 0.9, 0.3]
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_mock_decoder(bool media = false) {
    auto builder = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    if (!builder)
        return nullptr;
    auto network = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* inp = network->addInput("token_id", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {1}});
    float const_logits[4] = {0.1f, 0.2f, 0.9f, 0.3f};
    auto* cst = network->addConstant(
        nvinfer1::Dims{1, {4}}, nvinfer1::Weights{nvinfer1::DataType::kFLOAT, const_logits, 4});
    auto* logits = cst->getOutput(0);
    if (media) {
        auto* visual = network->addInput("visual_embedding", nvinfer1::DataType::kFLOAT,
                                         nvinfer1::Dims{2, {1, 4}});
        auto* selector = network->addInput("use_visual_embedding", nvinfer1::DataType::kINT32,
                                           nvinfer1::Dims{2, {1, 1}});
        auto* offsets = network->addInput("rotary_position_offsets", nvinfer1::DataType::kINT32,
                                          nvinfer1::Dims{1, {3}});
        auto* condition = network->addCast(*selector, nvinfer1::DataType::kBOOL)->getOutput(0);
        auto* reshape = network->addShuffle(*logits);
        reshape->setReshapeDimensions(nvinfer1::Dims{2, {1, 4}});
        logits = network->addSelect(*condition, *visual, *reshape->getOutput(0))->getOutput(0);
        auto* observed = network->addIdentity(*offsets)->getOutput(0);
        observed->setName("observed_offsets");
        network->markOutput(*observed);
    }
    logits->setName("logits");
    network->markOutput(*logits);

    // Use input so it's not optimized away
    auto* id = network->addIdentity(*inp);
    id->getOutput(0)->setName("_unused");

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan)
        return nullptr;
    auto rt = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

// Bounded sequence state for the actual TRT media-binding fixture. No model
// math is mocked: the graph selects visual/text logits and exposes its offsets.
class MediaFixtureState final : public trtmc::Qwen38InferenceState {
  public:
    void reset() override { position_ = 0; }
    void bind_to(trtmc::ITrtModule&) override {}
    void prepare_step(trtmc::TensorMap&, int32_t = 1) override {}
    void advance(int32_t count = 1) override { position_ += count; }
    int32_t position() const override { return position_; }
    int32_t max_length() const override { return 8; }
    int32_t num_layers() const override { return 1; }
    bool needs_attention_mask() const override { return false; }
    std::size_t device_memory_bytes() const override { return 0; }
    const char* state_type() const override { return "media-fixture"; }
    bool ok() const override { return true; }

  private:
    int32_t position_{0};
};

static void test_native_media_positions() {
    using namespace trtmc;
    using namespace trtmc::qwen3_8;
    const std::vector<int32_t> ids{9, 10, 12, 12, 12, 12, 11, 9, 10, 13, 13, 11};
    const std::vector<NativeVisualGroup> groups{{2, 2, false, 0, {0x3c00, 0x4000, 0x4200, 0x4400}},
                                                {1, 2, true, 1.5, {0x4500, 0x4600}}};
    check(native_visual_text({groups[0]}) ==
              "<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>",
          "native source image expansion is unchanged");
    check(native_visual_text({groups[1]}) ==
              "<|vision_start|><1.5 seconds><|vision_start|><|video_pad|><|video_pad|>"
              "<|vision_end|><|vision_end|>",
          "native source video retains outer wrapper and video-specific placeholders");
    const auto wrapped = native_media_prompt({9, 10, 12, 12, 12, 12, 11, 10, 9, 10, 13, 13, 11, 11},
                                             groups, 1, DType::kFloat32, 12, 13, 10, 11);
    check(wrapped.feature_rows ==
                  std::vector<int32_t>({-1, -1, 0, 1, 2, 3, -1, -1, -1, -1, 4, 5, -1, -1}) &&
              wrapped.positions[10] == std::array<int32_t, 3>{8, 8, 8} &&
              wrapped.positions[11] == std::array<int32_t, 3>{8, 8, 9} &&
              wrapped.decode_position == 12,
          "source video wrapper contributes text positions without consuming visual rows");
    const auto prompt = native_media_prompt(ids, groups, 1, DType::kFloat32, 12, 13, 10, 11);
    check(prompt.feature_rows == std::vector<int32_t>({-1, -1, 0, 1, 2, 3, -1, -1, -1, 4, 5, -1}),
          "native ordered image/video tokens map to exact feature rows");
    check(prompt.positions[2] == std::array<int32_t, 3>{2, 2, 2} &&
              prompt.positions[5] == std::array<int32_t, 3>{2, 3, 3} &&
              prompt.positions[6] == std::array<int32_t, 3>{4, 4, 4} &&
              prompt.positions[10] == std::array<int32_t, 3>{7, 7, 8} &&
              prompt.decode_position == 10,
          "native mixed-media positions preserve provider grids and trailing text");
    float features[6]{};
    std::memcpy(features, prompt.features.data(), sizeof(features));
    check(std::equal(std::begin(features), std::end(features),
                     std::vector<float>{1, 2, 3, 4, 5, 6}.begin()),
          "native visual features convert to the requested decoder precision");
    auto invalid = ids;
    invalid[4] = 9;
    bool threw = false;
    try {
        native_media_prompt(invalid, groups, 1, DType::kFloat16, 12, 13, 10, 11);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "native visual binding rejects interrupted media-token spans");
    NativeVisionConfig cfg;
    cfg.patch = 2;
    cfg.temporal = 2;
    cfg.merge = 2;
    cfg.positions = 16;
    cfg.head_dim = 8;
    cfg.hidden = 48;
    cfg.min_tokens = 1;
    cfg.max_tokens = 16;
    check(native_visual_size(cfg, 3, true, 8, 8) == std::array<int32_t, 2>{8, 8},
          "native odd video count preserves temporal padding within budget");
    threw = false;
    try {
        native_visual_size(cfg, 33, true, 8, 8);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "native visual profile rejects a video with no admissible grid");
}

static void test_native_media_bindings() {
    auto engine = build_mock_decoder(true);
    check(engine != nullptr, "native media fixture builds");
    if (!engine)
        return;
    cudaStream_t stream{};
    cudaStreamCreate(&stream);
    {
        auto module = std::make_unique<trtmc::TrtModuleImpl>(
            engine.get(), engine->createExecutionContext(), stream);
        auto* observed = module.get();
        auto state = std::make_unique<MediaFixtureState>();
        auto* sequence = state.get();
        trtmc::RecurrentGenConfig config;
        config.vocab_size = 4;
        config.id_eos_ids = {3};
        trtmc::RecurrentPipeline pipeline(std::move(module), std::move(state), config, stream,
                                          "NativeMedia");
        trtmc::Qwen38MediaPrompt media;
        media.positions = {{0, 0, 0}, {1, 1, 1}, {1, 1, 2}, {1, 2, 1}, {1, 2, 2}};
        media.feature_rows = {-1, 0, 1, 2, 3};
        media.hidden_size = 4;
        media.dtype = trtmc::DType::kFloat32;
        media.decode_position = 3;
        const float rows[16] = {0, 1, 2, 3, 0, 1, 3, 2, 0, 3, 2, 1, 0, 8, 1, 2};
        media.features.resize(sizeof(rows));
        std::memcpy(media.features.data(), rows, sizeof(rows));
        const std::vector<int32_t> ids{0, 1, 1, 1, 1};
        trtmc::TextGenerationConfig cfg;
        cfg.max_new_tokens = 1;
        auto result = pipeline.generate_media_ids(ids, media, cfg);
        check(result.token_ids.back() == 1,
              "native media injects the feature for the last visual token");
        check(sequence->position() == 5, "native media does not consume a terminal budget sample");
        int32_t offsets[3]{};
        cudaMemcpy(offsets, observed->device_ptr("observed_offsets"), sizeof(offsets),
                   cudaMemcpyDeviceToHost);
        check(offsets[0] == -3 && offsets[1] == -2 && offsets[2] == -2,
              "native media separates provider MRoPE coordinates from physical positions");
        cfg.max_new_tokens = 2;
        result = pipeline.generate_media_ids(ids, media, cfg);
        check(result.token_ids[result.token_ids.size() - 2] == 1 && result.token_ids.back() == 2,
              "native media decode returns to ordinary token embeddings");
        cudaMemcpy(offsets, observed->device_ptr("observed_offsets"), sizeof(offsets),
                   cudaMemcpyDeviceToHost);
        check(offsets[0] == -2 && offsets[1] == -2 && offsets[2] == -2,
              "native media decode retains the compressed rotary-position delta");
        auto invalid = media;
        invalid.feature_rows.back() = 4;
        bool threw = false;
        try {
            pipeline.generate_media_ids(ids, invalid, cfg);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "native media rejects an out-of-bounds feature row before execution");
        cfg.max_new_tokens = 4;
        threw = false;
        try {
            pipeline.generate_media_ids(ids, media, cfg);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "native media admits prompt plus output against actual cache capacity");
        cfg.max_new_tokens = 1;
        result = pipeline.generate_ids({0}, cfg);
        check(result.token_ids.back() == 2,
              "ordinary text request after media uses text embeddings");
        cudaMemcpy(offsets, observed->device_ptr("observed_offsets"), sizeof(offsets),
                   cudaMemcpyDeviceToHost);
        check(offsets[0] == 0 && offsets[1] == 0 && offsets[2] == 0,
              "ordinary text request after media resets all rotary offsets");
        result = pipeline.generate_media_ids(ids, media, cfg);
        check(result.token_ids.back() == 1,
              "native media request remains usable after invalid requests");
    }
    cudaStreamDestroy(stream);
}

static void test_mamba_pipeline() {
    auto engine = build_mock_decoder();
    if (!engine) {
        skip("can't build engine");
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);

    // Mamba: 2 state specs, 1 layer
    std::vector<trtmc::Qwen38RecurrentState::TensorSpec> specs = {
        {"conv_state", {12}},
        {"ssm_state", {32}},
    };
    auto rs = std::make_unique<trtmc::Qwen38RecurrentState>(1, specs, stream);

    trtmc::RecurrentGenConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos_ids = {99, 2}; // argmax=2 is the checkpoint's second EOS

    // Scoped so the pipeline destructor runs while `stream` is still valid.
    {
        trtmc::RecurrentPipeline pipeline(std::move(module), std::move(rs), cfg, stream,
                                          "MambaPipeline");
        trtmc::TextGenerationConfig gen_cfg;
        gen_cfg.max_new_tokens = 5;
        auto result = pipeline.generate_ids({1}, gen_cfg);

        // argmax=2=eos → stops after 1 generated token
        check(result.token_ids.size() == 2, "mamba: input + 1 generated");
        check(result.token_ids[1] == 2, "mamba: generated token = 2 (eos)");
    }
    cudaStreamDestroy(stream);
}

static void test_rwkv_pipeline() {
    auto engine = build_mock_decoder();
    if (!engine) {
        skip("can't build engine");
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);

    // RWKV: 5 state specs, 2 layers
    std::vector<trtmc::Qwen38RecurrentState::TensorSpec> specs = {
        {"attn_state", {8}}, {"ff_state", {8}},  {"num_state", {8}},
        {"den_state", {8}},  {"max_state", {8}},
    };
    auto rs = std::make_unique<trtmc::Qwen38RecurrentState>(2, specs, stream);

    trtmc::RecurrentGenConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos_ids = {99}; // never hit

    // Scoped so the pipeline destructor runs while `stream` is still valid.
    {
        trtmc::RecurrentPipeline pipeline(std::move(module), std::move(rs), cfg, stream,
                                          "RwkvPipeline");
        trtmc::TextGenerationConfig gen_cfg;
        gen_cfg.max_new_tokens = 3;
        auto result = pipeline.generate_ids({0}, gen_cfg);

        check(result.token_ids.size() == 4, "rwkv: input + 3 generated");
        check(result.token_ids[1] == 2, "rwkv: all gen tokens = 2");
        check(result.token_ids[3] == 2, "rwkv: last gen = 2");
    }
    cudaStreamDestroy(stream);
}

static void test_hybrid_pipeline() {
    auto engine = build_mock_decoder();
    if (!engine) {
        skip("can't build engine");
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Build mock engine with mask input too
    auto builder = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto network = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto bconfig = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    bconfig->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* tok = network->addInput("token_id", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {1}});
    auto* pos =
        network->addInput("position_id", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {1}});
    // Qwen38KvCache(1, /*max_length=*/4, ...) writes a decode mask of width
    // max_length + 1 = 5. Declaring 4 here would silently drop the final column.
    auto* mask =
        network->addInput("attention_mask", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {5}});

    float cl[4] = {0.1f, 0.2f, 0.9f, 0.3f};
    auto* c = network->addConstant(nvinfer1::Dims{1, {4}},
                                   nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cl, 4});
    c->getOutput(0)->setName("logits");
    network->markOutput(*c->getOutput(0));

    network->addIdentity(*tok)->getOutput(0)->setName("_t");
    network->addIdentity(*pos)->getOutput(0)->setName("_p");
    network->addIdentity(*mask)->getOutput(0)->setName("_m");

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *bconfig));
    if (!plan) {
        skip("can't build engine");
        cudaStreamDestroy(stream);
        return;
    }
    auto rt = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    auto hybrid_engine = trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
    if (!hybrid_engine) {
        skip("can't build engine");
        cudaStreamDestroy(stream);
        return;
    }

    auto module = std::make_unique<trtmc::TrtModuleImpl>(
        hybrid_engine.get(), hybrid_engine->createExecutionContext(), stream);
    auto kv = std::make_unique<trtmc::Qwen38KvCache>(1, 4, 2, stream);
    std::vector<trtmc::Qwen38RecurrentState::TensorSpec> specs = {{"ssm", {4}}};
    auto ssm = std::make_unique<trtmc::Qwen38RecurrentState>(1, specs, stream);
    auto hybrid = std::make_unique<trtmc::Qwen38HybridState>(std::move(kv), std::move(ssm));

    trtmc::RecurrentGenConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos_ids = {2};
    cfg.has_position_input = true;

    // Scoped so the pipeline destructor runs while `stream` is still valid.
    {
        trtmc::RecurrentPipeline pipeline(std::move(module), std::move(hybrid), cfg, stream,
                                          "HybridPipeline");
        trtmc::TextGenerationConfig gen_cfg;
        gen_cfg.max_new_tokens = 5;
        auto result = pipeline.generate_ids({0}, gen_cfg);

        check(result.token_ids.size() == 2, "hybrid: input + eos");
        check(result.token_ids[1] == 2, "hybrid: eos generated");
    }
    cudaStreamDestroy(stream);
}

static void test_generate_applies_chat_template() {
    auto engine = build_mock_decoder();
    if (!engine) {
        skip("can't build engine");
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);
    std::vector<trtmc::Qwen38RecurrentState::TensorSpec> specs = {
        {"conv_state", {12}},
        {"ssm_state", {32}},
    };
    auto rs = std::make_unique<trtmc::Qwen38RecurrentState>(1, specs, stream);

    trtmc::RecurrentGenConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_bos = 1;
    cfg.id_eos_ids = {2};
    cfg.chat_template_format = "chatml";

    auto tokenizer = std::make_shared<RecordingTokenizer>();
    // Scoped so the pipeline destructor runs while `stream` is still valid.
    {
        trtmc::RecurrentPipeline pipeline(std::move(module), std::move(rs), cfg, stream,
                                          "MambaPipeline", tokenizer);

        trtmc::TextGenerationConfig gen_cfg;
        gen_cfg.max_new_tokens = 1;
        gen_cfg.use_chat_template = true;
        gen_cfg.enable_thinking = false;
        auto result = pipeline.generate("What is the capital of France?", gen_cfg);

        check(result.text == "Paris", "chat template: generated text decodes");
        check(tokenizer->last_text.find("<|im_start|>user\n") != std::string::npos,
              "chat template: user prefix");
        check(tokenizer->last_text.find("What is the capital of France?") != std::string::npos,
              "chat template: prompt retained");
        check(tokenizer->last_text.find("<think>\n\n</think>\n\n") != std::string::npos,
              "chat template: no-thinking block is closed");
    }
    cudaStreamDestroy(stream);
}

static void test_argmax_recurrent() {
    std::vector<float> v = {-1.0f, 5.0f, 3.0f};
    check(trtmc::RecurrentPipeline::argmax(v) == 1, "argmax = 1");
}

// Optional real-bundle mode of this existing test. Quality is evaluated by the
// owning Python oracle; successful execution alone is not a quality pass.
static int test_typed_bundle(const char* bundle_path, const char* runtime_root,
                             const char* requests_path, const char* output_path) {
    const trtmc::BundleReader bundle(bundle_path);
    if (bundle.info().family != "qwen3_8" ||
        (!bundle.find_section("edge_llm.json") && !bundle.find_section("native_visual.json")))
        throw std::invalid_argument("Expected a qwen3_8 media-capable bundle");
    auto model = trtmc::Model::load(bundle_path, {runtime_root});
    std::ifstream input(requests_path);
    const auto requests = nlohmann::json::parse(input);
    if (!requests.is_array() || requests.empty())
        throw std::invalid_argument("Expected a nonempty request array");
    auto report = nlohmann::json::array();
    bool passed = true;
    for (const auto& request : requests) {
        nlohmann::json row{{"id", request.at("id")}};
        try {
            const auto maximum = request.value("max_new_tokens", 128);
            const trtmc::Config config{
                {"max_new_tokens", std::int64_t{maximum}},
                {"temperature", request.value("temperature", 0.0)},
                {"top_k", request.value("top_k", std::int64_t{1})},
                {"top_p", request.value("top_p", 1.0)},
                {"seed", request.value("seed", std::int64_t{-1})},
                {"use_chat_template", request.value("use_chat_template", true)},
                {"enable_thinking", request.value("enable_thinking", false)}};
            // Own pixels through synchronous run; descriptors only borrow them.
            std::deque<std::vector<std::uint8_t>> pixels;
            auto image = [&](const nlohmann::json& frame) {
                std::ifstream file(frame.at("rgb_path").get<std::string>(), std::ios::binary);
                if (!file)
                    throw std::invalid_argument("Cannot open RGB fixture");
                pixels.emplace_back(std::istreambuf_iterator<char>(file),
                                    std::istreambuf_iterator<char>());
                return trtmc::ImageInput{{pixels.back().data(), pixels.back().size()},
                                         frame.at("height").get<std::uint32_t>(),
                                         frame.at("width").get<std::uint32_t>()};
            };
            auto video = [&](const nlohmann::json& value) {
                trtmc::VideoInput result;
                for (const auto& frame : value.at("frames"))
                    result.frames.push_back(image(frame));
                result.timestamps_seconds =
                    value.value("timestamps_seconds", std::vector<double>{});
                return result;
            };
            const auto prompt = request.at("prompt").get<std::string>();
            auto run = [&]() -> trtmc::TextContinuationResult {
                const auto task = request.at("task").get<std::string>();
                if (task == "text_continuation")
                    return model.task<trtmc::TextContinuation>().run({prompt}, config);
                if (task == "images_text_to_text") {
                    std::vector<trtmc::ImagesTextPart> parts;
                    for (const auto& frame : request.at("images"))
                        parts.emplace_back(image(frame));
                    parts.emplace_back(trtmc::TextPart{prompt});
                    return model.task<trtmc::ImagesTextToText>().run(
                        trtmc::ImagesTextToTextRequest::from_parts(std::move(parts)), config);
                }
                if (task == "video_text_to_text")
                    return model.task<trtmc::VideoTextToText>().run(
                        trtmc::VideoTextToTextRequest::from_parts(
                            {video(request.at("video")), trtmc::TextPart{prompt}}),
                        config);
                if (task == "image_video_text_to_text")
                    return model.task<trtmc::ImageVideoTextToText>().run(
                        trtmc::ImageVideoTextToTextRequest::from_parts({image(request.at("image")),
                                                                        video(request.at("video")),
                                                                        trtmc::TextPart{prompt}}),
                        config);
                throw std::invalid_argument("Unknown typed request task");
            };
            const auto result = run();
            const auto ids = result.token_ids();
            if (result.text().empty() || ids.empty() ||
                ids.size() > static_cast<std::size_t>(maximum))
                throw std::runtime_error("Generation returned an invalid text/token result");
            row["text"] = std::string(result.text());
            row["token_ids"] = std::vector<std::int32_t>(ids.begin(), ids.end());
            if (request.contains("expect_error")) {
                row["unexpected_success"] = true;
                passed = false;
            }
        } catch (const std::exception& error) {
            row["error"] = error.what();
            row["expected_error"] =
                request.contains("expect_error") &&
                std::string(error.what()).find(request.at("expect_error").get<std::string>()) !=
                    std::string::npos;
            passed = passed && row["expected_error"].get<bool>();
        }
        report.push_back(std::move(row));
        std::ofstream(output_path) << report.dump(2) << '\n';
    }
    return passed ? 0 : 1;
}

void test_dspark_rejection_sampling() {
    using namespace trtmc;
    const std::vector<float> logits{std::log(0.6F), std::log(0.3F), std::log(0.1F)};
    Qwen38SamplingParams params;
    params.top_k = 0;
    params.seed = 38;
    auto probabilities = qwen38_probabilities(logits.data(), 3, params);
    check(std::fabs(probabilities[0] - 0.6F) < 1e-6F && std::fabs(probabilities[1] - 0.3F) < 1e-6F,
          "DSpark uses the existing target softmax distribution");
    params.top_k = 2;
    probabilities = qwen38_probabilities(logits.data(), 3, params);
    check(std::fabs(probabilities[0] - 2.0F / 3) < 1e-6F && probabilities[2] == 0,
          "DSpark preserves target top-k normalization");
    params.top_k = 0;
    params.top_p = 0.7F;
    probabilities = qwen38_probabilities(logits.data(), 3, params);
    check(std::fabs(probabilities[0] - 2.0F / 3) < 1e-6F && probabilities[2] == 0,
          "DSpark preserves target top-p normalization");
    params.top_p = 1;
    params.min_p = 0.7F;
    probabilities = qwen38_probabilities(logits.data(), 3, params);
    check(probabilities == std::vector<float>({1, 0, 0}), "DSpark preserves target min-p");
    uint64_t rng = 38;
    bool accepted = false;
    check(qwen38_verify_proposal(1, {0, 1, 0}, {0, 1, 0}, rng, accepted) == 1 && accepted,
          "Identical point-mass proposal is accepted");
    check(qwen38_verify_proposal(1, {0, 0, 1}, {0, 1, 0}, rng, accepted) == 2 && !accepted,
          "Disjoint proposal returns the exact target residual");
    const std::vector<float> target{0.7F, 0.2F, 0.1F}, draft{0.1F, 0.3F, 0.6F};
    int counts[3]{0, 0, 0};
    // Fixed seed makes this deterministic. This exercises rejection/residual
    // sampling against a deliberately very different proposal distribution.
    constexpr int trials = 200000;
    for (int i = 0; i < trials; ++i) {
        const int token = qwen38_draw_probabilities(draft, rng);
        ++counts[qwen38_verify_proposal(token, target, draft, rng, accepted)];
    }
    for (int i = 0; i < 3; ++i)
        check(std::fabs(static_cast<double>(counts[i]) / trials - target[i]) < 0.005,
              "DSpark acceptance plus residual retains the target distribution");
}

void test_dflash2_selector_distribution() {
    using namespace trtmc;
    Qwen38SamplingParams p;
    p.temperature = 1;
    p.top_k = 8;
    p.top_p = 0.1F;
    p.min_p = 0.9F;
    // Those target filters do not truncate the conditional proposal support.
    auto q = qwen38_dflash2_probabilities({std::log(0.6F), std::log(0.4F)}, {3, 7}, 10, p);
    check(std::fabs(q[3] - 0.6F) < 1e-6F && std::fabs(q[7] - 0.4F) < 1e-6F && q[0] == 0,
          "DFlash2 selector retains temperature-softmax over its sparse support");
    p.top_k = 1;
    q = qwen38_dflash2_probabilities({1, 1}, {7, 3}, 10, p);
    check(q[3] == 1 && q[7] == 0, "DFlash2 greedy tie selects smallest token ID");
    p.top_k = 8;
    p.temperature = 0.0001F;
    q = qwen38_dflash2_probabilities({2, 1}, {7, 3}, 10, p);
    check(q[7] == 1 && q[3] == 0, "DFlash2 near-zero temperature preserves source greedy boundary");
}

int main(int argc, char** argv) {
    if (argc != 1) {
        if (argc != 5) {
            std::cerr << "Usage: pipeline_test [BUNDLE RUNTIME_ROOT REQUESTS_JSON OUTPUT_JSON]\n";
            return 2;
        }
        try {
            return test_typed_bundle(argv[1], argv[2], argv[3], argv[4]);
        } catch (const std::exception& error) {
            std::cerr << error.what() << std::endl;
            return 1;
        }
    }
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        return 77;
    test_native_media_positions();
    test_native_media_bindings();
    test_dflash2_selector_distribution();
    test_dspark_rejection_sampling();
    test_mtp_draft_cache_isolation();
    test_mtp_sampling_lifecycle();
    test_argmax_recurrent();
    test_mamba_pipeline();
    test_rwkv_pipeline();
    test_hybrid_pipeline();
    test_generate_applies_chat_template();
    if (failures > 0) {
        std::cerr << failures << " FAILED\n";
        return failures;
    }
    return skipped ? 77 : 0;
}
