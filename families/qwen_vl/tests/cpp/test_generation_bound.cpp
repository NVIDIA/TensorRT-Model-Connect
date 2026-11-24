/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Generation stops when the KV cache is full instead of dropping the prompt's
// rows. A cache of L rows and a P-token prompt yield L - P + 1 new tokens: the
// final sampled token is never fed back, so it needs no row. A longer prompt is
// rejected before the decoder runs; a state without a length limit is bounded by
// max_new_tokens alone. Fake modules with a never-EOS argmax drive
// the real pipeline and cache through the text path and the image path (whose
// prompt takes the token-by-token prefill because there is no prefill engine).

#include "families/qwen_vl/runtime/image_preprocessor.h"
#include "families/qwen_vl/runtime/kv_cache.h"
#include "families/qwen_vl/runtime/pipeline.h"
#include "families/qwen_vl/runtime/tokenizer.h"
#include "trtmc/internal/config.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::int32_t kRows = 6;
constexpr std::int32_t kKvDim = 4;
constexpr std::int32_t kImageToken = 9;
constexpr std::int32_t kFeatureDim = 4;
int failures = 0;

void check(bool condition, const std::string& name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

// Shared ITrtModule surface; each fake overrides only forward and its inputs.
class FakeModule : public trtmc::ITrtModule {
  public:
    explicit FakeModule(cudaStream_t stream) : stream_(stream) {}
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap& inputs) override { (void)forward(inputs); }
    void sync() override {}
    cudaStream_t stream() const override { return stream_; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    std::int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_output(const std::string&) const override { return false; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<std::int64_t> input_profile_shape(const std::string&, std::int32_t,
                                                  trtmc::ProfileShapeSelector) const override {
        return {};
    }
    std::int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<std::int64_t>&) override {}
    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    cudaStream_t stream_{nullptr};
};

// One-layer decoder whose logits always pick token 3, which is never EOS.
class FakeDecoder final : public FakeModule {
  public:
    using FakeModule::FakeModule;
    int calls{0};
    bool mask_width_ok{true};

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        const auto mask = inputs.find("attention_mask");
        mask_width_ok = mask_width_ok && mask != inputs.end() &&
                        mask->second.shape == std::vector<std::int64_t>{1, kRows + 1};
        return {{"logits", trtmc::Tensor{logits_.data(), {1, 4}, trtmc::DType::kFloat32}}};
    }
    bool has_input(const std::string& name) const override {
        return name == "token_id" || name == "position_id" || name == "attention_mask" ||
               name == "cache_k_0" || name == "cache_v_0";
    }
    std::vector<std::int64_t> tensor_shape(const std::string& name) const override {
        if (name == "cache_k_0" || name == "cache_v_0")
            return {kRows, kKvDim};
        return {};
    }
    std::int32_t input_rank(const std::string& name) const override {
        return name == "token_id" || name == "position_id" ? 1 : 2;
    }

  private:
    std::vector<float> logits_{0.1F, 0.2F, 0.3F, 0.9F};
};

// Vision encoder returning two merged image features.
class FakeVision final : public FakeModule {
  public:
    using FakeModule::FakeModule;
    trtmc::TensorMap forward(const trtmc::TensorMap&) override {
        return {{"image_features",
                 trtmc::Tensor{features_.data(), {2, kFeatureDim}, trtmc::DType::kFloat32}}};
    }
    bool has_input(const std::string&) const override { return false; }
    std::vector<std::int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::int32_t input_rank(const std::string&) const override { return 2; }

  private:
    std::vector<float> features_ = std::vector<float>(2 * kFeatureDim, 0.5F);
};

// Returns the prompt IDs chosen by the test, two of them image tokens.
class FixedTokenizer final : public trtmc::ITokenizer {
  public:
    explicit FixedTokenizer(std::vector<std::int32_t>* ids) : ids_(ids) {}
    std::vector<std::int32_t> encode(const std::string&) const override { return *ids_; }
    std::string decode(const std::vector<std::int32_t>&) const override { return "out"; }
    std::int32_t id_for_token(std::string_view) const override { return -1; }
    std::string token_for_id(std::int32_t) const override { return ""; }

  private:
    std::vector<std::int32_t>* ids_;
};

// State with no length limit: max_length() is -1, as recurrent state reports.
class UnboundedState final : public trtmc::QwenVlInferenceState {
  public:
    void reset() override { position_ = 0; }
    void bind_to(trtmc::ITrtModule&) override {}
    void prepare_step(trtmc::TensorMap&, std::int32_t) override {}
    void advance(std::int32_t n_tokens) override { position_ += n_tokens; }
    std::int32_t position() const override { return position_; }
    std::int32_t max_length() const override { return -1; }
    std::int32_t num_layers() const override { return 1; }
    bool needs_attention_mask() const override { return false; }
    std::size_t device_memory_bytes() const override { return 0; }
    const char* state_type() const override { return "unbounded"; }
    bool ok() const override { return true; }

  private:
    std::int32_t position_{0};
};

struct Harness {
    FakeDecoder* decoder{nullptr};
    std::vector<std::int32_t> prompt;
    std::unique_ptr<trtmc::QwenVlPipeline> pipeline;
};

// Without a state, the pipeline gets a kRows-row KV cache.
std::unique_ptr<Harness>
make_harness(cudaStream_t stream, std::unique_ptr<trtmc::QwenVlInferenceState> state = nullptr) {
    auto harness = std::make_unique<Harness>();
    auto decoder = std::make_unique<FakeDecoder>(stream);
    harness->decoder = decoder.get();
    trtmc::QwenVlConfig config;
    config.vocab_size = 4;
    config.id_eos = 0;
    config.id_eos_ids = {0};
    config.image_token_id = kImageToken;
    config.vision_output_dim = kFeatureDim;
    config.num_layers = 1;
    config.prefill_max_length = kRows;
    trtmc::QwenVlPreprocessConfig preprocess;
    preprocess.fixed_image_size = 28;
    preprocess.num_image_pad_tokens = 2;
    preprocess.image_token_id = kImageToken;
    preprocess.vision_output_dim = kFeatureDim;
    if (!state)
        state = std::make_unique<trtmc::QwenVlKvCache>(1, kRows, kKvDim, stream);
    harness->pipeline = std::make_unique<trtmc::QwenVlPipeline>(
        std::move(decoder), std::make_unique<FakeVision>(stream), std::move(state),
        std::move(config), std::move(preprocess), stream,
        std::make_shared<FixedTokenizer>(&harness->prompt));
    return harness;
}

trtmc::TextGenerationConfig budget(std::int32_t max_new_tokens) {
    trtmc::TextGenerationConfig config;
    config.max_new_tokens = max_new_tokens;
    config.top_k = 1;
    return config;
}

void test_text_path(cudaStream_t stream) {
    for (const std::int32_t length : {kRows - 1, kRows}) {
        auto harness = make_harness(stream);
        const std::vector<std::int32_t> prompt(static_cast<std::size_t>(length), 1);
        const auto result = harness->pipeline->generate_ids(prompt, budget(100)).token_ids;
        const auto label = "text prompt of " + std::to_string(length) + " tokens";
        check(result.size() == prompt.size() + static_cast<std::size_t>(kRows - length + 1),
              label + " generates until the cache is full");
        check(harness->decoder->calls == kRows, label + " runs one decoder step per cache row");
        check(harness->decoder->mask_width_ok, label + " keeps the decode mask contract");
    }
    // A budget below the remaining rows still ends at max_new_tokens.
    auto short_budget = make_harness(stream);
    const auto result = short_budget->pipeline->generate_ids({1, 1}, budget(2)).token_ids;
    check(result.size() == 4 && short_budget->decoder->calls < kRows,
          "max_new_tokens ends generation before the cache fills");

    auto overflow = make_harness(stream);
    bool rejected = false;
    try {
        (void)overflow->pipeline->generate_ids(std::vector<std::int32_t>(kRows + 1, 1), budget(4));
    } catch (const trtmc::internal::ConfigError&) {
        rejected = true;
    }
    check(rejected, "a text prompt longer than the cache is rejected as invalid");
    check(overflow->decoder->calls == 0, "a rejected text prompt never reaches the decoder");
}

void test_image_path(cudaStream_t stream) {
    const std::vector<float> pixels(28 * 28 * 3, 0.5F);
    for (const std::int32_t length : {kRows - 1, kRows}) {
        auto harness = make_harness(stream);
        harness->prompt.assign(static_cast<std::size_t>(length), 1);
        harness->prompt[1] = kImageToken;
        harness->prompt[2] = kImageToken;
        const auto result = harness->pipeline->generate("q", pixels.data(), 28, 28, budget(100));
        const auto label = "image prompt of " + std::to_string(length) + " tokens";
        check(result.token_ids.size() == static_cast<std::size_t>(kRows - length + 1),
              label + " generates until the cache is full");
        check(harness->decoder->calls == kRows, label + " runs one decoder step per cache row");
    }
    auto overflow = make_harness(stream);
    overflow->prompt.assign(kRows + 1, 1);
    overflow->prompt[1] = kImageToken;
    overflow->prompt[2] = kImageToken;
    bool rejected = false;
    try {
        (void)overflow->pipeline->generate("q", pixels.data(), 28, 28, budget(4));
    } catch (const trtmc::internal::ConfigError&) {
        rejected = true;
    }
    check(rejected, "an image prompt longer than the cache is rejected as invalid");
    check(overflow->decoder->calls == 0, "a rejected image prompt never reaches the decoder");
}

void test_unbounded_state(cudaStream_t stream) {
    // A state without a length limit never fills, so only max_new_tokens ends
    // generation, even past the KV cache's row count.
    constexpr std::int32_t kBudget = 2 * kRows;
    auto harness = make_harness(stream, std::make_unique<UnboundedState>());
    const auto result = harness->pipeline->generate_ids({1, 1}, budget(kBudget)).token_ids;
    check(result.size() == static_cast<std::size_t>(2 + kBudget),
          "an unbounded state generates until max_new_tokens");
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 1)
        return 77;
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess)
        return 77;
    try {
        test_text_path(stream);
        test_image_path(stream);
        test_unbounded_state(stream);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        ++failures;
    }
    cudaStreamDestroy(stream);
    return failures == 0 ? 0 : 1;
}
