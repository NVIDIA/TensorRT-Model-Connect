/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/bark/runtime/kv_cache.h"
#include "families/bark/runtime/pipeline.h"
#include "trtmc/internal/audio.h"
#include "trtmc/internal/config.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

struct ModuleStats {
    int32_t calls{0};
    std::vector<int64_t> input_embed_shape;
};

class CountingModule final : public trtmc::ITrtModule {
  public:
    CountingModule(std::shared_ptr<ModuleStats> stats, std::vector<float> logits,
                   int32_t cache_length, cudaStream_t stream)
        : stats_(std::move(stats)), logits_(std::move(logits)), cache_length_(cache_length),
          stream_(stream) {}

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++stats_->calls;
        const auto found = inputs.find("input_embed");
        stats_->input_embed_shape =
            found == inputs.end() ? std::vector<int64_t>{} : found->second.shape;
        return {{"logits", trtmc::Tensor{logits_.data(),
                                         {1, static_cast<int64_t>(logits_.size())},
                                         trtmc::DType::kFloat32}}};
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap& inputs) override { (void)forward(inputs); }
    void sync() override {}
    cudaStream_t stream() const override { return stream_; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override {
        return name == "input_embed" || name == "position_id" || name == "attention_mask" ||
               name == "cache_k_0" || name == "cache_v_0";
    }
    bool has_output(const std::string& name) const override { return name == "logits"; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (name == "cache_k_0" || name == "cache_v_0")
            return {cache_length_, 0};
        return {};
    }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string& name, void* pointer,
                       const std::vector<int64_t>&) override {
        bind_external(name, pointer);
    }
    int32_t input_rank(const std::string& name) const override {
        return name == "position_id" ? 1 : 2;
    }
    bool input_is_dynamic(const std::string& name) const override { return name == "input_embed"; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    std::shared_ptr<ModuleStats> stats_;
    std::vector<float> logits_;
    int32_t cache_length_{0};
    cudaStream_t stream_{nullptr};
};

trtmc::BarkConfig config(int32_t semantic_output_vocab = 10048) {
    trtmc::BarkConfig value;
    value.hidden_size = 4;
    value.text_pad_token = 5;
    value.semantic_pad_token = 3;
    value.semantic_infer_token = 4;
    value.semantic_input_vocab = 6;
    value.semantic_output_vocab = semantic_output_vocab;
    value.semantic_vocab_size = 4;
    value.n_coarse_codebooks = 2;
    value.codebook_size = 4;
    value.coarse_semantic_pad_token = 10;
    value.coarse_infer_token = 9;
    value.max_coarse_input_length = 4;
    value.max_coarse_history = 4;
    value.sliding_window_len = 10;
    value.greedy = true;
    return value;
}

std::unique_ptr<trtmc::BarkKvCache> cache(int32_t max_length, cudaStream_t stream) {
    trtmc::BarkKvCacheNames names;
    names.cache_k = {"cache_k_0"};
    names.cache_v = {"cache_v_0"};
    names.present_k = {"present_k_0"};
    names.present_v = {"present_v_0"};
    return std::make_unique<trtmc::BarkKvCache>(0, max_length, 0, stream, trtmc::DType::kFloat32,
                                                std::move(names));
}

std::unique_ptr<CountingModule> module(std::shared_ptr<ModuleStats> stats,
                                       const std::vector<float>& logits, int32_t cache_length,
                                       cudaStream_t stream) {
    return std::make_unique<CountingModule>(std::move(stats), logits, cache_length, stream);
}

void test_bark_generate_audio() {
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    const std::vector<float> semantic_logits = {0.9F, 0.8F, 0.7F, 0.1F, 0.0F};
    const std::vector<float> coarse_logits(12, 0.1F);
    std::vector<float> semantic_embed(6 * 4, 0.1F);
    std::vector<float> coarse_embed(11 * 4, 0.1F);

    trtmc::BarkPipeline pipeline(
        module(std::make_shared<ModuleStats>(), semantic_logits, 512, stream),
        module(std::make_shared<ModuleStats>(), coarse_logits, 16, stream), cache(512, stream),
        cache(16, stream), std::move(semantic_embed), std::move(coarse_embed), config(), stream);

    trtmc::internal::ITextToAudio& audio = pipeline;
    trtmc::internal::TextToAudioRequest request;
    request.prompt = "";
    const trtmc::internal::ConfigEntry initial_entries[] = {
        {"max_new_tokens", trtmc::internal::ConfigValue{std::int64_t{1}}},
    };
    trtmc::internal::ConfigView initial_cfg{initial_entries, 1};
    const auto output = audio.run(request, initial_cfg);

    check(!output.samples.empty(), "bark run produces samples");
    check(output.sample_rate == 24000, "bark run sample_rate");

    const auto bindings = pipeline.task_bindings();
    check(bindings.size() == 1, "bark task bindings count");
    check(bindings[0].fields.size() == 2, "bark task bindings declared fields");

    const trtmc::internal::ConfigEntry custom_entries[] = {
        {"max_new_tokens", trtmc::internal::ConfigValue{std::int64_t{10}}},
        {"seed", trtmc::internal::ConfigValue{std::int64_t{42}}},
    };
    trtmc::internal::ConfigView custom_cfg{custom_entries, 2};
    const auto custom_output = audio.run(request, custom_cfg);
    check(!custom_output.samples.empty(), "bark run custom config produces samples");

    cudaStreamDestroy(stream);
}

void test_bark_seed_handling() {
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    const std::vector<float> semantic_logits = {0.5F, 0.4F, 0.3F, 0.2F, 0.1F};
    const std::vector<float> coarse_logits(12, 0.1F);
    std::vector<float> semantic_embed(6 * 4, 0.1F);
    std::vector<float> coarse_embed(11 * 4, 0.1F);

    auto cfg = config();
    cfg.greedy = false;

    trtmc::BarkPipeline pipeline(
        module(std::make_shared<ModuleStats>(), semantic_logits, 512, stream),
        module(std::make_shared<ModuleStats>(), coarse_logits, 16, stream), cache(512, stream),
        cache(16, stream), std::move(semantic_embed), std::move(coarse_embed), cfg, stream);

    trtmc::internal::ITextToAudio& audio = pipeline;
    trtmc::internal::TextToAudioRequest request;
    request.prompt = "";

    const std::int64_t seed_a = 42;
    const trtmc::internal::ConfigEntry entries_seed_a[] = {
        {"max_new_tokens", trtmc::internal::ConfigValue{std::int64_t{5}}},
        {"seed", trtmc::internal::ConfigValue{seed_a}},
    };
    trtmc::internal::ConfigView cfg_seed_a{entries_seed_a, 2};
    const auto output_a1 = audio.run(request, cfg_seed_a);
    const auto output_a2 = audio.run(request, cfg_seed_a);

    check(!output_a1.samples.empty(), "bark non-greedy run produces samples");
    check(output_a1.samples == output_a2.samples,
          "bark non-greedy run is deterministic for identical seed");

    const std::int64_t seed_b = seed_a | (std::int64_t{1} << 35);
    const trtmc::internal::ConfigEntry entries_seed_b[] = {
        {"max_new_tokens", trtmc::internal::ConfigValue{std::int64_t{5}}},
        {"seed", trtmc::internal::ConfigValue{seed_b}},
    };
    trtmc::internal::ConfigView cfg_seed_b{entries_seed_b, 2};
    const auto output_b = audio.run(request, cfg_seed_b);

    check(!output_b.samples.empty(), "bark non-greedy run with 64-bit seed produces samples");
    check(output_a1.samples != output_b.samples,
          "bark non-greedy run distinguishes seeds differing only in upper 32 bits");

    bool threw_invalid_seed = false;
    try {
        const trtmc::internal::ConfigEntry entries_invalid[] = {
            {"max_new_tokens", trtmc::internal::ConfigValue{std::int64_t{1}}},
            {"seed", trtmc::internal::ConfigValue{std::int64_t{-2}}},
        };
        trtmc::internal::ConfigView cfg_invalid{entries_invalid, 2};
        (void)audio.run(request, cfg_invalid);
    } catch (const trtmc::internal::ConfigError&) {
        threw_invalid_seed = true;
    }
    check(threw_invalid_seed, "bark run rejects seed < -1 with ConfigError");

    cudaStreamDestroy(stream);
}

void test_bark_batches_semantic_and_coarse_prefill() {
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    const std::vector<float> semantic_logits = {0.9F, 0.8F, 0.7F, 0.1F, 0.0F};
    const std::vector<float> coarse_logits(12, 0.1F);
    std::vector<float> semantic_embed(6 * 4, 0.1F);
    std::vector<float> coarse_embed(11 * 4, 0.1F);

    trtmc::BarkPipeline pipeline(
        module(std::make_shared<ModuleStats>(), semantic_logits, 512, stream),
        module(std::make_shared<ModuleStats>(), coarse_logits, 16, stream), cache(512, stream),
        cache(16, stream), std::move(semantic_embed), std::move(coarse_embed), config(), stream);
    auto semantic_stats = std::make_shared<ModuleStats>();
    auto coarse_stats = std::make_shared<ModuleStats>();
    pipeline.set_prefill_modules(module(semantic_stats, semantic_logits, 512, stream),
                                 module(coarse_stats, coarse_logits, 16, stream));

    trtmc::internal::TextToAudioRequest request;
    request.prompt = "";
    const trtmc::internal::ConfigEntry entries[] = {
        {"max_new_tokens", trtmc::internal::ConfigValue{std::int64_t{1}}},
    };
    trtmc::internal::ConfigView cfg{entries, 1};
    (void)pipeline.run(request, cfg);

    check(semantic_stats->calls == 1, "bark semantic prefill uses one batched call");
    check(semantic_stats->input_embed_shape == std::vector<int64_t>({257, 4}),
          "bark semantic prefill batches 256 text slots plus infer token");
    check(coarse_stats->calls == 1, "bark coarse prefill uses one batched call per window");
    check(coarse_stats->input_embed_shape.size() == 2 && coarse_stats->input_embed_shape[0] > 1 &&
              coarse_stats->input_embed_shape[1] == 4,
          "bark coarse prefill batches the complete window context");
    cudaStreamDestroy(stream);
}

void test_bark_dual_profile_decode_uses_one_embedding_row() {
    const std::vector<float> semantic_logits = {0.9F, 0.8F, 0.7F, 0.1F, 0.0F};
    const std::vector<float> coarse_logits(12, 0.1F);
    auto semantic_decode_stats = std::make_shared<ModuleStats>();
    auto coarse_decode_stats = std::make_shared<ModuleStats>();
    auto semantic_prefill_stats = std::make_shared<ModuleStats>();
    auto coarse_prefill_stats = std::make_shared<ModuleStats>();
    std::vector<float> semantic_embed(6 * 4, 0.1F);
    std::vector<float> coarse_embed(11 * 4, 0.1F);

    trtmc::BarkPipeline pipeline(module(semantic_decode_stats, semantic_logits, 512, nullptr),
                                 module(coarse_decode_stats, coarse_logits, 16, nullptr),
                                 cache(512, nullptr), cache(16, nullptr), std::move(semantic_embed),
                                 std::move(coarse_embed), config(5), nullptr);
    pipeline.set_prefill_modules(module(semantic_prefill_stats, semantic_logits, 512, nullptr),
                                 module(coarse_prefill_stats, coarse_logits, 16, nullptr));

    trtmc::internal::TextToAudioRequest request;
    request.prompt = "";
    const trtmc::internal::ConfigEntry entries[] = {
        {"max_new_tokens", trtmc::internal::ConfigValue{std::int64_t{2}}},
    };
    trtmc::internal::ConfigView cfg{entries, 1};
    (void)pipeline.run(request, cfg);

    check(semantic_decode_stats->calls == 2,
          "bark embed-only semantic engine decodes each generated token");
    check(semantic_decode_stats->input_embed_shape == std::vector<int64_t>({1, 4}),
          "bark semantic decode uses one rank-2 embedding row");
    check(coarse_decode_stats->calls > 0,
          "bark embed-only coarse engine decodes after batched prefill");
    check(coarse_decode_stats->input_embed_shape == std::vector<int64_t>({1, 4}),
          "bark coarse decode uses one rank-2 embedding row");
}

void test_bark_constructor_validates_semantic() {
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    std::vector<float> semantic_embed(6 * 4, 0.1F);
    std::vector<float> coarse_embed(11 * 4, 0.1F);
    bool threw = false;
    try {
        trtmc::BarkPipeline pipeline(
            nullptr,
            module(std::make_shared<ModuleStats>(), std::vector<float>(12, 0.1F), 16, stream),
            cache(512, stream), cache(16, stream), std::move(semantic_embed),
            std::move(coarse_embed), config(), stream);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "bark constructor rejects null semantic module");
    cudaStreamDestroy(stream);
}

void test_bark_constructor_validates_embed() {
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    const std::vector<float> semantic_logits = {0.9F, 0.8F, 0.7F, 0.1F, 0.0F};
    const std::vector<float> coarse_logits(12, 0.1F);
    bool threw = false;
    try {
        trtmc::BarkPipeline pipeline(
            module(std::make_shared<ModuleStats>(), semantic_logits, 512, stream),
            module(std::make_shared<ModuleStats>(), coarse_logits, 16, stream), cache(512, stream),
            cache(16, stream), {}, std::vector<float>(11 * 4, 0.1F), config(), stream);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "bark constructor rejects empty semantic embed");
    cudaStreamDestroy(stream);
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: CUDA device is unavailable\n";
        return 77;
    }
    test_bark_generate_audio();
    test_bark_seed_handling();
    test_bark_batches_semantic_and_coarse_prefill();
    test_bark_dual_profile_decode_uses_one_embedding_row();
    test_bark_constructor_validates_semantic();
    test_bark_constructor_validates_embed();
    if (failures > 0)
        std::cerr << failures << " bark pipeline test(s) FAILED\n";
    return failures;
}
