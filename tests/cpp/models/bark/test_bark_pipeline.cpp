/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../support/mock_trt_engines.h"
#include "runtime/backend/trt_module_impl.h"
#include "runtime/models/bark/bark_config.h"
#include "runtime/models/bark/kv_cache.h"
#include "runtime/models/bark/pipeline.h"

#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

struct PrefillStats {
    int32_t calls{0};
    std::vector<int64_t> input_embed_shape;
};

class CountingPrefillModule final : public trtmc::ITrtModule {
  public:
    CountingPrefillModule(std::shared_ptr<PrefillStats> stats, std::vector<float> logits,
                          cudaStream_t stream)
        : stats_(std::move(stats)), logits_(std::move(logits)), stream_(stream) {}

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
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override {
        return name == "input_embed" || name == "position_id" || name == "attention_mask";
    }
    bool has_output(const std::string& name) const override { return name == "logits"; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    int32_t input_rank(const std::string& name) const override {
        return name == "input_embed" || name == "attention_mask" ? 2 : 1;
    }
    bool input_is_dynamic(const std::string&) const override { return true; }
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    std::shared_ptr<PrefillStats> stats_;
    std::vector<float> logits_;
    cudaStream_t stream_{nullptr};
};

void test_bark_generate_audio() {
    trtmc::BarkConfig bcfg;
    bcfg.hidden_size = 4;
    bcfg.text_pad_token = 5;
    bcfg.semantic_pad_token = 3;
    bcfg.semantic_infer_token = 4;
    bcfg.semantic_input_vocab = 6;
    bcfg.semantic_output_vocab = 10048;
    bcfg.semantic_vocab_size = 4;
    bcfg.n_coarse_codebooks = 2;
    bcfg.codebook_size = 4;
    bcfg.coarse_semantic_pad_token = 10;
    bcfg.coarse_infer_token = 9;
    bcfg.max_coarse_input_length = 4;
    bcfg.max_coarse_history = 4;
    bcfg.sliding_window_len = 10;
    bcfg.greedy = true;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto sem_cache = std::make_unique<trtmc::BarkKvCache>(0, 512, 0, stream);
    auto coarse_cache = std::make_unique<trtmc::BarkKvCache>(0, 16, 0, stream);

    check(sem_cache->ok(), "bark semantic cache ok");
    check(coarse_cache->ok(), "bark coarse cache ok");

    const std::vector<float> sem_logits = {0.9F, 0.8F, 0.7F, 0.1F, 0.0F};
    auto sem_engine = trtmc::test::build_mock_mask_only_engine(513, 5, sem_logits);
    if (!sem_engine) {
        std::cerr << "WARNING: Could not build mock semantic engine, skipping\n";
        cudaStreamDestroy(stream);
        return;
    }

    const std::vector<float> coarse_logits(12, 0.1F);
    auto coarse_engine = trtmc::test::build_mock_mask_only_engine(17, 12, coarse_logits);
    if (!coarse_engine) {
        std::cerr << "WARNING: Could not build mock coarse engine, skipping\n";
        cudaStreamDestroy(stream);
        return;
    }

    auto semantic = std::make_unique<trtmc::TrtModuleImpl>(
        sem_engine.get(), sem_engine->createExecutionContext(), stream);
    auto coarse = std::make_unique<trtmc::TrtModuleImpl>(
        coarse_engine.get(), coarse_engine->createExecutionContext(), stream);

    std::vector<float> sem_embed(6 * 4, 0.1F);
    std::vector<float> coarse_embed(11 * 4, 0.1F);

    trtmc::BarkPipeline pipeline(std::move(semantic), std::move(coarse), std::move(sem_cache),
                                 std::move(coarse_cache), sem_embed, coarse_embed, bcfg, stream);

    check(std::string(pipeline.pipeline_type()) == "BarkPipeline", "bark pipeline_type");

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 1;
    auto out = pipeline.generate_audio("", gen_cfg);

    check(out.num_samples > 0, "bark generate_audio produces samples");
    check(out.sample_rate == 24000, "bark generate_audio sample_rate");

    cudaStreamDestroy(stream);
}

void test_bark_batches_semantic_and_coarse_prefill() {
    trtmc::BarkConfig bcfg;
    bcfg.hidden_size = 4;
    bcfg.text_pad_token = 5;
    bcfg.semantic_pad_token = 3;
    bcfg.semantic_infer_token = 4;
    bcfg.semantic_input_vocab = 6;
    bcfg.semantic_output_vocab = 10048;
    bcfg.semantic_vocab_size = 4;
    bcfg.n_coarse_codebooks = 2;
    bcfg.codebook_size = 4;
    bcfg.coarse_semantic_pad_token = 10;
    bcfg.coarse_infer_token = 9;
    bcfg.max_coarse_input_length = 4;
    bcfg.max_coarse_history = 4;
    bcfg.sliding_window_len = 10;
    bcfg.greedy = true;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto sem_cache = std::make_unique<trtmc::BarkKvCache>(0, 512, 0, stream);
    auto coarse_cache = std::make_unique<trtmc::BarkKvCache>(0, 16, 0, stream);
    const std::vector<float> sem_logits = {0.9F, 0.8F, 0.7F, 0.1F, 0.0F};
    const std::vector<float> coarse_logits(12, 0.1F);
    auto sem_engine = trtmc::test::build_mock_mask_only_engine(513, 5, sem_logits);
    auto coarse_engine = trtmc::test::build_mock_mask_only_engine(17, 12, coarse_logits);
    if (!sem_engine || !coarse_engine) {
        cudaStreamDestroy(stream);
        return;
    }

    auto semantic = std::make_unique<trtmc::TrtModuleImpl>(
        sem_engine.get(), sem_engine->createExecutionContext(), stream);
    auto coarse = std::make_unique<trtmc::TrtModuleImpl>(
        coarse_engine.get(), coarse_engine->createExecutionContext(), stream);
    std::vector<float> sem_embed(6 * 4, 0.1F);
    std::vector<float> coarse_embed(11 * 4, 0.1F);
    trtmc::BarkPipeline pipeline(std::move(semantic), std::move(coarse), std::move(sem_cache),
                                 std::move(coarse_cache), sem_embed, coarse_embed, bcfg, stream);

    auto semantic_stats = std::make_shared<PrefillStats>();
    auto coarse_stats = std::make_shared<PrefillStats>();
    pipeline.set_prefill_modules(
        std::make_unique<CountingPrefillModule>(semantic_stats, sem_logits, stream),
        std::make_unique<CountingPrefillModule>(coarse_stats, coarse_logits, stream));

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 1;
    (void)pipeline.generate_audio("", gen_cfg);

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
    trtmc::BarkConfig bcfg;
    bcfg.hidden_size = 4;
    bcfg.text_pad_token = 5;
    bcfg.semantic_pad_token = 3;
    bcfg.semantic_infer_token = 4;
    bcfg.semantic_input_vocab = 6;
    bcfg.semantic_output_vocab = 5;
    bcfg.semantic_vocab_size = 4;
    bcfg.n_coarse_codebooks = 2;
    bcfg.codebook_size = 4;
    bcfg.coarse_semantic_pad_token = 10;
    bcfg.coarse_infer_token = 9;
    bcfg.max_coarse_input_length = 4;
    bcfg.max_coarse_history = 4;
    bcfg.sliding_window_len = 10;
    bcfg.greedy = true;

    cudaStream_t stream = nullptr;
    auto sem_cache = std::make_unique<trtmc::BarkKvCache>(0, 512, 0, stream);
    auto coarse_cache = std::make_unique<trtmc::BarkKvCache>(0, 16, 0, stream);
    const std::vector<float> sem_logits = {0.9F, 0.8F, 0.7F, 0.1F, 0.0F};
    const std::vector<float> coarse_logits(12, 0.1F);

    auto semantic_decode_stats = std::make_shared<PrefillStats>();
    auto coarse_decode_stats = std::make_shared<PrefillStats>();
    auto semantic_prefill_stats = std::make_shared<PrefillStats>();
    auto coarse_prefill_stats = std::make_shared<PrefillStats>();
    auto semantic =
        std::make_unique<CountingPrefillModule>(semantic_decode_stats, sem_logits, stream);
    auto coarse =
        std::make_unique<CountingPrefillModule>(coarse_decode_stats, coarse_logits, stream);

    std::vector<float> sem_embed(6 * 4, 0.1F);
    std::vector<float> coarse_embed(11 * 4, 0.1F);
    trtmc::BarkPipeline pipeline(std::move(semantic), std::move(coarse), std::move(sem_cache),
                                 std::move(coarse_cache), sem_embed, coarse_embed, bcfg, stream);
    pipeline.set_prefill_modules(
        std::make_unique<CountingPrefillModule>(semantic_prefill_stats, sem_logits, stream),
        std::make_unique<CountingPrefillModule>(coarse_prefill_stats, coarse_logits, stream));

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 2;
    (void)pipeline.generate_audio("", gen_cfg);

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
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto coarse_cache = std::make_unique<trtmc::BarkKvCache>(0, 16, 0, stream);
    auto sem_cache = std::make_unique<trtmc::BarkKvCache>(0, 512, 0, stream);

    const std::vector<float> coarse_logits(12, 0.1F);
    auto coarse_engine = trtmc::test::build_mock_mask_only_engine(17, 12, coarse_logits);
    if (!coarse_engine) {
        cudaStreamDestroy(stream);
        return;
    }
    auto coarse = std::make_unique<trtmc::TrtModuleImpl>(
        coarse_engine.get(), coarse_engine->createExecutionContext(), stream);

    std::vector<float> sem_embed(24, 0.1F);
    std::vector<float> coarse_embed(44, 0.1F);

    bool threw = false;
    try {
        trtmc::BarkPipeline pipeline(nullptr, std::move(coarse), std::move(sem_cache),
                                     std::move(coarse_cache), sem_embed, coarse_embed,
                                     trtmc::BarkConfig{}, stream);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "bark constructor rejects null semantic module");

    cudaStreamDestroy(stream);
}

void test_bark_constructor_validates_embed() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const std::vector<float> sem_logits = {0.9F, 0.8F, 0.7F, 0.1F, 0.0F};
    auto sem_engine = trtmc::test::build_mock_mask_only_engine(513, 5, sem_logits);
    const std::vector<float> coarse_logits(12, 0.1F);
    auto coarse_engine = trtmc::test::build_mock_mask_only_engine(17, 12, coarse_logits);

    if (!sem_engine || !coarse_engine) {
        cudaStreamDestroy(stream);
        return;
    }

    auto semantic = std::make_unique<trtmc::TrtModuleImpl>(
        sem_engine.get(), sem_engine->createExecutionContext(), stream);
    auto coarse = std::make_unique<trtmc::TrtModuleImpl>(
        coarse_engine.get(), coarse_engine->createExecutionContext(), stream);
    auto sem_cache = std::make_unique<trtmc::BarkKvCache>(0, 512, 0, stream);
    auto coarse_cache = std::make_unique<trtmc::BarkKvCache>(0, 16, 0, stream);

    bool threw = false;
    try {
        std::vector<float> empty_embed;
        std::vector<float> coarse_embed(44, 0.1F);
        trtmc::BarkPipeline pipeline(std::move(semantic), std::move(coarse), std::move(sem_cache),
                                     std::move(coarse_cache), empty_embed, coarse_embed,
                                     trtmc::BarkConfig{}, stream);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "bark constructor rejects empty semantic embed");

    cudaStreamDestroy(stream);
}

} // namespace

int main() {
    test_bark_generate_audio();
    test_bark_batches_semantic_and_coarse_prefill();
    test_bark_dual_profile_decode_uses_one_embedding_row();
    test_bark_constructor_validates_semantic();
    test_bark_constructor_validates_embed();
    if (failures > 0) {
        std::cerr << failures << " bark pipeline test(s) FAILED\n";
    }
    return failures;
}
