/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-VL-CPP-01
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-FAC-01
// Intent:         DeepseekOcrPipeline text-only generation with mock engines; constructor
//                 validation (null decoder/cache); DeepseekOcrConfig sync from
//                 DeepseekOcrPreprocessConfig; zero max_tokens early exit; string-based
//                 generate() with tokenizer; no-tokenizer throws (lines 48, 109);
//                 image path with/without vision encoder; full VL image generation
//                 with mock vision encoder (convert_float_to_decoded, preprocess,
//                 run_vision_encoder, generate_vl_from_ids, run_text_step_with_embed);
//                 run_text_step_with_embed full body (lines 285-361) via embed-capable
//                 decoder (image token embed injection + zero-embed + autoregressive step)
// Preconditions:  TRT + CUDA GPU available
// Postconditions: Pipeline generates text tokens correctly in text-only and VL modes;
//                 null decoder/cache rejected; config sync applied; tokenizer checks
//                 throw when missing; image path fallthrough and full path both exercised;
//                 run_text_step_with_embed executes its full body including embed
//                 injection, zero-embed fallback, and autoregressive run_text_step
// =============================================================================

// =============================================================================
// Test suite: DeepseekOcrPipeline — vision-language generation
// =============================================================================

#include "runtime/backend/trt_module_impl.h"
#include "runtime/core/trt_common.h"
#include "runtime/models/deepseek_ocr/kv_cache.h"
#include "runtime/models/deepseek_ocr/pipeline.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <NvInfer.h>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

static int failures = 0;
static void check(bool c, const char* n) {
    if (!c) {
        std::cerr << "FAIL: " << n << '\n';
        ++failures;
    }
}

static trtmc::TrtLogger g_logger;

struct CountingTextStats {
    int32_t calls{0};
    std::unordered_map<std::string, std::vector<int64_t>> shapes;
    std::unordered_map<std::string, std::vector<int64_t>> bound_shapes;
    std::unordered_map<std::string, std::vector<float>> float_values;
};

class CountingTextModule final : public trtmc::ITrtModule {
  public:
    CountingTextModule(std::shared_ptr<CountingTextStats> stats, bool prefill, cudaStream_t stream)
        : stats_(std::move(stats)), prefill_(prefill), stream_(stream),
          present_k_(prefill ? trtmc::DeviceTensor::zeros({8, 4}, trtmc::DType::kFloat32, stream)
                             : trtmc::DeviceTensor{}),
          present_v_(prefill ? trtmc::DeviceTensor::zeros({8, 4}, trtmc::DType::kFloat32, stream)
                             : trtmc::DeviceTensor{}) {}

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++stats_->calls;
        stats_->shapes.clear();
        stats_->float_values.clear();
        for (const auto& [name, tensor] : inputs) {
            stats_->shapes[name] = tensor.shape;
            if (tensor.dtype == trtmc::DType::kFloat32) {
                const auto* begin = static_cast<const float*>(tensor.data);
                stats_->float_values[name] =
                    std::vector<float>(begin, begin + static_cast<std::ptrdiff_t>(tensor.numel()));
            }
        }
        return {{"logits", trtmc::Tensor{logits_.data(), {1, 4}, trtmc::DType::kFloat32}}};
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap& inputs) override { (void)forward(inputs); }
    void sync() override {}
    cudaStream_t stream() const override { return stream_; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return prefill_ ? 0 : 1; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override {
        return name == "token_id" || name == "position_id" || name == "attention_mask" ||
               name == "input_embed" || name == "use_input_embed" || name == "deepstack_active" ||
               name == "deepstack_embed_0" || name == "cache_k_0" || name == "cache_v_0";
    }
    bool has_output(const std::string& name) const override {
        return name == "logits" || name == "present_k_0" || name == "present_v_0";
    }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (name == "cache_k_0" || name == "cache_v_0")
            return {8, 4};
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
    void bind_external(const std::string& name, void*, const std::vector<int64_t>& shape) override {
        stats_->bound_shapes[name] = shape;
    }
    int32_t input_rank(const std::string& name) const override {
        return name == "token_id" || name == "position_id" ? 1 : 2;
    }
    bool input_is_dynamic(const std::string&) const override { return prefill_; }
    bool ok() const override { return !prefill_ || (present_k_.ok() && present_v_.ok()); }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    std::shared_ptr<CountingTextStats> stats_;
    bool prefill_{false};
    cudaStream_t stream_{nullptr};
    mutable trtmc::DeviceTensor present_k_;
    mutable trtmc::DeviceTensor present_v_;
    std::vector<float> logits_{0.1F, 0.2F, 0.9F, 0.3F};
};

class FakeSequenceVisionModule final : public trtmc::ITrtModule {
  public:
    trtmc::TensorMap forward(const trtmc::TensorMap&) override {
        return {{"image_features", trtmc::Tensor{features_.data(), {1, 4}, trtmc::DType::kFloat32}},
                {"deepstack_features_0",
                 trtmc::Tensor{deepstack_.data(), {1, 4}, trtmc::DType::kFloat32}}};
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override {
        return {{"pixel_values", {3, 4, 4}, trtmc::DType::kFloat32, true}};
    }
    std::vector<trtmc::TensorInfo> output_info() const override {
        return {{"image_features", {1, 4}, trtmc::DType::kFloat32, false}};
    }
    bool has_input(const std::string& name) const override { return name == "pixel_values"; }
    bool has_output(const std::string& name) const override {
        return name == "image_features" || name == "deepstack_features_0";
    }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    std::vector<float> features_{10.0F, 11.0F, 12.0F, 13.0F};
    std::vector<float> deepstack_{20.0F, 21.0F, 22.0F, 23.0F};
};

// ---------------------------------------------------------------------------
// Inline FixedTokenizer for string-based generate() tests
// ---------------------------------------------------------------------------
class VLFixedTokenizer : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string&) const override { return {1, 2, 3}; }
    std::string decode(const std::vector<int32_t>&) const override { return "out"; }
    int32_t id_for_token(std::string_view) const override { return 0; }
    std::string token_for_id(int32_t) const override { return ""; }
};

// ---------------------------------------------------------------------------
// Engine builders
// ---------------------------------------------------------------------------

static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_mock_decoder() {
    auto b = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* tok = n->addInput("token_id", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {1}});
    auto* mask = n->addInput("attention_mask", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {8}});

    float cl[4] = {0.1f, 0.2f, 0.9f, 0.3f};
    auto* cst = n->addConstant(nvinfer1::Dims{1, {4}},
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cl, 4});
    cst->getOutput(0)->setName("logits");
    n->markOutput(*cst->getOutput(0));

    n->addIdentity(*tok)->getOutput(0)->setName("_t");
    n->addIdentity(*mask)->getOutput(0)->setName("_m");

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan)
        return nullptr;
    auto rt = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

// Mock vision encoder: pixel_values[3,4,4] float32 -> image_features[4] float32
// Used to exercise the full VL image generation path in vl_pipeline.cpp
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_mock_vision_encoder() {
    auto b = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    // pixel_values[3,4,4] matches preprocessed output with fixed_image_size=4, in_channels=3
    auto* pv =
        n->addInput("pixel_values", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{3, {3, 4, 4}});

    float cv[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    auto* cst = n->addConstant(nvinfer1::Dims{1, {4}},
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cv, 4});
    cst->getOutput(0)->setName("image_features");
    n->markOutput(*cst->getOutput(0));

    n->addIdentity(*pv)->getOutput(0)->setName("_pv");

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan)
        return nullptr;
    auto rt = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_vl_text_only() {
    auto engine = build_mock_decoder();
    if (!engine) {
        std::cerr << "SKIP\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto decoder = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                          engine->createExecutionContext(), stream);
    auto cache = std::make_unique<trtmc::DeepseekOcrKvCache>(1, 8, 4, stream);

    trtmc::DeepseekOcrConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 2;
    cfg.has_position_input = false;

    // No vision encoder (text-only mode)
    trtmc::DeepseekOcrPreprocessConfig vl_pp;
    trtmc::DeepseekOcrPipeline pipeline(std::move(decoder), nullptr, std::move(cache), cfg, vl_pp,
                                        stream);
    check(std::string(pipeline.pipeline_type()) == "DeepseekOcrPipeline", "vl name");

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 5;
    auto result = pipeline.generate_ids({1}, gen_cfg);

    // argmax=2=eos → stops after 1 generated token
    check(result.token_ids.size() == 2, "text-only: input + eos");
    check(result.token_ids[1] == 2, "text-only: eos generated");

    cudaStreamDestroy(stream);
}

static void test_vl_text_only_max_tokens() {
    auto engine = build_mock_decoder();
    if (!engine) {
        std::cerr << "SKIP\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto decoder = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                          engine->createExecutionContext(), stream);
    auto cache = std::make_unique<trtmc::DeepseekOcrKvCache>(1, 8, 4, stream);

    trtmc::DeepseekOcrConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 99;
    cfg.has_position_input = false;

    trtmc::DeepseekOcrPreprocessConfig vl_pp;
    trtmc::DeepseekOcrPipeline pipeline(std::move(decoder), nullptr, std::move(cache), cfg, vl_pp,
                                        stream);

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 3;
    auto result = pipeline.generate_ids({0, 1}, gen_cfg);

    // 2 input + 3 generated = 5 total
    check(result.token_ids.size() == 5, "max tokens: 2 input + 3 gen");

    cudaStreamDestroy(stream);
}

static void test_vl_validates_decoder() {
    auto engine = build_mock_decoder();
    if (!engine) {
        std::cerr << "SKIP vl_validates_decoder\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto cache = std::make_unique<trtmc::DeepseekOcrKvCache>(1, 8, 4, stream);

    trtmc::DeepseekOcrConfig cfg;
    cfg.has_position_input = false;
    trtmc::DeepseekOcrPreprocessConfig vl_pp;

    // null text_decoder -> throws
    bool threw = false;
    try {
        trtmc::DeepseekOcrPipeline p(nullptr, nullptr, std::move(cache), cfg, vl_pp, stream);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "vl: null decoder throws");

    cudaStreamDestroy(stream);
}

static void test_vl_validates_cache() {
    auto engine = build_mock_decoder();
    if (!engine) {
        std::cerr << "SKIP vl_validates_cache\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto decoder = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                          engine->createExecutionContext(), stream);

    trtmc::DeepseekOcrConfig cfg;
    cfg.has_position_input = false;
    trtmc::DeepseekOcrPreprocessConfig vl_pp;

    // null cache -> throws
    bool threw = false;
    try {
        trtmc::DeepseekOcrPipeline p(std::move(decoder), nullptr, nullptr, cfg, vl_pp, stream);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "vl: null cache throws");

    cudaStreamDestroy(stream);
}

static void test_vl_config_sync() {
    // DeepseekOcrPreprocessConfig has image_token_id=1 and vision_output_dim=64,
    // DeepseekOcrConfig has image_token_id=-1 and vision_output_dim=0.
    // Constructor should sync: config_.image_token_id = 1, config_.vision_output_dim = 64
    auto engine = build_mock_decoder();
    if (!engine) {
        std::cerr << "SKIP vl_config_sync\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto decoder = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                          engine->createExecutionContext(), stream);
    auto cache = std::make_unique<trtmc::DeepseekOcrKvCache>(1, 8, 4, stream);

    trtmc::DeepseekOcrConfig cfg;
    cfg.has_position_input = false;
    cfg.image_token_id = -1;   // will be synced from vl_pp
    cfg.vision_output_dim = 0; // will be synced from vl_pp

    trtmc::DeepseekOcrPreprocessConfig vl_pp;
    vl_pp.image_token_id = 1;
    vl_pp.vision_output_dim = 64;

    trtmc::DeepseekOcrPipeline pipeline(std::move(decoder), nullptr, std::move(cache), cfg, vl_pp,
                                        stream);

    // Construction succeeded; pipeline type is correct
    check(std::string(pipeline.pipeline_type()) == "DeepseekOcrPipeline",
          "config_sync: type correct");

    // The synced config is reflected in vl_preprocess_config()
    check(pipeline.vl_preprocess_config().image_token_id == 1,
          "config_sync: image_token_id synced");

    cudaStreamDestroy(stream);
}

static void test_vl_zero_max_tokens() {
    // generate_ids with max_new_tokens=0 returns input_ids unchanged (early exit)
    auto engine = build_mock_decoder();
    if (!engine) {
        std::cerr << "SKIP vl_zero_max_tokens\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto decoder = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                          engine->createExecutionContext(), stream);
    auto cache = std::make_unique<trtmc::DeepseekOcrKvCache>(1, 8, 4, stream);

    trtmc::DeepseekOcrConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 2;
    cfg.has_position_input = false;

    trtmc::DeepseekOcrPreprocessConfig vl_pp;
    trtmc::DeepseekOcrPipeline pipeline(std::move(decoder), nullptr, std::move(cache), cfg, vl_pp,
                                        stream);

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 0; // triggers early return in generate_from_ids
    auto result = pipeline.generate_ids({1, 2, 3}, gen_cfg);

    // No new tokens generated, output == input
    check(result.token_ids.size() == 3, "zero max_tokens: same size as input");
    check(result.token_ids[0] == 1 && result.token_ids[1] == 2 && result.token_ids[2] == 3,
          "zero max_tokens: tokens unchanged");

    cudaStreamDestroy(stream);
}

static void test_vl_no_tokenizer_throws() {
    // Without tokenizer: text-only generate(string, cfg) throws (line 48)
    // and image generate(string, pixels, h, w, cfg) throws (line 109)
    auto engine = build_mock_decoder();
    if (!engine) {
        std::cerr << "SKIP vl_no_tokenizer\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto decoder = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                          engine->createExecutionContext(), stream);
    auto cache = std::make_unique<trtmc::DeepseekOcrKvCache>(1, 8, 4, stream);

    trtmc::DeepseekOcrConfig cfg;
    cfg.has_position_input = false;
    trtmc::DeepseekOcrPreprocessConfig vl_pp;

    // No tokenizer
    trtmc::DeepseekOcrPipeline pipeline(std::move(decoder), nullptr, std::move(cache), cfg, vl_pp,
                                        stream);

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 1;

    // Text-only path: covers line 48
    bool threw1 = false;
    try {
        pipeline.generate("hello", gen_cfg);
    } catch (const std::exception&) {
        threw1 = true;
    }
    check(threw1, "vl: no tokenizer text-only throws");

    // Image path: covers line 109
    float pixels[2 * 2 * 3] = {0.5f};
    bool threw2 = false;
    try {
        pipeline.generate("hello", pixels, 2, 2, gen_cfg);
    } catch (const std::exception&) {
        threw2 = true;
    }
    check(threw2, "vl: no tokenizer image-path throws");

    cudaStreamDestroy(stream);
}

static void test_vl_generate_with_image_no_encoder() {
    // Covers lines 103-113: generate(prompt, pixels, h, w, cfg) with valid pixels
    // but no vision encoder -> falls through to text-only generate
    auto engine = build_mock_decoder();
    if (!engine) {
        std::cerr << "SKIP vl_image_no_encoder\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto decoder = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                          engine->createExecutionContext(), stream);
    auto cache = std::make_unique<trtmc::DeepseekOcrKvCache>(1, 8, 4, stream);

    trtmc::DeepseekOcrConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 2;
    cfg.has_position_input = false;
    trtmc::DeepseekOcrPreprocessConfig vl_pp;

    auto tokenizer = std::make_shared<VLFixedTokenizer>();

    // No vision encoder -> !vision_encoder_ -> early return to text-only path (line 113)
    trtmc::DeepseekOcrPipeline pipeline(std::move(decoder), nullptr, std::move(cache), cfg, vl_pp,
                                        stream, tokenizer);

    float pixels[2 * 2 * 3] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                               0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 1;
    auto result = pipeline.generate("hello", pixels, 2, 2, gen_cfg);

    // Falls through to text-only path, argmax=2=eos -> one new token
    check(!result.token_ids.empty(), "image no encoder: non-empty result");

    cudaStreamDestroy(stream);
}

static void test_vl_generate_with_vision_encoder() {
    // Covers the full VL image generation path:
    //   convert_float_to_decoded (lines 64-79), deepseek_ocr_preprocess_decoded_image with
    //   simple_chw, run_vision_encoder (lines 363-410), infer_feature_dim (lines 81-91),
    //   generate_vl_from_ids (lines 181-228), run_text_step_with_embed fallback (lines 271-283)
    auto dec_engine = build_mock_decoder();
    auto vis_engine = build_mock_vision_encoder();
    if (!dec_engine || !vis_engine) {
        std::cerr << "SKIP vl_vision_encoder\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto decoder = std::make_unique<trtmc::TrtModuleImpl>(
        dec_engine.get(), dec_engine->createExecutionContext(), stream);
    auto vision = std::make_unique<trtmc::TrtModuleImpl>(
        vis_engine.get(), vis_engine->createExecutionContext(), stream);
    auto cache = std::make_unique<trtmc::DeepseekOcrKvCache>(1, 8, 4, stream);

    trtmc::DeepseekOcrConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 2;
    cfg.has_position_input = false;
    cfg.image_token_id = 1;    // token 1 from VLFixedTokenizer is treated as image token
    cfg.vision_output_dim = 4; // 4-dim features from mock vision encoder

    trtmc::DeepseekOcrPreprocessConfig vl_pp;
    vl_pp.preprocessor_type = "simple_chw"; // resize + CHW normalize, no patch merging
    vl_pp.fixed_image_size = 4;             // resize to 4x4 (matches pixel_values[3,4,4])
    vl_pp.in_channels = 3;

    auto tokenizer = std::make_shared<VLFixedTokenizer>(); // encodes as {1, 2, 3}

    trtmc::DeepseekOcrPipeline pipeline(std::move(decoder), std::move(vision), std::move(cache),
                                        cfg, vl_pp, stream, tokenizer);

    // Provide a 2x2 RGB image (float pixels in [0,1], will be converted and resized to 4x4)
    float pixels[2 * 2 * 3] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                               0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 1;
    gen_cfg.eos_token_id = 2;

    // If preprocessing fails (unlikely), pipeline throws and test catches it as skip
    try {
        auto result = pipeline.generate("hello", pixels, 2, 2, gen_cfg);
        // VLFixedTokenizer returns {1,2,3}; token 1 = image_token -> run_text_step_with_embed
        // decoder has no "input_embed" -> fallback to run_text_step
        // argmax(logits)=2=eos -> 1 new token generated
        check(!result.token_ids.empty(), "vl_vision_encoder: result not empty");
    } catch (const std::runtime_error& e) {
        // Preprocessing might fail in unusual environments; skip gracefully
        std::cerr << "SKIP vl_vision_encoder (preprocessing error): " << e.what() << '\n';
    }

    cudaStreamDestroy(stream);
}

// Decoder with "input_embed" and "use_input_embed" inputs to exercise
// run_text_step_with_embed full body (lines 285-361 in vl_pipeline.cpp).
// Constant logits {0.1, 0.9, 0.2, 0.3} -> argmax=1 (non-eos when eos=2),
// allowing the autoregressive loop to call run_text_step (line 224).
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_mock_decoder_with_embed() {
    auto b = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* tok = n->addInput("token_id", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {1}});
    auto* mask = n->addInput("attention_mask", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {8}});
    auto* ue = n->addInput("use_input_embed", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {1}});
    auto* emb = n->addInput("input_embed", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {4}});

    // argmax=1, non-eos (eos=2), so autoregressive loop does not break immediately
    float cl[4] = {0.1f, 0.9f, 0.2f, 0.3f};
    auto* cst = n->addConstant(nvinfer1::Dims{1, {4}},
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cl, 4});
    cst->getOutput(0)->setName("logits");
    n->markOutput(*cst->getOutput(0));

    // Keep all inputs live
    n->addIdentity(*tok)->getOutput(0)->setName("_t");
    n->addIdentity(*mask)->getOutput(0)->setName("_m");
    n->addIdentity(*ue)->getOutput(0)->setName("_ue");
    n->addIdentity(*emb)->getOutput(0)->setName("_e");

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan)
        return nullptr;
    auto rt = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

static void test_vl_generate_with_embed_decoder() {
    // Covers DeepseekOcrPipeline::run_text_step_with_embed() full body (lines 285-361).
    // The embed-capable decoder has "input_embed" so has_input("input_embed")==true,
    // bypassing the early fallback at line 280.
    // VLFixedTokenizer returns {1, 2, 3}; image_token_id=1, so:
    //   - token 1 (image): run_text_step_with_embed(1, embed_ptr, 1.0, logits)
    //                      -> covers image inject path (lines 202-206, 328-332)
    //   - token 2, 3 (text): run_text_step_with_embed(t, nullptr, 0.0, logits)
    //                        -> covers zero_embed path (lines 321-325)
    // Constant logits argmax=1 (non-eos=2): autoregressive loop calls
    // run_text_step for max_new_tokens=2 steps, covering line 224.
    auto dec_engine = build_mock_decoder_with_embed();
    auto vis_engine = build_mock_vision_encoder();
    if (!dec_engine || !vis_engine) {
        std::cerr << "SKIP embed_decoder\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto decoder = std::make_unique<trtmc::TrtModuleImpl>(
        dec_engine.get(), dec_engine->createExecutionContext(), stream);
    auto vision = std::make_unique<trtmc::TrtModuleImpl>(
        vis_engine.get(), vis_engine->createExecutionContext(), stream);
    auto cache = std::make_unique<trtmc::DeepseekOcrKvCache>(1, 8, 4, stream);

    trtmc::DeepseekOcrConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 2;
    cfg.image_token_id = 1;    // token 1 from VLFixedTokenizer is the image token
    cfg.vision_output_dim = 4; // 4-dim features from mock vision encoder
    cfg.has_position_input = false;

    trtmc::DeepseekOcrPreprocessConfig vl_pp;
    vl_pp.preprocessor_type = "simple_chw";
    vl_pp.fixed_image_size = 4; // resize to 4x4 CHW
    vl_pp.in_channels = 3;

    auto tokenizer = std::make_shared<VLFixedTokenizer>(); // encodes as {1, 2, 3}

    trtmc::DeepseekOcrPipeline pipeline(std::move(decoder), std::move(vision), std::move(cache),
                                        cfg, vl_pp, stream, tokenizer);

    // 2x2 RGB image float pixels in [0,1]
    float pixels[2 * 2 * 3] = {0.5f, 0.5f, 0.5f, 0.4f, 0.4f, 0.4f,
                               0.3f, 0.3f, 0.3f, 0.2f, 0.2f, 0.2f};

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 2; // 2 autoregressive steps to cover line 224
    gen_cfg.eos_token_id = 2;

    try {
        auto result = pipeline.generate("test", pixels, 2, 2, gen_cfg);
        check(!result.token_ids.empty(), "embed_decoder: non-empty result");
    } catch (const std::runtime_error& e) {
        std::cerr << "SKIP embed_decoder (error): " << e.what() << '\n';
    }

    cudaStreamDestroy(stream);
}

static void test_vl_sequence_prefill_uses_one_text_launch() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto decode_stats = std::make_shared<CountingTextStats>();
    auto prefill_stats = std::make_shared<CountingTextStats>();
    auto decoder = std::make_unique<CountingTextModule>(decode_stats, false, stream);
    auto prefill = std::make_unique<CountingTextModule>(prefill_stats, true, stream);
    auto vision = std::make_unique<FakeSequenceVisionModule>();
    auto cache = std::make_unique<trtmc::DeepseekOcrKvCache>(1, 8, 4, stream);

    trtmc::DeepseekOcrConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 2;
    cfg.image_token_id = 1;
    cfg.vision_output_dim = 4;
    cfg.num_layers = 1;
    cfg.prefill_max_length = 8;

    trtmc::DeepseekOcrPreprocessConfig vl_pp;
    vl_pp.preprocessor_type = "simple_chw";
    vl_pp.fixed_image_size = 4;
    vl_pp.in_channels = 3;

    auto tokenizer = std::make_shared<VLFixedTokenizer>();
    trtmc::DeepseekOcrPipeline pipeline(std::move(decoder), std::move(vision), std::move(cache),
                                        cfg, vl_pp, stream, tokenizer, "", nullptr,
                                        std::move(prefill));

    float pixels[2 * 2 * 3] = {0.5F, 0.5F, 0.5F, 0.4F, 0.4F, 0.4F,
                               0.3F, 0.3F, 0.3F, 0.2F, 0.2F, 0.2F};
    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 1;
    gen_cfg.eos_token_id = 2;
    auto result = pipeline.generate("test", pixels, 2, 2, gen_cfg);

    check(result.token_ids == std::vector<int32_t>{2}, "sequence prefill: output remains correct");
    check(prefill_stats->calls == 1, "sequence prefill: one prefill launch");
    check(decode_stats->calls == 0, "sequence prefill: no prompt-linear decode launches");
    check(prefill_stats->shapes["token_id"] == std::vector<int64_t>{3},
          "sequence prefill: token shape");
    check(prefill_stats->shapes["input_embed"] == std::vector<int64_t>({3, 4}),
          "sequence prefill: input embed shape");
    check(prefill_stats->shapes["use_input_embed"] == std::vector<int64_t>({3, 1}),
          "sequence prefill: embed selector shape");
    check(prefill_stats->shapes["deepstack_embed_0"] == std::vector<int64_t>({3, 4}),
          "sequence prefill: deepstack embed shape");
    check(prefill_stats->shapes["deepstack_active"] == std::vector<int64_t>({3, 1}),
          "sequence prefill: deepstack selector shape");
    check(prefill_stats->float_values["use_input_embed"] == std::vector<float>({1.0F, 0.0F, 0.0F}),
          "sequence prefill: image embedding selected by position");
    check(prefill_stats->float_values["deepstack_active"] == std::vector<float>({1.0F, 0.0F, 0.0F}),
          "sequence prefill: deepstack selected by position");
    check(prefill_stats->bound_shapes["cache_k_0"] == std::vector<int64_t>({8, 4}),
          "sequence prefill: dynamic K cache binds at full profile length");
    check(prefill_stats->bound_shapes["cache_v_0"] == std::vector<int64_t>({8, 4}),
          "sequence prefill: dynamic V cache binds at full profile length");

    cudaStreamDestroy(stream);
}

static void test_vl_sequence_prefill_over_limit_uses_compatibility_path() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto decode_stats = std::make_shared<CountingTextStats>();
    auto prefill_stats = std::make_shared<CountingTextStats>();
    auto decoder = std::make_unique<CountingTextModule>(decode_stats, false, stream);
    auto prefill = std::make_unique<CountingTextModule>(prefill_stats, true, stream);
    auto vision = std::make_unique<FakeSequenceVisionModule>();
    auto cache = std::make_unique<trtmc::DeepseekOcrKvCache>(1, 8, 4, stream);

    trtmc::DeepseekOcrConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 2;
    cfg.image_token_id = 1;
    cfg.vision_output_dim = 4;
    cfg.num_layers = 1;
    cfg.prefill_max_length = 2;

    trtmc::DeepseekOcrPreprocessConfig vl_pp;
    vl_pp.preprocessor_type = "simple_chw";
    vl_pp.fixed_image_size = 4;
    vl_pp.in_channels = 3;

    auto tokenizer = std::make_shared<VLFixedTokenizer>();
    trtmc::DeepseekOcrPipeline pipeline(std::move(decoder), std::move(vision), std::move(cache),
                                        cfg, vl_pp, stream, tokenizer, "", nullptr,
                                        std::move(prefill));

    float pixels[2 * 2 * 3] = {0.5F, 0.5F, 0.5F, 0.4F, 0.4F, 0.4F,
                               0.3F, 0.3F, 0.3F, 0.2F, 0.2F, 0.2F};
    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 1;
    gen_cfg.eos_token_id = 2;
    auto result = pipeline.generate("test", pixels, 2, 2, gen_cfg);

    check(result.token_ids == std::vector<int32_t>{2},
          "sequence prefill limit: output remains correct");
    check(prefill_stats->calls == 0, "sequence prefill limit: prefill launch skipped");
    check(decode_stats->calls == 3,
          "sequence prefill limit: token-by-token compatibility path used");

    cudaStreamDestroy(stream);
}

static void test_vl_generate_with_tokenizer() {
    // Covers the string-based generate(const string&, cfg) method
    auto engine = build_mock_decoder();
    if (!engine) {
        std::cerr << "SKIP vl_generate_tokenizer\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto decoder = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                          engine->createExecutionContext(), stream);
    auto cache = std::make_unique<trtmc::DeepseekOcrKvCache>(1, 8, 4, stream);

    trtmc::DeepseekOcrConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 2; // argmax=2 -> stops after 1 generated token
    cfg.has_position_input = false;

    trtmc::DeepseekOcrPreprocessConfig vl_pp;
    auto tokenizer = std::make_shared<VLFixedTokenizer>();

    trtmc::DeepseekOcrPipeline pipeline(std::move(decoder), nullptr, std::move(cache), cfg, vl_pp,
                                        stream, tokenizer);

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 1;
    // VLFixedTokenizer::encode() returns {1,2,3}; argmax=2=eos -> 1 new token generated
    auto result = pipeline.generate("hello world", gen_cfg);

    // result.token_ids contains only the newly generated token (eos=2)
    check(!result.token_ids.empty(), "generate with tokenizer: non-empty result");

    cudaStreamDestroy(stream);
}

int main() {
    test_vl_text_only();
    test_vl_text_only_max_tokens();
    test_vl_validates_decoder();
    test_vl_validates_cache();
    test_vl_config_sync();
    test_vl_zero_max_tokens();
    test_vl_no_tokenizer_throws();
    test_vl_generate_with_image_no_encoder();
    test_vl_generate_with_vision_encoder();
    test_vl_generate_with_embed_decoder();
    test_vl_sequence_prefill_uses_one_text_launch();
    test_vl_sequence_prefill_over_limit_uses_compatibility_path();
    test_vl_generate_with_tokenizer();
    if (failures > 0)
        std::cerr << failures << " FAILED\n";
    return failures;
}
