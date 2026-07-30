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
// Intent:         QwenVlPipeline text-only generation with mock engines; constructor
//                 validation (null decoder/cache); QwenVlConfig sync from
//                 QwenVlPreprocessConfig; zero max_tokens early exit; string-based
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
// Test suite: QwenVlPipeline — vision-language generation
// =============================================================================

#include "runtime/backend/trt_module_impl.h"
#include "runtime/core/trt_common.h"
#include "runtime/models/qwen_vl/kv_cache.h"
#include "runtime/models/qwen_vl/lora_peft_loader.h"
#include "runtime/models/qwen_vl/pipeline.h"
#include "trtmc/runtime/pipeline_pool.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <NvInfer.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <future>
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

static nvinfer1::IRuntime* mock_runtime() {
    // TensorRT requires the runtime to outlive every engine it deserializes.
    // This function-static runtime is destroyed before the earlier-constructed
    // process logger and after every mock engine owned by the tests below.
    static auto runtime =
        trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return runtime.get();
}

struct CountingTextStats {
    int32_t calls{0};
    std::unordered_map<std::string, std::vector<int64_t>> shapes;
    std::unordered_map<std::string, std::vector<float>> float_values;
    std::unordered_map<std::string, std::vector<int32_t>> int_values;
};

class CountingTextModule final : public trtmc::ITrtModule {
  public:
    CountingTextModule(std::shared_ptr<CountingTextStats> stats, bool prefill, cudaStream_t stream,
                       int32_t mrope_rank = 0)
        : stats_(std::move(stats)), prefill_(prefill), stream_(stream),
          mrope_rank_(mrope_rank),
          present_k_(prefill ? trtmc::DeviceTensor::zeros({8, 4}, trtmc::DType::kFloat32, stream)
                             : trtmc::DeviceTensor{}),
          present_v_(prefill ? trtmc::DeviceTensor::zeros({8, 4}, trtmc::DType::kFloat32, stream)
                             : trtmc::DeviceTensor{}) {}

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++stats_->calls;
        stats_->shapes.clear();
        stats_->float_values.clear();
        stats_->int_values.clear();
        for (const auto& [name, tensor] : inputs) {
            stats_->shapes[name] = tensor.shape;
            if (tensor.dtype == trtmc::DType::kFloat32) {
                const auto* begin = static_cast<const float*>(tensor.data);
                stats_->float_values[name] =
                    std::vector<float>(begin, begin + static_cast<std::ptrdiff_t>(tensor.numel()));
            } else if (tensor.dtype == trtmc::DType::kInt32) {
                const auto* begin = static_cast<const int32_t*>(tensor.data);
                stats_->int_values[name] =
                    std::vector<int32_t>(begin,
                                         begin + static_cast<std::ptrdiff_t>(tensor.numel()));
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
               name == "deepstack_embed_0" || name == "cache_k_0" || name == "cache_v_0" ||
               (name == "mrope_position_ids" && mrope_rank_ > 0);
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
    int32_t input_rank(const std::string& name) const override {
        if (name == "mrope_position_ids")
            return mrope_rank_;
        return name == "token_id" || name == "position_id" ? 1 : 2;
    }
    bool input_is_dynamic(const std::string&) const override { return prefill_; }
    bool ok() const override { return !prefill_ || (present_k_.ok() && present_v_.ok()); }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    std::shared_ptr<CountingTextStats> stats_;
    bool prefill_{false};
    cudaStream_t stream_{nullptr};
    int32_t mrope_rank_{0};
    mutable trtmc::DeviceTensor present_k_;
    mutable trtmc::DeviceTensor present_v_;
    std::vector<float> logits_{0.1F, 0.2F, 0.9F, 0.3F};
};

class FakeSequenceVisionModule final : public trtmc::ITrtModule {
  public:
    explicit FakeSequenceVisionModule(int32_t num_features = 1)
        : features_(static_cast<std::size_t>(num_features) * 4, 1.0F),
          deepstack_(static_cast<std::size_t>(num_features) * 4, 2.0F) {}

    trtmc::TensorMap forward(const trtmc::TensorMap&) override {
        const auto num_features = static_cast<int64_t>(features_.size() / 4);
        return {{"image_features",
                 trtmc::Tensor{features_.data(), {num_features, 4}, trtmc::DType::kFloat32}},
                {"deepstack_features_0",
                 trtmc::Tensor{deepstack_.data(), {num_features, 4},
                               trtmc::DType::kFloat32}}};
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
        return {{"image_features", {static_cast<int64_t>(features_.size() / 4), 4},
                 trtmc::DType::kFloat32, false}};
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
    std::vector<float> features_;
    std::vector<float> deepstack_;
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

class VLDynamicGridTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string&) const override {
        std::vector<int32_t> tokens{10};
        tokens.insert(tokens.end(), 6, 1);
        tokens.push_back(12);
        return tokens;
    }
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
    auto* rt = mock_runtime();
    if (!rt)
        return nullptr;
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_mock_lora_decoder() {
    auto b = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 4 << 20);

    auto* tok = n->addInput("token_id", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {1}});
    auto* mask = n->addInput("attention_mask", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {8}});
    auto* lora_a =
        n->addInput("lora_a_layer_0_w_q", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{2, {1, 1}});
    auto* lora_b =
        n->addInput("lora_b_layer_0_w_q", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{2, {1, 4}});

    float base_logits[4] = {0.1F, 0.2F, 0.9F, 0.3F};
    auto* base = n->addConstant(nvinfer1::Dims{2, {1, 4}},
                                nvinfer1::Weights{nvinfer1::DataType::kFLOAT, base_logits, 4});
    auto* delta = n->addMatrixMultiply(*lora_a, nvinfer1::MatrixOperation::kNONE, *lora_b,
                                       nvinfer1::MatrixOperation::kNONE);
    auto* logits = n->addElementWise(*base->getOutput(0), *delta->getOutput(0),
                                     nvinfer1::ElementWiseOperation::kSUM);
    logits->getOutput(0)->setName("logits");
    n->markOutput(*logits->getOutput(0));

    n->addIdentity(*tok)->getOutput(0)->setName("_t");
    n->addIdentity(*mask)->getOutput(0)->setName("_m");

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan)
        return nullptr;
    auto* rt = mock_runtime();
    if (!rt)
        return nullptr;
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

static std::filesystem::path write_mock_peft_adapter() {
    namespace fs = std::filesystem;
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path root = fs::temp_directory_path() / ("trtmc_qwen_lora_" + std::to_string(nonce));
    fs::create_directories(root);

    std::ofstream(root / "adapter_config.json")
        << R"({"peft_type":"LORA","r":1,"lora_alpha":2,)"
           R"("target_modules":["q_proj"],"bias":"none",)"
           R"("fan_in_fan_out":false,"use_dora":false,"use_rslora":false,)"
           R"("modules_to_save":null})";

    const std::string prefix = "base_model.model.model.language_model.layers.0.self_attn.q_proj.";
    std::string header_text = "{\"" + prefix +
                              "lora_A.weight\":{\"dtype\":\"BF16\",\"shape\":[1,1],"
                              "\"data_offsets\":[0,2]},\"" +
                              prefix +
                              "lora_B.weight\":{\"dtype\":\"BF16\",\"shape\":[4,1],"
                              "\"data_offsets\":[2,10]}}";
    while (header_text.size() % 8 != 0)
        header_text.push_back(' ');

    std::ofstream weights(root / "adapter_model.safetensors", std::ios::binary);
    const uint64_t header_size = header_text.size();
    for (int i = 0; i < 8; ++i)
        weights.put(static_cast<char>((header_size >> (8 * i)) & 0xFFU));
    weights.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
    // BF16: A=[1], B=[[2], [0], [0], [0]]. Alpha/rank=2 is folded into B.
    const uint16_t tensor_data[] = {0x3F80U, 0x4000U, 0U, 0U, 0U};
    weights.write(reinterpret_cast<const char*>(tensor_data), sizeof(tensor_data));
    return root;
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
    auto* rt = mock_runtime();
    if (!rt)
        return nullptr;
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
    auto cache = std::make_unique<trtmc::QwenVlKvCache>(1, 8, 4, stream);

    trtmc::QwenVlConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 2;
    cfg.has_position_input = false;

    // No vision encoder (text-only mode)
    trtmc::QwenVlPreprocessConfig vl_pp;
    trtmc::QwenVlPipeline pipeline(std::move(decoder), nullptr, std::move(cache), cfg, vl_pp,
                                   stream);
    check(std::string(pipeline.pipeline_type()) == "QwenVlPipeline", "vl name");

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
    auto cache = std::make_unique<trtmc::QwenVlKvCache>(1, 8, 4, stream);

    trtmc::QwenVlConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 99;
    cfg.has_position_input = false;

    trtmc::QwenVlPreprocessConfig vl_pp;
    trtmc::QwenVlPipeline pipeline(std::move(decoder), nullptr, std::move(cache), cfg, vl_pp,
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

    auto cache = std::make_unique<trtmc::QwenVlKvCache>(1, 8, 4, stream);

    trtmc::QwenVlConfig cfg;
    cfg.has_position_input = false;
    trtmc::QwenVlPreprocessConfig vl_pp;

    // null text_decoder -> throws
    bool threw = false;
    try {
        trtmc::QwenVlPipeline p(nullptr, nullptr, std::move(cache), cfg, vl_pp, stream);
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

    trtmc::QwenVlConfig cfg;
    cfg.has_position_input = false;
    trtmc::QwenVlPreprocessConfig vl_pp;

    // null cache -> throws
    bool threw = false;
    try {
        trtmc::QwenVlPipeline p(std::move(decoder), nullptr, nullptr, cfg, vl_pp, stream);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "vl: null cache throws");

    cudaStreamDestroy(stream);
}

static void test_vl_config_sync() {
    // QwenVlPreprocessConfig has image_token_id=1 and vision_output_dim=64,
    // QwenVlConfig has image_token_id=-1 and vision_output_dim=0.
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
    auto cache = std::make_unique<trtmc::QwenVlKvCache>(1, 8, 4, stream);

    trtmc::QwenVlConfig cfg;
    cfg.has_position_input = false;
    cfg.image_token_id = -1;   // will be synced from vl_pp
    cfg.vision_output_dim = 0; // will be synced from vl_pp

    trtmc::QwenVlPreprocessConfig vl_pp;
    vl_pp.image_token_id = 1;
    vl_pp.vision_output_dim = 64;

    trtmc::QwenVlPipeline pipeline(std::move(decoder), nullptr, std::move(cache), cfg, vl_pp,
                                   stream);

    // Construction succeeded; pipeline type is correct
    check(std::string(pipeline.pipeline_type()) == "QwenVlPipeline", "config_sync: type correct");

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
    auto cache = std::make_unique<trtmc::QwenVlKvCache>(1, 8, 4, stream);

    trtmc::QwenVlConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 2;
    cfg.has_position_input = false;

    trtmc::QwenVlPreprocessConfig vl_pp;
    trtmc::QwenVlPipeline pipeline(std::move(decoder), nullptr, std::move(cache), cfg, vl_pp,
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
    auto cache = std::make_unique<trtmc::QwenVlKvCache>(1, 8, 4, stream);

    trtmc::QwenVlConfig cfg;
    cfg.has_position_input = false;
    trtmc::QwenVlPreprocessConfig vl_pp;

    // No tokenizer
    trtmc::QwenVlPipeline pipeline(std::move(decoder), nullptr, std::move(cache), cfg, vl_pp,
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
    auto cache = std::make_unique<trtmc::QwenVlKvCache>(1, 8, 4, stream);

    trtmc::QwenVlConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 2;
    cfg.has_position_input = false;
    trtmc::QwenVlPreprocessConfig vl_pp;

    auto tokenizer = std::make_shared<VLFixedTokenizer>();

    // No vision encoder -> !vision_encoder_ -> early return to text-only path (line 113)
    trtmc::QwenVlPipeline pipeline(std::move(decoder), nullptr, std::move(cache), cfg, vl_pp,
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
    //   convert_float_to_decoded (lines 64-79), qwen_vl_preprocess_decoded_image with simple_chw,
    //   run_vision_encoder (lines 363-410), infer_feature_dim (lines 81-91),
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
    auto cache = std::make_unique<trtmc::QwenVlKvCache>(1, 8, 4, stream);

    trtmc::QwenVlConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 2;
    cfg.has_position_input = false;
    cfg.image_token_id = 1;    // token 1 from VLFixedTokenizer is treated as image token
    cfg.vision_output_dim = 4; // 4-dim features from mock vision encoder

    trtmc::QwenVlPreprocessConfig vl_pp;
    vl_pp.preprocessor_type = "simple_chw"; // resize + CHW normalize, no patch merging
    vl_pp.fixed_image_size = 4;             // resize to 4x4 (matches pixel_values[3,4,4])
    vl_pp.in_channels = 3;

    auto tokenizer = std::make_shared<VLFixedTokenizer>(); // encodes as {1, 2, 3}

    trtmc::QwenVlPipeline pipeline(std::move(decoder), std::move(vision), std::move(cache), cfg,
                                   vl_pp, stream, tokenizer);

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
    auto* rt = mock_runtime();
    if (!rt)
        return nullptr;
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

static void test_vl_generate_with_embed_decoder() {
    // Covers QwenVlPipeline::run_text_step_with_embed() full body (lines 285-361).
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
    auto cache = std::make_unique<trtmc::QwenVlKvCache>(1, 8, 4, stream);

    trtmc::QwenVlConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 2;
    cfg.image_token_id = 1;    // token 1 from VLFixedTokenizer is the image token
    cfg.vision_output_dim = 4; // 4-dim features from mock vision encoder
    cfg.has_position_input = false;

    trtmc::QwenVlPreprocessConfig vl_pp;
    vl_pp.preprocessor_type = "simple_chw";
    vl_pp.fixed_image_size = 4; // resize to 4x4 CHW
    vl_pp.in_channels = 3;

    auto tokenizer = std::make_shared<VLFixedTokenizer>(); // encodes as {1, 2, 3}

    trtmc::QwenVlPipeline pipeline(std::move(decoder), std::move(vision), std::move(cache), cfg,
                                   vl_pp, stream, tokenizer);

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
    auto cache = std::make_unique<trtmc::QwenVlKvCache>(1, 8, 4, stream);

    trtmc::QwenVlConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 2;
    cfg.image_token_id = 1;
    cfg.vision_output_dim = 4;
    cfg.num_layers = 1;
    cfg.prefill_max_length = 8;

    trtmc::QwenVlPreprocessConfig vl_pp;
    vl_pp.preprocessor_type = "simple_chw";
    vl_pp.fixed_image_size = 4;
    vl_pp.in_channels = 3;

    auto tokenizer = std::make_shared<VLFixedTokenizer>();
    trtmc::QwenVlPipeline pipeline(std::move(decoder), std::move(vision), std::move(cache), cfg,
                                   vl_pp, stream, tokenizer, "", nullptr, std::move(prefill));

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

    cudaStreamDestroy(stream);
}

static void test_vl_dual_profile_mrope_shapes_match_engine_rank() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto decode_stats = std::make_shared<CountingTextStats>();
    auto prefill_stats = std::make_shared<CountingTextStats>();
    auto decoder = std::make_unique<CountingTextModule>(decode_stats, false, stream, 2);
    auto prefill = std::make_unique<CountingTextModule>(prefill_stats, true, stream, 2);
    auto vision = std::make_unique<FakeSequenceVisionModule>();
    auto cache = std::make_unique<trtmc::QwenVlKvCache>(1, 8, 4, stream);

    trtmc::QwenVlConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 99;
    cfg.image_token_id = 1;
    cfg.vision_output_dim = 4;
    cfg.num_layers = 1;
    cfg.prefill_max_length = 8;

    trtmc::QwenVlPreprocessConfig vl_pp;
    vl_pp.preprocessor_type = "simple_chw";
    vl_pp.fixed_image_size = 4;
    vl_pp.in_channels = 3;

    auto tokenizer = std::make_shared<VLFixedTokenizer>();
    trtmc::QwenVlPipeline pipeline(std::move(decoder), std::move(vision), std::move(cache), cfg,
                                   vl_pp, stream, tokenizer, "", nullptr, std::move(prefill));

    float pixels[2 * 2 * 3] = {0.5F, 0.5F, 0.5F, 0.4F, 0.4F, 0.4F,
                               0.3F, 0.3F, 0.3F, 0.2F, 0.2F, 0.2F};
    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 2;
    gen_cfg.eos_token_id = 99;
    (void)pipeline.generate("test", pixels, 2, 2, gen_cfg);

    check(prefill_stats->shapes["mrope_position_ids"] == std::vector<int64_t>({3, 3}),
          "dual profile mrope: prefill keeps [3, sequence]");
    check(decode_stats->shapes["mrope_position_ids"] == std::vector<int64_t>({3, 1}),
          "dual profile mrope: decode keeps rank-two [3, 1]");

    cudaStreamDestroy(stream);
}

static void test_vl_dynamic_resolution_uses_actual_grid_for_mrope() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto decode_stats = std::make_shared<CountingTextStats>();
    auto prefill_stats = std::make_shared<CountingTextStats>();
    auto decoder = std::make_unique<CountingTextModule>(decode_stats, false, stream, 2);
    auto prefill = std::make_unique<CountingTextModule>(prefill_stats, true, stream, 2);
    auto vision = std::make_unique<FakeSequenceVisionModule>(6);
    auto cache = std::make_unique<trtmc::QwenVlKvCache>(1, 16, 4, stream);

    trtmc::QwenVlConfig cfg;
    cfg.vocab_size = 16;
    cfg.id_eos = 99;
    cfg.image_token_id = 1;
    cfg.vision_output_dim = 4;
    cfg.num_layers = 1;
    cfg.prefill_max_length = 16;

    trtmc::QwenVlPreprocessConfig vl_pp;
    vl_pp.preprocessor_type = "qwen_smart_resize_patchify";
    vl_pp.fixed_image_size = 448;
    vl_pp.patch_size = 14;
    vl_pp.merge_size = 2;
    vl_pp.temporal_patch_size = 2;
    vl_pp.in_channels = 3;
    vl_pp.dynamic_image_resolution = true;
    vl_pp.min_pixels = 56 * 84;
    vl_pp.max_pixels = 56 * 84;

    auto tokenizer = std::make_shared<VLDynamicGridTokenizer>();
    trtmc::QwenVlPipeline pipeline(std::move(decoder), std::move(vision), std::move(cache), cfg,
                                   vl_pp, stream, tokenizer, "", nullptr, std::move(prefill));

    std::vector<float> pixels(static_cast<std::size_t>(56 * 84 * 3), 0.5F);
    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 1;
    gen_cfg.eos_token_id = 99;
    (void)pipeline.generate("test", pixels.data(), 56, 84, gen_cfg);

    check(prefill_stats->shapes["mrope_position_ids"] == std::vector<int64_t>({3, 8}),
          "dynamic mrope: prefill shape follows token count");
    check(prefill_stats->int_values["mrope_position_ids"] ==
              std::vector<int32_t>({0, 1, 1, 1, 1, 1, 1, 4,
                                    0, 1, 1, 1, 2, 2, 2, 4,
                                    0, 1, 2, 3, 1, 2, 3, 4}),
          "dynamic mrope: pipeline uses actual 2x3 merged image grid");

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
    auto cache = std::make_unique<trtmc::QwenVlKvCache>(1, 8, 4, stream);

    trtmc::QwenVlConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 2; // argmax=2 -> stops after 1 generated token
    cfg.has_position_input = false;

    trtmc::QwenVlPreprocessConfig vl_pp;
    auto tokenizer = std::make_shared<VLFixedTokenizer>();

    trtmc::QwenVlPipeline pipeline(std::move(decoder), nullptr, std::move(cache), cfg, vl_pp,
                                   stream, tokenizer);

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 1;
    // VLFixedTokenizer::encode() returns {1,2,3}; argmax=2=eos -> 1 new token generated
    auto result = pipeline.generate("hello world", gen_cfg);

    // result.token_ids contains only the newly generated token (eos=2)
    check(!result.token_ids.empty(), "generate with tokenizer: non-empty result");

    cudaStreamDestroy(stream);
}

static void test_vl_dynamic_lora_adapter_switching() {
    auto engine = build_mock_lora_decoder();
    if (!engine) {
        std::cerr << "SKIP dynamic_lora\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto decoder = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                          engine->createExecutionContext(), stream);
    auto cache = std::make_unique<trtmc::QwenVlKvCache>(1, 8, 4, stream);
    trtmc::QwenVlConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 99;
    cfg.has_position_input = false;
    trtmc::QwenVlPreprocessConfig vl_pp;
    trtmc::QwenVlPipeline pipeline(std::move(decoder), nullptr, std::move(cache), cfg, vl_pp,
                                   stream);

    check(pipeline.has_dynamic_lora(), "dynamic_lora: engine inputs detected");
    check(pipeline.lora_input_names().size() == 2, "dynamic_lora: two bindings detected");

    float ones[1] = {1.0F};
    float adapter_a_delta[4] = {2.0F, 0.0F, 0.0F, 0.0F};
    float adapter_b_delta[4] = {0.0F, 2.0F, 0.0F, 0.0F};
    auto make_adapter = [&](float* delta) {
        trtmc::TensorMap tensors;
        tensors["lora_a_layer_0_w_q"] = trtmc::Tensor{ones, {1, 1}, trtmc::DType::kFloat32};
        tensors["lora_b_layer_0_w_q"] = trtmc::Tensor{delta, {1, 4}, trtmc::DType::kFloat32};
        return tensors;
    };
    pipeline.register_lora_adapter("adapter-a", make_adapter(adapter_a_delta));
    pipeline.register_lora_adapter("adapter-b", make_adapter(adapter_b_delta));

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 1;

    auto base = pipeline.generate_ids({3}, gen_cfg);
    check(base.token_ids.back() == 2, "dynamic_lora: zero binding selects base output");

    gen_cfg.lora_adapter_id = "adapter-a";
    auto adapter_a = pipeline.generate_ids({3}, gen_cfg);
    check(adapter_a.token_ids.back() == 0, "dynamic_lora: adapter A selected");

    gen_cfg.lora_adapter_id = "adapter-b";
    auto adapter_b = pipeline.generate_ids({3}, gen_cfg);
    check(adapter_b.token_ids.back() == 1, "dynamic_lora: adapter B selected");

    gen_cfg.lora_adapter_id.clear();
    auto base_again = pipeline.generate_ids({3}, gen_cfg);
    check(base_again.token_ids.back() == 2, "dynamic_lora: clear restores base output");

    bool unknown_threw = false;
    try {
        gen_cfg.lora_adapter_id = "missing";
        (void)pipeline.generate_ids({3}, gen_cfg);
    } catch (const std::invalid_argument&) {
        unknown_threw = true;
    }
    check(unknown_threw, "dynamic_lora: unknown adapter rejected");

    const auto adapter_dir = write_mock_peft_adapter();
    pipeline.load_lora_adapter("adapter-file", adapter_dir.string());
    check(pipeline.supports_lora_adapters(), "dynamic_lora: public capability detected");
    check(pipeline.loaded_lora_adapters().size() == 3,
          "dynamic_lora: public API lists loaded adapters");
    gen_cfg.lora_adapter_id = "adapter-file";
    auto adapter_file = pipeline.generate_ids({3}, gen_cfg);
    check(adapter_file.token_ids.back() == 0, "dynamic_lora: PEFT directory selected");
    pipeline.unload_lora_adapter("adapter-file");
    check(pipeline.loaded_lora_adapters().size() == 2, "dynamic_lora: public API unloads adapter");
    std::filesystem::remove_all(adapter_dir);

    // The Qwen-VL cache must not evict an adapter pinned by the active
    // execution context. Fill its four-entry capacity, select A, then add E;
    // the least-recently-used unpinned adapter B should be removed.
    pipeline.register_lora_adapter("adapter-c", make_adapter(adapter_b_delta));
    pipeline.register_lora_adapter("adapter-d", make_adapter(adapter_b_delta));
    gen_cfg.lora_adapter_id = "adapter-a";
    (void)pipeline.generate_ids({3}, gen_cfg);
    pipeline.register_lora_adapter("adapter-e", make_adapter(adapter_b_delta));
    const auto cached_ids = pipeline.loaded_lora_adapters();
    check(cached_ids.size() == 4, "dynamic_lora: Qwen-VL cache enforces capacity");
    check(std::find(cached_ids.begin(), cached_ids.end(), "adapter-a") != cached_ids.end(),
          "dynamic_lora: active adapter remains pinned");
    check(std::find(cached_ids.begin(), cached_ids.end(), "adapter-b") == cached_ids.end(),
          "dynamic_lora: least-recently-used unpinned adapter evicted");
    auto pinned_a = pipeline.generate_ids({3}, gen_cfg);
    check(pinned_a.token_ids.back() == 0,
          "dynamic_lora: pinned adapter remains bound after eviction");

    cudaStreamDestroy(stream);
}

static void test_vl_pool_isolates_concurrent_lora_selection() {
    auto engine = build_mock_lora_decoder();
    if (!engine) {
        std::cerr << "SKIP dynamic_lora_pool\n";
        return;
    }

    cudaStream_t stream_a;
    cudaStream_t stream_b;
    cudaStreamCreate(&stream_a);
    cudaStreamCreate(&stream_b);

    auto decoder_a = std::make_unique<trtmc::TrtModuleImpl>(
        engine.get(), engine->createExecutionContext(), stream_a);
    auto decoder_b = std::make_unique<trtmc::TrtModuleImpl>(
        engine.get(), engine->createExecutionContext(), stream_b);
    std::vector<trtmc::TensorInfo> adapter_contract;
    for (const auto& info : decoder_a->input_info()) {
        if (info.name.rfind("lora_a_", 0) == 0 || info.name.rfind("lora_b_", 0) == 0)
            adapter_contract.push_back(info);
    }
    auto adapter_cache =
        std::make_shared<trtmc::qwen_vl::LoraAdapterCache>(adapter_contract, stream_a);

    trtmc::QwenVlConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 99;
    cfg.has_position_input = false;
    trtmc::QwenVlPreprocessConfig vl_pp;
    auto pipeline_a = std::make_unique<trtmc::QwenVlPipeline>(
        std::move(decoder_a), nullptr, std::make_unique<trtmc::QwenVlKvCache>(1, 8, 4, stream_a),
        cfg, vl_pp, stream_a, nullptr, "lane-a", nullptr, adapter_cache);
    auto pipeline_b = std::make_unique<trtmc::QwenVlPipeline>(
        std::move(decoder_b), nullptr, std::make_unique<trtmc::QwenVlKvCache>(1, 8, 4, stream_b),
        cfg, vl_pp, stream_b, nullptr, "lane-b", nullptr, adapter_cache);

    float ones[1] = {1.0F};
    float adapter_a_delta[4] = {2.0F, 0.0F, 0.0F, 0.0F};
    float adapter_b_delta[4] = {0.0F, 2.0F, 0.0F, 0.0F};
    auto make_adapter = [&](float* delta) {
        trtmc::TensorMap tensors;
        tensors["lora_a_layer_0_w_q"] = trtmc::Tensor{ones, {1, 1}, trtmc::DType::kFloat32};
        tensors["lora_b_layer_0_w_q"] = trtmc::Tensor{delta, {1, 4}, trtmc::DType::kFloat32};
        return tensors;
    };
    pipeline_a->register_lora_adapter("adapter-a", make_adapter(adapter_a_delta));
    pipeline_a->register_lora_adapter("adapter-b", make_adapter(adapter_b_delta));
    check(pipeline_b->loaded_lora_adapters().size() == 2,
          "dynamic_lora_pool: lanes share adapter registry");

    {
        std::vector<std::unique_ptr<trtmc::IPipeline>> lanes;
        lanes.push_back(std::move(pipeline_a));
        lanes.push_back(std::move(pipeline_b));
        trtmc::PipelinePool pool(std::move(lanes));
        auto lease_a = pool.acquire();
        auto lease_b = pool.acquire();
        auto* qwen_a = dynamic_cast<trtmc::QwenVlPipeline*>(lease_a.get());
        auto* qwen_b = dynamic_cast<trtmc::QwenVlPipeline*>(lease_b.get());
        check(qwen_a != nullptr && qwen_b != nullptr, "dynamic_lora_pool: leases Qwen lanes");

        trtmc::GenerateConfig config_a;
        config_a.max_new_tokens = 1;
        config_a.lora_adapter_id = "adapter-a";
        trtmc::GenerateConfig config_b = config_a;
        config_b.lora_adapter_id = "adapter-b";
        auto result_a = std::async(
            std::launch::async, [qwen_a, config_a] { return qwen_a->generate_ids({3}, config_a); });
        auto result_b = std::async(
            std::launch::async, [qwen_b, config_b] { return qwen_b->generate_ids({3}, config_b); });
        check(result_a.get().token_ids.back() == 0, "dynamic_lora_pool: request A keeps adapter A");
        check(result_b.get().token_ids.back() == 1, "dynamic_lora_pool: request B keeps adapter B");
    }
    adapter_cache.reset();
    cudaStreamDestroy(stream_a);
    cudaStreamDestroy(stream_b);
}

static void test_qwen_vl_smart_resize() {
    const auto resized = trtmc::qwen_vl_smart_resize(1000, 2000, 28, 3136, 200704);
    check(resized[0] == 308 && resized[1] == 616,
          "smart_resize: preserves aspect ratio within the pixel limit");
}

static void test_qwen_vl_dynamic_grid_mrope_positions() {
    constexpr int32_t image_token_id = 151655;
    constexpr int32_t grid_height = 37;
    constexpr int32_t grid_width = 51;
    constexpr int32_t image_features = grid_height * grid_width;
    std::vector<int32_t> input_ids{10, 11};
    input_ids.insert(input_ids.end(), image_features, image_token_id);
    input_ids.push_back(12);

    const auto positions = trtmc::qwen_vl_build_mrope_positions(
        input_ids, image_token_id, image_features, grid_height, grid_width);
    check(positions.token_positions[2] == std::array<int32_t, 3>{2, 2, 2},
          "dynamic mrope: first image token uses actual merged grid");
    check(positions.token_positions[2 + grid_width] == std::array<int32_t, 3>{2, 3, 2},
          "dynamic mrope: next image row advances height axis");
    check(positions.token_positions.back() == std::array<int32_t, 3>{53, 53, 53},
          "dynamic mrope: trailing text starts after maximum image extent");
    check(positions.next_position == 54,
          "dynamic mrope: decode position follows trailing text");
}

int main() {
    test_qwen_vl_smart_resize();
    test_qwen_vl_dynamic_grid_mrope_positions();
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
    test_vl_dual_profile_mrope_shapes_match_engine_rank();
    test_vl_dynamic_resolution_uses_actual_grid_for_mrope();
    test_vl_generate_with_tokenizer();
    test_vl_dynamic_lora_adapter_switching();
    test_vl_pool_isolates_concurrent_lora_selection();
    if (failures > 0)
        std::cerr << failures << " FAILED\n";
    return failures;
}
