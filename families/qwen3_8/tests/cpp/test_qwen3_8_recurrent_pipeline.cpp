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

#include "families/qwen3_8/runtime/hybrid_state.h"
#include "families/qwen3_8/runtime/kv_cache.h"
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
#include <cstdint>
#include <cuda_runtime_api.h>
#include <deque>
#include <fstream>
#include <iostream>
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

// Mock decoder: token_id[1] → logits[4] = constant [0.1, 0.2, 0.9, 0.3]
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_mock_decoder() {
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
    cst->getOutput(0)->setName("logits");
    network->markOutput(*cst->getOutput(0));

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
    if (bundle.info().family != "qwen3_8" || !bundle.find_section("edge_llm.json"))
        throw std::invalid_argument("Expected a qwen3_8 Edge bundle");
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
                {"temperature", 0.0},
                {"top_k", std::int64_t{1}},
                {"top_p", 1.0},
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
