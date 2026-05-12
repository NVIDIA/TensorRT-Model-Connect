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
// Tests the RecurrentPipeline with mock engines and both RecurrentStateManager
// (for pure SSM) and HybridStateManager (for attention+SSM).
// =============================================================================

#include "runtime/models/recurrent/pipeline.h"
#include "trtmc/runtime/hybrid_state.h"
#include "trtmc/runtime/kv_cache.h"
#include "trtmc/runtime/recurrent_state.h"
#include "trtmc/runtime/trt_module.h"
// pipeline_interface.h was removed; GenerateConfig is in trtmc/pipeline.h
// (already included transitively via recurrent_pipeline.h)

#include "runtime/backend/trt_module_impl.h"
#include "runtime/core/trt_common.h"

#include <NvInfer.h>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
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
        std::cerr << "SKIP: can't build engine\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);

    // Mamba: 2 state specs, 1 layer
    std::vector<trtmc::RecurrentState::TensorSpec> specs = {
        {"conv_state", {12}},
        {"ssm_state", {32}},
    };
    auto rs = std::make_unique<trtmc::RecurrentState>(1, specs, stream);

    trtmc::RecurrentGenConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 2; // argmax=2=eos

    trtmc::RecurrentPipeline pipeline(std::move(module), std::move(rs), cfg, stream,
                                      "MambaPipeline");
    check(std::string(pipeline.pipeline_type()) == "MambaPipeline", "mamba name");

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 5;
    auto result = pipeline.generate_ids({1}, gen_cfg);

    // argmax=2=eos → stops after 1 generated token
    check(result.token_ids.size() == 2, "mamba: input + 1 generated");
    check(result.token_ids[1] == 2, "mamba: generated token = 2 (eos)");

    cudaStreamDestroy(stream);
}

static void test_rwkv_pipeline() {
    auto engine = build_mock_decoder();
    if (!engine) {
        std::cerr << "SKIP: can't build engine\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);

    // RWKV: 5 state specs, 2 layers
    std::vector<trtmc::RecurrentState::TensorSpec> specs = {
        {"attn_state", {8}}, {"ff_state", {8}},  {"num_state", {8}},
        {"den_state", {8}},  {"max_state", {8}},
    };
    auto rs = std::make_unique<trtmc::RecurrentState>(2, specs, stream);

    trtmc::RecurrentGenConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 99; // never hit

    trtmc::RecurrentPipeline pipeline(std::move(module), std::move(rs), cfg, stream,
                                      "RwkvPipeline");
    check(std::string(pipeline.pipeline_type()) == "RwkvPipeline", "rwkv name");

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 3;
    auto result = pipeline.generate_ids({0}, gen_cfg);

    check(result.token_ids.size() == 4, "rwkv: input + 3 generated");
    check(result.token_ids[1] == 2, "rwkv: all gen tokens = 2");
    check(result.token_ids[3] == 2, "rwkv: last gen = 2");

    cudaStreamDestroy(stream);
}

static void test_hybrid_pipeline() {
    auto engine = build_mock_decoder();
    if (!engine) {
        std::cerr << "SKIP: can't build engine\n";
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
    auto* mask =
        network->addInput("attention_mask", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {4}});

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
        cudaStreamDestroy(stream);
        return;
    }
    auto rt = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    auto hybrid_engine = trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
    if (!hybrid_engine) {
        cudaStreamDestroy(stream);
        return;
    }

    auto module = std::make_unique<trtmc::TrtModuleImpl>(
        hybrid_engine.get(), hybrid_engine->createExecutionContext(), stream);
    auto kv = std::make_unique<trtmc::KvCache>(1, 4, 2, stream);
    std::vector<trtmc::RecurrentState::TensorSpec> specs = {{"ssm", {4}}};
    auto ssm = std::make_unique<trtmc::RecurrentState>(1, specs, stream);
    auto hybrid = std::make_unique<trtmc::HybridState>(std::move(kv), std::move(ssm));

    trtmc::RecurrentGenConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_eos = 2;
    cfg.has_position_input = true;

    trtmc::RecurrentPipeline pipeline(std::move(module), std::move(hybrid), cfg, stream,
                                      "HybridPipeline");
    check(std::string(pipeline.pipeline_type()) == "HybridPipeline", "hybrid name");

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 5;
    auto result = pipeline.generate_ids({0}, gen_cfg);

    check(result.token_ids.size() == 2, "hybrid: input + eos");
    check(result.token_ids[1] == 2, "hybrid: eos generated");

    cudaStreamDestroy(stream);
}

static void test_generate_applies_chat_template() {
    auto engine = build_mock_decoder();
    if (!engine) {
        std::cerr << "SKIP: can't build engine\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);
    std::vector<trtmc::RecurrentState::TensorSpec> specs = {
        {"conv_state", {12}},
        {"ssm_state", {32}},
    };
    auto rs = std::make_unique<trtmc::RecurrentState>(1, specs, stream);

    trtmc::RecurrentGenConfig cfg;
    cfg.vocab_size = 4;
    cfg.id_bos = 1;
    cfg.id_eos = 2;
    cfg.chat_template_format = trtmc::ChatTemplateFormat::kNemotronH;

    auto tokenizer = std::make_shared<RecordingTokenizer>();
    trtmc::RecurrentPipeline pipeline(std::move(module), std::move(rs), cfg, stream,
                                      "MambaPipeline", tokenizer);

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 1;
    gen_cfg.use_chat_template = true;
    gen_cfg.enable_thinking = false;
    auto result = pipeline.generate("What is the capital of France?", gen_cfg);

    check(result.text == "Paris", "chat template: generated text decodes");
    check(tokenizer->last_text.find("<SPECIAL_10>System\n\n<SPECIAL_11>User\n") !=
              std::string::npos,
          "chat template: nemotron-h prefix");
    check(tokenizer->last_text.find("What is the capital of France?") != std::string::npos,
          "chat template: prompt retained");
    check(tokenizer->last_text.find("<think></think>") != std::string::npos,
          "chat template: no-thinking block is closed inline");
    check(tokenizer->last_text.find("<think>\n") == std::string::npos,
          "chat template: no thinking newline sentinel");

    cudaStreamDestroy(stream);
}

static void test_argmax_recurrent() {
    std::vector<float> v = {-1.0f, 5.0f, 3.0f};
    check(trtmc::RecurrentPipeline::argmax(v) == 1, "argmax = 1");
}

int main() {
    test_argmax_recurrent();
    test_mamba_pipeline();
    test_rwkv_pipeline();
    test_hybrid_pipeline();
    test_generate_applies_chat_template();
    if (failures > 0)
        std::cerr << failures << " FAILED\n";
    return failures;
}
