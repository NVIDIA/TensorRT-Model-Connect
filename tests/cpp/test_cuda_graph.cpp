/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-CUDA-CPP-03
// Architecture:   ARCH-MOD-001
// Unit Design:    UD-TRT-CORE-01
// Intent:         CudaGraphExec RAII wrapper + TrtModule CUDA Graph capture/replay
// Preconditions:  CUDA GPU available, TRT engine buildable
// Postconditions: CudaGraphExec captures/replays correctly; TrtModule produces
//                 identical output with and without CUDA Graphs
// =============================================================================

// =============================================================================
// Test suite: CUDA Graph capture and replay
// =============================================================================
//
// Validates:
// 1. CudaGraphExec RAII lifecycle (default state, capture, replay, reset,
//    move semantics, double-reset safety)
// 2. TrtModule::enable_cuda_graph() — first call captures, subsequent replays
// 3. CUDA Graph output matches normal execution
//
// Requires TRT + CUDA GPU. Skips gracefully without TRT.
// =============================================================================

#include "runtime/backend/trt_module_impl.h"
#include "runtime/core/trt_common.h"
#include "trtmc/runtime/tensor.h"
#include "trtmc/runtime/trt_module.h"

#include <NvInfer.h>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc {

class TrtModuleImplTestPeer {
  public:
    static void execute_enqueue(TrtModuleImpl& module) { module.execute_enqueue(); }
    static void set_cuda_graph_launch_override(TrtModuleImpl& module,
                                               bool (*launch)(const CudaGraphExec&, cudaStream_t)) {
        module.cuda_graph_launch_override_for_testing_ = launch;
    }
};

} // namespace trtmc

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static trtmc::TrtLogger g_logger;

static bool fail_cuda_graph_launch(const trtmc::CudaGraphExec&, cudaStream_t) {
    return false;
}

// Build a tiny TRT engine: y = x (identity), fixed shape [4] float32
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_identity_engine() {
    auto builder = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    if (!builder)
        return nullptr;

    auto network = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* inp = network->addInput("x", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {4}});
    if (!inp)
        return nullptr;

    auto* id_layer = network->addIdentity(*inp);
    if (!id_layer)
        return nullptr;
    auto* out = id_layer->getOutput(0);
    out->setName("y");
    network->markOutput(*out);

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan)
        return nullptr;

    auto runtime = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    if (!runtime)
        return nullptr;

    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
}

// Helper: run forward_async + sync, read back output
static void run_and_read(trtmc::ITrtModule& module, const float* input, float* output) {
    trtmc::Tensor input_tensor;
    input_tensor.data = const_cast<float*>(input);
    input_tensor.shape = {4};
    input_tensor.dtype = trtmc::DType::kFloat32;

    trtmc::TensorMap inputs;
    inputs["x"] = input_tensor;

    module.forward_async(inputs);
    module.sync();

    cudaMemcpy(output, module.device_ptr("y"), 4 * sizeof(float), cudaMemcpyDeviceToHost);
}

static std::unique_ptr<trtmc::TrtModuleImpl>
make_module(nvinfer1::ICudaEngine* engine, cudaStream_t stream, int32_t profile_idx = 0) {
    auto* ctx = engine->createExecutionContext();
    return std::make_unique<trtmc::TrtModuleImpl>(engine, ctx, stream, profile_idx);
}

// --- CudaGraphExec unit tests ---

static void test_default_state() {
    trtmc::CudaGraphExec graph;
    check(!graph.ready(), "default: not ready");
    check(!graph.launch(nullptr), "default: launch returns false");
}

static void test_capture_and_replay() {
    // Capture a simple cudaMemcpyAsync into a graph, then replay it
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float host_src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float host_dst[4] = {0};
    void* d_buf = nullptr;
    cudaMalloc(&d_buf, 16);

    trtmc::CudaGraphExec graph;

    // Capture: H2D copy
    check(graph.begin_capture(stream), "capture: begin ok");
    cudaMemcpyAsync(d_buf, host_src, 16, cudaMemcpyHostToDevice, stream);
    check(graph.end_capture(stream), "capture: end ok");
    check(graph.ready(), "capture: graph is ready");

    // Replay: should copy the same data
    check(graph.launch(stream), "replay: launch ok");
    cudaStreamSynchronize(stream);

    cudaMemcpy(host_dst, d_buf, 16, cudaMemcpyDeviceToHost);
    check(host_dst[0] == 1.0f, "replay: dst[0] = 1.0");
    check(host_dst[3] == 4.0f, "replay: dst[3] = 4.0");

    cudaFree(d_buf);
    cudaStreamDestroy(stream);
}

static void test_reset() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, 16);
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    trtmc::CudaGraphExec graph;
    graph.begin_capture(stream);
    cudaMemcpyAsync(d_buf, data, 16, cudaMemcpyHostToDevice, stream);
    graph.end_capture(stream);
    check(graph.ready(), "reset: ready before reset");

    graph.reset();
    check(!graph.ready(), "reset: not ready after reset");
    check(!graph.launch(stream), "reset: launch fails after reset");

    cudaFree(d_buf);
    cudaStreamDestroy(stream);
}

static void test_double_reset() {
    trtmc::CudaGraphExec graph;
    graph.reset(); // reset on default-constructed — should not crash
    graph.reset(); // double reset — should not crash
    check(!graph.ready(), "double_reset: still not ready");
}

static void test_move_constructor() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, 16);
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    trtmc::CudaGraphExec src;
    src.begin_capture(stream);
    cudaMemcpyAsync(d_buf, data, 16, cudaMemcpyHostToDevice, stream);
    src.end_capture(stream);
    check(src.ready(), "move_ctor: src ready before move");

    trtmc::CudaGraphExec dst(std::move(src));
    check(dst.ready(), "move_ctor: dst ready after move");
    check(!src.ready(), "move_ctor: src not ready after move");

    // dst should still be launchable
    check(dst.launch(stream), "move_ctor: dst launch ok");
    cudaStreamSynchronize(stream);

    cudaFree(d_buf);
    cudaStreamDestroy(stream);
}

static void test_move_assignment() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, 16);
    float data[4] = {5.0f, 6.0f, 7.0f, 8.0f};

    trtmc::CudaGraphExec src;
    src.begin_capture(stream);
    cudaMemcpyAsync(d_buf, data, 16, cudaMemcpyHostToDevice, stream);
    src.end_capture(stream);

    trtmc::CudaGraphExec dst;
    dst = std::move(src);
    check(dst.ready(), "move_assign: dst ready after move");
    check(!src.ready(), "move_assign: src not ready after move");
    check(dst.launch(stream), "move_assign: dst launch ok");
    cudaStreamSynchronize(stream);

    cudaFree(d_buf);
    cudaStreamDestroy(stream);
}

// --- TrtModule CUDA Graph integration tests ---

static void test_module_cuda_graph_correctness() {
    // Run the same inputs with and without CUDA Graphs — output must match
    auto engine = build_identity_engine();
    check(engine != nullptr, "graph_correctness: engine built");
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Run without CUDA Graph
    auto normal = make_module(engine.get(), stream);
    float input[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    float normal_out[4] = {0};
    run_and_read(*normal, input, normal_out);

    // Run with CUDA Graph
    auto graphed = make_module(engine.get(), stream);
    graphed->enable_cuda_graph();
    check(graphed->cuda_graph_active(), "graph_correctness: cuda_graph_active");

    // First call: capture + execute
    float graph_out1[4] = {0};
    run_and_read(*graphed, input, graph_out1);

    // Second call: replay
    float graph_out2[4] = {0};
    run_and_read(*graphed, input, graph_out2);

    // All three should produce identical results
    for (int i = 0; i < 4; ++i) {
        check(normal_out[i] == graph_out1[i], "graph_correctness: capture matches normal");
        check(normal_out[i] == graph_out2[i], "graph_correctness: replay matches normal");
    }

    cudaStreamDestroy(stream);
}

static void test_module_cuda_graph_multiple_runs() {
    // Verify CUDA Graph replay works over many iterations (stable, no drift)
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = make_module(engine.get(), stream);
    module->enable_cuda_graph();

    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    for (int iter = 0; iter < 10; ++iter) {
        float out[4] = {0};
        run_and_read(*module, input, out);
        for (int i = 0; i < 4; ++i) {
            check(out[i] == input[i], "multi_run: output matches input");
        }
    }

    cudaStreamDestroy(stream);
}

static void test_module_enable_after_normal_run() {
    // Enable CUDA Graphs after some normal forward calls
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = make_module(engine.get(), stream);

    // Run 3 normal steps
    float input[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    for (int i = 0; i < 3; ++i) {
        float out[4] = {0};
        run_and_read(*module, input, out);
    }

    // Now enable CUDA Graph
    module->enable_cuda_graph();
    check(module->cuda_graph_active(), "enable_after_normal: active");

    // First call captures, second replays
    float out1[4] = {0};
    run_and_read(*module, input, out1);
    float out2[4] = {0};
    run_and_read(*module, input, out2);

    for (int i = 0; i < 4; ++i) {
        check(out1[i] == input[i], "enable_after_normal: capture output correct");
        check(out2[i] == input[i], "enable_after_normal: replay output correct");
    }

    cudaStreamDestroy(stream);
}

static void test_module_cuda_graph_launch_failure() {
    auto engine = build_identity_engine();
    if (!engine)
        return;

    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    // Initial launch after a successful capture.
    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        auto module = make_module(engine.get(), stream);
        module->enable_cuda_graph();
        trtmc::TrtModuleImplTestPeer::set_cuda_graph_launch_override(*module,
                                                                     fail_cuda_graph_launch);

        bool rejected = false;
        try {
            module->forward_async({{"x", trtmc::Tensor{input, {4}, trtmc::DType::kFloat32}}});
        } catch (const std::runtime_error& error) {
            rejected = std::string(error.what()).find("CUDA Graph launch") != std::string::npos;
        }
        check(rejected, "graph_launch_failure: initial launch rejected");
        check(!module->cuda_graph_active(), "graph_launch_failure: initial graph disabled");
        check(!module->cuda_graph_captured(), "graph_launch_failure: initial graph reset");

        float output[4] = {0};
        run_and_read(*module, input, output);
        check(output[0] == input[0] && output[3] == input[3],
              "graph_launch_failure: initial-launch retry succeeds");
        module.reset();
        cudaStreamDestroy(stream);
    }

    // Replay of an already captured graph.
    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        auto module = make_module(engine.get(), stream);
        module->enable_cuda_graph();

        float output[4] = {0};
        run_and_read(*module, input, output);
        check(module->cuda_graph_captured(), "graph_launch_failure: replay graph captured");
        trtmc::TrtModuleImplTestPeer::set_cuda_graph_launch_override(*module,
                                                                     fail_cuda_graph_launch);

        bool rejected = false;
        try {
            trtmc::TrtModuleImplTestPeer::execute_enqueue(*module);
        } catch (const std::runtime_error& error) {
            rejected = std::string(error.what()).find("CUDA Graph launch") != std::string::npos;
        }
        check(rejected, "graph_launch_failure: replay rejected");
        check(!module->cuda_graph_active(), "graph_launch_failure: replay graph disabled");
        check(!module->cuda_graph_captured(), "graph_launch_failure: replay graph reset");

        std::memset(output, 0, sizeof(output));
        run_and_read(*module, input, output);
        check(output[0] == input[0] && output[3] == input[3],
              "graph_launch_failure: replay retry succeeds");
        module.reset();
        cudaStreamDestroy(stream);
    }
}

int main() {
    // CudaGraphExec unit tests
    test_default_state();
    test_capture_and_replay();
    test_reset();
    test_double_reset();
    test_move_constructor();
    test_move_assignment();

    // TrtModule CUDA Graph integration tests
    test_module_cuda_graph_correctness();
    test_module_cuda_graph_multiple_runs();
    test_module_enable_after_normal_run();
    test_module_cuda_graph_launch_failure();
    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All CUDA Graph tests passed.\n";
    return 0;
}
