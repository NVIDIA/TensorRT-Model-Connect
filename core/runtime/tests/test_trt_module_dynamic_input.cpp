/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/primitives/trt_common.h"
#include "runtime/tensorrt/trt_logger.h"
#include "runtime/tensorrt/trt_module_impl.h"

#include <NvInfer.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace trtmc {
class TrtModuleImplTestPeer {
  public:
    static std::size_t capacity(const TrtModuleImpl& module) { return module.activation_capacity_; }
    static std::size_t required(const TrtModuleImpl& module) { return module.activation_required_; }
    static void* memory(const TrtModuleImpl& module) { return module.activation_memory_; }
    static bool pending(const TrtModuleImpl& module) { return module.activation_work_pending_; }
    static bool dirty(const TrtModuleImpl& module) { return module.activation_memory_dirty_; }
    static bool shape_valid(const TrtModuleImpl& module) {
        return module.buffers_.at("input").shape_valid;
    }
    static std::size_t enqueues(const TrtModuleImpl& module) {
        return module.timing_events_.size();
    }
    static void reserve(TrtModuleImpl& module, std::size_t bytes) {
        module.reserve_activation_memory(bytes);
    }
    static std::size_t query_result(std::size_t bytes, bool error) {
        return TrtModuleImpl::validate_activation_query(bytes, error);
    }
    static bool coherent_alias(void* input, void* output) {
        TrtModuleImpl module(nullptr, nullptr, nullptr);
        module.buffers_["input"].d_ptr = input;
        module.buffers_["input"].is_external = true;
        module.buffers_["output"].d_ptr = output;
        module.buffers_["output"].is_external = true;
        module.alias_outputs_by_input_["input"] = {"output"};
        return module.alias_group_has_address("input", input);
    }
};
} // namespace trtmc

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_identity(bool dynamic, int profiles = 1) {
    static trtmc::TrtLogger logger;
    auto builder = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder)
        return nullptr;
    auto network = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    auto* input = network->addInput("input", nvinfer1::DataType::kFLOAT,
                                    nvinfer1::Dims{1, {dynamic ? -1 : 4}});
    auto* identity = network->addIdentity(*input);
    identity->getOutput(0)->setName("output");
    network->markOutput(*identity->getOutput(0));

    for (int index = 0; dynamic && index < profiles; ++index) {
        auto* profile = builder->createOptimizationProfile();
        profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims{1, {1}});
        profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT,
                               nvinfer1::Dims{1, {index ? 1 : 2}});
        profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX,
                               nvinfer1::Dims{1, {index ? 1 : 4}});
        config->addOptimizationProfile(profile);
    }

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan)
        return nullptr;
    auto runtime = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
}

trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_two_matmuls() {
    static trtmc::TrtLogger logger;
    auto builder = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder)
        return nullptr;
    auto network = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->clearFlag(nvinfer1::BuilderFlag::kTF32);
    config->setBuilderOptimizationLevel(1);
    std::vector<float> first_weights(64 * 512), second_weights(512 * 64);
    for (std::size_t i = 0; i < first_weights.size(); ++i) {
        first_weights[i] = static_cast<float>(static_cast<int>(i * 3 % 11) - 5) / 32.0F;
        second_weights[i] = static_cast<float>(static_cast<int>(i * 5 % 13) - 6) / 64.0F;
    }
    auto* input = network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2{-1, 64});
    auto* first =
        network->addConstant(nvinfer1::Dims2{64, 512},
                             nvinfer1::Weights{nvinfer1::DataType::kFLOAT, first_weights.data(),
                                               static_cast<int64_t>(first_weights.size())});
    auto* second =
        network->addConstant(nvinfer1::Dims2{512, 64},
                             nvinfer1::Weights{nvinfer1::DataType::kFLOAT, second_weights.data(),
                                               static_cast<int64_t>(second_weights.size())});
    auto* product =
        network->addMatrixMultiply(*input, nvinfer1::MatrixOperation::kNONE, *first->getOutput(0),
                                   nvinfer1::MatrixOperation::kNONE);
    auto* activation =
        network->addActivation(*product->getOutput(0), nvinfer1::ActivationType::kRELU);
    auto* output =
        network->addMatrixMultiply(*activation->getOutput(0), nvinfer1::MatrixOperation::kNONE,
                                   *second->getOutput(0), nvinfer1::MatrixOperation::kNONE);
    output->getOutput(0)->setName("output");
    network->markOutput(*output->getOutput(0));
    for (int profile_index = 0; profile_index < 2; ++profile_index) {
        auto* profile = builder->createOptimizationProfile();
        profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{1, 64});
        profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT,
                               nvinfer1::Dims2{profile_index ? 1 : 17, 64});
        profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX,
                               nvinfer1::Dims2{profile_index ? 1 : 257, 64});
        config->addOptimizationProfile(profile);
    }
    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan)
        return nullptr;
    auto runtime = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
}

// Hold the stream before inference so lifetime controls do not rely on a tiny
// network merely happening to remain in flight. The callback uses no CUDA API.
class PendingStream {
  public:
    explicit PendingStream(cudaStream_t stream)
        : stream_(stream), gate_(release_.get_future().share()) {
        const auto error = cudaLaunchHostFunc(
            stream_,
            [](void* value) {
                auto* pending = static_cast<PendingStream*>(value);
                if (pending->gate_.wait_for(std::chrono::seconds(5)) == std::future_status::timeout)
                    pending->timed_out_ = true;
            },
            this);
        if (error != cudaSuccess)
            throw std::runtime_error("unable to queue pending-work control");
    }
    ~PendingStream() {
        release();
        cudaStreamSynchronize(stream_);
        check(!timed_out_, "pending-work control released within its bounded deadline");
    }
    void release() {
        if (!released_) {
            release_.set_value();
            released_ = true;
        }
    }

  private:
    cudaStream_t stream_;
    std::promise<void> release_;
    std::shared_future<void> gate_;
    bool released_{false};
    std::atomic<bool> timed_out_{false};
};

template <class Operation>
void finish_pending_operation(PendingStream& pending, cudaEvent_t done, Operation operation) {
    check(cudaEventQuery(done) == cudaErrorNotReady,
          "lifetime control is actually pending before operation");
    std::promise<void> entered;
    auto started = entered.get_future();
    auto result = std::async(std::launch::async, [&] {
        entered.set_value();
        operation();
    });
    check(started.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
          "lifetime operation starts within its bounded deadline");
    check(result.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout,
          "operation cannot finish while preceding inference remains blocked");
    pending.release();
    result.get();
    check(cudaEventQuery(done) == cudaSuccess, "operation finishes with queued inference complete");
}

void test_first_forward_allocates_unbound_dynamic_input(nvinfer1::ICudaEngine& engine,
                                                        cudaStream_t stream) {
    trtmc::TrtModuleImpl module(&engine, engine.createExecutionContext(), stream);
    check(module.device_ptr("input") == nullptr,
          "dynamic input is not allocated during module construction");
    float values[] = {1.0F, 2.0F};
    const auto outputs =
        module.forward({{"input", trtmc::Tensor{values, {2}, trtmc::DType::kFloat32}}});
    check(module.device_ptr("input") != nullptr,
          "first forward allocates and binds an unbound dynamic input");
    const auto found = outputs.find("output");
    check(found != outputs.end() && found->second.numel() == 2,
          "dynamic identity returns its runtime shape");
    if (found != outputs.end() && found->second.numel() == 2) {
        const auto* output = static_cast<const float*>(found->second.data);
        check(std::fabs(output[0] - 1.0F) < 1.0e-6F && std::fabs(output[1] - 2.0F) < 1.0e-6F,
              "dynamic identity returns its input values");
    }
}

void test_external_dynamic_input_stays_external(nvinfer1::ICudaEngine& engine,
                                                cudaStream_t stream) {
    void* external = nullptr;
    if (cudaMalloc(&external, 4 * sizeof(float)) != cudaSuccess) {
        ++failures;
        return;
    }
    {
        trtmc::TrtModuleImpl module(&engine, engine.createExecutionContext(), stream);
        check(module.device_ptr("input") == nullptr,
              "second dynamic input also starts without an allocation");
        module.bind_external("input", external, {2});
        float values[] = {3.0F, 4.0F};
        (void)module.forward({{"input", trtmc::Tensor{values, {2}, trtmc::DType::kFloat32}}});
        check(module.device_ptr("input") == external,
              "forward preserves the pre-bound dynamic input buffer");
    }
    cudaFree(external);
}

void test_backend_managed_graph_does_not_enable_nested_capture(nvinfer1::ICudaEngine& engine,
                                                               cudaStream_t stream) {
    trtmc::TrtModuleImpl module(&engine, engine.createExecutionContext(), stream, 0, nullptr, {},
                                true);
    module.enable_cuda_graph();
    check(!module.cuda_graph_active(), "backend-managed graph leaves stream capture disabled");
}

void test_short_static_upload_does_not_keep_the_previous_input(nvinfer1::ICudaEngine& engine,
                                                               cudaStream_t stream) {
    for (bool graph : {false, true}) {
        trtmc::TrtModuleImpl module(&engine, engine.createExecutionContext(), stream);
        if (graph)
            module.enable_cuda_graph();
        float full[] = {1.0F, 2.0F, 3.0F, 4.0F};
        (void)module.forward({{"input", trtmc::Tensor{full, {4}, trtmc::DType::kFloat32}}});
        float short_input[] = {7.0F, 8.0F};
        const auto outputs =
            module.forward({{"input", trtmc::Tensor{short_input, {2}, trtmc::DType::kFloat32}}});
        const auto& tensor = outputs.at("output");
        const auto* values = static_cast<const float*>(tensor.data);
        check(tensor.numel() == 4 && values[0] == 7.0F && values[1] == 8.0F,
              "static input keeps the current uploaded prefix");
        check(values[2] == 0.0F && values[3] == 0.0F,
              "short static upload clears bytes from the previous longer request");
    }
}

void test_short_external_static_upload_preserves_its_tail(nvinfer1::ICudaEngine& engine,
                                                          cudaStream_t stream) {
    void* external = nullptr;
    if (cudaMalloc(&external, 4 * sizeof(float)) != cudaSuccess) {
        ++failures;
        return;
    }
    {
        trtmc::TrtModuleImpl module(&engine, engine.createExecutionContext(), stream);
        module.bind_external("input", external, {4});
        float full[] = {1.0F, 2.0F, 3.0F, 4.0F};
        (void)module.forward({{"input", trtmc::Tensor{full, {4}, trtmc::DType::kFloat32}}});
        float prefix[] = {7.0F, 8.0F};
        const auto outputs =
            module.forward({{"input", trtmc::Tensor{prefix, {2}, trtmc::DType::kFloat32}}});
        const auto* values = static_cast<const float*>(outputs.at("output").data);
        check(values[0] == 7.0F && values[1] == 8.0F && values[2] == 3.0F && values[3] == 4.0F,
              "short upload preserves externally owned static state beyond its prefix");
        check(module.device_ptr("input") == external, "static input keeps external ownership");
    }
    cudaFree(external);
}

void test_activation_query_errors_and_alias_metadata() {
    using Peer = trtmc::TrtModuleImplTestPeer;
    check(Peer::query_result(0, false) == 0,
          "zero activation requirement is valid without a TRT error");
    bool rejected = false;
    try {
        (void)Peer::query_result(0, true);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    check(rejected, "a query error returning zero cannot become an empty successful allocation");
    int left = 0, right = 0;
    check(Peer::coherent_alias(&left, &left), "same-address coherent alias metadata is a no-op");
    check(!Peer::coherent_alias(&left, &right), "partially rebound alias metadata is not a no-op");
    check(!Peer::coherent_alias(nullptr, nullptr), "unbound null aliases are not admitted");
}

void check_identity_result(trtmc::TrtModuleImpl& module, const float* values, int count) {
    const auto output = module.forward(
        {{"input", trtmc::Tensor{const_cast<float*>(values), {count}, trtmc::DType::kFloat32}}});
    const auto& tensor = output.at("output");
    check(tensor.numel() == static_cast<std::size_t>(count),
          "deferred context preserves actual output extent");
    check(std::memcmp(tensor.data, values, count * sizeof(float)) == 0,
          "deferred context preserves every identity output bit");
}

void test_deferred_shape_and_graph(nvinfer1::ICudaEngine& engine, cudaStream_t stream, bool graph) {
    using Peer = trtmc::TrtModuleImplTestPeer;
    trtmc::TrtModuleImpl module(
        &engine,
        engine.createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED),
        stream, 0, nullptr, {}, false, true);
    check(module.ok() && Peer::capacity(module) == 0 && Peer::memory(module) == nullptr,
          "deferred construction does not allocate MAX/OPT activation storage");
    if (graph)
        module.enable_cuda_graph();
    float values[] = {1.0F, -2.0F, 3.0F, -4.0F, 5.0F};
    for (int count : {2, 4, 1, 4}) {
        check_identity_result(module, values, count);
        check(Peer::capacity(module) >= Peer::required(module),
              "owned storage covers actual queried activation bytes");
    }
    bool rejected = false;
    try {
        (void)module.forward({{"input", trtmc::Tensor{values, {5}, trtmc::DType::kFloat32}}});
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    check(rejected && !Peer::shape_valid(module), "rejected shape is not cached as admitted");
    check_identity_result(module, values, 4);
    check(Peer::shape_valid(module), "retrying the previous shape revalidates the actual context");
    const auto before = Peer::enqueues(module);
    module.forward_async({{"input", trtmc::Tensor{values, {4}, trtmc::DType::kFloat32}}});
    module.bind_external("input", module.device_ptr("input"), {4});
    check(Peer::pending(module),
          "same-pointer/same-shape binding does not synchronize an async submission");
    module.forward_async({{"input", trtmc::Tensor{values, {4}, trtmc::DType::kFloat32}}});
    check(Peer::enqueues(module) == before + 2, "two async calls add exactly two timed enqueues");
    check(!graph || module.cuda_graph_captured(), "same-shape graph remains captured");
    module.sync();
}

void test_deferred_growth_and_allocation_failure(nvinfer1::ICudaEngine& engine,
                                                 cudaStream_t stream) {
    using Peer = trtmc::TrtModuleImplTestPeer;
    trtmc::DeviceTensor input({2}, trtmc::DType::kFloat32, stream);
    trtmc::TrtModuleImpl module(
        &engine,
        engine.createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED),
        stream, 0, nullptr, {}, false, true);
    float values[] = {7.0F, -8.0F};
    module.bind_external("input", input.data(), {2});
    check_identity_result(module, values, 2);
    module.enable_cuda_graph();
    check_identity_result(module, values, 2);
    cudaEvent_t done = nullptr;
    check(cudaEventCreate(&done) == cudaSuccess, "create async lifetime marker");
    const auto old_capacity = Peer::capacity(module);
    {
        PendingStream pending(stream);
        module.forward_device_async({{"input", &input}});
        check(cudaEventRecord(done, stream) == cudaSuccess, "record async lifetime marker");
        finish_pending_operation(pending, done, [&] { Peer::reserve(module, old_capacity + 256); });
    }
    check(!module.cuda_graph_captured(), "growth invalidates the captured graph");
    const auto capacity = Peer::capacity(module);
    const auto pointer = Peer::memory(module);
    check(capacity == old_capacity + 256 && pointer != nullptr,
          "bounded explicit reserve grows owned storage");
    bool rejected = false;
    try {
        Peer::reserve(module, static_cast<std::size_t>(std::numeric_limits<int64_t>::max()));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    (void)cudaGetLastError(); // Expected allocation failure is not a kernel failure.
    check(rejected && Peer::capacity(module) == capacity && Peer::memory(module) == pointer &&
              Peer::dirty(module),
          "failed allocation preserves prior ownership and requires fresh admission");
    check_identity_result(module, values, 2);
    check(Peer::capacity(module) == capacity, "smaller valid retry reuses high-water capacity");
    cudaEventDestroy(done);
}

void test_deferred_decode_and_destruction(nvinfer1::ICudaEngine& engine, cudaStream_t stream) {
    float value = -3.0F;
    trtmc::DeviceTensor input({1}, trtmc::DType::kFloat32, stream);
    trtmc::DeviceTensor output({1}, trtmc::DType::kFloat32, stream);
    cudaEvent_t done = nullptr;
    check(cudaEventCreate(&done) == cudaSuccess, "create destruction lifetime marker");
    {
        auto module = std::make_unique<trtmc::TrtModuleImpl>(
            &engine,
            engine.createExecutionContext(
                nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED),
            stream, 1, nullptr, std::vector<trtmc::ModuleExternalBinding>{}, false, true);
        check_identity_result(*module, &value, 1);
        check(cudaMemcpyAsync(input.data(), &value, sizeof(value), cudaMemcpyHostToDevice,
                              stream) == cudaSuccess,
              "copy device-path input");
        module->bind_external("input", input.data(), {1});
        module->bind_external("output", output.data());
        cudaStreamSynchronize(stream);
        PendingStream pending(stream);
        module->forward_device_async({{"input", &input}});
        check(cudaEventRecord(done, stream) == cudaSuccess,
              "record work before deferred destruction");
        finish_pending_operation(pending, done, [&] { module.reset(); });
    }
    check(cudaEventQuery(done) == cudaSuccess, "deferred destruction drains pending work");
    cudaEventDestroy(done);
}

std::vector<float> run_matrix(trtmc::TrtModuleImpl& module, const std::vector<float>& values,
                              int rows, bool device, cudaStream_t stream) {
    std::vector<float> output(values.size());
    if (device) {
        trtmc::DeviceTensor input({rows, 64}, trtmc::DType::kFloat32, stream);
        check(cudaMemcpyAsync(input.data(), values.data(), values.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream) == cudaSuccess,
              "upload matrix input");
        module.forward_device({{"input", &input}});
        check(cudaMemcpy(output.data(), module.device_ptr("output"), output.size() * sizeof(float),
                         cudaMemcpyDeviceToHost) == cudaSuccess,
              "read device-path matrix output");
    } else {
        const auto result =
            module.forward({{"input", trtmc::Tensor{const_cast<float*>(values.data()),
                                                    {rows, 64},
                                                    trtmc::DType::kFloat32}}});
        const auto& tensor = result.at("output");
        check(tensor.numel() == output.size(), "matrix output uses the actual logical extent");
        if (tensor.numel() != output.size())
            return {};
        std::memcpy(output.data(), tensor.data, output.size() * sizeof(float));
    }
    return output;
}

void compare_matrix(trtmc::TrtModuleImpl& baseline, trtmc::TrtModuleImpl& deferred, int rows,
                    bool device, cudaStream_t stream) {
    std::vector<float> values(static_cast<std::size_t>(rows) * 64);
    for (std::size_t index = 0; index < values.size(); ++index)
        values[index] = static_cast<float>(static_cast<int>(index % 17) - 8) / 16.0F;
    const auto expected = run_matrix(baseline, values, rows, device, stream);
    const auto actual = run_matrix(deferred, values, rows, device, stream);
    check(actual.size() == expected.size() && !actual.empty() &&
              std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0,
          "same-engine STATIC and USER_MANAGED nontrivial outputs are bit-identical");
    check(std::any_of(actual.begin(), actual.end(), [](float value) { return value != 0.0F; }),
          "nontrivial matrix control produces nonzero outputs");
}

void test_nonzero_activation_profile(nvinfer1::ICudaEngine& engine, cudaStream_t stream,
                                     int profile, bool graph) {
    using Peer = trtmc::TrtModuleImplTestPeer;
    trtmc::TrtModuleImpl baseline(&engine, engine.createExecutionContext(), stream, profile);
    trtmc::TrtModuleImpl deferred(
        &engine,
        engine.createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED),
        stream, profile, nullptr, {}, false, true);
    if (graph) {
        baseline.enable_cuda_graph();
        deferred.enable_cuda_graph();
    }
    const std::vector<int> rows =
        profile == 0 ? std::vector<int>{1, 17, 257, 17, 1, 1} : std::vector<int>{1, 1, 1};
    for (bool device : {false, true}) {
        for (int count : rows) {
            compare_matrix(baseline, deferred, count, device, stream);
            check(Peer::required(deferred) > 0,
                  "actual nontrivial query requires nonzero activation storage");
            check(Peer::capacity(deferred) >= Peer::required(deferred),
                  "actual query fits owned activation storage");
            std::cout << "activation profile=" << profile << " rows=" << count
                      << " device=" << device << " graph=" << graph
                      << " required=" << Peer::required(deferred)
                      << " capacity=" << Peer::capacity(deferred) << '\n';
        }
    }
    const auto pointer = Peer::memory(deferred);
    const auto capacity = Peer::capacity(deferred);
    deferred.reset_execution_context();
    compare_matrix(baseline, deferred, 1, false, stream);
    check(Peer::memory(deferred) == pointer && Peer::capacity(deferred) == capacity,
          "request reset preserves valid high-water storage and output behavior");
}

} // namespace

int main() {
    test_activation_query_errors_and_alias_metadata();
    if (failures != 0)
        return failures;
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        return 77;
    auto engine = build_identity(true);
    auto static_engine = build_identity(false);
    auto dual_engine = build_identity(true, 2);
    if (!engine || !static_engine || !dual_engine)
        return 1;
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess)
        return 1;
    test_first_forward_allocates_unbound_dynamic_input(*engine, stream);
    test_external_dynamic_input_stays_external(*engine, stream);
    test_backend_managed_graph_does_not_enable_nested_capture(*engine, stream);
    test_short_static_upload_does_not_keep_the_previous_input(*static_engine, stream);
    test_short_external_static_upload_preserves_its_tail(*static_engine, stream);
    test_deferred_shape_and_graph(*dual_engine, stream, false);
    test_deferred_shape_and_graph(*dual_engine, stream, true);
    test_deferred_growth_and_allocation_failure(*dual_engine, stream);
    test_deferred_decode_and_destruction(*dual_engine, stream);
    auto matrix_engine = build_two_matmuls();
    if (!matrix_engine)
        return 1;
    for (int profile : {0, 1}) {
        test_nonzero_activation_profile(*matrix_engine, stream, profile, false);
        test_nonzero_activation_profile(*matrix_engine, stream, profile, true);
    }
    cudaStreamDestroy(stream);
    return failures;
}
