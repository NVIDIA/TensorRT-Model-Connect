/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/primitives/trt_common.h"
#include "runtime/tensorrt/trt_logger.h"
#include "runtime/tensorrt/trt_module_impl.h"

#include <NvInfer.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace trtmc {

class TrtModuleImplTestPeer {
  public:
    static DType from_trt_dtype(nvinfer1::DataType dtype) {
        return TrtModuleImpl::from_trt_dtype(dtype);
    }
    static void validate_io_tensor_format(nvinfer1::DataType dtype, nvinfer1::TensorFormat format) {
        TrtModuleImpl::validate_io_tensor_format(dtype, format);
    }
    static bool has_no_bindings_or_buffers(const TrtModuleImpl& module) {
        return module.initial_external_bindings_.empty() && module.buffers_.empty();
    }
    static std::size_t allocation_bytes(const nvinfer1::Dims& dims, DType dtype) {
        std::vector<int64_t> shape;
        return TrtModuleImpl::compute_alloc_bytes(dims, dtype, shape);
    }

    static std::size_t input_capacity(const TrtModuleImpl& module, const std::string& name) {
        return module.buffers_.at(name).input_capacity_bytes;
    }

    static std::size_t profile_capacity(const TrtModuleImpl& module, const std::string& name) {
        return module.buffers_.at(name).nbytes;
    }

    static std::size_t output_capacity(const TrtModuleImpl& module, const std::string& name) {
        return module.buffers_.at(name).output_capacity_bytes;
    }

    static std::size_t host_staging_size(const TrtModuleImpl& module, const std::string& name) {
        const auto found = module.host_output_staging_.find(name);
        return found == module.host_output_staging_.end() ? 0 : found->second.size();
    }

    static bool shares_activation_arena(const TrtModuleImpl& first, const TrtModuleImpl& second) {
        return first.activation_arena_ && first.activation_arena_ == second.activation_arena_;
    }

    static std::int64_t activation_profile_bytes(const TrtModuleImpl& module) {
        return module.engine_->getDeviceMemorySizeForProfileV2(module.profile_idx_);
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

// The small test context owns its activation allocation. This coordinator
// exercises the serial opt-in and records shape invalidation independently.
class RecordingActivationArena final : public trtmc::ITrtActivationArena {
  public:
    void attach(nvinfer1::IExecutionContext*, cudaStream_t, std::int64_t) override {}
    void begin_enqueue(nvinfer1::IExecutionContext*) override {}
    void end_enqueue() noexcept override {}
    void invalidate_shapes(nvinfer1::IExecutionContext*) override { ++invalidations; }
    void detach(nvinfer1::IExecutionContext*) noexcept override {}

    int invalidations{0};
};

void test_fp8_tensor_storage_and_mapping() {
    const auto dtype = trtmc::TrtModuleImplTestPeer::from_trt_dtype(nvinfer1::DataType::kFP8);
    check(dtype == trtmc::DType::kFloat8E4M3, "TensorRT FP8 maps to E4M3, not float32");
    check(trtmc::dtype_size(dtype) == 1, "E4M3 elements occupy one byte");
    uint8_t values[] = {0x00, 0x38, 0xb8, 0x7e, 0xfe, 0x01};
    const trtmc::Tensor tensor{values, {2, 3}, dtype};
    check(tensor.numel() == 6 && tensor.nbytes() == sizeof(values),
          "FP8 host tensor sizes use byte storage");
    check(trtmc::TrtModuleImplTestPeer::allocation_bytes(nvinfer1::Dims{2, {2, 3}}, dtype) ==
              sizeof(values),
          "FP8 engine buffer allocation uses byte storage");
    check(trtmc::TrtModuleImplTestPeer::from_trt_dtype(nvinfer1::DataType::kINT8) ==
              trtmc::DType::kInt8,
          "signed integer storage remains distinct from E4M3");
    trtmc::TrtModuleImplTestPeer::validate_io_tensor_format(nvinfer1::DataType::kFP8,
                                                            nvinfer1::TensorFormat::kLINEAR);
    for (const auto format : {nvinfer1::TensorFormat::kHWC16, nvinfer1::TensorFormat::kCHW32}) {
        bool rejected = false;
        try {
            trtmc::TrtModuleImplTestPeer::validate_io_tensor_format(nvinfer1::DataType::kFP8,
                                                                    format);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "FP8 vectorized I/O is rejected before byte-buffer allocation");
    }
}

trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_identity(bool dynamic,
                                                          bool vectorized_fp8 = false) {
    static trtmc::TrtLogger logger;
    auto builder = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder)
        return nullptr;
    auto network = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    const auto dtype = vectorized_fp8 ? nvinfer1::DataType::kFP8 : nvinfer1::DataType::kFLOAT;
    const auto dims =
        vectorized_fp8 ? nvinfer1::Dims{4, {1, 4, 1, 1}} : nvinfer1::Dims{1, {dynamic ? -1 : 4}};
    auto* input = network->addInput("input", dtype, dims);
    auto* identity = network->addIdentity(*input);
    identity->getOutput(0)->setName("output");
    network->markOutput(*identity->getOutput(0));
    if (vectorized_fp8) {
        const auto formats = 1U << static_cast<unsigned>(nvinfer1::TensorFormat::kHWC16);
        input->setAllowedFormats(formats);
        identity->getOutput(0)->setAllowedFormats(formats);
    }

    if (dynamic) {
        auto* profile = builder->createOptimizationProfile();
        profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims{1, {1}});
        profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims{1, {2}});
        profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims{1, {4}});
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

void check_vectorized_fp8_rejected(nvinfer1::ICudaEngine& engine, cudaStream_t stream,
                                   void* external, bool prebind) {
    std::vector<trtmc::ModuleExternalBinding> bindings;
    if (prebind)
        bindings.push_back({"input", external, 16});
    auto* context = engine.createExecutionContext();
    check(context != nullptr, "FP8 fixture creates a valid context before rejection");
    if (!context)
        return;
    trtmc::TrtModuleImpl module(&engine, context, stream, 0, nullptr, bindings);
    check(!module.ok(), "constructor rejects non-linear FP8 I/O");
    check(trtmc::TrtModuleImplTestPeer::has_no_bindings_or_buffers(module),
          "constructor rejects FP8 format before prebinding or buffer allocation");
    check(module.device_ptr("input") == nullptr && module.device_ptr("output") == nullptr,
          "rejected FP8 module exposes no device buffers");
}

void test_constructor_rejects_vectorized_fp8(cudaStream_t stream) {
    // The storage checks above run everywhere. This real-engine fixture needs
    // FP8 tactics (compute capability 8.9 or newer), not merely byte storage.
    int device = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        check(false, "FP8 fixture can query the current CUDA device");
        return;
    }
    if (properties.major * 10 + properties.minor < 89) {
        std::cout << "SKIP: FP8 engine fixture requires FP8-capable TensorRT tactics\n";
        return;
    }
    auto engine = build_identity(false, true);
    check(engine != nullptr, "vectorized FP8 engine fixture builds");
    if (!engine)
        return;
    for (const auto* name : {"input", "output"}) {
        check(engine->getTensorDataType(name) == nvinfer1::DataType::kFP8 &&
                  engine->getTensorFormat(name) == nvinfer1::TensorFormat::kHWC16,
              "fixture really exposes vectorized FP8 I/O");
    }
    void* external = nullptr;
    if (cudaMalloc(&external, 16) != cudaSuccess) {
        check(false, "FP8 fixture can allocate a correctly padded external buffer");
        return;
    }
    for (bool prebind : {false, true})
        check_vectorized_fp8_rejected(*engine, stream, external, prebind);
    cudaFree(external);
}

void test_first_forward_allocates_unbound_dynamic_input(nvinfer1::ICudaEngine& engine,
                                                        cudaStream_t stream) {
    trtmc::TrtModuleImpl module(&engine, engine.createExecutionContext(), stream);
    check(module.device_ptr("input") == nullptr,
          "dynamic input is not allocated during module construction");
    check(module.device_ptr("output") != nullptr,
          "ordinary contexts retain eager output allocation");
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

void test_rejected_shape_fails_without_poisoning_the_next_request(nvinfer1::ICudaEngine& engine,
                                                                  cudaStream_t stream) {
    for (bool graph : {false, true}) {
        trtmc::TrtModuleImpl module(&engine, engine.createExecutionContext(), stream);
        if (graph)
            module.enable_cuda_graph();
        float initial[] = {1.0F, 2.0F};
        (void)module.forward({{"input", trtmc::Tensor{initial, {2}, trtmc::DType::kFloat32}}});
        float oversized[] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
        bool rejected = false;
        try {
            (void)module.forward(
                {{"input", trtmc::Tensor{oversized, {5}, trtmc::DType::kFloat32}}});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "out-of-profile shapes must fail instead of returning stale-shape outputs");
        float next[] = {7.0F, 8.0F};
        const auto output =
            module.forward({{"input", trtmc::Tensor{next, {2}, trtmc::DType::kFloat32}}});
        const auto& tensor = output.at("output");
        const auto* values = static_cast<const float*>(tensor.data);
        check(tensor.numel() == 2 && values[0] == 7.0F && values[1] == 8.0F,
              "a rejected shape leaves the next valid request usable");
    }
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

void check_identity_output(const trtmc::TensorMap& outputs, const float* expected, int64_t count) {
    const auto found = outputs.find("output");
    check(found != outputs.end() && found->second.shape == std::vector<int64_t>{count},
          "lazy output returns the actual runtime shape");
    if (found == outputs.end() || found->second.numel() != static_cast<std::size_t>(count))
        return;
    const auto* output = static_cast<const float*>(found->second.data);
    for (int64_t index = 0; index < count; ++index)
        check(std::fabs(output[index] - expected[index]) < 1.0e-6F,
              "lazy output preserves every output value");
}

void test_serial_dynamic_output_grows_and_reuses(nvinfer1::ICudaEngine& engine,
                                                 cudaStream_t stream) {
    auto arena = std::make_shared<RecordingActivationArena>();
    trtmc::TrtModuleImpl module(&engine, engine.createExecutionContext(), stream, 0, nullptr, {},
                                false, arena);
    check(module.device_ptr("output") == nullptr,
          "serial dynamic output does not allocate at profile MAX during construction");
    check(trtmc::TrtModuleImplTestPeer::host_staging_size(module, "output") == 0,
          "serial dynamic output does not allocate host staging during construction");
    check(module.tensor_shape("output") == std::vector<int64_t>{4},
          "lazy allocation preserves profile MAX output metadata");
    bool graph_rejected = false;
    try {
        module.enable_cuda_graph();
    } catch (const std::logic_error&) {
        graph_rejected = true;
    }
    check(graph_rejected, "serial live-shape outputs cannot enter stream graph capture");

    float values[] = {2.0F, -3.0F, 4.0F, 5.0F};
    check_identity_output(
        module.forward({{"input", trtmc::Tensor{values, {1}, trtmc::DType::kFloat32}}}), values, 1);
    check(trtmc::TrtModuleImplTestPeer::output_capacity(module, "output") == sizeof(float),
          "first enqueue allocates only its actual output shape");
    check(trtmc::TrtModuleImplTestPeer::input_capacity(module, "input") == sizeof(float),
          "first host forward allocates only its actual input shape");
    check(trtmc::TrtModuleImplTestPeer::profile_capacity(module, "input") == 4 * sizeof(float),
          "actual input allocation preserves the profile MAX capacity contract");
    check(trtmc::TrtModuleImplTestPeer::host_staging_size(module, "output") == sizeof(float),
          "first download allocates only its actual host output shape");

    check_identity_output(
        module.forward({{"input", trtmc::Tensor{values, {4}, trtmc::DType::kFloat32}}}), values, 4);
    void* const grown_output = module.device_ptr("output");
    void* const grown_input = module.device_ptr("input");
    check(trtmc::TrtModuleImplTestPeer::output_capacity(module, "output") == 4 * sizeof(float),
          "larger enqueue grows output storage through profile MAX");
    check(trtmc::TrtModuleImplTestPeer::input_capacity(module, "input") == 4 * sizeof(float),
          "larger host forward grows input storage through profile MAX");

    values[0] = -7.0F;
    check_identity_output(
        module.forward({{"input", trtmc::Tensor{values, {1}, trtmc::DType::kFloat32}}}), values, 1);
    check(module.device_ptr("output") == grown_output,
          "smaller enqueue reuses the high-water output allocation");
    check(module.device_ptr("input") == grown_input &&
              trtmc::TrtModuleImplTestPeer::input_capacity(module, "input") == 4 * sizeof(float),
          "smaller host forward reuses the high-water input allocation");
    check(trtmc::TrtModuleImplTestPeer::host_staging_size(module, "output") == sizeof(float),
          "smaller download reports only its actual host output shape");
    check(arena->invalidations == 3, "A to B to A invalidates activation shape requirements");
    (void)module.forward({{"input", trtmc::Tensor{values, {1}, trtmc::DType::kFloat32}}});
    check(arena->invalidations == 3,
          "unchanged input shape preserves activation requirement cache");
}

void test_serial_device_forward_defers_host_staging(nvinfer1::ICudaEngine& engine,
                                                    cudaStream_t stream) {
    auto arena = std::make_shared<RecordingActivationArena>();
    trtmc::TrtModuleImpl module(&engine, engine.createExecutionContext(), stream, 0, nullptr, {},
                                false, arena);
    float values[] = {8.0F, 9.0F, -10.0F, 11.0F};
    std::size_t high_water = 0;
    void* grown_input = nullptr;
    for (const int64_t count : {1, 4, 1}) {
        trtmc::DeviceTensor input({count}, trtmc::DType::kFloat32, stream);
        check(input.ok(), "device-forward input allocation succeeds");
        if (!input.ok())
            return;
        const auto bytes = static_cast<std::size_t>(count) * sizeof(float);
        cudaMemcpyAsync(input.data(), values, bytes, cudaMemcpyHostToDevice, stream);
        module.forward_device_async({{"input", &input}});
        module.sync();
        high_water = std::max(high_water, bytes);
        check(trtmc::TrtModuleImplTestPeer::input_capacity(module, "input") == high_water,
              "device forward allocates actual input capacity and reuses its high-water mark");
        if (count == 4)
            grown_input = module.device_ptr("input");
        else if (grown_input)
            check(module.device_ptr("input") == grown_input,
                  "smaller device forward preserves the grown input allocation");
        check(module.device_ptr("output") != nullptr,
              "device-forward path allocates its live output before enqueue");
        check(trtmc::TrtModuleImplTestPeer::host_staging_size(module, "output") == 0,
              "device-only forward never allocates host output staging");
        float output[4]{};
        cudaMemcpy(output, module.device_ptr("output"), bytes, cudaMemcpyDeviceToHost);
        for (int64_t index = 0; index < count; ++index)
            check(output[index] == values[index], "device-forward lazy output preserves values");
        values[0] += 1.0F;
    }
    check(arena->invalidations == 3,
          "device A to B to A invalidates activation shape requirements");
}

void test_serial_input_rejects_invalid_requests_before_allocation(nvinfer1::ICudaEngine& engine,
                                                                  cudaStream_t stream) {
    auto arena = std::make_shared<RecordingActivationArena>();
    trtmc::TrtModuleImpl module(&engine, engine.createExecutionContext(), stream, 0, nullptr, {},
                                false, arena);
    float values[] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    for (const auto& shape :
         {std::vector<int64_t>{0}, std::vector<int64_t>{5}, std::vector<int64_t>{1, 1}}) {
        bool rejected = false;
        try {
            module.forward_async({{"input", trtmc::Tensor{values, shape, trtmc::DType::kFloat32}}});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected && module.device_ptr("input") == nullptr,
              "invalid dynamic input shape is rejected before allocation");
    }
    bool dtype_rejected = false;
    try {
        module.forward_async({{"input", trtmc::Tensor{values, {1}, trtmc::DType::kInt32}}});
    } catch (const std::invalid_argument&) {
        dtype_rejected = true;
    }
    check(dtype_rejected && module.device_ptr("input") == nullptr,
          "invalid dynamic input dtype is rejected before allocation");
    check_identity_output(
        module.forward({{"input", trtmc::Tensor{values, {1}, trtmc::DType::kFloat32}}}), values, 1);
}

void test_serial_external_input_stays_external(nvinfer1::ICudaEngine& engine, cudaStream_t stream) {
    void* external = nullptr;
    if (cudaMalloc(&external, 4 * sizeof(float)) != cudaSuccess) {
        ++failures;
        return;
    }
    for (const bool initial_binding : {false, true}) {
        {
            auto arena = std::make_shared<RecordingActivationArena>();
            const std::vector<trtmc::ModuleExternalBinding> bindings =
                initial_binding ? std::vector<trtmc::ModuleExternalBinding>{{"input", external,
                                                                             4 * sizeof(float)}}
                                : std::vector<trtmc::ModuleExternalBinding>{};
            trtmc::TrtModuleImpl module(&engine, engine.createExecutionContext(), stream, 0,
                                        nullptr, bindings, false, arena);
            if (!initial_binding)
                module.bind_external("input", external, {1});
            float values[] = {-1.0F, 2.0F, 3.0F, 4.0F};
            for (const int64_t count : {1, 4, 1}) {
                check_identity_output(
                    module.forward(
                        {{"input", trtmc::Tensor{values, {count}, trtmc::DType::kFloat32}}}),
                    values, count);
                check(module.device_ptr("input") == external &&
                          trtmc::TrtModuleImplTestPeer::input_capacity(module, "input") == 0,
                      "serial dynamic forward never allocates or replaces an external input");
            }
        }
        check(cudaMemset(external, 0, 4 * sizeof(float)) == cudaSuccess,
              "destroying a serial module does not free externally owned input memory");
    }
    cudaFree(external);
}

void test_serial_external_output_stays_external(nvinfer1::ICudaEngine& engine,
                                                cudaStream_t stream) {
    void* external = nullptr;
    if (cudaMalloc(&external, 4 * sizeof(float)) != cudaSuccess) {
        ++failures;
        return;
    }
    {
        auto arena = std::make_shared<RecordingActivationArena>();
        const std::vector<trtmc::ModuleExternalBinding> bindings{
            {"output", external, 4 * sizeof(float)}};
        trtmc::TrtModuleImpl module(&engine, engine.createExecutionContext(), stream, 0, nullptr,
                                    bindings, false, arena);
        float values[] = {-1.0F, 2.0F, 3.0F, 4.0F};
        const auto outputs =
            module.forward({{"input", trtmc::Tensor{values, {4}, trtmc::DType::kFloat32}}});
        check(module.device_ptr("output") == external,
              "serial enqueue preserves the initial external output binding");
        check(outputs.count("output") == 0,
              "external output is not silently downloaded or returned as host data");
        check(trtmc::TrtModuleImplTestPeer::host_staging_size(module, "output") == 0,
              "external output never receives host staging");
        float output[4]{};
        cudaMemcpy(output, external, sizeof(output), cudaMemcpyDeviceToHost);
        for (int index = 0; index < 4; ++index)
            check(output[index] == values[index], "external output preserves all result values");
    }
    check(cudaMemset(external, 0, 4 * sizeof(float)) == cudaSuccess,
          "destroying a serial module does not free externally owned output memory");
    cudaFree(external);
}

// Own only this uniquely created directory and its one generated plan. Never
// overwrite an existing path or remove anything recursively.
struct TemporaryArenaPlan {
    std::filesystem::path directory;
    std::filesystem::path path;

    TemporaryArenaPlan() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (int attempt = 0; attempt < 16; ++attempt) {
            const auto candidate =
                std::filesystem::temp_directory_path() /
                ("trtmc-rtx-arena-" + std::to_string(stamp) + "-" + std::to_string(attempt));
            if (std::filesystem::create_directory(candidate)) {
                directory = candidate;
                path = directory / "dynamic.plan";
                return;
            }
        }
        throw std::runtime_error("unable to create a unique test plan directory");
    }

    ~TemporaryArenaPlan() {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
        std::filesystem::remove(directory, ignored);
    }
};

void add_softmax_test_profile(nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config) {
    auto* profile = builder.createOptimizationProfile();
    if (!profile ||
        !profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN,
                                nvinfer1::Dims{2, {8, 16}}) ||
        !profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT,
                                nvinfer1::Dims{2, {32, 16}}) ||
        !profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX,
                                nvinfer1::Dims{2, {512, 16}}) ||
        config.addOptimizationProfile(profile) < 0)
        throw std::runtime_error("cannot create the RTX test dynamic profile");
}

trtmc::TrtUniquePtr<nvinfer1::IHostMemory> build_dynamic_softmax_plan() {
    static trtmc::TrtLogger logger;
    auto builder = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder)
        throw std::runtime_error("cannot create the RTX test builder");
    auto network = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!network || !config)
        throw std::runtime_error("cannot create the RTX test network/config");
    auto* input =
        network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{2, {-1, 16}});
    if (!input)
        throw std::runtime_error("cannot create the RTX test input");
    auto* scores = network->addMatrixMultiply(*input, nvinfer1::MatrixOperation::kNONE, *input,
                                              nvinfer1::MatrixOperation::kTRANSPOSE);
    if (!scores)
        throw std::runtime_error("cannot create the RTX test matrix product");
    auto* softmax = network->addSoftMax(*scores->getOutput(0));
    if (!softmax)
        throw std::runtime_error("cannot create the RTX test softmax");
    softmax->setAxes(1U << 1U);
    softmax->getOutput(0)->setName("output");
    network->markOutput(*softmax->getOutput(0));
    add_softmax_test_profile(*builder, *config);
    return trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
}

void check_softmax_matrix(trtmc::ITrtModule& module, int64_t rows) {
    std::vector<float> values(static_cast<std::size_t>(rows) * 16, 1.0F);
    const auto outputs = module.forward(
        {{"input", trtmc::Tensor{values.data(), {rows, 16}, trtmc::DType::kFloat32}}});
    const auto found = outputs.find("output");
    const bool correct_shape =
        found != outputs.end() && found->second.shape == std::vector<int64_t>{rows, rows};
    check(correct_shape, "real arena preserves dynamic matrix/softmax output shape");
    if (!correct_shape)
        return;
    const auto* output = static_cast<const float*>(found->second.data);
    const float expected = 1.0F / static_cast<float>(rows);
    bool correct_values = true;
    for (int64_t index = 0; index < rows * rows; ++index)
        correct_values &= std::fabs(output[index] - expected) <= 1.0e-6F;
    check(correct_values, "real arena preserves all expected softmax probabilities");
}

void check_shared_activation(const trtmc::ITrtModule& first, const trtmc::ITrtModule& second) {
    const auto* first_impl = dynamic_cast<const trtmc::TrtModuleImpl*>(&first);
    const auto* second_impl = dynamic_cast<const trtmc::TrtModuleImpl*>(&second);
    if (!first_impl || !second_impl)
        throw std::runtime_error("unexpected RTX module implementation");
    check(trtmc::TrtModuleImplTestPeer::shares_activation_arena(*first_impl, *second_impl),
          "public RTX backend modules share a real arena on the explicit stream");
    check(trtmc::TrtModuleImplTestPeer::activation_profile_bytes(*first_impl) > 0,
          "real-arena test plan requires nonzero activation memory");
}

void test_real_rtx_activation_arena(cudaStream_t stream) {
    try {
        std::unique_ptr<trtmc::IBackend, decltype(&trtmc_destroy_backend)> backend(
            trtmc_create_backend(), &trtmc_destroy_backend);
        if (!backend)
            throw std::runtime_error("cannot create the selected backend");
        if (std::string(backend->name()) != "trt_rtx") {
            std::cout << "SKIP: real activation arena requires the RTX backend\n";
            return;
        }
        auto plan = build_dynamic_softmax_plan();
        if (!plan)
            throw std::runtime_error("cannot build the real-arena test plan");
        TemporaryArenaPlan temporary;
        {
            std::ofstream file(temporary.path, std::ios::binary);
            file.write(static_cast<const char*>(plan->data()),
                       static_cast<std::streamsize>(plan->size()));
            file.close();
            if (!file)
                throw std::runtime_error("cannot write the real-arena test plan");
        }
        trtmc::ModuleCreateOptions options;
        options.stream = stream;
        options.cuda_graphs = false;
        const auto path = temporary.path.u8string();
        auto first = backend->create_module_from_file(path.c_str(), 0, plan->size(), options, {},
                                                      -1, false, true);
        auto second = backend->create_module_from_file(path.c_str(), 0, plan->size(), options, {},
                                                       -1, false, true);
        if (!first || !second || !first->ok() || !second->ok())
            throw std::runtime_error("cannot create the real-arena serial modules");
        check_shared_activation(*first, *second);
        // Exercise another context raising the shared arena high-water mark
        // before the first context itself has ever used the larger shape.
        check_softmax_matrix(*first, 8);
        check_softmax_matrix(*second, 512);
        check_softmax_matrix(*first, 8);
        // The first context also transitions A -> B -> A without recreation.
        check_softmax_matrix(*first, 512);
        check_softmax_matrix(*first, 8);
        check(cudaStreamSynchronize(stream) == cudaSuccess,
              "real-arena shape/context transitions finish without a CUDA error");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: real RTX activation arena: " << error.what() << '\n';
        ++failures;
    }
}

} // namespace

int main() {
    test_fp8_tensor_storage_and_mapping();
    if (failures)
        return failures;
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        return 77;
    auto engine = build_identity(true);
    auto static_engine = build_identity(false);
    if (!engine || !static_engine)
        return 1;
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess)
        return 1;
    test_constructor_rejects_vectorized_fp8(stream);
    test_first_forward_allocates_unbound_dynamic_input(*engine, stream);
    test_external_dynamic_input_stays_external(*engine, stream);
    test_backend_managed_graph_does_not_enable_nested_capture(*engine, stream);
    test_rejected_shape_fails_without_poisoning_the_next_request(*engine, stream);
    test_short_static_upload_does_not_keep_the_previous_input(*static_engine, stream);
    test_short_external_static_upload_preserves_its_tail(*static_engine, stream);
    test_serial_dynamic_output_grows_and_reuses(*engine, stream);
    test_serial_device_forward_defers_host_staging(*engine, stream);
    test_serial_input_rejects_invalid_requests_before_allocation(*engine, stream);
    test_serial_external_input_stays_external(*engine, stream);
    test_serial_external_output_stays_external(*engine, stream);
    test_real_rtx_activation_arena(stream);
    cudaStreamDestroy(stream);
    return failures;
}
