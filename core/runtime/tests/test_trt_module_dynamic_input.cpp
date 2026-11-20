/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/primitives/trt_common.h"
#include "runtime/tensorrt/trt_logger.h"
#include "runtime/tensorrt/trt_module_impl.h"

#include <NvInfer.h>
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
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
    cudaStreamDestroy(stream);
    return failures;
}
