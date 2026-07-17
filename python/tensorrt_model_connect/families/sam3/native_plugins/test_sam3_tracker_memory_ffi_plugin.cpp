/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sam3_tracker_memory_ffi_plugin.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <initializer_list>
#include <iostream>
#include <string>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <vector>

namespace {

using Plugin = trtmc::sam3::TrackerMemoryFfiPlugin;

struct ShapeContext {
    int32_t batch_size;
};

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::vector<int64_t> output_shape(int32_t batch_size) {
    if (batch_size == 1)
        return {2, Plugin::kSpatialTokens, 1, Plugin::kMemoryChannels};
    return {2, 2, Plugin::kSpatialTokens, Plugin::kMemoryChannels};
}

std::array<std::vector<int64_t>, Plugin::kInputCount + Plugin::kOutputCount>
expected_shapes(int32_t batch_size) {
    return {
        std::vector<int64_t>{1, 256, 72, 72},
        std::vector<int64_t>{batch_size, 1, 288, 288},
        std::vector<int64_t>{batch_size, 1},
        std::vector<int64_t>{batch_size, 1},
        output_shape(batch_size),
    };
}

int shape_callback(void* self, const TVMFFIAny* arguments, int32_t argument_count,
                   TVMFFIAny* result) {
    if (self == nullptr || arguments == nullptr || result == nullptr ||
        argument_count != Plugin::kInputCount + Plugin::kOutputCount + 1)
        return -1;
    const auto batch_size = static_cast<const ShapeContext*>(self)->batch_size;
    const auto shapes = expected_shapes(batch_size);
    for (std::size_t index = 0; index < shapes.size(); ++index) {
        if (arguments[index].type_index != kTVMFFIDLTensorPtr)
            return -2;
        const auto* tensor = static_cast<const DLTensor*>(arguments[index].v_ptr);
        if (tensor == nullptr || tensor->shape == nullptr ||
            tensor->ndim != static_cast<int32_t>(shapes[index].size()))
            return -3;
        for (int32_t dimension = 0; dimension < tensor->ndim; ++dimension) {
            if (tensor->shape[dimension] != shapes[index][static_cast<std::size_t>(dimension)])
                return -4;
        }
        const bool integer = index == 3;
        const auto expected_code = integer ? kDLInt : kDLFloat;
        if (tensor->dtype.code != expected_code || tensor->dtype.bits != 32 ||
            tensor->dtype.lanes != 1)
            return -5;
    }
    const auto stream_index = shapes.size();
    if (arguments[stream_index].type_index != kTVMFFIOpaquePtr)
        return -6;
    const auto stream = reinterpret_cast<cudaStream_t>(arguments[stream_index].v_ptr);
    if (TVMFFIEnvGetStream(kDLCUDA, 0) != reinterpret_cast<TVMFFIStreamHandle>(stream))
        return -7;
    result->type_index = kTVMFFINone;
    result->v_int64 = 0;
    return 0;
}

int error_callback(void*, const TVMFFIAny*, int32_t, TVMFFIAny*) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "intentional SAM3 memory plugin test error");
    return -1;
}

nvinfer1::Dims make_dims(const std::vector<int64_t>& dimensions) {
    nvinfer1::Dims result{};
    result.nbDims = static_cast<int32_t>(dimensions.size());
    for (std::size_t index = 0; index < dimensions.size(); ++index)
        result.d[index] = dimensions[index];
    return result;
}

nvinfer1::PluginTensorDesc make_descriptor(nvinfer1::DataType type,
                                           const std::vector<int64_t>& dimensions) {
    nvinfer1::PluginTensorDesc descriptor{};
    descriptor.dims = make_dims(dimensions);
    descriptor.type = type;
    descriptor.format = nvinfer1::TensorFormat::kLINEAR;
    return descriptor;
}

std::array<nvinfer1::PluginTensorDesc, Plugin::kInputCount> make_actual_inputs(int32_t batch_size) {
    return {
        make_descriptor(nvinfer1::DataType::kFLOAT, {1, 256, 72, 72}),
        make_descriptor(nvinfer1::DataType::kFLOAT, {batch_size, 1, 288, 288}),
        make_descriptor(nvinfer1::DataType::kFLOAT, {batch_size, 1}),
        make_descriptor(nvinfer1::DataType::kINT32, {batch_size, 1}),
    };
}

std::array<nvinfer1::DynamicPluginTensorDesc, Plugin::kInputCount>
make_dynamic_inputs(const std::array<nvinfer1::PluginTensorDesc, Plugin::kInputCount>& actual) {
    std::array<nvinfer1::DynamicPluginTensorDesc, Plugin::kInputCount> dynamic{};
    for (std::size_t index = 0; index < dynamic.size(); ++index) {
        dynamic[index].desc = actual[index];
        dynamic[index].min = actual[index].dims;
        dynamic[index].max = actual[index].dims;
    }
    return dynamic;
}

void register_callback(const char* global_name, ShapeContext* context,
                       TVMFFISafeCallType callback) {
    TVMFFIObjectHandle function = nullptr;
    check(TVMFFIFunctionCreate(context, callback, nullptr, &function) == 0 && function != nullptr,
          "create TVM-FFI callback");
    const TVMFFIByteArray name{global_name, std::char_traits<char>::length(global_name)};
    const int status = TVMFFIFunctionSetGlobal(&name, function, 0);
    TVMFFIObjectDecRef(function);
    check(status == 0, "register TVM-FFI memory callback");
}

void exercise_plugin(const char* global_name, int32_t batch_size, cudaStream_t stream,
                     void* storage) {
    auto actual_inputs = make_actual_inputs(batch_size);
    const auto actual_output =
        make_descriptor(nvinfer1::DataType::kFLOAT, output_shape(batch_size));
    auto dynamic_inputs = make_dynamic_inputs(actual_inputs);
    nvinfer1::DynamicPluginTensorDesc dynamic_output{};
    dynamic_output.desc = actual_output;

    Plugin plugin(std::string(global_name), batch_size);
    plugin.configurePlugin(dynamic_inputs.data(), Plugin::kInputCount, &dynamic_output,
                           Plugin::kOutputCount);
    check(plugin.initialize() == 0, "initialize configured tracker-memory plugin");
    std::array<const void*, Plugin::kInputCount> inputs{};
    inputs.fill(storage);
    std::array<void*, Plugin::kOutputCount> outputs{storage};
    check(plugin.enqueue(actual_inputs.data(), &actual_output, inputs.data(), outputs.data(),
                         nullptr, stream) == 0,
          "memory callback receives exact fixed dimensions and dtype");
    auto* context_clone = plugin.clone();
    check(context_clone != nullptr, "memory clone resolves its own TVM-FFI callback reference");
    check(context_clone->enqueue(actual_inputs.data(), &actual_output, inputs.data(),
                                 outputs.data(), nullptr, stream) == 0,
          "memory context clone enqueues without a second initialize call");
    context_clone->destroy();

    auto invalid_inputs = actual_inputs;
    invalid_inputs[3].type = nvinfer1::DataType::kFLOAT;
    check(plugin.enqueue(invalid_inputs.data(), &actual_output, inputs.data(), outputs.data(),
                         nullptr, stream) != 0,
          "memory plugin rejects a non-INT32 suppression tensor");
    plugin.terminate();
}

} // namespace

int main() {
    constexpr const char* kSoftB1 = "trtmc.sam3.tracker_memory.soft.b1.fixed.0123456789abcdef0123";
    constexpr const char* kHardB2 = "trtmc.sam3.tracker_memory.hard.b2.fixed.123456789abcdef01234";
    constexpr const char* kErrorB2 = "trtmc.sam3.tracker_memory.soft.b2.fixed.abcdef0123456789abcd";
    check(cudaSetDevice(0) == cudaSuccess, "select CUDA device");
    ShapeContext b1{1};
    ShapeContext b2{2};
    register_callback(kSoftB1, &b1, shape_callback);
    register_callback(kHardB2, &b2, shape_callback);
    register_callback(kErrorB2, nullptr, error_callback);

    cudaStream_t previous_stream = nullptr;
    cudaStream_t execution_stream = nullptr;
    check(cudaStreamCreate(&previous_stream) == cudaSuccess, "create prior TVM-FFI stream");
    check(cudaStreamCreate(&execution_stream) == cudaSuccess, "create TensorRT execution stream");
    TVMFFIStreamHandle original_stream = nullptr;
    check(TVMFFIEnvSetStream(kDLCUDA, 0, reinterpret_cast<TVMFFIStreamHandle>(previous_stream),
                             &original_stream) == 0,
          "install prior TVM-FFI stream");

    void* storage = nullptr;
    check(cudaMalloc(&storage, 1) == cudaSuccess, "allocate memory callback test storage");
    exercise_plugin(kSoftB1, 1, execution_stream, storage);
    check(TVMFFIEnvGetStream(kDLCUDA, 0) == reinterpret_cast<TVMFFIStreamHandle>(previous_stream),
          "B1 plugin restores the prior TVM-FFI stream");
    exercise_plugin(kHardB2, 2, execution_stream, storage);
    check(TVMFFIEnvGetStream(kDLCUDA, 0) == reinterpret_cast<TVMFFIStreamHandle>(previous_stream),
          "B2 plugin restores the prior TVM-FFI stream");

    auto error_inputs = make_actual_inputs(2);
    const auto error_output = make_descriptor(nvinfer1::DataType::kFLOAT, output_shape(2));
    auto dynamic_inputs = make_dynamic_inputs(error_inputs);
    nvinfer1::DynamicPluginTensorDesc dynamic_output{};
    dynamic_output.desc = error_output;
    Plugin error_plugin(std::string(kErrorB2), int32_t{2});
    error_plugin.configurePlugin(dynamic_inputs.data(), Plugin::kInputCount, &dynamic_output,
                                 Plugin::kOutputCount);
    check(error_plugin.initialize() == 0, "initialize error-reporting memory plugin");
    std::array<const void*, Plugin::kInputCount> input_storage{};
    input_storage.fill(storage);
    std::array<void*, Plugin::kOutputCount> output_storage{storage};
    check(error_plugin.enqueue(error_inputs.data(), &error_output, input_storage.data(),
                               output_storage.data(), nullptr, execution_stream) != 0,
          "failing memory callback propagates an enqueue error");
    TVMFFIObjectHandle stale_error = nullptr;
    TVMFFIErrorMoveFromRaised(&stale_error);
    check(stale_error == nullptr, "memory plugin consumes structured TVM-FFI errors");
    error_plugin.terminate();

    check(cudaFree(storage) == cudaSuccess, "free memory callback storage");
    check(TVMFFIEnvSetStream(kDLCUDA, 0, original_stream, nullptr) == 0,
          "restore original TVM-FFI stream");
    check(cudaStreamDestroy(execution_stream) == cudaSuccess, "destroy execution stream");
    check(cudaStreamDestroy(previous_stream) == cudaSuccess, "destroy prior stream");
    std::cout << "PASS: SAM3 tracker-memory DLTensor bridge\n";
    return 0;
}
