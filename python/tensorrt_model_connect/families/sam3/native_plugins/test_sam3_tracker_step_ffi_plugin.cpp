/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sam3_tracker_step_ffi_plugin.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <initializer_list>
#include <iostream>
#include <tvm/ffi/c_api.h>
#include <vector>

namespace {

using Plugin = trtmc::sam3::TrackerStepFfiPlugin;

const std::array<std::vector<int64_t>, Plugin::kInputCount + Plugin::kOutputCount> kExpectedShapes{
    std::vector<int64_t>{1, 32, 288, 288},
    std::vector<int64_t>{1, 64, 144, 144},
    std::vector<int64_t>{1, 256, 72, 72},
    std::vector<int64_t>{1, 256, 72, 72},
    std::vector<int64_t>{2, 7, 72 * 72, 64},
    std::vector<int64_t>{2, 7, 72 * 72, 64},
    std::vector<int64_t>{2, 7},
    std::vector<int64_t>{2, 16, 256},
    std::vector<int64_t>{2, 16},
    std::vector<int64_t>{1},
    std::vector<int64_t>{2, Plugin::kPackedWidth},
};

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

int shape_callback(void*, const TVMFFIAny* arguments, int32_t argument_count, TVMFFIAny* result) {
    if (arguments == nullptr || result == nullptr ||
        argument_count != Plugin::kInputCount + Plugin::kOutputCount + 1)
        return -1;
    for (std::size_t index = 0; index < kExpectedShapes.size(); ++index) {
        if (arguments[index].type_index != kTVMFFIDLTensorPtr)
            return -2;
        const auto* tensor = static_cast<const DLTensor*>(arguments[index].v_ptr);
        if (tensor == nullptr || tensor->shape == nullptr ||
            tensor->ndim != static_cast<int32_t>(kExpectedShapes[index].size()))
            return -3;
        for (int32_t dimension = 0; dimension < tensor->ndim; ++dimension) {
            if (tensor->shape[dimension] !=
                kExpectedShapes[index][static_cast<std::size_t>(dimension)])
                return -4;
        }
    }
    if (arguments[kExpectedShapes.size()].type_index != kTVMFFIOpaquePtr)
        return -5;
    result->type_index = kTVMFFINone;
    result->v_int64 = 0;
    return 0;
}

int error_callback(void*, const TVMFFIAny*, int32_t, TVMFFIAny*) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "intentional SAM3 plugin test error");
    return -1;
}

nvinfer1::Dims make_dims(std::initializer_list<int64_t> dimensions) {
    nvinfer1::Dims result{};
    result.nbDims = static_cast<int32_t>(dimensions.size());
    std::size_t index = 0;
    for (const int64_t dimension : dimensions)
        result.d[index++] = dimension;
    return result;
}

nvinfer1::PluginTensorDesc make_descriptor(nvinfer1::DataType type,
                                           std::initializer_list<int64_t> dimensions) {
    nvinfer1::PluginTensorDesc descriptor{};
    descriptor.dims = make_dims(dimensions);
    descriptor.type = type;
    descriptor.format = nvinfer1::TensorFormat::kLINEAR;
    return descriptor;
}

std::array<nvinfer1::PluginTensorDesc, Plugin::kInputCount> make_actual_inputs() {
    return {
        make_descriptor(nvinfer1::DataType::kFLOAT, {1, 32, 288, 288}),
        make_descriptor(nvinfer1::DataType::kFLOAT, {1, 64, 144, 144}),
        make_descriptor(nvinfer1::DataType::kFLOAT, {1, 256, 72, 72}),
        make_descriptor(nvinfer1::DataType::kFLOAT, {1, 256, 72, 72}),
        make_descriptor(nvinfer1::DataType::kFLOAT, {2, 7, 72 * 72, 64}),
        make_descriptor(nvinfer1::DataType::kFLOAT, {2, 7, 72 * 72, 64}),
        make_descriptor(nvinfer1::DataType::kINT32, {2, 7}),
        make_descriptor(nvinfer1::DataType::kFLOAT, {2, 16, 256}),
        make_descriptor(nvinfer1::DataType::kINT32, {2, 16}),
        make_descriptor(nvinfer1::DataType::kINT32, {1}),
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
    for (const std::size_t index : {4U, 5U, 6U}) {
        dynamic[index].desc.dims.d[1] = -1;
        dynamic[index].min.d[1] = 1;
        dynamic[index].max.d[1] = 10;
    }
    for (const std::size_t index : {7U, 8U}) {
        dynamic[index].desc.dims.d[1] = -1;
        dynamic[index].min.d[1] = 1;
        dynamic[index].max.d[1] = 19;
    }
    return dynamic;
}

void register_callback(const char* global_name, TVMFFISafeCallType callback) {
    TVMFFIObjectHandle function = nullptr;
    check(TVMFFIFunctionCreate(nullptr, callback, nullptr, &function) == 0 && function != nullptr,
          "create TVM-FFI callback");
    const TVMFFIByteArray name{global_name, std::char_traits<char>::length(global_name)};
    const int status = TVMFFIFunctionSetGlobal(&name, function, 0);
    TVMFFIObjectDecRef(function);
    check(status == 0, "register TVM-FFI shape callback");
}

} // namespace

int main() {
    constexpr const char* kGlobalName =
        "trtmc.sam3.tracker_step.b2.split_aoti.0123456789abcdef0123";
    constexpr const char* kErrorGlobalName =
        "trtmc.sam3.tracker_step.b2.split_aoti.abcdef0123456789abcd";
    check(cudaSetDevice(0) == cudaSuccess, "select CUDA device");
    register_callback(kGlobalName, shape_callback);
    register_callback(kErrorGlobalName, error_callback);

    auto actual_inputs = make_actual_inputs();
    const auto actual_output =
        make_descriptor(nvinfer1::DataType::kFLOAT, {2, Plugin::kPackedWidth});
    auto dynamic_inputs = make_dynamic_inputs(actual_inputs);
    nvinfer1::DynamicPluginTensorDesc dynamic_output{};
    dynamic_output.desc = actual_output;
    // TensorRT does not define output min/max profile dimensions here. The
    // plugin must validate the exact declared output and only input profiles.

    Plugin plugin(std::string(kGlobalName), int32_t{2});
    plugin.configurePlugin(dynamic_inputs.data(), Plugin::kInputCount, &dynamic_output,
                           Plugin::kOutputCount);
    check(plugin.initialize() == 0, "initialize configured tracker-step plugin");

    void* storage = nullptr;
    check(cudaMalloc(&storage, 1) == cudaSuccess, "allocate callback test storage");
    std::array<const void*, Plugin::kInputCount> inputs{};
    inputs.fill(storage);
    std::array<void*, Plugin::kOutputCount> outputs{storage};
    check(plugin.enqueue(actual_inputs.data(), &actual_output, inputs.data(), outputs.data(),
                         nullptr, nullptr) == 0,
          "callback receives exact widened dimensions");
    auto* context_clone = plugin.clone();
    check(context_clone != nullptr, "clone resolves its own TVM-FFI callback reference");
    check(context_clone->enqueue(actual_inputs.data(), &actual_output, inputs.data(),
                                 outputs.data(), nullptr, nullptr) == 0,
          "TensorRT context clone can enqueue without a second initialize call");
    context_clone->destroy();
    plugin.terminate();

    Plugin error_plugin(std::string(kErrorGlobalName), int32_t{2});
    error_plugin.configurePlugin(dynamic_inputs.data(), Plugin::kInputCount, &dynamic_output,
                                 Plugin::kOutputCount);
    check(error_plugin.initialize() == 0, "initialize error-reporting tracker-step plugin");
    check(error_plugin.enqueue(actual_inputs.data(), &actual_output, inputs.data(), outputs.data(),
                               nullptr, nullptr) != 0,
          "failing callback propagates an enqueue error");
    TVMFFIObjectHandle stale_error = nullptr;
    TVMFFIErrorMoveFromRaised(&stale_error);
    check(stale_error == nullptr, "plugin consumes the callback's structured TVM-FFI error");
    error_plugin.terminate();

    check(cudaFree(storage) == cudaSuccess, "free callback test storage");
    std::cout << "PASS: SAM3 tracker-step DLTensor shape bridge\n";
    return 0;
}
