/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "plugins/tvm_ffi_kernel_plugin.h"

#include <NvInfer.h>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

/// @brief Record a named assertion failure.
void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

/// @brief Verify that an operation rejects an invalid shape specification.
template <typename Fn>
void check_throws(Fn&& fn, const char* name) {
    try {
        fn();
        check(false, name);
    } catch (const std::exception&) {
        check(true, name);
    }
}

/// @brief Cover nested JSON, output metadata, and serialization.
void test_shape_spec_parsing() {
    const std::string kernel_name = "test.shape_spec";
    const std::string shape_spec = R"({
        "num_inputs": 2,
        "num_outputs": 2,
        "outputs": [
            {
                "dims": "same_as_input_1",
                "dtype": "inherit",
                "metadata": {"nested": [1, {"text": "escaped \" ] }"}]}
            },
            {"dims": [2, 3], "dtype": "int32", "unused": [[], {}]}
        ],
        "workspace_bytes": 4096,
        "extra_args": [
            {"type": "int", "value": 7, "metadata": {"nested": [1, 2]}},
            {"type": "float", "value": 1.25, "label": "escaped \" ] }"}
        ],
        "extra": {"nested": {"array": [1, 2, 3]}}
    })";

    trtmc::TvmFfiKernelPlugin plugin(kernel_name, shape_spec);
    check(plugin.getNbOutputs() == 2, "shape spec output count");
    check(plugin.getWorkspaceSize(nullptr, 0, nullptr, 0) == 4096, "shape spec workspace size");

    const nvinfer1::DataType input_types[] = {nvinfer1::DataType::kFLOAT,
                                              nvinfer1::DataType::kHALF};
    check(plugin.getOutputDataType(0, input_types, 2) == nvinfer1::DataType::kHALF,
          "shape spec inherited output type");
    check(plugin.getOutputDataType(1, input_types, 2) == nvinfer1::DataType::kINT32,
          "shape spec explicit output type");

    std::vector<char> serialized(plugin.getSerializationSize());
    plugin.serialize(serialized.data());
    check(serialized.size() == sizeof(uint32_t) * 2 + kernel_name.size() + shape_spec.size(),
          "shape spec serialization size");

    const char* cursor = serialized.data();
    uint32_t serialized_kernel_length = 0;
    std::memcpy(&serialized_kernel_length, cursor, sizeof(serialized_kernel_length));
    cursor += sizeof(serialized_kernel_length);
    check(serialized_kernel_length == kernel_name.size(), "serialized kernel name length");
    check(std::string(cursor, serialized_kernel_length) == kernel_name, "serialized kernel name");
    cursor += serialized_kernel_length;
    uint32_t serialized_spec_length = 0;
    std::memcpy(&serialized_spec_length, cursor, sizeof(serialized_spec_length));
    cursor += sizeof(serialized_spec_length);
    check(serialized_spec_length == shape_spec.size(), "serialized shape spec length");
    check(std::string(cursor, serialized_spec_length) == shape_spec, "serialized shape spec");

    trtmc::TvmFfiKernelPlugin restored(serialized.data(), serialized.size());
    check(restored.getNbOutputs() == 2, "deserialized shape spec output count");
    check(restored.getWorkspaceSize(nullptr, 0, nullptr, 0) == 4096,
          "deserialized shape spec workspace size");
    check(restored.getSerializationSize() == serialized.size(),
          "shape spec serialization ABI preserved");
}

/// @brief Verify defaults for omitted and syntactically incomplete JSON.
void test_shape_spec_defaults_and_malformed_json() {
    const nvinfer1::DataType input_types[] = {nvinfer1::DataType::kHALF};
    for (const std::string& shape_spec : {
             std::string{"{}"},
             std::string{R"({"num_inputs": 2, "outputs": [)"},
             std::string{R"({"num_inputs": 2, "outputs": [{"dims": [2, 3]})"},
         }) {
        trtmc::TvmFfiKernelPlugin plugin("test.defaults", shape_spec);
        check(plugin.getNbOutputs() == 1, "malformed shape spec default output count");
        check(plugin.getWorkspaceSize(nullptr, 0, nullptr, 0) == 0,
              "malformed shape spec default workspace");
        check(plugin.getOutputDataType(0, input_types, 1) == nvinfer1::DataType::kHALF,
              "malformed shape spec default output type");
    }
}

/// @brief Verify invalid counts, workspace, dimensions, and input references.
void test_shape_spec_validation() {
    check_throws([] { trtmc::TvmFfiKernelPlugin("test.invalid", R"({"num_inputs":0})"); },
                 "reject zero inputs");
    check_throws([] { trtmc::TvmFfiKernelPlugin("test.invalid", R"({"num_outputs":0})"); },
                 "reject zero outputs");
    check_throws([] { trtmc::TvmFfiKernelPlugin("test.invalid", R"({"workspace_bytes":-1})"); },
                 "reject negative workspace");
    check_throws(
        [] {
            trtmc::TvmFfiKernelPlugin("test.invalid",
                                      R"({"num_inputs":1,"outputs":[{"dims":"same_as_input_1"}]})");
        },
        "reject out-of-range input index");
    check_throws(
        [] {
            trtmc::TvmFfiKernelPlugin(
                "test.invalid", R"({"outputs":[{"dims":"same_as_input_999999999999999999999"}]})");
        },
        "reject overflowing input index");
    check_throws(
        [] {
            trtmc::TvmFfiKernelPlugin("test.invalid",
                                      R"({"outputs":[{"dims":"same_as_input_-1"}]})");
        },
        "reject negative input index");
    check_throws(
        [] {
            trtmc::TvmFfiKernelPlugin("test.invalid",
                                      R"({"outputs":[{"dims":"same_as_input_0suffix"}]})");
        },
        "reject malformed input index");
    check_throws(
        [] { trtmc::TvmFfiKernelPlugin("test.invalid", R"({"outputs":[{"dims":[2,0,3]}]})"); },
        "reject non-positive fixed dimension");
}

} // namespace

/// @brief Run the TVM-FFI shape-specification unit tests.
int main() {
    test_shape_spec_parsing();
    test_shape_spec_defaults_and_malformed_json();
    test_shape_spec_validation();

    if (failures != 0) {
        std::cerr << failures << " FAILED\n";
        return 1;
    }
    std::cerr << "All TVM-FFI shape specification tests passed.\n";
    return 0;
}
