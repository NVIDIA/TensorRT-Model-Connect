/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <NvInferRuntime.h>
#include <iostream>
#include <string>

extern "C" void tvm_ffi_plugin_force_link();

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

nvinfer1::IPluginV2Ext* create(const std::string& spec) {
    auto* creator = static_cast<nvinfer1::IPluginCreator*>(
        getPluginRegistry()->getCreator("TvmFfiKernel", "1", ""));
    if (creator == nullptr)
        return nullptr;
    const std::string kernel = "test.kernel";
    nvinfer1::PluginField fields[] = {
        {"kernel_name", kernel.data(), nvinfer1::PluginFieldType::kCHAR,
         static_cast<int32_t>(kernel.size())},
        {"shape_spec", spec.data(), nvinfer1::PluginFieldType::kCHAR,
         static_cast<int32_t>(spec.size())},
    };
    nvinfer1::PluginFieldCollection collection{2, fields};
    auto* base = creator->createPlugin("shape_spec_test", &collection);
    auto* plugin = dynamic_cast<nvinfer1::IPluginV2Ext*>(base);
    if (base != nullptr && plugin == nullptr)
        base->destroy();
    return plugin;
}

void require_rejected(const std::string& spec, const char* name) {
    auto* plugin = create(spec);
    check(plugin == nullptr, name);
    if (plugin != nullptr)
        plugin->destroy();
}

} // namespace

int main() {
    tvm_ffi_plugin_force_link();
    const std::string valid =
        R"({"num_inputs":1,"num_outputs":1,"outputs":[{"dims":"same_as_input_0","dtype":"float32"}],"workspace_bytes":0})";
    auto* plugin = create(valid);
    check(plugin != nullptr, "valid shape spec accepted");
    if (plugin != nullptr) {
        const nvinfer1::DataType half_input[] = {nvinfer1::DataType::kHALF};
        check(plugin->getOutputDataType(0, half_input, 1) == nvinfer1::DataType::kFLOAT,
              "explicit float32 output does not inherit the input dtype");
        plugin->destroy();
    }

    require_rejected("{", "invalid JSON rejected");
    require_rejected(
        R"({"num_inputs":1,"num_outputs":1,"outputs":[{"dims":[1],"dtype":"unknown"}],"workspace_bytes":0})",
        "unknown dtype rejected");
    require_rejected(
        R"({"num_inputs":1,"num_outputs":1,"outputs":[{"dims":[1,1,1,1,1,1,1,1,1],"dtype":"float32"}],"workspace_bytes":0})",
        "rank above eight rejected");
    require_rejected(
        R"({"num_inputs":1,"num_outputs":2,"outputs":[{"dims":[1],"dtype":"float32"}],"workspace_bytes":0})",
        "output count mismatch rejected");
    require_rejected(
        R"({"num_inputs":1,"num_outputs":1,"outputs":[{"dims":[1],"dtype":"float32"}],"workspace_bytes":0,"unknown":1})",
        "unknown root field rejected");

    return failures == 0 ? 0 : 1;
}
