/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../byok/tvm_ffi_kernel_plugin.h"

#include <NvInfer.h>
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

class TestLogger : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kERROR)
            std::cerr << "[TRT] " << message << '\n';
    }
};

// TensorRT supplies the real expression builder during network shape analysis.
// Its public SDK does not support constructing one in the CPU parser test.
// This probe neither builds an engine nor enqueues the placeholder kernel.
bool saw_symbolic_dimensions = false;

void verify_output_expressions(const nvinfer1::DimsExprs& output,
                               const nvinfer1::DimsExprs* inputs) {
    if (output.nbDims != 3)
        return;
    check(output.d[0] == inputs[0].d[0], "first output preserves input0 axis0 expression");
    check(output.d[1] != nullptr && output.d[1]->isConstant() &&
              output.d[1]->getConstantValue() == 64,
          "middle output dimension is constant 64");
    check(output.d[2] == inputs[1].d[2], "last output preserves input1 axis2 expression");
    if (output.d[0] != nullptr && output.d[2] != nullptr)
        saw_symbolic_dimensions =
            saw_symbolic_dimensions || (!output.d[0]->isConstant() && !output.d[2]->isConstant());
}

class ShapeResolutionProbe final : public trtmc::TvmFfiKernelPlugin {
  public:
    ShapeResolutionProbe()
        : TvmFfiKernelPlugin(
              "example.identity_copy",
              R"({"num_inputs":2,"num_outputs":1,"outputs":[{"dims":[{"input":0,"axis":0},64,{"input":1,"axis":2}],"dtype":"float32"}],"workspace_bytes":0})") {
    }

    ShapeResolutionProbe* clone() const noexcept override { return new ShapeResolutionProbe(); }

    nvinfer1::DimsExprs getOutputDimensions(int32_t index, const nvinfer1::DimsExprs* inputs,
                                            int32_t count,
                                            nvinfer1::IExprBuilder& builder) noexcept override {
        const auto output = TvmFfiKernelPlugin::getOutputDimensions(index, inputs, count, builder);
        check(count == 2 && inputs[0].nbDims == 2 && inputs[1].nbDims == 3,
              "shape probe receives both input ranks");
        if (count != 2 || inputs[0].nbDims != 2 || inputs[1].nbDims != 3)
            return output;
        check(output.nbDims == 3, "mixed shape resolves to rank three");
        verify_output_expressions(output, inputs);
        nvinfer1::DimsExprs invalid_inputs[] = {inputs[0], inputs[1]};
        invalid_inputs[1].nbDims = 2;
        check(
            TvmFfiKernelPlugin::getOutputDimensions(index, invalid_inputs, count, builder).nbDims ==
                -1,
            "axis outside the selected input rank is rejected");
        return output;
    }
};

void test_symbolic_output_dimensions() {
    TestLogger logger;
    auto* builder = nvinfer1::createInferBuilder(logger);
    check(builder != nullptr, "GPU dimension test requires a TensorRT builder");
    if (builder == nullptr)
        return;
    auto* network = builder->createNetworkV2(0);
    nvinfer1::ITensor* inputs[] = {
        network->addInput("left", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{2, {-1, 3}}),
        network->addInput("right", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{3, {5, 7, -1}})};
    ShapeResolutionProbe plugin;
    auto* layer = network->addPluginV2(inputs, 2, plugin);
    check(layer != nullptr, "mixed-dimension plugin layer created");
    if (layer != nullptr) {
        const auto shape = layer->getOutput(0)->getDimensions();
        check(shape.nbDims == 3 && shape.d[0] == -1 && shape.d[1] == 64 && shape.d[2] == -1,
              "mixed output retains both independent dynamic dimensions");
    }
    check(saw_symbolic_dimensions, "dimension checks use real nonconstant TensorRT expressions");
    delete network;
    delete builder;
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

int main(int argc, char** argv) {
    tvm_ffi_plugin_force_link();
    if (argc > 1) {
        if (argc != 2 || std::string(argv[1]) != "--gpu-dimensions")
            return 2;
        test_symbolic_output_dimensions();
    }
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

    const std::string mixed =
        R"({"num_inputs":2,"num_outputs":1,"outputs":[{"dims":[{"input":0,"axis":0},64,{"input":1,"axis":2}],"dtype":"bfloat16"}],"workspace_bytes":0})";
    plugin = create(mixed);
    check(plugin != nullptr, "mixed fixed and input-axis dimensions accepted");
    if (plugin != nullptr) {
        auto* clone = plugin->clone();
        check(clone != nullptr, "input-axis shape specification survives cloning");
        if (clone != nullptr)
            clone->destroy();
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

    for (const auto* dimension :
         {R"({"input":1,"axis":0})", R"({"input":-1,"axis":0})", R"({"input":0,"axis":-1})",
          R"({"input":0,"axis":8})", R"({"input":0,"axis":0,"extra":0})", R"({"input":0})",
          R"({"input":0.0,"axis":0})", "2147483648", "0", "-1"}) {
        require_rejected(std::string(R"({"num_inputs":1,"num_outputs":1,"outputs":[{"dims":[)") +
                             dimension + R"(],"dtype":"float32"}],"workspace_bytes":0})",
                         "invalid fixed dimension or input-axis reference rejected");
    }

    return failures == 0 ? 0 : 1;
}
