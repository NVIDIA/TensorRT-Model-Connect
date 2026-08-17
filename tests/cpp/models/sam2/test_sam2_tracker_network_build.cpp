/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tools/sam2_native_builder/sam2_tracker_network.h"
#include "tools/sam2_native_builder/sam2_trt_layers.h"

#include <NvInfer.h>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

class Logger final : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kERROR)
            std::cerr << "TensorRT: " << (message == nullptr ? "" : message) << '\n';
    }
};

template <typename T>
struct TrtDelete {
    void operator()(T* object) const noexcept {
        if (object != nullptr)
            delete object;
    }
};

template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDelete<T>>;

bool sameContract(const nvinfer1::ITensor& tensor, const trtmc::sam2::TensorContract& contract) {
    const nvinfer1::DataType expected_type =
        contract.data_type == trtmc::sam2::TensorDataType::kFloat32 ? nvinfer1::DataType::kFLOAT
                                                                    : nvinfer1::DataType::kBF16;
    const nvinfer1::Dims dimensions = tensor.getDimensions();
    if (tensor.getType() != expected_type || dimensions.nbDims != contract.rank)
        return false;
    for (std::int32_t index = 0; index < dimensions.nbDims; ++index) {
        if (dimensions.d[index] != contract.dimensions[static_cast<std::size_t>(index)])
            return false;
    }
    return tensor.getName() != nullptr && contract.name == tensor.getName();
}

void testPlan(nvinfer1::IBuilder& builder, const trtmc::sam2::native::CheckpointReader& checkpoint,
              std::int32_t history_frames, bool serialize) {
    using namespace trtmc::sam2::native;

    TrtPtr<nvinfer1::INetworkDefinition> network(
        builder.createNetworkV2(sam2NetworkCreationFlags()));
    check(network != nullptr, "strongly typed tracker network creation");
    if (!network)
        return;

    const TrackerPlanSpec plan =
        history_frames == 0 ? promptTrackerPlanSpec() : recurrentTrackerPlanSpec(history_frames);
    Sam2TrackerNetworkBuilder tracker_builder(*network, checkpoint);
    const Sam2TrackerNetworkOutputs outputs = history_frames == 0
                                                  ? tracker_builder.buildPrompt()
                                                  : tracker_builder.buildRecurrent(history_frames);

    check(network->getNbInputs() == static_cast<std::int32_t>(plan.inputs.size()),
          "tracker graph input count matches its static plan");
    check(network->getNbOutputs() == static_cast<std::int32_t>(plan.outputs.size()),
          "tracker graph output count matches its static plan");
    check(outputs.mask_logits_256 != nullptr && outputs.object_pointer != nullptr &&
              outputs.memory_features != nullptr,
          "tracker builder reports every state output");
    check(outputs.referenced_tensor_count > 0 && outputs.referenced_tensor_count <= 309,
          "tracker builder reports a bounded checkpoint reference count");
    check(outputs.added_layer_count == network->getNbLayers() && outputs.added_layer_count > 0,
          "tracker builder reports its native layer count");
    const std::size_t expected_weights = history_frames == 0 ? 185U : 291U;
    const std::int32_t expected_layers =
        history_frames == 0 ? 882 : 1630 + (history_frames - 1) * 22;
    check(outputs.referenced_tensor_count == expected_weights,
          "tracker plan references its exact reachable checkpoint subset");
    check(outputs.added_layer_count == expected_layers,
          "tracker plan has its exact native layer topology");

    for (std::int32_t index = 0; index < network->getNbInputs(); ++index)
        check(sameContract(*network->getInput(index), plan.inputs[static_cast<std::size_t>(index)]),
              "tracker input name, shape, and type match the ABI");
    for (std::int32_t index = 0; index < network->getNbOutputs(); ++index)
        check(
            sameContract(*network->getOutput(index), plan.outputs[static_cast<std::size_t>(index)]),
            "tracker output name, shape, and type match the ABI");

    std::cout << (history_frames == 0 ? "prompt" : "recurrent") << " H=" << history_frames
              << " layers=" << outputs.added_layer_count
              << " weights=" << outputs.referenced_tensor_count;
    if (serialize) {
        TrtPtr<nvinfer1::IBuilderConfig> config(builder.createBuilderConfig());
        check(config != nullptr, "tracker builder configuration creation");
        if (config) {
            config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, std::size_t{8} << 30U);
            config->clearFlag(nvinfer1::BuilderFlag::kTF32);
            TrtPtr<nvinfer1::IHostMemory> plan_memory(
                builder.buildSerializedNetwork(*network, *config));
            check(plan_memory != nullptr && plan_memory->size() > 0,
                  "tracker graph serializes to a nonempty plan");
            if (plan_memory)
                std::cout << " bytes=" << plan_memory->size();
        }
    }
    std::cout << '\n';
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string_view(argv[2]) != "--serialize")) {
        std::cerr << "usage: " << argv[0] << " delivered-checkpoint [--serialize]\n";
        return 2;
    }

    try {
        trtmc::sam2::native::CheckpointReader checkpoint =
            trtmc::sam2::native::CheckpointReader::open(argv[1]);
        Logger logger;
        TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
        check(builder != nullptr, "TensorRT builder creation");
        if (builder) {
            for (std::int32_t history = 0; history <= 4; ++history)
                testPlan(*builder, checkpoint, history, argc == 3);
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: live tracker graph construction: " << error.what() << '\n';
        ++failures;
    }
    return failures == 0 ? 0 : 1;
}
