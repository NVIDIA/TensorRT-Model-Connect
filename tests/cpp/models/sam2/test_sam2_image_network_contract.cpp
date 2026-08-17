/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tools/sam2_native_builder/sam2_image_network.h"

#include <NvInfer.h>
#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <string_view>

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

bool sameShape(const nvinfer1::Dims& actual, const std::array<int32_t, 4>& expected) {
    if (actual.nbDims != 4)
        return false;
    for (int32_t i = 0; i < actual.nbDims; ++i) {
        if (actual.d[i] != expected[static_cast<std::size_t>(i)])
            return false;
    }
    return true;
}

void testStaticContract() {
    using namespace trtmc::sam2::native;

    static_assert(kDeliveredCheckpointTensorCount == 603);
    static_assert(kImageNetworkReferencedTensorCount == 282);
    static_assert(kImageNetworkLayerCount == 1139);
    static_assert(kImageNetworkConvolutionLayerCount == 23);
    static_assert(kImageNetworkActivationLayerCount == 28);
    static_assert(kImageNetworkPoolingLayerCount == 6);
    static_assert(kImageNetworkElementWiseLayerCount == 130);
    static_assert(kImageNetworkShuffleLayerCount == 313);
    static_assert(kImageNetworkConstantLayerCount == 216);
    static_assert(kImageNetworkSliceLayerCount == 67);
    static_assert(kImageNetworkResizeLayerCount == 2);
    static_assert(kImageNetworkNormalizationLayerCount == 32);
    static_assert(kImageNetworkCastLayerCount == 223);
    static_assert(kImageNetworkMatrixMultiplyLayerCount == 67);
    static_assert(kImageNetworkSoftmaxLayerCount == 0);
    static_assert(kImageNetworkPluginV3LayerCount == 0);
    static_assert(kImageNetworkAttentionInputLayerCount == 16);
    static_assert(kImageNetworkAttentionOutputLayerCount == 16);
    static_assert(kHieraSmallBlocks.size() == 16);
    static_assert(kTrackerFpnContracts.size() == 3);
    static_assert(kBboxMapContracts.size() == 6);
    static_assert(kImageInputContract.type == nvinfer1::DataType::kFLOAT);
    static_assert(kTrackerFpnContracts[0].type == nvinfer1::DataType::kBF16);
    static_assert(kTrackerFpnContracts[1].type == nvinfer1::DataType::kBF16);
    static_assert(kTrackerFpnContracts[2].type == nvinfer1::DataType::kFLOAT);

#if NV_TENSORRT_MAJOR >= 11
    check(sam2NetworkCreationFlags() == 0,
          "TensorRT 11 uses its unconditional strongly typed mode");
#else
    const uint32_t strong_type_bit =
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    check((sam2NetworkCreationFlags() & strong_type_bit) != 0,
          "network creation flags require strong typing");
#endif

    int32_t query_pool_blocks = 0;
    int32_t global_blocks = 0;
    int32_t current_channels = 96;
    int32_t current_height = 256;
    for (const HieraBlockContract& block : kHieraSmallBlocks) {
        check(block.input_channels == current_channels, "Hiera channel chain is contiguous");
        check(block.input_height == current_height, "Hiera spatial chain is contiguous");
        check(block.output_channels % block.heads == 0, "Hiera head width is integral");
        if (block.query_pool) {
            ++query_pool_blocks;
            current_height /= 2;
        }
        if (block.window_size == 0)
            ++global_blocks;
        current_channels = block.output_channels;
    }
    check(query_pool_blocks == 3, "Hiera has exactly three query-pooling transitions");
    check(global_blocks == 3, "Hiera has exactly three global-attention blocks");
    check(current_channels == 768 && current_height == 32, "Hiera terminates at 768x32x32");
    check(kHieraSmallBlocks[7].window_size == 0 && kHieraSmallBlocks[10].window_size == 0 &&
              kHieraSmallBlocks[13].window_size == 0,
          "global-attention block indices match the delivered config");

    std::set<std::string_view> output_names;
    for (const StaticTensorContract& contract : kTrackerFpnContracts)
        output_names.insert(contract.name);
    for (const StaticTensorContract& contract : kBboxMapContracts)
        output_names.insert(contract.name);
    check(output_names.size() == 9, "image output names are unique");
    check(kBboxMapContracts[0].dimensions == std::array<int32_t, 4>{1, 2, 128, 128},
          "stride-8 classification contract");
    check(kBboxMapContracts[5].dimensions == std::array<int32_t, 4>{1, 4, 32, 32},
          "stride-32 regression contract");
}

void testLiveStructure(const char* checkpoint_path, bool serialize) {
    using namespace trtmc::sam2::native;

    CheckpointReader checkpoint = CheckpointReader::open(checkpoint_path);
    Logger logger;
    TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    check(builder != nullptr, "TensorRT builder creation");
    if (!builder)
        return;
    TrtPtr<nvinfer1::INetworkDefinition> network(
        builder->createNetworkV2(sam2NetworkCreationFlags()));
    check(network != nullptr, "strongly typed network creation");
    if (!network)
        return;

    Sam2ImageNetworkBuilder image_builder(*network, checkpoint);
    const Sam2ImageNetworkOutputs result = image_builder.build();
    check(network->getNbInputs() == 1, "image graph has one input");
    check(network->getNbOutputs() == 9, "image graph has nine outputs");
    check(result.checkpoint_tensor_count == kDeliveredCheckpointTensorCount,
          "build report contains the full checkpoint inventory");
    check(result.referenced_tensor_count == kImageNetworkReferencedTensorCount,
          "build report contains the exact referenced tensor count");
    check(result.unreferenced_tensor_count == 321,
          "build report separates non-image checkpoint tensors");
    check(result.added_layer_count == network->getNbLayers() &&
              result.added_layer_count == kImageNetworkLayerCount,
          "build report contains the exact structural layer count");
    check(result.convolution_layer_count == kImageNetworkConvolutionLayerCount &&
              result.activation_layer_count == kImageNetworkActivationLayerCount &&
              result.pooling_layer_count == kImageNetworkPoolingLayerCount &&
              result.element_wise_layer_count == kImageNetworkElementWiseLayerCount &&
              result.shuffle_layer_count == kImageNetworkShuffleLayerCount &&
              result.constant_layer_count == kImageNetworkConstantLayerCount &&
              result.slice_layer_count == kImageNetworkSliceLayerCount &&
              result.resize_layer_count == kImageNetworkResizeLayerCount &&
              result.normalization_layer_count == kImageNetworkNormalizationLayerCount &&
              result.cast_layer_count == kImageNetworkCastLayerCount &&
              result.matrix_multiply_layer_count == kImageNetworkMatrixMultiplyLayerCount &&
              result.softmax_layer_count == kImageNetworkSoftmaxLayerCount &&
              result.plugin_v3_layer_count == kImageNetworkPluginV3LayerCount &&
              result.attention_input_layer_count == kImageNetworkAttentionInputLayerCount &&
              result.attention_output_layer_count == kImageNetworkAttentionOutputLayerCount,
          "build report contains the exact TensorRT IAttentionV2 layer inventory");

    std::map<std::string_view, const StaticTensorContract*> expected;
    for (const StaticTensorContract& contract : kTrackerFpnContracts)
        expected.emplace(contract.name, &contract);
    for (const StaticTensorContract& contract : kBboxMapContracts)
        expected.emplace(contract.name, &contract);
    for (int32_t i = 0; i < network->getNbOutputs(); ++i) {
        const nvinfer1::ITensor* output = network->getOutput(i);
        check(output != nullptr, "network output is non-null");
        if (output == nullptr)
            continue;
        const auto found = expected.find(output->getName());
        check(found != expected.end(), "network output name is in the ABI");
        if (found != expected.end()) {
            check(output->getType() == found->second->type, "network output type matches the ABI");
            check(sameShape(output->getDimensions(), found->second->dimensions),
                  "network output shape matches the ABI");
        }
    }

    std::map<nvinfer1::LayerType, int32_t> layer_counts;
    for (int32_t i = 0; i < network->getNbLayers(); ++i)
        ++layer_counts[network->getLayer(i)->getType()];
    check(layer_counts[nvinfer1::LayerType::kCONVOLUTION] == 23,
          "graph has the exact patch/FPN/detector convolution count");
    check(layer_counts[nvinfer1::LayerType::kNORMALIZATION] == 32,
          "graph has two layer normalizations per Hiera block");
    check(layer_counts[nvinfer1::LayerType::kSOFTMAX] == kImageNetworkSoftmaxLayerCount,
          "graph has no decomposed softmax layers");
    check(layer_counts[nvinfer1::LayerType::kMATRIX_MULTIPLY] ==
              kImageNetworkMatrixMultiplyLayerCount,
          "graph has the exact projection and native-attention multiplication count");
    check(layer_counts[nvinfer1::LayerType::kPLUGIN_V3] == kImageNetworkPluginV3LayerCount,
          "graph has no attention plugin layers");
    check(layer_counts[nvinfer1::LayerType::kATTENTION_INPUT] ==
              kImageNetworkAttentionInputLayerCount,
          "graph has one IAttention input boundary per Hiera block");
    check(layer_counts[nvinfer1::LayerType::kATTENTION_OUTPUT] ==
              kImageNetworkAttentionOutputLayerCount,
          "graph has one IAttention output boundary per Hiera block");
    check(layer_counts[nvinfer1::LayerType::kRESIZE] == 2,
          "graph has positional and FPN resize layers");

    if (serialize) {
        TrtPtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
        check(config != nullptr, "builder configuration creation");
        if (config) {
            config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, std::size_t{8} << 30U);
            TrtPtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
            check(plan != nullptr && plan->size() > 0,
                  "full image graph serializes to a nonempty plan");
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    testStaticContract();
    if (argc == 2 || (argc == 3 && std::string_view(argv[2]) == "--serialize")) {
        try {
            testLiveStructure(argv[1], argc == 3);
        } catch (const std::exception& error) {
            std::cerr << "FAIL: live image graph construction: " << error.what() << '\n';
            ++failures;
        }
    } else if (argc > 2) {
        std::cerr << "usage: " << argv[0] << " [delivered-checkpoint [--serialize]]\n";
        return 2;
    }
    return failures == 0 ? 0 : 1;
}
