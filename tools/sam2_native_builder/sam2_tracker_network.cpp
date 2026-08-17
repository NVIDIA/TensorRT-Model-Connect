/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sam2_tracker_network.h"

#include "sam2_trt_layers.h"

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trtmc::sam2::native {
namespace {

constexpr float kLayerNorm2dEpsilon = 1.0e-6F;
constexpr float kTransformerLayerNormEpsilon = 1.0e-5F;
constexpr float kTwoPi = 6.28318530717958647692F;
constexpr std::int32_t kImageEmbeddingSize = 64;
constexpr std::int32_t kImageTokens = kImageEmbeddingSize * kImageEmbeddingSize;
constexpr std::int32_t kHiddenDim = 256;
constexpr std::int32_t kMemoryDim = 64;

nvinfer1::Dims dims(std::initializer_list<std::int32_t> values) {
    if (values.size() > static_cast<std::size_t>(nvinfer1::Dims::MAX_DIMS))
        throw NetworkBuildError("SAM2 tracker tensor rank exceeds TensorRT's limit");
    nvinfer1::Dims result{};
    result.nbDims = static_cast<std::int32_t>(values.size());
    std::int32_t index = 0;
    for (const std::int32_t value : values) {
        if (value <= 0)
            throw NetworkBuildError("SAM2 tracker tensor has a non-positive extent");
        result.d[index++] = value;
    }
    return result;
}

nvinfer1::Dims contractDims(const TensorContract& contract) {
    nvinfer1::Dims result{};
    result.nbDims = contract.rank;
    for (std::int32_t index = 0; index < result.nbDims; ++index) {
        const std::int32_t extent = contract.dimensions[static_cast<std::size_t>(index)];
        if (extent <= 0)
            throw NetworkBuildError("SAM2 tracker contract has a non-positive extent: " +
                                    std::string(contract.name));
        result.d[index] = extent;
    }
    return result;
}

nvinfer1::DataType dataType(TensorDataType type) {
    switch (type) {
    case TensorDataType::kFloat32:
        return nvinfer1::DataType::kFLOAT;
    case TensorDataType::kBFloat16:
        return nvinfer1::DataType::kBF16;
    }
    throw NetworkBuildError("unknown SAM2 tracker tensor data type");
}

bool sameDimensions(const nvinfer1::Dims& actual, const TensorContract& expected) {
    if (actual.nbDims != expected.rank)
        return false;
    for (std::int32_t index = 0; index < actual.nbDims; ++index) {
        if (actual.d[index] != expected.dimensions[static_cast<std::size_t>(index)])
            return false;
    }
    return true;
}

std::string dimensionsString(const nvinfer1::Dims& value) {
    std::ostringstream stream;
    stream << '[';
    for (std::int32_t index = 0; index < value.nbDims; ++index) {
        if (index != 0)
            stream << ',';
        stream << value.d[index];
    }
    stream << ']';
    return stream.str();
}

std::int64_t elementCount(std::initializer_list<std::int64_t> shape) {
    std::int64_t result = 1;
    for (const std::int64_t extent : shape) {
        if (extent <= 0 || result > std::numeric_limits<std::int64_t>::max() / extent)
            throw NetworkBuildError("SAM2 tracker weight shape overflow");
        result *= extent;
    }
    return result;
}

} // namespace

struct Sam2TrackerNetworkBuilder::Impl {
    Impl(nvinfer1::INetworkDefinition& network, const CheckpointReader& checkpoint)
        : network(network), checkpoint(checkpoint), layers(network, checkpoint) {}

    nvinfer1::ILayer& requireLayer(nvinfer1::ILayer* layer, std::string_view operation,
                                   std::string_view name) {
        if (layer == nullptr)
            throw NetworkBuildError(std::string(operation) + " rejected SAM2 layer " +
                                    std::string(name));
        const std::string stable_name(name);
        layer->setName(stable_name.c_str());
        return *layer;
    }

    void requireDimensions(const nvinfer1::ITensor& tensor,
                           std::initializer_list<std::int32_t> expected,
                           std::string_view name) const {
        const nvinfer1::Dims actual = tensor.getDimensions();
        if (actual.nbDims != static_cast<std::int32_t>(expected.size()))
            throw NetworkBuildError("SAM2 tensor rank mismatch in " + std::string(name));
        std::int32_t index = 0;
        for (const std::int32_t extent : expected) {
            if (actual.d[index++] != extent) {
                throw NetworkBuildError("SAM2 tensor extent mismatch in " + std::string(name));
            }
        }
    }

    nvinfer1::ITensor& addInput(const TensorContract& contract) {
        nvinfer1::ITensor* input = network.addInput(
            contract.name.data(), dataType(contract.data_type), contractDims(contract));
        if (input == nullptr || input->getType() != dataType(contract.data_type) ||
            !sameDimensions(input->getDimensions(), contract)) {
            throw NetworkBuildError("TensorRT rejected SAM2 input contract " +
                                    std::string(contract.name));
        }
        return *input;
    }

    nvinfer1::Weights requiredWeights(std::string_view name,
                                      std::initializer_list<std::int64_t> shape) {
        const std::vector<std::int64_t> shape_vector(shape.begin(), shape.end());
        const WeightView view = checkpoint.requireTensor(name, DType::kFloat32, shape_vector);
        const std::int64_t count = elementCount(shape);
        if (!view.contiguous || view.data == nullptr ||
            view.bytes != static_cast<std::size_t>(count) * sizeof(float)) {
            throw NetworkBuildError("invalid contiguous SAM2 weight " + std::string(name));
        }
        directly_referenced_weights.emplace(name);
        return {nvinfer1::DataType::kFLOAT, view.data, count};
    }

    nvinfer1::Weights bfloat16Weights(nvinfer1::Weights source) {
        if (source.type != nvinfer1::DataType::kFLOAT || source.values == nullptr ||
            source.count <= 0) {
            throw NetworkBuildError("invalid FP32 source for SAM2 BF16 weight conversion");
        }
        const auto* input = static_cast<const float*>(source.values);
        std::vector<std::uint16_t> output(static_cast<std::size_t>(source.count));
        for (std::int64_t index = 0; index < source.count; ++index) {
            std::uint32_t bits = 0;
            std::memcpy(&bits, input + index, sizeof(bits));
            bits += 0x7FFFU + ((bits >> 16U) & 1U);
            output[static_cast<std::size_t>(index)] = static_cast<std::uint16_t>(bits >> 16U);
        }
        owned_bfloat16_weights.push_back(std::move(output));
        const auto& storage = owned_bfloat16_weights.back();
        return {nvinfer1::DataType::kBF16, storage.data(),
                static_cast<std::int64_t>(storage.size())};
    }

    nvinfer1::ITensor& floatConstant(const std::vector<float>& values,
                                     std::initializer_list<std::int32_t> shape,
                                     std::string_view name,
                                     nvinfer1::DataType type = nvinfer1::DataType::kFLOAT) {
        std::int64_t count = 1;
        for (const std::int32_t extent : shape)
            count *= extent;
        if (values.empty() || count != static_cast<std::int64_t>(values.size()))
            throw NetworkBuildError("generated SAM2 constant shape mismatch for " +
                                    std::string(name));
        owned_float_constants.push_back(values);
        const auto& storage = owned_float_constants.back();
        const nvinfer1::Weights weights{nvinfer1::DataType::kFLOAT, storage.data(), count};
        auto* layer = network.addConstant(dims(shape), weights);
        requireLayer(layer, "constant", name);
        if (type == nvinfer1::DataType::kFLOAT)
            return *layer->getOutput(0);
        return layers.cast(*layer->getOutput(0), type, std::string(name) + ".cast");
    }

    nvinfer1::ITensor& intConstant(std::int32_t value, std::initializer_list<std::int32_t> shape,
                                   std::string_view name) {
        std::int64_t count = 1;
        for (const std::int32_t extent : shape)
            count *= extent;
        owned_int_constants.emplace_back(static_cast<std::size_t>(count), value);
        const auto& storage = owned_int_constants.back();
        const nvinfer1::Weights weights{nvinfer1::DataType::kINT32, storage.data(), count};
        auto* layer = network.addConstant(dims(shape), weights);
        requireLayer(layer, "integer constant", name);
        return *layer->getOutput(0);
    }

    nvinfer1::ITensor& unary(nvinfer1::ITensor& input, nvinfer1::UnaryOperation operation,
                             std::string_view name) {
        auto* layer = network.addUnary(input, operation);
        requireLayer(layer, "unary", name);
        return *layer->getOutput(0);
    }

    nvinfer1::ITensor& activation(nvinfer1::ITensor& input, nvinfer1::ActivationType type,
                                  std::string_view name) {
        auto* layer = network.addActivation(input, type);
        requireLayer(layer, "activation", name);
        return *layer->getOutput(0);
    }

    nvinfer1::ITensor& reduce(nvinfer1::ITensor& input, nvinfer1::ReduceOperation operation,
                              std::uint32_t axes, bool keep_dimensions, std::string_view name) {
        auto* layer = network.addReduce(input, operation, axes, keep_dimensions);
        requireLayer(layer, "reduction", name);
        return *layer->getOutput(0);
    }

    nvinfer1::ITensor& promote(nvinfer1::ITensor& input, nvinfer1::DataType type,
                               std::string_view name) {
        return layers.cast(input, type, name);
    }

    nvinfer1::ITensor& binary(nvinfer1::ITensor& lhs, nvinfer1::ITensor& rhs,
                              nvinfer1::ElementWiseOperation operation, std::string_view name,
                              bool force_float = false) {
        nvinfer1::DataType type = lhs.getType();
        if (force_float || lhs.getType() == nvinfer1::DataType::kFLOAT ||
            rhs.getType() == nvinfer1::DataType::kFLOAT) {
            type = nvinfer1::DataType::kFLOAT;
        } else if (lhs.getType() != rhs.getType()) {
            throw NetworkBuildError("unsupported SAM2 elementwise type promotion in " +
                                    std::string(name));
        }
        nvinfer1::ITensor& left = promote(lhs, type, std::string(name) + ".lhs");
        nvinfer1::ITensor& right = promote(rhs, type, std::string(name) + ".rhs");
        return layers.elementWise(left, right, operation, name);
    }

    nvinfer1::ITensor& scale(nvinfer1::ITensor& input, float value, std::string_view name) {
        nvinfer1::ITensor& scalar = layers.scalar(value, input.getDimensions().nbDims,
                                                  input.getType(), std::string(name) + ".scalar");
        return layers.elementWise(input, scalar, nvinfer1::ElementWiseOperation::kPROD, name);
    }

    nvinfer1::ITensor& select(nvinfer1::ITensor& condition, nvinfer1::ITensor& then_value,
                              nvinfer1::ITensor& else_value, std::string_view name) {
        if (then_value.getType() != else_value.getType())
            throw NetworkBuildError("SAM2 select branch type mismatch in " + std::string(name));
        auto* layer = network.addSelect(condition, then_value, else_value);
        requireLayer(layer, "select", name);
        return *layer->getOutput(0);
    }

    nvinfer1::ITensor& checkpointConstant(std::string_view checkpoint_name,
                                          std::initializer_list<std::int64_t> checkpoint_shape,
                                          std::initializer_list<std::int64_t> tensor_shape,
                                          std::string_view name,
                                          nvinfer1::DataType type = nvinfer1::DataType::kFLOAT) {
        nvinfer1::ITensor& value =
            layers.constant(checkpoint_name, checkpoint_shape, tensor_shape, name);
        return layers.cast(value, type, std::string(name) + ".cast");
    }

    nvinfer1::ITensor& conv(nvinfer1::ITensor& input, std::string_view module,
                            std::int32_t input_channels, std::int32_t output_channels,
                            std::int32_t kernel, std::int32_t stride, std::int32_t padding,
                            std::int32_t groups, std::string_view name) {
        nvinfer1::ITensor& bf16 =
            layers.cast(input, nvinfer1::DataType::kBF16, std::string(name) + ".input");
        return layers.convolution(bf16, std::string(module) + ".weight",
                                  std::string(module) + ".bias", input_channels, output_channels,
                                  kernel, stride, padding, groups, name);
    }

    nvinfer1::ITensor& deconv(nvinfer1::ITensor& input, std::string_view module,
                              std::int32_t input_channels, std::int32_t output_channels,
                              std::int32_t kernel, std::int32_t stride, std::string_view name) {
        if (kernel != 2 || stride != 2)
            throw NetworkBuildError(
                "SAM2 tracker supports only exact 2x pixel-shuffle deconvolution");
        nvinfer1::ITensor& bf16 =
            layers.cast(input, nvinfer1::DataType::kBF16, std::string(name) + ".input");
        const nvinfer1::Weights source_kernel = requiredWeights(
            std::string(module) + ".weight", {input_channels, output_channels, kernel, kernel});
        const nvinfer1::Weights source_bias =
            requiredWeights(std::string(module) + ".bias", {output_channels});
        const auto* source_kernel_values = static_cast<const float*>(source_kernel.values);
        const auto* source_bias_values = static_cast<const float*>(source_bias.values);
        std::vector<float> projection_kernel(static_cast<std::size_t>(output_channels) * 4U *
                                             input_channels);
        std::vector<float> projection_bias(static_cast<std::size_t>(output_channels) * 4U);
        for (std::int32_t output = 0; output < output_channels; ++output) {
            for (std::int32_t y = 0; y < 2; ++y) {
                for (std::int32_t x = 0; x < 2; ++x) {
                    const std::int32_t phase = y * 2 + x;
                    const std::int32_t projection_output = output * 4 + phase;
                    projection_bias[static_cast<std::size_t>(projection_output)] =
                        source_bias_values[output];
                    for (std::int32_t input_channel = 0; input_channel < input_channels;
                         ++input_channel) {
                        const std::size_t source_offset =
                            ((static_cast<std::size_t>(input_channel) * output_channels + output) *
                                 2U +
                             static_cast<std::size_t>(y)) *
                                2U +
                            static_cast<std::size_t>(x);
                        const std::size_t destination_offset =
                            static_cast<std::size_t>(projection_output) * input_channels +
                            input_channel;
                        projection_kernel[destination_offset] = source_kernel_values[source_offset];
                    }
                }
            }
        }
        owned_transposed_convolution_weights.push_back(std::move(projection_kernel));
        const auto& kernel_storage = owned_transposed_convolution_weights.back();
        const nvinfer1::Weights kernel_weights =
            bfloat16Weights({nvinfer1::DataType::kFLOAT, kernel_storage.data(),
                             static_cast<std::int64_t>(kernel_storage.size())});
        owned_transposed_convolution_weights.push_back(std::move(projection_bias));
        const auto& bias_storage = owned_transposed_convolution_weights.back();
        const nvinfer1::Weights bias_weights =
            bfloat16Weights({nvinfer1::DataType::kFLOAT, bias_storage.data(),
                             static_cast<std::int64_t>(bias_storage.size())});
        auto* projection = network.addConvolutionNd(bf16, output_channels * 4, dims({1, 1}),
                                                    kernel_weights, bias_weights);
        requireLayer(projection, "pixel-shuffle projection", std::string(name) + ".projection");
        projection->setStrideNd(dims({1, 1}));
        projection->setNbGroups(1);

        const nvinfer1::Dims input_shape = bf16.getDimensions();
        if (input_shape.nbDims != 4 || input_shape.d[0] != 1 ||
            input_shape.d[1] != input_channels || input_shape.d[2] <= 0 || input_shape.d[3] <= 0 ||
            input_shape.d[2] > std::numeric_limits<std::int32_t>::max() / 2 ||
            input_shape.d[3] > std::numeric_limits<std::int32_t>::max() / 2) {
            throw NetworkBuildError("SAM2 pixel-shuffle input shape mismatch in " +
                                    std::string(name));
        }
        const std::int32_t input_height = static_cast<std::int32_t>(input_shape.d[2]);
        const std::int32_t input_width = static_cast<std::int32_t>(input_shape.d[3]);
        nvinfer1::ITensor& blocked = layers.shuffle(
            *projection->getOutput(0), {1, output_channels, 2, 2, input_height, input_width},
            std::string(name) + ".blocked");
        nvinfer1::ITensor& ordered =
            layers.transpose(blocked, {0, 1, 4, 2, 5, 3}, std::string(name) + ".ordered");
        nvinfer1::ITensor& result =
            layers.shuffle(ordered, {1, output_channels, input_height * 2, input_width * 2},
                           std::string(name) + ".pixel_shuffle");
        requireDimensions(result, {1, output_channels, input_height * 2, input_width * 2}, name);
        return result;
    }

    nvinfer1::ITensor& layerNorm2d(nvinfer1::ITensor& input, std::string_view module,
                                   std::int32_t channels, std::string_view name) {
        const nvinfer1::Dims input_shape = input.getDimensions();
        if (input_shape.nbDims != 4 || input_shape.d[0] != 1 || input_shape.d[1] != channels ||
            input_shape.d[2] <= 0 || input_shape.d[3] <= 0) {
            throw NetworkBuildError("SAM2 LayerNorm2d input shape mismatch in " +
                                    std::string(name));
        }
        nvinfer1::ITensor& mean = reduce(input, nvinfer1::ReduceOperation::kAVG, 1U << 1U, true,
                                         std::string(name) + ".mean_bf16");
        nvinfer1::ITensor& centered = binary(input, mean, nvinfer1::ElementWiseOperation::kSUB,
                                             std::string(name) + ".center_bf16");
        nvinfer1::ITensor& centered_fp32 =
            layers.cast(centered, nvinfer1::DataType::kFLOAT, std::string(name) + ".center_fp32");
        nvinfer1::ITensor& squared =
            layers.elementWise(centered_fp32, centered_fp32, nvinfer1::ElementWiseOperation::kPROD,
                               std::string(name) + ".square_fp32");
        nvinfer1::ITensor& variance = reduce(squared, nvinfer1::ReduceOperation::kAVG, 1U << 1U,
                                             true, std::string(name) + ".variance_fp32");
        nvinfer1::ITensor& epsilon =
            layers.scalar(kLayerNorm2dEpsilon, 4, nvinfer1::DataType::kFLOAT,
                          std::string(name) + ".epsilon_fp32");
        nvinfer1::ITensor& variance_with_epsilon =
            layers.elementWise(variance, epsilon, nvinfer1::ElementWiseOperation::kSUM,
                               std::string(name) + ".variance_with_epsilon");
        nvinfer1::ITensor& deviation = unary(variance_with_epsilon, nvinfer1::UnaryOperation::kSQRT,
                                             std::string(name) + ".deviation_fp32");
        nvinfer1::ITensor& normalized =
            layers.elementWise(centered_fp32, deviation, nvinfer1::ElementWiseOperation::kDIV,
                               std::string(name) + ".normalize_fp32");
        nvinfer1::ITensor& weight =
            checkpointConstant(std::string(module) + ".weight", {channels}, {1, channels, 1, 1},
                               std::string(name) + ".weight_fp32");
        nvinfer1::ITensor& scaled =
            layers.elementWise(normalized, weight, nvinfer1::ElementWiseOperation::kPROD,
                               std::string(name) + ".scale_fp32");
        nvinfer1::ITensor& bias =
            checkpointConstant(std::string(module) + ".bias", {channels}, {1, channels, 1, 1},
                               std::string(name) + ".bias_fp32");
        return layers.elementWise(scaled, bias, nvinfer1::ElementWiseOperation::kSUM, name);
    }

    nvinfer1::ITensor& relu(nvinfer1::ITensor& input, std::string_view name) {
        return activation(input, nvinfer1::ActivationType::kRELU, name);
    }

    nvinfer1::ITensor& linearRelu(nvinfer1::ITensor& input, std::string_view module,
                                  std::int32_t input_features, std::int32_t output_features,
                                  std::string_view name) {
        nvinfer1::ITensor& value = layers.linearAutocastBf16(
            input, module, input_features, output_features, std::string(name) + ".linear");
        return relu(value, name);
    }

    nvinfer1::ITensor& makeSinePosition(std::int32_t channels, std::string_view name) {
        if (channels != 64 && channels != 256)
            throw NetworkBuildError("unsupported SAM2 sine-position width");
        const std::int32_t half = channels / 2;
        std::vector<float> values(static_cast<std::size_t>(channels) * kImageTokens);
        for (std::int32_t channel = 0; channel < channels; ++channel) {
            const bool use_y = channel < half;
            const std::int32_t component = use_y ? channel : channel - half;
            const float exponent =
                2.0F * static_cast<float>(component / 2) / static_cast<float>(half);
            const float divisor = std::pow(10000.0F, exponent);
            for (std::int32_t y = 0; y < kImageEmbeddingSize; ++y) {
                const float y_position = static_cast<float>(y + 1) /
                                         (static_cast<float>(kImageEmbeddingSize) + 1.0e-6F) *
                                         kTwoPi;
                for (std::int32_t x = 0; x < kImageEmbeddingSize; ++x) {
                    const float x_position = static_cast<float>(x + 1) /
                                             (static_cast<float>(kImageEmbeddingSize) + 1.0e-6F) *
                                             kTwoPi;
                    const float phase = (use_y ? y_position : x_position) / divisor;
                    const float encoded = component % 2 == 0 ? std::sin(phase) : std::cos(phase);
                    const std::size_t offset =
                        (static_cast<std::size_t>(channel) * kImageEmbeddingSize + y) *
                            kImageEmbeddingSize +
                        x;
                    values[offset] = encoded;
                }
            }
        }
        return floatConstant(values, {1, channels, kImageEmbeddingSize, kImageEmbeddingSize}, name);
    }

    nvinfer1::ITensor& makeDenseRandomPosition() {
        std::vector<float> coordinates(static_cast<std::size_t>(kImageTokens) * 2U);
        for (std::int32_t y = 0; y < kImageEmbeddingSize; ++y) {
            const float normalized_y =
                (static_cast<float>(y) + 0.5F) / static_cast<float>(kImageEmbeddingSize);
            for (std::int32_t x = 0; x < kImageEmbeddingSize; ++x) {
                const float normalized_x =
                    (static_cast<float>(x) + 0.5F) / static_cast<float>(kImageEmbeddingSize);
                const std::size_t spatial = static_cast<std::size_t>(y) * kImageEmbeddingSize + x;
                coordinates[spatial * 2U] = 2.0F * normalized_x - 1.0F;
                coordinates[spatial * 2U + 1U] = 2.0F * normalized_y - 1.0F;
            }
        }
        nvinfer1::ITensor& grid =
            floatConstant(coordinates, {1, kImageTokens, 2}, "prompt.dense_position.grid",
                          nvinfer1::DataType::kBF16);
        nvinfer1::ITensor& gaussian = checkpointConstant(
            "sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix", {2, 128},
            {1, 2, 128}, "prompt.dense_position.gaussian", nvinfer1::DataType::kBF16);
        nvinfer1::ITensor& phase = layers.matrixMultiply(grid, nvinfer1::MatrixOperation::kNONE,
                                                         gaussian, nvinfer1::MatrixOperation::kNONE,
                                                         "prompt.dense_position.fourier");
        nvinfer1::ITensor& radians = scale(phase, kTwoPi, "prompt.dense_position.radians");
        nvinfer1::ITensor& sine =
            unary(radians, nvinfer1::UnaryOperation::kSIN, "prompt.dense_position.sin");
        nvinfer1::ITensor& cosine =
            unary(radians, nvinfer1::UnaryOperation::kCOS, "prompt.dense_position.cos");
        nvinfer1::ITensor& encoded =
            layers.concatenate({&sine, &cosine}, 2, "prompt.dense_position.encoded");
        nvinfer1::ITensor& nhwc =
            layers.shuffle(encoded, {1, 64, 64, 256}, "prompt.dense_position.nhwc");
        return layers.transpose(nhwc, {0, 3, 1, 2}, "prompt.dense_position.nchw");
    }

    nvinfer1::ITensor& makePointerTemporalPosition(std::int32_t distance, std::string_view name) {
        std::vector<float> values(256);
        const float position = static_cast<float>(distance) / 4.0F;
        for (std::int32_t feature = 0; feature < 128; ++feature) {
            const float exponent = 2.0F * static_cast<float>(feature / 2) / 128.0F;
            const float phase = position / std::pow(10000.0F, exponent);
            values[static_cast<std::size_t>(feature)] = std::sin(phase);
            values[static_cast<std::size_t>(128 + feature)] = std::cos(phase);
        }
        return floatConstant(values, {1, 256}, name);
    }

    nvinfer1::ITensor& buildSparsePrompt(nvinfer1::ITensor* box, TrackerPlanKind kind) {
        nvinfer1::ITensor& not_a_point =
            checkpointConstant("sam_prompt_encoder.not_a_point_embed.weight", {1, 256}, {1, 1, 256},
                               "prompt.not_a_point");
        if (kind == TrackerPlanKind::kRecurrent)
            return layers.concatenate({&not_a_point, &not_a_point}, 1, "prompt.recurrent_padding");
        if (box == nullptr)
            throw NetworkBuildError("SAM2 prompt graph requires box_xyxy_1024");

        nvinfer1::ITensor& coordinates = layers.shuffle(*box, {1, 2, 2}, "prompt.box.reshape");
        nvinfer1::ITensor& half =
            layers.scalar(0.5F, 3, nvinfer1::DataType::kFLOAT, "prompt.box.half");
        nvinfer1::ITensor& centered =
            binary(coordinates, half, nvinfer1::ElementWiseOperation::kSUM, "prompt.box.centered");
        nvinfer1::ITensor& normalized = scale(centered, 1.0F / 1024.0F, "prompt.box.normalized");
        nvinfer1::ITensor& doubled = scale(normalized, 2.0F, "prompt.box.doubled");
        nvinfer1::ITensor& minus_one =
            layers.scalar(-1.0F, 3, nvinfer1::DataType::kFLOAT, "prompt.box.minus_one");
        nvinfer1::ITensor& unit = binary(doubled, minus_one, nvinfer1::ElementWiseOperation::kSUM,
                                         "prompt.box.unit_square");
        nvinfer1::ITensor& unit_bf16 =
            layers.cast(unit, nvinfer1::DataType::kBF16, "prompt.box.unit_square_bf16");
        nvinfer1::ITensor& gaussian = checkpointConstant(
            "sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix", {2, 128},
            {1, 2, 128}, "prompt.box.gaussian", nvinfer1::DataType::kBF16);
        nvinfer1::ITensor& phase =
            layers.matrixMultiply(unit_bf16, nvinfer1::MatrixOperation::kNONE, gaussian,
                                  nvinfer1::MatrixOperation::kNONE, "prompt.box.fourier");
        nvinfer1::ITensor& radians = scale(phase, kTwoPi, "prompt.box.radians");
        nvinfer1::ITensor& sine = unary(radians, nvinfer1::UnaryOperation::kSIN, "prompt.box.sin");
        nvinfer1::ITensor& cosine =
            unary(radians, nvinfer1::UnaryOperation::kCOS, "prompt.box.cos");
        nvinfer1::ITensor& encoded = layers.concatenate({&sine, &cosine}, 2, "prompt.box.encoded");
        nvinfer1::ITensor& corner0 =
            layers.slice(encoded, {0, 0, 0}, {1, 1, 256}, {1, 1, 1},
                         nvinfer1::SampleMode::kSTRICT_BOUNDS, "prompt.box.corner0");
        nvinfer1::ITensor& corner1 =
            layers.slice(encoded, {0, 1, 0}, {1, 1, 256}, {1, 1, 1},
                         nvinfer1::SampleMode::kSTRICT_BOUNDS, "prompt.box.corner1");
        nvinfer1::ITensor& top_left =
            checkpointConstant("sam_prompt_encoder.point_embeddings.2.weight", {1, 256},
                               {1, 1, 256}, "prompt.box.top_left_embedding");
        nvinfer1::ITensor& bottom_right =
            checkpointConstant("sam_prompt_encoder.point_embeddings.3.weight", {1, 256},
                               {1, 1, 256}, "prompt.box.bottom_right_embedding");
        nvinfer1::ITensor& embedded0 =
            binary(corner0, top_left, nvinfer1::ElementWiseOperation::kSUM,
                   "prompt.box.corner0_embedding");
        nvinfer1::ITensor& embedded1 =
            binary(corner1, bottom_right, nvinfer1::ElementWiseOperation::kSUM,
                   "prompt.box.corner1_embedding");
        return layers.concatenate({&embedded0, &embedded1, &not_a_point}, 1,
                                  "prompt.box.sparse_embeddings");
    }

    std::pair<nvinfer1::ITensor*, nvinfer1::ITensor*>
    rotate(nvinfer1::ITensor& input, std::int32_t sequence, const std::vector<float>& cosine,
           const std::vector<float>& sine, std::string_view name) {
        const nvinfer1::Dims shape = input.getDimensions();
        if (shape.nbDims != 4 || shape.d[0] != 1 || shape.d[1] != 1 || shape.d[2] != sequence ||
            shape.d[3] != 256 || cosine.size() != static_cast<std::size_t>(sequence) * 128 ||
            sine.size() != cosine.size()) {
            throw NetworkBuildError("SAM2 RoPE shape mismatch in " + std::string(name));
        }
        nvinfer1::DataType original_type = input.getType();
        nvinfer1::ITensor& fp32 =
            layers.cast(input, nvinfer1::DataType::kFLOAT, std::string(name) + ".fp32");
        nvinfer1::ITensor& paired =
            layers.shuffle(fp32, {1, 1, sequence, 128, 2}, std::string(name) + ".pairs");
        nvinfer1::ITensor& real_rank5 =
            layers.slice(paired, {0, 0, 0, 0, 0}, {1, 1, sequence, 128, 1}, {1, 1, 1, 1, 1},
                         nvinfer1::SampleMode::kSTRICT_BOUNDS, std::string(name) + ".real_slice");
        nvinfer1::ITensor& imag_rank5 =
            layers.slice(paired, {0, 0, 0, 0, 1}, {1, 1, sequence, 128, 1}, {1, 1, 1, 1, 1},
                         nvinfer1::SampleMode::kSTRICT_BOUNDS, std::string(name) + ".imag_slice");
        nvinfer1::ITensor& cos_tensor =
            floatConstant(cosine, {1, 1, sequence, 128, 1}, std::string(name) + ".cos");
        nvinfer1::ITensor& sin_tensor =
            floatConstant(sine, {1, 1, sequence, 128, 1}, std::string(name) + ".sin");
        nvinfer1::ITensor& real_cos =
            binary(real_rank5, cos_tensor, nvinfer1::ElementWiseOperation::kPROD,
                   std::string(name) + ".real_cos", true);
        nvinfer1::ITensor& imag_sin =
            binary(imag_rank5, sin_tensor, nvinfer1::ElementWiseOperation::kPROD,
                   std::string(name) + ".imag_sin", true);
        nvinfer1::ITensor& rotated_real =
            binary(real_cos, imag_sin, nvinfer1::ElementWiseOperation::kSUB,
                   std::string(name) + ".rotated_real", true);
        nvinfer1::ITensor& real_sin =
            binary(real_rank5, sin_tensor, nvinfer1::ElementWiseOperation::kPROD,
                   std::string(name) + ".real_sin", true);
        nvinfer1::ITensor& imag_cos =
            binary(imag_rank5, cos_tensor, nvinfer1::ElementWiseOperation::kPROD,
                   std::string(name) + ".imag_cos", true);
        nvinfer1::ITensor& rotated_imag =
            binary(real_sin, imag_cos, nvinfer1::ElementWiseOperation::kSUM,
                   std::string(name) + ".rotated_imag", true);
        nvinfer1::ITensor& interleaved = layers.concatenate({&rotated_real, &rotated_imag}, 4,
                                                            std::string(name) + ".interleaved");
        requireDimensions(interleaved, {1, 1, sequence, 128, 2},
                          std::string(name) + ".interleaved");
        nvinfer1::ITensor& flattened =
            layers.shuffle(interleaved, {1, 1, sequence, 256}, std::string(name) + ".flatten");
        requireDimensions(flattened, {1, 1, sequence, 256}, std::string(name) + ".flatten");
        nvinfer1::ITensor& result =
            layers.cast(flattened, original_type, std::string(name) + ".restore_type");
        return {&result, nullptr};
    }

    std::pair<std::vector<float>, std::vector<float>> axialFrequencies(std::int32_t repeats) {
        if (repeats <= 0)
            throw NetworkBuildError("SAM2 RoPE repeat count must be positive");
        std::vector<float> cosine(static_cast<std::size_t>(repeats) * kImageTokens * 128);
        std::vector<float> sine(cosine.size());
        for (std::int32_t repeat = 0; repeat < repeats; ++repeat) {
            for (std::int32_t token = 0; token < kImageTokens; ++token) {
                const float x = static_cast<float>(token % kImageEmbeddingSize);
                const float y = static_cast<float>(token / kImageEmbeddingSize);
                for (std::int32_t feature = 0; feature < 64; ++feature) {
                    const float frequency =
                        1.0F / std::pow(10000.0F, static_cast<float>(feature * 4) / 256.0F);
                    for (std::int32_t axis = 0; axis < 2; ++axis) {
                        const float phase = (axis == 0 ? x : y) * frequency;
                        const std::int32_t complex_feature = axis * 64 + feature;
                        const std::size_t offset =
                            (static_cast<std::size_t>(repeat) * kImageTokens + token) * 128 +
                            complex_feature;
                        cosine[offset] = std::cos(phase);
                        sine[offset] = std::sin(phase);
                    }
                }
            }
        }
        return {std::move(cosine), std::move(sine)};
    }

    nvinfer1::ITensor& attention(nvinfer1::ITensor& query, nvinfer1::ITensor& key,
                                 nvinfer1::ITensor& value, std::string_view module,
                                 std::int32_t query_features, std::int32_t key_value_features,
                                 std::int32_t internal_features, std::int32_t heads,
                                 std::string_view name, bool rope = false,
                                 std::int32_t rope_key_tokens = 0) {
        if (internal_features % heads != 0)
            throw NetworkBuildError("SAM2 attention head width does not divide");
        const std::int32_t query_tokens = query.getDimensions().d[1];
        const std::int32_t key_tokens = key.getDimensions().d[1];
        const std::int32_t head_dim = internal_features / heads;
        nvinfer1::ITensor& projected_q =
            layers.linearAutocastBf16(query, std::string(module) + ".q_proj", query_features,
                                      internal_features, std::string(name) + ".q_proj");
        nvinfer1::ITensor& projected_k =
            layers.linearAutocastBf16(key, std::string(module) + ".k_proj", key_value_features,
                                      internal_features, std::string(name) + ".k_proj");
        nvinfer1::ITensor& projected_v =
            layers.linearAutocastBf16(value, std::string(module) + ".v_proj", key_value_features,
                                      internal_features, std::string(name) + ".v_proj");
        nvinfer1::ITensor& q_grouped = layers.shuffle(
            projected_q, {1, query_tokens, heads, head_dim}, std::string(name) + ".q_group");
        nvinfer1::ITensor& k_grouped = layers.shuffle(projected_k, {1, key_tokens, heads, head_dim},
                                                      std::string(name) + ".k_group");
        nvinfer1::ITensor& v_grouped = layers.shuffle(projected_v, {1, key_tokens, heads, head_dim},
                                                      std::string(name) + ".v_group");
        nvinfer1::ITensor* q_heads =
            &layers.transpose(q_grouped, {0, 2, 1, 3}, std::string(name) + ".q_heads");
        nvinfer1::ITensor* k_heads =
            &layers.transpose(k_grouped, {0, 2, 1, 3}, std::string(name) + ".k_heads");
        nvinfer1::ITensor& v_heads =
            layers.transpose(v_grouped, {0, 2, 1, 3}, std::string(name) + ".v_heads");
        if (rope) {
            if (heads != 1 || head_dim != 256 || query_tokens != kImageTokens ||
                rope_key_tokens < query_tokens || rope_key_tokens > key_tokens ||
                rope_key_tokens % query_tokens != 0) {
                throw NetworkBuildError("SAM2 axial RoPE static geometry mismatch");
            }
            const auto q_frequency = axialFrequencies(1);
            q_heads = rotate(*q_heads, query_tokens, q_frequency.first, q_frequency.second,
                             std::string(name) + ".q_rope")
                          .first;
            nvinfer1::ITensor& k_rope_part = layers.slice(
                *k_heads, {0, 0, 0, 0}, {1, 1, rope_key_tokens, 256}, {1, 1, 1, 1},
                nvinfer1::SampleMode::kSTRICT_BOUNDS, std::string(name) + ".k_rope_slice");
            const auto k_frequency = axialFrequencies(rope_key_tokens / query_tokens);
            nvinfer1::ITensor* rotated_k = rotate(k_rope_part, rope_key_tokens, k_frequency.first,
                                                  k_frequency.second, std::string(name) + ".k_rope")
                                               .first;
            if (rope_key_tokens != key_tokens) {
                nvinfer1::ITensor& excluded = layers.slice(
                    *k_heads, {0, 0, rope_key_tokens, 0}, {1, 1, key_tokens - rope_key_tokens, 256},
                    {1, 1, 1, 1}, nvinfer1::SampleMode::kSTRICT_BOUNDS,
                    std::string(name) + ".k_pointer_slice");
                k_heads = &layers.concatenate({rotated_k, &excluded}, 2,
                                              std::string(name) + ".k_with_pointers");
            } else {
                k_heads = rotated_k;
            }
        }
        nvinfer1::ITensor& scores = layers.matrixMultiply(
            *q_heads, nvinfer1::MatrixOperation::kNONE, *k_heads,
            nvinfer1::MatrixOperation::kTRANSPOSE, std::string(name) + ".scores");
        nvinfer1::ITensor& scaled = scale(scores, 1.0F / std::sqrt(static_cast<float>(head_dim)),
                                          std::string(name) + ".scaled");
        nvinfer1::ITensor& probabilities =
            layers.softmax(scaled, 1U << 3U, std::string(name) + ".softmax");
        nvinfer1::ITensor& attended =
            layers.matrixMultiply(probabilities, nvinfer1::MatrixOperation::kNONE, v_heads,
                                  nvinfer1::MatrixOperation::kNONE, std::string(name) + ".values");
        nvinfer1::ITensor& token_major =
            layers.transpose(attended, {0, 2, 1, 3}, std::string(name) + ".token_major");
        nvinfer1::ITensor& recombined = layers.shuffle(
            token_major, {1, query_tokens, internal_features}, std::string(name) + ".recombine");
        return layers.linearAutocastBf16(recombined, std::string(module) + ".out_proj",
                                         internal_features, query_features,
                                         std::string(name) + ".out_proj");
    }

    std::pair<nvinfer1::ITensor*, nvinfer1::ITensor*>
    twoWayTransformer(nvinfer1::ITensor& image, nvinfer1::ITensor& image_position,
                      nvinfer1::ITensor& prompt_tokens, std::int32_t decoder_tokens) {
        nvinfer1::ITensor& image_flat =
            layers.shuffle(image, {1, 256, kImageTokens}, "decoder.image.flatten");
        nvinfer1::ITensor* keys =
            &layers.transpose(image_flat, {0, 2, 1}, "decoder.image.token_major");
        nvinfer1::ITensor& position_flat =
            layers.shuffle(image_position, {1, 256, kImageTokens}, "decoder.position.flatten");
        nvinfer1::ITensor& key_position =
            layers.transpose(position_flat, {0, 2, 1}, "decoder.position.token_major");
        nvinfer1::ITensor* queries = &prompt_tokens;
        nvinfer1::ITensor& query_position = prompt_tokens;
        for (std::int32_t index = 0; index < 2; ++index) {
            const std::string prefix =
                "sam_mask_decoder.transformer.layers." + std::to_string(index);
            const std::string name = "decoder.two_way." + std::to_string(index);
            if (index == 0) {
                queries = &attention(*queries, *queries, *queries, prefix + ".self_attn", 256, 256,
                                     256, 8, name + ".self");
            } else {
                nvinfer1::ITensor& q_with_position =
                    binary(*queries, query_position, nvinfer1::ElementWiseOperation::kSUM,
                           name + ".self.q_with_position", true);
                nvinfer1::ITensor& attended =
                    attention(q_with_position, q_with_position, *queries, prefix + ".self_attn",
                              256, 256, 256, 8, name + ".self.attention");
                queries = &binary(*queries, attended, nvinfer1::ElementWiseOperation::kSUM,
                                  name + ".self.residual", true);
            }
            queries = &layers.layerNormFp32(*queries, prefix + ".norm1", 256,
                                            kTransformerLayerNormEpsilon, name + ".norm1");

            nvinfer1::ITensor& token_query =
                binary(*queries, query_position, nvinfer1::ElementWiseOperation::kSUM,
                       name + ".token_to_image.query", true);
            nvinfer1::ITensor& image_key =
                binary(*keys, key_position, nvinfer1::ElementWiseOperation::kSUM,
                       name + ".token_to_image.key", true);
            nvinfer1::ITensor& token_attended =
                attention(token_query, image_key, *keys, prefix + ".cross_attn_token_to_image", 256,
                          256, 128, 8, name + ".token_to_image.attention");
            queries = &binary(*queries, token_attended, nvinfer1::ElementWiseOperation::kSUM,
                              name + ".token_to_image.residual", true);
            queries = &layers.layerNormFp32(*queries, prefix + ".norm2", 256,
                                            kTransformerLayerNormEpsilon, name + ".norm2");

            nvinfer1::ITensor& hidden =
                linearRelu(*queries, prefix + ".mlp.layers.0", 256, 2048, name + ".mlp.relu");
            nvinfer1::ITensor& mlp = layers.linearAutocastBf16(hidden, prefix + ".mlp.layers.1",
                                                               2048, 256, name + ".mlp.output");
            queries = &binary(*queries, mlp, nvinfer1::ElementWiseOperation::kSUM,
                              name + ".mlp.residual", true);
            queries = &layers.layerNormFp32(*queries, prefix + ".norm3", 256,
                                            kTransformerLayerNormEpsilon, name + ".norm3");

            nvinfer1::ITensor& image_query =
                binary(*keys, key_position, nvinfer1::ElementWiseOperation::kSUM,
                       name + ".image_to_token.query", true);
            nvinfer1::ITensor& token_key =
                binary(*queries, query_position, nvinfer1::ElementWiseOperation::kSUM,
                       name + ".image_to_token.key", true);
            nvinfer1::ITensor& image_attended =
                attention(image_query, token_key, *queries, prefix + ".cross_attn_image_to_token",
                          256, 256, 128, 8, name + ".image_to_token.attention");
            keys = &binary(*keys, image_attended, nvinfer1::ElementWiseOperation::kSUM,
                           name + ".image_to_token.residual", true);
            keys = &layers.layerNormFp32(*keys, prefix + ".norm4", 256,
                                         kTransformerLayerNormEpsilon, name + ".norm4");
        }
        nvinfer1::ITensor& final_q =
            binary(*queries, query_position, nvinfer1::ElementWiseOperation::kSUM,
                   "decoder.final.query", true);
        nvinfer1::ITensor& final_k = binary(
            *keys, key_position, nvinfer1::ElementWiseOperation::kSUM, "decoder.final.key", true);
        nvinfer1::ITensor& final_attention = attention(
            final_q, final_k, *keys, "sam_mask_decoder.transformer.final_attn_token_to_image", 256,
            256, 128, 8, "decoder.final.attention");
        queries = &binary(*queries, final_attention, nvinfer1::ElementWiseOperation::kSUM,
                          "decoder.final.residual", true);
        queries = &layers.layerNormFp32(*queries, "sam_mask_decoder.transformer.norm_final_attn",
                                        256, kTransformerLayerNormEpsilon, "decoder.final.norm");
        if (queries->getDimensions().d[1] != decoder_tokens)
            throw NetworkBuildError("SAM2 decoder token extent drifted");
        return {queries, keys};
    }

    nvinfer1::ITensor& topCandidateIndex(nvinfer1::ITensor& scores, std::string_view name) {
#if NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR < 14
        auto* topk = network.addTopK(scores, nvinfer1::TopKOperation::kMAX, 1, 1U << 1U);
#else
        auto* topk = network.addTopK(scores, nvinfer1::TopKOperation::kMAX, 1, 1U << 1U,
                                     nvinfer1::DataType::kINT32);
#endif
        requireLayer(topk, "top-k", name);
        return *topk->getOutput(1);
    }

    nvinfer1::ITensor& selectCandidate(nvinfer1::ITensor& candidates, nvinfer1::ITensor& indices,
                                       std::int32_t candidate_count,
                                       std::initializer_list<std::int64_t> candidate_shape,
                                       std::initializer_list<std::int64_t> condition_shape,
                                       std::string_view name) {
        nvinfer1::ITensor* result = nullptr;
        for (std::int32_t index = 0; index < candidate_count; ++index) {
            std::vector<std::int64_t> start(candidate_shape.size(), 0);
            std::vector<std::int64_t> size(candidate_shape.begin(), candidate_shape.end());
            std::vector<std::int64_t> stride(candidate_shape.size(), 1);
            start[1] = index;
            size[1] = 1;
            auto as_list = [](const std::vector<std::int64_t>& values) {
                nvinfer1::Dims result{};
                result.nbDims = static_cast<std::int32_t>(values.size());
                for (std::int32_t i = 0; i < result.nbDims; ++i)
                    result.d[i] = static_cast<std::int32_t>(values[static_cast<std::size_t>(i)]);
                return result;
            };
            auto* slice_layer =
                network.addSlice(candidates, as_list(start), as_list(size), as_list(stride));
            requireLayer(slice_layer, "candidate slice",
                         std::string(name) + ".candidate." + std::to_string(index));
            nvinfer1::ITensor* candidate = slice_layer->getOutput(0);
            if (result == nullptr) {
                result = candidate;
                continue;
            }
            nvinfer1::ITensor& expected =
                intConstant(index, {1, 1}, std::string(name) + ".index." + std::to_string(index));
            nvinfer1::ITensor& condition =
                layers.elementWise(indices, expected, nvinfer1::ElementWiseOperation::kEQUAL,
                                   std::string(name) + ".equal." + std::to_string(index));
            nvinfer1::ITensor& expanded =
                layers.shuffle(condition, condition_shape,
                               std::string(name) + ".condition." + std::to_string(index));
            result = &select(expanded, *candidate, *result,
                             std::string(name) + ".select." + std::to_string(index));
        }
        if (result == nullptr)
            throw NetworkBuildError("SAM2 candidate selector has no candidates");
        return *result;
    }

    struct DecoderOutput {
        nvinfer1::ITensor* mask{nullptr};
        nvinfer1::ITensor* object_pointer{nullptr};
        nvinfer1::ITensor* object_present{nullptr};
        nvinfer1::ITensor* object_score{nullptr};
    };

    DecoderOutput decode(nvinfer1::ITensor& image, nvinfer1::ITensor& fpn0, nvinfer1::ITensor& fpn1,
                         nvinfer1::ITensor& sparse, const TrackerPlanSpec& plan) {
        nvinfer1::ITensor& no_mask = checkpointConstant(
            "sam_prompt_encoder.no_mask_embed.weight", {1, 256}, {1, 256, 1, 1}, "decoder.no_mask");
        nvinfer1::ITensor& source = binary(image, no_mask, nvinfer1::ElementWiseOperation::kSUM,
                                           "decoder.image_with_no_mask", true);
        nvinfer1::ITensor& dense_position = makeDenseRandomPosition();
        nvinfer1::ITensor& object_token =
            checkpointConstant("sam_mask_decoder.obj_score_token.weight", {1, 256}, {1, 1, 256},
                               "decoder.object_token");
        nvinfer1::ITensor& iou_token = checkpointConstant(
            "sam_mask_decoder.iou_token.weight", {1, 256}, {1, 1, 256}, "decoder.iou_token");
        nvinfer1::ITensor& mask_tokens = checkpointConstant(
            "sam_mask_decoder.mask_tokens.weight", {4, 256}, {1, 4, 256}, "decoder.mask_tokens");
        nvinfer1::ITensor& tokens = layers.concatenate(
            {&object_token, &iou_token, &mask_tokens, &sparse}, 1, "decoder.input_tokens");
        const auto transformed =
            twoWayTransformer(source, dense_position, tokens, plan.decoder_tokens);
        nvinfer1::ITensor& states = *transformed.first;
        nvinfer1::ITensor& transformed_image = *transformed.second;
        nvinfer1::ITensor& object_state_rank3 =
            layers.slice(states, {0, 0, 0}, {1, 1, 256}, {1, 1, 1},
                         nvinfer1::SampleMode::kSTRICT_BOUNDS, "decoder.object_state");
        nvinfer1::ITensor& iou_state_rank3 =
            layers.slice(states, {0, 1, 0}, {1, 1, 256}, {1, 1, 1},
                         nvinfer1::SampleMode::kSTRICT_BOUNDS, "decoder.iou_state");
        nvinfer1::ITensor& mask_states =
            layers.slice(states, {0, 2, 0}, {1, 4, 256}, {1, 1, 1},
                         nvinfer1::SampleMode::kSTRICT_BOUNDS, "decoder.mask_states");

        nvinfer1::ITensor& image_token_channels =
            layers.transpose(transformed_image, {0, 2, 1}, "decoder.image.channels");
        nvinfer1::ITensor& image_nchw =
            layers.shuffle(image_token_channels, {1, 256, 64, 64}, "decoder.image.nchw");
        nvinfer1::ITensor& up1 = deconv(image_nchw, "sam_mask_decoder.output_upscaling.0", 256, 64,
                                        2, 2, "decoder.upscale.0");
        nvinfer1::ITensor& high1 =
            conv(fpn1, "sam_mask_decoder.conv_s1", 256, 64, 1, 1, 0, 1, "decoder.high_res.1");
        nvinfer1::ITensor& merged1 =
            binary(up1, high1, nvinfer1::ElementWiseOperation::kSUM, "decoder.upscale.0.skip");
        nvinfer1::ITensor& norm1 = layerNorm2d(merged1, "sam_mask_decoder.output_upscaling.1", 64,
                                               "decoder.upscale.0.norm");
        nvinfer1::ITensor& act1 = layers.gelu(norm1, "decoder.upscale.0.gelu");
        nvinfer1::ITensor& up2 =
            deconv(act1, "sam_mask_decoder.output_upscaling.3", 64, 32, 2, 2, "decoder.upscale.1");
        nvinfer1::ITensor& high0 =
            conv(fpn0, "sam_mask_decoder.conv_s0", 256, 32, 1, 1, 0, 1, "decoder.high_res.0");
        nvinfer1::ITensor& merged2 =
            binary(up2, high0, nvinfer1::ElementWiseOperation::kSUM, "decoder.upscale.1.skip");
        nvinfer1::ITensor& upscaled = layers.gelu(merged2, "decoder.upscale.1.gelu");

        std::vector<nvinfer1::ITensor*> hypernetworks;
        for (std::int32_t token = 0; token < 4; ++token) {
            nvinfer1::ITensor& state =
                layers.slice(mask_states, {0, token, 0}, {1, 1, 256}, {1, 1, 1},
                             nvinfer1::SampleMode::kSTRICT_BOUNDS,
                             "decoder.hyper." + std::to_string(token) + ".state");
            const std::string module =
                "sam_mask_decoder.output_hypernetworks_mlps." + std::to_string(token) + ".layers.";
            nvinfer1::ITensor& hidden0 = linearRelu(
                state, module + "0", 256, 256, "decoder.hyper." + std::to_string(token) + ".relu0");
            nvinfer1::ITensor& hidden1 =
                linearRelu(hidden0, module + "1", 256, 256,
                           "decoder.hyper." + std::to_string(token) + ".relu1");
            hypernetworks.push_back(
                &layers.linearAutocastBf16(hidden1, module + "2", 256, 32,
                                           "decoder.hyper." + std::to_string(token) + ".output"));
        }
        nvinfer1::ITensor& hyper = layers.concatenate(hypernetworks, 1, "decoder.hyper.stack");
        nvinfer1::ITensor& upscaled_flat =
            layers.shuffle(upscaled, {1, 32, 256 * 256}, "decoder.upscaled.flatten");
        nvinfer1::ITensor& masks_flat =
            layers.matrixMultiply(hyper, nvinfer1::MatrixOperation::kNONE, upscaled_flat,
                                  nvinfer1::MatrixOperation::kNONE, "decoder.masks.matmul");
        nvinfer1::ITensor& all_masks =
            layers.shuffle(masks_flat, {1, 4, 256, 256}, "decoder.masks.reshape");

        nvinfer1::ITensor& iou_hidden0 =
            linearRelu(iou_state_rank3, "sam_mask_decoder.iou_prediction_head.layers.0", 256, 256,
                       "decoder.iou.relu0");
        nvinfer1::ITensor& iou_hidden1 =
            linearRelu(iou_hidden0, "sam_mask_decoder.iou_prediction_head.layers.1", 256, 256,
                       "decoder.iou.relu1");
        nvinfer1::ITensor& iou_rank3 =
            layers.linearAutocastBf16(iou_hidden1, "sam_mask_decoder.iou_prediction_head.layers.2",
                                      256, 4, "decoder.iou.output");
        nvinfer1::ITensor& iou = layers.shuffle(iou_rank3, {1, 4}, "decoder.iou.squeeze");
        nvinfer1::ITensor& iou_probability =
            activation(iou, nvinfer1::ActivationType::kSIGMOID, "decoder.iou.sigmoid");

        nvinfer1::ITensor& object_hidden0 =
            linearRelu(object_state_rank3, "sam_mask_decoder.pred_obj_score_head.layers.0", 256,
                       256, "decoder.object.relu0");
        nvinfer1::ITensor& object_hidden1 =
            linearRelu(object_hidden0, "sam_mask_decoder.pred_obj_score_head.layers.1", 256, 256,
                       "decoder.object.relu1");
        nvinfer1::ITensor& object_score_rank3 = layers.linearAutocastBf16(
            object_hidden1, "sam_mask_decoder.pred_obj_score_head.layers.2", 256, 1,
            "decoder.object.output");
        nvinfer1::ITensor& object_score =
            layers.cast(layers.shuffle(object_score_rank3, {1, 1}, "decoder.object.squeeze"),
                        nvinfer1::DataType::kFLOAT, "decoder.object.fp32");
        nvinfer1::ITensor& zero =
            layers.scalar(0.0F, 2, nvinfer1::DataType::kFLOAT, "decoder.object.zero");
        nvinfer1::ITensor& object_present = layers.elementWise(
            object_score, zero, nvinfer1::ElementWiseOperation::kGREATER, "decoder.object.present");
        nvinfer1::ITensor& all_masks_fp32 =
            layers.cast(all_masks, nvinfer1::DataType::kFLOAT, "decoder.masks.fp32");

        nvinfer1::ITensor* selected_mask = nullptr;
        nvinfer1::ITensor* selected_state = nullptr;
        if (!plan.multimask_output) {
            nvinfer1::ITensor& stability_mask =
                layers.slice(all_masks, {0, 0, 0, 0}, {1, 1, 256, 256}, {1, 1, 1, 1},
                             nvinfer1::SampleMode::kSTRICT_BOUNDS, "decoder.stability.mask_bf16");
            nvinfer1::ITensor& single_mask =
                layers.slice(all_masks_fp32, {0, 0, 0, 0}, {1, 1, 256, 256}, {1, 1, 1, 1},
                             nvinfer1::SampleMode::kSTRICT_BOUNDS, "decoder.mask.single");
            nvinfer1::ITensor& candidate_masks = layers.slice(
                all_masks_fp32, {0, 1, 0, 0}, {1, 3, 256, 256}, {1, 1, 1, 1},
                nvinfer1::SampleMode::kSTRICT_BOUNDS, "decoder.mask.fallback_candidates");
            nvinfer1::ITensor& candidate_iou = layers.slice(iou_probability, {0, 1}, {1, 3}, {1, 1},
                                                            nvinfer1::SampleMode::kSTRICT_BOUNDS,
                                                            "decoder.iou.fallback_candidates");
            nvinfer1::ITensor& indices =
                topCandidateIndex(candidate_iou, "decoder.iou.fallback_argmax");
            nvinfer1::ITensor& best_multimask =
                selectCandidate(candidate_masks, indices, 3, {1, 3, 256, 256}, {1, 1, 1, 1},
                                "decoder.mask.fallback_best");

            nvinfer1::ITensor& upper =
                layers.scalar(0.05F, 4, nvinfer1::DataType::kBF16, "decoder.stability.upper");
            nvinfer1::ITensor& lower =
                layers.scalar(-0.05F, 4, nvinfer1::DataType::kBF16, "decoder.stability.lower");
            nvinfer1::ITensor& intersection_bool =
                layers.elementWise(stability_mask, upper, nvinfer1::ElementWiseOperation::kGREATER,
                                   "decoder.stability.intersection_bool");
            nvinfer1::ITensor& union_bool =
                layers.elementWise(stability_mask, lower, nvinfer1::ElementWiseOperation::kGREATER,
                                   "decoder.stability.union_bool");
            nvinfer1::ITensor& intersection =
                reduce(layers.cast(intersection_bool, nvinfer1::DataType::kFLOAT,
                                   "decoder.stability.intersection_fp32"),
                       nvinfer1::ReduceOperation::kSUM, (1U << 2U) | (1U << 3U), false,
                       "decoder.stability.intersection");
            nvinfer1::ITensor& union_area = reduce(
                layers.cast(union_bool, nvinfer1::DataType::kFLOAT, "decoder.stability.union_fp32"),
                nvinfer1::ReduceOperation::kSUM, (1U << 2U) | (1U << 3U), false,
                "decoder.stability.union");
            nvinfer1::ITensor& ratio =
                layers.elementWise(intersection, union_area, nvinfer1::ElementWiseOperation::kDIV,
                                   "decoder.stability.ratio");
            nvinfer1::ITensor& area_zero =
                layers.scalar(0.0F, 2, nvinfer1::DataType::kFLOAT, "decoder.stability.area_zero");
            nvinfer1::ITensor& nonempty =
                layers.elementWise(union_area, area_zero, nvinfer1::ElementWiseOperation::kGREATER,
                                   "decoder.stability.nonempty");
            nvinfer1::ITensor& one =
                layers.scalar(1.0F, 2, nvinfer1::DataType::kFLOAT, "decoder.stability.one");
            nvinfer1::ITensor& stability = select(nonempty, ratio, one, "decoder.stability.score");
            nvinfer1::ITensor& threshold =
                layers.scalar(0.98F, 2, nvinfer1::DataType::kFLOAT, "decoder.stability.threshold");
            nvinfer1::ITensor& unstable =
                layers.elementWise(stability, threshold, nvinfer1::ElementWiseOperation::kLESS,
                                   "decoder.stability.unstable");
            nvinfer1::ITensor& unstable_mask =
                layers.shuffle(unstable, {1, 1, 1, 1}, "decoder.stability.unstable_mask");
            selected_mask = &select(unstable_mask, best_multimask, single_mask,
                                    "decoder.mask.dynamic_fallback");
            selected_state =
                &layers.slice(mask_states, {0, 0, 0}, {1, 1, 256}, {1, 1, 1},
                              nvinfer1::SampleMode::kSTRICT_BOUNDS, "decoder.pointer.single_state");
        } else {
            nvinfer1::ITensor& candidate_masks =
                layers.slice(all_masks_fp32, {0, 1, 0, 0}, {1, 3, 256, 256}, {1, 1, 1, 1},
                             nvinfer1::SampleMode::kSTRICT_BOUNDS, "decoder.mask.candidates");
            nvinfer1::ITensor& candidate_iou =
                layers.slice(iou_probability, {0, 1}, {1, 3}, {1, 1},
                             nvinfer1::SampleMode::kSTRICT_BOUNDS, "decoder.iou.candidates");
            nvinfer1::ITensor& candidate_states =
                layers.slice(mask_states, {0, 1, 0}, {1, 3, 256}, {1, 1, 1},
                             nvinfer1::SampleMode::kSTRICT_BOUNDS, "decoder.pointer.candidates");
            nvinfer1::ITensor& indices = topCandidateIndex(candidate_iou, "decoder.iou.argmax");
            selected_mask = &selectCandidate(candidate_masks, indices, 3, {1, 3, 256, 256},
                                             {1, 1, 1, 1}, "decoder.mask.best");
            selected_state = &selectCandidate(candidate_states, indices, 3, {1, 3, 256}, {1, 1, 1},
                                              "decoder.pointer.best");
        }

        nvinfer1::ITensor& object_present_rank4 =
            layers.shuffle(object_present, {1, 1, 1, 1}, "decoder.object.present_mask");
        nvinfer1::ITensor& no_object_mask =
            layers.scalar(-1024.0F, 4, nvinfer1::DataType::kFLOAT, "decoder.mask.no_object");
        selected_mask = &select(object_present_rank4, *selected_mask, no_object_mask,
                                "decoder.mask.object_gate");

        nvinfer1::ITensor& pointer_state =
            layers.shuffle(*selected_state, {1, 256}, "decoder.pointer.state");
        nvinfer1::ITensor& pointer0 =
            linearRelu(pointer_state, "obj_ptr_proj.layers.0", 256, 256, "decoder.pointer.relu0");
        nvinfer1::ITensor& pointer1 =
            linearRelu(pointer0, "obj_ptr_proj.layers.1", 256, 256, "decoder.pointer.relu1");
        nvinfer1::ITensor& pointer2 = layers.linearAutocastBf16(pointer1, "obj_ptr_proj.layers.2",
                                                                256, 256, "decoder.pointer.output");
        nvinfer1::ITensor& pointer_fp32 =
            layers.cast(pointer2, nvinfer1::DataType::kFLOAT, "decoder.pointer.fp32");
        nvinfer1::ITensor& no_object_pointer =
            checkpointConstant("no_obj_ptr", {1, 256}, {1, 256}, "decoder.pointer.no_object");
        nvinfer1::ITensor& gated_pointer =
            select(object_present, pointer_fp32, no_object_pointer, "decoder.pointer.object_gate");
        return {selected_mask, &gated_pointer, &object_present, &object_score};
    }

    nvinfer1::ITensor& encodeMemory(nvinfer1::ITensor& current_image,
                                    nvinfer1::ITensor& low_resolution_mask,
                                    nvinfer1::ITensor& object_present,
                                    bool binarize_interacted_mask) {
        nvinfer1::ITensor& high_resolution_mask = layers.resizeNchw(
            low_resolution_mask, 1024, 1024, nvinfer1::InterpolationMode::kLINEAR,
            nvinfer1::ResizeCoordinateTransformation::kHALF_PIXEL, "memory.mask.resize");
        nvinfer1::ITensor* probability = nullptr;
        if (binarize_interacted_mask) {
            nvinfer1::ITensor& zero =
                layers.scalar(0.0F, 4, nvinfer1::DataType::kFLOAT, "memory.mask.binary_zero");
            nvinfer1::ITensor& foreground = layers.elementWise(
                high_resolution_mask, zero, nvinfer1::ElementWiseOperation::kGREATER,
                "memory.mask.binary_foreground");
            probability =
                &layers.cast(foreground, nvinfer1::DataType::kFLOAT, "memory.mask.binary_fp32");
        } else {
            probability = &activation(high_resolution_mask, nvinfer1::ActivationType::kSIGMOID,
                                      "memory.mask.sigmoid");
        }
        nvinfer1::ITensor& scaled = scale(*probability, 20.0F, "memory.mask.scale");
        nvinfer1::ITensor& bias = layers.scalar(-10.0F, 4, scaled.getType(), "memory.mask.bias");
        nvinfer1::ITensor* mask = &layers.elementWise(
            scaled, bias, nvinfer1::ElementWiseOperation::kSUM, "memory.mask.scaled_bias");
        constexpr std::array<std::int32_t, 4> channels = {4, 16, 64, 256};
        constexpr std::array<std::int32_t, 4> conv_indices = {0, 3, 6, 9};
        constexpr std::array<std::int32_t, 4> norm_indices = {1, 4, 7, 10};
        std::int32_t input_channels = 1;
        for (std::size_t index = 0; index < channels.size(); ++index) {
            const std::string base = "memory_encoder.mask_downsampler.encoder.";
            nvinfer1::ITensor& convolved = conv(
                *mask, base + std::to_string(conv_indices[index]), input_channels, channels[index],
                3, 2, 1, 1, "memory.downsample." + std::to_string(index) + ".conv");
            nvinfer1::ITensor& normalized =
                layerNorm2d(convolved, base + std::to_string(norm_indices[index]), channels[index],
                            "memory.downsample." + std::to_string(index) + ".norm");
            mask = &layers.gelu(normalized, "memory.downsample." + std::to_string(index) + ".gelu");
            input_channels = channels[index];
        }
        mask = &conv(*mask, "memory_encoder.mask_downsampler.encoder.12", 256, 256, 1, 1, 0, 1,
                     "memory.downsample.project");
        nvinfer1::ITensor& projected_image = conv(current_image, "memory_encoder.pix_feat_proj",
                                                  256, 256, 1, 1, 0, 1, "memory.image.project");
        nvinfer1::ITensor* fused = &binary(
            projected_image, *mask, nvinfer1::ElementWiseOperation::kSUM, "memory.fuser.input");
        for (std::int32_t index = 0; index < 2; ++index) {
            const std::string module = "memory_encoder.fuser.layers." + std::to_string(index);
            const std::string name = "memory.fuser." + std::to_string(index);
            nvinfer1::ITensor& depthwise =
                conv(*fused, module + ".dwconv", 256, 256, 7, 1, 3, 256, name + ".depthwise");
            nvinfer1::ITensor& normalized =
                layerNorm2d(depthwise, module + ".norm", 256, name + ".norm");
            nvinfer1::ITensor& nhwc = layers.transpose(normalized, {0, 2, 3, 1}, name + ".nhwc");
            nvinfer1::ITensor& expanded =
                layers.linearAutocastBf16(nhwc, module + ".pwconv1", 256, 1024, name + ".expand");
            nvinfer1::ITensor& hidden = layers.gelu(expanded, name + ".gelu");
            nvinfer1::ITensor& projected = layers.linearAutocastBf16(hidden, module + ".pwconv2",
                                                                     1024, 256, name + ".project");
            nvinfer1::ITensor& gamma =
                checkpointConstant(module + ".gamma", {256}, {1, 1, 1, 256}, name + ".gamma");
            nvinfer1::ITensor& scaled_block = binary(
                projected, gamma, nvinfer1::ElementWiseOperation::kPROD, name + ".scale", true);
            nvinfer1::ITensor& nchw = layers.transpose(scaled_block, {0, 3, 1, 2}, name + ".nchw");
            fused = &binary(*fused, nchw, nvinfer1::ElementWiseOperation::kSUM, name + ".residual",
                            true);
        }
        nvinfer1::ITensor& compressed =
            conv(*fused, "memory_encoder.out_proj", 256, 64, 1, 1, 0, 1, "memory.output.project");
        nvinfer1::ITensor& compressed_fp32 =
            layers.cast(compressed, nvinfer1::DataType::kFLOAT, "memory.output.fp32");
        nvinfer1::ITensor& visible =
            layers.shuffle(object_present, {1, 1, 1, 1}, "memory.object.visible");
        nvinfer1::ITensor& no_object_embedding = checkpointConstant(
            "no_obj_embed_spatial", {1, 64}, {1, 64, 1, 1}, "memory.object.embedding");
        nvinfer1::ITensor& zero =
            layers.scalar(0.0F, 4, nvinfer1::DataType::kFLOAT, "memory.object.zero");
        nvinfer1::ITensor& occlusion =
            select(visible, zero, no_object_embedding, "memory.object.select_embedding");
        nvinfer1::ITensor& result =
            binary(compressed_fp32, occlusion, nvinfer1::ElementWiseOperation::kSUM,
                   "memory.output.with_object_state", true);
        return layers.cast(result, nvinfer1::DataType::kBF16, "memory.output.bf16");
    }

    nvinfer1::ITensor& memoryAttention(nvinfer1::ITensor& current,
                                       nvinfer1::ITensor& current_position,
                                       nvinfer1::ITensor& history_memory,
                                       nvinfer1::ITensor& history_pointers,
                                       const TrackerPlanSpec& plan) {
        std::vector<nvinfer1::ITensor*> memory_features;
        std::vector<nvinfer1::ITensor*> memory_positions;
        nvinfer1::ITensor& spatial_position_fp32 =
            makeSinePosition(64, "attention.memory.spatial_fp32");
        nvinfer1::ITensor& spatial_position = layers.cast(
            spatial_position_fp32, nvinfer1::DataType::kBF16, "attention.memory.spatial_bf16");
        nvinfer1::ITensor& temporal_table = checkpointConstant(
            "maskmem_tpos_enc", {7, 1, 1, 64}, {7, 1, 1, 64}, "attention.memory.temporal_table");
        for (std::int32_t index = 0; index < plan.history_frames; ++index) {
            nvinfer1::ITensor& feature_nchw =
                layers.slice(history_memory, {index, 0, 0, 0}, {1, 64, 64, 64}, {1, 1, 1, 1},
                             nvinfer1::SampleMode::kSTRICT_BOUNDS,
                             "attention.memory.feature." + std::to_string(index));
            nvinfer1::ITensor& feature_flat =
                layers.shuffle(feature_nchw, {1, 64, kImageTokens},
                               "attention.memory.feature_flat." + std::to_string(index));
            memory_features.push_back(
                &layers.transpose(feature_flat, {0, 2, 1},
                                  "attention.memory.feature_tokens." + std::to_string(index)));

            const std::int32_t row =
                plan.memory_temporal_embedding_rows[static_cast<std::size_t>(index)];
            nvinfer1::ITensor& temporal_nhwc =
                layers.slice(temporal_table, {row, 0, 0, 0}, {1, 1, 1, 64}, {1, 1, 1, 1},
                             nvinfer1::SampleMode::kSTRICT_BOUNDS,
                             "attention.memory.temporal." + std::to_string(index));
            nvinfer1::ITensor& temporal =
                layers.transpose(temporal_nhwc, {0, 3, 1, 2},
                                 "attention.memory.temporal_nchw." + std::to_string(index));
            nvinfer1::ITensor& positioned =
                binary(spatial_position, temporal, nvinfer1::ElementWiseOperation::kSUM,
                       "attention.memory.position." + std::to_string(index), true);
            nvinfer1::ITensor& position_flat =
                layers.shuffle(positioned, {1, 64, kImageTokens},
                               "attention.memory.position_flat." + std::to_string(index));
            memory_positions.push_back(
                &layers.transpose(position_flat, {0, 2, 1},
                                  "attention.memory.position_tokens." + std::to_string(index)));
        }

        std::vector<nvinfer1::ITensor*> pointer_features;
        std::vector<nvinfer1::ITensor*> pointer_positions;
        for (std::int32_t index = 0; index < plan.history_frames; ++index) {
            const std::int32_t frame =
                plan.object_pointer_frame_order[static_cast<std::size_t>(index)];
            nvinfer1::ITensor& pointer =
                layers.slice(history_pointers, {frame, 0}, {1, 256}, {1, 1},
                             nvinfer1::SampleMode::kSTRICT_BOUNDS,
                             "attention.pointer.feature." + std::to_string(index));
            pointer_features.push_back(&layers.shuffle(
                pointer, {1, 4, 64}, "attention.pointer.tokens." + std::to_string(index)));
            nvinfer1::ITensor& raw_position = makePointerTemporalPosition(
                plan.object_pointer_temporal_distances[static_cast<std::size_t>(index)],
                "attention.pointer.raw_position." + std::to_string(index));
            nvinfer1::ITensor& projected_position = layers.linearAutocastBf16(
                raw_position, "obj_ptr_tpos_proj", 256, 64,
                "attention.pointer.project_position." + std::to_string(index));
            nvinfer1::ITensor& position_token =
                layers.shuffle(projected_position, {1, 1, 64},
                               "attention.pointer.position_token." + std::to_string(index));
            pointer_positions.push_back(&layers.concatenate(
                {&position_token, &position_token, &position_token, &position_token}, 1,
                "attention.pointer.positions." + std::to_string(index)));
        }
        nvinfer1::ITensor& spatial_memory =
            layers.concatenate(memory_features, 1, "attention.memory.spatial_features");
        nvinfer1::ITensor& spatial_positions =
            layers.concatenate(memory_positions, 1, "attention.memory.spatial_positions");
        nvinfer1::ITensor& pointers =
            layers.concatenate(pointer_features, 1, "attention.memory.pointer_features");
        nvinfer1::ITensor& pointer_position =
            layers.concatenate(pointer_positions, 1, "attention.memory.pointer_positions");
        nvinfer1::ITensor& memory =
            layers.concatenate({&layers.cast(spatial_memory, nvinfer1::DataType::kFLOAT,
                                             "attention.memory.features_fp32"),
                                &pointers},
                               1, "attention.memory.features");
        nvinfer1::ITensor& position = layers.concatenate(
            {&spatial_positions, &layers.cast(pointer_position, nvinfer1::DataType::kFLOAT,
                                              "attention.memory.pointer_position_fp32")},
            1, "attention.memory.positions");

        nvinfer1::ITensor& current_flat =
            layers.shuffle(current, {1, 256, kImageTokens}, "attention.current.flatten");
        nvinfer1::ITensor* output =
            &layers.transpose(current_flat, {0, 2, 1}, "attention.current.tokens");
        nvinfer1::ITensor& current_position_flat = layers.shuffle(
            current_position, {1, 256, kImageTokens}, "attention.current_position.flatten");
        nvinfer1::ITensor& current_position_tokens =
            layers.transpose(current_position_flat, {0, 2, 1}, "attention.current_position.tokens");
        nvinfer1::ITensor& scaled_position =
            scale(current_position_tokens, 0.1F, "attention.current_position.scale");
        output = &binary(*output, scaled_position, nvinfer1::ElementWiseOperation::kSUM,
                         "attention.input_with_position", true);
        const std::int32_t spatial_tokens = plan.history_frames * kImageTokens;
        for (std::int32_t index = 0; index < 4; ++index) {
            const std::string module = "memory_attention.layers." + std::to_string(index);
            const std::string name = "attention.layer." + std::to_string(index);
            nvinfer1::ITensor& self_norm = layers.layerNormFp32(
                *output, module + ".norm1", 256, kTransformerLayerNormEpsilon, name + ".self.norm");
            nvinfer1::ITensor& self_attended =
                attention(self_norm, self_norm, self_norm, module + ".self_attn", 256, 256, 256, 1,
                          name + ".self.attention", true, kImageTokens);
            output = &binary(*output, self_attended, nvinfer1::ElementWiseOperation::kSUM,
                             name + ".self.residual", true);
            nvinfer1::ITensor& cross_norm =
                layers.layerNormFp32(*output, module + ".norm2", 256, kTransformerLayerNormEpsilon,
                                     name + ".cross.norm");
            nvinfer1::ITensor& positioned_memory =
                binary(memory, position, nvinfer1::ElementWiseOperation::kSUM,
                       name + ".cross.positioned_memory", true);
            nvinfer1::ITensor& cross_attended =
                attention(cross_norm, positioned_memory, memory, module + ".cross_attn_image", 256,
                          64, 256, 1, name + ".cross.attention", true, spatial_tokens);
            output = &binary(*output, cross_attended, nvinfer1::ElementWiseOperation::kSUM,
                             name + ".cross.residual", true);
            nvinfer1::ITensor& mlp_norm = layers.layerNormFp32(
                *output, module + ".norm3", 256, kTransformerLayerNormEpsilon, name + ".mlp.norm");
            nvinfer1::ITensor& hidden =
                linearRelu(mlp_norm, module + ".linear1", 256, 2048, name + ".mlp.relu");
            nvinfer1::ITensor& projected = layers.linearAutocastBf16(
                hidden, module + ".linear2", 2048, 256, name + ".mlp.output");
            output = &binary(*output, projected, nvinfer1::ElementWiseOperation::kSUM,
                             name + ".mlp.residual", true);
        }
        output = &layers.layerNormFp32(*output, "memory_attention.norm", 256,
                                       kTransformerLayerNormEpsilon, "attention.output.norm");
        nvinfer1::ITensor& channel_major =
            layers.transpose(*output, {0, 2, 1}, "attention.output.channel_major");
        return layers.shuffle(channel_major, {1, 256, 64, 64}, "attention.output.nchw");
    }

    void validateNetwork() const {
#if NV_TENSORRT_MAJOR == 10
        if (!network.getFlag(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED))
            throw NetworkBuildError("SAM2 tracker requires a strongly typed TensorRT network");
#elif NV_TENSORRT_MAJOR < 10
#error "The native SAM2 tracker builder requires TensorRT 10 or newer"
#endif
        if (built)
            throw NetworkBuildError("SAM2 tracker network builder is single-use");
        if (network.getNbInputs() != 0 || network.getNbOutputs() != 0 ||
            network.getNbLayers() != 0) {
            throw NetworkBuildError("SAM2 tracker construction requires an empty network");
        }
        validateTrackerCheckpoint(checkpoint);
    }

    void markOutput(nvinfer1::ITensor& tensor, const TensorContract& contract) {
        if (tensor.getType() != dataType(contract.data_type) ||
            !sameDimensions(tensor.getDimensions(), contract)) {
            throw NetworkBuildError("SAM2 tracker output contract mismatch for " +
                                    std::string(contract.name) + ": actual " +
                                    dimensionsString(tensor.getDimensions()));
        }
        tensor.setName(contract.name.data());
        network.markOutput(tensor);
    }

    Sam2TrackerNetworkOutputs build(const TrackerPlanSpec& plan) {
        validateNetwork();
        built = true;
        nvinfer1::ITensor& fpn0 = addInput(kTrackerFpn[0]);
        nvinfer1::ITensor& fpn1 = addInput(kTrackerFpn[1]);
        nvinfer1::ITensor& fpn2 = addInput(kTrackerFpn[2]);
        nvinfer1::ITensor* box = nullptr;
        nvinfer1::ITensor* history_memory = nullptr;
        nvinfer1::ITensor* history_pointers = nullptr;
        if (plan.kind == TrackerPlanKind::kPrompt) {
            box = &addInput(kBoxPrompt);
        } else {
            history_memory = &addInput(historyMemoryFeatures(plan.history_frames));
            history_pointers = &addInput(historyObjectPointers(plan.history_frames));
        }
        nvinfer1::ITensor& current_position = makeSinePosition(256, "tracker.current_position");
        nvinfer1::ITensor* decoder_image = &fpn2;
        if (plan.kind == TrackerPlanKind::kPrompt) {
            nvinfer1::ITensor& no_memory = checkpointConstant("no_mem_embed", {1, 1, 256},
                                                              {1, 256, 1, 1}, "tracker.no_memory");
            decoder_image = &binary(fpn2, no_memory, nvinfer1::ElementWiseOperation::kSUM,
                                    "tracker.prompt.no_memory", true);
        } else {
            decoder_image =
                &memoryAttention(fpn2, current_position, *history_memory, *history_pointers, plan);
        }
        nvinfer1::ITensor& sparse = buildSparsePrompt(box, plan.kind);
        DecoderOutput decoded = decode(*decoder_image, fpn0, fpn1, sparse, plan);
        nvinfer1::ITensor& mask =
            layers.cast(*decoded.mask, nvinfer1::DataType::kFLOAT, "tracker.output.mask_fp32");
        nvinfer1::ITensor& pointer = layers.cast(
            *decoded.object_pointer, nvinfer1::DataType::kFLOAT, "tracker.output.pointer_fp32");
        nvinfer1::ITensor& memory = encodeMemory(fpn2, mask, *decoded.object_present,
                                                 plan.kind == TrackerPlanKind::kPrompt);
        markOutput(mask, kMaskLogits256);
        markOutput(pointer, kObjectPointer);
        markOutput(memory, kMemoryFeatures);

        Sam2TrackerNetworkOutputs result;
        result.mask_logits_256 = &mask;
        result.object_pointer = &pointer;
        result.memory_features = &memory;
        result.referenced_tensor_count =
            layers.referencedTensorCount() + directly_referenced_weights.size();
        result.added_layer_count = network.getNbLayers();
        return result;
    }

    nvinfer1::INetworkDefinition& network;
    const CheckpointReader& checkpoint;
    TrtLayers layers;
    std::deque<std::vector<float>> owned_float_constants;
    std::deque<std::vector<std::int32_t>> owned_int_constants;
    std::deque<std::vector<std::uint16_t>> owned_bfloat16_weights;
    std::deque<std::vector<float>> owned_transposed_convolution_weights;
    std::set<std::string> directly_referenced_weights;
    bool built{false};
};

Sam2TrackerNetworkBuilder::Sam2TrackerNetworkBuilder(nvinfer1::INetworkDefinition& network,
                                                     const CheckpointReader& checkpoint)
    : impl_(std::make_unique<Impl>(network, checkpoint)) {}

Sam2TrackerNetworkBuilder::~Sam2TrackerNetworkBuilder() = default;
Sam2TrackerNetworkBuilder::Sam2TrackerNetworkBuilder(Sam2TrackerNetworkBuilder&&) noexcept =
    default;
Sam2TrackerNetworkBuilder&
Sam2TrackerNetworkBuilder::operator=(Sam2TrackerNetworkBuilder&&) noexcept = default;

Sam2TrackerNetworkOutputs Sam2TrackerNetworkBuilder::buildPrompt() {
    if (!impl_)
        throw NetworkBuildError("SAM2 tracker builder was moved from");
    return impl_->build(promptTrackerPlanSpec());
}

Sam2TrackerNetworkOutputs Sam2TrackerNetworkBuilder::buildRecurrent(std::int32_t history_frames) {
    if (!impl_)
        throw NetworkBuildError("SAM2 tracker builder was moved from");
    return impl_->build(recurrentTrackerPlanSpec(history_frames));
}

} // namespace trtmc::sam2::native
