/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sam2_trt_layers.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>

namespace trtmc::sam2::native {
namespace {

std::string shapeString(const std::vector<int64_t>& shape) {
    std::ostringstream stream;
    stream << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0)
            stream << ',';
        stream << shape[i];
    }
    stream << ']';
    return stream.str();
}

int64_t elementCount(const std::vector<int64_t>& shape) {
    int64_t count = 1;
    for (const int64_t value : shape) {
        if (value <= 0 || count > std::numeric_limits<int64_t>::max() / value)
            throw NetworkBuildError("invalid or overflowing static shape " + shapeString(shape));
        count *= value;
    }
    return count;
}

std::vector<int64_t> asVector(std::initializer_list<int64_t> values) {
    return {values.begin(), values.end()};
}

} // namespace

TrtLayers::TrtLayers(nvinfer1::INetworkDefinition& network, const CheckpointReader& checkpoint)
    : network_(network), checkpoint_(checkpoint) {}

nvinfer1::Dims TrtLayers::dims(std::initializer_list<int64_t> values) {
    return dims(std::vector<int64_t>(values));
}

nvinfer1::Dims TrtLayers::dims(const std::vector<int64_t>& values) {
    if (values.size() > static_cast<std::size_t>(nvinfer1::Dims::MAX_DIMS))
        throw NetworkBuildError("tensor rank exceeds TensorRT's static rank limit");
    nvinfer1::Dims result{};
    result.nbDims = static_cast<int32_t>(values.size());
    int32_t index = 0;
    for (const int64_t value : values) {
        if (value < std::numeric_limits<int32_t>::min() ||
            value > std::numeric_limits<int32_t>::max()) {
            throw NetworkBuildError("tensor extent cannot be represented by TensorRT");
        }
        result.d[index++] = static_cast<int32_t>(value);
    }
    return result;
}

nvinfer1::Permutation TrtLayers::permutation(std::initializer_list<int32_t> values) {
    if (values.size() > static_cast<std::size_t>(nvinfer1::Dims::MAX_DIMS))
        throw NetworkBuildError("permutation rank exceeds TensorRT's static rank limit");
    nvinfer1::Permutation result{};
    std::fill(std::begin(result.order), std::end(result.order), 0);
    int32_t index = 0;
    for (const int32_t value : values)
        result.order[index++] = value;
    return result;
}

std::string TrtLayers::joined(std::string_view prefix, std::string_view suffix) {
    std::string result(prefix);
    result.append(suffix);
    return result;
}

nvinfer1::ILayer& TrtLayers::requireLayer(nvinfer1::ILayer* layer, std::string_view operation,
                                          std::string_view name) {
    if (layer == nullptr)
        throw NetworkBuildError(std::string(operation) + " rejected layer " + std::string(name));
    const std::string stable_name(name);
    layer->setName(stable_name.c_str());
    return *layer;
}

nvinfer1::Weights TrtLayers::requiredWeights(std::string_view name,
                                             const std::vector<int64_t>& shape) {
    const WeightView view = checkpoint_.requireTensor(name, DType::kFloat32, shape);
    if (!view.contiguous)
        throw NetworkBuildError("checkpoint tensor is not contiguous: " + std::string(name));
    const int64_t count = elementCount(shape);
    if (view.data == nullptr || view.bytes != static_cast<std::size_t>(count) * sizeof(float))
        throw NetworkBuildError("checkpoint tensor has an invalid byte count: " +
                                std::string(name));
    referenced_tensors_.emplace(name);
    return {nvinfer1::DataType::kFLOAT, view.data, count};
}

nvinfer1::Weights TrtLayers::convertedWeights(nvinfer1::Weights source, nvinfer1::DataType type) {
    if (source.count == 0)
        return {type, nullptr, 0};
    if (source.type == type)
        return source;
    if (source.type != nvinfer1::DataType::kFLOAT || type != nvinfer1::DataType::kBF16 ||
        source.values == nullptr) {
        throw NetworkBuildError("unsupported convolution weight conversion");
    }
    const auto* input = static_cast<const float*>(source.values);
    std::vector<uint16_t> output(static_cast<std::size_t>(source.count));
    for (int64_t i = 0; i < source.count; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, input + i, sizeof(bits));
        const uint32_t least_significant_retained_bit = (bits >> 16U) & 1U;
        bits += 0x7FFFU + least_significant_retained_bit;
        output[static_cast<std::size_t>(i)] = static_cast<uint16_t>(bits >> 16U);
    }
    owned_bf16_weights_.push_back(std::move(output));
    const auto& storage = owned_bf16_weights_.back();
    return {nvinfer1::DataType::kBF16, storage.data(), static_cast<int64_t>(storage.size())};
}

nvinfer1::ITensor& TrtLayers::cast(nvinfer1::ITensor& input, nvinfer1::DataType type,
                                   std::string_view name) {
    if (input.getType() == type)
        return input;
    auto* layer = network_.addCast(input, type);
    requireLayer(layer, "cast", name);
    return *layer->getOutput(0);
}

nvinfer1::ITensor& TrtLayers::shuffle(nvinfer1::ITensor& input,
                                      std::initializer_list<int64_t> reshape,
                                      std::string_view name) {
    return shuffle(input, std::vector<int64_t>(reshape), name);
}

nvinfer1::ITensor& TrtLayers::shuffle(nvinfer1::ITensor& input, const std::vector<int64_t>& reshape,
                                      std::string_view name) {
    auto* layer = network_.addShuffle(input);
    requireLayer(layer, "shuffle", name);
    layer->setReshapeDimensions(dims(reshape));
    return *layer->getOutput(0);
}

nvinfer1::ITensor& TrtLayers::transpose(nvinfer1::ITensor& input,
                                        std::initializer_list<int32_t> order,
                                        std::string_view name) {
    if (order.size() != static_cast<std::size_t>(input.getDimensions().nbDims))
        throw NetworkBuildError("transpose rank mismatch for " + std::string(name));
    auto* layer = network_.addShuffle(input);
    requireLayer(layer, "transpose", name);
    layer->setFirstTranspose(permutation(order));
    return *layer->getOutput(0);
}

nvinfer1::ITensor& TrtLayers::slice(nvinfer1::ITensor& input, std::initializer_list<int64_t> start,
                                    std::initializer_list<int64_t> size,
                                    std::initializer_list<int64_t> stride,
                                    nvinfer1::SampleMode mode, std::string_view name) {
    const int32_t rank = input.getDimensions().nbDims;
    if (start.size() != static_cast<std::size_t>(rank) ||
        size.size() != static_cast<std::size_t>(rank) ||
        stride.size() != static_cast<std::size_t>(rank)) {
        throw NetworkBuildError("slice rank mismatch for " + std::string(name));
    }
    auto* layer = network_.addSlice(input, dims(start), dims(size), dims(stride));
    requireLayer(layer, "slice", name);
    layer->setMode(mode);
    return *layer->getOutput(0);
}

nvinfer1::ITensor& TrtLayers::concatenate(const std::vector<nvinfer1::ITensor*>& inputs,
                                          int32_t axis, std::string_view name) {
    if (inputs.empty() ||
        std::any_of(inputs.begin(), inputs.end(),
                    [](const nvinfer1::ITensor* tensor) { return tensor == nullptr; })) {
        throw NetworkBuildError("concatenation has no inputs for " + std::string(name));
    }
    auto* layer = network_.addConcatenation(inputs.data(), static_cast<int32_t>(inputs.size()));
    requireLayer(layer, "concatenation", name);
    layer->setAxis(axis);
    return *layer->getOutput(0);
}

nvinfer1::ITensor& TrtLayers::elementWise(nvinfer1::ITensor& lhs, nvinfer1::ITensor& rhs,
                                          nvinfer1::ElementWiseOperation operation,
                                          std::string_view name) {
    if (lhs.getType() != rhs.getType())
        throw NetworkBuildError("elementwise input type mismatch for " + std::string(name));
    auto* layer = network_.addElementWise(lhs, rhs, operation);
    requireLayer(layer, "elementwise", name);
    return *layer->getOutput(0);
}

nvinfer1::ITensor& TrtLayers::matrixMultiply(nvinfer1::ITensor& lhs,
                                             nvinfer1::MatrixOperation lhs_operation,
                                             nvinfer1::ITensor& rhs,
                                             nvinfer1::MatrixOperation rhs_operation,
                                             std::string_view name) {
    if (lhs.getType() != rhs.getType())
        throw NetworkBuildError("matrix multiplication input type mismatch for " +
                                std::string(name));
    auto* layer = network_.addMatrixMultiply(lhs, lhs_operation, rhs, rhs_operation);
    requireLayer(layer, "matrix multiplication", name);
    return *layer->getOutput(0);
}

nvinfer1::ITensor& TrtLayers::softmax(nvinfer1::ITensor& input, uint32_t axes,
                                      std::string_view name) {
    auto* layer = network_.addSoftMax(input);
    requireLayer(layer, "softmax", name);
    layer->setAxes(axes);
    return *layer->getOutput(0);
}

nvinfer1::ITensor& TrtLayers::constant(std::string_view checkpoint_name,
                                       std::initializer_list<int64_t> checkpoint_shape,
                                       std::initializer_list<int64_t> tensor_shape,
                                       std::string_view name) {
    const std::vector<int64_t> source_shape = asVector(checkpoint_shape);
    const std::vector<int64_t> result_shape = asVector(tensor_shape);
    if (elementCount(source_shape) != elementCount(result_shape))
        throw NetworkBuildError("constant reshape changes element count for " +
                                std::string(checkpoint_name));
    const nvinfer1::Weights weights = requiredWeights(checkpoint_name, source_shape);
    auto* layer = network_.addConstant(dims(tensor_shape), weights);
    requireLayer(layer, "constant", name);
    return *layer->getOutput(0);
}

nvinfer1::ITensor& TrtLayers::ownedConstant(const std::vector<float>& values,
                                            std::initializer_list<int64_t> tensor_shape,
                                            nvinfer1::DataType type, std::string_view name) {
    if (values.empty() ||
        elementCount(asVector(tensor_shape)) != static_cast<int64_t>(values.size())) {
        throw NetworkBuildError("owned constant shape mismatch for " + std::string(name));
    }
    owned_float_weights_.push_back(values);
    const auto& storage = owned_float_weights_.back();
    const nvinfer1::Weights weights{nvinfer1::DataType::kFLOAT, storage.data(),
                                    static_cast<int64_t>(storage.size())};
    auto* layer = network_.addConstant(dims(tensor_shape), weights);
    requireLayer(layer, "constant", name);
    if (type == nvinfer1::DataType::kFLOAT)
        return *layer->getOutput(0);
    return cast(*layer->getOutput(0), type, joined(name, ".cast"));
}

nvinfer1::ITensor& TrtLayers::scalar(float value, int32_t rank, nvinfer1::DataType type,
                                     std::string_view name) {
    if (rank <= 0 || rank > nvinfer1::Dims::MAX_DIMS)
        throw NetworkBuildError("invalid scalar broadcast rank for " + std::string(name));
    owned_float_weights_.push_back({value});
    nvinfer1::Dims shape{};
    shape.nbDims = rank;
    for (int32_t i = 0; i < rank; ++i)
        shape.d[i] = 1;
    const auto& storage = owned_float_weights_.back();
    const nvinfer1::Weights weights{nvinfer1::DataType::kFLOAT, storage.data(), 1};
    auto* layer = network_.addConstant(shape, weights);
    requireLayer(layer, "constant", name);
    if (type == nvinfer1::DataType::kFLOAT)
        return *layer->getOutput(0);
    return cast(*layer->getOutput(0), type, joined(name, ".cast"));
}

nvinfer1::ITensor& TrtLayers::convolution(nvinfer1::ITensor& input, std::string_view weight_name,
                                          std::string_view bias_name, int32_t input_channels,
                                          int32_t output_channels, int32_t kernel, int32_t stride,
                                          int32_t padding, int32_t groups, std::string_view name) {
    if (input_channels <= 0 || output_channels <= 0 || kernel <= 0 || stride <= 0 || groups <= 0 ||
        input_channels % groups != 0 || output_channels % groups != 0) {
        throw NetworkBuildError("invalid convolution geometry for " + std::string(name));
    }
    const nvinfer1::DataType input_type = input.getType();
    if (input_type != nvinfer1::DataType::kFLOAT && input_type != nvinfer1::DataType::kBF16) {
        throw NetworkBuildError("unsupported convolution input type for " + std::string(name));
    }
    const nvinfer1::Weights kernel_weights = convertedWeights(
        requiredWeights(weight_name, {output_channels, input_channels / groups, kernel, kernel}),
        input_type);
    nvinfer1::Weights bias_weights{input_type, nullptr, 0};
    if (!bias_name.empty())
        bias_weights = convertedWeights(requiredWeights(bias_name, {output_channels}), input_type);
    auto* layer = network_.addConvolutionNd(input, output_channels, dims({kernel, kernel}),
                                            kernel_weights, bias_weights);
    requireLayer(layer, "convolution", name);
    layer->setStrideNd(dims({stride, stride}));
    layer->setPaddingNd(dims({padding, padding}));
    layer->setNbGroups(groups);
    return *layer->getOutput(0);
}

TrtLayers::FoldedWeights TrtLayers::foldBatchNorm(std::string_view module_name,
                                                  int32_t input_channels, int32_t output_channels,
                                                  int32_t kernel, int32_t groups, float epsilon) {
    const std::string conv_name = joined(module_name, ".conv.weight");
    const nvinfer1::Weights raw_kernel =
        requiredWeights(conv_name, {output_channels, input_channels / groups, kernel, kernel});
    const nvinfer1::Weights gamma =
        requiredWeights(joined(module_name, ".bn.weight"), {output_channels});
    const nvinfer1::Weights beta =
        requiredWeights(joined(module_name, ".bn.bias"), {output_channels});
    const nvinfer1::Weights mean =
        requiredWeights(joined(module_name, ".bn.running_mean"), {output_channels});
    const nvinfer1::Weights variance =
        requiredWeights(joined(module_name, ".bn.running_var"), {output_channels});

    const auto* kernel_data = static_cast<const float*>(raw_kernel.values);
    const auto* gamma_data = static_cast<const float*>(gamma.values);
    const auto* beta_data = static_cast<const float*>(beta.values);
    const auto* mean_data = static_cast<const float*>(mean.values);
    const auto* variance_data = static_cast<const float*>(variance.values);
    const int64_t values_per_output =
        static_cast<int64_t>(input_channels / groups) * kernel * kernel;

    std::vector<float> folded_kernel(static_cast<std::size_t>(raw_kernel.count));
    std::vector<float> folded_bias(static_cast<std::size_t>(output_channels));
    for (int32_t output = 0; output < output_channels; ++output) {
        if (!(variance_data[output] >= 0.0F) || !std::isfinite(variance_data[output]))
            throw NetworkBuildError("invalid batch-normalization variance in " +
                                    std::string(module_name));
        const float scale = gamma_data[output] / std::sqrt(variance_data[output] + epsilon);
        folded_bias[static_cast<std::size_t>(output)] =
            beta_data[output] - mean_data[output] * scale;
        const int64_t offset = static_cast<int64_t>(output) * values_per_output;
        for (int64_t i = 0; i < values_per_output; ++i)
            folded_kernel[static_cast<std::size_t>(offset + i)] = kernel_data[offset + i] * scale;
    }

    owned_float_weights_.push_back(std::move(folded_kernel));
    const auto* folded_kernel_data = owned_float_weights_.back().data();
    const int64_t folded_kernel_count = static_cast<int64_t>(owned_float_weights_.back().size());
    owned_float_weights_.push_back(std::move(folded_bias));
    const auto* folded_bias_data = owned_float_weights_.back().data();
    return {{nvinfer1::DataType::kFLOAT, folded_kernel_data, folded_kernel_count},
            {nvinfer1::DataType::kFLOAT, folded_bias_data, output_channels}};
}

nvinfer1::ITensor&
TrtLayers::convolutionBatchNormSilu(nvinfer1::ITensor& input, std::string_view module_name,
                                    int32_t input_channels, int32_t output_channels, int32_t kernel,
                                    int32_t stride, int32_t padding, int32_t groups, float epsilon,
                                    std::string_view name) {
    nvinfer1::ITensor& bf16_input =
        cast(input, nvinfer1::DataType::kBF16, joined(name, ".input_bf16"));
    const FoldedWeights weights =
        foldBatchNorm(module_name, input_channels, output_channels, kernel, groups, epsilon);
    const nvinfer1::Weights kernel_weights =
        convertedWeights(weights.kernel, nvinfer1::DataType::kBF16);
    const nvinfer1::Weights bias_weights =
        convertedWeights(weights.bias, nvinfer1::DataType::kBF16);
    auto* layer = network_.addConvolutionNd(bf16_input, output_channels, dims({kernel, kernel}),
                                            kernel_weights, bias_weights);
    requireLayer(layer, "convolution", joined(name, ".conv_bn_folded"));
    layer->setStrideNd(dims({stride, stride}));
    layer->setPaddingNd(dims({padding, padding}));
    layer->setNbGroups(groups);
    return silu(*layer->getOutput(0), joined(name, ".silu"));
}

nvinfer1::ITensor& TrtLayers::linearAutocastBf16(nvinfer1::ITensor& input,
                                                 std::string_view module_name,
                                                 int32_t input_features, int32_t output_features,
                                                 std::string_view name) {
    const nvinfer1::Dims input_dims = input.getDimensions();
    if (input_dims.nbDims < 2 || input_dims.d[input_dims.nbDims - 1] != input_features) {
        throw NetworkBuildError("linear input shape mismatch for " + std::string(name));
    }
    nvinfer1::ITensor& bf16_input =
        cast(input, nvinfer1::DataType::kBF16, joined(name, ".input_bf16"));
    nvinfer1::ITensor& weight_fp32 =
        constant(joined(module_name, ".weight"), {output_features, input_features},
                 {output_features, input_features}, joined(name, ".weight"));
    nvinfer1::ITensor& weight_bf16 =
        cast(weight_fp32, nvinfer1::DataType::kBF16, joined(name, ".weight_bf16"));
    std::vector<int64_t> weight_shape(static_cast<std::size_t>(input_dims.nbDims), 1);
    weight_shape[weight_shape.size() - 2] = output_features;
    weight_shape.back() = input_features;
    nvinfer1::ITensor& weight =
        shuffle(weight_bf16, weight_shape, joined(name, ".weight_broadcast"));
    auto* multiply = network_.addMatrixMultiply(bf16_input, nvinfer1::MatrixOperation::kNONE,
                                                weight, nvinfer1::MatrixOperation::kTRANSPOSE);
    requireLayer(multiply, "matrix multiplication", joined(name, ".matmul"));

    std::vector<int64_t> bias_shape(static_cast<std::size_t>(input_dims.nbDims), 1);
    bias_shape.back() = output_features;
    const nvinfer1::Weights bias_weights =
        requiredWeights(joined(module_name, ".bias"), {output_features});
    nvinfer1::Dims bias_dims{};
    bias_dims.nbDims = input_dims.nbDims;
    for (int32_t i = 0; i < input_dims.nbDims; ++i)
        bias_dims.d[i] = static_cast<int32_t>(bias_shape[static_cast<std::size_t>(i)]);
    auto* bias_constant = network_.addConstant(bias_dims, bias_weights);
    requireLayer(bias_constant, "constant", joined(name, ".bias"));
    nvinfer1::ITensor& bias =
        cast(*bias_constant->getOutput(0), nvinfer1::DataType::kBF16, joined(name, ".bias_bf16"));
    return elementWise(*multiply->getOutput(0), bias, nvinfer1::ElementWiseOperation::kSUM,
                       joined(name, ".bias_add"));
}

nvinfer1::ITensor& TrtLayers::layerNormFp32(nvinfer1::ITensor& input, std::string_view module_name,
                                            int32_t channels, float epsilon,
                                            std::string_view name) {
    const nvinfer1::Dims input_dims = input.getDimensions();
    if (input_dims.nbDims <= 0 || input_dims.d[input_dims.nbDims - 1] != channels)
        throw NetworkBuildError("layer-normalization input shape mismatch for " +
                                std::string(name));
    nvinfer1::ITensor& fp32_input =
        cast(input, nvinfer1::DataType::kFLOAT, joined(name, ".input_fp32"));

    nvinfer1::Dims parameter_dims{};
    parameter_dims.nbDims = input_dims.nbDims;
    for (int32_t i = 0; i < input_dims.nbDims; ++i)
        parameter_dims.d[i] = i == input_dims.nbDims - 1 ? channels : 1;
    const nvinfer1::Weights scale_weights =
        requiredWeights(joined(module_name, ".weight"), {channels});
    const nvinfer1::Weights bias_weights =
        requiredWeights(joined(module_name, ".bias"), {channels});
    auto* scale = network_.addConstant(parameter_dims, scale_weights);
    auto* bias = network_.addConstant(parameter_dims, bias_weights);
    requireLayer(scale, "constant", joined(name, ".scale"));
    requireLayer(bias, "constant", joined(name, ".bias"));
    const uint32_t axes = 1U << static_cast<uint32_t>(input_dims.nbDims - 1);
#if NV_TENSORRT_MAJOR >= 11
    auto* layer =
        network_.addNormalizationV2(fp32_input, *scale->getOutput(0), *bias->getOutput(0), axes);
#else
    auto* layer =
        network_.addNormalization(fp32_input, *scale->getOutput(0), *bias->getOutput(0), axes);
#endif
    requireLayer(layer, "normalization", name);
    layer->setEpsilon(epsilon);
    return *layer->getOutput(0);
}

nvinfer1::ITensor& TrtLayers::gelu(nvinfer1::ITensor& input, std::string_view name) {
    auto* layer = network_.addActivation(input, nvinfer1::ActivationType::kGELU_ERF);
    requireLayer(layer, "GELU", name);
    return *layer->getOutput(0);
}

nvinfer1::ITensor& TrtLayers::silu(nvinfer1::ITensor& input, std::string_view name) {
    auto* sigmoid = network_.addActivation(input, nvinfer1::ActivationType::kSIGMOID);
    requireLayer(sigmoid, "sigmoid", joined(name, ".sigmoid"));
    return elementWise(input, *sigmoid->getOutput(0), nvinfer1::ElementWiseOperation::kPROD, name);
}

nvinfer1::ITensor& TrtLayers::maxPoolNchw(nvinfer1::ITensor& input, int32_t kernel, int32_t stride,
                                          std::string_view name) {
    auto* layer = network_.addPoolingNd(input, nvinfer1::PoolingType::kMAX, dims({kernel, kernel}));
    requireLayer(layer, "max pooling", name);
    layer->setStrideNd(dims({stride, stride}));
    layer->setPaddingNd(dims({0, 0}));
    layer->setAverageCountExcludesPadding(true);
    return *layer->getOutput(0);
}

nvinfer1::ITensor& TrtLayers::maxPoolNhwc(nvinfer1::ITensor& input, int32_t kernel, int32_t stride,
                                          std::string_view name) {
    nvinfer1::ITensor& nchw = transpose(input, {0, 3, 1, 2}, joined(name, ".to_nchw"));
    nvinfer1::ITensor& pooled = maxPoolNchw(nchw, kernel, stride, joined(name, ".pool"));
    return transpose(pooled, {0, 2, 3, 1}, joined(name, ".to_nhwc"));
}

nvinfer1::ITensor& TrtLayers::resizeNchw(nvinfer1::ITensor& input, int32_t output_height,
                                         int32_t output_width,
                                         nvinfer1::InterpolationMode interpolation,
                                         nvinfer1::ResizeCoordinateTransformation coordinates,
                                         std::string_view name, float cubic_coefficient) {
    const nvinfer1::Dims input_dims = input.getDimensions();
    if (input_dims.nbDims != 4 || output_height <= 0 || output_width <= 0)
        throw NetworkBuildError("invalid NCHW resize geometry for " + std::string(name));
    auto* layer = network_.addResize(input);
    requireLayer(layer, "resize", name);
    layer->setResizeMode(interpolation);
    layer->setCoordinateTransformation(coordinates);
    layer->setOutputDimensions(
        dims({input_dims.d[0], input_dims.d[1], output_height, output_width}));
    if (interpolation == nvinfer1::InterpolationMode::kCUBIC)
        layer->setCubicCoeff(cubic_coefficient);
    return *layer->getOutput(0);
}

WindowTensor TrtLayers::windowPartition(nvinfer1::ITensor& input, int32_t height, int32_t width,
                                        int32_t channels, int32_t window_size,
                                        std::string_view name) {
    if (height <= 0 || width <= 0 || channels <= 0 || window_size <= 0)
        throw NetworkBuildError("invalid window geometry for " + std::string(name));
    const int32_t padded_height = height + (window_size - height % window_size) % window_size;
    const int32_t padded_width = width + (window_size - width % window_size) % window_size;
    nvinfer1::ITensor* padded = &input;
    if (padded_height != height || padded_width != width) {
        padded = &slice(input, {0, 0, 0, 0}, {1, padded_height, padded_width, channels},
                        {1, 1, 1, 1}, nvinfer1::SampleMode::kFILL, joined(name, ".pad"));
    }
    nvinfer1::ITensor& blocked = shuffle(*padded,
                                         {1, padded_height / window_size, window_size,
                                          padded_width / window_size, window_size, channels},
                                         joined(name, ".block"));
    nvinfer1::ITensor& ordered = transpose(blocked, {0, 1, 3, 2, 4, 5}, joined(name, ".order"));
    const int32_t window_count = (padded_height / window_size) * (padded_width / window_size);
    nvinfer1::ITensor& result = shuffle(ordered, {window_count, window_size, window_size, channels},
                                        joined(name, ".windows"));
    return {&result, height, width, padded_height, padded_width, window_size};
}

nvinfer1::ITensor& TrtLayers::windowUnpartition(const WindowTensor& windows,
                                                nvinfer1::ITensor& input, int32_t output_height,
                                                int32_t output_width, int32_t channels,
                                                int32_t output_window_size, std::string_view name) {
    if (windows.tensor == nullptr || output_window_size <= 0 ||
        windows.window_size % output_window_size != 0) {
        throw NetworkBuildError("invalid window unpartition geometry for " + std::string(name));
    }
    const int32_t reduction = windows.window_size / output_window_size;
    const int32_t padded_height = windows.padded_height / reduction;
    const int32_t padded_width = windows.padded_width / reduction;
    if (padded_height % output_window_size != 0 || padded_width % output_window_size != 0 ||
        output_height > padded_height || output_width > padded_width) {
        throw NetworkBuildError("window unpartition dimensions do not close for " +
                                std::string(name));
    }
    nvinfer1::ITensor& blocked =
        shuffle(input,
                {1, padded_height / output_window_size, padded_width / output_window_size,
                 output_window_size, output_window_size, channels},
                joined(name, ".block"));
    nvinfer1::ITensor& ordered = transpose(blocked, {0, 1, 3, 2, 4, 5}, joined(name, ".order"));
    nvinfer1::ITensor& padded =
        shuffle(ordered, {1, padded_height, padded_width, channels}, joined(name, ".merge"));
    if (padded_height == output_height && padded_width == output_width)
        return padded;
    return slice(padded, {0, 0, 0, 0}, {1, output_height, output_width, channels}, {1, 1, 1, 1},
                 nvinfer1::SampleMode::kSTRICT_BOUNDS, joined(name, ".crop"));
}

std::size_t TrtLayers::referencedTensorCount() const noexcept {
    return referenced_tensors_.size();
}

} // namespace trtmc::sam2::native
