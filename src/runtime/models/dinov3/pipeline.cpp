/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/dinov3/pipeline.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

using HalfBits = uint16_t;

float fp16_to_fp32(HalfBits value) {
    const uint32_t sign = (static_cast<uint32_t>(value) & 0x8000U) << 16U;
    uint32_t exponent = (value >> 10U) & 0x1FU;
    uint32_t mantissa = value & 0x03FFU;
    uint32_t bits = sign;
    if (exponent == 0U) {
        if (mantissa != 0U) {
            exponent = 113U;
            while ((mantissa & 0x0400U) == 0U) {
                mantissa <<= 1U;
                --exponent;
            }
            bits |= (exponent << 23U) | ((mantissa & 0x03FFU) << 13U);
        }
    } else if (exponent == 0x1FU) {
        bits |= 0x7F800000U | (mantissa << 13U);
    } else {
        bits |= (exponent + 112U) << 23U;
        bits |= mantissa << 13U;
    }
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float bf16_to_fp32(HalfBits value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

bool round_to_nearest_even(uint32_t remainder, uint32_t halfway, uint32_t rounded) {
    return remainder > halfway || (remainder == halfway && (rounded & 1U) != 0U);
}

HalfBits fp32_subnormal_to_fp16(uint32_t sign, uint32_t mantissa, int32_t exponent) {
    if (exponent < -10)
        return static_cast<HalfBits>(sign);
    const uint32_t normalized = mantissa | 0x00800000U;
    const uint32_t shift = static_cast<uint32_t>(14 - exponent);
    uint32_t rounded = normalized >> shift;
    const uint32_t remainder = normalized & ((1U << shift) - 1U);
    const uint32_t halfway = 1U << (shift - 1U);
    if (round_to_nearest_even(remainder, halfway, rounded))
        ++rounded;
    return static_cast<HalfBits>(sign | rounded);
}

HalfBits fp32_normal_to_fp16(uint32_t sign, uint32_t mantissa, int32_t exponent) {
    uint32_t rounded = mantissa >> 13U;
    const uint32_t remainder = mantissa & 0x1FFFU;
    if (!round_to_nearest_even(remainder, 0x1000U, rounded))
        return static_cast<HalfBits>(sign | (static_cast<uint32_t>(exponent) << 10U) | rounded);

    ++rounded;
    if (rounded != 0x0400U)
        return static_cast<HalfBits>(sign | (static_cast<uint32_t>(exponent) << 10U) | rounded);
    if (exponent + 1 >= 31)
        return static_cast<HalfBits>(sign | 0x7C00U);
    return static_cast<HalfBits>(sign | (static_cast<uint32_t>(exponent + 1) << 10U));
}

HalfBits fp32_to_fp16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16U) & 0x8000U;
    const uint32_t mantissa = bits & 0x007FFFFFU;
    const uint32_t exponent_bits = (bits >> 23U) & 0xFFU;
    if (exponent_bits == 0xFFU)
        return static_cast<HalfBits>(sign | 0x7C00U | (mantissa == 0U ? 0U : 0x0200U));

    const int32_t exponent = static_cast<int32_t>(exponent_bits) - 127 + 15;
    if (exponent >= 31)
        return static_cast<HalfBits>(sign | 0x7C00U);
    if (exponent <= 0)
        return fp32_subnormal_to_fp16(sign, mantissa, exponent);
    return fp32_normal_to_fp16(sign, mantissa, exponent);
}

HalfBits fp32_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding_bias = 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<HalfBits>((bits + rounding_bias) >> 16U);
}

std::size_t validated_numel(const Tensor& tensor, const char* name) {
    if (tensor.data == nullptr)
        throw std::runtime_error(std::string("DINOv3 output '") + name + "' has no data");
    if (tensor.shape.empty())
        throw std::runtime_error(std::string("DINOv3 output '") + name + "' has no shape");
    std::size_t count = 1;
    for (int64_t dim : tensor.shape) {
        if (dim <= 0 ||
            static_cast<uint64_t>(dim) > std::numeric_limits<std::size_t>::max() / count) {
            throw std::runtime_error(std::string("DINOv3 output '") + name +
                                     "' has an invalid shape");
        }
        count *= static_cast<std::size_t>(dim);
    }
    return count;
}

std::vector<float> tensor_to_floats(const Tensor& tensor, const char* name) {
    const auto count = validated_numel(tensor, name);
    std::vector<float> result(count);
    if (tensor.dtype == DType::kFloat32) {
        const auto* source = static_cast<const float*>(tensor.data);
        std::copy_n(source, count, result.data());
    } else if (tensor.dtype == DType::kFloat16 || tensor.dtype == DType::kBFloat16) {
        const auto* source = static_cast<const HalfBits*>(tensor.data);
        for (std::size_t i = 0; i < count; ++i) {
            result[i] =
                tensor.dtype == DType::kFloat16 ? fp16_to_fp32(source[i]) : bf16_to_fp32(source[i]);
        }
    } else {
        throw std::runtime_error(std::string("DINOv3 output '") + name +
                                 "' must be a floating-point tensor");
    }
    return result;
}

const Tensor& require_output(const TensorMap& outputs, const char* name) {
    const auto output = outputs.find(name);
    if (output == outputs.end())
        throw std::runtime_error(std::string("DINOv3 engine did not return required output '") +
                                 name + "'");
    return output->second;
}

Tensor make_input_tensor(const std::vector<float>& values, std::vector<HalfBits>& values_16,
                         DType dtype, const std::vector<int64_t>& shape) {
    if (dtype == DType::kFloat32)
        return Tensor{const_cast<float*>(values.data()), shape, dtype};
    if (dtype != DType::kFloat16 && dtype != DType::kBFloat16)
        throw std::runtime_error("DINOv3 engine input 'pixel_values' must be floating point");
    values_16.resize(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        values_16[i] = dtype == DType::kFloat16 ? fp32_to_fp16(values[i]) : fp32_to_bf16(values[i]);
    }
    return Tensor{values_16.data(), shape, dtype};
}

} // namespace

Dinov3ImageFeaturePipeline::Dinov3ImageFeaturePipeline(std::unique_ptr<TrtModule> model,
                                                       Dinov3PreprocessConfig preprocess_config,
                                                       std::string model_id)
    : model_(std::move(model)), preprocess_config_(std::move(preprocess_config)),
      model_id_(std::move(model_id)) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("Dinov3ImageFeaturePipeline: invalid model");
}

ImageFeaturesResult Dinov3ImageFeaturePipeline::extract_image_features(const float* pixels,
                                                                       int32_t height,
                                                                       int32_t width) {
    const auto pixel_values = preprocess_dinov3_image(pixels, height, width, preprocess_config_);
    const std::vector<int64_t> input_shape{1, 3, preprocess_config_.input_image_h,
                                           preprocess_config_.input_image_w};
    const auto input_dtype = model_->tensor_dtype("pixel_values");
    std::vector<HalfBits> pixel_values_16;
    auto input = make_input_tensor(pixel_values, pixel_values_16, input_dtype, input_shape);
    const auto outputs = model_->forward({{"pixel_values", input}});

    const auto& hidden = require_output(outputs, "last_hidden_state");
    const auto& pooled = require_output(outputs, "pooler_output");
    ImageFeaturesResult result;
    result.last_hidden_state = tensor_to_floats(hidden, "last_hidden_state");
    result.last_hidden_state_shape = hidden.shape;
    result.pooler_output = tensor_to_floats(pooled, "pooler_output");
    result.pooler_output_shape = pooled.shape;
    return result;
}

} // namespace trtmc
