/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/moge/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

constexpr int32_t kFocalRecoverySize = 64;
constexpr double kDenominatorEpsilon = 1.0e-9;
constexpr int32_t kMinImageSize = 64;
constexpr int32_t kMaxImageSize = 2048;
constexpr float kMinAspectRatio = 0.5F;
constexpr float kMaxAspectRatio = 2.0F;
constexpr float kMaskThreshold = 0.5F;

struct FocalSample {
    double u{0.0};
    double v{0.0};
    double x{0.0};
    double y{0.0};
    double z{0.0};
};

struct FocalShift {
    float focal{1.0F}; // relative to half the image diagonal
    float shift{0.0F};
};

double normalized_coordinate(int32_t index, int32_t size, double span) {
    return span * (2.0 * index + 1.0 - size) / size;
}

std::vector<FocalSample> make_focal_samples(const float* points, const std::vector<uint8_t>& mask,
                                            int32_t height, int32_t width) {
    const double aspect = static_cast<double>(width) / height;
    const double diagonal_factor = std::sqrt(1.0 + aspect * aspect);
    const double span_x = aspect / diagonal_factor;
    const double span_y = 1.0 / diagonal_factor;
    std::vector<FocalSample> samples;
    samples.reserve(static_cast<std::size_t>(kFocalRecoverySize) * kFocalRecoverySize);

    // torch.nn.functional.interpolate(..., mode="nearest") selects
    // floor(output_index * input_size / output_size). The official MoGe
    // recovery downsamples points, UVs, and the mask this way to 64x64.
    for (int32_t out_y = 0; out_y < kFocalRecoverySize; ++out_y) {
        const int32_t y = std::min(height - 1, static_cast<int32_t>(static_cast<int64_t>(out_y) *
                                                                    height / kFocalRecoverySize));
        for (int32_t out_x = 0; out_x < kFocalRecoverySize; ++out_x) {
            const int32_t x = std::min(width - 1, static_cast<int32_t>(static_cast<int64_t>(out_x) *
                                                                       width / kFocalRecoverySize));
            const auto pixel = static_cast<std::size_t>(y) * width + x;
            if (mask[pixel] == 0)
                continue;
            const auto point = pixel * 3U;
            const double px = points[point];
            const double py = points[point + 1U];
            const double pz = points[point + 2U];
            if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz))
                continue;
            samples.push_back({normalized_coordinate(x, width, span_x),
                               normalized_coordinate(y, height, span_y), px, py, pz});
        }
    }
    return samples;
}

struct FocalObjective {
    double focal{1.0};
    double cost{std::numeric_limits<double>::infinity()};
    double gradient{0.0};
    double curvature{0.0};
    bool ok{false};
};

bool valid_sample_depth(double depth) {
    return std::isfinite(depth) && std::abs(depth) > kDenominatorEpsilon;
}

bool estimate_focal(const std::vector<FocalSample>& samples, double shift, double& focal) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (const auto& sample : samples) {
        const double depth = sample.z + shift;
        if (!valid_sample_depth(depth))
            return false;
        const double projected_x = sample.x / depth;
        const double projected_y = sample.y / depth;
        numerator += projected_x * sample.u + projected_y * sample.v;
        denominator += projected_x * projected_x + projected_y * projected_y;
    }
    if (!std::isfinite(numerator) || !std::isfinite(denominator) ||
        denominator <= kDenominatorEpsilon) {
        return false;
    }

    focal = numerator / denominator;
    return std::isfinite(focal) && focal > kDenominatorEpsilon;
}

bool valid_focal_objective(const FocalObjective& objective) {
    return std::isfinite(objective.cost) && std::isfinite(objective.gradient) &&
           std::isfinite(objective.curvature) && objective.curvature > kDenominatorEpsilon;
}

FocalObjective accumulate_focal_objective(const std::vector<FocalSample>& samples, double shift,
                                          double focal) {
    FocalObjective result;
    result.focal = focal;
    result.cost = 0.0;
    result.gradient = 0.0;
    result.curvature = 0.0;
    for (const auto& sample : samples) {
        const double depth = sample.z + shift;
        const double inv_depth = 1.0 / depth;
        const double projected_x = sample.x * inv_depth;
        const double projected_y = sample.y * inv_depth;
        const double residual_x = focal * projected_x - sample.u;
        const double residual_y = focal * projected_y - sample.v;
        // At the least-squares-optimal focal, residuals are orthogonal to
        // projected XY. The focal derivative therefore drops from d(cost)/ds.
        const double derivative_x = -focal * sample.x * inv_depth * inv_depth;
        const double derivative_y = -focal * sample.y * inv_depth * inv_depth;
        result.cost += residual_x * residual_x + residual_y * residual_y;
        result.gradient += residual_x * derivative_x + residual_y * derivative_y;
        result.curvature += derivative_x * derivative_x + derivative_y * derivative_y;
    }
    result.ok = valid_focal_objective(result);
    return result;
}

FocalObjective evaluate_focal_objective(const std::vector<FocalSample>& samples, double shift) {
    double focal = 1.0;
    if (!estimate_focal(samples, shift, focal))
        return {};
    return accumulate_focal_objective(samples, shift, focal);
}

bool objective_improves(const FocalObjective& candidate, const FocalObjective& current) {
    return candidate.ok && candidate.cost < current.cost;
}

bool valid_focal_recovery(const FocalObjective& objective, double shift) {
    return objective.ok && std::isfinite(shift) && std::isfinite(objective.focal) &&
           objective.focal > 0.0;
}

const float* require_float_output(const TensorMap& outputs, const char* name,
                                  const std::vector<int64_t>& shape) {
    const auto iterator = outputs.find(name);
    if (iterator == outputs.end())
        throw std::runtime_error(std::string("MoGe engine did not return required output '") +
                                 name + "'");
    const auto& tensor = iterator->second;
    if (tensor.data == nullptr || tensor.dtype != DType::kFloat32 || tensor.shape != shape) {
        throw std::runtime_error(std::string("MoGe output contract mismatch for '") + name + "'");
    }
    return static_cast<const float*>(tensor.data);
}

void validate_metric_scale(float metric_scale) {
    if (!std::isfinite(metric_scale) || metric_scale <= 0.0F)
        throw std::invalid_argument("MoGe metric scale must be finite and positive");
}

bool valid_raw_geometry_pixel(const float* affine_points, const float* mask_probabilities,
                              std::size_t pixel) {
    const auto point = pixel * 3U;
    return std::isfinite(mask_probabilities[pixel]) && mask_probabilities[pixel] > kMaskThreshold &&
           std::isfinite(affine_points[point]) && std::isfinite(affine_points[point + 1U]) &&
           std::isfinite(affine_points[point + 2U]);
}

std::vector<uint8_t> make_raw_mask(const float* affine_points, const float* mask_probabilities,
                                   std::size_t area) {
    std::vector<uint8_t> raw_mask(area, 0);
    for (std::size_t pixel = 0; pixel < area; ++pixel) {
        raw_mask[pixel] = static_cast<uint8_t>(
            valid_raw_geometry_pixel(affine_points, mask_probabilities, pixel));
    }
    return raw_mask;
}

struct MogeCalibration {
    double span_x{1.0};
    double span_y{1.0};
    float fx{1.0F};
    float fy{1.0F};
};

bool valid_intrinsics(const MogeCalibration& calibration) {
    return std::isfinite(calibration.fx) && std::isfinite(calibration.fy) &&
           calibration.fx > 0.0F && calibration.fy > 0.0F;
}

MogeCalibration make_calibration(const FocalShift& recovered, int32_t height, int32_t width) {
    const double aspect = static_cast<double>(width) / height;
    const double diagonal_factor = std::sqrt(1.0 + aspect * aspect);
    MogeCalibration calibration;
    calibration.span_x = aspect / diagonal_factor;
    calibration.span_y = 1.0 / diagonal_factor;
    calibration.fx = static_cast<float>(recovered.focal * diagonal_factor / (2.0 * aspect));
    calibration.fy = static_cast<float>(recovered.focal * diagonal_factor / 2.0);
    if (!valid_intrinsics(calibration))
        throw std::runtime_error("MoGe recovered invalid camera intrinsics");
    return calibration;
}

moge::GeometryResult make_geometry_result(int32_t height, int32_t width, std::size_t area,
                                          const MogeCalibration& calibration) {
    moge::GeometryResult result;
    result.height = height;
    result.width = width;
    result.intrinsics = {calibration.fx, 0.0F, 0.5F, 0.0F, calibration.fy, 0.5F, 0.0F, 0.0F, 1.0F};
    result.points.resize(area * 3U);
    result.depth.resize(area);
    result.mask.resize(area);
    return result;
}

bool valid_metric_depth(uint8_t raw_mask, double depth) {
    return raw_mask != 0 && std::isfinite(depth) && depth > 0.0;
}

void write_invalid_geometry(moge::GeometryResult& result, std::size_t pixel, float infinity) {
    const auto point = pixel * 3U;
    result.depth[pixel] = infinity;
    result.points[point] = infinity;
    result.points[point + 1U] = infinity;
    result.points[point + 2U] = infinity;
}

void populate_geometry_result(moge::GeometryResult& result, const float* affine_points,
                              const std::vector<uint8_t>& raw_mask, const FocalShift& recovered,
                              const MogeCalibration& calibration, float metric_scale) {
    const float infinity = std::numeric_limits<float>::infinity();
    for (int32_t y = 0; y < result.height; ++y) {
        const double v = normalized_coordinate(y, result.height, calibration.span_y);
        for (int32_t x = 0; x < result.width; ++x) {
            const auto pixel = static_cast<std::size_t>(y) * result.width + x;
            const auto point = pixel * 3U;
            const double depth_unscaled =
                static_cast<double>(affine_points[point + 2U]) + recovered.shift;
            const bool valid = valid_metric_depth(raw_mask[pixel], depth_unscaled);
            result.mask[pixel] = static_cast<uint8_t>(valid);
            if (!valid) {
                write_invalid_geometry(result, pixel, infinity);
                continue;
            }
            const double depth = depth_unscaled * metric_scale;
            const double u = normalized_coordinate(x, result.width, calibration.span_x);
            result.depth[pixel] = static_cast<float>(depth);
            result.points[point] = static_cast<float>(u * depth / recovered.focal);
            result.points[point + 1U] = static_cast<float>(v * depth / recovered.focal);
            result.points[point + 2U] = static_cast<float>(depth);
        }
    }
}

bool supported_image_size(int32_t height, int32_t width) {
    return height >= kMinImageSize && width >= kMinImageSize && height <= kMaxImageSize &&
           width <= kMaxImageSize;
}

bool supported_aspect_ratio(int32_t height, int32_t width) {
    const float aspect = static_cast<float>(width) / height;
    return aspect >= kMinAspectRatio && aspect <= kMaxAspectRatio;
}

bool valid_rgb_value(float value) {
    return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
}

void validate_image_input(const float* pixels, int32_t height, int32_t width) {
    if (pixels == nullptr)
        throw std::invalid_argument("MoGe image pointer is null");
    if (!supported_image_size(height, width))
        throw std::invalid_argument("MoGe image dimensions are outside the bundle profile");
    if (!supported_aspect_ratio(height, width))
        throw std::invalid_argument("MoGe image aspect ratio is outside the supported range");
    const auto area = static_cast<std::size_t>(height) * width;
    for (std::size_t index = 0; index < area * 3U; ++index) {
        if (!valid_rgb_value(pixels[index]))
            throw std::invalid_argument("MoGe RGB input values must be finite and in [0, 1]");
    }
}

std::optional<FocalShift> recover_focal_shift(const float* affine_points,
                                              const std::vector<uint8_t>& mask, int32_t height,
                                              int32_t width) {
    const auto samples = make_focal_samples(affine_points, mask, height, width);
    if (samples.size() < 2U)
        return std::nullopt;

    double shift = 0.0;
    double damping = 1.0e-3;
    auto current = evaluate_focal_objective(samples, shift);
    if (!current.ok)
        return std::nullopt;

    for (int iteration = 0; iteration < 500; ++iteration) {
        const double step = -current.gradient / (current.curvature + damping);
        if (!std::isfinite(step))
            break;
        if (std::abs(step) <= 1.0e-10 * (1.0 + std::abs(shift)))
            break;
        const double candidate_shift = shift + step;
        const auto candidate = evaluate_focal_objective(samples, candidate_shift);
        if (objective_improves(candidate, current)) {
            const double improvement = current.cost - candidate.cost;
            shift = candidate_shift;
            current = candidate;
            damping = std::max(1.0e-12, damping * 0.25);
            if (improvement <= 1.0e-12 * (1.0 + current.cost))
                break;
        } else {
            damping = std::min(1.0e12, damping * 10.0);
        }
    }

    if (!valid_focal_recovery(current, shift))
        return std::nullopt;
    return FocalShift{static_cast<float>(current.focal), static_cast<float>(shift)};
}

moge::GeometryResult postprocess_geometry(const float* affine_points,
                                          const float* mask_probabilities, float metric_scale,
                                          int32_t height, int32_t width) {
    const auto area = static_cast<std::size_t>(height) * width;
    validate_metric_scale(metric_scale);
    const auto raw_mask = make_raw_mask(affine_points, mask_probabilities, area);

    const auto recovered = recover_focal_shift(affine_points, raw_mask, height, width);
    if (!recovered)
        throw std::runtime_error("MoGe could not recover camera focal length and shift");
    const auto calibration = make_calibration(*recovered, height, width);
    auto result = make_geometry_result(height, width, area, calibration);
    populate_geometry_result(result, affine_points, raw_mask, *recovered, calibration,
                             metric_scale);
    return result;
}

} // namespace

MogePipeline::MogePipeline(std::unique_ptr<ITrtModule> model, std::string model_id)
    : model_(std::move(model)), model_id_(std::move(model_id)) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("MogePipeline: invalid model");
}

moge::GeometryResult MogePipeline::estimate_geometry(const float* pixels, int32_t height,
                                                     int32_t width) {
    validate_image_input(pixels, height, width);

    Tensor image{const_cast<float*>(pixels), {1, height, width, 3}, DType::kFloat32};
    const auto outputs = model_->forward({{"image", image}});
    const auto* points = require_float_output(outputs, "points", {1, height, width, 3});
    const auto* mask = require_float_output(outputs, "mask", {1, height, width});
    const auto* scale = require_float_output(outputs, "metric_scale", {1});
    return postprocess_geometry(points, mask, scale[0], height, width);
}

} // namespace trtmc
