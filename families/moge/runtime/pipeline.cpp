/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/moge/runtime/pipeline.h"

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
constexpr float kMinAspectRatio = 0.5F;
constexpr float kMaxAspectRatio = 2.0F;

struct ImageProfileBounds {
    int32_t min_height{0};
    int32_t min_width{0};
    int32_t max_height{0};
    int32_t max_width{0};
};

bool valid_image_profile_shape(const std::vector<int64_t>& shape) {
    constexpr auto kInt32Max = std::numeric_limits<int32_t>::max();
    return shape.size() == 4U && shape[0] == 1 && shape[1] > 0 && shape[1] <= kInt32Max &&
           shape[2] > 0 && shape[2] <= kInt32Max && shape[3] == 3;
}

ImageProfileBounds image_profile_bounds(const ITrtModule& model) {
    const auto profile = model.profile_idx();
    const auto minimum = model.input_profile_shape("image", profile, ProfileShapeSelector::kMin);
    const auto maximum = model.input_profile_shape("image", profile, ProfileShapeSelector::kMax);
    if (!valid_image_profile_shape(minimum) || !valid_image_profile_shape(maximum) ||
        minimum[1] > maximum[1] || minimum[2] > maximum[2]) {
        throw std::runtime_error("MogePipeline: invalid TensorRT image profile");
    }
    return {static_cast<int32_t>(minimum[1]), static_cast<int32_t>(minimum[2]),
            static_cast<int32_t>(maximum[1]), static_cast<int32_t>(maximum[2])};
}

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

int32_t nearest_sample_index(int32_t output_index, int32_t input_size) {
    return std::min(input_size - 1, static_cast<int32_t>(static_cast<int64_t>(output_index) *
                                                         input_size / kFocalRecoverySize));
}

bool valid_focal_neighborhood(const uint16_t* valid, int32_t height, int32_t width, int32_t y,
                              int32_t x) {
    if (y <= 0 || x <= 0 || y >= height - 1 || x >= width - 1)
        return false;
    for (int32_t neighbor_y = y - 1; neighbor_y <= y + 1; ++neighbor_y) {
        for (int32_t neighbor_x = x - 1; neighbor_x <= x + 1; ++neighbor_x) {
            const auto neighbor = static_cast<std::size_t>(neighbor_y) * width + neighbor_x;
            if (valid[neighbor] == 0)
                return false;
        }
    }
    return true;
}

std::vector<FocalSample> make_focal_samples(const float* sampled_points, const uint16_t* valid,
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
        const int32_t y = nearest_sample_index(out_y, height);
        for (int32_t out_x = 0; out_x < kFocalRecoverySize; ++out_x) {
            const int32_t x = nearest_sample_index(out_x, width);
            const auto pixel = static_cast<std::size_t>(y) * width + x;
            if (valid[pixel] == 0)
                continue;
            if (!valid_focal_neighborhood(valid, height, width, y, x))
                continue;
            const auto sample = (static_cast<std::size_t>(out_y) * kFocalRecoverySize + out_x) * 3U;
            const double px = sampled_points[sample];
            const double py = sampled_points[sample + 1U];
            const double pz = sampled_points[sample + 2U];
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

const uint16_t* require_float16_output(const TensorMap& outputs, const char* name,
                                       const std::vector<int64_t>& shape) {
    const auto iterator = outputs.find(name);
    if (iterator == outputs.end())
        throw std::runtime_error(std::string("MoGe engine did not return required output '") +
                                 name + "'");
    const auto& tensor = iterator->second;
    if (tensor.data == nullptr || tensor.dtype != DType::kFloat16 || tensor.shape != shape) {
        throw std::runtime_error(std::string("MoGe output contract mismatch for '") + name + "'");
    }
    return static_cast<const uint16_t*>(tensor.data);
}

void validate_metric_scale(float metric_scale) {
    if (!std::isfinite(metric_scale) || metric_scale <= 0.0F)
        throw std::invalid_argument("MoGe metric scale must be finite and positive");
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

GeometryResult make_geometry_result(int32_t height, int32_t width, std::size_t area,
                                    const MogeCalibration& calibration) {
    GeometryResult result;
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

void write_invalid_geometry(GeometryResult& result, std::size_t pixel, float infinity) {
    const auto point = pixel * 3U;
    result.depth[pixel] = infinity;
    result.points[point] = infinity;
    result.points[point + 1U] = infinity;
    result.points[point + 2U] = infinity;
}

void populate_geometry_result(GeometryResult& result, const float* affine_depth,
                              const uint16_t* valid_pixels, const FocalShift& recovered,
                              const MogeCalibration& calibration, float metric_scale) {
    const float infinity = std::numeric_limits<float>::infinity();
    for (int32_t y = 0; y < result.height; ++y) {
        const double v = normalized_coordinate(y, result.height, calibration.span_y);
        for (int32_t x = 0; x < result.width; ++x) {
            const auto pixel = static_cast<std::size_t>(y) * result.width + x;
            const auto point = pixel * 3U;
            const double depth_unscaled =
                static_cast<double>(affine_depth[pixel]) + recovered.shift;
            const bool valid =
                valid_metric_depth(static_cast<uint8_t>(valid_pixels[pixel] != 0), depth_unscaled);
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

bool supported_image_size(int32_t height, int32_t width, const ImageProfileBounds& profile) {
    return height >= profile.min_height && width >= profile.min_width &&
           height <= profile.max_height && width <= profile.max_width;
}

bool supported_aspect_ratio(int32_t height, int32_t width) {
    const float aspect = static_cast<float>(width) / height;
    return aspect >= kMinAspectRatio && aspect <= kMaxAspectRatio;
}

bool valid_rgb_value(float value) {
    return value >= 0.0F && value <= 1.0F;
}

void validate_image_input(const float* pixels, int32_t height, int32_t width,
                          const ImageProfileBounds& profile) {
    if (pixels == nullptr)
        throw std::invalid_argument("MoGe image pointer is null");
    if (!supported_image_size(height, width, profile))
        throw std::invalid_argument("MoGe image dimensions are outside the bundle profile");
    if (!supported_aspect_ratio(height, width))
        throw std::invalid_argument("MoGe image aspect ratio is outside the supported range");
    const auto area = static_cast<std::size_t>(height) * width;
    for (std::size_t index = 0; index < area * 3U; ++index) {
        if (!valid_rgb_value(pixels[index]))
            throw std::invalid_argument("MoGe RGB input values must be finite and in [0, 1]");
    }
}

std::optional<FocalShift> recover_focal_shift(const float* focal_samples, const uint16_t* valid,
                                              int32_t height, int32_t width) {
    const auto samples = make_focal_samples(focal_samples, valid, height, width);
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

GeometryResult postprocess_geometry(const float* affine_depth, const uint16_t* valid,
                                    const float* focal_samples, float metric_scale, int32_t height,
                                    int32_t width) {
    const auto area = static_cast<std::size_t>(height) * width;
    validate_metric_scale(metric_scale);
    const auto recovered = recover_focal_shift(focal_samples, valid, height, width);
    if (!recovered)
        throw std::runtime_error("MoGe could not recover camera focal length and shift");
    const auto calibration = make_calibration(*recovered, height, width);
    auto result = make_geometry_result(height, width, area, calibration);
    populate_geometry_result(result, affine_depth, valid, *recovered, calibration, metric_scale);
    return result;
}

} // namespace

MogePipeline::MogePipeline(std::unique_ptr<ITrtModule> model) : model_(std::move(model)) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("MogePipeline: invalid model");
    const auto profile = image_profile_bounds(*model_);
    min_image_height_ = profile.min_height;
    min_image_width_ = profile.min_width;
    max_image_height_ = profile.max_height;
    max_image_width_ = profile.max_width;
}

GeometryResult MogePipeline::estimate_geometry(const float* pixels, int32_t height, int32_t width) {
    validate_image_input(
        pixels, height, width,
        {min_image_height_, min_image_width_, max_image_height_, max_image_width_});
    Tensor image{const_cast<float*>(pixels), {1, height, width, 3}, DType::kFloat32};
    const auto outputs = model_->forward({{"image", image}});
    const auto* affine_depth = require_float_output(outputs, "affine_depth", {1, height, width});
    const auto* valid = require_float16_output(outputs, "valid", {1, height, width});
    const auto* focal_samples = require_float_output(
        outputs, "focal_samples", {1, kFocalRecoverySize, kFocalRecoverySize, 3});
    const auto* scale = require_float_output(outputs, "metric_scale", {1});
    return postprocess_geometry(affine_depth, valid, focal_samples, scale[0], height, width);
}

} // namespace trtmc
