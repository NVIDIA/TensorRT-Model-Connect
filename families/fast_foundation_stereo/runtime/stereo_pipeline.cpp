/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/fast_foundation_stereo/runtime/stereo_pipeline.h"

#include <algorithm>
#include <cstring>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <utility>

namespace trtmc {

namespace {

constexpr int32_t kInputHeight = 700;
constexpr int32_t kInputWidth = 700;
constexpr int32_t kEngineHeight = 704;
constexpr int32_t kEngineWidth = 704;
constexpr int32_t kPadTop = 2;
constexpr int32_t kPadLeft = 2;

void check_cuda(const char* operation, cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("FastFoundationStereoPipeline: ") + operation +
                                 " failed: " + cudaGetErrorString(status));
    }
}

} // namespace

void prepare_fast_foundation_stereo_image(const float* pixels, int32_t height, int32_t width,
                                          std::vector<float>& output) {
    if (pixels == nullptr)
        throw std::invalid_argument("Fast Foundation Stereo image pointer is null");
    if (height != kInputHeight || width != kInputWidth) {
        throw std::invalid_argument("Fast Foundation Stereo requires 700x700 rectified images");
    }
    output.resize(static_cast<std::size_t>(3) * kEngineHeight * kEngineWidth);
    const std::size_t channel_size = static_cast<std::size_t>(kEngineHeight) * kEngineWidth;
    for (int32_t source_y = 0; source_y < height; ++source_y) {
        const float* source_row = pixels + static_cast<std::size_t>(source_y) * width * 3;
        float* output_rows[3] = {
            output.data() + static_cast<std::size_t>(source_y + kPadTop) * kEngineWidth,
            output.data() + channel_size +
                static_cast<std::size_t>(source_y + kPadTop) * kEngineWidth,
            output.data() + 2 * channel_size +
                static_cast<std::size_t>(source_y + kPadTop) * kEngineWidth,
        };
        for (int32_t source_x = 0; source_x < width; ++source_x) {
            const std::size_t source_offset = static_cast<std::size_t>(source_x) * 3;
            const int32_t output_x = source_x + kPadLeft;
            output_rows[0][output_x] = source_row[source_offset] * 255.0F;
            output_rows[1][output_x] = source_row[source_offset + 1] * 255.0F;
            output_rows[2][output_x] = source_row[source_offset + 2] * 255.0F;
        }
        for (float* output_row : output_rows) {
            std::fill_n(output_row, kPadLeft, output_row[kPadLeft]);
            std::fill_n(output_row + kPadLeft + width, kEngineWidth - kPadLeft - width,
                        output_row[kPadLeft + width - 1]);
        }
    }
    for (int32_t channel = 0; channel < 3; ++channel) {
        float* channel_output = output.data() + static_cast<std::size_t>(channel) * channel_size;
        const std::size_t row_bytes = static_cast<std::size_t>(kEngineWidth) * sizeof(float);
        const float* first_source_row = channel_output + kPadTop * kEngineWidth;
        for (int32_t y = 0; y < kPadTop; ++y) {
            std::memcpy(channel_output + static_cast<std::size_t>(y) * kEngineWidth,
                        first_source_row, row_bytes);
        }
        const float* last_source_row = channel_output + (kPadTop + height - 1) * kEngineWidth;
        for (int32_t y = kPadTop + height; y < kEngineHeight; ++y) {
            std::memcpy(channel_output + static_cast<std::size_t>(y) * kEngineWidth,
                        last_source_row, row_bytes);
        }
    }
}

FastFoundationStereoPipeline::FastFoundationStereoPipeline(std::unique_ptr<ITrtModule> feature,
                                                           std::unique_ptr<ITrtModule> post,
                                                           std::string model_id)
    : feature_(std::move(feature)), post_(std::move(post)), model_id_(std::move(model_id)) {
    if (!feature_ || !feature_->ok() || !post_ || !post_->ok())
        throw std::runtime_error("FastFoundationStereoPipeline: invalid engine module");
    if (feature_->stream() != post_->stream())
        throw std::runtime_error(
            "FastFoundationStereoPipeline: engines must share one CUDA stream");
    const std::vector<int64_t> expected_shape{1, 1, kEngineHeight, kEngineWidth};
    if (!post_->has_output("disp") || post_->tensor_dtype("disp") != DType::kFloat32 ||
        post_->tensor_shape("disp") != expected_shape) {
        throw std::runtime_error(
            "FastFoundationStereoPipeline: disparity output contract mismatch");
    }
}

FastFoundationStereoPipeline::~FastFoundationStereoPipeline() {
    if (left_input_pinned_)
        cudaHostUnregister(left_input_.data());
    if (right_input_pinned_)
        cudaHostUnregister(right_input_.data());
}

void FastFoundationStereoPipeline::pin_input(std::vector<float>& input, bool& pinned,
                                             const char* name) {
    if (pinned)
        return;
    check_cuda(name, cudaHostRegister(input.data(), input.size() * sizeof(float),
                                      cudaHostRegisterDefault));
    pinned = true;
}

void FastFoundationStereoPipeline::upload_feature_input(const char* name,
                                                        const std::vector<float>& input) {
    void* destination = feature_->device_ptr(name);
    if (destination == nullptr) {
        throw std::runtime_error(
            std::string("FastFoundationStereoPipeline: missing feature input ") + name);
    }
    check_cuda(name, cudaMemcpyAsync(destination, input.data(), input.size() * sizeof(float),
                                     cudaMemcpyHostToDevice, feature_->stream()));
}

void FastFoundationStereoPipeline::bind_post_inputs() {
    static constexpr const char* names[] = {
        "features_left_04", "features_left_08",  "features_left_16",
        "features_left_32", "features_right_04", "stem_2x",
    };
    for (const char* name : names) {
        if (!feature_->has_output(name) || !post_->has_input(name)) {
            throw std::runtime_error(
                std::string("FastFoundationStereoPipeline: invalid feature/post tensor role for ") +
                name);
        }
        const auto feature_shape = feature_->tensor_shape(name);
        if (feature_shape != post_->tensor_shape(name) ||
            feature_->tensor_dtype(name) != post_->tensor_dtype(name)) {
            throw std::runtime_error(
                std::string("FastFoundationStereoPipeline: feature/post contract mismatch for ") +
                name);
        }
        void* pointer = feature_->device_ptr(name);
        if (pointer == nullptr)
            throw std::runtime_error(std::string("FastFoundationStereoPipeline: missing feature ") +
                                     name);
        post_->bind_external(name, pointer, feature_shape);
    }
    post_inputs_bound_ = true;
}

StereoDisparityResult FastFoundationStereoPipeline::estimate_disparity(const float* left_pixels,
                                                                       const float* right_pixels,
                                                                       int32_t height,
                                                                       int32_t width) {
    prepare_fast_foundation_stereo_image(left_pixels, height, width, left_input_);
    pin_input(left_input_, left_input_pinned_, "left input pinning");
    upload_feature_input("left", left_input_);
    prepare_fast_foundation_stereo_image(right_pixels, height, width, right_input_);
    pin_input(right_input_, right_input_pinned_, "right input pinning");
    upload_feature_input("right", right_input_);
    feature_->forward_async({});
    if (!post_inputs_bound_)
        bind_post_inputs();
    post_->forward_async({});

    const auto* disparity = static_cast<const float*>(post_->device_ptr("disp"));
    if (disparity == nullptr)
        throw std::runtime_error("FastFoundationStereoPipeline: missing disparity output");
    padded_output_.resize(static_cast<std::size_t>(kEngineHeight) * kEngineWidth);
    check_cuda("disparity download", cudaMemcpyAsync(padded_output_.data(), disparity,
                                                     padded_output_.size() * sizeof(float),
                                                     cudaMemcpyDeviceToHost, post_->stream()));
    post_->sync();

    StereoDisparityResult result;
    result.height = height;
    result.width = width;
    result.disparity.resize(static_cast<std::size_t>(height) * width);
    for (int32_t y = 0; y < height; ++y) {
        const auto* source =
            padded_output_.data() + static_cast<std::size_t>(y + kPadTop) * kEngineWidth + kPadLeft;
        auto* destination = result.disparity.data() + static_cast<std::size_t>(y) * width;
        std::transform(source, source + width, destination,
                       [](float value) { return std::max(value, 0.0F); });
    }
    return result;
}

} // namespace trtmc
