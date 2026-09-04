/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/segformer/runtime/segment_pipeline.h"

#include "families/segformer/runtime/segformer_postprocess_cuda.h"
#include "families/segformer/runtime/segformer_postprocess_seam.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

// Extract (num_classes, H, W) from output shape, handling optional batch dim.
bool parse_segmentation_shape(const std::vector<int64_t>& shape, int32_t& num_classes,
                              int32_t& out_h, int32_t& out_w) {
    if (shape.size() == 4) {
        num_classes = static_cast<int32_t>(shape[1]);
        out_h = static_cast<int32_t>(shape[2]);
        out_w = static_cast<int32_t>(shape[3]);
    } else if (shape.size() == 3) {
        num_classes = static_cast<int32_t>(shape[0]);
        out_h = static_cast<int32_t>(shape[1]);
        out_w = static_cast<int32_t>(shape[2]);
    } else {
        return false;
    }
    return num_classes > 1 && out_h > 0 && out_w > 0;
}

// Find the logits/output tensor from the model output map.
const Tensor* find_segmentation_output(const TensorMap& outputs) {
    for (const auto& [name, tensor] : outputs) {
        if (name.find("logits") != std::string::npos || name.find("output") != std::string::npos ||
            outputs.size() == 1)
            return &tensor;
    }
    return nullptr;
}

const TensorInfo* find_segmentation_output(const std::vector<TensorInfo>& outputs) {
    for (const auto& tensor : outputs) {
        if (tensor.name.find("logits") != std::string::npos ||
            tensor.name.find("output") != std::string::npos || outputs.size() == 1)
            return &tensor;
    }
    return nullptr;
}

void check_cuda(const char* operation, cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("SegmentPipeline: ") + operation +
                                 " failed: " + cudaGetErrorString(status));
    }
}

} // namespace

// ─── SegmentPipeline ───

SegmentPipeline::SegmentPipeline(std::unique_ptr<ITrtModule> model,
                                 SegformerPreprocessConfig preprocess_config,
                                 std::string model_id_str)
    : model_(std::move(model)), preprocess_config_(std::move(preprocess_config)),
      model_id_(std::move(model_id_str)) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("SegmentPipeline: invalid model");
}

SegmentResult SegmentPipeline::segment(const float* pixels, int32_t height, int32_t width) {
    auto pixel_values = preprocess_segformer_image(pixels, height, width, preprocess_config_);

    Tensor img_t;
    img_t.data = pixel_values.data();
    img_t.shape = {3, preprocess_config_.input_image_h, preprocess_config_.input_image_w};
    img_t.dtype = DType::kFloat32;

    SegmentResult result;
    if (try_segment_logits_on_device(img_t, height, width, result))
        return result;

    auto outputs = model_->forward({{"pixel_values", img_t}});

    const Tensor* out_tensor = find_segmentation_output(outputs);
    if (!out_tensor)
        return result;

    const auto* data = static_cast<const float*>(out_tensor->data);
    int32_t num_classes = 0, out_h = 0, out_w = 0;

    if (parse_segmentation_shape(out_tensor->shape, num_classes, out_h, out_w)) {
        const auto logits_size = static_cast<std::size_t>(num_classes) * out_h * out_w;
        const std::vector<float> logits(data, data + logits_size);
        const SegformerLogitsShape logits_shape{num_classes, out_h, out_w};
        const auto status = compute_segformer_class_map_from_logits(logits, logits_shape, height,
                                                                    width, result.mask);
        if (status != SegformerPostprocessStatus::kOk)
            throw std::runtime_error("SegmentPipeline: invalid SegFormer logits shape");
        result.height = height;
        result.width = width;
    } else {
        auto n = out_tensor->numel();
        result.height = height;
        result.width = width;
        result.mask.resize(static_cast<std::size_t>(n));
        for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i)
            result.mask[i] = static_cast<int32_t>(data[i]);
    }

    return result;
}

bool SegmentPipeline::try_segment_logits_on_device(const Tensor& input, int32_t target_h,
                                                   int32_t target_w, SegmentResult& result) {
    const auto outputs = model_->output_info();
    const TensorInfo* output = find_segmentation_output(outputs);
    int32_t num_classes = 0, logits_h = 0, logits_w = 0;
    if (output == nullptr || output->dtype != DType::kFloat32 ||
        !parse_segmentation_shape(output->shape, num_classes, logits_h, logits_w)) {
        return false;
    }

    const auto target_size = static_cast<std::size_t>(target_h) * target_w;
    const std::vector<int64_t> target_shape{target_h, target_w};
    if (!device_class_map_.ok() || device_class_map_.shape() != target_shape) {
        device_class_map_ = DeviceTensor(target_shape, DType::kInt32, model_->stream());
        if (!device_class_map_.ok())
            throw std::runtime_error("SegmentPipeline: failed to allocate device class map");
    }

    const auto* logits = static_cast<const float*>(model_->device_ptr(output->name));
    if (logits == nullptr)
        return false;

    model_->forward_async({{"pixel_values", input}});
    check_cuda("GPU postprocess launch",
               launch_segformer_bilinear_argmax(
                   logits, num_classes, logits_h, logits_w, target_h, target_w,
                   static_cast<int32_t*>(device_class_map_.data()), model_->stream()));

    result.mask.resize(target_size);
    check_cuda("class-map download", cudaMemcpyAsync(result.mask.data(), device_class_map_.data(),
                                                     target_size * sizeof(int32_t),
                                                     cudaMemcpyDeviceToHost, model_->stream()));
    check_cuda("GPU postprocess synchronization", cudaStreamSynchronize(model_->stream()));
    result.height = target_h;
    result.width = target_w;
    return true;
}

} // namespace trtmc
