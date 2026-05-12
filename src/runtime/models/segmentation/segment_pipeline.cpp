#include "runtime/models/segmentation/segment_pipeline.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc {

namespace {

// Argmax over class dimension: logits[C, H, W] -> class_map[H, W].
void argmax_class_map(const float* logits, int32_t num_classes, int32_t out_h, int32_t out_w,
                      std::vector<int32_t>& class_map) {
    const auto plane_size = static_cast<std::size_t>(out_h) * out_w;
    class_map.resize(plane_size);

    for (std::size_t px = 0; px < plane_size; ++px) {
        int32_t best_class = 0;
        float best_val = -1e30F;
        for (int32_t c = 0; c < num_classes; ++c) {
            const float val = logits[static_cast<std::size_t>(c) * plane_size + px];
            if (val > best_val) {
                best_val = val;
                best_class = c;
            }
        }
        class_map[px] = best_class;
    }
}

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

} // namespace

// ─── SegmentPipeline ───

SegmentPipeline::SegmentPipeline(std::unique_ptr<TrtModule> model, std::string model_id_str)
    : model_(std::move(model)), model_id_(std::move(model_id_str)) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("SegmentPipeline: invalid model");
}

SegmentResult SegmentPipeline::segment(const float* pixels, int32_t height, int32_t width) {
    Tensor img_t;
    img_t.data = const_cast<float*>(pixels);
    img_t.shape = {3, height, width};
    img_t.dtype = DType::kFloat32;

    auto outputs = model_->forward({{"pixel_values", img_t}});
    SegmentResult result;

    const Tensor* out_tensor = find_segmentation_output(outputs);
    if (!out_tensor)
        return result;

    const auto* data = static_cast<const float*>(out_tensor->data);
    int32_t num_classes = 0, out_h = 0, out_w = 0;

    if (parse_segmentation_shape(out_tensor->shape, num_classes, out_h, out_w)) {
        result.height = out_h;
        result.width = out_w;
        argmax_class_map(data, num_classes, out_h, out_w, result.mask);
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

} // namespace trtmc
