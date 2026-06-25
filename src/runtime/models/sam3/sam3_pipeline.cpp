#include "runtime/models/sam3/sam3_pipeline.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

struct Sam3ImagePreprocessPlan {
    std::vector<float> pixel_values;
    int32_t original_width{0};
    int32_t original_height{0};
};

struct Sam3RawOutputs {
    std::vector<float> masks;
    std::vector<int64_t> masks_shape;
    std::vector<float> boxes;
    std::vector<int64_t> boxes_shape;
    std::vector<float> logits;
    std::vector<int64_t> logits_shape;
    std::vector<float> presence_logits;
};

struct Sam3PostprocessGeometry {
    int32_t num_queries{0};
    int32_t mask_height{0};
    int32_t mask_width{0};
    int32_t mask_stride{0};
    std::size_t output_mask_area{0};
};

std::vector<float> copy_float_tensor(const Tensor& tensor) {
    const auto* data = static_cast<const float*>(tensor.data);
    if (data == nullptr)
        return {};
    const auto count = static_cast<std::size_t>(tensor.numel());
    return std::vector<float>(data, data + count);
}

const Tensor* require_tensor(const TensorMap& outputs, const std::string& name,
                             const char* producer) {
    const auto it = outputs.find(name);
    if (it == outputs.end() || it->second.data == nullptr)
        throw std::runtime_error(std::string("Sam3Pipeline: ") + producer + " missing output " +
                                 name);
    return &it->second;
}

float sigmoid(float value) {
    if (value >= 0.0F) {
        const float z = std::exp(-value);
        return 1.0F / (1.0F + z);
    }
    const float z = std::exp(value);
    return z / (1.0F + z);
}

std::array<float, 3> sam3_channel_values(const std::vector<float>& values,
                                         std::array<float, 3> defaults) {
    const auto count = std::min(values.size(), defaults.size());
    for (std::size_t i = 0; i < count; ++i)
        defaults[i] = values[i];
    return defaults;
}

float sample_hwc_channel(const float* pixels, int32_t height, int32_t width, int32_t channel,
                         float x, float y) {
    const float clamped_x = std::clamp(x, 0.0F, static_cast<float>(width - 1));
    const float clamped_y = std::clamp(y, 0.0F, static_cast<float>(height - 1));
    const int32_t x0 = static_cast<int32_t>(std::floor(clamped_x));
    const int32_t y0 = static_cast<int32_t>(std::floor(clamped_y));
    const int32_t x1 = std::min(x0 + 1, width - 1);
    const int32_t y1 = std::min(y0 + 1, height - 1);
    const float wx = clamped_x - static_cast<float>(x0);
    const float wy = clamped_y - static_cast<float>(y0);

    const auto idx = [width](int32_t yy, int32_t xx, int32_t cc) {
        return static_cast<std::size_t>((yy * width + xx) * 3 + cc);
    };

    const float v00 = pixels[idx(y0, x0, channel)];
    const float v01 = pixels[idx(y0, x1, channel)];
    const float v10 = pixels[idx(y1, x0, channel)];
    const float v11 = pixels[idx(y1, x1, channel)];
    const float top = v00 * (1.0F - wx) + v01 * wx;
    const float bottom = v10 * (1.0F - wx) + v11 * wx;
    return top * (1.0F - wy) + bottom * wy;
}

Sam3ImagePreprocessPlan build_sam3_preprocess_plan(const float* image_pixels, int32_t height,
                                                   int32_t width, const Sam3Config& config) {
    Sam3ImagePreprocessPlan plan;
    if (image_pixels == nullptr || height <= 0 || width <= 0 || config.image_size <= 0)
        return plan;

    const int32_t image_size = config.image_size;
    plan.original_width = width;
    plan.original_height = height;
    plan.pixel_values.assign(static_cast<std::size_t>(3) * static_cast<std::size_t>(image_size) *
                                 static_cast<std::size_t>(image_size),
                             0.0F);

    const auto mean = sam3_channel_values(config.image_mean, {0.5F, 0.5F, 0.5F});
    const auto stdv = sam3_channel_values(config.image_std, {0.5F, 0.5F, 0.5F});
    for (int32_t y = 0; y < image_size; ++y) {
        const float src_y = (static_cast<float>(y) + 0.5F) * static_cast<float>(height) /
                                static_cast<float>(image_size) -
                            0.5F;
        for (int32_t x = 0; x < image_size; ++x) {
            const float src_x = (static_cast<float>(x) + 0.5F) * static_cast<float>(width) /
                                    static_cast<float>(image_size) -
                                0.5F;
            for (int32_t c = 0; c < 3; ++c) {
                const float value =
                    sample_hwc_channel(image_pixels, height, width, c, src_x, src_y);
                const float normalized =
                    (value - mean[static_cast<std::size_t>(c)]) / stdv[static_cast<std::size_t>(c)];
                plan.pixel_values[static_cast<std::size_t>(c) *
                                      static_cast<std::size_t>(image_size) *
                                      static_cast<std::size_t>(image_size) +
                                  static_cast<std::size_t>(y) *
                                      static_cast<std::size_t>(image_size) +
                                  static_cast<std::size_t>(x)] = normalized;
            }
        }
    }

    return plan;
}

std::vector<int64_t> batched_text_shape(const std::vector<int64_t>& shape) {
    if (shape.size() == 2)
        return {1, shape[0], shape[1]};
    return shape;
}

void add_required_vision_inputs(TensorMap& inputs, const TensorMap& outputs) {
    for (int32_t level = 0; level < 3; ++level) {
        const std::string hidden_name = "sam3_fpn_hidden_" + std::to_string(level);
        const std::string pos_name = "sam3_fpn_position_" + std::to_string(level);
        inputs[hidden_name] = *require_tensor(outputs, hidden_name, "vision encoder");
        inputs[pos_name] = *require_tensor(outputs, pos_name, "vision encoder");
    }
}

int32_t query_count_from_logits_shape(const std::vector<int64_t>& shape) {
    if (shape.size() == 2)
        return static_cast<int32_t>(shape[1]);
    if (shape.size() == 1)
        return static_cast<int32_t>(shape[0]);
    return 0;
}

int32_t mask_query_stride(const std::vector<int64_t>& shape, int32_t num_queries, int32_t& height,
                          int32_t& width) {
    if (shape.size() == 4) {
        num_queries = static_cast<int32_t>(shape[1]);
        height = static_cast<int32_t>(shape[2]);
        width = static_cast<int32_t>(shape[3]);
    } else if (shape.size() == 3) {
        num_queries = static_cast<int32_t>(shape[0]);
        height = static_cast<int32_t>(shape[1]);
        width = static_cast<int32_t>(shape[2]);
    } else {
        return 0;
    }
    if (num_queries <= 0 || height <= 0 || width <= 0)
        return 0;
    return height * width;
}

std::vector<float> resize_sigmoid_mask_to_original(const float* src, int32_t src_h, int32_t src_w,
                                                   int32_t dst_h, int32_t dst_w,
                                                   float mask_threshold) {
    std::vector<float> out(static_cast<std::size_t>(dst_h) * static_cast<std::size_t>(dst_w), 0.0F);
    if (src == nullptr || src_h <= 0 || src_w <= 0 || dst_h <= 0 || dst_w <= 0)
        return out;

    const auto at = [src, src_w](int32_t yy, int32_t xx) {
        return sigmoid(src[static_cast<std::size_t>(yy) * static_cast<std::size_t>(src_w) +
                           static_cast<std::size_t>(xx)]);
    };
    for (int32_t y = 0; y < dst_h; ++y) {
        const float src_y =
            (static_cast<float>(y) + 0.5F) * static_cast<float>(src_h) / static_cast<float>(dst_h) -
            0.5F;
        const float clamped_y = std::clamp(src_y, 0.0F, static_cast<float>(src_h - 1));
        const int32_t y0 = static_cast<int32_t>(std::floor(clamped_y));
        const int32_t y1 = std::min(y0 + 1, src_h - 1);
        const float wy = clamped_y - static_cast<float>(y0);
        for (int32_t x = 0; x < dst_w; ++x) {
            const float src_x = (static_cast<float>(x) + 0.5F) * static_cast<float>(src_w) /
                                    static_cast<float>(dst_w) -
                                0.5F;
            const float clamped_x = std::clamp(src_x, 0.0F, static_cast<float>(src_w - 1));
            const int32_t x0 = static_cast<int32_t>(std::floor(clamped_x));
            const int32_t x1 = std::min(x0 + 1, src_w - 1);
            const float wx = clamped_x - static_cast<float>(x0);
            const float top = at(y0, x0) * (1.0F - wx) + at(y0, x1) * wx;
            const float bottom = at(y1, x0) * (1.0F - wx) + at(y1, x1) * wx;
            const float value = top * (1.0F - wy) + bottom * wy;
            out[static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_w) +
                static_cast<std::size_t>(x)] = value > mask_threshold ? 1.0F : 0.0F;
        }
    }
    return out;
}

Sam3RawOutputs parse_sam3_raw_outputs(const TensorMap& outputs) {
    Sam3RawOutputs raw;
    const auto* masks = require_tensor(outputs, "pred_masks", "core engine");
    const auto* boxes = require_tensor(outputs, "pred_boxes", "core engine");
    const auto* logits = require_tensor(outputs, "pred_logits", "core engine");
    raw.masks = copy_float_tensor(*masks);
    raw.masks_shape = masks->shape;
    raw.boxes = copy_float_tensor(*boxes);
    raw.boxes_shape = boxes->shape;
    raw.logits = copy_float_tensor(*logits);
    raw.logits_shape = logits->shape;

    const auto presence_it = outputs.find("presence_logits");
    if (presence_it != outputs.end() && presence_it->second.data != nullptr)
        raw.presence_logits = copy_float_tensor(presence_it->second);
    return raw;
}

bool init_sam3_postprocess_geometry(const Sam3RawOutputs& raw, const Sam3ImagePreprocessPlan& image,
                                    Sam3PostprocessGeometry& geometry) {
    int32_t mask_h = 0;
    int32_t mask_w = 0;
    const int32_t num_queries = query_count_from_logits_shape(raw.logits_shape);
    const int32_t mask_stride = mask_query_stride(raw.masks_shape, num_queries, mask_h, mask_w);
    if (num_queries <= 0)
        return false;
    if (mask_stride <= 0)
        return false;
    if (image.original_height <= 0)
        return false;
    if (image.original_width <= 0)
        return false;

    geometry.num_queries = num_queries;
    geometry.mask_height = mask_h;
    geometry.mask_width = mask_w;
    geometry.mask_stride = mask_stride;
    geometry.output_mask_area = static_cast<std::size_t>(image.original_height) *
                                static_cast<std::size_t>(image.original_width);
    return true;
}

void append_sam3_query_if_kept(const Sam3RawOutputs& raw, int32_t query,
                               const Sam3PostprocessGeometry& geometry,
                               const Sam3ImagePreprocessPlan& image, const Sam3Config& config,
                               float presence_score, PromptedSegmentationResult& result) {
    const auto box_offset = static_cast<std::size_t>(query) * 4U;
    if (raw.boxes.size() < box_offset + 4U)
        return;

    const float score = sigmoid(raw.logits[static_cast<std::size_t>(query)]) * presence_score;
    if (score <= config.score_threshold)
        return;

    const auto mask_offset =
        static_cast<std::size_t>(query) * static_cast<std::size_t>(geometry.mask_stride);
    if (raw.masks.size() < mask_offset + static_cast<std::size_t>(geometry.mask_stride))
        return;

    auto resized = resize_sigmoid_mask_to_original(
        raw.masks.data() + mask_offset, geometry.mask_height, geometry.mask_width,
        image.original_height, image.original_width, config.mask_threshold);
    if (resized.size() != geometry.output_mask_area)
        return;

    result.masks.insert(result.masks.end(), resized.begin(), resized.end());
    result.iou_scores.push_back(score);
    result.boxes.push_back(raw.boxes[box_offset] * static_cast<float>(image.original_width));
    result.boxes.push_back(raw.boxes[box_offset + 1U] * static_cast<float>(image.original_height));
    result.boxes.push_back(raw.boxes[box_offset + 2U] * static_cast<float>(image.original_width));
    result.boxes.push_back(raw.boxes[box_offset + 3U] * static_cast<float>(image.original_height));
    ++result.num_masks;
}

PromptedSegmentationResult postprocess_sam3_raw_outputs(Sam3RawOutputs raw,
                                                        const Sam3ImagePreprocessPlan& image,
                                                        const Sam3Config& config) {
    PromptedSegmentationResult result;
    Sam3PostprocessGeometry geometry;
    if (!init_sam3_postprocess_geometry(raw, image, geometry))
        return result;

    const float presence_score =
        raw.presence_logits.empty() ? 1.0F : sigmoid(raw.presence_logits.front());
    for (int32_t query = 0; query < geometry.num_queries; ++query)
        append_sam3_query_if_kept(raw, query, geometry, image, config, presence_score, result);

    result.height = image.original_height;
    result.width = image.original_width;
    return result;
}

} // namespace

Sam3Pipeline::Sam3Pipeline(std::unique_ptr<TrtModule> text_encoder,
                           std::shared_ptr<ITokenizer> tokenizer, Sam3Config config,
                           std::string model_id_str)
    : Sam3Pipeline(std::move(text_encoder), nullptr, nullptr, std::move(tokenizer),
                   std::move(config), std::move(model_id_str)) {}

Sam3Pipeline::Sam3Pipeline(std::unique_ptr<TrtModule> text_encoder,
                           std::unique_ptr<TrtModule> vision_encoder,
                           std::shared_ptr<ITokenizer> tokenizer, Sam3Config config,
                           std::string model_id_str)
    : Sam3Pipeline(std::move(text_encoder), std::move(vision_encoder), nullptr,
                   std::move(tokenizer), std::move(config), std::move(model_id_str)) {}

Sam3Pipeline::Sam3Pipeline(std::unique_ptr<TrtModule> text_encoder,
                           std::unique_ptr<TrtModule> vision_encoder,
                           std::unique_ptr<TrtModule> core_engine,
                           std::shared_ptr<ITokenizer> tokenizer, Sam3Config config,
                           std::string model_id_str)
    : text_encoder_(std::move(text_encoder)), vision_encoder_(std::move(vision_encoder)),
      core_engine_(std::move(core_engine)), tokenizer_(std::move(tokenizer)),
      config_(std::move(config)), model_id_(std::move(model_id_str)) {
    if (!text_encoder_ || !text_encoder_->ok())
        throw std::runtime_error("Sam3Pipeline: invalid text_encoder");
    if (vision_encoder_ && !vision_encoder_->ok())
        throw std::runtime_error("Sam3Pipeline: invalid vision_encoder");
    if (core_engine_ && !core_engine_->ok())
        throw std::runtime_error("Sam3Pipeline: invalid core_engine");
    if (!tokenizer_)
        throw std::runtime_error("Sam3Pipeline: tokenizer is required for text-prompt PCS");
    if (config_.text_max_position_embeddings <= 0)
        throw std::runtime_error("Sam3Pipeline: invalid text max position embeddings");
}

PromptedSegmentationResult Sam3Pipeline::segment_prompted(const float* image_pixels,
                                                          int32_t image_height, int32_t image_width,
                                                          float point_x, float point_y,
                                                          bool is_foreground) {
    (void)image_pixels;
    (void)image_height;
    (void)image_width;
    (void)point_x;
    (void)point_y;
    (void)is_foreground;
    throw std::runtime_error(
        "Sam3Pipeline: SAM3 image PCS requires a text prompt; use segment_prompted_text()");
}

PromptedSegmentationResult Sam3Pipeline::segment_prompted_text(const float* image_pixels,
                                                               int32_t image_height,
                                                               int32_t image_width,
                                                               const std::string& text_prompt) {
    auto text = encode_text_prompt(text_prompt);
    if (text.features.empty() || text.hidden_states.empty()) {
        throw std::runtime_error("Sam3Pipeline: SAM3 text encoder produced no features");
    }

    auto image = build_sam3_preprocess_plan(image_pixels, image_height, image_width, config_);
    if (image.pixel_values.empty()) {
        throw std::runtime_error("Sam3Pipeline: invalid image for SAM3 image preprocessing");
    }

    if (!vision_encoder_) {
        throw std::runtime_error("Sam3Pipeline: missing vision_engine_plan for SAM3 image PCS");
    }

    Tensor image_tensor;
    image_tensor.data = image.pixel_values.data();
    image_tensor.shape = {1, 3, config_.image_size, config_.image_size};
    image_tensor.dtype = DType::kFloat32;
    const auto vision_outputs = vision_encoder_->forward({{"pixel_values", image_tensor}});
    if (vision_outputs.empty()) {
        throw std::runtime_error("Sam3Pipeline: SAM3 vision encoder produced no outputs");
    }

    if (!core_engine_) {
        TensorMap required;
        add_required_vision_inputs(required, vision_outputs);
        throw std::runtime_error(
            "Sam3Pipeline: missing sam3_core_engine_plan for SAM3 DETR/mask/scoring");
    }

    Tensor text_features_tensor;
    text_features_tensor.data = text.features.data();
    text_features_tensor.shape = batched_text_shape(text.features_shape);
    text_features_tensor.dtype = DType::kFloat32;

    Tensor text_mask_tensor;
    text_mask_tensor.data = text.attention_mask.data();
    text_mask_tensor.shape = {1, config_.text_max_position_embeddings};
    text_mask_tensor.dtype = DType::kInt32;

    TensorMap core_inputs;
    core_inputs["sam3_text_features"] = text_features_tensor;
    core_inputs["sam3_text_attention_mask"] = text_mask_tensor;
    add_required_vision_inputs(core_inputs, vision_outputs);

    auto raw_outputs = core_engine_->forward(core_inputs);
    return postprocess_sam3_raw_outputs(parse_sam3_raw_outputs(raw_outputs), image, config_);
}

Sam3TextFeatures Sam3Pipeline::encode_text_prompt_for_test(const std::string& text_prompt) const {
    return encode_text_prompt(text_prompt);
}

Sam3TextFeatures Sam3Pipeline::encode_text_prompt(const std::string& text_prompt) const {
    auto ids = tokenizer_->encode(text_prompt);
    const auto max_seq = static_cast<std::size_t>(config_.text_max_position_embeddings);
    if (ids.size() > max_seq) {
        ids.resize(max_seq);
    }

    std::vector<int32_t> input_ids(max_seq, config_.text_pad_token_id);
    std::vector<int32_t> attention_mask(max_seq, 0);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        input_ids[i] = ids[i];
        attention_mask[i] = 1;
    }

    Tensor ids_tensor;
    ids_tensor.data = input_ids.data();
    ids_tensor.shape = {config_.text_max_position_embeddings};
    ids_tensor.dtype = DType::kInt32;

    Tensor mask_tensor;
    mask_tensor.data = attention_mask.data();
    mask_tensor.shape = {config_.text_max_position_embeddings};
    mask_tensor.dtype = DType::kInt32;

    const auto outputs =
        text_encoder_->forward({{"input_ids", ids_tensor}, {"attention_mask", mask_tensor}});
    const auto* features = require_tensor(outputs, "sam3_text_features", "text encoder");
    const auto* hidden = require_tensor(outputs, "sam3_text_hidden_states", "text encoder");

    Sam3TextFeatures result;
    result.features = copy_float_tensor(*features);
    result.features_shape = features->shape;
    result.hidden_states = copy_float_tensor(*hidden);
    result.hidden_states_shape = hidden->shape;
    result.attention_mask = std::move(attention_mask);
    return result;
}

} // namespace trtmc
