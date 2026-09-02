/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/fl2va_runtime.h"

#include "runtime/models/minimax_h3/torch_cuda_normal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::minimax_h3 {
namespace {

constexpr int32_t kVisionPatch = 16;
constexpr int32_t kVisionTemporalPatch = 2;
constexpr int32_t kVisionMerge = 2;
constexpr int32_t kVisionPatchWidth = 3 * kVisionTemporalPatch * kVisionPatch * kVisionPatch;
constexpr int32_t kVisionTableSize = 48;
constexpr int32_t kTextDim = 5120;
constexpr int32_t kLabelTokens = 6;
constexpr int32_t kVisionStartToken = 151652;
constexpr int32_t kVisionEndToken = 151653;
constexpr int32_t kImagePadToken = 151655;
constexpr int32_t kLatentChannels = 24;
constexpr int32_t kPosteriorChannels = 48;
constexpr int32_t kTileSize = 256;
constexpr int32_t kLatentTileSize = 16;
constexpr int32_t kPatchHeight = 2;
constexpr int32_t kPatchWidth = 2;
constexpr int32_t kPatchDim = 96;
constexpr uint64_t kKeyframeSeed = 42;

constexpr std::array<float, kLatentChannels> kLatentMean = {
    0.8580903411F,  -0.9606591463F, 1.0661640167F,  -0.5090325475F, -0.2727581859F, -1.3675414324F,
    -0.2553254962F, -0.2690755427F, -0.5376840830F, -0.0464097299F, 0.6657370329F,  0.1969012767F,
    -0.5460608006F, -0.4035342038F, -0.2368302494F, 0.2592845261F,  -0.3013394475F, 0.2113419920F,
    -1.1206848621F, 0.3581933379F,  -0.0422514379F, 0.2604829967F,  0.2286409289F,  0.7056031823F};
constexpr std::array<float, kLatentChannels> kLatentStd = {
    1.2223774195F, 1.2767263651F, 1.6831774712F, 1.7549455166F, 1.5636216402F, 2.1941435337F,
    0.9653137922F, 1.0569885969F, 0.8419489264F, 0.7729952931F, 1.8955937624F, 0.9468418360F,
    0.7996809483F, 0.4498890042F, 0.7197399735F, 0.6936293244F, 2.9610950947F, 2.7694199085F,
    3.0496184826F, 2.1088054180F, 3.2762262821F, 3.1627357006F, 2.2816812992F, 2.6127843857F};
constexpr std::array<float, 3> kPixelMean = {0.485F, 0.456F, 0.406F};
constexpr std::array<float, 3> kPixelStd = {0.229F, 0.224F, 0.225F};

std::size_t checked_product(std::initializer_list<std::size_t> values, const char* label) {
    std::size_t result = 1;
    for (std::size_t value : values) {
        if (value != 0 && result > std::numeric_limits<std::size_t>::max() / value)
            throw std::overflow_error(std::string("MiniMax-H3 ") + label + " size overflow");
        result *= value;
    }
    return result;
}

void validate_image(const VideoImageInput& image) {
    if (image.height <= 0 || image.width <= 0 || image.channels != 3 || image.height % 32 != 0 ||
        image.width % 32 != 0) {
        throw std::invalid_argument(
            "MiniMax-H3 FL2VA prepared image must be RGB on a multiple-of-32 canvas");
    }
    const std::size_t expected = checked_product(
        {static_cast<std::size_t>(image.height), static_cast<std::size_t>(image.width), 3U},
        "FL2VA image");
    if (image.pixels.size() != expected)
        throw std::invalid_argument("MiniMax-H3 FL2VA image buffer size is invalid");
    for (float value : image.pixels) {
        if (!std::isfinite(value) || value < 0.0F || value > 1.0F)
            throw std::invalid_argument("MiniMax-H3 FL2VA pixels must be finite in [0, 1]");
    }
}

const Tensor& require_output(const TensorMap& outputs, const char* name) {
    const auto it = outputs.find(name);
    if (it == outputs.end() || it->second.data == nullptr)
        throw std::runtime_error(std::string("MiniMax-H3 FL2VA plan did not return ") + name);
    return it->second;
}

std::vector<float> copy_float_output(const TensorMap& outputs, const char* name,
                                     std::size_t expected) {
    const Tensor& tensor = require_output(outputs, name);
    if (tensor.dtype != DType::kFloat32 || tensor.numel() != expected)
        throw std::runtime_error(std::string("MiniMax-H3 FL2VA invalid ") + name + " output");
    const auto* data = static_cast<const float*>(tensor.data);
    return std::vector<float>(data, data + expected);
}

void require_binding(ITrtModule& module, const char* name, bool input, DType dtype) {
    if ((input ? module.has_input(name) : module.has_output(name)) == false ||
        module.tensor_dtype(name) != dtype) {
        throw std::runtime_error(std::string("MiniMax-H3 FL2VA plan ABI mismatch for ") + name);
    }
}

bool dynamic_input_matches(ITrtModule& module, const char* name, DType dtype,
                           const std::vector<int64_t>& minimum, const std::vector<int64_t>& optimum,
                           const std::vector<int64_t>& maximum) {
    return module.has_input(name) && module.input_is_dynamic(name) &&
           module.tensor_dtype(name) == dtype && module.optimization_profile_count() == 1 &&
           module.input_profile_shape(name, 0, ProfileShapeSelector::kMin) == minimum &&
           module.input_profile_shape(name, 0, ProfileShapeSelector::kOpt) == optimum &&
           module.input_profile_shape(name, 0, ProfileShapeSelector::kMax) == maximum;
}

void require_dynamic_input(ITrtModule& module, const char* name, DType dtype,
                           const std::vector<int64_t>& minimum, const std::vector<int64_t>& optimum,
                           const std::vector<int64_t>& maximum) {
    if (!dynamic_input_matches(module, name, dtype, minimum, optimum, maximum))
        throw std::runtime_error(std::string("MiniMax-H3 FL2VA dynamic input ABI mismatch for ") +
                                 name);
}

void require_static_input(ITrtModule& module, const char* name, DType dtype,
                          const std::vector<int64_t>& shape) {
    if (!module.has_input(name) || module.input_is_dynamic(name) ||
        module.tensor_dtype(name) != dtype || module.tensor_shape(name) != shape) {
        throw std::runtime_error(std::string("MiniMax-H3 FL2VA static input ABI mismatch for ") +
                                 name);
    }
}

void require_profile_output(ITrtModule& module, const char* name, DType dtype,
                            const std::vector<int64_t>& maximum) {
    if (!module.has_output(name) || module.tensor_dtype(name) != dtype ||
        module.tensor_shape(name) != maximum) {
        throw std::runtime_error(std::string("MiniMax-H3 FL2VA output ABI mismatch for ") + name);
    }
}

void require_io_counts(ITrtModule& module, std::size_t inputs, std::size_t outputs,
                       const char* label) {
    if (!module.ok() || module.optimization_profile_count() != 1 ||
        module.input_info().size() != inputs || module.output_info().size() != outputs) {
        throw std::runtime_error(std::string("MiniMax-H3 FL2VA ") + label +
                                 " plan has an unexpected I/O contract");
    }
}

void append_text_positions(std::array<std::vector<int32_t>, 3>& axes, int32_t& current,
                           int32_t length) {
    for (int32_t index = 0; index < length; ++index) {
        for (auto& axis : axes)
            axis.push_back(current + index);
    }
    current += length;
}

std::size_t tile_index(int32_t tile, int32_t channel, int32_t y, int32_t x) {
    return (
        ((static_cast<std::size_t>(tile) * kPosteriorChannels + channel) * kLatentTileSize + y) *
            kLatentTileSize +
        x);
}

float half_round(float value) {
    const __half rounded = __float2half_rn(value);
    return __half2float(rounded);
}

} // namespace

void validate_fl2va_plan(ITrtModule& module, Fl2vaPlanKind kind) {
    if (kind == Fl2vaPlanKind::kVisionEncoder) {
        require_io_counts(module, 4, 4, "vision encoder");
        constexpr int64_t kSmallMaxPatches = 4176;
        constexpr int64_t kSupersetMaxPatches = 65536;
        const auto matches_envelope = [&](int64_t maximum) {
            return dynamic_input_matches(module, "pixel_values", DType::kFloat32, {2040, 1536},
                                         {4032, 1536}, {maximum, 1536});
        };
        const int64_t maximum =
            matches_envelope(kSmallMaxPatches)
                ? kSmallMaxPatches
                : (matches_envelope(kSupersetMaxPatches) ? kSupersetMaxPatches : 0);
        if (maximum == 0)
            throw std::runtime_error(
                "MiniMax-H3 FL2VA vision plan is neither the exact public nor Ref2VA-superset "
                "profile");
        require_dynamic_input(module, "interp_indices", DType::kInt32, {2040, 4}, {4032, 4},
                              {maximum, 4});
        require_dynamic_input(module, "interp_weights", DType::kFloat32, {2040, 4}, {4032, 4},
                              {maximum, 4});
        require_dynamic_input(module, "vision_position_ids", DType::kInt32, {2040, 2}, {4032, 2},
                              {maximum, 2});
        for (const char* name : {"vision_embeds", "deepstack_0", "deepstack_1", "deepstack_2"})
            require_profile_output(module, name, DType::kFloat32, {maximum / 4, kTextDim});
        return;
    }
    if (kind == Fl2vaPlanKind::kTextEncoder) {
        require_io_counts(module, 9, 1, "text encoder");
        constexpr int64_t kSmallMaxRows = 2641;
        constexpr int64_t kSmallMaxVisionRows = 2088;
        constexpr int64_t kSupersetMaxRows = 262144;
        const auto matches_envelope = [&](int64_t maximum) {
            return dynamic_input_matches(module, "input_ids", DType::kInt32, {1}, {1144},
                                         {maximum});
        };
        const int64_t maximum = matches_envelope(kSmallMaxRows)
                                    ? kSmallMaxRows
                                    : (matches_envelope(kSupersetMaxRows) ? kSupersetMaxRows : 0);
        if (maximum == 0)
            throw std::runtime_error(
                "MiniMax-H3 FL2VA text plan is neither the exact public nor Ref2VA-superset "
                "profile");
        const int64_t maximum_vision =
            maximum == kSmallMaxRows ? kSmallMaxVisionRows : kSupersetMaxRows;
        require_dynamic_input(module, "mrope_position_ids", DType::kInt32, {3, 1}, {3, 1144},
                              {3, maximum});
        require_dynamic_input(module, "vision_mask", DType::kFloat32, {1, 1}, {1144, 1},
                              {maximum, 1});
        require_static_input(module, "vision_count", DType::kInt32, {1});
        require_dynamic_input(module, "vision_row_indices", DType::kInt32, {1}, {1008},
                              {maximum_vision});
        for (const char* name : {"vision_embeds", "deepstack_0", "deepstack_1", "deepstack_2"})
            require_dynamic_input(module, name, DType::kFloat32, {1, kTextDim}, {1008, kTextDim},
                                  {maximum_vision, kTextDim});
        require_profile_output(module, "encoder_hidden_states", DType::kFloat32,
                               {maximum, kTextDim});
        return;
    }
    if (kind == Fl2vaPlanKind::kKeyframeVaeEncoder) {
        require_io_counts(module, 1, 1, "keyframe VAE encoder");
        require_dynamic_input(module, "pixel_tiles", DType::kFloat32,
                              {1, 3, 1, kTileSize, kTileSize}, {28, 3, 1, kTileSize, kTileSize},
                              {33, 3, 1, kTileSize, kTileSize});
        require_profile_output(module, "posterior_parameter_tiles", DType::kFloat32,
                               {33, kPosteriorChannels, 1, kLatentTileSize, kLatentTileSize});
        return;
    }
    throw std::invalid_argument("MiniMax-H3 FL2VA plan kind is invalid");
}

Fl2vaConditioningResult run_fl2va_conditioning(const VideoGenerationRequest& request,
                                               int32_t output_height, int32_t output_width,
                                               int32_t output_frames, ITokenizer& tokenizer,
                                               const Fl2vaPlanLoader& loader) {
    if (request.mode != VideoGenerationMode::kFirstLastFrameToVideoAudio ||
        (!request.first_frame && !request.last_frame) || !request.references.empty()) {
        throw std::invalid_argument(
            "MiniMax-H3 FL2VA structured request needs an endpoint and no references");
    }
    if (!request.config.negative_prompt.empty())
        throw std::invalid_argument(
            "MiniMax-H3 is guidance-distilled and does not accept negative_prompt");
    auto keyframes = prepare_minimax_h3_keyframes(request.first_frame, request.last_frame,
                                                  output_height, output_width, output_frames);
    auto result = run_fl2va_conditioning(request.prompt, keyframes, tokenizer, loader);
    result.keyframes = std::move(keyframes);
    return result;
}

Fl2vaConditioningResult run_fl2va_conditioning(const std::string& prompt,
                                               const MiniMaxH3PreparedKeyframes& keyframes,
                                               ITokenizer& tokenizer,
                                               const Fl2vaPlanLoader& loader) {
    if (!loader)
        throw std::invalid_argument("MiniMax-H3 FL2VA plan loader is missing");
    if (keyframes.images.empty() || keyframes.images.size() > 2U ||
        keyframes.images.size() != keyframes.anchors.size()) {
        throw std::invalid_argument("MiniMax-H3 FL2VA prepared keyframes are inconsistent");
    }
    const int32_t height = keyframes.images.front().height;
    const int32_t width = keyframes.images.front().width;
    for (const auto& image : keyframes.images) {
        validate_image(image);
        if (image.height != height || image.width != width)
            throw std::invalid_argument(
                "MiniMax-H3 FL2VA prepared keyframes do not share a canvas");
    }

    Fl2vaConditioningResult result;
    auto presentation = make_fl2va_text_presentation(
        prompt, static_cast<int32_t>(keyframes.images.size()), height, width, tokenizer);

    result.keyframe_latents.reserve(keyframes.images.size());
    {
        auto module = loader("fl2va_keyframe_vae_encoder_plan");
        if (!module)
            throw std::runtime_error("MiniMax-H3 FL2VA keyframe VAE plan is missing");
        module->set_timing_label("fl2va_keyframe_vae_encoder_plan");
        for (const auto& image : keyframes.images)
            result.keyframe_latents.push_back(run_fl2va_keyframe_vae_encoder(*module, image));
        module->sync();
    }

    Fl2vaVisionFeatures compact_features;
    {
        auto module = loader("vision_encoder_plan");
        if (!module)
            throw std::runtime_error("MiniMax-H3 FL2VA vision plan is missing");
        module->set_timing_label("vision_encoder_plan");
        const auto append = [](std::vector<float>& target, const std::vector<float>& source) {
            target.insert(target.end(), source.begin(), source.end());
        };
        for (const auto& image : keyframes.images) {
            auto features = run_fl2va_vision_encoder(*module, make_fl2va_vision_inputs(image));
            compact_features.rows += features.rows;
            append(compact_features.vision_embeds, features.vision_embeds);
            append(compact_features.deepstack_0, features.deepstack_0);
            append(compact_features.deepstack_1, features.deepstack_1);
            append(compact_features.deepstack_2, features.deepstack_2);
        }
        module->sync();
    }
    {
        auto module = loader("text_encoder_plan");
        if (!module)
            throw std::runtime_error("MiniMax-H3 FL2VA text plan is missing");
        module->set_timing_label("text_encoder_plan");
        result.text_embeddings = run_fl2va_text_encoder(*module, presentation, compact_features);
        module->sync();
    }
    result.text_token_tags = std::move(presentation.token_tags);
    return result;
}

Fl2vaTextPresentation make_fl2va_text_presentation(const std::string& prompt,
                                                   int32_t keyframe_count, int32_t height,
                                                   int32_t width, const ITokenizer& tokenizer) {
    if (keyframe_count < 1 || keyframe_count > 2)
        throw std::invalid_argument("MiniMax-H3 FL2VA requires one or two keyframes");
    if (height <= 0 || width <= 0 || height % 32 != 0 || width % 32 != 0)
        throw std::invalid_argument("MiniMax-H3 FL2VA text canvas must be divisible by 32");

    const auto prompt_ids = tokenizer.encode(prompt);
    const int32_t vision_height = height / 32;
    const int32_t vision_width = width / 32;
    const int32_t vision_rows = vision_height * vision_width;
    Fl2vaTextPresentation result;
    result.keyframe_count = keyframe_count;
    result.vision_rows_per_keyframe = vision_rows;

    for (int32_t image = 0; image < keyframe_count; ++image) {
        const auto label = tokenizer.encode("<Picture " + std::to_string(image + 1) + ">: ");
        if (label.size() != static_cast<std::size_t>(kLabelTokens))
            throw std::runtime_error(
                "MiniMax-H3 tokenizer does not match the released six-token picture labels");
        result.input_ids.insert(result.input_ids.end(), label.begin(), label.end());
        result.token_tags.insert(result.token_tags.end(), label.size(), 1);
        result.vision_mask.insert(result.vision_mask.end(), label.size(), 0.0F);

        result.input_ids.push_back(kVisionStartToken);
        result.token_tags.push_back(0);
        result.vision_mask.push_back(0.0F);
        for (int32_t row = 0; row < vision_rows; ++row) {
            result.vision_row_indices.push_back(static_cast<int32_t>(result.input_ids.size()));
            result.input_ids.push_back(kImagePadToken);
            result.token_tags.push_back(0);
            result.vision_mask.push_back(1.0F);
        }
        result.input_ids.push_back(kVisionEndToken);
        result.token_tags.push_back(0);
        result.vision_mask.push_back(0.0F);
    }
    result.input_ids.insert(result.input_ids.end(), prompt_ids.begin(), prompt_ids.end());
    result.token_tags.insert(result.token_tags.end(), prompt_ids.size(), 1);
    result.vision_mask.insert(result.vision_mask.end(), prompt_ids.size(), 0.0F);

    if (result.input_ids.empty() || result.input_ids.size() > 2641U)
        throw std::invalid_argument("MiniMax-H3 FL2VA presentation exceeds 2641 text rows");

    std::array<std::vector<int32_t>, 3> axes;
    int32_t current = 0;
    for (int32_t image = 0; image < keyframe_count; ++image) {
        append_text_positions(axes, current, kLabelTokens + 1 + (image > 0 ? 1 : 0));
        for (int32_t row = 0; row < vision_height; ++row) {
            for (int32_t column = 0; column < vision_width; ++column) {
                axes[0].push_back(current);
                axes[1].push_back(current + row);
                axes[2].push_back(current + column);
            }
        }
        current += std::max(vision_height, vision_width);
    }
    append_text_positions(axes, current, 1 + static_cast<int32_t>(prompt_ids.size()));
    for (const auto& axis : axes) {
        if (axis.size() != result.input_ids.size())
            throw std::logic_error("MiniMax-H3 FL2VA MRoPE row accounting failed");
        result.mrope_position_ids.insert(result.mrope_position_ids.end(), axis.begin(), axis.end());
    }
    return result;
}

Fl2vaVisionInputs make_fl2va_vision_inputs(const VideoImageInput& image) {
    validate_image(image);
    const int32_t grid_height = image.height / kVisionPatch;
    const int32_t grid_width = image.width / kVisionPatch;
    Fl2vaVisionInputs result;
    result.patch_rows = grid_height * grid_width;
    result.pixel_values.resize(static_cast<std::size_t>(result.patch_rows) * kVisionPatchWidth);
    result.interp_indices.resize(static_cast<std::size_t>(result.patch_rows) * 4);
    result.interp_weights.resize(static_cast<std::size_t>(result.patch_rows) * 4);
    result.vision_position_ids.resize(static_cast<std::size_t>(result.patch_rows) * 2);

    int32_t patch_row = 0;
    for (int32_t merge_y = 0; merge_y < grid_height; merge_y += kVisionMerge) {
        for (int32_t merge_x = 0; merge_x < grid_width; merge_x += kVisionMerge) {
            for (int32_t inner_y = 0; inner_y < kVisionMerge; ++inner_y) {
                for (int32_t inner_x = 0; inner_x < kVisionMerge; ++inner_x) {
                    const int32_t patch_y = merge_y + inner_y;
                    const int32_t patch_x = merge_x + inner_x;
                    result.vision_position_ids[static_cast<std::size_t>(patch_row) * 2] = patch_y;
                    result.vision_position_ids[static_cast<std::size_t>(patch_row) * 2 + 1] =
                        patch_x;

                    std::size_t column = static_cast<std::size_t>(patch_row) * kVisionPatchWidth;
                    for (int32_t channel = 0; channel < 3; ++channel) {
                        for (int32_t temporal = 0; temporal < kVisionTemporalPatch; ++temporal) {
                            (void)temporal;
                            for (int32_t y = 0; y < kVisionPatch; ++y) {
                                for (int32_t x = 0; x < kVisionPatch; ++x) {
                                    const int32_t source_y = patch_y * kVisionPatch + y;
                                    const int32_t source_x = patch_x * kVisionPatch + x;
                                    const std::size_t source =
                                        (static_cast<std::size_t>(source_y) * image.width +
                                         source_x) *
                                            3 +
                                        channel;
                                    result.pixel_values[column++] =
                                        (image.pixels[source] - 0.5F) / 0.5F;
                                }
                            }
                        }
                    }

                    // Transformers builds this arithmetic from integer Torch
                    // tensors; true division therefore promotes to FP32.
                    const float source_y = grid_height == 1
                                               ? 0.0F
                                               : static_cast<float>(patch_y) *
                                                     static_cast<float>(kVisionTableSize - 1) /
                                                     static_cast<float>(grid_height - 1);
                    const float source_x = grid_width == 1
                                               ? 0.0F
                                               : static_cast<float>(patch_x) *
                                                     static_cast<float>(kVisionTableSize - 1) /
                                                     static_cast<float>(grid_width - 1);
                    const int32_t y0 = static_cast<int32_t>(std::floor(source_y));
                    const int32_t x0 = static_cast<int32_t>(std::floor(source_x));
                    const int32_t y1 = std::min(y0 + 1, kVisionTableSize - 1);
                    const int32_t x1 = std::min(x0 + 1, kVisionTableSize - 1);
                    const float wy = source_y - static_cast<float>(y0);
                    const float wx = source_x - static_cast<float>(x0);
                    const std::array<int32_t, 4> indices = {
                        y0 * kVisionTableSize + x0, y0 * kVisionTableSize + x1,
                        y1 * kVisionTableSize + x0, y1 * kVisionTableSize + x1};
                    const std::array<float, 4> weights = {
                        (1.0F - wy) * (1.0F - wx), (1.0F - wy) * wx, wy * (1.0F - wx), wy * wx};
                    std::copy(indices.begin(), indices.end(),
                              result.interp_indices.begin() +
                                  static_cast<std::ptrdiff_t>(patch_row * 4));
                    std::copy(weights.begin(), weights.end(),
                              result.interp_weights.begin() +
                                  static_cast<std::ptrdiff_t>(patch_row * 4));
                    ++patch_row;
                }
            }
        }
    }
    if (patch_row != result.patch_rows)
        throw std::logic_error("MiniMax-H3 FL2VA vision patch accounting failed");
    return result;
}

Fl2vaVisionFeatures run_fl2va_vision_encoder(ITrtModule& module, const Fl2vaVisionInputs& inputs) {
    if (inputs.patch_rows <= 0 || inputs.patch_rows % 4 != 0 ||
        inputs.pixel_values.size() !=
            static_cast<std::size_t>(inputs.patch_rows) * kVisionPatchWidth ||
        inputs.interp_indices.size() != static_cast<std::size_t>(inputs.patch_rows) * 4 ||
        inputs.interp_weights.size() != static_cast<std::size_t>(inputs.patch_rows) * 4 ||
        inputs.vision_position_ids.size() != static_cast<std::size_t>(inputs.patch_rows) * 2) {
        throw std::invalid_argument("MiniMax-H3 FL2VA vision inputs are inconsistent");
    }
    validate_fl2va_plan(module, Fl2vaPlanKind::kVisionEncoder);
    require_binding(module, "pixel_values", true, DType::kFloat32);
    require_binding(module, "interp_indices", true, DType::kInt32);
    require_binding(module, "interp_weights", true, DType::kFloat32);
    require_binding(module, "vision_position_ids", true, DType::kInt32);
    for (const char* name : {"vision_embeds", "deepstack_0", "deepstack_1", "deepstack_2"})
        require_binding(module, name, false, DType::kFloat32);

    TensorMap plan_inputs;
    plan_inputs.emplace("pixel_values", Tensor{const_cast<float*>(inputs.pixel_values.data()),
                                               {inputs.patch_rows, kVisionPatchWidth},
                                               DType::kFloat32});
    plan_inputs.emplace("interp_indices", Tensor{const_cast<int32_t*>(inputs.interp_indices.data()),
                                                 {inputs.patch_rows, 4},
                                                 DType::kInt32});
    plan_inputs.emplace("interp_weights", Tensor{const_cast<float*>(inputs.interp_weights.data()),
                                                 {inputs.patch_rows, 4},
                                                 DType::kFloat32});
    plan_inputs.emplace("vision_position_ids",
                        Tensor{const_cast<int32_t*>(inputs.vision_position_ids.data()),
                               {inputs.patch_rows, 2},
                               DType::kInt32});
    const auto outputs = module.forward(plan_inputs);
    Fl2vaVisionFeatures result;
    result.rows = inputs.patch_rows / 4;
    const std::size_t count = static_cast<std::size_t>(result.rows) * kTextDim;
    result.vision_embeds = copy_float_output(outputs, "vision_embeds", count);
    result.deepstack_0 = copy_float_output(outputs, "deepstack_0", count);
    result.deepstack_1 = copy_float_output(outputs, "deepstack_1", count);
    result.deepstack_2 = copy_float_output(outputs, "deepstack_2", count);
    return result;
}

std::vector<float> run_fl2va_text_encoder(ITrtModule& module,
                                          const Fl2vaTextPresentation& presentation,
                                          const Fl2vaVisionFeatures& features) {
    const int32_t text_rows = static_cast<int32_t>(presentation.input_ids.size());
    const int32_t vision_rows = static_cast<int32_t>(presentation.vision_row_indices.size());
    if (text_rows <= 0 || text_rows > 2641 || features.rows != vision_rows ||
        presentation.mrope_position_ids.size() != static_cast<std::size_t>(3) * text_rows ||
        presentation.vision_mask.size() != static_cast<std::size_t>(text_rows) ||
        presentation.token_tags.size() != static_cast<std::size_t>(text_rows)) {
        throw std::invalid_argument("MiniMax-H3 FL2VA text presentation is inconsistent");
    }
    const std::size_t feature_count = static_cast<std::size_t>(vision_rows) * kTextDim;
    if (features.vision_embeds.size() != feature_count ||
        features.deepstack_0.size() != feature_count ||
        features.deepstack_1.size() != feature_count ||
        features.deepstack_2.size() != feature_count) {
        throw std::invalid_argument("MiniMax-H3 FL2VA compact vision features are inconsistent");
    }
    validate_fl2va_plan(module, Fl2vaPlanKind::kTextEncoder);
    for (const char* name :
         {"input_ids", "mrope_position_ids", "vision_mask", "vision_count", "vision_row_indices",
          "vision_embeds", "deepstack_0", "deepstack_1", "deepstack_2"}) {
        const DType dtype = (std::string(name) == "vision_mask" ||
                             std::string(name).find("embeds") != std::string::npos ||
                             std::string(name).find("deepstack") != std::string::npos)
                                ? DType::kFloat32
                                : DType::kInt32;
        require_binding(module, name, true, dtype);
    }
    require_binding(module, "encoder_hidden_states", false, DType::kFloat32);

    int32_t vision_count = vision_rows;
    TensorMap inputs;
    inputs.emplace(
        "input_ids",
        Tensor{const_cast<int32_t*>(presentation.input_ids.data()), {text_rows}, DType::kInt32});
    inputs.emplace("mrope_position_ids",
                   Tensor{const_cast<int32_t*>(presentation.mrope_position_ids.data()),
                          {3, text_rows},
                          DType::kInt32});
    inputs.emplace("vision_mask", Tensor{const_cast<float*>(presentation.vision_mask.data()),
                                         {text_rows, 1},
                                         DType::kFloat32});
    inputs.emplace("vision_count", Tensor{&vision_count, {1}, DType::kInt32});
    inputs.emplace("vision_row_indices",
                   Tensor{const_cast<int32_t*>(presentation.vision_row_indices.data()),
                          {vision_rows},
                          DType::kInt32});
    inputs.emplace("vision_embeds", Tensor{const_cast<float*>(features.vision_embeds.data()),
                                           {vision_rows, kTextDim},
                                           DType::kFloat32});
    inputs.emplace("deepstack_0", Tensor{const_cast<float*>(features.deepstack_0.data()),
                                         {vision_rows, kTextDim},
                                         DType::kFloat32});
    inputs.emplace("deepstack_1", Tensor{const_cast<float*>(features.deepstack_1.data()),
                                         {vision_rows, kTextDim},
                                         DType::kFloat32});
    inputs.emplace("deepstack_2", Tensor{const_cast<float*>(features.deepstack_2.data()),
                                         {vision_rows, kTextDim},
                                         DType::kFloat32});
    const auto outputs = module.forward(inputs);
    return copy_float_output(outputs, "encoder_hidden_states",
                             static_cast<std::size_t>(text_rows) * kTextDim);
}

std::vector<float> stitch_fl2va_posterior_tiles(const std::vector<float>& tiles, int32_t height,
                                                int32_t width) {
    const MiniMaxH3VaeTileLayout layout = make_minimax_h3_vae_tile_layout(height, width);
    const int32_t tile_rows = static_cast<int32_t>(layout.y_starts.size());
    const int32_t tile_columns = static_cast<int32_t>(layout.x_starts.size());
    const int32_t tile_count = tile_rows * tile_columns;
    const std::size_t expected = checked_product(
        {static_cast<std::size_t>(tile_count), static_cast<std::size_t>(kPosteriorChannels),
         static_cast<std::size_t>(kLatentTileSize), static_cast<std::size_t>(kLatentTileSize)},
        "FL2VA posterior tiles");
    if (tiles.size() != expected)
        throw std::invalid_argument("MiniMax-H3 FL2VA posterior tile buffer is invalid");
    const int32_t latent_height = height / 16;
    const int32_t latent_width = width / 16;
    std::vector<float> result(static_cast<std::size_t>(kPosteriorChannels) * latent_height *
                              latent_width);
    for (int32_t tile_y = 0; tile_y < tile_rows; ++tile_y) {
        const int32_t y_start = layout.y_starts[static_cast<std::size_t>(tile_y)] / 16;
        const int32_t y_overlap =
            tile_y > 0 ? layout.y_overlaps[static_cast<std::size_t>(tile_y - 1)] / 16 : 0;
        const int32_t kept_height =
            kLatentTileSize -
            (tile_y + 1 < tile_rows ? layout.y_overlaps[static_cast<std::size_t>(tile_y)] / 16 : 0);
        for (int32_t tile_x = 0; tile_x < tile_columns; ++tile_x) {
            const int32_t x_start = layout.x_starts[static_cast<std::size_t>(tile_x)] / 16;
            const int32_t x_overlap =
                tile_x > 0 ? layout.x_overlaps[static_cast<std::size_t>(tile_x - 1)] / 16 : 0;
            const int32_t kept_width =
                kLatentTileSize - (tile_x + 1 < tile_columns
                                       ? layout.x_overlaps[static_cast<std::size_t>(tile_x)] / 16
                                       : 0);
            const int32_t tile = tile_y * tile_columns + tile_x;
            for (int32_t channel = 0; channel < kPosteriorChannels; ++channel) {
                for (int32_t y = 0; y < kept_height; ++y) {
                    for (int32_t x = 0; x < kept_width; ++x) {
                        float value = tiles[tile_index(tile, channel, y, x)];
                        if (tile_y > 0 && y < y_overlap) {
                            const int32_t above = tile - tile_columns;
                            const float weight = static_cast<float>(y) / y_overlap;
                            value = tiles[tile_index(above, channel,
                                                     kLatentTileSize - y_overlap + y, x)] *
                                        (1.0F - weight) +
                                    value * weight;
                        }
                        if (tile_x > 0 && x < x_overlap) {
                            const int32_t left = tile - 1;
                            const float weight = static_cast<float>(x) / x_overlap;
                            value = tiles[tile_index(left, channel, y,
                                                     kLatentTileSize - x_overlap + x)] *
                                        (1.0F - weight) +
                                    value * weight;
                        }
                        const std::size_t target =
                            (static_cast<std::size_t>(channel) * latent_height + y_start + y) *
                                latent_width +
                            x_start + x;
                        result[target] = value;
                    }
                }
            }
        }
    }
    return result;
}

std::vector<float>
sample_and_normalize_fl2va_posterior(const std::vector<float>& posterior_parameters,
                                     int32_t latent_height, int32_t latent_width,
                                     const std::vector<float>& standard_normal) {
    if (latent_height <= 0 || latent_width <= 0)
        throw std::invalid_argument("MiniMax-H3 FL2VA latent geometry is invalid");
    const std::size_t plane = static_cast<std::size_t>(latent_height) * latent_width;
    const std::size_t sample_count = static_cast<std::size_t>(kLatentChannels) * plane;
    if (posterior_parameters.size() != sample_count * 2 || standard_normal.size() != sample_count) {
        throw std::invalid_argument("MiniMax-H3 FL2VA posterior sampling buffers are invalid");
    }
    std::vector<float> result(sample_count);
    for (int32_t channel = 0; channel < kLatentChannels; ++channel) {
        for (std::size_t index = 0; index < plane; ++index) {
            const std::size_t sample = static_cast<std::size_t>(channel) * plane + index;
            const float mean = posterior_parameters[sample];
            const float logvar =
                std::max(-30.0F, std::min(20.0F, posterior_parameters[sample_count + sample]));
            const float value = mean + std::exp(0.5F * logvar) * standard_normal[sample];
            result[sample] = (half_round(value) - kLatentMean[channel]) / kLatentStd[channel];
        }
    }
    return result;
}

std::vector<float> patchify_fl2va_keyframe_latent(const std::vector<float>& latent,
                                                  int32_t latent_height, int32_t latent_width) {
    if (latent_height <= 0 || latent_width <= 0 || latent_height % 2 != 0 ||
        latent_width % 2 != 0 ||
        latent.size() != static_cast<std::size_t>(kLatentChannels) * latent_height * latent_width) {
        throw std::invalid_argument("MiniMax-H3 FL2VA keyframe latent is invalid");
    }
    std::vector<float> rows(static_cast<std::size_t>(latent_height / 2) * (latent_width / 2) *
                            kPatchDim);
    std::size_t target = 0;
    for (int32_t y = 0; y < latent_height; y += kPatchHeight) {
        for (int32_t x = 0; x < latent_width; x += kPatchWidth) {
            for (int32_t channel = 0; channel < kLatentChannels; ++channel) {
                for (int32_t py = 0; py < kPatchHeight; ++py) {
                    for (int32_t px = 0; px < kPatchWidth; ++px) {
                        rows[target++] =
                            latent[(static_cast<std::size_t>(channel) * latent_height + y + py) *
                                       latent_width +
                                   x + px];
                    }
                }
            }
        }
    }
    return rows;
}

std::vector<float> run_fl2va_keyframe_vae_encoder(ITrtModule& module,
                                                  const VideoImageInput& image) {
    validate_image(image);
    validate_fl2va_plan(module, Fl2vaPlanKind::kKeyframeVaeEncoder);
    require_binding(module, "pixel_tiles", true, DType::kFloat32);
    require_binding(module, "posterior_parameter_tiles", false, DType::kFloat32);
    const MiniMaxH3VaeTileLayout layout =
        make_minimax_h3_vae_tile_layout(image.height, image.width);
    const int32_t tile_rows = static_cast<int32_t>(layout.y_starts.size());
    const int32_t tile_columns = static_cast<int32_t>(layout.x_starts.size());
    const int32_t tile_count = tile_rows * tile_columns;
    std::vector<float> pixel_tiles(
        checked_product({static_cast<std::size_t>(tile_count), 3U, 1U,
                         static_cast<std::size_t>(kTileSize), static_cast<std::size_t>(kTileSize)},
                        "FL2VA VAE pixel tiles"));
    for (int32_t tile_y = 0; tile_y < tile_rows; ++tile_y) {
        for (int32_t tile_x = 0; tile_x < tile_columns; ++tile_x) {
            const int32_t tile = tile_y * tile_columns + tile_x;
            const int32_t source_y = layout.y_starts[static_cast<std::size_t>(tile_y)];
            const int32_t source_x = layout.x_starts[static_cast<std::size_t>(tile_x)];
            for (int32_t channel = 0; channel < 3; ++channel) {
                for (int32_t y = 0; y < kTileSize; ++y) {
                    for (int32_t x = 0; x < kTileSize; ++x) {
                        const std::size_t source =
                            (static_cast<std::size_t>(source_y + y) * image.width + source_x + x) *
                                3 +
                            channel;
                        const std::size_t target =
                            (((static_cast<std::size_t>(tile) * 3 + channel) * kTileSize + y) *
                                 kTileSize +
                             x);
                        pixel_tiles[target] =
                            (image.pixels[source] - kPixelMean[channel]) / kPixelStd[channel];
                    }
                }
            }
        }
    }
    TensorMap inputs;
    inputs.emplace(
        "pixel_tiles",
        Tensor{pixel_tiles.data(), {tile_count, 3, 1, kTileSize, kTileSize}, DType::kFloat32});
    const auto outputs = module.forward(inputs);
    auto posterior_tiles =
        copy_float_output(outputs, "posterior_parameter_tiles",
                          static_cast<std::size_t>(tile_count) * kPosteriorChannels *
                              kLatentTileSize * kLatentTileSize);
    auto posterior = stitch_fl2va_posterior_tiles(posterior_tiles, image.height, image.width);
    const int32_t latent_height = image.height / 16;
    const int32_t latent_width = image.width / 16;
    auto epsilon = torch_cuda_normal(
        static_cast<std::size_t>(kLatentChannels) * latent_height * latent_width, kKeyframeSeed);
    return sample_and_normalize_fl2va_posterior(posterior, latent_height, latent_width, epsilon);
}

} // namespace trtmc::minimax_h3
