/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/llava/runtime/image_preprocessor.h"

#include "families/llava/runtime/image_preprocess_seam.h"

#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc {
LlavaPreprocessedImage
llava_preprocess_decoded_image(const runtime::adapters::io::DecodedImage& image,
                               const LlavaPreprocessConfig& config) {
    LlavaPreprocessedImage result;
    if (image.empty() || image.channels != 3)
        throw std::invalid_argument("LLaVA requires a nonempty RGB image");
    std::vector<float> pixels(image.pixels.size());
    for (std::size_t i = 0; i < pixels.size(); ++i)
        pixels[i] = static_cast<float>(image.pixels[i]) / 255.0F;
    LlavaClipConfig clip;
    clip.input_image_h = clip.input_image_w = config.fixed_image_size;
    result.pixel_values = preprocess_llava_image(pixels.data(), image.height, image.width, clip);
    result.height = result.width = config.fixed_image_size;
    result.ok = true;
    return result;
}

std::string llava_format_prompt(const std::string& prompt, const LlavaPreprocessConfig& config,
                                int32_t count) {
    if (prompt.find("<image>") != std::string::npos)
        throw std::invalid_argument(
            "LLaVA accepts an unformatted user prompt without image markers");
    if (count != config.num_image_pad_tokens)
        throw std::runtime_error("LLaVA vision feature count differs from its patch grid");
    std::string image;
    for (int32_t i = 0; i < count; ++i)
        image += config.image_token_str;
    std::string text = config.vl_prompt_template;
    text.replace(text.find("{image_pads}"), 12, image);
    text.replace(text.find("{prompt}"), 8, prompt);
    return text;
}

LlavaPreprocessConfig llava_parse_preprocess_config(const std::string& text) {
    auto json = nlohmann::json::parse(text);
    LlavaPreprocessConfig config;
    config.fixed_image_size = json.at("fixed_image_size").get<int32_t>();
    config.num_image_pad_tokens = json.at("num_image_pad_tokens").get<int32_t>();
    config.image_token_id = json.at("image_token_id").get<int32_t>();
    config.vision_output_dim = json.at("vision_output_dim").get<int32_t>();
    config.vl_prompt_template = json.at("vl_prompt_template").get<std::string>();
    config.image_token_str = json.at("image_token_str").get<std::string>();
    return config;
}
} // namespace trtmc
