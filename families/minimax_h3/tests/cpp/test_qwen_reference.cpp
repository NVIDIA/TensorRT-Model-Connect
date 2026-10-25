/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/qwen_reference.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace trtmc::minimax_h3;

void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}

class FixtureTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string& value) const override {
        return std::vector<int32_t>(value.begin(), value.end());
    }
    std::string decode(const std::vector<int32_t>&) const override { return {}; }
    int32_t id_for_token(std::string_view value) const override {
        if (value == "<|vision_start|>")
            return 151652;
        if (value == "<|vision_end|>")
            return 151653;
        if (value == "<|image_pad|>")
            return 151655;
        return -1;
    }
    std::string token_for_id(int32_t) const override { return {}; }
};

std::vector<uint8_t> rgb_fixture() {
    std::vector<uint8_t> rgb(32 * 64 * 3);
    for (std::size_t index = 0; index < rgb.size(); ++index)
        rgb[index] = static_cast<uint8_t>(index % 256);
    return rgb;
}

void test_metadata_and_pixels() {
    const auto metadata = make_qwen_vision_metadata({2, 4});
    require(metadata.position_indices.size() == 8 * 4, "interpolation index shape");
    require(metadata.rotary_cos.size() == 8 * 72, "full rotary cache shape");
    for (std::size_t row = 0; row < 8; ++row) {
        float total = 0;
        for (std::size_t tap = 0; tap < 4; ++tap) {
            const auto index = metadata.position_indices[row * 4 + tap];
            require(index >= 0 && index < 2304, "interpolation index bounds");
            total += metadata.position_weights[row * 4 + tap];
        }
        require(std::abs(total - 1.0F) < 1.0e-6F, "bilinear partition of unity");
        for (std::size_t column = 0; column < 36; ++column)
            require(metadata.rotary_cos[row * 72 + column] ==
                        metadata.rotary_cos[row * 72 + 36 + column],
                    "rotate-half frequency duplication");
    }
    require(std::abs(metadata.rotary_cos[2 * 72] - std::cos(1.0F)) < 1.0e-7F,
            "merge-block row two has height one");
    require(std::abs(metadata.rotary_cos[4 * 72 + 18] - std::cos(2.0F)) < 1.0e-7F,
            "merge-block row four has width two");
    const auto rgb = rgb_fixture();
    const auto packed = pack_resized_qwen_image(rgb.data(), rgb.size(), 32, 64);
    require(packed.size() == 8 * 1536, "patch matrix shape");
    for (std::size_t row = 0; row < 8; ++row) {
        for (std::size_t channel = 0; channel < 3; ++channel) {
            const auto base = row * 1536 + channel * 512;
            for (std::size_t pixel = 0; pixel < 256; ++pixel) {
                require(packed[base + pixel] == packed[base + 256 + pixel],
                        "single image duplicates each temporal patch");
                require(packed[base + pixel] >= -1.0F && packed[base + pixel] <= 1.0F,
                        "Qwen normalized range");
            }
        }
    }
    const float expected = (static_cast<float>(rgb[(16 * 64) * 3]) - 127.5F) / 127.5F;
    require(packed[2 * 1536] == expected, "patch row order is block-major, not raster-major");
    for (const QwenImageGrid bad : {QwenImageGrid{1, 4}, {2, -4}, {512, 512}}) {
        bool rejected = false;
        try {
            (void)make_qwen_vision_metadata(bad);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "invalid vision geometry rejected");
    }
}

QwenImagePresentation presentation_fixture() {
    return make_qwen_image_presentation(FixtureTokenizer{}, "walk", {{4, 4}, {2, 6}}, 128);
}

void test_presentation() {
    const auto value = presentation_fixture();
    const auto rows = value.input_ids.size();
    require(rows == 41 && value.position_ids.size() == 3 * rows,
            "original reference presentation row count");
    require(value.rope_delta == -2, "visual positions compress by grid extent, not pad count");
    require(value.visual_rows.size() == 7, "all and only merged visual features have row indexes");
    for (std::size_t index = 0; index < rows; ++index) {
        const auto token = value.input_ids[index];
        const bool visual = token == 151655;
        require(value.visual_mask[index] == static_cast<int32_t>(visual),
                "native int32 mask marks pads only");
        if (visual || token == 151652 || token == 151653)
            require(value.h3_token_tags[index] == 0, "H3 tags whole vision block as video");
        else
            require(value.h3_token_tags[index] == 1, "H3 tags labels and prompt as text");
    }
    const auto first = static_cast<std::size_t>(value.visual_rows[0]);
    require(value.position_ids[first] == 14 && value.position_ids[rows + first] == 14 &&
                value.position_ids[2 * rows + first] == 14,
            "first visual position starts after label and vision_start");
    require(value.position_ids[first + 3] == 14 && value.position_ids[rows + first + 3] == 15 &&
                value.position_ids[2 * rows + first + 3] == 15,
            "merged visual T/H/W coordinates");
    bool rejected = false;
    try {
        (void)make_qwen_image_presentation(FixtureTokenizer{}, "walk", {{4, 4}, {2, 6}}, 40);
    } catch (const std::length_error&) {
        rejected = true;
    }
    require(rejected, "profile overflow is rejected, never truncated or downsampled");
}

template <class T>
void dump(const std::vector<T>& values) {
    std::cout.write(reinterpret_cast<const char*>(values.data()),
                    static_cast<std::streamsize>(values.size() * sizeof(T)));
}

void dump_fixture(const std::string& field) {
    const auto metadata = make_qwen_vision_metadata({2, 4});
    const auto presentation = presentation_fixture();
    if (field == "indices")
        dump(metadata.position_indices);
    else if (field == "weights")
        dump(metadata.position_weights);
    else if (field == "cos")
        dump(metadata.rotary_cos);
    else if (field == "sin")
        dump(metadata.rotary_sin);
    else if (field == "ids")
        dump(presentation.input_ids);
    else if (field == "tags")
        dump(presentation.h3_token_tags);
    else if (field == "mask")
        dump(presentation.visual_mask);
    else if (field == "positions")
        dump(presentation.position_ids);
    else if (field == "visual_rows")
        dump(presentation.visual_rows);
    else if (field == "pixels") {
        const auto rgb = rgb_fixture();
        dump(pack_resized_qwen_image(rgb.data(), rgb.size(), 32, 64));
    } else
        throw std::invalid_argument("unknown Qwen reference fixture field");
}

} // namespace

int main(int argc, char** argv) {
    try {
        test_metadata_and_pixels();
        test_presentation();
        if (argc == 3 && std::string(argv[1]) == "--dump-fixture")
            dump_fixture(argv[2]);
        else if (argc != 1)
            throw std::invalid_argument("expected no args or --dump-fixture FIELD");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
