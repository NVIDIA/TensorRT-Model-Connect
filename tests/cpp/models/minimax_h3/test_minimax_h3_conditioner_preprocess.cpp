/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/conditioner_preprocess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

template <typename Callable>
void check_throws(Callable&& callable, const char* label) {
    try {
        callable();
        check(false, label);
    } catch (const std::invalid_argument&) {
    }
}

trtmc::MiniMaxH3ConditionerTokenizer make_tokenizer(std::vector<std::string>& calls) {
    return [&calls](const std::string& text) {
        calls.push_back(text);
        if (text == "<Picture 1>: ")
            return std::vector<int32_t>{101, 102};
        if (text == "<Picture 2>: ")
            return std::vector<int32_t>{201};
        if (text == "verbatim prompt")
            return std::vector<int32_t>{301, 302, 303};
        throw std::invalid_argument("unexpected tokenizer input");
    };
}

int32_t mrope_at(const trtmc::MiniMaxH3ConditionerPresentation& presentation, int32_t axis,
                 int32_t row) {
    return presentation
        .mrope_position_ids[static_cast<std::size_t>(axis) * presentation.sequence_rows + row];
}

bool range_is(const std::vector<int32_t>& values, int32_t begin, int32_t end, int32_t expected) {
    for (int32_t index = begin; index < end; ++index) {
        if (values[static_cast<std::size_t>(index)] != expected)
            return false;
    }
    return true;
}

void test_text_only_presentation_is_verbatim_and_one_dimensional() {
    std::vector<std::string> calls;
    const auto presentation = trtmc::minimax_h3_make_conditioner_presentation(
        "verbatim prompt", false, false, make_tokenizer(calls), {900, 901, 902});
    check(calls == std::vector<std::string>({"verbatim prompt"}),
          "H3 T2VA tokenizes the prompt verbatim once");
    check(presentation.input_ids == std::vector<int32_t>({301, 302, 303}),
          "H3 T2VA presentation is the verbatim prompt tokens");
    check(presentation.h3_token_tags == std::vector<int32_t>({1, 1, 1}),
          "H3 T2VA H3 tags are text");
    check(presentation.qwen_mm_token_type_ids == std::vector<int32_t>({0, 0, 0}),
          "H3 T2VA Qwen types are text");
    check(presentation.vision_scatter_indices.empty(), "H3 T2VA has no vision scatter rows");
    check(presentation.vision_selector == std::vector<int32_t>({0, 0, 0}),
          "H3 T2VA selector is inactive");
    check(presentation.mrope_position_ids == std::vector<int32_t>({0, 1, 2, 0, 1, 2, 0, 1, 2}),
          "H3 T2VA MRoPE is ordinary one-dimensional text position");
    check(presentation.next_mrope_position == 3, "H3 T2VA advances the MRoPE cursor");
}

void check_single_keyframe_presentation(const trtmc::MiniMaxH3ConditionerPresentation& presentation,
                                        const char* label) {
    check(presentation.sequence_rows == 1015, label);
    check(presentation.num_keyframes == 1, "H3 single-keyframe count is one");
    check(presentation.input_ids[0] == 101, "H3 Picture 1 label begins first");
    check(presentation.input_ids[1] == 102, "H3 Picture 1 label keeps tokenizer order");
    check(presentation.input_ids[2] == 900, "H3 vision-start follows Picture 1 label");
    check(presentation.input_ids[3] == 901, "H3 image pads follow vision-start");
    check(presentation.input_ids[1010] == 901, "H3 emits exactly 1008 image pads");
    check(presentation.input_ids[1011] == 902, "H3 vision-end follows image pads");
    check(presentation.input_ids[1012] == 301, "H3 prompt follows vision-end");
    check(presentation.input_ids[1014] == 303, "H3 prompt remains verbatim at the tail");
    check(range_is(presentation.h3_token_tags, 0, 2, 1), "H3 Picture 1 label tags are text");
    check(range_is(presentation.h3_token_tags, 2, 1012, 0),
          "H3 complete vision block tags are video");
    check(range_is(presentation.h3_token_tags, 1012, 1015, 1), "H3 prompt tags are text");
    check(range_is(presentation.qwen_mm_token_type_ids, 0, 3, 0),
          "Qwen treats the label and vision-start as text");
    check(range_is(presentation.qwen_mm_token_type_ids, 3, 1011, 1),
          "Qwen treats only image pads as image");
    check(range_is(presentation.qwen_mm_token_type_ids, 1011, 1015, 0),
          "Qwen treats vision-end and prompt as text");
    check(presentation.vision_scatter_indices.size() == 1008,
          "H3 single-keyframe scatter has 1008 rows");
    check(presentation.vision_scatter_indices.front() == 3,
          "H3 scatter begins at the first image pad");
    check(presentation.vision_scatter_indices.back() == 1010,
          "H3 scatter ends at the last image pad");
    check(std::accumulate(presentation.vision_selector.begin(), presentation.vision_selector.end(),
                          int64_t{0}) == 1008,
          "H3 selector activates every compact vision row");
    check(mrope_at(presentation, 0, 3) == 3, "H3 image temporal MRoPE starts at 3");
    check(mrope_at(presentation, 1, 3) == 3, "H3 image height MRoPE starts at 3");
    check(mrope_at(presentation, 2, 3) == 3, "H3 image width MRoPE starts at 3");
    check(mrope_at(presentation, 0, 1010) == 3, "H3 image temporal MRoPE stays fixed");
    check(mrope_at(presentation, 1, 1010) == 26, "H3 image height MRoPE spans 24 rows");
    check(mrope_at(presentation, 2, 1010) == 44, "H3 image width MRoPE spans 42 columns");
    check(mrope_at(presentation, 0, 1011) == 45, "H3 text resumes after the image span");
    check(mrope_at(presentation, 0, 1014) == 48, "H3 prompt advances one dimensionally");
    check(presentation.next_mrope_position == 49, "H3 single-keyframe MRoPE cursor is exact");
}

void test_first_only_uses_picture_one() {
    std::vector<std::string> calls;
    const auto presentation = trtmc::minimax_h3_make_conditioner_presentation(
        "verbatim prompt", true, false, make_tokenizer(calls), {900, 901, 902});
    check(calls == std::vector<std::string>({"<Picture 1>: ", "verbatim prompt"}),
          "H3 first-only presentation labels its packed keyframe Picture 1");
    check_single_keyframe_presentation(presentation, "H3 first-only presentation has one grid");
}

void test_last_only_still_uses_picture_one() {
    std::vector<std::string> calls;
    const auto presentation = trtmc::minimax_h3_make_conditioner_presentation(
        "verbatim prompt", false, true, make_tokenizer(calls), {900, 901, 902});
    check(calls == std::vector<std::string>({"<Picture 1>: ", "verbatim prompt"}),
          "H3 last-only presentation labels its sole packed keyframe Picture 1");
    check_single_keyframe_presentation(presentation, "H3 last-only presentation has one grid");
}

void test_both_keyframes_are_sequential_picture_grids() {
    std::vector<std::string> calls;
    const auto presentation = trtmc::minimax_h3_make_conditioner_presentation(
        "verbatim prompt", true, true, make_tokenizer(calls), {900, 901, 902});
    check(calls == std::vector<std::string>({"<Picture 1>: ", "<Picture 2>: ", "verbatim prompt"}),
          "H3 both-keyframe presentation tokenizes labels in packed order");
    check(presentation.sequence_rows == 2026,
          "H3 both-keyframe presentation contains both vision blocks");
    check(presentation.num_keyframes == 2, "H3 both-keyframe count is two");
    check(presentation.input_ids[1011] == 902, "H3 Picture 1 closes before Picture 2");
    check(presentation.input_ids[1012] == 201, "H3 Picture 2 label follows Picture 1");
    check(presentation.input_ids[1013] == 900, "H3 Picture 2 vision-start follows its label");
    check(presentation.input_ids[1014] == 901, "H3 Picture 2 pads follow its vision-start");
    check(presentation.input_ids[2021] == 901, "H3 Picture 2 has 1008 image pads");
    check(presentation.input_ids[2022] == 902, "H3 Picture 2 closes before the prompt");
    check(presentation.input_ids[2023] == 301, "H3 verbatim prompt follows Picture 2");
    check(presentation.vision_scatter_indices.size() == 2016,
          "H3 both-keyframe scatter has 2016 rows");
    check(presentation.vision_scatter_indices[1007] == 1010,
          "H3 compact Picture 1 rows end before Picture 2");
    check(presentation.vision_scatter_indices[1008] == 1014,
          "H3 compact Picture 2 rows begin at its first pad");
    check(presentation.vision_scatter_indices.back() == 2021,
          "H3 compact Picture 2 rows end at its last pad");
    check(range_is(presentation.h3_token_tags, 1012, 1013, 1), "H3 Picture 2 label is text");
    check(range_is(presentation.h3_token_tags, 1013, 2023, 0),
          "H3 Picture 2 vision boundaries and pads are video");
    check(range_is(presentation.qwen_mm_token_type_ids, 1014, 2022, 1),
          "Qwen Picture 2 pads alone are image");
    check(mrope_at(presentation, 0, 1014) == 48, "H3 Picture 2 temporal MRoPE starts at 48");
    check(mrope_at(presentation, 1, 1014) == 48, "H3 Picture 2 height MRoPE starts at 48");
    check(mrope_at(presentation, 2, 1014) == 48, "H3 Picture 2 width MRoPE starts at 48");
    check(mrope_at(presentation, 0, 2021) == 48, "H3 Picture 2 temporal MRoPE stays fixed");
    check(mrope_at(presentation, 1, 2021) == 71, "H3 Picture 2 height MRoPE spans 24 rows");
    check(mrope_at(presentation, 2, 2021) == 89, "H3 Picture 2 width MRoPE spans 42 columns");
    check(mrope_at(presentation, 0, 2022) == 90, "H3 text resumes after Picture 2");
    check(presentation.next_mrope_position == 94, "H3 both-keyframe MRoPE cursor is exact");
}

std::size_t image_offset(int32_t y, int32_t x, int32_t channel) {
    return (static_cast<std::size_t>(y) * trtmc::kMiniMaxH3ConditionerImageWidth + x) * 3U +
           channel;
}

std::size_t patch_offset(int32_t row, int32_t channel, int32_t temporal, int32_t y, int32_t x) {
    return static_cast<std::size_t>(row) * trtmc::kMiniMaxH3ConditionerPatchVector +
           (((static_cast<std::size_t>(channel) * trtmc::kMiniMaxH3ConditionerTemporalPatchSize +
              temporal) *
                 trtmc::kMiniMaxH3ConditionerPatchSize +
             y) *
                trtmc::kMiniMaxH3ConditionerPatchSize +
            x);
}

trtmc::MediaImageInput make_patch_order_keyframe() {
    trtmc::MediaImageInput image;
    image.height = trtmc::kMiniMaxH3ConditionerImageHeight;
    image.width = trtmc::kMiniMaxH3ConditionerImageWidth;
    image.pixels.assign(static_cast<std::size_t>(image.height) * image.width * 3U, 0.5F);
    image.pixels[image_offset(0, 0, 0)] = 1.0F;
    image.pixels[image_offset(0, 16, 1)] = 0.0F;
    image.pixels[image_offset(18, 3, 2)] = 0.75F;
    image.pixels[image_offset(0, 32, 0)] = 0.25F;
    image.pixels[image_offset(767, 1343, 2)] = 1.0F;
    return image;
}

void test_keyframe_patch_tensor_matches_official_merge_order() {
    auto image = make_patch_order_keyframe();
    const auto pixels = trtmc::minimax_h3_preprocess_conditioner_keyframe(image);
    check(pixels.size() == static_cast<std::size_t>(trtmc::kMiniMaxH3ConditionerPatchRows) *
                               trtmc::kMiniMaxH3ConditionerPatchVector,
          "H3 conditioner emits [4032,1536] processor pixels");
    check(pixels[patch_offset(0, 0, 0, 0, 0)] == 1.0F,
          "H3 conditioner normalizes the first temporal slot");
    check(pixels[patch_offset(0, 0, 1, 0, 0)] == 1.0F,
          "H3 conditioner duplicates the still image temporally");
    check(pixels[patch_offset(1, 1, 0, 0, 0)] == -1.0F,
          "H3 conditioner orders merge-column before channels");
    check(pixels[patch_offset(1, 1, 1, 0, 0)] == -1.0F,
          "H3 conditioner duplicates every channel temporally");
    check(pixels[patch_offset(2, 2, 0, 2, 3)] == 0.5F,
          "H3 conditioner orders merge-row before patch pixels");
    check(pixels[patch_offset(4, 0, 0, 0, 0)] == -0.5F,
          "H3 conditioner advances to the next merge group");
    check(pixels[patch_offset(3, 0, 0, 0, 0)] == 0.0F,
          "H3 conditioner keeps the fourth first-group patch distinct");
    check(pixels[patch_offset(4031, 2, 1, 15, 15)] == 1.0F,
          "H3 conditioner reaches the last channel and temporal patch element");

    auto invalid = image;
    invalid.pixels[0] = std::numeric_limits<float>::quiet_NaN();
    check_throws([&] { (void)trtmc::minimax_h3_preprocess_conditioner_keyframe(invalid); },
                 "H3 conditioner rejects non-finite keyframe pixels");
    invalid = {};
    invalid.height = 1;
    invalid.width = 1;
    invalid.pixels = {0.0F, 0.0F, 0.0F};
    check_throws([&] { (void)trtmc::minimax_h3_preprocess_conditioner_keyframe(invalid); },
                 "H3 conditioner rejects keyframes outside the fixed processor canvas");
}

std::vector<float> make_compact_features(int32_t rows, int32_t feature_dim, float base) {
    std::vector<float> result(static_cast<std::size_t>(rows) * feature_dim);
    for (int32_t row = 0; row < rows; ++row) {
        for (int32_t column = 0; column < feature_dim; ++column) {
            result[static_cast<std::size_t>(row) * feature_dim + column] =
                base + static_cast<float>(row * feature_dim + column);
        }
    }
    return result;
}

float sequence_feature(const std::vector<float>& values, int32_t feature_dim, int32_t row,
                       int32_t column) {
    return values[static_cast<std::size_t>(row) * feature_dim + column];
}

void test_first_keyframe_scatter_materializes_all_four_outputs() {
    std::vector<std::string> calls;
    const auto presentation = trtmc::minimax_h3_make_conditioner_presentation(
        "verbatim prompt", true, false, make_tokenizer(calls), {900, 901, 902});
    constexpr int32_t feature_dim = 2;
    const auto main = make_compact_features(1008, feature_dim, 100.0F);
    const std::array<std::vector<float>, 3> deepstack = {
        make_compact_features(1008, feature_dim, 1000.0F),
        make_compact_features(1008, feature_dim, 2000.0F),
        make_compact_features(1008, feature_dim, 3000.0F),
    };
    const auto scattered =
        trtmc::minimax_h3_scatter_vision_features(presentation, main, deepstack, feature_dim);

    check(scattered.sequence_rows == 1015, "H3 scattered first-keyframe rows match presentation");
    check(scattered.feature_dim == feature_dim, "H3 scattered first-keyframe feature dim is kept");
    check(scattered.vision_selector == presentation.vision_selector,
          "H3 scattered first-keyframe selector binds as [L,1]");
    check(scattered.vision_embeddings.size() == 2030, "H3 main vision output materializes [L,D]");
    check(sequence_feature(scattered.vision_embeddings, feature_dim, 0, 0) == 0.0F,
          "H3 main scatter zero-fills inactive label rows");
    check(sequence_feature(scattered.vision_embeddings, feature_dim, 2, 1) == 0.0F,
          "H3 main scatter zero-fills inactive vision-start rows");
    check(sequence_feature(scattered.vision_embeddings, feature_dim, 3, 0) == 100.0F,
          "H3 main scatter places compact row zero at the first pad");
    check(sequence_feature(scattered.vision_embeddings, feature_dim, 1010, 1) == 2115.0F,
          "H3 main scatter places compact row 1007 at the last pad");
    check(sequence_feature(scattered.vision_embeddings, feature_dim, 1011, 0) == 0.0F,
          "H3 main scatter zero-fills inactive vision-end rows");
    check(sequence_feature(scattered.deepstack_embeddings[0], feature_dim, 3, 0) == 1000.0F,
          "H3 DeepStack 0 uses the main scatter rows");
    check(sequence_feature(scattered.deepstack_embeddings[1], feature_dim, 3, 1) == 2001.0F,
          "H3 DeepStack 1 uses the main scatter rows");
    check(sequence_feature(scattered.deepstack_embeddings[2], feature_dim, 1010, 0) == 5014.0F,
          "H3 DeepStack 2 preserves the last compact row");
}

void test_both_keyframe_scatter_preserves_keyframe_major_order() {
    std::vector<std::string> calls;
    const auto presentation = trtmc::minimax_h3_make_conditioner_presentation(
        "verbatim prompt", true, true, make_tokenizer(calls), {900, 901, 902});
    const auto main = make_compact_features(2016, 1, 1.0F);
    const std::array<std::vector<float>, 3> deepstack = {
        make_compact_features(2016, 1, 10001.0F),
        make_compact_features(2016, 1, 20001.0F),
        make_compact_features(2016, 1, 30001.0F),
    };
    const auto scattered =
        trtmc::minimax_h3_scatter_vision_features(presentation, main, deepstack, 1);

    check(sequence_feature(scattered.vision_embeddings, 1, 3, 0) == 1.0F,
          "H3 Picture 1 scatter begins with compact row zero");
    check(sequence_feature(scattered.vision_embeddings, 1, 1010, 0) == 1008.0F,
          "H3 Picture 1 scatter ends before the separator");
    check(sequence_feature(scattered.vision_embeddings, 1, 1011, 0) == 0.0F,
          "H3 both-keyframe scatter zero-fills the inter-picture separator");
    check(sequence_feature(scattered.vision_embeddings, 1, 1014, 0) == 1009.0F,
          "H3 Picture 2 scatter continues compact keyframe-major order");
    check(sequence_feature(scattered.vision_embeddings, 1, 2021, 0) == 2016.0F,
          "H3 Picture 2 scatter ends with the final compact row");
    check(sequence_feature(scattered.deepstack_embeddings[2], 1, 1014, 0) == 31009.0F,
          "H3 DeepStack scatter keeps Picture 2 compact ordering");
}

void test_vision_scatter_rejects_shape_finite_and_order_errors() {
    std::vector<std::string> calls;
    const auto presentation = trtmc::minimax_h3_make_conditioner_presentation(
        "verbatim prompt", true, false, make_tokenizer(calls), {900, 901, 902});
    auto main = make_compact_features(1008, 1, 1.0F);
    std::array<std::vector<float>, 3> deepstack = {
        make_compact_features(1008, 1, 1001.0F),
        make_compact_features(1008, 1, 2001.0F),
        make_compact_features(1008, 1, 3001.0F),
    };

    main.pop_back();
    check_throws(
        [&] { (void)trtmc::minimax_h3_scatter_vision_features(presentation, main, deepstack, 1); },
        "H3 vision scatter rejects the wrong compact main shape");
    main = make_compact_features(1008, 1, 1.0F);
    deepstack[1][17] = std::numeric_limits<float>::quiet_NaN();
    check_throws(
        [&] { (void)trtmc::minimax_h3_scatter_vision_features(presentation, main, deepstack, 1); },
        "H3 vision scatter rejects non-finite DeepStack values");
    deepstack[1][17] = 2018.0F;
    auto wrong_order = presentation;
    std::swap(wrong_order.vision_scatter_indices[0], wrong_order.vision_scatter_indices[1]);
    check_throws(
        [&] { (void)trtmc::minimax_h3_scatter_vision_features(wrong_order, main, deepstack, 1); },
        "H3 vision scatter rejects reordered compact rows");
    auto wrong_type = presentation;
    wrong_type.qwen_mm_token_type_ids[0] = 2;
    check_throws(
        [&] { (void)trtmc::minimax_h3_scatter_vision_features(wrong_type, main, deepstack, 1); },
        "H3 vision scatter rejects non-image multimodal token types");
}

void test_presentation_validation_fails_closed() {
    check_throws(
        [&] {
            (void)trtmc::minimax_h3_make_conditioner_presentation("verbatim prompt", false, false,
                                                                  {}, {900, 901, 902});
        },
        "H3 conditioner rejects a missing tokenizer callback");
    const auto reserved = [](const std::string&) { return std::vector<int32_t>{901}; };
    check_throws(
        [&] {
            (void)trtmc::minimax_h3_make_conditioner_presentation("verbatim prompt", false, false,
                                                                  reserved, {900, 901, 902});
        },
        "H3 conditioner rejects reserved vision tokens in text output");
}

} // namespace

int main() {
    test_text_only_presentation_is_verbatim_and_one_dimensional();
    test_first_only_uses_picture_one();
    test_last_only_still_uses_picture_one();
    test_both_keyframes_are_sequential_picture_grids();
    test_keyframe_patch_tensor_matches_official_merge_order();
    test_first_keyframe_scatter_materializes_all_four_outputs();
    test_both_keyframe_scatter_preserves_keyframe_major_order();
    test_vision_scatter_rejects_shape_finite_and_order_errors();
    test_presentation_validation_fails_closed();
    return failures == 0 ? 0 : 1;
}
