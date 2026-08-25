/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/ref2va_conditioner.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

int32_t mrope_at(const trtmc::MiniMaxH3Ref2VAConditionerPresentation& presentation, int32_t axis,
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

trtmc::MiniMaxH3Ref2VATokenizer make_tokenizer(std::vector<std::string>& calls) {
    return [&calls](const std::string& text) {
        calls.push_back(text);
        return std::vector<int32_t>{1000 + static_cast<int32_t>(calls.size())};
    };
}

trtmc::MultiChannelAudioResult prepared_audio() {
    trtmc::MultiChannelAudioResult audio;
    audio.samples.assign(16, 0.25F);
    audio.num_samples = 8;
    audio.sample_rate = 32000;
    audio.num_channels = 2;
    return audio;
}

trtmc::AudioVideoReference source_reference(trtmc::AudioVideoReferenceKind kind,
                                            bool soundtrack = false) {
    trtmc::AudioVideoReference reference;
    reference.kind = kind;
    if (soundtrack)
        reference.video.soundtrack = prepared_audio();
    return reference;
}

std::size_t image_offset(int32_t width, int32_t y, int32_t x, int32_t channel) {
    return (static_cast<std::size_t>(y) * width + x) * 3U + channel;
}

trtmc::MediaImageInput prepared_image() {
    trtmc::MediaImageInput image;
    image.height = trtmc::kMiniMaxH3Ref2VAImageShortEdge;
    image.width = trtmc::kMiniMaxH3Ref2VAImageShortEdge;
    image.pixels.assign(static_cast<std::size_t>(image.height) * image.width * 3U, 0.5F);
    image.pixels[image_offset(image.width, 0, 0, 0)] = 1.0F;
    image.pixels[image_offset(image.width, 0, 16, 1)] = 0.0F;
    return image;
}

trtmc::MediaVideoInput prepared_video(int32_t num_frames = 13) {
    trtmc::MediaVideoInput video;
    video.num_frames = num_frames;
    video.height = 768;
    video.width = 768;
    video.fps = 24.0F;
    const auto frame_elements = static_cast<std::size_t>(video.height) * video.width * 3U;
    video.pixels.assign(static_cast<std::size_t>(video.num_frames) * frame_elements, 0.5F);
    video.pixels[image_offset(video.width, 0, 0, 0)] = 1.0F;
    video.pixels[12U * frame_elements + image_offset(video.width, 0, 0, 1)] = 0.0F;
    return video;
}

trtmc::MiniMaxH3PreparedReference image_reference(std::size_t index) {
    trtmc::MiniMaxH3PreparedReference reference;
    reference.reference_index = index;
    reference.kind = trtmc::AudioVideoReferenceKind::kImage;
    reference.image = prepared_image();
    reference.qwen_grid_h = reference.image.height / trtmc::kMiniMaxH3Ref2VAPatchSize;
    reference.qwen_grid_w = reference.image.width / trtmc::kMiniMaxH3Ref2VAPatchSize;
    reference.qwen_patch_rows = reference.qwen_grid_h * reference.qwen_grid_w;
    return reference;
}

trtmc::MiniMaxH3PreparedReference video_reference(std::size_t index, bool soundtrack,
                                                  int32_t num_frames = 13) {
    trtmc::MiniMaxH3PreparedReference reference;
    reference.reference_index = index;
    reference.kind = trtmc::AudioVideoReferenceKind::kVideo;
    reference.video = prepared_video(num_frames);
    reference.qwen_grid_h = reference.video.height / trtmc::kMiniMaxH3Ref2VAPatchSize;
    reference.qwen_grid_w = reference.video.width / trtmc::kMiniMaxH3Ref2VAPatchSize;
    reference.qwen_patch_rows = reference.qwen_grid_h * reference.qwen_grid_w;
    if (soundtrack)
        reference.audio = prepared_audio();
    return reference;
}

trtmc::MiniMaxH3PreparedReference audio_reference(std::size_t index) {
    trtmc::MiniMaxH3PreparedReference reference;
    reference.reference_index = index;
    reference.kind = trtmc::AudioVideoReferenceKind::kAudio;
    reference.audio = prepared_audio();
    return reference;
}

void check_first_position_bindings(const trtmc::MiniMaxH3Ref2VAVisionInput& input) {
    check(input.position_indices.size() ==
              static_cast<std::size_t>(input.grid_h) * input.grid_w * 4U,
          "Ref2VA learned-position indices cover every patch");
    check(input.position_weights.size() == input.position_indices.size(),
          "Ref2VA FP32 learned-position weights align with indices");
    check(input.vision_position_ids.size() ==
              static_cast<std::size_t>(input.grid_h) * input.grid_w * 2U,
          "Ref2VA 2-D rotary IDs cover every patch");
    check(input.position_indices[0] == 0, "Ref2VA first learned-position tap is top-left");
    check(input.position_indices[1] == 1, "Ref2VA second learned-position tap is top-right");
    check(input.position_indices[2] == 48, "Ref2VA third learned-position tap is bottom-left");
    check(input.position_indices[3] == 49, "Ref2VA fourth learned-position tap is bottom-right");
    check(input.position_weights[0] == 1.0F,
          "Ref2VA first learned-position weight is exact FP32 one");
    check(input.position_weights[1] == 0.0F, "Ref2VA second learned-position weight is zero");
    check(input.position_weights[2] == 0.0F, "Ref2VA third learned-position weight is zero");
    check(input.position_weights[3] == 0.0F, "Ref2VA fourth learned-position weight is zero");
    check(input.vision_position_ids[0] == 0, "Ref2VA first rotary row is zero");
    check(input.vision_position_ids[1] == 0, "Ref2VA first rotary column is zero");
    check(input.vision_position_ids[2] == 0, "Ref2VA second rotary row remains zero");
    check(input.vision_position_ids[3] == 1, "Ref2VA second rotary column increments");
    check(input.vision_position_ids[4] == 1, "Ref2VA third rotary row increments");
    check(input.vision_position_ids[5] == 0, "Ref2VA third rotary column resets");
}

void test_image_reference_builds_dynamic_qwen_run() {
    std::vector<std::string> calls;
    std::vector<trtmc::AudioVideoReference> sources = {
        source_reference(trtmc::AudioVideoReferenceKind::kImage)};
    std::vector<trtmc::MiniMaxH3PreparedReference> prepared;
    prepared.push_back(image_reference(0));
    const auto presentation = trtmc::minimax_h3_build_ref2va_conditioner_presentation(
        "prompt", sources, prepared, make_tokenizer(calls), {900, 901, 902, 903});

    check(calls == std::vector<std::string>({"<Picture 1>: ", "prompt"}),
          "Ref2VA image presentation uses exact Picture label then prompt");
    check(presentation.sequence_rows == 4100, "Ref2VA square image produces 4096 vision rows");
    check(presentation.input_ids[1] == 900, "Ref2VA image block starts with vision-start");
    check(presentation.input_ids[2] == 901, "Ref2VA image block begins with image-pad");
    check(presentation.input_ids[4097] == 901, "Ref2VA image block ends with image-pad");
    check(presentation.input_ids[4098] == 903, "Ref2VA image block closes with vision-end");
    check(range_is(presentation.h3_token_tags, 1, 4099, 0),
          "Ref2VA complete image vision block carries the H3 video tag");
    check(range_is(presentation.qwen_mm_token_type_ids, 2, 4098, 1),
          "Ref2VA image pads alone use the Qwen image type");
    check(range_is(presentation.vision_selector, 2, 4098, 1),
          "Ref2VA image selector covers exactly the image pads");
    check(presentation.vision_selector[1] == 0, "Ref2VA image vision-start is not scattered");
    check(presentation.vision_selector[4098] == 0, "Ref2VA image vision-end is not scattered");
    check(presentation.vision_run_lengths == std::vector<int32_t>({4096}),
          "Ref2VA image run metadata retains its length");
    check(presentation.vision_run_reference_ids == std::vector<int32_t>({0}),
          "Ref2VA image run metadata retains reference identity");
    check(presentation.vision_scatter[0].sequence_row_begin == 2,
          "Ref2VA image scatter begins at the first image pad");
    check(presentation.vision_scatter[0].compact_row_count == 4096,
          "Ref2VA image scatter covers every compact row");
    check(presentation.vision_scatter_indices.front() == 2,
          "Ref2VA image scatter index starts at the first image pad");
    check(presentation.vision_scatter_indices.back() == 4097,
          "Ref2VA image scatter index ends at the last image pad");
    check(mrope_at(presentation, 0, 2) == 2, "Ref2VA image temporal MRoPE starts after text");
    check(mrope_at(presentation, 1, 4097) == 65, "Ref2VA image height MRoPE reaches 65");
    check(mrope_at(presentation, 2, 4097) == 65, "Ref2VA image width MRoPE reaches 65");
    check(presentation.next_mrope_position == 68,
          "Ref2VA image MRoPE advances past vision-end and prompt");

    const auto& input = presentation.vision_inputs[0];
    check(input.grid_h == 128, "Ref2VA square image grid height is 128 patches");
    check(input.grid_w == 128, "Ref2VA square image grid width is 128 patches");
    check(input.pixel_values.size() == 16384U * trtmc::kMiniMaxH3Ref2VAPatchVectorSize,
          "Ref2VA image patchification emits the dynamic Qwen pixel ABI");
    check(input.pixel_values[0] == 1.0F, "Ref2VA image first temporal pixel is normalized");
    check(input.pixel_values[256] == 1.0F, "Ref2VA image pixel is duplicated in time");
    check(input.pixel_values[1536U + 512U] == -1.0F,
          "Ref2VA image merge ordering keeps the next patch distinct");
    check_first_position_bindings(input);
    check(std::abs(input.position_weights[4] - 0.62992126F) < 1.0e-7F,
          "Ref2VA image interpolation preserves the nontrivial left weight in FP32");
    check(std::abs(input.position_weights[5] - 0.37007874F) < 1.0e-7F,
          "Ref2VA image interpolation preserves the nontrivial right weight in FP32");
}

void test_video_soundtrack_precedes_timestamped_video_pair() {
    std::vector<std::string> calls;
    std::vector<trtmc::AudioVideoReference> sources = {
        source_reference(trtmc::AudioVideoReferenceKind::kVideo, true)};
    std::vector<trtmc::MiniMaxH3PreparedReference> prepared;
    prepared.push_back(video_reference(0, true, 25));
    const auto presentation = trtmc::minimax_h3_build_ref2va_conditioner_presentation(
        "prompt", sources, prepared, make_tokenizer(calls), {900, 901, 902, 903});

    check(calls == std::vector<std::string>(
                       {"<Audio 1>: ", "<Video 1>: ", "<0.2 seconds>", "<1.0 seconds>", "prompt"}),
          "Ref2VA soundtrack precedes paired video timestamps with repeat-last semantics");
    check(presentation.sequence_rows == 1161, "Ref2VA three samples produce two temporal pairs");
    check(presentation.input_ids[3] == 900, "Ref2VA temporal pair starts with vision-start");
    check(presentation.input_ids[4] == 902, "Ref2VA temporal pair begins with video-pad");
    check(presentation.input_ids[579] == 902, "Ref2VA temporal pair ends with video-pad");
    check(presentation.input_ids[580] == 903, "Ref2VA temporal pair closes with vision-end");
    check(range_is(presentation.qwen_mm_token_type_ids, 4, 580, 2),
          "Ref2VA video pads alone use the Qwen video type");
    check(range_is(presentation.vision_selector, 4, 580, 1),
          "Ref2VA video selector covers exactly the first pair pads");
    check(presentation.audio_labels.size() == 1, "Ref2VA video exposes one soundtrack label");
    check(presentation.audio_labels[0].reference_index == 0,
          "Ref2VA soundtrack metadata retains request order");
    check(presentation.audio_labels[0].from_video_soundtrack,
          "Ref2VA soundtrack metadata retains video ownership");
    check(mrope_at(presentation, 0, 4) == 4, "Ref2VA video temporal MRoPE starts after text");
    check(mrope_at(presentation, 1, 579) == 27, "Ref2VA video height MRoPE reaches 27");
    check(mrope_at(presentation, 2, 579) == 27, "Ref2VA video width MRoPE reaches 27");
    check(presentation.vision_run_lengths == std::vector<int32_t>({576, 576}),
          "Ref2VA video emits one vision run per temporal pair");
    check(presentation.vision_run_reference_ids == std::vector<int32_t>({0, 0}),
          "Ref2VA temporal-pair runs retain their shared reference ID");
    check(presentation.vision_scatter[1].sequence_row_begin == 583,
          "Ref2VA second temporal pair scatters after its timestamp");
    check(presentation.next_mrope_position == 57,
          "Ref2VA multi-pair MRoPE advances once per separated run");

    const auto& input = presentation.vision_inputs[0];
    check(input.kind == trtmc::MiniMaxH3Ref2VAVisionKind::kVideo,
          "Ref2VA temporal pair is a video vision run");
    check(input.timestamp_seconds == 0.25F, "Ref2VA first temporal-pair timestamp is 0.25s");
    check(input.grid_h == 48, "Ref2VA video grid height is 48 patches");
    check(input.grid_w == 48, "Ref2VA video grid width is 48 patches");
    check(input.pixel_values[0] == 1.0F, "Ref2VA temporal patch starts with frame zero");
    check(input.pixel_values[768U] == -1.0F,
          "Ref2VA temporal patch follows with normalized frame twelve");
    check(presentation.vision_inputs[1].timestamp_seconds == 1.0F,
          "Ref2VA odd sampled frame repeats its timestamp in the final pair");
    check_first_position_bindings(input);
}

void test_image_then_audio_preserves_reference_order() {
    std::vector<std::string> calls;
    std::vector<trtmc::AudioVideoReference> sources = {
        source_reference(trtmc::AudioVideoReferenceKind::kImage),
        source_reference(trtmc::AudioVideoReferenceKind::kAudio),
    };
    std::vector<trtmc::MiniMaxH3PreparedReference> prepared;
    prepared.push_back(image_reference(0));
    prepared.push_back(audio_reference(1));
    const auto presentation = trtmc::minimax_h3_build_ref2va_conditioner_presentation(
        "prompt", sources, prepared, make_tokenizer(calls), {900, 901, 902, 903});

    check(calls == std::vector<std::string>({"<Picture 1>: ", "<Audio 1>: ", "prompt"}),
          "Ref2VA image plus audio labels follow request order");
    check(presentation.input_ids[4099] == 1002 && presentation.input_ids[4100] == 1003,
          "Ref2VA standalone Audio label follows the complete image block");
    check(presentation.audio_labels.size() == 1, "Ref2VA image plus audio has one audio label");
    check(presentation.audio_labels[0].reference_index == 1,
          "Ref2VA standalone audio metadata keeps its original reference ID");
    check(!presentation.audio_labels[0].from_video_soundtrack,
          "Ref2VA standalone audio metadata is not a soundtrack");
    check(presentation.vision_run_reference_ids == std::vector<int32_t>({0}),
          "Ref2VA image plus audio emits no fake audio vision run");
}

void test_mixed_reordered_references_keep_semantic_order() {
    std::vector<std::string> calls;
    std::vector<trtmc::AudioVideoReference> sources = {
        source_reference(trtmc::AudioVideoReferenceKind::kAudio),
        source_reference(trtmc::AudioVideoReferenceKind::kVideo, true),
        source_reference(trtmc::AudioVideoReferenceKind::kImage),
    };
    std::vector<trtmc::MiniMaxH3PreparedReference> prepared;
    prepared.push_back(audio_reference(0));
    prepared.push_back(video_reference(1, true));
    prepared.push_back(image_reference(2));
    const auto presentation = trtmc::minimax_h3_build_ref2va_conditioner_presentation(
        "prompt", sources, prepared, make_tokenizer(calls), {900, 901, 902, 903});

    check(calls == std::vector<std::string>({"<Audio 1>: ", "<Audio 2>: ", "<Video 1>: ",
                                             "<0.2 seconds>", "<Picture 1>: ", "prompt"}),
          "Ref2VA mixed labels preserve request order with independent modality numbering");
    check(presentation.vision_run_reference_ids == std::vector<int32_t>({1, 2}),
          "Ref2VA mixed vision runs retain video-before-image reference order");
    check(presentation.vision_run_lengths == std::vector<int32_t>({576, 4096}),
          "Ref2VA mixed vision runs retain per-grid lengths");
    check(presentation.vision_scatter[0].compact_row_begin == 0,
          "Ref2VA mixed video compact rows start first");
    check(presentation.vision_scatter[0].sequence_row_begin == 5,
          "Ref2VA mixed video scatter follows its timestamp");
    check(presentation.vision_scatter[1].compact_row_begin == 576,
          "Ref2VA mixed image compact rows follow the video");
    check(presentation.vision_scatter[1].sequence_row_begin == 584,
          "Ref2VA mixed image scatter follows its Picture label");
    check(presentation.vision_scatter[0].grid_h == 48,
          "Ref2VA mixed video scatter retains its grid height");
    check(presentation.vision_scatter[0].grid_w == 48,
          "Ref2VA mixed video scatter retains its grid width");
    check(presentation.vision_scatter[1].grid_h == 128,
          "Ref2VA mixed image scatter retains its grid height");
    check(presentation.vision_scatter[1].grid_w == 128,
          "Ref2VA mixed image scatter retains its grid width");
    check(presentation.audio_labels.size() == 2, "Ref2VA mixed request has two audio labels");
    check(presentation.audio_labels[0].reference_index == 0,
          "Ref2VA mixed standalone audio stays first");
    check(presentation.audio_labels[1].reference_index == 1,
          "Ref2VA mixed soundtrack stays second");
    check(presentation.next_mrope_position == 98,
          "Ref2VA mixed multi-run MRoPE reaches the exact next position");
    check(presentation.mrope_position_delta == -4584,
          "Ref2VA mixed multi-run MRoPE delta is independent of token count");
}

} // namespace

int main() {
    test_image_reference_builds_dynamic_qwen_run();
    test_video_soundtrack_precedes_timestamped_video_pair();
    test_image_then_audio_preserves_reference_order();
    test_mixed_reordered_references_keep_semantic_order();
    if (failures != 0)
        return 1;
    std::cout << "MiniMax-H3 Ref2VA conditioner tests passed\n";
    return 0;
}
