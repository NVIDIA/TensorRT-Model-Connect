/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_video_session.h"
#include "trtmc/models/sam2_video.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void check_status(int32_t actual, int32_t expected, const char* message) {
    if (actual != expected) {
        std::cerr << "FAIL: " << message << " (status " << actual << ", expected " << expected
                  << ", error '" << trtmc_sam2_video_last_error() << "')\n";
        std::exit(1);
    }
}

void check_error_contains(const char* needle, const char* message) {
    const char* error = trtmc_sam2_video_last_error();
    check(error != nullptr && std::strstr(error, needle) != nullptr, message);
}

using Handle = std::unique_ptr<TrtmcSam2VideoSession, decltype(&trtmc_sam2_video_session_destroy)>;
using FrameStorage = std::array<std::vector<float>, trtmc::kSam2VideoFrameCount>;

FrameStorage make_frame_storage() {
    FrameStorage storage;
    for (std::size_t index = 0; index < storage.size(); ++index)
        storage[index] = std::vector<float>(18, 0.1F + static_cast<float>(index) * 0.1F);
    return storage;
}

trtmc::Sam2VideoFrameResult make_host_frame_result(const trtmc::Sam2VideoFrameView& frame) {
    trtmc::Sam2VideoFrameResult result;
    result.frame_index = frame.frame_index;
    result.height = frame.height;
    result.width = frame.width;
    const auto area =
        static_cast<std::size_t>(frame.height) * static_cast<std::size_t>(frame.width);
    std::vector<uint8_t> mask(area);
    for (std::size_t index = 0; index < mask.size(); ++index)
        mask[index] =
            static_cast<uint8_t>((index + static_cast<std::size_t>(frame.frame_index)) & 1U);
    result.mask = trtmc::Sam2VideoMaskBuffer::host(std::move(mask));
    return result;
}

trtmc::Sam2VideoPromptResult make_host_prompt_result(const trtmc::Sam2VideoFrames& frames) {
    trtmc::Sam2VideoPromptResult result;
    result.track.label = 7;
    result.track.detector_score = 0.875F;
    result.track.prompt_box_xyxy = {-4.5F, -2.0F, static_cast<float>(frames.front().width) + 7.0F,
                                    static_cast<float>(frames.front().height) + 3.0F};
    result.frame_zero = make_host_frame_result(frames.front());
    return result;
}

trtmc::Sam2VideoFrameResults make_host_propagation(const trtmc::Sam2VideoFrames& frames) {
    trtmc::Sam2VideoFrameResults results;
    for (std::size_t index = 0; index < results.size(); ++index)
        results[index] = make_host_frame_result(frames[index]);
    return results;
}

trtmc::Sam2VideoProcessor make_host_processor() {
    trtmc::Sam2VideoProcessor processor;
    processor.reset = [] {};
    processor.run_bbox_prompt = [](const trtmc::Sam2VideoFrames& frames) {
        return make_host_prompt_result(frames);
    };
    processor.propagate = [](const trtmc::Sam2VideoPromptResult&,
                             const trtmc::Sam2VideoFrames& frames) {
        return make_host_propagation(frames);
    };
    return processor;
}

Handle make_handle(trtmc::Sam2VideoProcessor processor = make_host_processor(),
                   trtmc::Sam2VideoLimits limits = {}) {
    return Handle(trtmc::make_sam2_video_session_handle(std::move(processor), limits),
                  &trtmc_sam2_video_session_destroy);
}

void append_five_frames(TrtmcSam2VideoSession* session, FrameStorage& storage) {
    check_status(trtmc_sam2_video_begin_v1(session), TRTMC_SAM2_VIDEO_STATUS_OK,
                 "sam2 begins its fixed five-frame run");
    for (std::size_t index = 0; index < storage.size(); ++index) {
        check_status(trtmc_sam2_video_append_frame_v1(session, storage[index].data(), 2, 3),
                     TRTMC_SAM2_VIDEO_STATUS_OK,
                     "sam2 accepts the next contiguous frame in its fixed run");
    }
}

void test_processor_owned_state_reset_and_failure() {
    int reset_calls = 0;
    auto processor = make_host_processor();
    processor.reset = [&reset_calls] { ++reset_calls; };
    auto handle = make_handle(std::move(processor));
    auto storage = make_frame_storage();
    append_five_frames(handle.get(), storage);
    check_status(trtmc_sam2_video_run_bbox_prompt_v1(handle.get()), TRTMC_SAM2_VIDEO_STATUS_OK,
                 "sam2 creates processor-owned run state");
    check_status(trtmc_sam2_video_reset_v1(handle.get()), TRTMC_SAM2_VIDEO_STATUS_OK,
                 "sam2 resets wrapper and processor-owned state together");
    check(reset_calls == 1, "sam2 invokes the processor reset callback exactly once");
    check_status(trtmc_sam2_video_begin_v1(handle.get()), TRTMC_SAM2_VIDEO_STATUS_OK,
                 "sam2 permits reuse only after the processor reset succeeds");

    auto failing_processor = make_host_processor();
    failing_processor.reset = [] { throw std::runtime_error("synthetic reset failure"); };
    auto failing = make_handle(std::move(failing_processor));
    check_status(trtmc_sam2_video_reset_v1(failing.get()), TRTMC_SAM2_VIDEO_STATUS_PROCESSOR_ERROR,
                 "sam2 surfaces a processor-owned reset failure");
    check_error_contains("reset", "sam2 reset failure identifies the processor stage");
    check_status(trtmc_sam2_video_begin_v1(failing.get()), TRTMC_SAM2_VIDEO_STATUS_INVALID_STATE,
                 "sam2 poisons a session whose processor reset failed");
}

void test_exact_contract_track_and_frame_lifetime() {
    check(TRTMC_SAM2_VIDEO_FRAME_COUNT_V1 == 5U && TRTMC_SAM2_VIDEO_OBJECT_COUNT_V1 == 1U,
          "sam2 ABI declares exactly five frames and one selected object");
    check(trtmc_sam2_video_abi_version() == TRTMC_SAM2_VIDEO_ABI_VERSION_1,
          "sam2 reports version-one ABI");

    auto handle = make_handle();
    auto storage = make_frame_storage();
    append_five_frames(handle.get(), storage);
    check_status(trtmc_sam2_video_run_bbox_prompt_v1(handle.get()), TRTMC_SAM2_VIDEO_STATUS_OK,
                 "sam2 selects exactly one frame-zero detector result");

    TrtmcSam2VideoTrackV1 track{};
    check_status(trtmc_sam2_video_get_track_v1(handle.get(), &track, sizeof(track)),
                 TRTMC_SAM2_VIDEO_STATUS_OK, "sam2 exposes selected-track metadata once");
    check(track.struct_size == sizeof(track) &&
              track.abi_version == TRTMC_SAM2_VIDEO_ABI_VERSION_1 &&
              track.flags == TRTMC_SAM2_VIDEO_TRACK_PROMPT_BOX_ABSOLUTE_XYXY && track.label == 7 &&
              track.detector_score == 0.875F && track.prompt_box_xyxy[0] == -4.5F &&
              track.prompt_box_xyxy[1] == -2.0F && track.prompt_box_xyxy[2] == 10.0F &&
              track.prompt_box_xyxy[3] == 5.0F,
          "sam2 track keeps the detector label, score, and unclipped original-space prompt box");

    uint64_t result_count = 0;
    check_status(trtmc_sam2_video_result_count_v1(handle.get(), &result_count),
                 TRTMC_SAM2_VIDEO_STATUS_OK, "sam2 exposes the prompt mask before propagation");
    check(result_count == 1, "sam2 prompt state has exactly one frame-zero mask view");

    TrtmcSam2VideoFrameResultV1 prompt{};
    check_status(trtmc_sam2_video_get_frame_result_v1(handle.get(), 0, TRTMC_SAM2_VIDEO_GET_DEFAULT,
                                                      &prompt, sizeof(prompt)),
                 TRTMC_SAM2_VIDEO_STATUS_OK, "sam2 exposes the frame-zero prompt mask");
    check(prompt.struct_size == sizeof(prompt) &&
              prompt.abi_version == TRTMC_SAM2_VIDEO_ABI_VERSION_1 &&
              prompt.flags == TRTMC_SAM2_VIDEO_FRAME_MASK_BINARY && prompt.frame_index == 0 &&
              prompt.height == 2 && prompt.width == 3 &&
              prompt.mask_memory_kind == TRTMC_SAM2_VIDEO_MASK_MEMORY_HOST &&
              prompt.mask_device_ordinal == -1 && prompt.mask_byte_count == 6 &&
              prompt.mask_row_stride_bytes == 3,
          "sam2 per-frame view contains only its binary singular-mask contract");
    const uint8_t* prompt_mask = static_cast<const uint8_t*>(prompt.mask);

    uint64_t frame_count = 0;
    check_status(trtmc_sam2_video_propagate_v1(handle.get(), &frame_count),
                 TRTMC_SAM2_VIDEO_STATUS_OK, "sam2 runs its fixed five-frame propagation once");
    check(frame_count == TRTMC_SAM2_VIDEO_FRAME_COUNT_V1,
          "sam2 propagation reports exactly five frames");
    check(prompt_mask[0] == 0 && prompt_mask[1] == 1,
          "sam2 keeps the pre-propagation prompt mask view alive");

    check_status(trtmc_sam2_video_result_count_v1(handle.get(), &result_count),
                 TRTMC_SAM2_VIDEO_STATUS_OK, "sam2 reports propagated result count");
    check(result_count == TRTMC_SAM2_VIDEO_FRAME_COUNT_V1,
          "sam2 exposes exactly one mask for each of five frames");
    for (uint64_t index = 0; index < result_count; ++index) {
        TrtmcSam2VideoFrameResultV1 frame{};
        check_status(trtmc_sam2_video_get_frame_result_v1(
                         handle.get(), index, TRTMC_SAM2_VIDEO_GET_DEFAULT, &frame, sizeof(frame)),
                     TRTMC_SAM2_VIDEO_STATUS_OK,
                     "sam2 exposes each temporally ordered propagated mask");
        check(frame.frame_index == static_cast<int32_t>(index) && frame.mask_byte_count == 6 &&
                  static_cast<const uint8_t*>(frame.mask)[0] == (index & 1U),
              "sam2 propagated result has a contiguous index and singular mask");
    }

    TrtmcSam2VideoTrackV1 track_after_propagation{};
    check_status(trtmc_sam2_video_get_track_v1(handle.get(), &track_after_propagation,
                                               sizeof(track_after_propagation)),
                 TRTMC_SAM2_VIDEO_STATUS_OK,
                 "sam2 selected-track metadata remains queryable after propagation");
    check(track_after_propagation.label == track.label &&
              track_after_propagation.detector_score == track.detector_score &&
              track_after_propagation.prompt_box_xyxy[0] == track.prompt_box_xyxy[0],
          "sam2 does not invent propagated detector metadata");

    check_status(trtmc_sam2_video_run_bbox_prompt_v1(handle.get()),
                 TRTMC_SAM2_VIDEO_STATUS_INVALID_STATE, "sam2 rejects a second bbox prompt");
    check_status(trtmc_sam2_video_propagate_v1(handle.get(), &frame_count),
                 TRTMC_SAM2_VIDEO_STATUS_INVALID_STATE, "sam2 rejects a second propagation");
    check_status(trtmc_sam2_video_reset_v1(handle.get()), TRTMC_SAM2_VIDEO_STATUS_OK,
                 "sam2 reset makes the session reusable");
    check_status(trtmc_sam2_video_get_track_v1(handle.get(), &track, sizeof(track)),
                 TRTMC_SAM2_VIDEO_STATUS_INVALID_STATE,
                 "sam2 reset invalidates the selected-track query");
    check_status(trtmc_sam2_video_result_count_v1(handle.get(), &result_count),
                 TRTMC_SAM2_VIDEO_STATUS_INVALID_STATE,
                 "sam2 reset invalidates previously queryable masks");
}

void test_state_and_tightly_packed_input_validation() {
    trtmc::Sam2VideoLimits validation_limits;
    validation_limits.validate_input_values = true;
    auto handle = make_handle(make_host_processor(), validation_limits);
    std::vector<float> pixels(18, 0.5F);
    check_status(trtmc_sam2_video_append_frame_v1(handle.get(), pixels.data(), 2, 3),
                 TRTMC_SAM2_VIDEO_STATUS_INVALID_STATE, "sam2 rejects append before begin");
    check_status(trtmc_sam2_video_begin_v1(handle.get()), TRTMC_SAM2_VIDEO_STATUS_OK,
                 "sam2 begins without advertising a variable frame count");
    check_status(trtmc_sam2_video_begin_v1(handle.get()), TRTMC_SAM2_VIDEO_STATUS_INVALID_STATE,
                 "sam2 rejects a second begin");
    check_status(trtmc_sam2_video_append_frame_v1(handle.get(), nullptr, 2, 3),
                 TRTMC_SAM2_VIDEO_STATUS_INVALID_ARGUMENT, "sam2 rejects null pixels");
    check_status(trtmc_sam2_video_append_frame_v1(handle.get(), pixels.data(), 0, 3),
                 TRTMC_SAM2_VIDEO_STATUS_INVALID_ARGUMENT, "sam2 rejects invalid geometry");

    std::vector<float> non_finite = pixels;
    non_finite[4] = std::numeric_limits<float>::quiet_NaN();
    check_status(trtmc_sam2_video_append_frame_v1(handle.get(), non_finite.data(), 2, 3),
                 TRTMC_SAM2_VIDEO_STATUS_INVALID_ARGUMENT,
                 "sam2 diagnostic validation rejects non-finite decoded pixels");
    std::vector<float> out_of_range = pixels;
    out_of_range[4] = 1.01F;
    check_status(trtmc_sam2_video_append_frame_v1(handle.get(), out_of_range.data(), 2, 3),
                 TRTMC_SAM2_VIDEO_STATUS_INVALID_ARGUMENT,
                 "sam2 diagnostic validation rejects decoded pixels outside [0,1]");

    check_status(trtmc_sam2_video_append_frame_v1(handle.get(), pixels.data(), 2, 3),
                 TRTMC_SAM2_VIDEO_STATUS_OK, "sam2 accepts a tightly packed RGB HWC frame");
    check_status(trtmc_sam2_video_append_frame_v1(handle.get(), pixels.data(), 3, 2),
                 TRTMC_SAM2_VIDEO_STATUS_INVALID_ARGUMENT,
                 "sam2 rejects mixed video geometry even at equal element count");
    check_status(trtmc_sam2_video_run_bbox_prompt_v1(handle.get()),
                 TRTMC_SAM2_VIDEO_STATUS_INVALID_STATE,
                 "sam2 prompt waits for all five contiguous frames");
    for (std::size_t index = 1; index < trtmc::kSam2VideoFrameCount; ++index) {
        check_status(trtmc_sam2_video_append_frame_v1(handle.get(), pixels.data(), 2, 3),
                     TRTMC_SAM2_VIDEO_STATUS_OK,
                     "sam2 accepts the remainder of the fixed five-frame input");
    }
    check_status(trtmc_sam2_video_append_frame_v1(handle.get(), pixels.data(), 2, 3),
                 TRTMC_SAM2_VIDEO_STATUS_INVALID_STATE,
                 "sam2 rejects a sixth frame instead of implying a longer graph");

    TrtmcSam2VideoTrackV1 track{};
    TrtmcSam2VideoFrameResultV1 result{};
    check_status(trtmc_sam2_video_get_track_v1(handle.get(), &track, sizeof(track) - 1U),
                 TRTMC_SAM2_VIDEO_STATUS_UNSUPPORTED_ABI,
                 "sam2 rejects an undersized track ABI structure");
    check_status(trtmc_sam2_video_get_frame_result_v1(handle.get(), 0, TRTMC_SAM2_VIDEO_GET_DEFAULT,
                                                      &result, sizeof(result) - 1U),
                 TRTMC_SAM2_VIDEO_STATUS_UNSUPPORTED_ABI,
                 "sam2 rejects an undersized frame-result ABI structure");
    check_status(trtmc_sam2_video_get_frame_result_v1(handle.get(), 0, UINT32_C(0x80000000),
                                                      &result, sizeof(result)),
                 TRTMC_SAM2_VIDEO_STATUS_UNSUPPORTED_ABI,
                 "sam2 rejects unknown version-one query flags");
    check_status(trtmc_sam2_video_begin_v1(nullptr), TRTMC_SAM2_VIDEO_STATUS_INVALID_ARGUMENT,
                 "sam2 rejects a null session");

    check_status(trtmc_sam2_video_reset_v1(handle.get()), TRTMC_SAM2_VIDEO_STATUS_OK,
                 "sam2 resets after input validation errors");
    trtmc::Sam2VideoLimits small_limits;
    small_limits.max_frame_elements = 18;
    auto bounded = make_handle(make_host_processor(), small_limits);
    check_status(trtmc_sam2_video_begin_v1(bounded.get()), TRTMC_SAM2_VIDEO_STATUS_OK,
                 "sam2 begins a bounded geometry run");
    float one_pixel = 0.5F;
    check_status(trtmc_sam2_video_append_frame_v1(bounded.get(), &one_pixel,
                                                  std::numeric_limits<int32_t>::max(),
                                                  std::numeric_limits<int32_t>::max()),
                 TRTMC_SAM2_VIDEO_STATUS_INVALID_ARGUMENT,
                 "sam2 rejects impractical dimensions before reading the pixel buffer");
}

using PromptMutator = std::function<void(trtmc::Sam2VideoPromptResult&)>;

void expect_invalid_prompt_result(PromptMutator mutate, const char* error_fragment) {
    auto processor = make_host_processor();
    processor.run_bbox_prompt = [mutate = std::move(mutate)](const trtmc::Sam2VideoFrames& frames) {
        auto result = make_host_prompt_result(frames);
        mutate(result);
        return result;
    };
    auto handle = make_handle(std::move(processor));
    auto storage = make_frame_storage();
    append_five_frames(handle.get(), storage);
    const auto status = trtmc_sam2_video_run_bbox_prompt_v1(handle.get());
    if (status != TRTMC_SAM2_VIDEO_STATUS_INVALID_RESULT) {
        std::cerr << "FAIL: sam2 rejects malformed prompt result containing '" << error_fragment
                  << "' (status " << status << ", error '" << trtmc_sam2_video_last_error()
                  << "')\n";
        std::exit(1);
    }
    check_error_contains(error_fragment, "sam2 reports the malformed prompt-result field");
    uint64_t count = 0;
    check_status(trtmc_sam2_video_result_count_v1(handle.get(), &count),
                 TRTMC_SAM2_VIDEO_STATUS_INVALID_STATE,
                 "sam2 poisons the session after malformed prompt output");
    check_status(trtmc_sam2_video_reset_v1(handle.get()), TRTMC_SAM2_VIDEO_STATUS_OK,
                 "sam2 reset recovers a poisoned prompt session");
}

void test_strict_selected_track_and_prompt_mask_validation() {
    expect_invalid_prompt_result([](auto& result) { result.track.label = -1; }, "label");
    expect_invalid_prompt_result(
        [](auto& result) { result.track.detector_score = std::numeric_limits<float>::infinity(); },
        "finite probability");
    expect_invalid_prompt_result([](auto& result) { result.track.detector_score = 1.1F; },
                                 "finite probability");
    expect_invalid_prompt_result(
        [](auto& result) {
            result.track.prompt_box_xyxy[0] = std::numeric_limits<float>::quiet_NaN();
        },
        "finite");
    expect_invalid_prompt_result([](auto& result) { result.track.prompt_box_xyxy[0] = 20.0F; },
                                 "ordered");
    expect_invalid_prompt_result([](auto& result) { result.frame_zero.frame_index = 1; },
                                 "geometry");
    expect_invalid_prompt_result(
        [](auto& result) { result.frame_zero.mask = trtmc::Sam2VideoMaskBuffer::host({0, 1}); },
        "byte count");
    expect_invalid_prompt_result(
        [](auto& result) {
            std::vector<uint8_t> mask(6, 0);
            mask[4] = 2;
            result.frame_zero.mask = trtmc::Sam2VideoMaskBuffer::host(std::move(mask));
        },
        "binary");
    expect_invalid_prompt_result(
        [](auto& result) {
            static const uint8_t device_bytes[6]{};
            result.frame_zero.mask = trtmc::Sam2VideoMaskBuffer::cuda_device_binary(
                device_bytes, sizeof(device_bytes), 0, {},
                [] { return std::vector<uint8_t>(6, uint8_t{0}); });
        },
        "require an owner");
    expect_invalid_prompt_result(
        [](auto& result) {
            auto storage = std::make_shared<std::vector<uint8_t>>(6, uint8_t{0});
            std::shared_ptr<const void> owner(storage, storage->data());
            result.frame_zero.mask = trtmc::Sam2VideoMaskBuffer::cuda_device_binary(
                nullptr, storage->size(), 0, std::move(owner), [storage] { return *storage; });
        },
        "must not be null");
    expect_invalid_prompt_result(
        [](auto& result) {
            auto storage = std::make_shared<std::vector<uint8_t>>(6, uint8_t{0});
            std::shared_ptr<const void> owner(storage, storage->data());
            result.frame_zero.mask = trtmc::Sam2VideoMaskBuffer::cuda_device_binary(
                storage->data(), storage->size(), -1, std::move(owner),
                [storage] { return *storage; });
        },
        "device ordinal");
    expect_invalid_prompt_result(
        [](auto& result) {
            auto storage = std::make_shared<std::vector<uint8_t>>(6, uint8_t{0});
            std::shared_ptr<const void> owner(storage, storage->data());
            result.frame_zero.mask = trtmc::Sam2VideoMaskBuffer::cuda_device_binary(
                storage->data(), storage->size(), 0, std::move(owner), {});
        },
        "host materializer");
}

using PropagationMutator = std::function<void(trtmc::Sam2VideoFrameResults&)>;

void expect_invalid_propagation(PropagationMutator mutate, const char* error_fragment) {
    auto processor = make_host_processor();
    processor.propagate = [mutate = std::move(mutate)](const trtmc::Sam2VideoPromptResult&,
                                                       const trtmc::Sam2VideoFrames& frames) {
        auto results = make_host_propagation(frames);
        mutate(results);
        return results;
    };
    auto handle = make_handle(std::move(processor));
    auto storage = make_frame_storage();
    append_five_frames(handle.get(), storage);
    check_status(trtmc_sam2_video_run_bbox_prompt_v1(handle.get()), TRTMC_SAM2_VIDEO_STATUS_OK,
                 "sam2 propagation fixture runs its prompt");
    uint64_t frames = 0;
    check_status(trtmc_sam2_video_propagate_v1(handle.get(), &frames),
                 TRTMC_SAM2_VIDEO_STATUS_INVALID_RESULT,
                 "sam2 rejects malformed propagated output");
    check_error_contains(error_fragment, "sam2 reports malformed propagation detail");
    check_status(trtmc_sam2_video_propagate_v1(handle.get(), &frames),
                 TRTMC_SAM2_VIDEO_STATUS_INVALID_STATE,
                 "sam2 poisons a session after malformed propagation");
}

void test_propagated_results_are_exact_contiguous_masks() {
    static_assert(std::tuple_size_v<trtmc::Sam2VideoFrameResults> == 5,
                  "the processor output type must be structurally five frames");
    expect_invalid_propagation([](auto& results) { results[3].frame_index = 2; }, "geometry");
    expect_invalid_propagation([](auto& results) { results[4].width = 4; }, "geometry");
    expect_invalid_propagation(
        [](auto& results) { results[1].mask = trtmc::Sam2VideoMaskBuffer::host({0}); },
        "byte count");
    expect_invalid_propagation(
        [](auto& results) {
            std::vector<uint8_t> mask(6, 0);
            mask[5] = 3;
            results[2].mask = trtmc::Sam2VideoMaskBuffer::host(std::move(mask));
        },
        "binary");
}

trtmc::Sam2VideoFrameResult
make_device_frame_result(const trtmc::Sam2VideoFrameView& frame, int* materialize_calls,
                         const std::shared_ptr<std::weak_ptr<std::vector<uint8_t>>>& owner_probe) {
    auto result = make_host_frame_result(frame);
    const auto bytes =
        static_cast<std::size_t>(frame.height) * static_cast<std::size_t>(frame.width);
    auto storage = std::make_shared<std::vector<uint8_t>>(bytes, uint8_t{1});
    *owner_probe = storage;
    const void* device_pointer = storage->data();
    std::shared_ptr<const void> owner(storage, storage->data());
    std::weak_ptr<std::vector<uint8_t>> weak_storage = storage;
    result.mask = trtmc::Sam2VideoMaskBuffer::cuda_device_binary(
        device_pointer, bytes, 0, std::move(owner), [weak_storage, materialize_calls] {
            ++*materialize_calls;
            const auto locked = weak_storage.lock();
            if (!locked)
                throw std::runtime_error("device allocation expired");
            return *locked;
        });
    return result;
}

void test_device_masks_stay_lazy_and_session_owned() {
    int materialize_calls = 0;
    auto prompt_owner_probe = std::make_shared<std::weak_ptr<std::vector<uint8_t>>>();
    auto processor = make_host_processor();
    processor.run_bbox_prompt = [prompt_owner_probe,
                                 &materialize_calls](const trtmc::Sam2VideoFrames& frames) {
        auto result = make_host_prompt_result(frames);
        result.frame_zero =
            make_device_frame_result(frames.front(), &materialize_calls, prompt_owner_probe);
        return result;
    };
    processor.propagate = [&materialize_calls](const trtmc::Sam2VideoPromptResult&,
                                               const trtmc::Sam2VideoFrames& frames) {
        trtmc::Sam2VideoFrameResults results;
        for (std::size_t index = 0; index < results.size(); ++index) {
            auto unused_probe = std::make_shared<std::weak_ptr<std::vector<uint8_t>>>();
            results[index] =
                make_device_frame_result(frames[index], &materialize_calls, unused_probe);
        }
        return results;
    };

    auto handle = make_handle(std::move(processor));
    auto storage = make_frame_storage();
    append_five_frames(handle.get(), storage);
    check_status(trtmc_sam2_video_run_bbox_prompt_v1(handle.get()), TRTMC_SAM2_VIDEO_STATUS_OK,
                 "sam2 accepts the binary device prompt mask");
    check(materialize_calls == 0 && !prompt_owner_probe->expired(),
          "sam2 prompt validation neither copies nor releases its device mask");

    TrtmcSam2VideoFrameResultV1 prompt{};
    check_status(trtmc_sam2_video_get_frame_result_v1(handle.get(), 0, TRTMC_SAM2_VIDEO_GET_DEFAULT,
                                                      &prompt, sizeof(prompt)),
                 TRTMC_SAM2_VIDEO_STATUS_OK, "sam2 exposes a zero-copy device mask view");
    check(prompt.mask_memory_kind == TRTMC_SAM2_VIDEO_MASK_MEMORY_CUDA_DEVICE &&
              prompt.mask_device_ordinal == 0 && prompt.mask != nullptr && materialize_calls == 0,
          "sam2 default query preserves CUDA mask residency");

    uint64_t frames = 0;
    check_status(trtmc_sam2_video_propagate_v1(handle.get(), &frames), TRTMC_SAM2_VIDEO_STATUS_OK,
                 "sam2 propagation validates device metadata without D2H masks");
    check(materialize_calls == 0 && !prompt_owner_probe->expired(),
          "sam2 propagation stays copy-free and keeps prompt storage alive");

    TrtmcSam2VideoFrameResultV1 host_view{};
    check_status(trtmc_sam2_video_get_frame_result_v1(handle.get(), 4,
                                                      TRTMC_SAM2_VIDEO_GET_MATERIALIZE_MASK_HOST,
                                                      &host_view, sizeof(host_view)),
                 TRTMC_SAM2_VIDEO_STATUS_OK,
                 "sam2 accuracy query lazily materializes one propagated mask");
    check(host_view.mask_memory_kind == TRTMC_SAM2_VIDEO_MASK_MEMORY_HOST &&
              host_view.mask_device_ordinal == -1 &&
              static_cast<const uint8_t*>(host_view.mask)[0] == 1 && materialize_calls == 1,
          "sam2 host accuracy view is binary and materialized once");
    check_status(trtmc_sam2_video_get_frame_result_v1(handle.get(), 4,
                                                      TRTMC_SAM2_VIDEO_GET_MATERIALIZE_MASK_HOST,
                                                      &host_view, sizeof(host_view)),
                 TRTMC_SAM2_VIDEO_STATUS_OK, "sam2 reuses its cached host accuracy view");
    check(materialize_calls == 1, "sam2 does not repeat a device-to-host mask copy");

    TrtmcSam2VideoFrameResultV1 device_view{};
    check_status(trtmc_sam2_video_get_frame_result_v1(handle.get(), 4, TRTMC_SAM2_VIDEO_GET_DEFAULT,
                                                      &device_view, sizeof(device_view)),
                 TRTMC_SAM2_VIDEO_STATUS_OK,
                 "sam2 default view remains device-resident after materialization");
    check(device_view.mask_memory_kind == TRTMC_SAM2_VIDEO_MASK_MEMORY_CUDA_DEVICE,
          "sam2 materialization does not alter default result residency");

    check_status(trtmc_sam2_video_reset_v1(handle.get()), TRTMC_SAM2_VIDEO_STATUS_OK,
                 "sam2 resets a device-backed session");
    check(prompt_owner_probe->expired(), "sam2 reset releases session-owned device mask storage");
}

void test_lazy_materialization_and_processor_failures_poison() {
    {
        int calls = 0;
        auto probe = std::make_shared<std::weak_ptr<std::vector<uint8_t>>>();
        auto processor = make_host_processor();
        processor.run_bbox_prompt = [probe, &calls](const trtmc::Sam2VideoFrames& frames) {
            auto result = make_host_prompt_result(frames);
            result.frame_zero = make_device_frame_result(frames.front(), &calls, probe);
            const auto bytes = static_cast<std::size_t>(frames.front().height) *
                               static_cast<std::size_t>(frames.front().width);
            auto storage = probe->lock();
            std::shared_ptr<const void> owner(storage, storage->data());
            result.frame_zero.mask = trtmc::Sam2VideoMaskBuffer::cuda_device_binary(
                storage->data(), bytes, 0, std::move(owner), [&calls, bytes] {
                    ++calls;
                    std::vector<uint8_t> invalid(bytes, uint8_t{0});
                    invalid.back() = 2;
                    return invalid;
                });
            return result;
        };
        auto handle = make_handle(std::move(processor));
        auto storage = make_frame_storage();
        append_five_frames(handle.get(), storage);
        check_status(trtmc_sam2_video_run_bbox_prompt_v1(handle.get()), TRTMC_SAM2_VIDEO_STATUS_OK,
                     "sam2 accepts a device mask without eager host validation");
        TrtmcSam2VideoFrameResultV1 view{};
        check_status(
            trtmc_sam2_video_get_frame_result_v1(
                handle.get(), 0, TRTMC_SAM2_VIDEO_GET_MATERIALIZE_MASK_HOST, &view, sizeof(view)),
            TRTMC_SAM2_VIDEO_STATUS_INVALID_RESULT,
            "sam2 validates binary bytes when host materialization is requested");
        check(calls == 1, "sam2 runs an invalid materializer exactly once");
        check_status(trtmc_sam2_video_get_frame_result_v1(
                         handle.get(), 0, TRTMC_SAM2_VIDEO_GET_DEFAULT, &view, sizeof(view)),
                     TRTMC_SAM2_VIDEO_STATUS_INVALID_STATE,
                     "sam2 poisons the session after invalid materialized output");
    }

    {
        auto processor = make_host_processor();
        processor.run_bbox_prompt = [](const trtmc::Sam2VideoFrames&) {
            throw std::runtime_error("synthetic prompt failure");
            return trtmc::Sam2VideoPromptResult{};
        };
        auto handle = make_handle(std::move(processor));
        auto storage = make_frame_storage();
        append_five_frames(handle.get(), storage);
        check_status(trtmc_sam2_video_run_bbox_prompt_v1(handle.get()),
                     TRTMC_SAM2_VIDEO_STATUS_PROCESSOR_ERROR,
                     "sam2 translates prompt processor exceptions across the C ABI");
        check_error_contains("synthetic prompt failure", "sam2 preserves prompt failure context");
        uint64_t count = 0;
        check_status(trtmc_sam2_video_result_count_v1(handle.get(), &count),
                     TRTMC_SAM2_VIDEO_STATUS_INVALID_STATE,
                     "sam2 poisons the session after prompt processor failure");
    }

    {
        auto processor = make_host_processor();
        processor.propagate = [](const trtmc::Sam2VideoPromptResult&,
                                 const trtmc::Sam2VideoFrames&) {
            throw std::bad_alloc();
            return trtmc::Sam2VideoFrameResults{};
        };
        auto handle = make_handle(std::move(processor));
        auto storage = make_frame_storage();
        append_five_frames(handle.get(), storage);
        check_status(trtmc_sam2_video_run_bbox_prompt_v1(handle.get()), TRTMC_SAM2_VIDEO_STATUS_OK,
                     "sam2 allocation-failure fixture prompts");
        uint64_t frames = 0;
        check_status(trtmc_sam2_video_propagate_v1(handle.get(), &frames),
                     TRTMC_SAM2_VIDEO_STATUS_INTERNAL_ERROR,
                     "sam2 reports allocation failure without allocating its C diagnostic");
        check_error_contains("allocation failed", "sam2 exposes bounded allocation-failure text");
        check_status(trtmc_sam2_video_propagate_v1(handle.get(), &frames),
                     TRTMC_SAM2_VIDEO_STATUS_INVALID_STATE,
                     "sam2 poisons the session after propagation allocation failure");
    }
}

void test_result_bounds_and_bundle_factory_fail_closed() {
    auto handle = make_handle();
    auto storage = make_frame_storage();
    append_five_frames(handle.get(), storage);
    check_status(trtmc_sam2_video_run_bbox_prompt_v1(handle.get()), TRTMC_SAM2_VIDEO_STATUS_OK,
                 "sam2 bounds fixture prompts");
    uint64_t frames = 0;
    check_status(trtmc_sam2_video_propagate_v1(handle.get(), &frames), TRTMC_SAM2_VIDEO_STATUS_OK,
                 "sam2 bounds fixture propagates");
    TrtmcSam2VideoFrameResultV1 result{};
    check_status(trtmc_sam2_video_get_frame_result_v1(handle.get(), 5, TRTMC_SAM2_VIDEO_GET_DEFAULT,
                                                      &result, sizeof(result)),
                 TRTMC_SAM2_VIDEO_STATUS_OUT_OF_RANGE,
                 "sam2 rejects a result beyond its exact five-frame graph");

    check(trtmc_sam2_video_create_from_bundle_v1(nullptr, "plugins", "backends") == nullptr,
          "sam2 bundle factory rejects missing paths");
    check_error_contains("paths are required", "sam2 bundle factory reports invalid paths");
    check(trtmc_sam2_video_create_from_bundle_v1("model.bundle", "plugins", "backends") == nullptr,
          "sam2 legacy bundle factory stays fail-closed");
    check_error_contains("qualified constructor",
                         "sam2 legacy bundle factory names the qualified admission path");
    check_error_contains("explicit qualification record",
                         "sam2 legacy bundle factory reports its missing admission input");
}

} // namespace

int main() {
    test_exact_contract_track_and_frame_lifetime();
    test_processor_owned_state_reset_and_failure();
    test_state_and_tightly_packed_input_validation();
    test_strict_selected_track_and_prompt_mask_validation();
    test_propagated_results_are_exact_contiguous_masks();
    test_device_masks_stay_lazy_and_session_owned();
    test_lazy_materialization_and_processor_failures_poison();
    test_result_bounds_and_bundle_factory_fail_closed();
    std::cout << "SAM2 exact five-frame, one-object video session contract tests passed\n";
    return 0;
}
