/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/sam2/runtime/sam2_pipeline.h"

#include "families/sam2/runtime/sam2_engine_contract.h"

#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace trtmc::sam2 {
namespace {

std::vector<char> requirePlan(const BundleReader& bundle, std::string_view name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::invalid_argument("SAM2 bundle is missing plan section: " + std::string(name));
    return bundle.read_section(name);
}

void validateBundleInventory(const BundleReader& bundle) {
    for (const auto section : kRequiredPlanSections)
        (void)requirePlan(bundle, section);
    const auto* config = bundle.find_section(kConfigSection);
    if (config == nullptr || config->length == 0)
        throw std::invalid_argument("SAM2 bundle is missing runtime.json");
}

constexpr std::size_t kRgbElementCount = static_cast<std::size_t>(kOriginalImageHeight) *
                                         static_cast<std::size_t>(kOriginalImageWidth) * 3U;

void requireFrameShape(const VideoFrameView& frame) {
    if (frame.pixels == nullptr || frame.height != kOriginalImageHeight ||
        frame.width != kOriginalImageWidth || frame.element_count != kRgbElementCount) {
        throw std::invalid_argument("SAM2 frame does not match its fixed RGB contract");
    }
}

using ConvertedRgb8Frames = std::array<std::vector<std::uint8_t>, kVideoFrameCount>;

NativeRgb8Frames prepareRgb8Frames(const VideoSegmentationRequest& request,
                                   ConvertedRgb8Frames& converted_frames) {
    if (!request.text_prompt.empty())
        throw std::invalid_argument("SAM2 bbox tracking does not accept a text prompt");
    if (request.frames.size() != kVideoFrameCount)
        throw std::invalid_argument("SAM2 requires exactly five frames");

    NativeRgb8Frames frames{};
    for (std::size_t index = 0; index < request.frames.size(); ++index) {
        const auto& frame = request.frames[index];
        requireFrameShape(frame);
        if (frame.format == VideoFrameFormat::kRgb8) {
            frames[index] = static_cast<const std::uint8_t*>(frame.pixels);
        } else if (frame.format == VideoFrameFormat::kRgbFloat32) {
            converted_frames[index] = convertSam2FloatFrameToRgb8(frame);
            frames[index] = converted_frames[index].data();
        } else {
            throw std::invalid_argument("SAM2 frame has an unsupported RGB format");
        }
    }
    return frames;
}

class Session final : public IVideoSegmentationSession, public ISam2DeviceMaskSession {
  public:
    explicit Session(std::unique_ptr<NativeVideoProcessor> processor)
        : processor_(std::move(processor)) {}

    VideoSegmentationResult segment(const VideoSegmentationRequest& request) override {
        ConvertedRgb8Frames converted_frames;
        const auto frames = prepareRgb8Frames(request, converted_frames);
        const auto view = processor_->run(frames, true);
        const auto mask_size = static_cast<std::size_t>(kOriginalImageHeight) *
                               static_cast<std::size_t>(kOriginalImageWidth);
        VideoSegmentationResult result;
        result.frames.resize(kVideoFrameCount);
        for (std::size_t index = 0; index < result.frames.size(); ++index) {
            auto& frame = result.frames[index];
            frame.masks.resize(mask_size);
            std::memcpy(frame.masks.data(), view.masks[index], mask_size);
            frame.object_ids = {view.label};
            frame.detection_scores = {view.detector_score};
            frame.boxes.assign(view.prompt_box_xyxy.begin(), view.prompt_box_xyxy.end());
            frame.num_objects = 1;
            frame.height = kOriginalImageHeight;
            frame.width = kOriginalImageWidth;
        }
        return result;
    }

    Sam2DeviceMaskResultView segment_device(const VideoSegmentationRequest& request) override {
        ConvertedRgb8Frames converted_frames;
        const auto frames = prepareRgb8Frames(request, converted_frames);
        const auto native = processor_->run(frames, false);
        if (native.mask_device_ordinal < 0 ||
            std::any_of(
                native.masks.begin(), native.masks.end(),
                [](const void* mask) { return mask == nullptr; })) {
            throw std::runtime_error("SAM2 did not return five CUDA device masks");
        }
        Sam2DeviceMaskResultView result;
        result.masks = native.masks;
        result.mask_device_ordinal = native.mask_device_ordinal;
        result.label = native.label;
        result.detector_score = native.detector_score;
        result.prompt_box_xyxy = native.prompt_box_xyxy;
        return result;
    }

  private:
    std::unique_ptr<NativeVideoProcessor> processor_;
};

} // namespace

std::vector<std::uint8_t> convertSam2FloatFrameToRgb8(const VideoFrameView& frame) {
    requireFrameShape(frame);
    if (frame.format != VideoFrameFormat::kRgbFloat32)
        throw std::invalid_argument("SAM2 float conversion requires RGB float32 input");

    const auto* source = static_cast<const float*>(frame.pixels);
    std::vector<std::uint8_t> result(kRgbElementCount);
    for (std::size_t index = 0; index < result.size(); ++index) {
        const float value = source[index];
        if (!std::isfinite(value) || value < 0.0F || value > 1.0F) {
            throw std::invalid_argument(
                "SAM2 RGB float32 input must contain finite values in [0, 1]");
        }
        result[index] = static_cast<std::uint8_t>(std::lround(value * 255.0F));
    }
    return result;
}

NativeVideoEngineSet makeNativeVideoEngineSet(const BundleReader& bundle,
                                              const NativePlanModuleFactory& module_factory) {
    if (!module_factory)
        throw std::invalid_argument("SAM2 requires a module factory");
    validateBundleInventory(bundle);
    std::array<std::unique_ptr<ITrtModule>, 6> modules;
    for (std::size_t index = 0; index < modules.size(); ++index) {
        const auto& plan = requirePlan(bundle, kRequiredPlanSections[index]);
        modules[index] = module_factory(kRequiredPlanSections[index], plan.data(), plan.size());
        if (!modules[index] || !modules[index]->ok())
            throw std::runtime_error("SAM2 failed to create module: " +
                                     std::string(kRequiredPlanSections[index]));
    }
    NativeVideoEngineSet engines;
    engines.image = std::move(modules[0]);
    engines.prompt = std::move(modules[1]);
    for (std::size_t index = 0; index < engines.recurrent.size(); ++index)
        engines.recurrent[index] = std::move(modules[index + 2]);
    return engines;
}

Sam2Pipeline::Sam2Pipeline(std::unique_ptr<NativeVideoProcessor> processor)
    : processor_(std::move(processor)) {
    if (!processor_)
        throw std::invalid_argument("SAM2 pipeline requires a native video processor");
}

std::unique_ptr<IVideoSegmentationSession> Sam2Pipeline::create_video_segmentation_session() {
    if (!processor_)
        throw std::logic_error("SAM2 pipeline already created its video session");
    return std::make_unique<Session>(std::move(processor_));
}

} // namespace trtmc::sam2
