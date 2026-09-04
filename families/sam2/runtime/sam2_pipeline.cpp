/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/sam2/runtime/sam2_pipeline.h"

#include "families/sam2/runtime/sam2_engine_contract.h"

#include <array>
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

class Session final : public IVideoSegmentationSession {
  public:
    explicit Session(std::unique_ptr<NativeVideoProcessor> processor)
        : processor_(std::move(processor)) {}

    VideoSegmentationResult segment(const VideoSegmentationRequest& request) override {
        if (!request.text_prompt.empty())
            throw std::invalid_argument("SAM2 bbox tracking does not accept a text prompt");
        if (request.frames.size() != kVideoFrameCount)
            throw std::invalid_argument("SAM2 requires exactly five frames");

        NativeRgb8Frames frames{};
        const auto element_count = static_cast<std::size_t>(kOriginalImageHeight) *
                                   static_cast<std::size_t>(kOriginalImageWidth) * 3U;
        for (std::size_t index = 0; index < request.frames.size(); ++index) {
            const auto& frame = request.frames[index];
            if (frame.pixels == nullptr || frame.format != VideoFrameFormat::kRgb8 ||
                frame.height != kOriginalImageHeight || frame.width != kOriginalImageWidth ||
                frame.element_count != element_count) {
                throw std::invalid_argument("SAM2 frame does not match its fixed RGB8 contract");
            }
            frames[index] = static_cast<const std::uint8_t*>(frame.pixels);
        }

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

  private:
    std::unique_ptr<NativeVideoProcessor> processor_;
};

} // namespace

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
