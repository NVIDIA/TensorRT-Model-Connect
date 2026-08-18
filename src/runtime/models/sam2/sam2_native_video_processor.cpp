/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_native_video_processor.h"

#include "runtime/models/sam2/sam2_bbox_postprocess.h"
#include "runtime/models/sam2/sam2_device_workspace.h"
#include "runtime/models/sam2/sam2_engine_contract.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trtmc::sam2 {

namespace {

using ContractList = std::vector<TensorContract>;

DType runtimeDtype(TensorDataType data_type) {
    if (data_type == TensorDataType::kFloat32)
        return DType::kFloat32;
    if (data_type == TensorDataType::kBFloat16)
        return DType::kBFloat16;
    throw std::logic_error("SAM2 engine contract contains an unknown data type");
}

std::vector<int64_t> runtimeShape(const TensorContract& contract) {
    return std::vector<int64_t>(contract.dimensions.begin(),
                                contract.dimensions.begin() + contract.rank);
}

ContractList imageOutputs() {
    ContractList result(kTrackerFpn.begin(), kTrackerFpn.end());
    result.insert(result.end(), kBboxMaps.begin(), kBboxMaps.end());
    return result;
}

ContractList promptInputs() {
    ContractList result(kTrackerFpn.begin(), kTrackerFpn.end());
    result.push_back(kBoxPrompt);
    return result;
}

ContractList trackerOutputs() {
    return {kMaskLogits256, kObjectPointer, kMemoryFeatures};
}

ContractList recurrentInputs(std::int32_t history_frames) {
    ContractList result(kTrackerFpn.begin(), kTrackerFpn.end());
    result.push_back(historyMemoryFeatures(history_frames));
    result.push_back(historyObjectPointers(history_frames));
    return result;
}

std::string moduleMessage(std::string_view label, std::string_view detail) {
    return "SAM2 " + std::string(label) + " module " + std::string(detail);
}

void validateDirection(const ITrtModule& module, const ContractList& contracts, bool input,
                       std::string_view label) {
    const auto metadata = input ? module.input_info() : module.output_info();
    if (metadata.size() != contracts.size())
        throw std::invalid_argument(
            moduleMessage(label, input ? "input count drifted" : "output count drifted"));
    for (const auto& contract : contracts) {
        const auto matches = std::count_if(metadata.begin(), metadata.end(), [&](const auto& info) {
            return info.name == contract.name && info.is_input == input &&
                   info.shape == runtimeShape(contract) &&
                   info.dtype == runtimeDtype(contract.data_type);
        });
        if (matches != 1)
            throw std::invalid_argument(
                moduleMessage(label, "tensor contract drifted for " + std::string(contract.name)));
    }
}

void validateModule(const ITrtModule* module, const ContractList& inputs,
                    const ContractList& outputs, std::string_view label) {
    if (module == nullptr || !module->ok())
        throw std::invalid_argument(moduleMessage(label, "is missing or not ready"));
    if (module->optimization_profile_count() != 1 || module->profile_idx() != 0)
        throw std::invalid_argument(moduleMessage(label, "requires one static profile at index 0"));
    for (const auto& input : inputs) {
        if (module->input_is_dynamic(std::string(input.name)))
            throw std::invalid_argument(moduleMessage(label, "contains a dynamic input"));
    }
    validateDirection(*module, inputs, true, label);
    validateDirection(*module, outputs, false, label);
}

void bindDeviceTensor(ITrtModule& module, const TensorContract& contract, void* address,
                      std::string_view label) {
    if (address == nullptr)
        throw std::invalid_argument(moduleMessage(label, "cannot bind a null device tensor"));
    const std::string name(contract.name);
    module.bind_external(name, address);
    if (module.device_ptr(name) != address)
        throw std::runtime_error(moduleMessage(label, "rejected external binding for " + name));
}

cudaStream_t requireSharedStream(const NativeVideoEngineSet& engines) {
    if (engines.image == nullptr || engines.prompt == nullptr)
        throw std::invalid_argument("SAM2 image or prompt module is missing");
    const auto stream = engines.image->stream();
    if (stream == nullptr || engines.prompt->stream() != stream)
        throw std::invalid_argument("SAM2 modules require one non-null shared CUDA stream");
    for (const auto& recurrent : engines.recurrent) {
        if (recurrent == nullptr || recurrent->stream() != stream)
            throw std::invalid_argument("SAM2 modules require one non-null shared CUDA stream");
    }
    return stream;
}

class ScopedCudaDevice final {
  public:
    explicit ScopedCudaDevice(std::int32_t desired) : desired_(desired) {
        auto status = cudaGetDevice(&previous_);
        if (status != cudaSuccess)
            throw std::runtime_error("SAM2 CUDA device query failed: " +
                                     std::string(cudaGetErrorString(status)));
        if (previous_ != desired_ && (status = cudaSetDevice(desired_)) != cudaSuccess)
            throw std::runtime_error("SAM2 CUDA device selection failed: " +
                                     std::string(cudaGetErrorString(status)));
    }

    ~ScopedCudaDevice() {
        if (previous_ != desired_)
            (void)cudaSetDevice(previous_);
    }

  private:
    std::int32_t desired_{-1};
    std::int32_t previous_{-1};
};

class WorkspaceDrainGuard final {
  public:
    explicit WorkspaceDrainGuard(Sam2DeviceWorkspace& workspace) : workspace_(&workspace) {}
    ~WorkspaceDrainGuard() {
        if (workspace_ != nullptr)
            workspace_->drainNoexcept();
    }
    void dismiss() noexcept { workspace_ = nullptr; }

  private:
    Sam2DeviceWorkspace* workspace_;
};

void destroyModuleOnStreamDevice(std::unique_ptr<ITrtModule>& module) noexcept {
    if (module == nullptr)
        return;
    try {
        std::int32_t device = -1;
        if (module->stream() != nullptr &&
            cudaStreamGetDevice(module->stream(), &device) == cudaSuccess) {
            ScopedCudaDevice selected(device);
            module.reset();
            return;
        }
        (void)cudaGetLastError();
    } catch (...) {
    }
    module.reset();
}

void destroyEngineSet(NativeVideoEngineSet& engines) noexcept {
    for (auto iterator = engines.recurrent.rbegin(); iterator != engines.recurrent.rend();
         ++iterator) {
        destroyModuleOnStreamDevice(*iterator);
    }
    destroyModuleOnStreamDevice(engines.prompt);
    // The first module owns the shared stream, so it must be released last.
    destroyModuleOnStreamDevice(engines.image);
}

} // namespace

struct NativeVideoProcessor::Impl final {
    explicit Impl(NativeVideoEngineSet engines) : engines_(std::move(engines)) {
        try {
            const auto stream = requireSharedStream(engines_);
            std::int32_t device = -1;
            const auto status = cudaStreamGetDevice(stream, &device);
            if (status != cudaSuccess)
                throw std::runtime_error("SAM2 CUDA stream-device query failed: " +
                                         std::string(cudaGetErrorString(status)));
            ScopedCudaDevice selected(device);
            validateModules();
            workspace_ = std::make_unique<Sam2DeviceWorkspace>(stream);
            bindGraph();
        } catch (...) {
            if (workspace_ != nullptr)
                workspace_->drainNoexcept();
            destroyEngineSet(engines_);
            throw;
        }
    }

    ~Impl() {
        if (workspace_ == nullptr)
            return;
        try {
            ScopedCudaDevice selected(workspace_->deviceOrdinal());
            workspace_->drainNoexcept();
            destroyEngineSet(engines_);
        } catch (...) {
            workspace_->drainNoexcept();
            destroyEngineSet(engines_);
        }
    }

    NativeVideoRunView run(const NativeRgb8Frames& frames, bool materialize_masks_host) {
        if (poisoned_)
            throw std::logic_error("SAM2 video session is poisoned; recreate it");
        try {
            for (const auto* frame : frames) {
                if (frame == nullptr)
                    throw std::invalid_argument("SAM2 RGB8 frame pointers must not be null");
            }

            ScopedCudaDevice selected(workspace_->deviceOrdinal());
            WorkspaceDrainGuard drain_guard(*workspace_);
            for (auto& mask : host_masks_)
                mask.clear();
            workspace_->beginRun();

            enqueueImageFrame(frames[0]);
            workspace_->enqueueBboxDownload(*engines_.image);
            const auto decoded = decode_sam2_bbox_outputs(workspace_->waitForBbox());
            const auto& detection = require_exactly_one_sam2_bbox_detection(decoded);

            auto model_box = detection.model_xyxy_1024;
            TensorMap prompt_inputs;
            prompt_inputs.emplace(std::string(kBoxPrompt.name),
                                  Tensor{model_box.data(), runtimeShape(kBoxPrompt),
                                         runtimeDtype(kBoxPrompt.data_type)});
            engines_.prompt->forward_async(prompt_inputs);
            workspace_->enqueueTrackerPostprocess(*engines_.prompt, 0U);
            workspace_->finishTrackerStage("prompt");

            for (std::size_t frame = 1; frame < frames.size(); ++frame) {
                enqueueImageFrame(frames[frame]);
                auto& recurrent = *engines_.recurrent[frame - 1U];
                recurrent.forward_async(TensorMap{});
                workspace_->enqueueTrackerPostprocess(recurrent, frame);
            }
            workspace_->finishTrackerStage("propagation");

            NativeVideoRunView result;
            result.label = detection.label;
            result.detector_score = detection.score;
            result.prompt_box_xyxy = detection.original_xyxy;
            result.mask_device_ordinal = materialize_masks_host ? -1 : workspace_->deviceOrdinal();
            for (std::size_t frame = 0; frame < result.masks.size(); ++frame) {
                if (materialize_masks_host) {
                    host_masks_[frame] = workspace_->materializeMask(frame);
                    result.masks[frame] = host_masks_[frame].data();
                } else {
                    result.masks[frame] = workspace_->maskPointer(frame);
                }
            }
            drain_guard.dismiss();
            return result;
        } catch (...) {
            poisoned_ = true;
            throw;
        }
    }

  private:
    void validateModules() const {
        validateModule(engines_.image.get(), {kPixelValues}, imageOutputs(), "image");
        validateModule(engines_.prompt.get(), promptInputs(), trackerOutputs(), "prompt");
        for (std::size_t index = 0; index < engines_.recurrent.size(); ++index) {
            validateModule(engines_.recurrent[index].get(),
                           recurrentInputs(static_cast<std::int32_t>(index + 1U)), trackerOutputs(),
                           "recurrent H" + std::to_string(index + 1U));
        }
    }

    std::array<ITrtModule*, kVideoFrameCount> trackers() const {
        return {engines_.prompt.get(), engines_.recurrent[0].get(), engines_.recurrent[1].get(),
                engines_.recurrent[2].get(), engines_.recurrent[3].get()};
    }

    void bindGraph() {
        bindDeviceTensor(*engines_.image, kPixelValues, workspace_->preprocessedPixelValues(),
                         "image preprocess");
        for (auto* tracker : trackers()) {
            for (const auto& fpn : kTrackerFpn)
                bindDeviceTensor(*tracker, fpn, engines_.image->device_ptr(std::string(fpn.name)),
                                 "tracker FPN");
        }

        std::array<ITrtModule*, 4> history_producers = {
            engines_.prompt.get(), engines_.recurrent[0].get(), engines_.recurrent[1].get(),
            engines_.recurrent[2].get()};
        for (std::size_t frame = 0; frame < history_producers.size(); ++frame) {
            bindDeviceTensor(*history_producers[frame], kMemoryFeatures,
                             workspace_->historyMemorySlot(frame), "history producer");
            bindDeviceTensor(*history_producers[frame], kObjectPointer,
                             workspace_->historyPointerSlot(frame), "history producer");
        }
        for (std::size_t index = 0; index < engines_.recurrent.size(); ++index) {
            const auto history = static_cast<std::int32_t>(index + 1U);
            bindDeviceTensor(*engines_.recurrent[index], historyMemoryFeatures(history),
                             workspace_->historyMemoryBase(), "recurrent history");
            bindDeviceTensor(*engines_.recurrent[index], historyObjectPointers(history),
                             workspace_->historyPointerBase(), "recurrent history");
        }
    }

    void enqueueImageFrame(const std::uint8_t* frame) {
        workspace_->enqueueRgb8Preprocess(frame, kOriginalImageHeight, kOriginalImageWidth);
        engines_.image->forward_async(TensorMap{});
    }

    // Modules release external pointers before their workspace allocation.
    std::unique_ptr<Sam2DeviceWorkspace> workspace_;
    NativeVideoEngineSet engines_;
    std::array<std::vector<std::uint8_t>, kVideoFrameCount> host_masks_;
    bool poisoned_{false};
};

NativeVideoProcessor::NativeVideoProcessor(NativeVideoEngineSet engines)
    : implementation_(std::make_unique<Impl>(std::move(engines))) {}

NativeVideoProcessor::~NativeVideoProcessor() = default;

NativeVideoRunView NativeVideoProcessor::run(const NativeRgb8Frames& frames,
                                             bool materialize_masks_host) {
    return implementation_->run(frames, materialize_masks_host);
}

} // namespace trtmc::sam2
