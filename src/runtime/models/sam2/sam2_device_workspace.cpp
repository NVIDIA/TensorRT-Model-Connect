/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_device_workspace.h"

#include "runtime/models/sam2/sam2_engine_contract.h"
#include "runtime/models/sam2/sam2_mask_postprocess_cuda.h"
#include "runtime/models/sam2/sam2_preprocess.h"
#include "runtime/models/sam2/sam2_preprocess_cuda.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::sam2 {

namespace {

constexpr std::size_t kHistoryFrames = 4U;
constexpr std::size_t kMemoryFeatureElements = 64U * 64U * 64U;
constexpr std::size_t kMemoryFeatureBytes = kMemoryFeatureElements * sizeof(std::uint16_t);
constexpr std::size_t kObjectPointerElements = 256U;
constexpr std::size_t kObjectPointerBytes = kObjectPointerElements * sizeof(float);
constexpr std::size_t kMaskBytes =
    static_cast<std::size_t>(kOriginalImageHeight) * static_cast<std::size_t>(kOriginalImageWidth);
constexpr std::size_t kRgbChannels = 3U;
constexpr std::size_t kSourceRgbBytes = static_cast<std::size_t>(kOriginalImageHeight) *
                                        static_cast<std::size_t>(kOriginalImageWidth) *
                                        kRgbChannels;
constexpr std::size_t kHorizontalRgbBytes = static_cast<std::size_t>(kOriginalImageHeight) *
                                            static_cast<std::size_t>(kPreprocessImageSize) *
                                            kRgbChannels;
constexpr std::size_t kPixelValueBytes = static_cast<std::size_t>(kPreprocessImageSize) *
                                         static_cast<std::size_t>(kPreprocessImageSize) *
                                         kRgbChannels * sizeof(float);

[[noreturn]] void failCuda(cudaError_t status, const std::string& operation) {
    throw std::runtime_error("SAM2 " + operation + " failed: " + cudaGetErrorString(status));
}

void checkCuda(cudaError_t status, const std::string& operation) {
    if (status != cudaSuccess)
        failCuda(status, operation);
}

class ScopedDevice final {
  public:
    explicit ScopedDevice(std::int32_t desired) : desired_(desired) {
        checkCuda(cudaGetDevice(&previous_), "CUDA device query");
        if (previous_ != desired_)
            checkCuda(cudaSetDevice(desired_), "CUDA device selection");
    }

    ~ScopedDevice() {
        if (previous_ != desired_)
            (void)cudaSetDevice(previous_);
    }

    ScopedDevice(const ScopedDevice&) = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;

  private:
    std::int32_t desired_{-1};
    std::int32_t previous_{-1};
};

class DeviceAllocation final {
  public:
    DeviceAllocation(std::size_t bytes, std::int32_t device) : bytes_(bytes), device_(device) {
        if (bytes_ == 0U)
            throw std::invalid_argument("SAM2 CUDA allocation size must be positive");
        ScopedDevice selected(device_);
        checkCuda(cudaMalloc(&data_, bytes_), "CUDA allocation");
    }

    ~DeviceAllocation() {
        if (data_ == nullptr)
            return;
        try {
            ScopedDevice selected(device_);
            (void)cudaFree(data_);
        } catch (...) {
        }
    }

    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;

    void* data() const noexcept { return data_; }
    std::size_t bytes() const noexcept { return bytes_; }

  private:
    void* data_{nullptr};
    std::size_t bytes_{0U};
    std::int32_t device_{-1};
};

class PinnedAllocation final {
  public:
    explicit PinnedAllocation(std::size_t bytes) {
        if (bytes == 0U)
            throw std::invalid_argument("SAM2 pinned allocation size must be positive");
        checkCuda(cudaHostAlloc(&data_, bytes, cudaHostAllocDefault), "pinned host allocation");
    }

    ~PinnedAllocation() {
        if (data_ != nullptr)
            (void)cudaFreeHost(data_);
    }

    PinnedAllocation(const PinnedAllocation&) = delete;
    PinnedAllocation& operator=(const PinnedAllocation&) = delete;

    void* data() const noexcept { return data_; }

  private:
    void* data_{nullptr};
};

std::size_t tensorBytes(const TensorContract& contract) {
    std::size_t elements = 1U;
    for (std::uint8_t index = 0; index < contract.rank; ++index) {
        const auto dimension = contract.dimensions[index];
        if (dimension <= 0)
            throw std::logic_error("SAM2 workspace contract dimension must be positive");
        const auto extent = static_cast<std::size_t>(dimension);
        if (extent > std::numeric_limits<std::size_t>::max() / elements)
            throw std::overflow_error("SAM2 workspace tensor size overflowed");
        elements *= extent;
    }
    const std::size_t element_bytes =
        contract.data_type == TensorDataType::kFloat32 ? sizeof(float) : sizeof(std::uint16_t);
    if (element_bytes > std::numeric_limits<std::size_t>::max() / elements)
        throw std::overflow_error("SAM2 workspace tensor byte count overflowed");
    return elements * element_bytes;
}

std::array<std::size_t, kBboxMaps.size()> bboxOffsets() {
    std::array<std::size_t, kBboxMaps.size()> offsets{};
    std::size_t current = 0U;
    for (std::size_t index = 0; index < offsets.size(); ++index) {
        offsets[index] = current;
        current += tensorBytes(kBboxMaps[index]);
    }
    return offsets;
}

std::size_t bboxBytes() {
    std::size_t result = 0U;
    for (const auto& contract : kBboxMaps)
        result += tensorBytes(contract);
    return result;
}

std::array<std::int64_t, 4> bboxShape(const TensorContract& contract) {
    if (contract.rank != 4U)
        throw std::logic_error("SAM2 bbox workspace requires rank-four tensors");
    return {contract.dimensions[0], contract.dimensions[1], contract.dimensions[2],
            contract.dimensions[3]};
}

} // namespace

struct Sam2DeviceWorkspace::Impl final {
    explicit Impl(cudaStream_t requested_stream)
        : stream(requireStream(requested_stream)), device(queryStreamDevice(stream)),
          horizontal_plan(makePillowBicubicAxisPlan(kOriginalImageWidth, kPreprocessImageSize)),
          vertical_plan(makePillowBicubicAxisPlan(kOriginalImageHeight, kPreprocessImageSize)),
          source_rgb(kSourceRgbBytes, device), horizontal_rgb(kHorizontalRgbBytes, device),
          pixel_values(kPixelValueBytes, device),
          horizontal_spans(horizontal_plan.spans.size() * sizeof(PillowResizeSpan), device),
          horizontal_weights(horizontal_plan.weights.size() * sizeof(std::int32_t), device),
          vertical_spans(vertical_plan.spans.size() * sizeof(PillowResizeSpan), device),
          vertical_weights(vertical_plan.weights.size() * sizeof(std::int32_t), device),
          normalization_table(kSam2Rgb8NormalizationTableElements * sizeof(float), device),
          history_memory(kHistoryFrames * kMemoryFeatureBytes, device),
          history_pointers(kHistoryFrames * kObjectPointerBytes, device),
          masks(static_cast<std::size_t>(kFrameCount) * kMaskBytes, device),
          status(sizeof(std::uint32_t), device), bbox_staging(bboxBytes()),
          status_staging(sizeof(std::uint32_t)), bbox_offsets(bboxOffsets()) {
        copyPlanToDevice(horizontal_plan, horizontal_spans, horizontal_weights, "horizontal");
        copyPlanToDevice(vertical_plan, vertical_spans, vertical_weights, "vertical");
        const auto& normalization = sam2Rgb8NormalizationTable();
        checkCuda(cudaMemcpy(normalization_table.data(), normalization.data(),
                             normalization_table.bytes(), cudaMemcpyHostToDevice),
                  "RGB8 normalization-table upload");
    }

    static cudaStream_t requireStream(cudaStream_t requested_stream) {
        if (requested_stream == nullptr)
            throw std::invalid_argument("SAM2 device workspace requires a non-null CUDA stream");
        return requested_stream;
    }

    static std::int32_t queryStreamDevice(cudaStream_t requested_stream) {
        std::int32_t result = -1;
        checkCuda(cudaStreamGetDevice(requested_stream, &result), "CUDA stream-device query");
        return result;
    }

    static void copyPlanToDevice(const PillowResizeAxisPlan& plan,
                                 const DeviceAllocation& device_spans,
                                 const DeviceAllocation& device_weights, const char* axis) {
        if (plan.spans.empty() || plan.weights.empty())
            throw std::logic_error(std::string("SAM2 ") + axis + " resize plan is empty");
        checkCuda(cudaMemcpy(device_spans.data(), plan.spans.data(), device_spans.bytes(),
                             cudaMemcpyHostToDevice),
                  std::string(axis) + " resize-span upload");
        checkCuda(cudaMemcpy(device_weights.data(), plan.weights.data(), device_weights.bytes(),
                             cudaMemcpyHostToDevice),
                  std::string(axis) + " resize-weight upload");
    }

    cudaStream_t stream{nullptr};
    std::int32_t device{-1};
    PillowResizeAxisPlan horizontal_plan;
    PillowResizeAxisPlan vertical_plan;
    DeviceAllocation source_rgb;
    DeviceAllocation horizontal_rgb;
    DeviceAllocation pixel_values;
    DeviceAllocation horizontal_spans;
    DeviceAllocation horizontal_weights;
    DeviceAllocation vertical_spans;
    DeviceAllocation vertical_weights;
    DeviceAllocation normalization_table;
    DeviceAllocation history_memory;
    DeviceAllocation history_pointers;
    DeviceAllocation masks;
    DeviceAllocation status;
    PinnedAllocation bbox_staging;
    PinnedAllocation status_staging;
    std::array<std::size_t, kBboxMaps.size()> bbox_offsets{};
};

Sam2DeviceWorkspace::Sam2DeviceWorkspace(cudaStream_t stream)
    : implementation_(std::make_unique<Impl>(stream)) {}

Sam2DeviceWorkspace::~Sam2DeviceWorkspace() = default;

std::int32_t Sam2DeviceWorkspace::deviceOrdinal() const noexcept {
    return implementation_->device;
}

void* Sam2DeviceWorkspace::historyMemoryBase() const noexcept {
    return implementation_->history_memory.data();
}

void* Sam2DeviceWorkspace::historyPointerBase() const noexcept {
    return implementation_->history_pointers.data();
}

void* Sam2DeviceWorkspace::historyMemorySlot(std::size_t frame_index) const {
    if (frame_index >= kHistoryFrames)
        throw std::out_of_range("SAM2 history-memory slot is out of range");
    auto* base = static_cast<std::uint8_t*>(implementation_->history_memory.data());
    return base + frame_index * kMemoryFeatureBytes;
}

void* Sam2DeviceWorkspace::historyPointerSlot(std::size_t frame_index) const {
    if (frame_index >= kHistoryFrames)
        throw std::out_of_range("SAM2 history-pointer slot is out of range");
    auto* base = static_cast<std::uint8_t*>(implementation_->history_pointers.data());
    return base + frame_index * kObjectPointerBytes;
}

void* Sam2DeviceWorkspace::preprocessedPixelValues() const noexcept {
    return implementation_->pixel_values.data();
}

void Sam2DeviceWorkspace::enqueueRgb8Preprocess(const std::uint8_t* rgb_hwc, std::int32_t height,
                                                std::int32_t width) {
    if (rgb_hwc == nullptr)
        throw std::invalid_argument("SAM2 RGB8 preprocess source must not be null");
    if (height != kOriginalImageHeight || width != kOriginalImageWidth) {
        throw std::invalid_argument(
            "SAM2 RGB8 CUDA preprocess requires the fixed 1280x1088 frame contract");
    }

    ScopedDevice selected(implementation_->device);
    checkCuda(cudaMemcpyAsync(implementation_->source_rgb.data(), rgb_hwc, kSourceRgbBytes,
                              cudaMemcpyHostToDevice, implementation_->stream),
              "RGB8 source upload");
    checkCuda(enqueueSam2PillowRgb8Preprocess(
                  static_cast<const std::uint8_t*>(implementation_->source_rgb.data()),
                  static_cast<std::uint8_t*>(implementation_->horizontal_rgb.data()),
                  static_cast<float*>(implementation_->pixel_values.data()),
                  static_cast<const PillowResizeSpan*>(implementation_->horizontal_spans.data()),
                  static_cast<const std::int32_t*>(implementation_->horizontal_weights.data()),
                  static_cast<const PillowResizeSpan*>(implementation_->vertical_spans.data()),
                  static_cast<const std::int32_t*>(implementation_->vertical_weights.data()),
                  static_cast<const float*>(implementation_->normalization_table.data()),
                  implementation_->stream),
              "RGB8 Pillow preprocess launch");
}

void Sam2DeviceWorkspace::beginRun() {
    ScopedDevice selected(implementation_->device);
    *static_cast<std::uint32_t*>(implementation_->status_staging.data()) = UINT32_MAX;
    checkCuda(cudaMemsetAsync(implementation_->status.data(), 0, sizeof(std::uint32_t),
                              implementation_->stream),
              "device status reset");
}

void Sam2DeviceWorkspace::enqueueBboxDownload(const ITrtModule& image) {
    ScopedDevice selected(implementation_->device);
    auto* staging = static_cast<std::uint8_t*>(implementation_->bbox_staging.data());
    for (std::size_t index = 0; index < kBboxMaps.size(); ++index) {
        const std::string name(kBboxMaps[index].name);
        const void* source = image.device_ptr(name);
        if (source == nullptr)
            throw std::runtime_error("SAM2 image bbox output has no device storage: " + name);
        checkCuda(cudaMemcpyAsync(staging + implementation_->bbox_offsets[index], source,
                                  tensorBytes(kBboxMaps[index]), cudaMemcpyDeviceToHost,
                                  implementation_->stream),
                  "bbox output download for " + name);
    }
}

Sam2BBoxRawOutputs Sam2DeviceWorkspace::waitForBbox() {
    ScopedDevice selected(implementation_->device);
    checkCuda(cudaStreamSynchronize(implementation_->stream), "bbox completion wait");
    const auto* staging = static_cast<const std::uint8_t*>(implementation_->bbox_staging.data());
    std::array<Sam2BBoxTensorView, kBboxMaps.size()> views{};
    for (std::size_t index = 0; index < views.size(); ++index) {
        views[index] = {
            reinterpret_cast<const std::uint16_t*>(staging + implementation_->bbox_offsets[index]),
            bboxShape(kBboxMaps[index]), tensorBytes(kBboxMaps[index]) / sizeof(std::uint16_t)};
    }
    return {views[0], views[1], views[2], views[3], views[4], views[5]};
}

void Sam2DeviceWorkspace::enqueueTrackerPostprocess(const ITrtModule& tracker,
                                                    std::size_t frame_index) {
    if (frame_index >= static_cast<std::size_t>(kFrameCount))
        throw std::out_of_range("SAM2 mask frame index is out of range");
    const auto* mask_logits =
        static_cast<const float*>(tracker.device_ptr(std::string(kMaskLogits256.name)));
    const auto* object_pointer =
        static_cast<const float*>(tracker.device_ptr(std::string(kObjectPointer.name)));
    const auto* memory_features =
        static_cast<const std::uint16_t*>(tracker.device_ptr(std::string(kMemoryFeatures.name)));
    auto* mask =
        static_cast<std::uint8_t*>(implementation_->masks.data()) + frame_index * kMaskBytes;

    ScopedDevice selected(implementation_->device);
    checkCuda(enqueueSam2ValidateAndResizeMask(
                  mask_logits, object_pointer, memory_features, mask,
                  static_cast<std::uint32_t*>(implementation_->status.data()),
                  implementation_->stream),
              "tracker device validation and mask postprocess launch");
}

void Sam2DeviceWorkspace::finishTrackerStage(const char* stage) {
    const std::string label = stage == nullptr ? "tracker" : stage;
    ScopedDevice selected(implementation_->device);
    checkCuda(cudaMemcpyAsync(implementation_->status_staging.data(),
                              implementation_->status.data(), sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost, implementation_->stream),
              label + " device status download");
    checkCuda(cudaStreamSynchronize(implementation_->stream), label + " completion wait");
    const auto status = *static_cast<const std::uint32_t*>(implementation_->status_staging.data());
    if (status != 0U) {
        throw std::runtime_error("SAM2 " + label + " produced non-finite tracker output; status=" +
                                 std::to_string(status));
    }
}

void Sam2DeviceWorkspace::drainNoexcept() noexcept {
    try {
        ScopedDevice selected(implementation_->device);
        (void)cudaStreamSynchronize(implementation_->stream);
    } catch (...) {
    }
}

const void* Sam2DeviceWorkspace::maskPointer(std::size_t frame_index) const {
    if (frame_index >= static_cast<std::size_t>(kFrameCount))
        throw std::out_of_range("SAM2 mask frame index is out of range");
    const auto* base = static_cast<const std::uint8_t*>(implementation_->masks.data());
    return base + frame_index * kMaskBytes;
}

std::vector<std::uint8_t> Sam2DeviceWorkspace::materializeMask(std::size_t frame_index) const {
    const void* source = maskPointer(frame_index);
    std::vector<std::uint8_t> result(kMaskBytes);
    ScopedDevice selected(implementation_->device);
    checkCuda(cudaMemcpy(result.data(), source, result.size(), cudaMemcpyDeviceToHost),
              "host mask materialization");
    return result;
}

} // namespace trtmc::sam2
