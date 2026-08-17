/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_native_video_processor.h"

#include "runtime/models/sam2/sam2_bbox_postprocess.h"
#include "runtime/models/sam2/sam2_device_workspace.h"
#include "runtime/models/sam2/sam2_engine_contract.h"
#include "runtime/models/sam2/sam2_mask_postprocess.h"
#include "runtime/models/sam2/sam2_preprocess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trtmc::sam2 {

namespace {

using ContractList = std::vector<TensorContract>;

constexpr std::size_t kMaskLogitElements = 256U * 256U;
constexpr std::size_t kMemoryFeatureElements = 64U * 64U * 64U;
constexpr std::size_t kObjectPointerElements = 256U;

DType runtimeDtype(TensorDataType data_type) {
    switch (data_type) {
    case TensorDataType::kFloat32:
        return DType::kFloat32;
    case TensorDataType::kBFloat16:
        return DType::kBFloat16;
    }
    throw std::logic_error("SAM2 engine contract contains an unknown data type");
}

std::vector<int64_t> runtimeShape(const TensorContract& contract) {
    std::vector<int64_t> shape;
    shape.reserve(contract.rank);
    for (std::uint8_t index = 0; index < contract.rank; ++index)
        shape.push_back(contract.dimensions[index]);
    return shape;
}

std::size_t checkedElementCount(const TensorContract& contract) {
    std::size_t count = 1;
    for (const auto dimension : runtimeShape(contract)) {
        if (dimension <= 0)
            throw std::logic_error("SAM2 engine contract contains a non-positive dimension");
        const auto value = static_cast<std::size_t>(dimension);
        if (value > std::numeric_limits<std::size_t>::max() / count)
            throw std::overflow_error("SAM2 engine contract element count overflows");
        count *= value;
    }
    return count;
}

std::size_t checkedByteCount(const TensorContract& contract) {
    const auto elements = checkedElementCount(contract);
    const auto element_bytes = dtype_size(runtimeDtype(contract.data_type));
    if (element_bytes > std::numeric_limits<std::size_t>::max() / elements)
        throw std::overflow_error("SAM2 engine contract byte count overflows");
    return elements * element_bytes;
}

ContractList imageInputs() {
    return {kPixelValues};
}

ContractList imageOutputs() {
    ContractList result;
    result.reserve(kTrackerFpn.size() + kBboxMaps.size());
    result.insert(result.end(), kTrackerFpn.begin(), kTrackerFpn.end());
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

const TensorContract* findContract(const ContractList& contracts, const std::string& name) {
    const auto found = std::find_if(contracts.begin(), contracts.end(),
                                    [&](const auto& contract) { return contract.name == name; });
    return found == contracts.end() ? nullptr : &*found;
}

void validateDirection(const ITrtModule& module, const ContractList& contracts, bool input,
                       std::string_view label) {
    const auto metadata = input ? module.input_info() : module.output_info();
    if (metadata.size() != contracts.size())
        throw std::invalid_argument(
            moduleMessage(label, input ? "input count drifted" : "output count drifted"));

    for (const auto& info : metadata) {
        const auto* contract = findContract(contracts, info.name);
        if (contract == nullptr)
            throw std::invalid_argument(
                moduleMessage(label, input ? "has an unknown input" : "has an unknown output"));
        if (info.is_input != input || info.shape != runtimeShape(*contract) ||
            info.dtype != runtimeDtype(contract->data_type)) {
            throw std::invalid_argument(
                moduleMessage(label, std::string("metadata drifted for ") + info.name));
        }
        const auto duplicate_count =
            std::count_if(metadata.begin(), metadata.end(),
                          [&](const auto& candidate) { return candidate.name == info.name; });
        if (duplicate_count != 1)
            throw std::invalid_argument(moduleMessage(label, "contains duplicate tensor names"));
    }

    for (const auto& contract : contracts) {
        const std::string name(contract.name);
        const bool has_direction = input ? module.has_input(name) : module.has_output(name);
        const bool has_other_direction = input ? module.has_output(name) : module.has_input(name);
        if (!has_direction || has_other_direction ||
            module.tensor_shape(name) != runtimeShape(contract) ||
            module.tensor_dtype(name) != runtimeDtype(contract.data_type)) {
            throw std::invalid_argument(
                moduleMessage(label, std::string("binding drifted for ") + name));
        }
    }
}

void validateModule(const ITrtModule* module, const ContractList& inputs,
                    const ContractList& outputs, std::string_view label) {
    if (module == nullptr)
        throw std::invalid_argument(moduleMessage(label, "is missing"));
    if (!module->ok())
        throw std::invalid_argument(moduleMessage(label, "is not ready"));
    if (module->optimization_profile_count() != 1 || module->profile_idx() != 0) {
        throw std::invalid_argument(
            moduleMessage(label, "requires exactly one static optimization profile at index 0"));
    }
    for (const auto& contract : inputs) {
        if (module->input_is_dynamic(std::string(contract.name))) {
            throw std::invalid_argument(
                moduleMessage(label, "contains a dynamic input in the fixed native ABI"));
        }
    }
    validateDirection(*module, inputs, true, label);
    validateDirection(*module, outputs, false, label);
}

void validateReturnedOutputs(const TensorMap& outputs, const ContractList& contracts,
                             std::string_view label) {
    if (outputs.size() != contracts.size())
        throw std::runtime_error(moduleMessage(label, "returned the wrong output count"));
    for (const auto& contract : contracts) {
        const auto found = outputs.find(std::string(contract.name));
        if (found == outputs.end())
            throw std::runtime_error(
                moduleMessage(label, std::string("did not return ") + std::string(contract.name)));
        const auto& tensor = found->second;
        if (tensor.data == nullptr || tensor.shape != runtimeShape(contract) ||
            tensor.dtype != runtimeDtype(contract.data_type) ||
            tensor.numel() != checkedElementCount(contract)) {
            throw std::runtime_error(
                moduleMessage(label, std::string("returned an invalid tensor for ") +
                                         std::string(contract.name)));
        }
    }
}

const Tensor& requiredOutput(const TensorMap& outputs, const TensorContract& contract) {
    return outputs.at(std::string(contract.name));
}

TensorMap imageInput(PreprocessedFrame& frame) {
    TensorMap inputs;
    inputs.emplace(std::string(kPixelValues.name),
                   Tensor{frame.pixel_values.data(), runtimeShape(kPixelValues),
                          runtimeDtype(kPixelValues.data_type)});
    return inputs;
}

void appendFpnInputs(TensorMap& inputs, const TensorMap& image_outputs) {
    for (const auto& contract : kTrackerFpn) {
        const auto& output = requiredOutput(image_outputs, contract);
        inputs.emplace(std::string(contract.name), Tensor{output.data, output.shape, output.dtype});
    }
}

Sam2BBoxTensorView bboxView(const TensorMap& outputs, const TensorContract& contract) {
    const auto& tensor = requiredOutput(outputs, contract);
    if (tensor.dtype != DType::kBFloat16)
        throw std::runtime_error("SAM2 image module returned a non-BF16 bbox map");
    std::array<int64_t, 4> shape{};
    std::copy(tensor.shape.begin(), tensor.shape.end(), shape.begin());
    return {tensor.data, Sam2BBoxDataType::kBFloat16, shape, tensor.numel()};
}

Sam2BBoxRawOutputs bboxOutputs(const TensorMap& outputs) {
    return {
        bboxView(outputs, kBboxMaps[0]), bboxView(outputs, kBboxMaps[1]),
        bboxView(outputs, kBboxMaps[2]), bboxView(outputs, kBboxMaps[3]),
        bboxView(outputs, kBboxMaps[4]), bboxView(outputs, kBboxMaps[5]),
    };
}

std::vector<float> copyFloatTensor(const Tensor& tensor, std::size_t expected_elements,
                                   std::string_view label) {
    if (tensor.dtype != DType::kFloat32 || tensor.data == nullptr ||
        tensor.numel() != expected_elements) {
        throw std::runtime_error("SAM2 " + std::string(label) + " tensor copy contract drifted");
    }
    const auto* values = static_cast<const float*>(tensor.data);
    std::vector<float> result(values, values + expected_elements);
    if (!std::all_of(result.begin(), result.end(),
                     [](float value) { return std::isfinite(value); }))
        throw std::runtime_error("SAM2 " + std::string(label) + " contains NaN or infinity");
    return result;
}

std::vector<std::uint16_t> copyBfloat16Tensor(const Tensor& tensor, std::size_t expected_elements,
                                              std::string_view label) {
    if (tensor.dtype != DType::kBFloat16 || tensor.data == nullptr ||
        tensor.numel() != expected_elements) {
        throw std::runtime_error("SAM2 " + std::string(label) + " tensor copy contract drifted");
    }
    const auto* values = static_cast<const std::uint16_t*>(tensor.data);
    std::vector<std::uint16_t> result(values, values + expected_elements);
    if (!std::all_of(result.begin(), result.end(), [](std::uint16_t value) {
            return (value & UINT16_C(0x7F80)) != UINT16_C(0x7F80);
        })) {
        throw std::runtime_error("SAM2 " + std::string(label) + " contains NaN or infinity");
    }
    return result;
}

std::size_t expectedFrameElements() {
    return static_cast<std::size_t>(kOriginalImageHeight) *
           static_cast<std::size_t>(kOriginalImageWidth) * 3U;
}

void validateFrames(const Sam2VideoFrames& frames) {
    const auto format = frames.front().pixel_format;
    for (std::size_t index = 0; index < frames.size(); ++index) {
        const auto& frame = frames[index];
        const bool valid_float = frame.pixel_format == Sam2VideoPixelFormat::kFloat32Rgb01 &&
                                 frame.pixels != nullptr &&
                                 frame.pixel_elements == expectedFrameElements() &&
                                 frame.rgb8_pixels == nullptr && frame.rgb8_bytes == 0U;
        const bool valid_rgb8 = frame.pixel_format == Sam2VideoPixelFormat::kUint8Rgb &&
                                frame.pixels == nullptr && frame.pixel_elements == 0U &&
                                frame.rgb8_pixels != nullptr &&
                                frame.rgb8_bytes == expectedFrameElements();
        if (frame.frame_index != static_cast<std::int32_t>(index) ||
            frame.height != kOriginalImageHeight || frame.width != kOriginalImageWidth ||
            frame.pixel_format != format || (!valid_float && !valid_rgb8)) {
            throw std::invalid_argument(
                "SAM2 native v1 requires five contiguous, uniformly encoded 1280x1088 RGB "
                "frames");
        }
    }
}

const void* frameStorage(const Sam2VideoFrameView& frame) noexcept {
    return frame.pixel_format == Sam2VideoPixelFormat::kUint8Rgb
               ? static_cast<const void*>(frame.rgb8_pixels)
               : static_cast<const void*>(frame.pixels);
}

PreprocessedFrame preprocessFrameView(const Sam2VideoFrameView& frame) {
    if (frame.pixel_format == Sam2VideoPixelFormat::kUint8Rgb)
        return preprocessRgb8Frame(frame.rgb8_pixels, frame.height, frame.width);
    return preprocessFrame(frame.pixels, frame.height, frame.width);
}

bool sameTrack(const Sam2VideoTrack& left, const Sam2VideoTrack& right) {
    return left.label == right.label && left.detector_score == right.detector_score &&
           left.prompt_box_xyxy == right.prompt_box_xyxy;
}

class NativeVideoProcessorState final {
  public:
    explicit NativeVideoProcessorState(NativeVideoEngineSet engines)
        : engines_(std::move(engines)) {
        validateAllModules();
    }

    Sam2VideoPromptResult runBboxPrompt(const Sam2VideoFrames& frames) {
        std::lock_guard<std::mutex> lock(mutex_);
        requirePhase(Phase::kIdle, "bbox prompt requires reset before reuse");
        phase_ = Phase::kFailed;
        clearRunState();
        validateAllModules();
        validateFrames(frames);

        auto preprocessed = preprocessFrameView(frames[0]);
        auto image_outputs = engines_.image->forward(imageInput(preprocessed));
        validateReturnedOutputs(image_outputs, imageOutputs(), "image");

        const auto decoded =
            decode_sam2_bbox_outputs(bboxOutputs(image_outputs), frames[0].height, frames[0].width);
        const auto& detection = require_exactly_one_sam2_bbox_detection(decoded);

        std::array<float, 4> model_box = detection.model_xyxy_1024;
        TensorMap prompt_inputs;
        appendFpnInputs(prompt_inputs, image_outputs);
        prompt_inputs.emplace(
            std::string(kBoxPrompt.name),
            Tensor{model_box.data(), runtimeShape(kBoxPrompt), runtimeDtype(kBoxPrompt.data_type)});
        auto tracker_outputs = engines_.prompt->forward(prompt_inputs);
        validateReturnedOutputs(tracker_outputs, trackerOutputs(), "prompt");

        auto mask_logits = copyFloatTensor(requiredOutput(tracker_outputs, kMaskLogits256),
                                           kMaskLogitElements, "prompt mask logits");
        auto frame_zero_mask =
            resizeAndThresholdMask(mask_logits.data(), 256, 256, frames[0].height, frames[0].width);
        memory_history_ = copyBfloat16Tensor(requiredOutput(tracker_outputs, kMemoryFeatures),
                                             kMemoryFeatureElements, "prompt memory features");
        pointer_history_ = copyFloatTensor(requiredOutput(tracker_outputs, kObjectPointer),
                                           kObjectPointerElements, "prompt object pointer");

        track_ = {detection.label, detection.score, detection.original_xyxy};
        frame_zero_mask_ = frame_zero_mask;
        for (std::size_t index = 0; index < frames.size(); ++index)
            frame_pointers_[index] = frameStorage(frames[index]);
        phase_ = Phase::kPrompted;

        Sam2VideoPromptResult result;
        result.track = track_;
        result.frame_zero = makeFrameResult(frames[0], std::move(frame_zero_mask));
        return result;
    }

    Sam2VideoFrameResults propagate(const Sam2VideoPromptResult& prompt,
                                    const Sam2VideoFrames& frames) {
        std::lock_guard<std::mutex> lock(mutex_);
        requirePhase(Phase::kPrompted, "propagation requires one native prompt result");
        phase_ = Phase::kFailed;
        validateAllModules();
        validateFrames(frames);
        validatePromptAndFrames(prompt, frames);

        Sam2VideoFrameResults results;
        results[0] = makeFrameResult(frames[0], frame_zero_mask_);
        for (std::size_t frame_index = 1; frame_index < frames.size(); ++frame_index) {
            auto preprocessed = preprocessFrameView(frames[frame_index]);
            auto image_outputs = engines_.image->forward(imageInput(preprocessed));
            validateReturnedOutputs(image_outputs, imageOutputs(), "image");

            const auto history_frames = static_cast<std::int32_t>(frame_index);
            TensorMap recurrent_inputs;
            appendFpnInputs(recurrent_inputs, image_outputs);
            const auto memory_contract = historyMemoryFeatures(history_frames);
            const auto pointer_contract = historyObjectPointers(history_frames);
            recurrent_inputs.emplace(std::string(memory_contract.name),
                                     Tensor{memory_history_.data(), runtimeShape(memory_contract),
                                            runtimeDtype(memory_contract.data_type)});
            recurrent_inputs.emplace(std::string(pointer_contract.name),
                                     Tensor{pointer_history_.data(), runtimeShape(pointer_contract),
                                            runtimeDtype(pointer_contract.data_type)});

            auto tracker_outputs = engines_.recurrent[frame_index - 1U]->forward(recurrent_inputs);
            validateReturnedOutputs(tracker_outputs, trackerOutputs(), "recurrent");
            auto mask_logits = copyFloatTensor(requiredOutput(tracker_outputs, kMaskLogits256),
                                               kMaskLogitElements, "recurrent mask logits");
            auto mask =
                resizeAndThresholdMask(mask_logits.data(), 256, 256, frames[frame_index].height,
                                       frames[frame_index].width);
            appendHistory(tracker_outputs, history_frames);
            results[frame_index] = makeFrameResult(frames[frame_index], std::move(mask));
        }

        phase_ = Phase::kComplete;
        return results;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        phase_ = Phase::kFailed;
        clearRunState();

        std::exception_ptr first_failure;
        resetModule(*engines_.image, first_failure);
        resetModule(*engines_.prompt, first_failure);
        for (auto& recurrent : engines_.recurrent)
            resetModule(*recurrent, first_failure);
        if (first_failure != nullptr)
            std::rethrow_exception(first_failure);
        phase_ = Phase::kIdle;
    }

  private:
    enum class Phase { kIdle, kPrompted, kComplete, kFailed };

    static void resetModule(ITrtModule& module, std::exception_ptr& first_failure) {
        try {
            module.reset_execution_context();
        } catch (...) {
            if (first_failure == nullptr)
                first_failure = std::current_exception();
        }
    }

    void validateAllModules() const {
        validateModule(engines_.image.get(), imageInputs(), imageOutputs(), "image");
        validateModule(engines_.prompt.get(), promptInputs(), trackerOutputs(), "prompt");
        for (std::size_t index = 0; index < engines_.recurrent.size(); ++index) {
            validateModule(engines_.recurrent[index].get(),
                           recurrentInputs(static_cast<std::int32_t>(index + 1U)), trackerOutputs(),
                           "recurrent H" + std::to_string(index + 1U));
        }
    }

    void requirePhase(Phase expected, const char* message) const {
        if (phase_ != expected)
            throw std::logic_error(std::string("SAM2 native processor ") + message);
    }

    void clearRunState() {
        track_ = {};
        frame_pointers_ = {};
        std::vector<std::uint8_t>().swap(frame_zero_mask_);
        std::vector<std::uint16_t>().swap(memory_history_);
        std::vector<float>().swap(pointer_history_);
    }

    static Sam2VideoFrameResult makeFrameResult(const Sam2VideoFrameView& frame,
                                                std::vector<std::uint8_t> mask) {
        Sam2VideoFrameResult result;
        result.frame_index = frame.frame_index;
        result.height = frame.height;
        result.width = frame.width;
        result.mask = Sam2VideoMaskBuffer::host(std::move(mask));
        return result;
    }

    void validatePromptAndFrames(const Sam2VideoPromptResult& prompt,
                                 const Sam2VideoFrames& frames) const {
        if (!sameTrack(prompt.track, track_) || prompt.frame_zero.frame_index != 0 ||
            prompt.frame_zero.height != kOriginalImageHeight ||
            prompt.frame_zero.width != kOriginalImageWidth ||
            prompt.frame_zero.mask.memory_kind() != TRTMC_SAM2_VIDEO_MASK_MEMORY_HOST ||
            prompt.frame_zero.mask.byte_count() != frame_zero_mask_.size() ||
            prompt.frame_zero.mask.data() == nullptr ||
            !std::equal(frame_zero_mask_.begin(), frame_zero_mask_.end(),
                        static_cast<const std::uint8_t*>(prompt.frame_zero.mask.data()))) {
            throw std::invalid_argument(
                "SAM2 native propagation received a foreign or modified prompt result");
        }
        for (std::size_t index = 0; index < frames.size(); ++index) {
            if (frameStorage(frames[index]) != frame_pointers_[index]) {
                throw std::invalid_argument(
                    "SAM2 native propagation frame storage changed after prompting");
            }
        }
    }

    void appendHistory(const TensorMap& outputs, std::int32_t previous_history_frames) {
        auto memory = copyBfloat16Tensor(requiredOutput(outputs, kMemoryFeatures),
                                         kMemoryFeatureElements, "recurrent memory features");
        auto pointer = copyFloatTensor(requiredOutput(outputs, kObjectPointer),
                                       kObjectPointerElements, "recurrent object pointer");
        const auto expected_memory =
            static_cast<std::size_t>(previous_history_frames) * kMemoryFeatureElements;
        const auto expected_pointers =
            static_cast<std::size_t>(previous_history_frames) * kObjectPointerElements;
        if (memory_history_.size() != expected_memory ||
            pointer_history_.size() != expected_pointers)
            throw std::logic_error("SAM2 native recurrent history length drifted");
        memory_history_.insert(memory_history_.end(), memory.begin(), memory.end());
        pointer_history_.insert(pointer_history_.end(), pointer.begin(), pointer.end());
    }

    NativeVideoEngineSet engines_;
    mutable std::mutex mutex_;
    Phase phase_{Phase::kIdle};
    Sam2VideoTrack track_{};
    std::array<const void*, kSam2VideoFrameCount> frame_pointers_{};
    std::vector<std::uint8_t> frame_zero_mask_;
    std::vector<std::uint16_t> memory_history_;
    std::vector<float> pointer_history_;
};

void bindDeviceTensorExactly(ITrtModule& module, const TensorContract& contract, bool input,
                             void* address, std::string_view label) {
    const std::string name(contract.name);
    if (address == nullptr)
        throw std::invalid_argument(moduleMessage(label, "cannot bind a null device tensor"));
    const bool has_direction = input ? module.has_input(name) : module.has_output(name);
    const bool has_other_direction = input ? module.has_output(name) : module.has_input(name);
    if (!has_direction || has_other_direction ||
        module.tensor_shape(name) != runtimeShape(contract) ||
        module.tensor_dtype(name) != runtimeDtype(contract.data_type)) {
        throw std::invalid_argument(
            moduleMessage(label, std::string("cannot bind drifted tensor ") + name));
    }
    module.bind_external(name, address);
    if (module.device_ptr(name) != address) {
        throw std::runtime_error(
            moduleMessage(label, std::string("rejected external binding for ") + name));
    }
}

cudaStream_t requireSharedDeviceStream(const NativeVideoEngineSet& engines) {
    if (engines.image == nullptr)
        throw std::invalid_argument("SAM2 image module is missing");
    const auto stream = engines.image->stream();
    if (stream == nullptr)
        throw std::invalid_argument("SAM2 device processor requires a non-null shared CUDA stream");
    if (engines.prompt == nullptr || engines.prompt->stream() != stream) {
        throw std::invalid_argument(
            "SAM2 device processor requires one shared CUDA stream for all six modules");
    }
    for (const auto& recurrent : engines.recurrent) {
        if (recurrent == nullptr || recurrent->stream() != stream) {
            throw std::invalid_argument(
                "SAM2 device processor requires one shared CUDA stream for all six modules");
        }
    }
    return stream;
}

class WorkspaceDrainGuard final {
  public:
    explicit WorkspaceDrainGuard(Sam2DeviceWorkspace& workspace) noexcept
        : workspace_(&workspace) {}
    ~WorkspaceDrainGuard() {
        if (workspace_ != nullptr)
            workspace_->drainNoexcept();
    }

    WorkspaceDrainGuard(const WorkspaceDrainGuard&) = delete;
    WorkspaceDrainGuard& operator=(const WorkspaceDrainGuard&) = delete;

    void dismiss() noexcept { workspace_ = nullptr; }

  private:
    Sam2DeviceWorkspace* workspace_{nullptr};
};

class ScopedCudaDevice final {
  public:
    explicit ScopedCudaDevice(std::int32_t desired) : desired_(desired) {
        auto status = cudaGetDevice(&previous_);
        if (status != cudaSuccess) {
            throw std::runtime_error("SAM2 CUDA device query failed: " +
                                     std::string(cudaGetErrorString(status)));
        }
        if (previous_ != desired_) {
            status = cudaSetDevice(desired_);
            if (status != cudaSuccess) {
                throw std::runtime_error("SAM2 CUDA device selection failed: " +
                                         std::string(cudaGetErrorString(status)));
            }
        }
    }

    ~ScopedCudaDevice() {
        if (previous_ != desired_)
            (void)cudaSetDevice(previous_);
    }

    ScopedCudaDevice(const ScopedCudaDevice&) = delete;
    ScopedCudaDevice& operator=(const ScopedCudaDevice&) = delete;

  private:
    std::int32_t desired_{-1};
    std::int32_t previous_{-1};
};

void destroyModuleOnStreamDevice(std::unique_ptr<ITrtModule>& module) noexcept {
    if (module == nullptr)
        return;
    try {
        const auto stream = module->stream();
        std::int32_t device = -1;
        if (stream != nullptr && cudaStreamGetDevice(stream, &device) == cudaSuccess) {
            std::int32_t previous = -1;
            const auto query_status = cudaGetDevice(&previous);
            if (cudaSetDevice(device) == cudaSuccess) {
                module.reset();
                if (query_status == cudaSuccess && previous != device)
                    (void)cudaSetDevice(previous);
                return;
            }
        } else {
            (void)cudaGetLastError();
        }
    } catch (...) {
    }
    module.reset();
}

void destroyEngineSetOnOwnDevices(NativeVideoEngineSet& engines) noexcept {
    destroyModuleOnStreamDevice(engines.image);
    destroyModuleOnStreamDevice(engines.prompt);
    for (auto& recurrent : engines.recurrent)
        destroyModuleOnStreamDevice(recurrent);
}

class NativeDeviceVideoProcessorState final {
  public:
    explicit NativeDeviceVideoProcessorState(NativeVideoEngineSet engines)
        : workspace_(), engines_(std::move(engines)) {
        try {
            const auto stream = requireSharedDeviceStream(engines_);
            std::int32_t stream_device = -1;
            const auto status = cudaStreamGetDevice(stream, &stream_device);
            if (status != cudaSuccess) {
                throw std::runtime_error("SAM2 CUDA stream-device query failed: " +
                                         std::string(cudaGetErrorString(status)));
            }
            ScopedCudaDevice selected(stream_device);
            try {
                validateAllModules();
                workspace_ = Sam2DeviceWorkspace::create(stream);
                validateAllDeviceStorage();
                bindDeviceGraph();
                validateDeviceGraph();
                retainWorkspaceWithModules();
            } catch (...) {
                if (workspace_ != nullptr)
                    workspace_->drainNoexcept();
                engines_ = {};
                throw;
            }
        } catch (...) {
            destroyEngineSetOnOwnDevices(engines_);
            throw;
        }
    }

    ~NativeDeviceVideoProcessorState() {
        if (workspace_ == nullptr)
            return;
        try {
            ScopedCudaDevice selected(workspace_->deviceOrdinal());
            workspace_->drainNoexcept();
            engines_ = {};
        } catch (...) {
            workspace_->drainNoexcept();
        }
    }

    Sam2VideoPromptResult runBboxPrompt(const Sam2VideoFrames& frames) {
        std::lock_guard<std::mutex> lock(mutex_);
        ScopedCudaDevice selected(workspace_->deviceOrdinal());
        requirePhase(Phase::kIdle, "bbox prompt requires reset before reuse");
        phase_ = Phase::kFailed;
        clearRunState();
        PreprocessedFrame preprocessed;
        std::array<float, 4> model_box{};
        WorkspaceDrainGuard drain_guard(*workspace_);
        try {
            validateAllModules();
            validateDeviceGraph();
            validateFrames(frames);
            workspace_->beginRun();

            enqueueImageFrame(frames[0], preprocessed);
            workspace_->enqueueBboxDownload(*engines_.image);
            const auto decoded = decode_sam2_bbox_outputs(workspace_->waitForBbox(),
                                                          frames[0].height, frames[0].width);
            const auto& detection = require_exactly_one_sam2_bbox_detection(decoded);

            model_box = detection.model_xyxy_1024;
            TensorMap prompt_inputs;
            prompt_inputs.emplace(std::string(kBoxPrompt.name),
                                  Tensor{model_box.data(), runtimeShape(kBoxPrompt),
                                         runtimeDtype(kBoxPrompt.data_type)});
            engines_.prompt->forward_async(prompt_inputs);
            workspace_->enqueueTrackerPostprocess(*engines_.prompt, 0U);
            workspace_->finishTrackerStage("prompt");

            track_ = {detection.label, detection.score, detection.original_xyxy};
            for (std::size_t index = 0; index < frames.size(); ++index)
                frame_pointers_[index] = frameStorage(frames[index]);
            phase_ = Phase::kPrompted;

            Sam2VideoPromptResult result;
            result.track = track_;
            result.frame_zero = makeFrameResult(frames[0], 0U);
            drain_guard.dismiss();
            return result;
        } catch (...) {
            throw;
        }
    }

    Sam2VideoFrameResults propagate(const Sam2VideoPromptResult& prompt,
                                    const Sam2VideoFrames& frames) {
        std::lock_guard<std::mutex> lock(mutex_);
        ScopedCudaDevice selected(workspace_->deviceOrdinal());
        requirePhase(Phase::kPrompted, "propagation requires one native prompt result");
        phase_ = Phase::kFailed;
        std::array<PreprocessedFrame, kSam2VideoFrameCount - 1U> preprocessed_frames;
        WorkspaceDrainGuard drain_guard(*workspace_);
        try {
            validateAllModules();
            validateDeviceGraph();
            validateFrames(frames);
            validatePromptAndFrames(prompt, frames);

            Sam2VideoFrameResults results;
            results[0] = makeFrameResult(frames[0], 0U);
            for (std::size_t frame_index = 1; frame_index < frames.size(); ++frame_index) {
                auto& preprocessed = preprocessed_frames[frame_index - 1U];
                enqueueImageFrame(frames[frame_index], preprocessed);
                auto& recurrent = *engines_.recurrent[frame_index - 1U];
                recurrent.forward_async(TensorMap{});
                workspace_->enqueueTrackerPostprocess(recurrent, frame_index);
                results[frame_index] = makeFrameResult(frames[frame_index], frame_index);
            }
            workspace_->finishTrackerStage("propagation");
            phase_ = Phase::kComplete;
            drain_guard.dismiss();
            return results;
        } catch (...) {
            throw;
        }
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        ScopedCudaDevice selected(workspace_->deviceOrdinal());
        phase_ = Phase::kFailed;
        clearRunState();

        std::exception_ptr first_failure;
        try {
            workspace_->drain();
        } catch (...) {
            first_failure = std::current_exception();
        }
        workspace_->invalidateRun();
        resetModule(*engines_.image, first_failure);
        resetModule(*engines_.prompt, first_failure);
        for (auto& recurrent : engines_.recurrent)
            resetModule(*recurrent, first_failure);
        if (first_failure != nullptr)
            std::rethrow_exception(first_failure);
        validateDeviceGraph();
        phase_ = Phase::kIdle;
    }

  private:
    enum class Phase { kIdle, kPrompted, kComplete, kFailed };

    static void resetModule(ITrtModule& module, std::exception_ptr& first_failure) {
        try {
            module.reset_execution_context();
        } catch (...) {
            if (first_failure == nullptr)
                first_failure = std::current_exception();
        }
    }

    void validateAllModules() const {
        validateModule(engines_.image.get(), imageInputs(), imageOutputs(), "image");
        validateModule(engines_.prompt.get(), promptInputs(), trackerOutputs(), "prompt");
        for (std::size_t index = 0; index < engines_.recurrent.size(); ++index) {
            validateModule(engines_.recurrent[index].get(),
                           recurrentInputs(static_cast<std::int32_t>(index + 1U)), trackerOutputs(),
                           "recurrent H" + std::to_string(index + 1U));
        }
    }

    void validateDeviceStorage(const ITrtModule& module, const ContractList& contracts,
                               std::string_view label) const {
        for (const auto& contract : contracts) {
            const std::string name(contract.name);
            const void* pointer = module.device_ptr(name);
            if (pointer == nullptr ||
                !workspace_->isDeviceSpan(pointer, checkedByteCount(contract))) {
                throw std::invalid_argument(moduleMessage(
                    label,
                    "tensor is not CUDA device storage on the shared stream device: " + name));
            }
        }
    }

    void validateAllDeviceStorage() const {
        validateDeviceStorage(*engines_.image, imageInputs(), "device image");
        validateDeviceStorage(*engines_.image, imageOutputs(), "device image");
        validateDeviceStorage(*engines_.prompt, promptInputs(), "device prompt");
        validateDeviceStorage(*engines_.prompt, trackerOutputs(), "device prompt");
        for (std::size_t index = 0; index < engines_.recurrent.size(); ++index) {
            const auto label = "device recurrent H" + std::to_string(index + 1U);
            validateDeviceStorage(*engines_.recurrent[index],
                                  recurrentInputs(static_cast<std::int32_t>(index + 1U)), label);
            validateDeviceStorage(*engines_.recurrent[index], trackerOutputs(), label);
        }
    }

    std::array<ITrtModule*, 5> trackers() const {
        return {engines_.prompt.get(), engines_.recurrent[0].get(), engines_.recurrent[1].get(),
                engines_.recurrent[2].get(), engines_.recurrent[3].get()};
    }

    void bindDeviceGraph() {
        bindDeviceTensorExactly(*engines_.image, kPixelValues, true,
                                workspace_->preprocessedPixelValues(), "device image preprocess");
        for (auto* tracker : trackers()) {
            for (const auto& fpn : kTrackerFpn) {
                void* producer = engines_.image->device_ptr(std::string(fpn.name));
                if (producer == nullptr) {
                    throw std::invalid_argument("SAM2 image FPN output has no device storage: " +
                                                std::string(fpn.name));
                }
                bindDeviceTensorExactly(*tracker, fpn, true, producer, "device tracker");
            }
        }

        std::array<ITrtModule*, 4> history_producers = {
            engines_.prompt.get(), engines_.recurrent[0].get(), engines_.recurrent[1].get(),
            engines_.recurrent[2].get()};
        for (std::size_t frame = 0; frame < history_producers.size(); ++frame) {
            bindDeviceTensorExactly(*history_producers[frame], kMemoryFeatures, false,
                                    workspace_->historyMemorySlot(frame),
                                    "device history producer");
            bindDeviceTensorExactly(*history_producers[frame], kObjectPointer, false,
                                    workspace_->historyPointerSlot(frame),
                                    "device history producer");
        }

        for (std::size_t index = 0; index < engines_.recurrent.size(); ++index) {
            const auto history_frames = static_cast<std::int32_t>(index + 1U);
            bindDeviceTensorExactly(*engines_.recurrent[index],
                                    historyMemoryFeatures(history_frames), true,
                                    workspace_->historyMemoryBase(), "device recurrent history");
            bindDeviceTensorExactly(*engines_.recurrent[index],
                                    historyObjectPointers(history_frames), true,
                                    workspace_->historyPointerBase(), "device recurrent history");
        }
    }

    void validateDeviceGraph() const {
        const auto shared_stream = requireSharedDeviceStream(engines_);
        if (workspace_ == nullptr || workspace_->stream() != shared_stream)
            throw std::runtime_error("SAM2 device workspace stream binding drifted");
        validateAllDeviceStorage();

        if (engines_.image->device_ptr(std::string(kPixelValues.name)) !=
            workspace_->preprocessedPixelValues()) {
            throw std::runtime_error("SAM2 device image preprocess binding drifted");
        }

        for (auto* tracker : trackers()) {
            for (const auto& fpn : kTrackerFpn) {
                const std::string name(fpn.name);
                if (engines_.image->device_ptr(name) == nullptr ||
                    tracker->device_ptr(name) != engines_.image->device_ptr(name)) {
                    throw std::runtime_error("SAM2 device FPN binding drifted for " + name);
                }
            }
            for (const auto& output : trackerOutputs()) {
                if (tracker->device_ptr(std::string(output.name)) == nullptr) {
                    throw std::runtime_error("SAM2 tracker output has no device storage: " +
                                             std::string(output.name));
                }
            }
        }

        std::array<ITrtModule*, 4> history_producers = {
            engines_.prompt.get(), engines_.recurrent[0].get(), engines_.recurrent[1].get(),
            engines_.recurrent[2].get()};
        for (std::size_t frame = 0; frame < history_producers.size(); ++frame) {
            if (history_producers[frame]->device_ptr(std::string(kMemoryFeatures.name)) !=
                    workspace_->historyMemorySlot(frame) ||
                history_producers[frame]->device_ptr(std::string(kObjectPointer.name)) !=
                    workspace_->historyPointerSlot(frame)) {
                throw std::runtime_error("SAM2 device history output binding drifted");
            }
        }
        for (const auto& recurrent : engines_.recurrent) {
            if (recurrent->device_ptr(std::string(historyMemoryFeatures(1).name)) !=
                    workspace_->historyMemoryBase() ||
                recurrent->device_ptr(std::string(historyObjectPointers(1).name)) !=
                    workspace_->historyPointerBase()) {
                throw std::runtime_error("SAM2 device recurrent history binding drifted");
            }
        }
    }

    void retainWorkspaceWithModules() {
        engines_.image->keep_alive(workspace_);
        engines_.prompt->keep_alive(workspace_);
        for (auto& recurrent : engines_.recurrent)
            recurrent->keep_alive(workspace_);
    }

    void requirePhase(Phase expected, const char* message) const {
        if (phase_ != expected)
            throw std::logic_error(std::string("SAM2 native device processor ") + message);
    }

    void clearRunState() {
        track_ = {};
        frame_pointers_ = {};
    }

    void enqueueImageFrame(const Sam2VideoFrameView& frame, PreprocessedFrame& host_storage) {
        if (frame.pixel_format == Sam2VideoPixelFormat::kUint8Rgb) {
            workspace_->enqueueRgb8Preprocess(frame.rgb8_pixels, frame.height, frame.width);
            // pixel_values is an external image input filled by the two CUDA
            // kernels immediately before this enqueue on the identical stream.
            engines_.image->forward_async(TensorMap{});
            return;
        }
        host_storage = preprocessFrame(frame.pixels, frame.height, frame.width);
        engines_.image->forward_async(imageInput(host_storage));
    }

    Sam2VideoFrameResult makeFrameResult(const Sam2VideoFrameView& frame, std::size_t frame_index) {
        Sam2VideoFrameResult result;
        result.frame_index = frame.frame_index;
        result.height = frame.height;
        result.width = frame.width;
        result.mask = workspace_->makeMaskBuffer(frame_index);
        return result;
    }

    void validatePromptAndFrames(const Sam2VideoPromptResult& prompt,
                                 const Sam2VideoFrames& frames) const {
        if (!sameTrack(prompt.track, track_) || prompt.frame_zero.frame_index != 0 ||
            prompt.frame_zero.height != kOriginalImageHeight ||
            prompt.frame_zero.width != kOriginalImageWidth ||
            !workspace_->ownsMask(prompt.frame_zero.mask, 0U)) {
            throw std::invalid_argument(
                "SAM2 native device propagation received a foreign or modified prompt result");
        }
        for (std::size_t index = 0; index < frames.size(); ++index) {
            if (frameStorage(frames[index]) != frame_pointers_[index]) {
                throw std::invalid_argument(
                    "SAM2 native device propagation frame storage changed after prompting");
            }
        }
    }

    // Declaration order ensures every module releases its external bindings
    // before the workspace allocation is destroyed.
    std::shared_ptr<Sam2DeviceWorkspace> workspace_;
    NativeVideoEngineSet engines_;
    mutable std::mutex mutex_;
    Phase phase_{Phase::kIdle};
    Sam2VideoTrack track_{};
    std::array<const void*, kSam2VideoFrameCount> frame_pointers_{};
};

} // namespace

Sam2VideoProcessor makeNativeVideoProcessor(NativeVideoEngineSet engines) {
    auto state = std::make_shared<NativeVideoProcessorState>(std::move(engines));
    Sam2VideoProcessor processor;
    processor.run_bbox_prompt = [state](const Sam2VideoFrames& frames) {
        return state->runBboxPrompt(frames);
    };
    processor.propagate = [state](const Sam2VideoPromptResult& prompt,
                                  const Sam2VideoFrames& frames) {
        return state->propagate(prompt, frames);
    };
    processor.reset = [state] { state->reset(); };
    return processor;
}

Sam2VideoProcessor makeNativeDeviceVideoProcessor(NativeVideoEngineSet engines) {
    auto state = std::make_shared<NativeDeviceVideoProcessorState>(std::move(engines));
    Sam2VideoProcessor processor;
    processor.run_bbox_prompt = [state](const Sam2VideoFrames& frames) {
        return state->runBboxPrompt(frames);
    };
    processor.propagate = [state](const Sam2VideoPromptResult& prompt,
                                  const Sam2VideoFrames& frames) {
        return state->propagate(prompt, frames);
    };
    processor.reset = [state] { state->reset(); };
    return processor;
}

} // namespace trtmc::sam2
