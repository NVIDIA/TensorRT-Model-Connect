/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_engine_contract.h"
#include "runtime/models/sam2/sam2_native_video_processor.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Contracts = std::vector<trtmc::sam2::TensorContract>;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void check(bool condition, const char* message) {
    if (!condition)
        fail(message);
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        fail(std::string(operation) + " failed: " + cudaGetErrorString(status));
}

template <typename Exception, typename Function>
void checkThrows(Function&& function, const char* needle, const char* message) {
    static_assert(std::is_base_of<std::exception, Exception>::value);
    try {
        function();
    } catch (const Exception& error) {
        if (std::strstr(error.what(), needle) != nullptr)
            return;
        fail(std::string(message) + ": wrong message: " + error.what());
    } catch (const std::exception& error) {
        fail(std::string(message) + ": wrong exception: " + error.what());
    }
    fail(std::string(message) + ": no exception");
}

trtmc::DType dtype(trtmc::sam2::TensorDataType data_type) {
    return data_type == trtmc::sam2::TensorDataType::kFloat32 ? trtmc::DType::kFloat32
                                                              : trtmc::DType::kBFloat16;
}

std::vector<std::int64_t> shape(const trtmc::sam2::TensorContract& contract) {
    std::vector<std::int64_t> result;
    result.reserve(contract.rank);
    for (std::uint8_t index = 0; index < contract.rank; ++index)
        result.push_back(contract.dimensions[index]);
    return result;
}

std::size_t elements(const trtmc::sam2::TensorContract& contract) {
    std::size_t result = 1U;
    for (const auto dimension : shape(contract))
        result *= static_cast<std::size_t>(dimension);
    return result;
}

std::size_t bytes(const trtmc::sam2::TensorContract& contract) {
    return elements(contract) * trtmc::dtype_size(dtype(contract.data_type));
}

Contracts imageInputs() {
    return {trtmc::sam2::kPixelValues};
}

Contracts imageOutputs() {
    Contracts result(trtmc::sam2::kTrackerFpn.begin(), trtmc::sam2::kTrackerFpn.end());
    result.insert(result.end(), trtmc::sam2::kBboxMaps.begin(), trtmc::sam2::kBboxMaps.end());
    return result;
}

Contracts promptInputs() {
    Contracts result(trtmc::sam2::kTrackerFpn.begin(), trtmc::sam2::kTrackerFpn.end());
    result.push_back(trtmc::sam2::kBoxPrompt);
    return result;
}

Contracts trackerOutputs() {
    return {trtmc::sam2::kMaskLogits256, trtmc::sam2::kObjectPointer, trtmc::sam2::kMemoryFeatures};
}

Contracts recurrentInputs(std::int32_t history) {
    Contracts result(trtmc::sam2::kTrackerFpn.begin(), trtmc::sam2::kTrackerFpn.end());
    result.push_back(trtmc::sam2::historyMemoryFeatures(history));
    result.push_back(trtmc::sam2::historyObjectPointers(history));
    return result;
}

std::uint16_t bfloat16(float value) {
    std::uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += UINT32_C(0x7FFF) + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

class StreamOwner final {
  public:
    StreamOwner() {
        checkCuda(cudaGetDevice(&device_), "stream device query");
        checkCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream creation");
    }
    ~StreamOwner() {
        if (stream_ == nullptr)
            return;
        std::int32_t previous = -1;
        if (cudaGetDevice(&previous) == cudaSuccess && previous != device_)
            (void)cudaSetDevice(device_);
        (void)cudaStreamDestroy(stream_);
        if (previous >= 0 && previous != device_)
            (void)cudaSetDevice(previous);
    }
    StreamOwner(const StreamOwner&) = delete;
    StreamOwner& operator=(const StreamOwner&) = delete;
    cudaStream_t get() const noexcept { return stream_; }

  private:
    cudaStream_t stream_{nullptr};
    std::int32_t device_{-1};
};

class DeviceBuffer final {
  public:
    explicit DeviceBuffer(std::size_t size) : size_(size) {
        checkCuda(cudaGetDevice(&device_), "fake allocation device query");
        checkCuda(cudaMalloc(&data_, size_), "fake module CUDA allocation");
    }
    ~DeviceBuffer() {
        if (data_ == nullptr)
            return;
        std::int32_t previous = -1;
        if (cudaGetDevice(&previous) == cudaSuccess && previous != device_)
            (void)cudaSetDevice(device_);
        (void)cudaFree(data_);
        if (previous >= 0 && previous != device_)
            (void)cudaSetDevice(previous);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void* data() const noexcept { return data_; }

  private:
    void* data_{nullptr};
    std::size_t size_{0U};
    std::int32_t device_{-1};
};

class FakeDeviceModule final : public trtmc::ITrtModule {
  public:
    enum class Kind { kImage, kTracker };

    FakeDeviceModule(Contracts inputs, Contracts outputs, Kind kind, cudaStream_t stream,
                     std::int32_t tracker_ordinal = -1, bool allocate_outputs = true)
        : inputs_(std::move(inputs)), outputs_(std::move(outputs)), kind_(kind), stream_(stream),
          tracker_ordinal_(tracker_ordinal) {
        if (!allocate_outputs)
            return;
        for (const auto& contract : inputs_)
            allocate(contract);
        for (const auto& contract : outputs_)
            allocate(contract);
        if (kind_ == Kind::kImage)
            initializeImageOutputs();
        else
            initializeTrackerOutputs();
    }

    trtmc::TensorMap forward(const trtmc::TensorMap&) override {
        ++host_forward_calls;
        fail("device fake unexpectedly used synchronous host forward");
    }

    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override {
        fail("device fake unexpectedly used synchronous device forward");
    }

    void forward_device_async(const trtmc::DeviceTensorMap&) override {
        fail("device fake unexpectedly used copying device forward");
    }

    void forward_async(const trtmc::TensorMap& inputs) override {
        ++async_forward_calls;
        if (kind_ == Kind::kImage) {
            check(inputs.size() == 1U &&
                      inputs.count(std::string(trtmc::sam2::kPixelValues.name)) == 1U,
                  "device image receives only host pixel input");
            return;
        }

        validateExternalTrackerInputs(inputs);
        if (tracker_ordinal_ == 0) {
            const auto& box = inputs.at(std::string(trtmc::sam2::kBoxPrompt.name));
            deferred_box_source_ = static_cast<const float*>(box.data);
            checkCuda(cudaLaunchHostFunc(stream_, observeBox, this),
                      "deferred prompt-box observation");
        }
        uploadTrackerOutputs();
        if (invalidate_mask_after_forward) {
            bindings_[std::string(trtmc::sam2::kMaskLogits256.name)] = nullptr;
        }
    }

    void sync() override { checkCuda(cudaStreamSynchronize(stream_), "fake module stream sync"); }
    cudaStream_t stream() const override { return stream_; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    std::int32_t profile_idx() const override { return active_profile; }

    std::vector<trtmc::TensorInfo> input_info() const override { return metadata(inputs_, true); }
    std::vector<trtmc::TensorInfo> output_info() const override {
        return metadata(outputs_, false);
    }

    bool has_input(const std::string& name) const override {
        return find(inputs_, name) != nullptr;
    }
    bool has_output(const std::string& name) const override {
        return find(outputs_, name) != nullptr;
    }

    trtmc::DType tensor_dtype(const std::string& name) const override {
        const auto* contract = find(inputs_, name);
        if (contract == nullptr)
            contract = find(outputs_, name);
        if (contract == nullptr)
            fail("fake module dtype query used an unknown tensor");
        return dtype(contract->data_type);
    }

    std::vector<std::int64_t> tensor_shape(const std::string& name) const override {
        const auto* contract = find(inputs_, name);
        if (contract == nullptr)
            contract = find(outputs_, name);
        return contract == nullptr ? std::vector<std::int64_t>{} : shape(*contract);
    }

    std::vector<std::int64_t> input_profile_shape(const std::string& name, std::int32_t profile,
                                                  trtmc::ProfileShapeSelector) const override {
        if (profile != 0 || !has_input(name))
            return {};
        return tensor_shape(name);
    }

    std::int32_t optimization_profile_count() const override { return profile_count; }
    bool input_is_dynamic(const std::string& name) const override { return name == dynamic_input; }

    void* device_ptr(const std::string& name) const override {
        const auto found = bindings_.find(name);
        return found == bindings_.end() ? nullptr : found->second;
    }

    void bind_external(const std::string& name, void* pointer) override {
        if (!has_input(name) && !has_output(name))
            fail("fake module received an unknown external binding");
        if (pointer == nullptr)
            fail("fake module received a null external binding");
        if (name == rejected_binding)
            throw std::runtime_error("synthetic external binding rejection");
        if (name == ignored_binding)
            return;
        bindings_[name] = pointer;
        ++external_bind_calls;
    }

    void reset_execution_context() override { ++reset_calls; }
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void> resource) override {
        keep_alive_.push_back(std::move(resource));
    }

    void tamperBinding(const std::string& name, void* pointer) { bindings_[name] = pointer; }

    void* ownedPointer(const std::string& name) const {
        const auto found = owned_.find(name);
        return found == owned_.end() ? nullptr : found->second->data();
    }

    void setMaskNonFinite(bool enabled) {
        mask_logits_.front() = enabled ? std::numeric_limits<float>::quiet_NaN()
                                       : (tracker_ordinal_ % 2 == 0 ? 1.0F : -1.0F);
    }

    std::int32_t async_forward_calls{0};
    std::int32_t host_forward_calls{0};
    std::int32_t external_bind_calls{0};
    std::int32_t reset_calls{0};
    std::int32_t profile_count{1};
    std::int32_t active_profile{0};
    std::string dynamic_input;
    bool invalidate_mask_after_forward{false};
    bool deferred_box_observed{false};
    std::string rejected_binding;
    std::string ignored_binding;
    std::array<float, 4> observed_box{};

  private:
    static void CUDART_CB observeBox(void* context) {
        auto* module = static_cast<FakeDeviceModule*>(context);
        std::copy_n(module->deferred_box_source_, module->observed_box.size(),
                    module->observed_box.begin());
        module->deferred_box_observed = true;
    }

    static const trtmc::sam2::TensorContract* find(const Contracts& contracts,
                                                   const std::string& name) {
        const auto found = std::find_if(contracts.begin(), contracts.end(),
                                        [&](const auto& item) { return item.name == name; });
        return found == contracts.end() ? nullptr : &*found;
    }

    static std::vector<trtmc::TensorInfo> metadata(const Contracts& contracts, bool input) {
        std::vector<trtmc::TensorInfo> result;
        for (const auto& contract : contracts) {
            result.push_back(
                {std::string(contract.name), shape(contract), dtype(contract.data_type), input});
        }
        return result;
    }

    void allocate(const trtmc::sam2::TensorContract& contract) {
        const std::string name(contract.name);
        auto buffer = std::make_unique<DeviceBuffer>(bytes(contract));
        bindings_[name] = buffer->data();
        owned_.emplace(name, std::move(buffer));
    }

    void initializeImageOutputs() {
        for (const auto& fpn : trtmc::sam2::kTrackerFpn) {
            checkCuda(cudaMemsetAsync(device_ptr(std::string(fpn.name)), 0, bytes(fpn), stream_),
                      "fake FPN initialization");
        }

        constexpr std::array<std::int32_t, 3> sizes = {128, 64, 32};
        for (std::size_t level = 0; level < sizes.size(); ++level) {
            const auto area = static_cast<std::size_t>(sizes[level]) * sizes[level];
            bbox_cls_[level].assign(2U * area, bfloat16(-20.0F));
            bbox_reg_[level].assign(4U * area, bfloat16(0.0F));
        }
        const std::size_t stride8_area = 128U * 128U;
        const std::size_t anchor = 84U * 128U + 76U;
        bbox_cls_[0][stride8_area + anchor] = bfloat16(4.0F);
        constexpr std::array<float, 4> distances = {5.0F, 4.5F, 5.0F, 4.0F};
        for (std::size_t coordinate = 0; coordinate < distances.size(); ++coordinate)
            bbox_reg_[0][coordinate * stride8_area + anchor] = bfloat16(distances[coordinate]);

        for (std::size_t level = 0; level < 3U; ++level) {
            checkCuda(cudaMemcpyAsync(device_ptr(std::string(trtmc::sam2::kBboxMaps[level].name)),
                                      bbox_cls_[level].data(),
                                      bbox_cls_[level].size() * sizeof(std::uint16_t),
                                      cudaMemcpyHostToDevice, stream_),
                      "fake bbox classification initialization");
            checkCuda(cudaMemcpyAsync(
                          device_ptr(std::string(trtmc::sam2::kBboxMaps[level + 3U].name)),
                          bbox_reg_[level].data(), bbox_reg_[level].size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice, stream_),
                      "fake bbox regression initialization");
        }
    }

    void initializeTrackerOutputs() {
        mask_logits_.assign(elements(trtmc::sam2::kMaskLogits256),
                            tracker_ordinal_ % 2 == 0 ? 1.0F : -1.0F);
        object_pointer_.assign(elements(trtmc::sam2::kObjectPointer),
                               10.0F + static_cast<float>(tracker_ordinal_));
        memory_features_.assign(elements(trtmc::sam2::kMemoryFeatures),
                                bfloat16(1.0F + static_cast<float>(tracker_ordinal_)));
    }

    void validateExternalTrackerInputs(const trtmc::TensorMap& inputs) const {
        const bool prompt = tracker_ordinal_ == 0;
        if (inputs.size() != (prompt ? 1U : 0U))
            fail("device tracker received unexpected host inputs");
        for (const auto& contract : inputs_) {
            const std::string name(contract.name);
            if (name == trtmc::sam2::kBoxPrompt.name) {
                const auto found = inputs.find(name);
                if (!prompt || found == inputs.end() || found->second.data == nullptr ||
                    found->second.shape != shape(contract) ||
                    found->second.dtype != dtype(contract.data_type)) {
                    fail("device prompt box input drifted");
                }
                continue;
            }
            if (device_ptr(name) == nullptr || inputs.count(name) != 0U)
                fail("device tracker external input binding drifted");
        }
    }

    void uploadTrackerOutputs() {
        checkCuda(cudaMemcpyAsync(device_ptr(std::string(trtmc::sam2::kMaskLogits256.name)),
                                  mask_logits_.data(), mask_logits_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, stream_),
                  "fake mask output upload");
        checkCuda(cudaMemcpyAsync(device_ptr(std::string(trtmc::sam2::kObjectPointer.name)),
                                  object_pointer_.data(), object_pointer_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, stream_),
                  "fake pointer output upload");
        checkCuda(cudaMemcpyAsync(device_ptr(std::string(trtmc::sam2::kMemoryFeatures.name)),
                                  memory_features_.data(),
                                  memory_features_.size() * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice, stream_),
                  "fake memory output upload");
    }

    Contracts inputs_;
    Contracts outputs_;
    Kind kind_;
    cudaStream_t stream_{nullptr};
    std::int32_t tracker_ordinal_{-1};
    std::unordered_map<std::string, std::unique_ptr<DeviceBuffer>> owned_;
    std::unordered_map<std::string, void*> bindings_;
    std::array<std::vector<std::uint16_t>, 3> bbox_cls_;
    std::array<std::vector<std::uint16_t>, 3> bbox_reg_;
    std::vector<float> mask_logits_;
    std::vector<float> object_pointer_;
    std::vector<std::uint16_t> memory_features_;
    std::vector<std::shared_ptr<void>> keep_alive_;
    const float* deferred_box_source_{nullptr};
};

struct EngineFixture {
    trtmc::sam2::NativeVideoEngineSet engines;
    std::array<FakeDeviceModule*, 6> modules{};
    std::shared_ptr<StreamOwner> stream_owner;
};

EngineFixture makeEngines(cudaStream_t image_stream, cudaStream_t tracker_stream,
                          bool allocate_outputs) {
    EngineFixture fixture;
    auto image = std::make_unique<FakeDeviceModule>(imageInputs(), imageOutputs(),
                                                    FakeDeviceModule::Kind::kImage, image_stream,
                                                    -1, allocate_outputs);
    fixture.modules[0] = image.get();
    fixture.engines.image = std::move(image);

    auto prompt = std::make_unique<FakeDeviceModule>(promptInputs(), trackerOutputs(),
                                                     FakeDeviceModule::Kind::kTracker,
                                                     tracker_stream, 0, allocate_outputs);
    fixture.modules[1] = prompt.get();
    fixture.engines.prompt = std::move(prompt);
    for (std::size_t index = 0; index < fixture.engines.recurrent.size(); ++index) {
        auto recurrent = std::make_unique<FakeDeviceModule>(
            recurrentInputs(static_cast<std::int32_t>(index + 1U)), trackerOutputs(),
            FakeDeviceModule::Kind::kTracker, tracker_stream, static_cast<std::int32_t>(index + 1U),
            allocate_outputs);
        fixture.modules[index + 2U] = recurrent.get();
        fixture.engines.recurrent[index] = std::move(recurrent);
    }
    return fixture;
}

EngineFixture makeGpuEngines() {
    auto stream_owner = std::make_shared<StreamOwner>();
    auto fixture = makeEngines(stream_owner->get(), stream_owner->get(), true);
    fixture.stream_owner = stream_owner;
    for (auto* module : fixture.modules)
        module->keep_alive(stream_owner);
    return fixture;
}

struct FrameFixture {
    std::vector<float> pixels;
    trtmc::Sam2VideoFrames views{};
};

FrameFixture makeFrames() {
    FrameFixture fixture;
    const auto count = static_cast<std::size_t>(trtmc::sam2::kOriginalImageHeight) *
                       trtmc::sam2::kOriginalImageWidth * 3U;
    fixture.pixels.assign(count, 0.5F);
    for (std::size_t index = 0; index < fixture.views.size(); ++index) {
        fixture.views[index] = {static_cast<std::int32_t>(index), trtmc::sam2::kOriginalImageHeight,
                                trtmc::sam2::kOriginalImageWidth, fixture.pixels.data(), count};
    }
    return fixture;
}

void testStreamGateWithoutCuda() {
    {
        auto fixture = makeEngines(nullptr, nullptr, false);
        checkThrows<std::invalid_argument>(
            [&] { (void)trtmc::sam2::makeNativeDeviceVideoProcessor(std::move(fixture.engines)); },
            "non-null shared CUDA stream", "device processor rejects null streams");
    }
    {
        const auto stream_a = reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(1U));
        const auto stream_b = reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(2U));
        auto fixture = makeEngines(stream_a, stream_b, false);
        checkThrows<std::invalid_argument>(
            [&] { (void)trtmc::sam2::makeNativeDeviceVideoProcessor(std::move(fixture.engines)); },
            "one shared CUDA stream", "device processor rejects mismatched streams");
    }
}

void testNonCurrentStreamDevice(std::int32_t device_count) {
    if (device_count < 2)
        return;
    std::int32_t previous = -1;
    checkCuda(cudaGetDevice(&previous), "current device query");
    try {
        {
            checkCuda(cudaSetDevice(1), "secondary device selection for invalid profile");
            auto invalid_profile = makeGpuEngines();
            invalid_profile.modules[0]->profile_count = 2;
            checkCuda(cudaSetDevice(0), "non-stream device selection for invalid profile");
            checkThrows<std::invalid_argument>(
                [&] {
                    (void)trtmc::sam2::makeNativeDeviceVideoProcessor(
                        std::move(invalid_profile.engines));
                },
                "exactly one static optimization profile",
                "multi-GPU constructor rejects an invalid profile safely");
            std::int32_t observed = -1;
            checkCuda(cudaGetDevice(&observed), "invalid-profile device query");
            check(observed == 0,
                  "invalid-profile multi-GPU construction restores the caller's device");
        }
        {
            checkCuda(cudaSetDevice(1), "secondary device selection for rejected binding");
            auto rejected = makeGpuEngines();
            rejected.modules[1]->ignored_binding = std::string(trtmc::sam2::kTrackerFpn[0].name);
            checkCuda(cudaSetDevice(0), "non-stream device selection for rejected binding");
            checkThrows<std::runtime_error>(
                [&] {
                    (void)trtmc::sam2::makeNativeDeviceVideoProcessor(std::move(rejected.engines));
                },
                "rejected external binding",
                "multi-GPU constructor rejects ignored external bindings safely");
            std::int32_t observed = -1;
            checkCuda(cudaGetDevice(&observed), "rejected-binding device query");
            check(observed == 0,
                  "rejected multi-GPU construction restores the caller's current device");
        }
        {
            checkCuda(cudaSetDevice(1), "secondary device selection for undersized storage");
            auto undersized = makeGpuEngines();
            DeviceBuffer tiny_storage(1U);
            undersized.modules[0]->tamperBinding(std::string(trtmc::sam2::kPixelValues.name),
                                                 tiny_storage.data());
            checkCuda(cudaSetDevice(0), "non-stream device selection for undersized storage");
            checkThrows<std::invalid_argument>(
                [&] {
                    (void)trtmc::sam2::makeNativeDeviceVideoProcessor(
                        std::move(undersized.engines));
                },
                "not CUDA device storage",
                "multi-GPU constructor rejects undersized storage safely");
            std::int32_t observed = -1;
            checkCuda(cudaGetDevice(&observed), "undersized-storage device query");
            check(observed == 0,
                  "undersized multi-GPU construction restores the caller's current device");
        }
        checkCuda(cudaSetDevice(1), "secondary device selection");
        auto fixture = makeGpuEngines();
        checkCuda(cudaSetDevice(0), "non-stream device selection");
        std::int32_t observed = -1;
        {
            auto processor =
                trtmc::sam2::makeNativeDeviceVideoProcessor(std::move(fixture.engines));
            checkCuda(cudaGetDevice(&observed), "post-construction device query");
            check(observed == 0, "device validation restores the caller's current device");

            auto frames = makeFrames();
            auto prompt = processor.run_bbox_prompt(frames.views);
            checkCuda(cudaGetDevice(&observed), "post-prompt device query");
            check(observed == 0, "device prompt restores the caller's current device");
            auto results = processor.propagate(prompt, frames.views);
            checkCuda(cudaGetDevice(&observed), "post-propagation device query");
            check(observed == 0, "device propagation restores the caller's current device");
            const auto mask_bytes = static_cast<std::size_t>(trtmc::sam2::kOriginalImageHeight) *
                                    trtmc::sam2::kOriginalImageWidth;
            (void)results.back().mask.materialize_host(mask_bytes);
            checkCuda(cudaGetDevice(&observed), "post-materialization device query");
            check(observed == 0,
                  "lazy device-mask materialization restores the caller's current device");

            processor.reset();
            checkCuda(cudaGetDevice(&observed), "post-reset device query");
            check(observed == 0, "device reset restores the caller's current device");
        }
        checkCuda(cudaGetDevice(&observed), "post-destruction device query");
        check(observed == 0, "device destruction restores the caller's current device");
        checkCuda(cudaSetDevice(previous), "original device restoration");
    } catch (...) {
        (void)cudaSetDevice(previous);
        throw;
    }
}

void checkHostMask(const trtmc::Sam2VideoFrameResultView& result, std::uint8_t expected) {
    const auto mask_bytes = static_cast<std::size_t>(trtmc::sam2::kOriginalImageHeight) *
                            trtmc::sam2::kOriginalImageWidth;
    const auto* mask = static_cast<const std::uint8_t*>(result.mask);
    check(result.mask_memory_kind == TRTMC_SAM2_VIDEO_MASK_MEMORY_HOST && mask != nullptr &&
              result.mask_byte_count == mask_bytes &&
              std::all_of(mask, mask + mask_bytes,
                          [expected](std::uint8_t value) { return value == expected; }),
          "lazy host mask content drifted");
}

void testDeviceSequenceBindingsLazyMasksAndReset() {
    auto fixture = makeGpuEngines();
    const auto modules = fixture.modules;
    auto processor = trtmc::sam2::makeNativeDeviceVideoProcessor(std::move(fixture.engines));
    auto frames = makeFrames();
    trtmc::Sam2VideoSegmentationSession session(std::move(processor));
    session.begin();
    for (const auto& frame : frames.views)
        session.append_frame(frame.pixels, frame.height, frame.width);

    session.run_bbox_prompt();
    check(modules[0]->async_forward_calls == 1 && modules[1]->async_forward_calls == 1 &&
              modules[0]->host_forward_calls == 0 && modules[1]->host_forward_calls == 0,
          "device prompt uses only asynchronous module execution");
    check(modules[1]->observed_box == std::array<float, 4>{572.0F, 640.0F, 652.0F, 708.0F},
          "device prompt retains the exact model-space box");
    const auto prompt_device = session.result(0, false);
    check(prompt_device.mask_memory_kind == TRTMC_SAM2_VIDEO_MASK_MEMORY_CUDA_DEVICE &&
              prompt_device.mask != nullptr,
          "default prompt result remains CUDA-device resident");
    checkHostMask(session.result(0, true), 1U);

    check(session.propagate() == trtmc::kSam2VideoFrameCount,
          "device propagation returns all five frames");
    check(modules[0]->async_forward_calls == 5,
          "device sequence runs the image module once per frame");
    for (std::size_t frame = 0; frame < trtmc::kSam2VideoFrameCount; ++frame) {
        check(modules[frame + 1U]->async_forward_calls == 1,
              "device sequence runs every tracker module once");
        const auto device = session.result(frame, false);
        check(device.mask_memory_kind == TRTMC_SAM2_VIDEO_MASK_MEMORY_CUDA_DEVICE,
              "propagated mask remains device resident by default");
        checkHostMask(session.result(frame, true), static_cast<std::uint8_t>(frame % 2U == 0U));
    }

    for (const auto& fpn : trtmc::sam2::kTrackerFpn) {
        const auto name = std::string(fpn.name);
        for (std::size_t tracker = 1U; tracker < modules.size(); ++tracker) {
            check(modules[tracker]->device_ptr(name) == modules[0]->device_ptr(name),
                  "all tracker FPN inputs share the image output address");
        }
    }
    const auto history_name = std::string(trtmc::sam2::historyMemoryFeatures(1).name);
    const auto pointer_name = std::string(trtmc::sam2::historyObjectPointers(1).name);
    void* const memory_base = modules[2]->device_ptr(history_name);
    void* const pointer_base = modules[2]->device_ptr(pointer_name);
    for (std::size_t recurrent = 2U; recurrent < modules.size(); ++recurrent) {
        check(modules[recurrent]->device_ptr(history_name) == memory_base &&
                  modules[recurrent]->device_ptr(pointer_name) == pointer_base,
              "all recurrent plans share contiguous history bases");
    }
    constexpr std::size_t kMemorySlotBytes = 64U * 64U * 64U * sizeof(std::uint16_t);
    constexpr std::size_t kPointerSlotBytes = 256U * sizeof(float);
    for (std::size_t producer = 1U; producer < 5U; ++producer) {
        const auto memory_address = reinterpret_cast<std::uintptr_t>(
            modules[producer]->device_ptr(std::string(trtmc::sam2::kMemoryFeatures.name)));
        const auto pointer_address = reinterpret_cast<std::uintptr_t>(
            modules[producer]->device_ptr(std::string(trtmc::sam2::kObjectPointer.name)));
        check(memory_address == reinterpret_cast<std::uintptr_t>(memory_base) +
                                    (producer - 1U) * kMemorySlotBytes &&
                  pointer_address == reinterpret_cast<std::uintptr_t>(pointer_base) +
                                         (producer - 1U) * kPointerSlotBytes,
              "history producer outputs occupy exact chronological slots");
    }

    session.reset();
    for (const auto* module : modules)
        check(module->reset_calls == 1, "device reset reaches all six modules");
    session.begin();
    for (const auto& frame : frames.views)
        session.append_frame(frame.pixels, frame.height, frame.width);
    session.run_bbox_prompt();
    check(session.result(0, false).mask_memory_kind == TRTMC_SAM2_VIDEO_MASK_MEMORY_CUDA_DEVICE,
          "device processor reuses its persistent bindings after reset");
}

void testBindingAndFiniteFailures() {
    {
        auto fixture = makeGpuEngines();
        fixture.modules[0]->profile_count = 2;
        checkThrows<std::invalid_argument>(
            [&] { (void)trtmc::sam2::makeNativeDeviceVideoProcessor(std::move(fixture.engines)); },
            "exactly one static optimization profile",
            "device processor rejects multiple optimization profiles");
    }
    {
        auto fixture = makeGpuEngines();
        fixture.modules[0]->active_profile = 1;
        checkThrows<std::invalid_argument>(
            [&] { (void)trtmc::sam2::makeNativeDeviceVideoProcessor(std::move(fixture.engines)); },
            "exactly one static optimization profile",
            "device processor rejects a nonzero active profile");
    }
    {
        auto fixture = makeGpuEngines();
        fixture.modules[0]->dynamic_input = std::string(trtmc::sam2::kPixelValues.name);
        checkThrows<std::invalid_argument>(
            [&] { (void)trtmc::sam2::makeNativeDeviceVideoProcessor(std::move(fixture.engines)); },
            "dynamic input", "device processor rejects a dynamic input ABI");
    }
    {
        auto fixture = makeGpuEngines();
        std::int32_t host_storage = 0;
        fixture.modules[0]->tamperBinding(std::string(trtmc::sam2::kPixelValues.name),
                                          &host_storage);
        checkThrows<std::invalid_argument>(
            [&] { (void)trtmc::sam2::makeNativeDeviceVideoProcessor(std::move(fixture.engines)); },
            "not CUDA device storage",
            "device processor rejects a tensor pointer outside the stream device");
    }
    {
        auto fixture = makeGpuEngines();
        DeviceBuffer undersized_storage(1U);
        fixture.modules[0]->tamperBinding(std::string(trtmc::sam2::kPixelValues.name),
                                          undersized_storage.data());
        checkThrows<std::invalid_argument>(
            [&] { (void)trtmc::sam2::makeNativeDeviceVideoProcessor(std::move(fixture.engines)); },
            "not CUDA device storage", "device processor rejects an undersized device allocation");
    }
    {
        auto fixture = makeGpuEngines();
        fixture.modules[1]->ignored_binding = std::string(trtmc::sam2::kTrackerFpn[0].name);
        checkThrows<std::runtime_error>(
            [&] { (void)trtmc::sam2::makeNativeDeviceVideoProcessor(std::move(fixture.engines)); },
            "rejected external binding", "device processor verifies accepted binding addresses");
    }
    {
        auto fixture = makeGpuEngines();
        const auto modules = fixture.modules;
        modules[1]->invalidate_mask_after_forward = true;
        auto processor = trtmc::sam2::makeNativeDeviceVideoProcessor(std::move(fixture.engines));
        auto frames = makeFrames();
        checkThrows<std::runtime_error>([&] { (void)processor.run_bbox_prompt(frames.views); },
                                        "mask postprocess launch",
                                        "device prompt drains a deferred box after later failure");
        check(modules[1]->deferred_box_observed &&
                  modules[1]->observed_box == std::array<float, 4>{572.0F, 640.0F, 652.0F, 708.0F},
              "deferred prompt-box consumption completes before input destruction");
    }
    {
        auto fixture = makeGpuEngines();
        const auto modules = fixture.modules;
        auto processor = trtmc::sam2::makeNativeDeviceVideoProcessor(std::move(fixture.engines));
        auto frames = makeFrames();
        modules[1]->setMaskNonFinite(true);
        checkThrows<std::runtime_error>([&] { (void)processor.run_bbox_prompt(frames.views); },
                                        "non-finite tracker output",
                                        "device processor rejects non-finite prompt output");
        processor.reset();
        modules[1]->setMaskNonFinite(false);
        const auto prompt = processor.run_bbox_prompt(frames.views);
        check(prompt.frame_zero.mask.memory_kind() == TRTMC_SAM2_VIDEO_MASK_MEMORY_CUDA_DEVICE,
              "device processor recovers after reset and finite output restoration");
    }
    {
        auto fixture = makeGpuEngines();
        const auto modules = fixture.modules;
        auto processor = trtmc::sam2::makeNativeDeviceVideoProcessor(std::move(fixture.engines));
        const auto fpn_name = std::string(trtmc::sam2::kTrackerFpn[0].name);
        modules[1]->tamperBinding(fpn_name, modules[1]->ownedPointer(fpn_name));
        auto frames = makeFrames();
        checkThrows<std::runtime_error>([&] { (void)processor.run_bbox_prompt(frames.views); },
                                        "FPN binding drifted",
                                        "device processor detects post-construction binding drift");
    }
    {
        auto fixture = makeGpuEngines();
        auto processor = trtmc::sam2::makeNativeDeviceVideoProcessor(std::move(fixture.engines));
        auto frames = makeFrames();
        auto prompt = processor.run_bbox_prompt(frames.views);

        trtmc::Sam2VideoPromptResult forged;
        forged.track = prompt.track;
        forged.frame_zero.frame_index = prompt.frame_zero.frame_index;
        forged.frame_zero.height = prompt.frame_zero.height;
        forged.frame_zero.width = prompt.frame_zero.width;
        const auto mask_bytes = prompt.frame_zero.mask.byte_count();
        forged.frame_zero.mask = trtmc::Sam2VideoMaskBuffer::cuda_device_binary(
            prompt.frame_zero.mask.data(), mask_bytes, prompt.frame_zero.mask.device_ordinal(),
            std::make_shared<std::int32_t>(7),
            [mask_bytes] { return std::vector<std::uint8_t>(mask_bytes, 1U); });
        checkThrows<std::invalid_argument>(
            [&] { (void)processor.propagate(forged, frames.views); },
            "foreign or modified prompt result",
            "device processor rejects a same-pointer mask with a foreign owner");
    }
    {
        auto fixture = makeGpuEngines();
        auto processor = trtmc::sam2::makeNativeDeviceVideoProcessor(std::move(fixture.engines));
        auto frames = makeFrames();
        auto stale_prompt = processor.run_bbox_prompt(frames.views);
        processor.reset();
        auto current_prompt = processor.run_bbox_prompt(frames.views);
        check(current_prompt.frame_zero.mask.memory_kind() ==
                  TRTMC_SAM2_VIDEO_MASK_MEMORY_CUDA_DEVICE,
              "device processor produces a new provenance after reset");
        checkThrows<std::invalid_argument>(
            [&] { (void)processor.propagate(stale_prompt, frames.views); },
            "foreign or modified prompt result",
            "device processor rejects a prompt provenance from before reset");
    }
}

} // namespace

int main() {
    try {
        testStreamGateWithoutCuda();
        std::int32_t device_count = 0;
        const auto probe = cudaGetDeviceCount(&device_count);
        if (probe != cudaSuccess || device_count <= 0) {
            std::cout << "SKIP: CUDA device unavailable after SAM2 stream-gate tests\n";
            return 0;
        }
        testNonCurrentStreamDevice(device_count);
        testDeviceSequenceBindingsLazyMasksAndReset();
        testBindingAndFiniteFailures();
        std::cout << "PASS: SAM2 native device video processor\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
