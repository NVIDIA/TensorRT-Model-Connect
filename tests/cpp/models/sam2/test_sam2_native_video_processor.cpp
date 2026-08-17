/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_engine_contract.h"
#include "runtime/models/sam2/sam2_native_video_processor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

template <typename Exception, typename Function>
void checkThrows(Function&& function, const char* needle, const char* message) {
    static_assert(std::is_base_of<std::exception, Exception>::value,
                  "test exception must derive from std::exception");
    try {
        function();
    } catch (const Exception& error) {
        if (std::strstr(error.what(), needle) != nullptr)
            return;
        std::cerr << "FAIL: " << message << " (wrong message '" << error.what() << "')\n";
        std::exit(1);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << message << " (wrong exception '" << error.what() << "')\n";
        std::exit(1);
    }
    std::cerr << "FAIL: " << message << " (no exception)\n";
    std::exit(1);
}

trtmc::DType dtype(trtmc::sam2::TensorDataType data_type) {
    return data_type == trtmc::sam2::TensorDataType::kFloat32 ? trtmc::DType::kFloat32
                                                              : trtmc::DType::kBFloat16;
}

std::vector<int64_t> shape(const trtmc::sam2::TensorContract& contract) {
    std::vector<int64_t> result;
    for (std::uint8_t index = 0; index < contract.rank; ++index)
        result.push_back(contract.dimensions[index]);
    return result;
}

std::size_t elements(const trtmc::sam2::TensorContract& contract) {
    std::size_t result = 1;
    for (const auto dimension : shape(contract))
        result *= static_cast<std::size_t>(dimension);
    return result;
}

using Contracts = std::vector<trtmc::sam2::TensorContract>;

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
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += UINT32_C(0x7FFF) + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

class FakeModule final : public trtmc::ITrtModule {
  public:
    enum class Kind { kImage, kTracker };

    FakeModule(Contracts inputs, Contracts outputs, Kind kind, std::int32_t tracker_ordinal = -1)
        : inputs_(std::move(inputs)), outputs_(std::move(outputs)), kind_(kind),
          tracker_ordinal_(tracker_ordinal) {
        if (kind_ == Kind::kImage)
            initializeImageStorage();
        else
            initializeTrackerStorage();
    }

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        validateInputs(inputs);
        ++forward_calls;
        if (kind_ == Kind::kImage)
            return imageOutputMap();
        recordTrackerInputs(inputs);
        auto result = trackerOutputMap();
        if (!omitted_output.empty())
            result.erase(omitted_output);
        return result;
    }

    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap& /*inputs*/) override {
        return {};
    }
    void forward_device_async(const trtmc::DeviceTensorMap& /*inputs*/) override {}
    void forward_async(const trtmc::TensorMap& /*inputs*/) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }

    std::vector<trtmc::TensorInfo> input_info() const override { return metadata(inputs_, true); }
    std::vector<trtmc::TensorInfo> output_info() const override {
        auto result = metadata(outputs_, false);
        if (!metadata_shape_drift.empty()) {
            for (auto& info : result) {
                if (info.name == metadata_shape_drift)
                    info.shape.back() -= 1;
            }
        }
        return result;
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
        return contract == nullptr ? trtmc::DType::kFloat32 : dtype(contract->data_type);
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        const auto* contract = find(inputs_, name);
        if (contract == nullptr)
            contract = find(outputs_, name);
        return contract == nullptr ? std::vector<int64_t>{} : shape(*contract);
    }
    std::vector<int64_t>
    input_profile_shape(const std::string& name, int32_t /*profile_idx*/,
                        trtmc::ProfileShapeSelector /*selector*/) const override {
        return tensor_shape(name);
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string& /*name*/) const override { return nullptr; }
    void bind_external(const std::string& /*name*/, void* /*ptr*/) override {}
    bool ok() const override { return ready; }
    void keep_alive(std::shared_ptr<void> resource) override { keep_alive_ = std::move(resource); }
    void reset_execution_context() override {
        ++reset_calls;
        if (throw_on_reset)
            throw std::runtime_error("synthetic reset failure");
    }

    void overwriteTrackerState(float pointer_value, float memory_value) {
        std::fill(object_pointer_.begin(), object_pointer_.end(), pointer_value);
        std::fill(memory_features_.begin(), memory_features_.end(), bfloat16(memory_value));
    }

    bool ready{true};
    bool throw_on_reset{false};
    std::string omitted_output;
    std::string metadata_shape_drift;
    std::int32_t forward_calls{0};
    std::int32_t reset_calls{0};
    std::int32_t observed_history{-1};
    std::vector<float> observed_pointer_history;
    std::vector<std::uint16_t> observed_memory_history;
    std::array<float, 4> observed_box{};

  private:
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

    void validateInputs(const trtmc::TensorMap& inputs) const {
        if (inputs.size() != inputs_.size())
            throw std::runtime_error("fake SAM2 module input count drifted");
        for (const auto& contract : inputs_) {
            const auto found = inputs.find(std::string(contract.name));
            if (found == inputs.end() || found->second.data == nullptr ||
                found->second.shape != shape(contract) ||
                found->second.dtype != dtype(contract.data_type)) {
                throw std::runtime_error("fake SAM2 module input contract drifted");
            }
        }
    }

    void initializeImageStorage() {
        fpn_0_.resize(elements(trtmc::sam2::kTrackerFpn[0]), bfloat16(1.0F));
        fpn_1_.resize(elements(trtmc::sam2::kTrackerFpn[1]), bfloat16(2.0F));
        fpn_2_.resize(elements(trtmc::sam2::kTrackerFpn[2]), 3.0F);
        constexpr std::array<std::int32_t, 3> sizes = {128, 64, 32};
        for (std::size_t level = 0; level < sizes.size(); ++level) {
            const auto area = static_cast<std::size_t>(sizes[level]) * sizes[level];
            bbox_cls_[level].resize(2U * area, bfloat16(-20.0F));
            bbox_reg_[level].resize(4U * area, bfloat16(0.0F));
        }
        const std::size_t area = 128U * 128U;
        const std::size_t anchor = 84U * 128U + 76U;
        bbox_cls_[0][area + anchor] = bfloat16(4.0F);
        constexpr std::array<float, 4> distances = {5.0F, 4.5F, 5.0F, 4.0F};
        for (std::size_t coordinate = 0; coordinate < distances.size(); ++coordinate)
            bbox_reg_[0][coordinate * area + anchor] = bfloat16(distances[coordinate]);
    }

    void initializeTrackerStorage() {
        const float mask_value = tracker_ordinal_ % 2 == 0 ? 1.0F : -1.0F;
        mask_logits_.resize(elements(trtmc::sam2::kMaskLogits256), mask_value);
        object_pointer_.resize(elements(trtmc::sam2::kObjectPointer),
                               10.0F + static_cast<float>(tracker_ordinal_));
        memory_features_.resize(elements(trtmc::sam2::kMemoryFeatures),
                                bfloat16(1.0F + static_cast<float>(tracker_ordinal_)));
    }

    trtmc::TensorMap imageOutputMap() {
        trtmc::TensorMap result;
        result.emplace(std::string(trtmc::sam2::kTrackerFpn[0].name),
                       trtmc::Tensor{fpn_0_.data(), shape(trtmc::sam2::kTrackerFpn[0]),
                                     trtmc::DType::kBFloat16});
        result.emplace(std::string(trtmc::sam2::kTrackerFpn[1].name),
                       trtmc::Tensor{fpn_1_.data(), shape(trtmc::sam2::kTrackerFpn[1]),
                                     trtmc::DType::kBFloat16});
        result.emplace(std::string(trtmc::sam2::kTrackerFpn[2].name),
                       trtmc::Tensor{fpn_2_.data(), shape(trtmc::sam2::kTrackerFpn[2]),
                                     trtmc::DType::kFloat32});
        for (std::size_t level = 0; level < 3; ++level) {
            const auto& cls_contract = trtmc::sam2::kBboxMaps[level];
            const auto& reg_contract = trtmc::sam2::kBboxMaps[level + 3U];
            result.emplace(std::string(cls_contract.name),
                           trtmc::Tensor{bbox_cls_[level].data(), shape(cls_contract),
                                         trtmc::DType::kBFloat16});
            result.emplace(std::string(reg_contract.name),
                           trtmc::Tensor{bbox_reg_[level].data(), shape(reg_contract),
                                         trtmc::DType::kBFloat16});
        }
        return result;
    }

    trtmc::TensorMap trackerOutputMap() {
        return {
            {std::string(trtmc::sam2::kMaskLogits256.name),
             {mask_logits_.data(), shape(trtmc::sam2::kMaskLogits256), trtmc::DType::kFloat32}},
            {std::string(trtmc::sam2::kObjectPointer.name),
             {object_pointer_.data(), shape(trtmc::sam2::kObjectPointer), trtmc::DType::kFloat32}},
            {std::string(trtmc::sam2::kMemoryFeatures.name),
             {memory_features_.data(), shape(trtmc::sam2::kMemoryFeatures),
              trtmc::DType::kBFloat16}},
        };
    }

    void recordTrackerInputs(const trtmc::TensorMap& inputs) {
        const auto box = inputs.find(std::string(trtmc::sam2::kBoxPrompt.name));
        if (box != inputs.end()) {
            const auto* values = static_cast<const float*>(box->second.data);
            std::copy_n(values, observed_box.size(), observed_box.begin());
            return;
        }
        const auto history = inputs.find(std::string(trtmc::sam2::historyObjectPointers(1).name));
        const auto memory = inputs.find(std::string(trtmc::sam2::historyMemoryFeatures(1).name));
        if (history == inputs.end() || memory == inputs.end())
            throw std::runtime_error("fake recurrent module did not receive complete history");
        observed_history = static_cast<std::int32_t>(history->second.shape.front());
        const auto* values = static_cast<const float*>(history->second.data);
        const auto* memory_values = static_cast<const std::uint16_t*>(memory->second.data);
        observed_pointer_history.reserve(static_cast<std::size_t>(observed_history));
        observed_memory_history.reserve(static_cast<std::size_t>(observed_history));
        for (std::int32_t index = 0; index < observed_history; ++index) {
            observed_pointer_history.push_back(values[static_cast<std::size_t>(index) * 256U]);
            observed_memory_history.push_back(
                memory_values[static_cast<std::size_t>(index) * 64U * 64U * 64U]);
        }
    }

    Contracts inputs_;
    Contracts outputs_;
    Kind kind_;
    std::int32_t tracker_ordinal_;
    std::vector<std::uint16_t> fpn_0_;
    std::vector<std::uint16_t> fpn_1_;
    std::vector<float> fpn_2_;
    std::array<std::vector<std::uint16_t>, 3> bbox_cls_;
    std::array<std::vector<std::uint16_t>, 3> bbox_reg_;
    std::vector<float> mask_logits_;
    std::vector<float> object_pointer_;
    std::vector<std::uint16_t> memory_features_;
    std::shared_ptr<void> keep_alive_;
};

struct EngineFixture {
    trtmc::sam2::NativeVideoEngineSet engines;
    std::array<FakeModule*, 6> modules{};
};

EngineFixture makeEngines() {
    EngineFixture fixture;
    auto image =
        std::make_unique<FakeModule>(imageInputs(), imageOutputs(), FakeModule::Kind::kImage);
    fixture.modules[0] = image.get();
    fixture.engines.image = std::move(image);

    auto prompt = std::make_unique<FakeModule>(promptInputs(), trackerOutputs(),
                                               FakeModule::Kind::kTracker, 0);
    fixture.modules[1] = prompt.get();
    fixture.engines.prompt = std::move(prompt);
    for (std::size_t index = 0; index < fixture.engines.recurrent.size(); ++index) {
        auto recurrent = std::make_unique<FakeModule>(
            recurrentInputs(static_cast<std::int32_t>(index + 1U)), trackerOutputs(),
            FakeModule::Kind::kTracker, static_cast<std::int32_t>(index + 1U));
        fixture.modules[index + 2U] = recurrent.get();
        fixture.engines.recurrent[index] = std::move(recurrent);
    }
    return fixture;
}

struct FrameFixture {
    std::vector<float> pixels;
    trtmc::Sam2VideoFrames views{};
};

FrameFixture makeFrames() {
    FrameFixture fixture;
    const auto elements = static_cast<std::size_t>(trtmc::sam2::kOriginalImageHeight) *
                          trtmc::sam2::kOriginalImageWidth * 3U;
    fixture.pixels.resize(elements, 0.5F);
    for (std::size_t index = 0; index < fixture.views.size(); ++index) {
        fixture.views[index] = {static_cast<std::int32_t>(index), trtmc::sam2::kOriginalImageHeight,
                                trtmc::sam2::kOriginalImageWidth, fixture.pixels.data(), elements};
    }
    return fixture;
}

void checkMask(const trtmc::Sam2VideoFrameResultView& result, std::uint8_t expected,
               const char* message) {
    const auto area = static_cast<std::size_t>(trtmc::sam2::kOriginalImageHeight) *
                      trtmc::sam2::kOriginalImageWidth;
    const auto* values = static_cast<const std::uint8_t*>(result.mask);
    check(result.height == trtmc::sam2::kOriginalImageHeight &&
              result.width == trtmc::sam2::kOriginalImageWidth &&
              result.mask_memory_kind == TRTMC_SAM2_VIDEO_MASK_MEMORY_HOST &&
              result.mask_byte_count == area && values != nullptr &&
              std::all_of(values, values + area,
                          [expected](std::uint8_t value) { return value == expected; }),
          message);
}

void testPromptH1ThroughH4HistoryAndReset() {
    auto fixture = makeEngines();
    const auto modules = fixture.modules;
    auto processor = trtmc::sam2::makeNativeVideoProcessor(std::move(fixture.engines));
    auto frames = makeFrames();
    trtmc::Sam2VideoSegmentationSession session(std::move(processor));
    session.begin();
    for (const auto& frame : frames.views)
        session.append_frame(frame.pixels, frame.height, frame.width);

    session.run_bbox_prompt();
    check(modules[0]->forward_calls == 1 && modules[1]->forward_calls == 1,
          "sam2 prompt runs image and prompt plans exactly once");
    const auto& track = session.track();
    check(track.label == 1 && track.detector_score > 0.9F &&
              track.prompt_box_xyxy == std::array<float, 4>{607.75F, 800.0F, 692.75F, 885.0F},
          "sam2 prompt retains the selected original-space detector result");
    check(modules[1]->observed_box == std::array<float, 4>{572.0F, 640.0F, 652.0F, 708.0F},
          "sam2 prompt engine receives the unclipped model-space box");
    checkMask(session.result(0, true), 1,
              "sam2 prompt postprocesses the frame-zero mask through the session seam");
    modules[1]->overwriteTrackerState(99.0F, 99.0F);

    check(session.propagate() == trtmc::kSam2VideoFrameCount,
          "sam2 session exposes all five native propagated results");
    check(modules[0]->forward_calls == 5,
          "sam2 native propagation runs the image plan once per frame");
    checkMask(session.result(0, true), 1,
              "sam2 propagation preserves frame zero from the prompt result");
    for (std::size_t index = 1; index < trtmc::kSam2VideoFrameCount; ++index) {
        check(modules[index + 1U]->forward_calls == 1 &&
                  modules[index + 1U]->observed_history == static_cast<std::int32_t>(index),
              "sam2 propagation selects the matching fixed-history recurrent plan");
        const auto expected_mask = static_cast<std::uint8_t>((index % 2U) == 0U);
        checkMask(session.result(index, true), expected_mask,
                  "sam2 recurrent plan mask is postprocessed for its original frame");
        check(modules[index + 1U]->observed_pointer_history.size() == index,
              "sam2 recurrent input contains all prior object pointers");
        for (std::size_t history = 0; history < index; ++history) {
            check(modules[index + 1U]->observed_pointer_history[history] ==
                      10.0F + static_cast<float>(history),
                  "sam2 recurrent history remains chronological and deep copied");
            check(modules[index + 1U]->observed_memory_history[history] ==
                      bfloat16(1.0F + static_cast<float>(history)),
                  "sam2 recurrent memory history remains chronological and deep copied");
        }
    }

    checkThrows<trtmc::Sam2VideoStateError>([&] { (void)session.propagate(); }, "requires one",
                                            "sam2 propagation is one-shot before reset");
    session.reset();
    for (const auto* module : modules)
        check(module->reset_calls == 1, "sam2 reset reaches every owned TensorRT module");
}

void testExactMetadataAndReturnedOutputsFailClosed() {
    {
        auto fixture = makeEngines();
        fixture.modules[1]->metadata_shape_drift = std::string(trtmc::sam2::kMaskLogits256.name);
        checkThrows<std::invalid_argument>(
            [&] { (void)trtmc::sam2::makeNativeVideoProcessor(std::move(fixture.engines)); },
            "metadata drifted", "sam2 constructor rejects tracker metadata shape drift");
    }
    {
        trtmc::sam2::NativeVideoEngineSet missing;
        checkThrows<std::invalid_argument>(
            [&] { (void)trtmc::sam2::makeNativeVideoProcessor(std::move(missing)); }, "missing",
            "sam2 constructor rejects a missing native plan");
    }
    {
        auto fixture = makeEngines();
        const auto modules = fixture.modules;
        modules[1]->omitted_output = std::string(trtmc::sam2::kMaskLogits256.name);
        auto processor = trtmc::sam2::makeNativeVideoProcessor(std::move(fixture.engines));
        auto frames = makeFrames();
        checkThrows<std::runtime_error>([&] { (void)processor.run_bbox_prompt(frames.views); },
                                        "wrong output count",
                                        "sam2 prompt rejects an incomplete returned TensorMap");
        checkThrows<std::logic_error>([&] { (void)processor.run_bbox_prompt(frames.views); },
                                      "requires reset",
                                      "sam2 processor stays failed after consumed inference");
        processor.reset();
        for (const auto* module : modules)
            check(module->reset_calls == 1,
                  "sam2 failed processor reset still reaches all six modules");
    }
    {
        auto fixture = makeEngines();
        const auto modules = fixture.modules;
        modules[1]->throw_on_reset = true;
        auto processor = trtmc::sam2::makeNativeVideoProcessor(std::move(fixture.engines));
        checkThrows<std::runtime_error>([&] { processor.reset(); }, "synthetic reset failure",
                                        "sam2 reset surfaces the first module failure");
        for (const auto* module : modules)
            check(module->reset_calls == 1,
                  "sam2 reset visits all six modules even after one reset failure");
        modules[1]->throw_on_reset = false;
        processor.reset();
        for (const auto* module : modules)
            check(module->reset_calls == 2,
                  "sam2 processor recovers only after a complete successful reset");
    }
}

} // namespace

int main() {
    testPromptH1ThroughH4HistoryAndReset();
    testExactMetadataAndReturnedOutputsFailClosed();
    std::cout << "SAM2 native video processor tests passed\n";
    return 0;
}
