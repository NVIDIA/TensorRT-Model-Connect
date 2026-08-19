/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-SAM2-HOI-CPP-03
// Architecture:   ARCH-MODPLUG-001
// Unit Design:    UD-SAM2-HOI-03
// Intent:         SAM2 HOI split-engine prebinding and one-frame tracking orchestration
// Preconditions:  Six valid module contracts and one decoded RGB frame
// Postconditions: Producer buffers are shared directly and two masks are serialized
// =============================================================================

#include "runtime/models/sam2_hoi/hoi_postprocess.h"
#include "runtime/models/sam2_hoi/jpeg_decoder.h"
#include "runtime/models/sam2_hoi/ordered_async_mask_postprocessor.h"
#include "runtime/models/sam2_hoi/pafpn_composite.h"
#include "runtime/models/sam2_hoi/pipeline.h"
#include "runtime/models/sam2_hoi/rolling_async_preprocessor.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using trtmc::DType;
using trtmc::ITrtModule;
using trtmc::ProfileShapeSelector;
using trtmc::Tensor;
using trtmc::TensorInfo;
using trtmc::TensorMap;

constexpr int32_t kImageSize = 1024;
constexpr int32_t kObjectBatch = 2;
constexpr int32_t kMaskSize = 256;
constexpr int32_t kMemoryChannels = 64;
constexpr int32_t kMemorySize = 64;
constexpr int32_t kPointerWidth = 256;

int g_failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++g_failures;
    }
}

struct TensorContract {
    std::vector<int64_t> shape;
    DType dtype{DType::kFloat32};
};

using ContractMap = std::unordered_map<std::string, TensorContract>;

class FakeModule final : public ITrtModule {
  public:
    struct ExternalBinding {
        void* pointer{nullptr};
        std::vector<int64_t> shape;
    };

    using Forward = std::function<TensorMap(const TensorMap&)>;

    FakeModule(ContractMap inputs, ContractMap outputs, Forward forward = {},
               std::vector<std::string>* events = nullptr, std::string event_name = {})
        : inputs_(std::move(inputs)), outputs_(std::move(outputs)), forward_(std::move(forward)),
          events_(events), event_name_(std::move(event_name)) {
        for (const auto& [name, contract] : outputs_) {
            (void)contract;
            output_anchors_.emplace(name, std::make_unique<std::uint8_t>(0));
        }
    }

    TensorMap forward(const TensorMap& inputs) override {
        record_event("forward");
        ++forward_count;
        last_forward_shapes.clear();
        last_forward_dtypes.clear();
        for (const auto& [name, tensor] : inputs) {
            last_forward_shapes[name] = tensor.shape;
            last_forward_dtypes[name] = tensor.dtype;
        }
        return forward_ ? forward_(inputs) : TensorMap{};
    }

    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}

    void forward_async(const TensorMap& inputs) override {
        record_event("forward_async");
        ++forward_async_count;
        last_async_shapes.clear();
        last_async_dtypes.clear();
        for (const auto& [name, tensor] : inputs) {
            last_async_shapes[name] = tensor.shape;
            last_async_dtypes[name] = tensor.dtype;
        }
    }

    void sync() override {
        record_event("sync");
        ++sync_count;
    }
    cudaStream_t stream() const override {
        return reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(1));
    }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }

    std::vector<TensorInfo> input_info() const override { return tensor_info(inputs_, true); }
    std::vector<TensorInfo> output_info() const override { return tensor_info(outputs_, false); }
    bool has_input(const std::string& name) const override { return inputs_.count(name) != 0; }
    bool has_output(const std::string& name) const override { return outputs_.count(name) != 0; }
    DType tensor_dtype(const std::string& name) const override { return contract(name).dtype; }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        return contract(name).shape;
    }
    std::vector<int64_t> input_profile_shape(const std::string& name, int32_t,
                                             ProfileShapeSelector) const override {
        return contract(name).shape;
    }
    int32_t optimization_profile_count() const override { return 1; }

    void* device_ptr(const std::string& name) const override {
        const auto binding = external_bindings_.find(name);
        if (binding != external_bindings_.end())
            return binding->second.pointer;
        const auto output = output_anchors_.find(name);
        return output == output_anchors_.end() ? nullptr : output->second.get();
    }

    void bind_external(const std::string& name, void* pointer) override {
        bind_external(name, pointer, tensor_shape(name));
    }

    void bind_external(const std::string& name, void* pointer,
                       const std::vector<int64_t>& shape) override {
        if (!has_input(name))
            throw std::runtime_error("fake module cannot bind unknown input " + name);
        external_bindings_[name] = {pointer, shape};
    }

    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

    const ExternalBinding* external_binding(const std::string& name) const {
        const auto binding = external_bindings_.find(name);
        return binding == external_bindings_.end() ? nullptr : &binding->second;
    }

    std::size_t external_binding_count() const { return external_bindings_.size(); }

    int forward_count{0};
    int forward_async_count{0};
    int sync_count{0};
    std::unordered_map<std::string, std::vector<int64_t>> last_forward_shapes;
    std::unordered_map<std::string, DType> last_forward_dtypes;
    std::unordered_map<std::string, std::vector<int64_t>> last_async_shapes;
    std::unordered_map<std::string, DType> last_async_dtypes;

  private:
    void record_event(const char* action) {
        if (events_ != nullptr)
            events_->push_back(event_name_ + ":" + action);
    }

    static std::vector<TensorInfo> tensor_info(const ContractMap& contracts, bool is_input) {
        std::vector<TensorInfo> info;
        info.reserve(contracts.size());
        for (const auto& [name, value] : contracts)
            info.push_back({name, value.shape, value.dtype, is_input});
        return info;
    }

    const TensorContract& contract(const std::string& name) const {
        const auto input = inputs_.find(name);
        if (input != inputs_.end())
            return input->second;
        const auto output = outputs_.find(name);
        if (output != outputs_.end())
            return output->second;
        throw std::out_of_range("fake module has no tensor named " + name);
    }

    ContractMap inputs_;
    ContractMap outputs_;
    Forward forward_;
    std::vector<std::string>* events_{nullptr};
    std::string event_name_;
    std::unordered_map<std::string, std::unique_ptr<std::uint8_t>> output_anchors_;
    std::unordered_map<std::string, ExternalBinding> external_bindings_;
};

class FakePafpnComposite final : public trtmc::sam2_hoi::IPafpnComposite {
  public:
    void bind_external_input(const std::string& composite_name, ITrtModule& producer,
                             const std::string& producer_output) override {
        if (producer.stream() != stream() || producer.device_ptr(producer_output) == nullptr)
            throw std::runtime_error("fake PAFPN external binding is invalid");
        roots.emplace_back(composite_name, producer_output);
    }

    void bind_output_to(const std::string& composite_name, ITrtModule& consumer,
                        const std::string& consumer_input) override {
        if (consumer.stream() != stream())
            throw std::runtime_error("fake PAFPN detector stream mismatch");
        auto& anchor = outputs[composite_name];
        if (!anchor)
            anchor = std::make_unique<std::uint8_t>(0);
        consumer.bind_external(consumer_input, anchor.get(), consumer.tensor_shape(consumer_input));
        detector_bindings.emplace_back(composite_name, consumer_input);
    }

    void forward_async() override { ++forward_async_count; }
    void sync() override { ++sync_count; }
    cudaStream_t stream() const override {
        return reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(1));
    }

    int forward_async_count{0};
    int sync_count{0};
    std::vector<std::pair<std::string, std::string>> roots;
    std::vector<std::pair<std::string, std::string>> detector_bindings;

  private:
    std::unordered_map<std::string, std::unique_ptr<std::uint8_t>> outputs;
};

TensorContract f32(std::vector<int64_t> shape) {
    return {std::move(shape), DType::kFloat32};
}

TensorContract i32(std::vector<int64_t> shape) {
    return {std::move(shape), DType::kInt32};
}

ContractMap image_outputs() {
    return {
        {"fpn_input_0", f32({1})},
        {"fpn_input_2", f32({1})},
        {"tracker_feature_0", f32({1, 32, 256, 256})},
        {"tracker_feature_1", f32({1, 64, 128, 128})},
        {"tracker_feature_2", f32({1, 256, 64, 64})},
        {"tracker_position_2", f32({1, 256, 64, 64})},
    };
}

ContractMap detector_inputs() {
    return {
        {"detector_feature_0", f32({1, 256, 128, 128})},
        {"detector_feature_1", f32({1, 256, 64, 64})},
        {"detector_feature_2", f32({1, 256, 32, 32})},
    };
}

ContractMap feature_inputs(const ContractMap& outputs, const std::vector<std::string>& names) {
    ContractMap inputs;
    for (const auto& name : names)
        inputs.emplace(name, outputs.at(name));
    return inputs;
}

void set_box(std::vector<float>& boxes, std::size_t query, const std::array<float, 4>& xyxy) {
    const std::size_t offset = query * 4;
    boxes[offset] = (xyxy[0] + xyxy[2]) * 0.5F / static_cast<float>(kImageSize);
    boxes[offset + 1] = (xyxy[1] + xyxy[3]) * 0.5F / static_cast<float>(kImageSize);
    boxes[offset + 2] = (xyxy[2] - xyxy[0]) / static_cast<float>(kImageSize);
    boxes[offset + 3] = (xyxy[3] - xyxy[1]) / static_cast<float>(kImageSize);
}

void check_direct_binding(const FakeModule& producer, const std::string& output_name,
                          const FakeModule& consumer, const std::string& input_name,
                          const char* label) {
    const auto* binding = consumer.external_binding(input_name);
    check(binding != nullptr, label);
    if (binding == nullptr)
        return;
    check(binding->pointer == producer.device_ptr(output_name), label);
    check(binding->shape == producer.tensor_shape(output_name), label);
}

struct TemporaryDirectory {
    TemporaryDirectory() {
        const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
               ("trtmc_sam2_hoi_pipeline_" + std::to_string(suffix));
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }

    std::filesystem::path path;
};

void update_maximum(std::atomic<int>& maximum, int value) {
    int observed = maximum.load(std::memory_order_relaxed);
    while (observed < value &&
           !maximum.compare_exchange_weak(observed, value, std::memory_order_relaxed)) {
    }
}

void test_rolling_preprocess_is_bounded_and_ordered() {
    std::atomic<int> active{0};
    std::atomic<int> maximum_active{0};
    std::atomic<std::size_t> completed{0U};
    std::mutex gate_mutex;
    std::condition_variable gate_condition;
    std::size_t gate_arrivals = 0U;
    bool gate_open = false;

    trtmc::sam2_hoi::RollingAsyncPreprocessor scheduler(
        7U, trtmc::sam2_hoi::kMaxConcurrentPreprocessTasks, [&](std::size_t index) {
            const int active_now = active.fetch_add(1, std::memory_order_relaxed) + 1;
            update_maximum(maximum_active, active_now);
            {
                std::unique_lock<std::mutex> lock(gate_mutex);
                if (!gate_open) {
                    ++gate_arrivals;
                    if (gate_arrivals == trtmc::sam2_hoi::kMaxConcurrentPreprocessTasks) {
                        gate_open = true;
                        gate_condition.notify_all();
                    } else if (!gate_condition.wait_for(lock, std::chrono::seconds(2),
                                                        [&]() { return gate_open; })) {
                        gate_open = true;
                        gate_condition.notify_all();
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            completed.fetch_add(1U, std::memory_order_relaxed);
            active.fetch_sub(1, std::memory_order_relaxed);
            return std::vector<float>{static_cast<float>(index)};
        });

    std::vector<std::size_t> observed_order;
    while (!scheduler.empty()) {
        const auto output = scheduler.take_next();
        observed_order.push_back(static_cast<std::size_t>(output.at(0)));
    }
    check(observed_order == std::vector<std::size_t>({0U, 1U, 2U, 3U, 4U, 5U, 6U}),
          "rolling preprocess returns exact input order");
    check(maximum_active.load(std::memory_order_relaxed) ==
              static_cast<int>(trtmc::sam2_hoi::kMaxConcurrentPreprocessTasks),
          "rolling preprocess enforces its five-task concurrency bound");
    check(completed.load(std::memory_order_relaxed) == 7U,
          "rolling preprocess completes every requested input");
}

void test_rolling_preprocess_joins_before_lowest_failure() {
    std::atomic<std::size_t> completed{0U};
    trtmc::sam2_hoi::RollingAsyncPreprocessor scheduler(
        5U, trtmc::sam2_hoi::kMaxConcurrentPreprocessTasks, [&](std::size_t index) {
            if (index == 0U)
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            else if (index == 1U)
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            completed.fetch_add(1U, std::memory_order_relaxed);
            if (index == 1U || index == 2U)
                throw std::runtime_error("preprocess failure " + std::to_string(index));
            return std::vector<float>{static_cast<float>(index)};
        });

    const auto first = scheduler.take_next();
    check(first == std::vector<float>{0.0F},
          "rolling preprocess returns the first success before later failures");
    bool reported_lowest = false;
    try {
        (void)scheduler.take_next();
    } catch (const std::runtime_error& error) {
        reported_lowest = std::string(error.what()) == "preprocess failure 1";
    }
    check(reported_lowest, "rolling preprocess reports the lowest-index failure");
    check(completed.load(std::memory_order_relaxed) == 5U,
          "rolling preprocess joins every launched task before throwing");
}

void test_rolling_preprocess_destructor_joins_pending_tasks() {
    std::atomic<std::size_t> completed{0U};
    {
        trtmc::sam2_hoi::RollingAsyncPreprocessor scheduler(
            7U, trtmc::sam2_hoi::kMaxConcurrentPreprocessTasks, [&](std::size_t index) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                completed.fetch_add(1U, std::memory_order_relaxed);
                return std::vector<float>{static_cast<float>(index)};
            });
        check(scheduler.max_in_flight() == trtmc::sam2_hoi::kMaxConcurrentPreprocessTasks,
              "rolling preprocess exposes its configured memory bound");
    }
    check(completed.load(std::memory_order_relaxed) == 5U,
          "rolling preprocess destructor joins all initially launched tasks");
}

void test_rolling_preprocess_falls_back_when_thread_launch_fails() {
    std::atomic<std::size_t> launch_attempts{0U};
    trtmc::sam2_hoi::RollingAsyncPreprocessor::Launcher unavailable =
        [&](const trtmc::sam2_hoi::RollingAsyncPreprocessor::Callback&,
            std::size_t) -> std::future<std::vector<float>> {
        launch_attempts.fetch_add(1U, std::memory_order_relaxed);
        throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));
    };
    trtmc::sam2_hoi::RollingAsyncPreprocessor scheduler(
        3U, trtmc::sam2_hoi::kMaxConcurrentPreprocessTasks,
        [](std::size_t index) { return std::vector<float>{static_cast<float>(index)}; },
        std::move(unavailable));

    std::vector<std::size_t> observed;
    while (!scheduler.empty())
        observed.push_back(static_cast<std::size_t>(scheduler.take_next().at(0)));
    check(observed == std::vector<std::size_t>({0U, 1U, 2U}),
          "rolling preprocess preserves results through deferred launch fallback");
    check(launch_attempts.load(std::memory_order_relaxed) == 3U,
          "rolling preprocess attempts async launch once per deferred item");
}

void test_rolling_preprocess_prefers_lower_pending_failure_to_launch_failure() {
    std::atomic<std::size_t> completed{0U};
    trtmc::sam2_hoi::RollingAsyncPreprocessor::Launcher launcher =
        [](const trtmc::sam2_hoi::RollingAsyncPreprocessor::Callback& callback, std::size_t index) {
            if (index == 2U)
                throw std::runtime_error("launcher failure 2");
            return std::async(std::launch::async,
                              [callback, index]() mutable { return callback(index); });
        };
    trtmc::sam2_hoi::RollingAsyncPreprocessor scheduler(
        3U, 2U,
        [&](std::size_t index) {
            if (index == 0U)
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            completed.fetch_add(1U, std::memory_order_relaxed);
            if (index == 1U)
                throw std::runtime_error("callback failure 1");
            return std::vector<float>{static_cast<float>(index)};
        },
        std::move(launcher));

    bool reported_pending_failure = false;
    try {
        (void)scheduler.take_next();
    } catch (const std::runtime_error& error) {
        reported_pending_failure = std::string(error.what()) == "callback failure 1";
    }
    check(reported_pending_failure,
          "rolling preprocess prefers a lower pending callback failure to launch failure");
    check(completed.load(std::memory_order_relaxed) == 2U,
          "rolling preprocess joins all lower-index work before launch failure arbitration");
}

void test_ordered_mask_postprocess_is_bounded_and_ordered() {
    std::atomic<int> active{0};
    std::atomic<int> maximum_active{0};
    std::atomic<std::size_t> completed{0U};
    std::mutex gate_mutex;
    std::condition_variable gate_condition;
    std::size_t gate_arrivals = 0U;
    bool gate_open = false;
    trtmc::sam2_hoi::OrderedAsyncMaskPostprocessor processor;
    std::vector<std::size_t> observed;

    for (std::size_t index = 0; index < 7U; ++index) {
        if (processor.full()) {
            auto result = processor.take_next();
            observed.push_back(result.index);
            check(result.output ==
                      std::vector<std::uint8_t>{static_cast<std::uint8_t>(result.index)},
                  "ordered mask postprocess preserves result payload");
        }
        processor.submit(index, [&, index] {
            const int active_now = active.fetch_add(1, std::memory_order_relaxed) + 1;
            update_maximum(maximum_active, active_now);
            {
                std::unique_lock<std::mutex> lock(gate_mutex);
                if (!gate_open) {
                    ++gate_arrivals;
                    if (gate_arrivals == trtmc::sam2_hoi::kMaxConcurrentMaskPostprocessTasks) {
                        gate_open = true;
                        gate_condition.notify_all();
                    } else if (!gate_condition.wait_for(lock, std::chrono::seconds(2),
                                                        [&]() { return gate_open; })) {
                        gate_open = true;
                        gate_condition.notify_all();
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            completed.fetch_add(1U, std::memory_order_relaxed);
            active.fetch_sub(1, std::memory_order_relaxed);
            return std::vector<std::uint8_t>{static_cast<std::uint8_t>(index)};
        });
    }
    while (!processor.empty()) {
        auto result = processor.take_next();
        observed.push_back(result.index);
        check(result.output == std::vector<std::uint8_t>{static_cast<std::uint8_t>(result.index)},
              "ordered mask postprocess preserves drained payload");
    }

    check(observed == std::vector<std::size_t>({0U, 1U, 2U, 3U, 4U, 5U, 6U}),
          "ordered mask postprocess returns exact submission order");
    check(maximum_active.load(std::memory_order_relaxed) ==
              static_cast<int>(trtmc::sam2_hoi::kMaxConcurrentMaskPostprocessTasks),
          "ordered mask postprocess enforces its five-task bound");
    check(completed.load(std::memory_order_relaxed) == 7U,
          "ordered mask postprocess completes every submitted task");
}

void test_ordered_mask_postprocess_destructor_joins_pending_tasks() {
    std::atomic<std::size_t> completed{0U};
    {
        trtmc::sam2_hoi::OrderedAsyncMaskPostprocessor processor;
        for (std::size_t index = 0; index < trtmc::sam2_hoi::kMaxConcurrentMaskPostprocessTasks;
             ++index) {
            processor.submit(index, [&] {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                completed.fetch_add(1U, std::memory_order_relaxed);
                return std::vector<std::uint8_t>{1U};
            });
        }
    }
    check(completed.load(std::memory_order_relaxed) ==
              trtmc::sam2_hoi::kMaxConcurrentMaskPostprocessTasks,
          "ordered mask postprocess destructor joins every submitted task");
}

void test_ordered_mask_postprocess_falls_back_when_thread_launch_fails() {
    std::atomic<std::size_t> launch_attempts{0U};
    trtmc::sam2_hoi::OrderedAsyncMaskPostprocessor::Launcher unavailable =
        [&](trtmc::sam2_hoi::OrderedAsyncMaskPostprocessor::CallbackHandle,
            std::size_t) -> std::future<std::vector<std::uint8_t>> {
        launch_attempts.fetch_add(1U, std::memory_order_relaxed);
        throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));
    };
    trtmc::sam2_hoi::OrderedAsyncMaskPostprocessor processor(
        trtmc::sam2_hoi::kMaxConcurrentMaskPostprocessTasks, std::move(unavailable));
    for (std::size_t index = 0; index < 3U; ++index) {
        processor.submit(
            index, [index] { return std::vector<std::uint8_t>{static_cast<uint8_t>(index)}; });
    }
    std::vector<std::size_t> observed;
    while (!processor.empty())
        observed.push_back(processor.take_next().index);
    check(observed == std::vector<std::size_t>({0U, 1U, 2U}),
          "ordered mask postprocess preserves order through deferred fallback");
    check(launch_attempts.load(std::memory_order_relaxed) == 3U,
          "ordered mask postprocess attempts async launch once per deferred task");
}

struct CopyObservedMaskCallback {
    explicit CopyObservedMaskCallback(std::shared_ptr<std::atomic<int>> copies)
        : copies(std::move(copies)) {}
    CopyObservedMaskCallback(const CopyObservedMaskCallback& other) : copies(other.copies) {
        copies->fetch_add(1, std::memory_order_relaxed);
    }
    CopyObservedMaskCallback(CopyObservedMaskCallback&&) noexcept = default;

    std::vector<std::uint8_t> operator()() const { return {7U}; }

    std::shared_ptr<std::atomic<int>> copies;
};

void test_ordered_mask_postprocess_does_not_copy_owned_callback_payload() {
    auto copies = std::make_shared<std::atomic<int>>(0);
    trtmc::sam2_hoi::OrderedAsyncMaskPostprocessor::Callback callback{
        CopyObservedMaskCallback(copies)};
    copies->store(0, std::memory_order_relaxed);
    trtmc::sam2_hoi::OrderedAsyncMaskPostprocessor processor;
    processor.submit(0U, std::move(callback));
    const auto result = processor.take_next();
    check(result.output == std::vector<std::uint8_t>{7U},
          "ordered mask postprocess executes its owned callback payload");
    check(copies->load(std::memory_order_relaxed) == 0,
          "ordered mask postprocess shares callback ownership without deep copies");
}

void test_ordered_mask_postprocess_prefers_lower_pending_failure_to_launch_failure() {
    std::atomic<std::size_t> completed{0U};
    trtmc::sam2_hoi::OrderedAsyncMaskPostprocessor::Launcher launcher =
        [](trtmc::sam2_hoi::OrderedAsyncMaskPostprocessor::CallbackHandle callback,
           std::size_t index) {
            if (index == 2U)
                throw std::runtime_error("mask launcher failure 2");
            return std::async(std::launch::async,
                              [callback = std::move(callback)]() mutable { return (*callback)(); });
        };
    trtmc::sam2_hoi::OrderedAsyncMaskPostprocessor processor(2U, std::move(launcher));
    processor.submit(0U, [&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        completed.fetch_add(1U, std::memory_order_relaxed);
        return std::vector<std::uint8_t>{0U};
    });
    processor.submit(1U, [&]() -> std::vector<std::uint8_t> {
        completed.fetch_add(1U, std::memory_order_relaxed);
        throw std::runtime_error("mask callback failure 1");
    });
    const auto first = processor.take_next();
    check(first.index == 0U && first.output == std::vector<std::uint8_t>{0U},
          "ordered mask postprocess returns the first successful result");
    bool reported_pending_failure = false;
    try {
        processor.submit(2U, [] { return std::vector<std::uint8_t>{2U}; });
    } catch (const std::runtime_error& error) {
        reported_pending_failure = std::string(error.what()) == "mask callback failure 1";
    }
    check(reported_pending_failure,
          "ordered mask postprocess prefers lower callback failure to launch failure");
    check(completed.load(std::memory_order_relaxed) == 2U,
          "ordered mask postprocess joins pending work before launch failure arbitration");
}

void test_direct_prebinding_and_one_frame_tracking() {
    const ContractMap features = image_outputs();
    auto image = std::make_unique<FakeModule>(
        ContractMap{{"pixel_values", f32({1, 3, kImageSize, kImageSize})}}, features);
    FakeModule* image_raw = image.get();

    std::vector<float> class_scores(
        trtmc::sam2_hoi::kHoiQueryCount * trtmc::sam2_hoi::kHoiClassCount, 0.0F);
    std::vector<float> boxes(trtmc::sam2_hoi::kHoiQueryCount * 4, 0.0F);
    std::vector<float> embeddings(
        trtmc::sam2_hoi::kHoiQueryCount * trtmc::sam2_hoi::kHoiEmbeddingSize, 0.0F);
    class_scores[0] = 0.90F;
    class_scores[trtmc::sam2_hoi::kHoiClassCount + 1] = 0.80F;
    set_box(boxes, 0, {64.0F, 96.0F, 256.0F, 384.0F});
    set_box(boxes, 1, {512.0F, 128.0F, 896.0F, 512.0F});
    bool detector_received_only_prebound_features = false;
    auto detector = std::make_unique<FakeModule>(
        detector_inputs(),
        ContractMap{{"class_scores", f32({1500, 4})},
                    {"boxes_cxcywh", f32({1500, 4})},
                    {"query_embeddings", f32({1500, 256})}},
        [&](const TensorMap& inputs) {
            detector_received_only_prebound_features = inputs.empty();
            return TensorMap{
                {"class_scores", Tensor{class_scores.data(), {1500, 4}, DType::kFloat32}},
                {"boxes_cxcywh", Tensor{boxes.data(), {1500, 4}, DType::kFloat32}},
                {"query_embeddings", Tensor{embeddings.data(), {1500, 256}, DType::kFloat32}},
            };
        });
    FakeModule* detector_raw = detector.get();

    auto interaction =
        std::make_unique<FakeModule>(ContractMap{{"pair_features", f32({1, 512})}},
                                     ContractMap{{"interaction_probabilities", f32({1, 2})}});
    FakeModule* interaction_raw = interaction.get();

    std::vector<float> masks(static_cast<std::size_t>(kObjectBatch) * kMaskSize * kMaskSize, -1.0F);
    std::fill(masks.begin(), masks.begin() + kMaskSize * kMaskSize, 1.0F);
    std::vector<float> pointers(static_cast<std::size_t>(kObjectBatch) * kPointerWidth, 0.25F);
    std::array<float, kObjectBatch> object_scores{2.0F, 3.0F};
    bool prompt_received_expected_points = false;
    ContractMap prompt_inputs =
        feature_inputs(features, {"tracker_feature_0", "tracker_feature_1", "tracker_feature_2"});
    prompt_inputs.emplace("point_coords", f32({2, 3, 2}));
    prompt_inputs.emplace("point_labels", i32({2, 3}));
    auto prompt = std::make_unique<FakeModule>(
        std::move(prompt_inputs),
        ContractMap{{"pred_masks", f32({2, 1, 256, 256})},
                    {"object_pointer", f32({2, 256})},
                    {"object_score_logits", f32({2})}},
        [&](const TensorMap& inputs) {
            const auto coordinates = inputs.find("point_coords");
            const auto labels = inputs.find("point_labels");
            if (coordinates != inputs.end() && labels != inputs.end() &&
                coordinates->second.data != nullptr && labels->second.data != nullptr &&
                coordinates->second.shape == std::vector<int64_t>({2, 3, 2}) &&
                labels->second.shape == std::vector<int64_t>({2, 3})) {
                const auto* values = static_cast<const float*>(coordinates->second.data);
                const auto* kinds = static_cast<const int32_t*>(labels->second.data);
                const std::array<float, 12> expected_coordinates{
                    64.0F,  96.0F,  256.0F, 384.0F, 160.0F, 240.0F,
                    512.0F, 128.0F, 896.0F, 512.0F, 704.0F, 320.0F,
                };
                const std::array<int32_t, 6> expected_labels{2, 3, 1, 2, 3, 1};
                prompt_received_expected_points =
                    std::equal(expected_coordinates.begin(), expected_coordinates.end(), values) &&
                    std::equal(expected_labels.begin(), expected_labels.end(), kinds);
            }
            return TensorMap{
                {"pred_masks", Tensor{masks.data(), {2, 1, 256, 256}, DType::kFloat32}},
                {"object_pointer", Tensor{pointers.data(), {2, 256}, DType::kFloat32}},
                {"object_score_logits", Tensor{object_scores.data(), {2}, DType::kFloat32}},
            };
        });
    FakeModule* prompt_raw = prompt.get();

    ContractMap recurrent_inputs =
        feature_inputs(features, {"tracker_feature_0", "tracker_feature_1", "tracker_feature_2",
                                  "tracker_position_2"});
    recurrent_inputs.emplace("memory_features", f32({2, 1, 64, 64, 64}));
    recurrent_inputs.emplace("memory_position", f32({2, 1, 64, 64, 64}));
    recurrent_inputs.emplace("memory_temporal_offsets", i32({2, 1}));
    recurrent_inputs.emplace("object_pointers", f32({2, 1, 256}));
    recurrent_inputs.emplace("object_pointer_temporal_offsets", f32({2, 1}));
    recurrent_inputs.emplace("object_pointer_time_denominator", f32({1}));
    auto recurrent = std::make_unique<FakeModule>(std::move(recurrent_inputs),
                                                  ContractMap{{"pred_masks", f32({2, 1, 256, 256})},
                                                              {"object_pointer", f32({2, 256})},
                                                              {"object_score_logits", f32({2})}});
    FakeModule* recurrent_raw = recurrent.get();

    const std::size_t memory_values =
        static_cast<std::size_t>(kObjectBatch) * kMemoryChannels * kMemorySize * kMemorySize;
    std::vector<float> memory_features(memory_values, 0.125F);
    std::vector<float> memory_positions(memory_values, 0.25F);
    bool memory_received_prompt_outputs = false;
    ContractMap memory_inputs = feature_inputs(features, {"tracker_feature_2"});
    memory_inputs.emplace("pred_masks", f32({2, 1, 256, 256}));
    memory_inputs.emplace("object_score_logits", f32({2, 1}));
    memory_inputs.emplace("is_mask_from_points", i32({2, 1}));
    auto memory = std::make_unique<FakeModule>(
        std::move(memory_inputs),
        ContractMap{{"new_memory_features", f32({2, 64, 64, 64})},
                    {"new_memory_position", f32({2, 64, 64, 64})}},
        [&](const TensorMap& inputs) {
            const auto mask = inputs.find("pred_masks");
            const auto scores = inputs.find("object_score_logits");
            const auto flags = inputs.find("is_mask_from_points");
            if (mask != inputs.end() && scores != inputs.end() && flags != inputs.end() &&
                mask->second.data != nullptr && scores->second.data != nullptr &&
                flags->second.data != nullptr) {
                const auto* mask_values = static_cast<const float*>(mask->second.data);
                const auto* score_values = static_cast<const float*>(scores->second.data);
                const auto* flag_values = static_cast<const int32_t*>(flags->second.data);
                memory_received_prompt_outputs =
                    mask->second.shape == std::vector<int64_t>({2, 1, 256, 256}) &&
                    scores->second.shape == std::vector<int64_t>({2, 1}) &&
                    flags->second.shape == std::vector<int64_t>({2, 1}) && mask_values[0] == 1.0F &&
                    mask_values[kMaskSize * kMaskSize] == -1.0F && score_values[0] == 2.0F &&
                    score_values[1] == 3.0F && flag_values[0] == 1 && flag_values[1] == 1;
            }
            return TensorMap{
                {"new_memory_features",
                 Tensor{memory_features.data(), {2, 64, 64, 64}, DType::kFloat32}},
                {"new_memory_position",
                 Tensor{memory_positions.data(), {2, 64, 64, 64}, DType::kFloat32}},
            };
        });
    FakeModule* memory_raw = memory.get();

    auto pafpn = std::make_unique<FakePafpnComposite>();
    FakePafpnComposite* pafpn_raw = pafpn.get();
    trtmc::sam2_hoi::Sam2HoiPipeline pipeline(
        std::make_shared<int>(1), std::move(image), std::move(pafpn), std::move(detector),
        std::move(interaction), std::move(prompt), std::move(recurrent), std::move(memory),
        "sam2-hoi-test");

    check(std::string(pipeline.model_id()) == "sam2-hoi-test", "pipeline retains model id");
    check(pipeline.max_video_frame_load_concurrency() == trtmc::sam2_hoi::kMaxConcurrentJpegDecodes,
          "pipeline preserves the model-owned bounded batch decoder");
    check(detector_raw->external_binding_count() == 3,
          "detector receives three direct feature bindings");
    check(prompt_raw->external_binding_count() == 3,
          "prompt tracker receives three direct feature bindings");
    check(recurrent_raw->external_binding_count() == 4,
          "recurrent tracker receives four direct feature bindings");
    check(memory_raw->external_binding_count() == 1,
          "memory encoder receives one direct feature binding");
    check(pafpn_raw->roots ==
              std::vector<std::pair<std::string, std::string>>{{"fpn_input_0", "fpn_input_0"},
                                                               {"fpn_input_1", "tracker_feature_2"},
                                                               {"fpn_input_2", "fpn_input_2"}},
          "PAFPN roots alias the front outputs including shared root one");
    check(pafpn_raw->detector_bindings.size() == 3,
          "detector receives three PAFPN output bindings");
    for (const char* name : {"tracker_feature_0", "tracker_feature_1", "tracker_feature_2"}) {
        check_direct_binding(*image_raw, name, *prompt_raw, name,
                             "prompt binding aliases image output with exact shape");
        check_direct_binding(*image_raw, name, *recurrent_raw, name,
                             "recurrent binding aliases image output with exact shape");
    }
    check_direct_binding(*image_raw, "tracker_position_2", *recurrent_raw, "tracker_position_2",
                         "position binding aliases image output with exact shape");
    check_direct_binding(*image_raw, "tracker_feature_2", *memory_raw, "tracker_feature_2",
                         "memory binding aliases image output with exact shape");

    TemporaryDirectory temporary;
    const auto output_json = temporary.path / "tracking.json";
    const auto mask_directory = temporary.path / "masks";
    const std::array<float, 3> pixel{0.0F, 0.5F, 1.0F};
    const std::vector<trtmc::sam2_hoi::Sam2HoiVideoFrameView> frames{{pixel.data(), 1, 1}};

    bool rejected_partial_output = false;
    try {
        (void)pipeline.track_video(frames, output_json.string(), "");
    } catch (const std::invalid_argument&) {
        rejected_partial_output = true;
    }
    check(rejected_partial_output, "tracking rejects exactly one empty output path");

    bool rejected_output_collision = false;
    try {
        (void)pipeline.track_video(frames, (mask_directory / "frame_000000.npy").string(),
                                   mask_directory.string());
    } catch (const std::invalid_argument& error) {
        rejected_output_collision =
            std::string(error.what()).find("generated mask") != std::string::npos;
    }
    check(rejected_output_collision,
          "tracking rejects JSON output that would replace a generated mask");
    check(image_raw->forward_async_count == 0,
          "tracking rejects colliding outputs before executing inference");

    try {
        check(pipeline.track_video(frames, output_json.string(), mask_directory.string()) == 1,
              "one-frame tracking reports one result");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: one-frame tracking threw: " << error.what() << '\n';
        ++g_failures;
        return;
    }

    check(image_raw->forward_async_count == 1 && image_raw->sync_count == 0 &&
              pafpn_raw->forward_async_count == 1 && pafpn_raw->sync_count == 0,
          "front and PAFPN enqueue without an intermediate host synchronization");
    check(image_raw->last_async_shapes["pixel_values"] ==
                  std::vector<int64_t>({1, 3, kImageSize, kImageSize}) &&
              image_raw->last_async_dtypes["pixel_values"] == DType::kFloat32,
          "image engine receives the fixed preprocessed tensor");
    check(detector_raw->forward_count == 1 && detector_received_only_prebound_features,
          "detector executes with only its direct feature bindings");
    check(interaction_raw->forward_count == 0,
          "interaction engine is skipped when no object pair exists");
    check(prompt_raw->forward_count == 1 && prompt_received_expected_points,
          "prompt tracker receives two box-and-center prompts");
    check(recurrent_raw->forward_count == 0, "recurrent tracker is skipped on the prompt frame");
    check(memory_raw->forward_count == 0 && !memory_received_prompt_outputs,
          "one-frame tracking skips unused final-frame memory encoding");

    std::ifstream json_input(output_json, std::ios::binary);
    const std::string json{std::istreambuf_iterator<char>(json_input),
                           std::istreambuf_iterator<char>()};
    check(json.find("\"object_ids\": [0,1]") != std::string::npos,
          "JSON preserves the two selected object ids");
    check(json.find("\"det_labels\": [0,1]") != std::string::npos,
          "JSON preserves the two hand labels");
    check(json.find("\"interaction_pairs\": []") != std::string::npos,
          "JSON records the empty interaction set");

    const auto mask_path = mask_directory / "frame_000000.npy";
    std::ifstream mask_input(mask_path, std::ios::binary);
    const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(mask_input),
                                          std::istreambuf_iterator<char>()};
    check(bytes.size() >= 12, "mask output contains a NumPy header and two values");
    if (bytes.size() >= 12) {
        const std::size_t header_size =
            static_cast<std::size_t>(bytes[8]) | (static_cast<std::size_t>(bytes[9]) << 8U);
        check(10U + header_size + 2U == bytes.size(), "mask output payload has two objects");
        if (10U + header_size + 2U == bytes.size()) {
            const std::string header(
                bytes.begin() + 10, bytes.begin() + static_cast<std::ptrdiff_t>(10U + header_size));
            check(header.find("'shape': (2, 1, 1, 1)") != std::string::npos,
                  "mask output declares two one-pixel masks");
            check(bytes[10U + header_size] == 1U && bytes[10U + header_size + 1U] == 0U,
                  "mask output serializes one foreground and one background object");
        }
    }

    try {
        check(pipeline.track_video(frames, "", "") == 1,
              "discard-output tracking reports the same result count");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: discard-output tracking threw: " << error.what() << '\n';
        ++g_failures;
    }
    check(image_raw->forward_async_count == 2 && image_raw->sync_count == 0 &&
              pafpn_raw->forward_async_count == 2 && pafpn_raw->sync_count == 0,
          "discard-output tracking executes the same unsynchronized front/PAFPN path");
    check(detector_raw->forward_count == 2 && prompt_raw->forward_count == 2 &&
              memory_raw->forward_count == 0,
          "discard-output tracking executes inference without serialization");
}

void test_prompt_can_start_after_empty_detection_frame() {
    const ContractMap features = image_outputs();
    auto image = std::make_unique<FakeModule>(
        ContractMap{{"pixel_values", f32({1, 3, kImageSize, kImageSize})}}, features);
    FakeModule* image_raw = image.get();

    std::vector<float> empty_scores(
        trtmc::sam2_hoi::kHoiQueryCount * trtmc::sam2_hoi::kHoiClassCount, 0.0F);
    std::vector<float> prompt_scores = empty_scores;
    std::vector<float> boxes(trtmc::sam2_hoi::kHoiQueryCount * 4, 0.0F);
    std::vector<float> embeddings(
        trtmc::sam2_hoi::kHoiQueryCount * trtmc::sam2_hoi::kHoiEmbeddingSize, 0.0F);
    prompt_scores[0] = 0.90F;
    prompt_scores[trtmc::sam2_hoi::kHoiClassCount + 1] = 0.80F;
    set_box(boxes, 0, {64.0F, 96.0F, 256.0F, 384.0F});
    set_box(boxes, 1, {512.0F, 128.0F, 896.0F, 512.0F});
    int detector_response = 0;
    auto detector = std::make_unique<FakeModule>(
        detector_inputs(),
        ContractMap{{"class_scores", f32({1500, 4})},
                    {"boxes_cxcywh", f32({1500, 4})},
                    {"query_embeddings", f32({1500, 256})}},
        [&](const TensorMap&) {
            auto& scores = detector_response++ == 0 ? empty_scores : prompt_scores;
            return TensorMap{
                {"class_scores", Tensor{scores.data(), {1500, 4}, DType::kFloat32}},
                {"boxes_cxcywh", Tensor{boxes.data(), {1500, 4}, DType::kFloat32}},
                {"query_embeddings", Tensor{embeddings.data(), {1500, 256}, DType::kFloat32}},
            };
        });
    FakeModule* detector_raw = detector.get();

    auto interaction =
        std::make_unique<FakeModule>(ContractMap{{"pair_features", f32({1, 512})}},
                                     ContractMap{{"interaction_probabilities", f32({1, 2})}});
    FakeModule* interaction_raw = interaction.get();

    std::vector<float> masks(static_cast<std::size_t>(kObjectBatch) * kMaskSize * kMaskSize, 1.0F);
    masks[0] = -1.0F;
    std::vector<float> pointers(static_cast<std::size_t>(kObjectBatch) * kPointerWidth, 0.25F);
    std::array<float, kObjectBatch> object_scores{2.0F, 3.0F};
    ContractMap prompt_inputs =
        feature_inputs(features, {"tracker_feature_0", "tracker_feature_1", "tracker_feature_2"});
    prompt_inputs.emplace("point_coords", f32({2, 3, 2}));
    prompt_inputs.emplace("point_labels", i32({2, 3}));
    auto prompt = std::make_unique<FakeModule>(
        std::move(prompt_inputs),
        ContractMap{{"pred_masks", f32({2, 1, 256, 256})},
                    {"object_pointer", f32({2, 256})},
                    {"object_score_logits", f32({2})}},
        [&](const TensorMap&) {
            return TensorMap{
                {"pred_masks", Tensor{masks.data(), {2, 1, 256, 256}, DType::kFloat32}},
                {"object_pointer", Tensor{pointers.data(), {2, 256}, DType::kFloat32}},
                {"object_score_logits", Tensor{object_scores.data(), {2}, DType::kFloat32}},
            };
        });
    FakeModule* prompt_raw = prompt.get();

    int recurrent_response = 0;
    bool recurrent_received_first_history = false;
    bool recurrent_received_second_history = false;
    ContractMap recurrent_inputs =
        feature_inputs(features, {"tracker_feature_0", "tracker_feature_1", "tracker_feature_2",
                                  "tracker_position_2"});
    recurrent_inputs.emplace("memory_features", f32({2, 1, 64, 64, 64}));
    recurrent_inputs.emplace("memory_position", f32({2, 1, 64, 64, 64}));
    recurrent_inputs.emplace("memory_temporal_offsets", i32({2, 1}));
    recurrent_inputs.emplace("object_pointers", f32({2, 1, 256}));
    recurrent_inputs.emplace("object_pointer_temporal_offsets", f32({2, 1}));
    recurrent_inputs.emplace("object_pointer_time_denominator", f32({1}));
    auto recurrent = std::make_unique<FakeModule>(
        std::move(recurrent_inputs),
        ContractMap{{"pred_masks", f32({2, 1, 256, 256})},
                    {"object_pointer", f32({2, 256})},
                    {"object_score_logits", f32({2})}},
        [&](const TensorMap& inputs) {
            const auto memory_features_input = inputs.find("memory_features");
            const auto memory_position_input = inputs.find("memory_position");
            const auto memory_offsets = inputs.find("memory_temporal_offsets");
            const auto pointer_values = inputs.find("object_pointers");
            const auto pointer_offsets = inputs.find("object_pointer_temporal_offsets");
            const auto denominator = inputs.find("object_pointer_time_denominator");
            if (memory_features_input != inputs.end() && memory_position_input != inputs.end() &&
                memory_offsets != inputs.end() && pointer_values != inputs.end() &&
                pointer_offsets != inputs.end() && denominator != inputs.end() &&
                memory_offsets->second.data != nullptr && pointer_offsets->second.data != nullptr &&
                denominator->second.data != nullptr) {
                const auto* memory_offset_values =
                    static_cast<const int32_t*>(memory_offsets->second.data);
                const auto* pointer_offset_values =
                    static_cast<const float*>(pointer_offsets->second.data);
                const auto* denominator_value = static_cast<const float*>(denominator->second.data);
                if (recurrent_response == 0) {
                    recurrent_received_first_history =
                        memory_features_input->second.shape ==
                            std::vector<int64_t>({2, 1, 64, 64, 64}) &&
                        memory_position_input->second.shape ==
                            std::vector<int64_t>({2, 1, 64, 64, 64}) &&
                        memory_offsets->second.shape == std::vector<int64_t>({2, 1}) &&
                        pointer_values->second.shape == std::vector<int64_t>({2, 1, 256}) &&
                        pointer_offsets->second.shape == std::vector<int64_t>({2, 1}) &&
                        denominator->second.shape == std::vector<int64_t>({1}) &&
                        memory_offset_values[0] == 0 && memory_offset_values[1] == 0 &&
                        pointer_offset_values[0] == 1.0F && pointer_offset_values[1] == 1.0F &&
                        denominator_value[0] == 3.0F;
                } else if (recurrent_response == 1) {
                    recurrent_received_second_history =
                        memory_features_input->second.shape ==
                            std::vector<int64_t>({2, 2, 64, 64, 64}) &&
                        memory_position_input->second.shape ==
                            std::vector<int64_t>({2, 2, 64, 64, 64}) &&
                        memory_offsets->second.shape == std::vector<int64_t>({2, 2}) &&
                        pointer_values->second.shape == std::vector<int64_t>({2, 2, 256}) &&
                        pointer_offsets->second.shape == std::vector<int64_t>({2, 2}) &&
                        denominator->second.shape == std::vector<int64_t>({1}) &&
                        memory_offset_values[0] == 0 && memory_offset_values[1] == 6 &&
                        pointer_offset_values[0] == 2.0F && pointer_offset_values[1] == 1.0F &&
                        denominator_value[0] == 3.0F;
                }
            }
            ++recurrent_response;
            return TensorMap{
                {"pred_masks", Tensor{masks.data(), {2, 1, 256, 256}, DType::kFloat32}},
                {"object_pointer", Tensor{pointers.data(), {2, 256}, DType::kFloat32}},
                {"object_score_logits", Tensor{object_scores.data(), {2}, DType::kFloat32}},
            };
        });
    FakeModule* recurrent_raw = recurrent.get();

    const std::size_t memory_values =
        static_cast<std::size_t>(kObjectBatch) * kMemoryChannels * kMemorySize * kMemorySize;
    std::vector<float> memory_features(memory_values, 0.125F);
    std::vector<float> memory_positions(memory_values, 0.25F);
    std::vector<float> observed_memory_mask_values;
    std::vector<int32_t> observed_memory_point_flags;
    ContractMap memory_inputs = feature_inputs(features, {"tracker_feature_2"});
    memory_inputs.emplace("pred_masks", f32({2, 1, 256, 256}));
    memory_inputs.emplace("object_score_logits", f32({2, 1}));
    memory_inputs.emplace("is_mask_from_points", i32({2, 1}));
    auto memory = std::make_unique<FakeModule>(
        std::move(memory_inputs),
        ContractMap{{"new_memory_features", f32({2, 64, 64, 64})},
                    {"new_memory_position", f32({2, 64, 64, 64})}},
        [&](const TensorMap& inputs) {
            const auto mask = inputs.find("pred_masks");
            const auto flags = inputs.find("is_mask_from_points");
            if (mask != inputs.end() && flags != inputs.end() && mask->second.data != nullptr &&
                flags->second.data != nullptr) {
                observed_memory_mask_values.push_back(
                    static_cast<const float*>(mask->second.data)[0]);
                observed_memory_point_flags.push_back(
                    static_cast<const int32_t*>(flags->second.data)[0]);
            }
            return TensorMap{
                {"new_memory_features",
                 Tensor{memory_features.data(), {2, 64, 64, 64}, DType::kFloat32}},
                {"new_memory_position",
                 Tensor{memory_positions.data(), {2, 64, 64, 64}, DType::kFloat32}},
            };
        });
    FakeModule* memory_raw = memory.get();

    auto pafpn = std::make_unique<FakePafpnComposite>();
    FakePafpnComposite* pafpn_raw = pafpn.get();
    trtmc::sam2_hoi::Sam2HoiPipeline pipeline(
        std::make_shared<int>(1), std::move(image), std::move(pafpn), std::move(detector),
        std::move(interaction), std::move(prompt), std::move(recurrent), std::move(memory),
        "sam2-hoi-delayed-prompt-test");

    TemporaryDirectory temporary;
    const auto output_json = temporary.path / "tracking.json";
    const auto mask_directory = temporary.path / "masks";
    const std::array<float, 3> first_pixel{0.0F, 0.5F, 1.0F};
    const std::array<float, 3> second_pixel{1.0F, 0.5F, 0.0F};
    const std::array<float, 3> third_pixel{0.25F, 0.5F, 0.75F};
    const std::array<float, 3> fourth_pixel{0.75F, 0.5F, 0.25F};
    const std::vector<trtmc::sam2_hoi::Sam2HoiVideoFrameView> frames{
        {first_pixel.data(), 1, 1},
        {second_pixel.data(), 1, 1},
        {third_pixel.data(), 1, 1},
        {fourth_pixel.data(), 1, 1},
    };

    try {
        check(pipeline.track_video(frames, output_json.string(), mask_directory.string()) == 3,
              "delayed prompt emits prompt and two recurrent results");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: delayed-prompt tracking threw: " << error.what() << '\n';
        ++g_failures;
        return;
    }

    check(image_raw->forward_async_count == 4 && image_raw->sync_count == 0 &&
              pafpn_raw->forward_async_count == 4 && pafpn_raw->sync_count == 0,
          "delayed prompt evaluates front and PAFPN without intermediate syncs");
    check(detector_raw->forward_count == 4,
          "delayed prompt evaluates the detector on all source frames");
    check(interaction_raw->forward_count == 0,
          "delayed prompt skips interaction when no object pair exists");
    check(prompt_raw->forward_count == 1,
          "delayed prompt invokes the prompt tracker once on frame one");
    check(recurrent_raw->forward_count == 2 && recurrent_received_first_history &&
              recurrent_received_second_history,
          "delayed prompt recurrent frames receive ordered M=P=1 then M=P=2 history");
    check(memory_raw->forward_count == 2,
          "delayed prompt stores both memories consumed by recurrent frames");
    check(observed_memory_mask_values == std::vector<float>({0.1F, -1.0F}) &&
              observed_memory_point_flags == std::vector<int32_t>({1, 0}),
          "memory encoder receives filled prompt masks then raw recurrent masks");

    std::ifstream json_input(output_json, std::ios::binary);
    const std::string json{std::istreambuf_iterator<char>(json_input),
                           std::istreambuf_iterator<char>()};
    check(json.find("\"frame_index\": 1") != std::string::npos,
          "delayed prompt JSON preserves source frame index one");
    check(json.find("\"frame_index\": 2") != std::string::npos,
          "delayed prompt JSON preserves source frame index two");
    check(json.find("\"frame_index\": 3") != std::string::npos,
          "delayed prompt JSON preserves source frame index three");
    check(json.find("\"frame_index\": 0") == std::string::npos,
          "delayed prompt JSON omits the empty source frame");

    std::vector<std::string> mask_assets;
    for (const auto& entry : std::filesystem::directory_iterator(mask_directory)) {
        if (entry.is_regular_file())
            mask_assets.push_back(entry.path().filename().string());
    }
    std::sort(mask_assets.begin(), mask_assets.end());
    check(mask_assets ==
              std::vector<std::string>{"frame_000001.npy", "frame_000002.npy", "frame_000003.npy"},
          "delayed prompt writes source-indexed prompt and recurrent mask assets");
}

void test_legacy_six_plan_sync_boundary() {
    const ContractMap features = image_outputs();
    ContractMap monolithic_outputs = features;
    monolithic_outputs.erase("fpn_input_0");
    monolithic_outputs.erase("fpn_input_2");
    for (const auto& [name, contract] : detector_inputs())
        monolithic_outputs.emplace(name, contract);

    std::vector<std::string> events;
    auto image = std::make_unique<FakeModule>(
        ContractMap{{"pixel_values", f32({1, 3, kImageSize, kImageSize})}},
        std::move(monolithic_outputs), FakeModule::Forward{}, &events, "image");
    FakeModule* image_raw = image.get();

    std::vector<float> class_scores(
        trtmc::sam2_hoi::kHoiQueryCount * trtmc::sam2_hoi::kHoiClassCount, 0.0F);
    std::vector<float> boxes(trtmc::sam2_hoi::kHoiQueryCount * 4, 0.0F);
    std::vector<float> embeddings(
        trtmc::sam2_hoi::kHoiQueryCount * trtmc::sam2_hoi::kHoiEmbeddingSize, 0.0F);
    auto detector = std::make_unique<FakeModule>(
        detector_inputs(),
        ContractMap{{"class_scores", f32({1500, 4})},
                    {"boxes_cxcywh", f32({1500, 4})},
                    {"query_embeddings", f32({1500, 256})}},
        [&](const TensorMap&) {
            return TensorMap{
                {"class_scores", Tensor{class_scores.data(), {1500, 4}, DType::kFloat32}},
                {"boxes_cxcywh", Tensor{boxes.data(), {1500, 4}, DType::kFloat32}},
                {"query_embeddings", Tensor{embeddings.data(), {1500, 256}, DType::kFloat32}},
            };
        },
        &events, "detector");
    FakeModule* detector_raw = detector.get();

    auto interaction =
        std::make_unique<FakeModule>(ContractMap{{"pair_features", f32({1, 512})}},
                                     ContractMap{{"interaction_probabilities", f32({1, 2})}});
    auto prompt = std::make_unique<FakeModule>(
        feature_inputs(features, {"tracker_feature_0", "tracker_feature_1", "tracker_feature_2"}),
        ContractMap{});
    auto recurrent = std::make_unique<FakeModule>(
        feature_inputs(features, {"tracker_feature_0", "tracker_feature_1", "tracker_feature_2",
                                  "tracker_position_2"}),
        ContractMap{});
    auto memory = std::make_unique<FakeModule>(feature_inputs(features, {"tracker_feature_2"}),
                                               ContractMap{});

    trtmc::sam2_hoi::Sam2HoiPipeline pipeline(
        std::move(image), std::move(detector), std::move(interaction), std::move(prompt),
        std::move(recurrent), std::move(memory), "sam2-hoi-legacy-test");
    for (const char* name : {"detector_feature_0", "detector_feature_1", "detector_feature_2"}) {
        check_direct_binding(*image_raw, name, *detector_raw, name,
                             "legacy detector input aliases monolithic image output");
    }

    const std::array<float, 3> pixel{0.0F, 0.5F, 1.0F};
    const std::vector<trtmc::sam2_hoi::Sam2HoiVideoFrameView> frames{{pixel.data(), 1, 1}};
    bool rejected_empty_detection = false;
    try {
        (void)pipeline.track_video(frames, "", "");
    } catch (const std::runtime_error& error) {
        rejected_empty_detection =
            std::string(error.what()).find("no detections") != std::string::npos;
    }
    check(rejected_empty_detection, "legacy test reaches the expected empty-detection exit");
    check(events.size() >= 3 && events[0] == "image:forward_async" && events[1] == "image:sync" &&
              events[2] == "detector:forward",
          "legacy path synchronizes the monolithic image engine before detector execution");
    check(image_raw->forward_async_count == 1 && image_raw->sync_count == 1 &&
              detector_raw->forward_count == 1,
          "legacy path preserves one front enqueue, one front sync, and one detector call");
}

} // namespace

int main() {
    test_rolling_preprocess_is_bounded_and_ordered();
    test_rolling_preprocess_joins_before_lowest_failure();
    test_rolling_preprocess_destructor_joins_pending_tasks();
    test_rolling_preprocess_falls_back_when_thread_launch_fails();
    test_rolling_preprocess_prefers_lower_pending_failure_to_launch_failure();
    test_ordered_mask_postprocess_is_bounded_and_ordered();
    test_ordered_mask_postprocess_destructor_joins_pending_tasks();
    test_ordered_mask_postprocess_falls_back_when_thread_launch_fails();
    test_ordered_mask_postprocess_does_not_copy_owned_callback_payload();
    test_ordered_mask_postprocess_prefers_lower_pending_failure_to_launch_failure();
    test_direct_prebinding_and_one_frame_tracking();
    test_prompt_can_start_after_empty_detection_frame();
    test_legacy_six_plan_sync_boundary();
    if (g_failures != 0) {
        std::cerr << g_failures << " SAM2 HOI pipeline test(s) failed\n";
        return 1;
    }
    std::cout << "SAM2 HOI pipeline tests passed\n";
    return 0;
}
