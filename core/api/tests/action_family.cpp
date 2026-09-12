/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/internal/action.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/family_factory.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>

namespace {
using namespace trtmc;
using namespace trtmc::internal;

// Only the family fixture implements queue mechanics. No physical controller or
// TensorRT workload is executed by these protocol tests.
std::atomic<bool> block_next{false}, entered{false};
std::mutex block_mutex;
std::condition_variable block_condition;
bool unblock = false;
void maybe_block() {
    if (!block_next.exchange(false))
        return;
    std::unique_lock<std::mutex> lock(block_mutex);
    entered = true;
    if (!block_condition.wait_for(lock, std::chrono::seconds(5), [] { return unblock; }))
        throw std::runtime_error("action fixture blocking test timed out");
}
std::string tag(ConfigView config) {
    std::string value = "default";
    bool seen = false;
    for (const auto& entry : config) {
        if (entry.name != "tag" || seen || !std::holds_alternative<std::string_view>(entry.value))
            throw ConfigError("unknown, duplicate or mistyped action tag");
        value = std::get<std::string_view>(entry.value);
        seen = true;
    }
    return value;
}
void shape_and_state(const ImageStateObservation& input) {
    if (input.image.height != 1 || input.image.width != 1 || input.image.channels != 3 ||
        input.state.size() != 2)
        throw std::invalid_argument("fixture requires one RGB pixel and two state values");
    for (const auto value : input.state)
        if (!std::isfinite(value))
            throw std::invalid_argument("fixture state must be finite on every action call");
}
float image_value(const ImageView& image) {
    return image.format == ImageFormat::Float32
               ? static_cast<const float*>(image.data)[0]
               : static_cast<const uint8_t*>(image.data)[0] / 255.0F;
}
ActionSequenceResult predict(const ImageStateObservation& input, const std::string& name) {
    shape_and_state(input);
    const auto pixel = image_value(input.image);
    if (!std::isfinite(pixel) || pixel < 0 || pixel > 1)
        throw std::invalid_argument("fixture image is outside input range");
    return {{{input.state[0] + pixel, -3, input.state[1], 8}, 2, 2},
            {"fixture." + name, {"left", "right"}, {}, "", "unnormalized"},
            {},
            {}};
}
bool within(Span<const float> values) {
    for (const auto value : values)
        if (!std::isfinite(value) || std::abs(value) > 5)
            return false;
    return true;
}
class Session final : public IImageStateActionSession {
  public:
    Session(std::string name, std::string mode) : name_(std::move(name)), mode_(std::move(mode)) {}
    ActionStepResult act(const ImageStateObservation& input, ConfigView config) override {
        if (!config.empty())
            throw ConfigError("fixture act accepts no config; tag belongs to creation");
        shape_and_state(input);
        maybe_block();
        bool started = false;
        double milliseconds = 0;
        if (cursor_ >= queue_.values.rows) {
            queue_ = predict(input, name_);
            cursor_ = 0;
            started = true;
            milliseconds = static_cast<double>(++refills_);
        }
        const auto begin = queue_.values.values.begin() + static_cast<std::ptrdiff_t>(cursor_ * 2);
        ActionSequenceResult step{{{begin, begin + 2}, 1, 2}, queue_.schema, {}, {}};
        ++cursor_;
        const bool bounds = within({step.values.values.data(), step.values.values.size()});
        if (mode_ == "bad_step") {
            step.values.rows = 2;
            step.values.values.insert(step.values.values.end(), {0, 0});
        }
        return {std::move(step), bounds, started, milliseconds};
    }
    void reset() override {
        queue_ = {};
        cursor_ = 0; // Family-owned queue and execution-state reset.
    }

  private:
    std::string name_, mode_;
    ActionSequenceResult queue_;
    uint64_t cursor_{0}, refills_{0};
};
class ActionFixture final : public IModel,
                            public IImageStateToActionChunk,
                            public IImageStateActionQueue {
  public:
    explicit ActionFixture(std::string mode) : mode_(std::move(mode)) {}
    const char* task() const noexcept override { return mode_.c_str(); }
    std::vector<TaskInstance> task_bindings() override {
        if (mode_ == "none" || mode_ == "example_recorded_unsupported")
            return {};
        return {bind<IImageStateToActionChunk>(*this, fields_for(IImageStateToActionChunk::kTask)),
                bind<IImageStateActionQueue>(*this, fields_for(IImageStateActionQueue::kTask))};
    }
    trtmc::Span<const ConfigField> fields_for(std::string_view) const {
        if (mode_.find("example_recorded") == 0)
            return {};
        static const ConfigField declared[] = {
            {"tag", ConfigKind::String, ConfigValue{std::string_view{"default"}},
             "Synthetic schema tag for chunk run or queue creation; act accepts no config."}};
        return declared;
    }
    ImageStateActionChunkResult run(const ImageStateToActionChunkRequest& input,
                                    ConfigView config) override {
        if (mode_.find("example_recorded") == 0) {
            if (mode_ == "example_recorded_fail")
                throw std::runtime_error("recorded fixture provider failed");
            const auto& observation = input.observation;
            if (!config.empty() || observation.image.height != 480 ||
                observation.image.width != 640 || observation.image.channels != 3 ||
                observation.image.format != ImageFormat::Float32 || observation.state.size() != 14)
                throw std::invalid_argument(
                    "recorded fixture requires the exact example input contract");
            for (size_t index = 0; index < observation.state.size(); ++index)
                if (observation.state[index] != static_cast<float>(index))
                    throw std::invalid_argument("recorded fixture state was changed or reordered");
            if (image_value(observation.image) != 1.0F)
                throw std::invalid_argument("recorded image normalization was changed");
            if (++stateless_calls_ != 1)
                throw std::logic_error("recorded replay predicted more than one chunk");
            ActionSequenceResult result;
            result.values.rows = mode_ == "example_recorded_bad_shape" ? 99 : 100;
            result.values.columns = 14;
            result.schema = {"fixture.recorded", {}, {}, "", "unnormalized"};
            for (uint64_t step = 0; step < result.values.rows; ++step)
                for (int32_t joint = 0; joint < 14; ++joint)
                    result.values.values.push_back(static_cast<float>(100 * step + joint) - 700);
            return {std::move(result), false, 101.0};
        }
        auto result = predict(input.observation, tag(config));
        const bool bounds = within({result.values.values.data(), result.values.values.size()});
        if (mode_ == "bad_chunk")
            result.values.values.pop_back();
        return {std::move(result), bounds, static_cast<double>(100 + (++stateless_calls_))};
    }
    std::unique_ptr<IImageStateActionSession> create_action_session(ConfigView config) override {
        const auto name = tag(config);
        if (mode_ == "null_queue")
            return nullptr;
        return std::make_unique<Session>(name, mode_);
    }

  private:
    std::string mode_;
    uint64_t stateless_calls_{0};
};
class DeclaredOnly final : public IModel {
  public:
    const char* task() const noexcept override { return "missing"; }
    std::vector<TaskInstance> task_bindings() override { return {}; }
};
// Independent old-path fixture, not an adapter over a new SDK implementation.
class ExistingRecorded final : public trtmc::IRobotControl {
  public:
    RobotActionChunk predict_action_chunk(const RobotObservation& input) override {
        if (++calls_ != 1 || input.image_height != 480 || input.image_width != 640 ||
            input.image_channels != 3 || input.state.size() != 14 || input.image_pixels.empty() ||
            input.image_pixels[0] != 1.0F)
            throw std::invalid_argument("old recorded fixture received the wrong observation");
        for (size_t index = 0; index < input.state.size(); ++index)
            if (input.state[index] != static_cast<float>(index))
                throw std::invalid_argument("old recorded state was changed");
        RobotActionChunk result{{}, 100, 14, false, 101};
        for (int32_t step = 0; step < 100; ++step)
            for (int32_t joint = 0; joint < 14; ++joint)
                result.actions.push_back(static_cast<float>(100 * step + joint - 700));
        return result;
    }
    RobotAction act(const RobotObservation&) override {
        throw std::logic_error("recorded example must not consume a queue");
    }
    void reset() override { throw std::logic_error("stateless example must not reset a queue"); }

  private:
    int calls_{0};
};
} // namespace
extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.reader.info().family != "action_fixture")
        throw std::runtime_error("wrong fixture family");
    if (context.reader.info().task == "missing")
        return new DeclaredOnly;
    if (context.reader.info().task == trtmc::IRobotControl::kTask)
        return new ExistingRecorded;
    return new ActionFixture(context.reader.info().task);
}
// Deterministic synchronization belongs only to this protocol fixture.
extern "C" void trtmc_action_fixture_block_next() {
    std::lock_guard<std::mutex> lock(block_mutex);
    unblock = false;
    entered = false;
    block_next = true;
}
extern "C" int trtmc_action_fixture_entered() {
    return entered.load() ? 1 : 0;
}
extern "C" void trtmc_action_fixture_unblock() {
    {
        std::lock_guard<std::mutex> lock(block_mutex);
        unblock = true;
    }
    block_condition.notify_all();
}
