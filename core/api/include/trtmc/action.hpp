/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "trtmc/action.h"
#include "trtmc/core.hpp"
#include "trtmc/video.hpp"

namespace trtmc {
struct ImageStateObservation {
    ImageInput image;
    Span<const float> state; // Borrowed through the synchronous call.
};
struct ImageStateToActionChunkRequest {
    ImageStateObservation observation;
};

class ImageStateActionChunkResult
    : public detail::ViewResult<trtmc_image_state_action_chunk_view_v1> {
  public:
    using ViewResult::ViewResult;
    FloatMatrixView actions() const { return detail::action_values(view().actions); }
    bool within_training_bounds() const { return view().within_training_bounds != 0; }
    double inference_ms() const { return view().inference_ms; }
};
class ActionStepResult : public detail::ViewResult<trtmc_action_step_view_v1> {
  public:
    using ViewResult::ViewResult;
    Span<const float> values() const {
        const auto matrix = view().action.values;
        return {matrix.data, static_cast<size_t>(matrix.count)};
    }
    bool within_training_bounds() const { return view().within_training_bounds != 0; }
    bool started_new_chunk() const { return view().started_new_chunk != 0; }
    double inference_ms() const { return view().inference_ms; }
};
namespace detail {
inline trtmc_image_state_observation_v1 action_observation(const ImageStateObservation& input) {
    return {input.image.wire, input.state.data(), input.state.size()};
}
template <class Table>
void validate_action_table(const trtmc_api_header* table) {
    if (!table || table->major != 1 || table->minor != 0 || table->byte_size < sizeof(Table))
        throw Error(TRTMC_VERSION_MISMATCH, "incompatible action Task table");
}
} // namespace detail

class ImageStateToActionChunk {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_IMAGE_STATE_TO_ACTION_CHUNK;
    static constexpr uint32_t kMajor = 1, kMinor = 0;
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_action_table<trtmc_image_state_to_action_chunk_api_v1>(table);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(model_, kTask, kMajor, kMinor);
    }
    ImageStateActionChunkResult run(const ImageStateToActionChunkRequest& input,
                                    const Config& config = {}) const {
        const trtmc_image_state_to_action_chunk_request_v1 request{
            detail::action_observation(input.observation)};
        const auto entries = config.c_entries();
        const auto options = entries.view();
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->run(model_->handle, &request, &options, &raw, &error);
        detail::ResultOwner owner(model_, raw);
        detail::check(model_->api, status, error);
        return ImageStateActionChunkResult(std::move(owner), api_->result_view);
    }

  private:
    friend class Model;
    ImageStateToActionChunk(std::shared_ptr<detail::ModelState> model, const trtmc_api_header* api)
        : model_(std::move(model)),
          api_(reinterpret_cast<const trtmc_image_state_to_action_chunk_api_v1*>(api)) {}
    std::shared_ptr<detail::ModelState> model_;
    const trtmc_image_state_to_action_chunk_api_v1* api_;
};
class ImageStateActionSession {
  public:
    ~ImageStateActionSession() { close(); }
    ImageStateActionSession(const ImageStateActionSession&) = delete;
    ImageStateActionSession& operator=(const ImageStateActionSession&) = delete;
    ImageStateActionSession(ImageStateActionSession&& other) noexcept
        : model_(std::move(other.model_)), api_(other.api_),
          handle_(std::exchange(other.handle_, nullptr)) {}
    ImageStateActionSession& operator=(ImageStateActionSession&& other) noexcept {
        if (this != &other) {
            close();
            model_ = std::move(other.model_);
            api_ = other.api_;
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }
    ActionStepResult act(const ImageStateObservation& input, const Config& config = {}) const {
        require_open();
        const auto request = detail::action_observation(input);
        const auto entries = config.c_entries();
        const auto options = entries.view();
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->act(handle_, &request, &options, &raw, &error);
        detail::ResultOwner owner(model_, raw);
        detail::check(model_->api, status, error);
        return ActionStepResult(std::move(owner), api_->result_view);
    }
    void reset() const {
        require_open();
        trtmc_error* error = nullptr;
        const auto status = api_->reset(handle_, &error);
        detail::check(model_->api, status, error);
    }
    // Must not race act/reset. Closing discards the family-owned queue.
    void close() noexcept {
        if (handle_)
            api_->release(std::exchange(handle_, nullptr));
        model_.reset();
    }

  private:
    friend class ImageStateActionQueue;
    ImageStateActionSession(std::shared_ptr<detail::ModelState> model,
                            const trtmc_image_state_action_queue_api_v1* api,
                            trtmc_image_state_action_session* handle) noexcept
        : model_(std::move(model)), api_(api), handle_(handle) {}
    void require_open() const {
        if (!handle_)
            throw Error(TRTMC_INVALID_ARGUMENT, "action session is closed");
    }
    std::shared_ptr<detail::ModelState> model_;
    const trtmc_image_state_action_queue_api_v1* api_;
    trtmc_image_state_action_session* handle_{nullptr};
};
class ImageStateActionQueue {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_IMAGE_STATE_ACTION_QUEUE;
    static constexpr uint32_t kMajor = 1, kMinor = 0;
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_action_table<trtmc_image_state_action_queue_api_v1>(table);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(model_, kTask, kMajor, kMinor);
    }
    ImageStateActionSession create(const Config& config = {}) const {
        const auto entries = config.c_entries();
        const auto options = entries.view();
        trtmc_image_state_action_session* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->create(model_->handle, &options, &raw, &error);
        ImageStateActionSession session(model_, api_, raw);
        detail::check(model_->api, status, error);
        return session;
    }

  private:
    friend class Model;
    ImageStateActionQueue(std::shared_ptr<detail::ModelState> model, const trtmc_api_header* api)
        : model_(std::move(model)),
          api_(reinterpret_cast<const trtmc_image_state_action_queue_api_v1*>(api)) {}
    std::shared_ptr<detail::ModelState> model_;
    const trtmc_image_state_action_queue_api_v1* api_;
};
} // namespace trtmc
