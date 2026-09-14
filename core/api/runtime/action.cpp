/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/internal/action.h"

#include "api_internal.h"
#include "trtmc/action.h"

#include <cmath>

struct trtmc_image_state_action_session {
    std::shared_ptr<std::mutex> operation = std::make_shared<std::mutex>();
    std::shared_ptr<trtmc::api::ModelState> owner;
    std::unique_ptr<trtmc::api::ModelSession> model;
    std::unique_ptr<trtmc::internal::IImageStateActionSession> implementation;
    ~trtmc_image_state_action_session() {
        // Keep the reservation and family state until an independent chunk ends.
        // No chunk waits for operation while holding model_mutex.
        std::lock_guard<std::mutex> lock(*operation);
        implementation.reset();
        model.reset();
    }
};

namespace trtmc::api {
namespace {
// std::mutex::try_lock requires that this thread does not already own it.
// Track only action entrypoints; existing mutexes still exclude other threads.
class ActionCall {
  public:
    explicit ActionCall(const ModelState* model) : model_(model), previous_(current_) {
        for (const auto* call = current_; call; call = call->previous_)
            if (call->model_ == model_)
                throw ApiFailure{TRTMC_BUSY, "model is executing a reentrant action operation"};
        current_ = this;
    }
    ~ActionCall() { current_ = previous_; }
    ActionCall(const ActionCall&) = delete;
    ActionCall& operator=(const ActionCall&) = delete;

  private:
    const ModelState* model_;
    const ActionCall* previous_;
    inline static thread_local const ActionCall* current_ = nullptr;
};

const std::shared_ptr<ModelState>& session_owner(trtmc_image_state_action_session* session) {
    require(session && session->implementation, "action session is null");
    return session->owner;
}

void output_check(bool condition, const char* message) {
    if (!condition)
        throw ApiFailure{TRTMC_INTERNAL_ERROR, message};
}
std::unique_lock<std::mutex> lock_model(const std::shared_ptr<ModelState>& owner) {
    std::unique_lock<std::mutex> lock(model_mutex(owner), std::try_to_lock);
    if (!lock.owns_lock())
        throw ApiFailure{TRTMC_BUSY, "model is executing another operation"};
    return lock;
}
internal::ImageStateObservation observation(const trtmc_image_state_observation_v1& input) {
    const auto state = checked_span(input.state, input.state_count);
    require(!state.empty(), "action observation requires a state vector");
    return {image_input(input.image), state};
}
void timing(double milliseconds) {
    output_check(std::isfinite(milliseconds) && milliseconds >= 0,
                 "action inference timing is invalid");
}
struct ActionChunkStorage final : ResultStorage {
    explicit ActionChunkStorage(internal::ImageStateActionChunkResult result)
        : actions(std::move(result.actions)) {
        output_check(actions.view.values.rows > 0, "action chunk must contain at least one step");
        timing(result.inference_ms);
        view = {actions.view, result.within_training_bounds ? 1U : 0U, result.inference_ms};
    }
    ActionSequenceStorage actions;
    trtmc_image_state_action_chunk_view_v1 view{};
};
struct ActionStepStorage final : ResultStorage {
    explicit ActionStepStorage(internal::ActionStepResult result)
        : action(std::move(result.action)) {
        output_check(action.view.values.rows == 1, "queued action must contain exactly one step");
        timing(result.inference_ms);
        view = {action.view, result.within_training_bounds ? 1U : 0U,
                result.started_new_chunk ? 1U : 0U, result.inference_ms};
    }
    ActionSequenceStorage action;
    trtmc_action_step_view_v1 view{};
};
template <class Storage, class View>
trtmc_status TRTMC_CALL result_view(const trtmc_result* result, View* output,
                                    trtmc_error** error) noexcept {
    if (output)
        *output = {};
    return guarded(error, [&] {
        require(output != nullptr, "action result view output is null");
        *output = require_result<Storage>(result).view;
    });
}
trtmc_status TRTMC_CALL predict(trtmc_model* model,
                                const trtmc_image_state_to_action_chunk_request_v1* input,
                                const trtmc_config_view_v1* config, trtmc_result** output,
                                trtmc_error** error) noexcept {
    if (output)
        *output = nullptr;
    return guarded(error, [&] {
        require(input && output, "action chunk request and output are required");
        const internal::ImageStateToActionChunkRequest request{observation(input->observation)};
        const ConvertedConfig options(config);
        const auto owner = model_owner(model);
        const ActionCall call(owner.get());
        auto lock = lock_model(owner);
        const auto operation = action_chunk_operation(owner);
        std::unique_lock<std::mutex> queue_lock;
        if (operation) {
            queue_lock = std::unique_lock<std::mutex>(*operation, std::try_to_lock);
            if (!queue_lock.owns_lock())
                throw ApiFailure{TRTMC_BUSY, "another action session operation is active"};
            lock.unlock();
        }
        const auto key = internal::contract_key<internal::IImageStateToActionChunk>();
        auto& family =
            *static_cast<internal::IImageStateToActionChunk*>(task_implementation(owner, key));
        validate_task_config(owner, key, options.view());
        *output = make_result<ActionChunkStorage>(family.run(request, options.view()));
    });
}
trtmc_status TRTMC_CALL create(trtmc_model* model, const trtmc_config_view_v1* config,
                               trtmc_image_state_action_session** output,
                               trtmc_error** error) noexcept {
    if (output)
        *output = nullptr;
    return guarded(error, [&] {
        require(output != nullptr, "action session output is null");
        const ConvertedConfig options(config);
        auto session = std::make_unique<trtmc_image_state_action_session>();
        const auto owner = model_owner(model);
        const ActionCall call(owner.get());
        {
            auto lock = lock_model(owner);
            auto& family = require_interface<internal::IImageStateActionQueue>(
                model, internal::IImageStateActionQueue::kTask);
            session->owner = owner;
            validate_task_config(session->owner,
                                 internal::contract_key<internal::IImageStateActionQueue>(),
                                 options.view());
            session->model = std::make_unique<ModelSession>(model, session->operation);
            session->implementation = family.create_action_session(options.view());
        }
        output_check(session->implementation != nullptr, "family returned no action session");
        *output = session.release();
    });
}
std::unique_lock<std::mutex> operation(trtmc_image_state_action_session* session) {
    std::unique_lock<std::mutex> lock(*session->operation, std::try_to_lock);
    if (!lock.owns_lock())
        throw ApiFailure{TRTMC_BUSY, "another action session operation is active"};
    return lock;
}
trtmc_status TRTMC_CALL act(trtmc_image_state_action_session* session,
                            const trtmc_image_state_observation_v1* input,
                            const trtmc_config_view_v1* config, trtmc_result** output,
                            trtmc_error** error) noexcept {
    if (output)
        *output = nullptr;
    return guarded(error, [&] {
        require(input && output, "action observation and output are required");
        const ActionCall call(session_owner(session).get());
        auto lock = operation(session);
        const auto request = observation(*input);
        const ConvertedConfig options(config);
        validate_task_config(session->owner,
                             internal::contract_key<internal::IImageStateActionQueue>(),
                             options.view());
        *output =
            make_result<ActionStepStorage>(session->implementation->act(request, options.view()));
    });
}
trtmc_status TRTMC_CALL reset(trtmc_image_state_action_session* session,
                              trtmc_error** error) noexcept {
    return guarded(error, [&] {
        const ActionCall call(session_owner(session).get());
        auto lock = operation(session);
        session->implementation->reset();
    });
}
void TRTMC_CALL release(trtmc_image_state_action_session* session) noexcept {
    delete session;
}

const trtmc_image_state_to_action_chunk_api_v1 chunk_api = {
    {1, 0, sizeof(trtmc_image_state_to_action_chunk_api_v1)},
    predict,
    result_view<ActionChunkStorage, trtmc_image_state_action_chunk_view_v1>};
const trtmc_image_state_action_queue_api_v1 queue_api = {
    {1, 0, sizeof(trtmc_image_state_action_queue_api_v1)},     create, act,
    result_view<ActionStepStorage, trtmc_action_step_view_v1>, reset,  release};
static_assert(offsetof(trtmc_image_state_to_action_chunk_api_v1, header) == 0);
static_assert(offsetof(trtmc_image_state_action_queue_api_v1, header) == 0);
const TaskBinding bindings[] = {
    {internal::IImageStateToActionChunk::kTask, 1, 0, &chunk_api.header},
    {internal::IImageStateActionQueue::kTask, 1, 0, &queue_api.header}};
} // namespace
Span<const TaskBinding> action_task_bindings() noexcept {
    return bindings;
}
} // namespace trtmc::api
