/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "trtmc/internal/video.h"

#include <memory>

namespace trtmc::internal {

struct ImageStateObservation {
    ImageView image;
    Span<const float> state; // One ordered state vector, interpreted by the family.
};
struct ImageStateToActionChunkRequest {
    ImageStateObservation observation;
};
struct ImageStateActionChunkResult {
    ActionSequenceResult actions; // [step,component], in actual unnormalized model output units.
    bool within_training_bounds{false}; // Range metadata, not a physical safety guarantee.
    double inference_ms{0};
};
struct ActionStepResult {
    ActionSequenceResult action; // Exactly one [1,component] row; same action schema.
    bool within_training_bounds{false};
    bool started_new_chunk{false};
    double inference_ms{0}; // Actual family work on this call; zero is meaningful.
};
class IImageStateToActionChunk {
  public:
    using TaskInterface = IImageStateToActionChunk;
    static constexpr std::string_view kTask = "image_state_to_action_chunk";
    virtual ~IImageStateToActionChunk() = default;
    // Does not create, consume or mutate the family's action queue.
    virtual ImageStateActionChunkResult run(const ImageStateToActionChunkRequest&, ConfigView) = 0;
};
class IImageStateActionSession {
  public:
    virtual ~IImageStateActionSession() = default;
    // The family validates each observation and owns refill/consumption policy.
    // Input and config storage is borrowed only through return.
    virtual ActionStepResult act(const ImageStateObservation&, ConfigView) = 0;
    virtual void reset() = 0; // Discard queued actions and reset family execution state.
};
class IImageStateActionQueue {
  public:
    using TaskInterface = IImageStateActionQueue;
    static constexpr std::string_view kTask = "image_state_action_queue";
    virtual ~IImageStateActionQueue() = default;
    // Copy or parse retained config before returning. No shared queue algorithm.
    virtual std::unique_ptr<IImageStateActionSession> create_action_session(ConfigView) = 0;
};
} // namespace trtmc::internal
