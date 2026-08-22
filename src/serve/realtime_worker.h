/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/pipeline.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <memory>

namespace trtmc::serve {

// Transport limits for one native JSONL realtime session. The defaults keep
// every input and output line below 1 MiB, accept at most 100 ms of 24 kHz
// mono PCM16 per append, and bound both queued commands and session lifetime.
struct RealtimeWorkerLimits {
    std::size_t max_line_bytes{1024U * 1024U};
    std::size_t max_audio_bytes{4800U};
    // Keep transport lifetime below the model-owned recurrent/TTS cache
    // horizon after accounting for agent output and function-channel steps.
    std::size_t max_session_audio_samples{24000U * 60U * 5U};
    std::size_t max_session_config_bytes{256U * 1024U};
    std::size_t max_queued_commands{64U};
    std::size_t max_queued_bytes{4U * 1024U * 1024U};
    std::size_t max_native_events_per_drain{256U};
    std::size_t max_native_audio_samples_per_drain{24000U};
    std::size_t max_event_text_bytes{256U * 1024U};
    std::size_t max_event_audio_samples{2400U};
    int32_t close_drain_timeout_ms{120000};
    int32_t idle_poll_ms{5};
};

using RealtimePipelineFactory = std::function<std::unique_ptr<IPipeline>()>;

// Successful commit, clear, cancel, and truncate controls emit the internal
// JSONL acknowledgements input_audio_buffer.committed,
// input_audio_buffer.cleared, response.cancelled, and
// conversation.item.truncated. response.create emits no synthetic success;
// the native kTurnStarted event is the response-creation authority.

// Run one generic native realtime transport session. The caller retains
// ownership of pipeline and both streams. Returns zero after an orderly close
// or EOF and non-zero only when the worker itself cannot be started or read.
int run_realtime_worker(IPipeline& pipeline, std::istream& input, std::ostream& output,
                        const RealtimeWorkerLimits& limits = {});

// Factory overload for callers that want the worker to own the pipeline. The
// factory is invoked exactly once before session.ready is emitted.
int run_realtime_worker(const RealtimePipelineFactory& factory, std::istream& input,
                        std::ostream& output, const RealtimeWorkerLimits& limits = {});

} // namespace trtmc::serve
