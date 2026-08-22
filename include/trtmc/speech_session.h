/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/pipeline.h"

#include <cstdint>
#include <memory>
#include <string>

namespace trtmc {

// Optional capability implemented only by persistent speech-session
// pipelines. Keeping it separate from IPipeline preserves the optimized
// runtime ABI for providers compiled against an earlier public header.
class ISpeechSessionProvider {
  public:
    virtual ~ISpeechSessionProvider();
    virtual std::unique_ptr<ISpeechSession>
    create_speech_session(const SpeechSessionConfig& cfg = {}) = 0;
};

// Optional capability for finite-recording inference. Batch sessions preserve
// the model's native frame-by-frame decisions and do not apply live transport
// policies such as autonomous silence ticks or RNN-T turn forcing. Keeping the
// factory separate avoids changing ISpeechSessionProvider or
// SpeechSessionConfig across dynamically loaded providers.
class ISpeechBatchSessionProvider {
  public:
    virtual ~ISpeechBatchSessionProvider();
    virtual std::unique_ptr<ISpeechSession>
    create_batch_speech_session(const SpeechSessionConfig& cfg = {}) = 0;
};

// Optional full-duplex controls implemented by speech sessions that can keep
// accepting input while a response is in flight. Keeping these methods off
// ISpeechSession preserves the existing session vtable and value-type ABI.
class ISpeechRealtimeControl {
  public:
    virtual ~ISpeechRealtimeControl();

    // Commit seals the current input buffer without closing the persistent
    // session. create_response=false supports protocols that acknowledge the
    // input item before a later, independent response.create request.
    virtual void commit_input_turn(bool create_response = true) = 0;
    // Start one response from the current response-available conversation point.
    virtual void create_response() = 0;
    virtual void clear_pending_input() = 0;
    virtual void cancel_response() = 0;

    // This is a model/conversation-state rollback to the point the caller
    // actually played, not merely an output-queue hint. Implementations must
    // perform that rollback or report failure; callers cannot assume success
    // just because they stopped playback.
    virtual void truncate_response(std::uint64_t epoch, std::int64_t played_output_samples) = 0;
};

// Tool definitions use the OpenAI function-tool JSON shape. Keeping this
// configuration on a new optional interface avoids changing the size of
// SpeechSessionConfig across dynamically loaded runtime providers.
struct SpeechToolSessionConfig {
    std::string tools_json;
    // Optional JSON object mapping a tool name (or "default") to a short,
    // ASCII, TTS-friendly phrase spoken while the caller executes the tool.
    std::string on_hold_messages_json;
};

// Optional capability implemented by a persistent speech session whose model
// exposes a function channel. A response is accepted only for a currently
// pending kFunctionCall from the same epoch.
class ISpeechToolSession {
  public:
    virtual ~ISpeechToolSession();
    virtual void submit_tool_response(std::uint64_t epoch, const std::string& call_id,
                                      const std::string& output) = 0;
};

// Separate factory so existing speech providers and SpeechSessionConfig keep
// their ABI. The returned object also implements ISpeechToolSession.
class ISpeechToolSessionProvider {
  public:
    virtual ~ISpeechToolSessionProvider();
    virtual std::unique_ptr<ISpeechSession>
    create_tool_speech_session(const SpeechSessionConfig& session_config,
                               const SpeechToolSessionConfig& tool_config) = 0;
};

} // namespace trtmc
