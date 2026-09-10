/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace trtmc {

// Public optional family capability for an application that keeps its microphone
// stream open while discarding the dialogue. The synchronous barrier stops
// the old reply and clears generation, transcription, and conversation state.
// Queued input, resampling phase, and the bounded acoustic encoder history
// remain continuous. ISpeechSession::reset() still clears the entire stream;
// ISpeechRealtimeControl::cancel_response() still permits response recreation.
class INemotronVoiceChatLiveControl {
  public:
    virtual ~INemotronVoiceChatLiveControl() = default;
    virtual void reset_conversation_context() = 0;
};

} // namespace trtmc
