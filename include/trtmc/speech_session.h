/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/pipeline.h"

#include <memory>

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

} // namespace trtmc
