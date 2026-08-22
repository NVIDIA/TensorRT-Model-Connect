/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/speech_session.h"

namespace trtmc {

ISpeechSessionProvider::~ISpeechSessionProvider() = default;
ISpeechBatchSessionProvider::~ISpeechBatchSessionProvider() = default;
ISpeechRealtimeControl::~ISpeechRealtimeControl() = default;
ISpeechToolSession::~ISpeechToolSession() = default;
ISpeechToolSessionProvider::~ISpeechToolSessionProvider() = default;

} // namespace trtmc
