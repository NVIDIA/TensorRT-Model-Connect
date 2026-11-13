/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/voxtral/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"

#include <stdexcept>
#include <string>

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("voxtral does not support --kv-cache-size");
    const auto& task = context.reader.info().task;
    if (task != trtmc::internal::ISpeechTranscription::kTask)
        throw std::runtime_error("voxtral bundle declares an unsupported task: " + task);
    return new trtmc::voxtral::VoxtralPipeline(context);
}
