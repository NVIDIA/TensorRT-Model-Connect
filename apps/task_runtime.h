/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/task.h"

#include <string_view>

namespace trtmc::app {

// During family-by-family migration, existing bundle modes retain their
// existing application path. A migrated family publishes its semantic Task ID.
// Selection happens before execution; an SDK error never retries the old path.
// Delete this selection with the old application paths after the last migration.
inline bool uses_existing_task_runtime(std::string_view primary_task) noexcept {
    constexpr const char* tasks[] = {
        ITextGeneration::kTask,
        IVisionLanguageGeneration::kTask,
        IImageGeneration::kTask,
        IImageEditing::kTask,
        IImageBatchGeneration::kTask,
        IWorldModelGeneration::kTask,
        IAudioGeneration::kTask,
        ITranscription::kTask,
        IBatchTranscription::kTask,
        IStreamingTranscription::kTask,
        ISpeechToSpeech::kTask,
        ISpeechSessionProvider::kTask,
        ISpeechBatchSessionProvider::kTask,
        ISpeechToolSessionProvider::kTask,
        IEmbedding::kTask,
        IEncoding::kTask,
        IReranking::kTask,
        ISegmentation::kTask,
        IPointPromptedSegmentation::kTask,
        ITextPromptedSegmentation::kTask,
        IStereoDisparity::kTask,
        IMonocularGeometry::kTask,
        IImageClassification::kTask,
        IObjectDetection::kTask,
        IPoseHypothesisRefinement::kTask,
        IImageFeatureExtractor::kTask,
        IVideoSegmentation::kTask,
        INeuralOperator::kTask,
        ITimeSeriesForecast::kTask,
        IRobotControl::kTask,
    };
    for (const auto* task : tasks) {
        if (primary_task == task)
            return true;
    }
    return false;
}

} // namespace trtmc::app
