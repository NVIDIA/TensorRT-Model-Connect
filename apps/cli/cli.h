/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/bundle.h"
#include "trtmc/task.h"

#include <iosfwd>
#include <string>
#include <unordered_map>
#include <vector>

namespace trtmc::cli {

enum class CommandKind {
    kHelp,
    kVersion,
    kInspect,
    kRun,
    kEncode,
    kEmbed,
    kRerank,
    kClassify,
    kExtractFeatures,
    kDisparity,
    kGeometry,
    kSegment,
    kSegmentPrompted,
    kVideoSegment,
    kGenerateAudio,
    kTranscribe,
    kTranscribeBatch,
    kTranscribeStreaming,
    kSpeak,
    kSpeechSession,
    kGenerateImage,
    kGenerateImageBatch,
    kGenerateVideo,
    kSolve,
    kForecast,
    kControl,
    kGenerateWorld,
};

struct Command {
    CommandKind kind{CommandKind::kHelp};
    std::string name;
    std::string bundle;
    std::string runtime_root;
    std::unordered_map<std::string, std::string> options;
    std::vector<std::string> frames;
    std::vector<std::string> inputs;
};

Command parse_args(int argc, char** argv);
int dispatch(const Command& command, ITask& task, std::ostream& output);
void print_usage(std::ostream& output);
int run(int argc, char** argv, std::ostream& output, std::ostream& error);

} // namespace trtmc::cli
