/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/task.h"

#include <string>
#include <vector>

namespace trtmc::vibevoice_asr {

struct ParsedTranscript {
    bool structured{false}; // the output parsed as a list of segments
    std::string text;       // Contents joined by one space, or the unparsed output
    std::vector<TranscriptionSegment> segments;
};

// Parse the decoded model output, e.g.
//   assistant\n[{"Start":0.0,"End":7.56,"Speaker":0,"Content":"..."}]\n
// The leading role name is dropped; a ```json fence and a single segment object
// are accepted as the official post-processing does; a list truncated by the
// token limit keeps its complete segments. Other output is returned stripped.
ParsedTranscript parse_transcript(const std::string& decoded);

} // namespace trtmc::vibevoice_asr
