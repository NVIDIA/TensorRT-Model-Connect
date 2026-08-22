/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/bundle.h"
#include "trtmc/pipeline.h"

#include <iosfwd>

namespace trtmc::serve {

// Run the private JSONL protocol used by the Python HTTP facade. The caller
// owns `pipeline` for the full call. Requests are deliberately serialized so a
// single pipeline's mutable execution state is never used concurrently.
//
// Requests use {"id": <non-empty-string>, "op": <name>, ...}; responses use
// {"id": ..., "ok": true, "result": {...}} or
// {"id": ..., "ok": false, "error": {"type": ..., "message": ...}}.
// A worker owns at most one transcription stream. Streaming audio chunks carry
// base64-encoded little-endian mono PCM16 in the `audio` field.
//
// A ready event is written before the first request is read. The loop exits on
// EOF or after replying to a `shutdown` request.
int run_worker_protocol(IPipeline& pipeline, const BundleInfo& bundle_info, std::istream& input,
                        std::ostream& output);

} // namespace trtmc::serve
