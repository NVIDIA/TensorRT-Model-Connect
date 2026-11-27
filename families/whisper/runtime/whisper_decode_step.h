/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// One decode step of a Whisper decoder whose copies ride the engine's wait.
//
// A step enqueues the decoder, downloads its logits and moves the step's K/V
// rows into the cache. Issued after the wait for the engine, the download and
// the row copies keep the device idle while the host issues them. Issued on
// the decoder's stream right after the enqueue, they run behind the engine and
// one wait covers all of them: the same copies of the same bytes, in the same
// order on that stream.
//
// Templates over the module and the state, so the facts and the order can be
// checked without a GPU.

#include <cstddef>
#include <cstdint>
#include <string>

namespace trtmc {

constexpr const char* kWhisperLogitsOutput = "logits";

// Whether the factory asks its pipeline for the step that queues its copies:
// a single-GPU decoder on the standard TensorRT backend. The queued order is
// not validated on a tensor-parallel decoder or on any other backend, so
// those keep the step that waits for the engine before its copies.
inline bool whisper_queues_decode_step_copies(int32_t tensor_parallel_size,
                                              const char* backend_name) {
    return tensor_parallel_size == 1 && backend_name != nullptr &&
           std::string(backend_name) == "trt";
}

// The step a pipeline runs, in words, for the line the factory prints at load.
inline const char* whisper_decode_step_description(bool queues_copies) {
    return queues_copies ? "copies queued behind the engine"
                         : "waits for the engine before its copies";
}

// What the factory does with its new pipeline: asks it for the queued step
// where the choice above says so (the pipeline throws if it cannot run one),
// and returns the line to print, read from the pipeline.
template <typename Pipeline>
inline std::string whisper_choose_decode_step(Pipeline& pipeline, int32_t tensor_parallel_size,
                                              const char* backend_name) {
    if (whisper_queues_decode_step_copies(tensor_parallel_size, backend_name))
        pipeline.queue_decode_step_copies();
    return std::string("[whisper] decode step: ") +
           whisper_decode_step_description(pipeline.decode_steps_queue_copies());
}

// How many values the decoder's logits output holds: the product of its
// dimensions, zero for a decoder without that output.
template <typename Module>
inline std::size_t whisper_logits_count(const Module& decoder) {
    const auto shape = decoder.tensor_shape(kWhisperLogitsOutput);
    if (shape.empty())
        return 0;
    std::size_t count = 1;
    for (const int64_t dim : shape)
        count *= static_cast<std::size_t>(dim);
    return count;
}

// Why a step of `decoder` cannot queue its copies behind the engine; empty
// when it can. Such a step needs a state whose advance() only queues work on
// the decoder's stream, and logits that are a fixed number of float32 values:
// no input is dynamic (an engine's output sizes follow its input shapes).
template <typename Module, typename State, typename DTypeT>
inline std::string whisper_queued_step_refusal(const Module& decoder, const State& state,
                                               DTypeT float32) {
    if (!state.advance_queues_on(decoder.stream()))
        return "the inference state does not advance on the decoder's stream";
    if (!decoder.has_output(kWhisperLogitsOutput) || whisper_logits_count(decoder) == 0)
        return "the decoder has no logits: rebuild the bundle";
    if (decoder.tensor_dtype(kWhisperLogitsOutput) != float32)
        return "the decoder's logits are not float32: rebuild the bundle";
    for (const auto& input : decoder.input_info()) {
        if (decoder.input_is_dynamic(input.name))
            return "the decoder's input " + input.name + " is dynamic: rebuild the bundle";
    }
    return {};
}

// One step: the enqueue; behind it on the decoder's stream the logits download
// (queue_logits_download is handed the device logits and says whether it
// queued the copy) and the state's advance(); then the one wait. The wait is
// taken on every way out, so nothing issued for the step is still running
// when it returns or throws. Returns whether the download was queued.
template <typename Module, typename State, typename Inputs, typename DownloadFn>
inline bool run_whisper_queued_decode_step(Module& decoder, State& state, const Inputs& inputs,
                                           DownloadFn&& queue_logits_download) {
    bool queued = false;
    try {
        decoder.forward_async(inputs);
        queued = queue_logits_download(decoder.device_ptr(kWhisperLogitsOutput));
        state.advance();
    } catch (...) {
        decoder.sync();
        throw;
    }
    decoder.sync();
    return queued;
}

} // namespace trtmc
