/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The host side of a Whisper decode step that queues its copies behind the
// engine, without a GPU and over stand-ins: when a decoder and a state can run
// such a step, which step the factory asks for, what the step issues and in
// which order, and that every way out of it waits; the decode loop over either holder of the
// logits; the token selection over a pointer; and the page-locked buffer's ownership, against CPU
// CUDA stubs. test_whisper_decode_pipeline.cpp runs the pipeline itself.

#include "families/whisper/runtime/decode_runtime.h"
#include "families/whisper/runtime/device_buffer.h"
#include "families/whisper/runtime/whisper_decode_policy.h"
#include "families/whisper/runtime/whisper_decode_step.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

// --- CPU CUDA stubs for the page-locked buffer -----------------------------

bool g_fail_host_allocation = false;
std::set<void*> g_host_outstanding;
int g_host_frees = 0;
std::uintptr_t g_next_host_address = 0x2000;

} // namespace

extern "C" {

cudaError_t cudaMallocHost(void** ptr, size_t size) {
    (void)size;
    if (g_fail_host_allocation) {
        // A failed call may leave anything behind: the owner must not keep it.
        *ptr = reinterpret_cast<void*>(std::uintptr_t{0xdead});
        return cudaErrorMemoryAllocation;
    }
    void* address = reinterpret_cast<void*>(g_next_host_address);
    g_next_host_address += 0x1000;
    g_host_outstanding.insert(address);
    *ptr = address;
    return cudaSuccess;
}

cudaError_t cudaFreeHost(void* ptr) {
    ++g_host_frees;
    check(g_host_outstanding.erase(ptr) == 1, "only a held page-locked buffer is freed");
    return cudaSuccess;
}

} // extern "C"

namespace {

// --- Stand-ins --------------------------------------------------------------

constexpr int kFloat32 = 0;
constexpr int kFloat16 = 1;
constexpr int32_t kVocab = 37;
constexpr int kStreamA = 1;
constexpr int kStreamB = 2;

struct InputInfo {
    std::string name;
};

// What a step issued, in order. The stand-ins only record it: the pipeline
// test runs the work.
std::vector<std::string> g_issued;

// The members of ITrtModule a decode step and its conditions call.
struct Decoder {
    int stream_{kStreamA};
    std::vector<int64_t> logits_shape{1, kVocab};
    int logits_dtype{kFloat32};
    bool has_logits{true};
    std::string dynamic_input;
    bool fail_enqueue{false};
    float device_logits{0.0F};

    int stream() const { return stream_; }
    bool has_output(const std::string& name) const { return has_logits && name == "logits"; }
    int tensor_dtype(const std::string&) const { return logits_dtype; }
    std::vector<int64_t> tensor_shape(const std::string& name) const {
        return has_output(name) ? logits_shape : std::vector<int64_t>{};
    }
    std::vector<InputInfo> input_info() const {
        return {{"token_id"}, {"position_id"}, {"attention_mask"}, {"cache_k_0"}};
    }
    bool input_is_dynamic(const std::string& name) const { return name == dynamic_input; }
    void* device_ptr(const std::string& name) {
        return name == "logits" ? &device_logits : nullptr;
    }
    // As the module: the uploads are issued first, then the engine; the stage
    // timer's end event is recorded when the enqueue returns.
    void forward_async(int) {
        g_issued.emplace_back("uploads");
        if (fail_enqueue)
            throw std::runtime_error("enqueue failed");
        g_issued.emplace_back("engine");
    }
    void sync() { g_issued.emplace_back("wait"); }
};

struct State {
    int stream{kStreamA};
    bool throw_on_advance{false};
    int position{0};

    bool advance_queues_on(int other) const { return other == stream; }
    void advance() {
        if (throw_on_advance)
            throw std::runtime_error("advance failed");
        g_issued.emplace_back("row copies");
        ++position;
    }
};

// --- Checks -----------------------------------------------------------------

void check_conditions() {
    const auto refusal = [](const Decoder& decoder, const State& state) {
        return trtmc::whisper_queued_step_refusal(decoder, state, kFloat32);
    };
    {
        Decoder decoder;
        State state;
        check(refusal(decoder, state).empty(),
              "fixed float32 logits behind a state on the decoder's stream can queue");
        check(trtmc::whisper_logits_count(decoder) == kVocab, "the logits count is the vocabulary");
        decoder.logits_shape = {2, 3, 5};
        check(trtmc::whisper_logits_count(decoder) == 30,
              "the logits count is the product of the dimensions");
    }
    {
        Decoder decoder;
        State state;
        state.stream = kStreamB;
        check(refusal(decoder, state) ==
                  "the inference state does not advance on the decoder's stream",
              "a state that advances on another stream is refused");
    }
    {
        Decoder decoder;
        State state;
        decoder.has_logits = false;
        check(trtmc::whisper_logits_count(decoder) == 0 &&
                  refusal(decoder, state) == "the decoder has no logits: rebuild the bundle",
              "a decoder without a logits output is refused");
    }
    {
        Decoder decoder;
        State state;
        decoder.logits_shape = {1, 0};
        check(refusal(decoder, state) == "the decoder has no logits: rebuild the bundle",
              "empty logits are refused");
    }
    {
        Decoder decoder;
        State state;
        decoder.logits_dtype = kFloat16;
        check(refusal(decoder, state) == "the decoder's logits are not float32: rebuild the bundle",
              "logits that are not float32 are refused");
    }
    {
        Decoder decoder;
        State state;
        decoder.dynamic_input = "attention_mask";
        check(refusal(decoder, state) ==
                  "the decoder's input attention_mask is dynamic: rebuild the bundle",
              "a dynamic input is refused, by name");
    }
}

// The factory's choice of step, and the words of its line at load.
void check_factory_choice() {
    check(trtmc::whisper_queues_decode_step_copies(1, "trt"),
          "a single-GPU decoder on the standard TensorRT backend queues its copies");
    check(!trtmc::whisper_queues_decode_step_copies(2, "trt") &&
              !trtmc::whisper_queues_decode_step_copies(4, "trt") &&
              !trtmc::whisper_queues_decode_step_copies(0, "trt"),
          "a tensor-parallel decoder waits first");
    check(!trtmc::whisper_queues_decode_step_copies(1, "trt_rtx") &&
              !trtmc::whisper_queues_decode_step_copies(1, "TRT") &&
              !trtmc::whisper_queues_decode_step_copies(1, "") &&
              !trtmc::whisper_queues_decode_step_copies(1, nullptr),
          "another backend, or none named, waits first");
    check(std::string(trtmc::whisper_decode_step_description(true)) ==
                  "copies queued behind the engine" &&
              std::string(trtmc::whisper_decode_step_description(false)) ==
                  "waits for the engine before its copies",
          "the line at load names the step");
}

void check_step_order() {
    Decoder decoder;
    State state;
    g_issued.clear();
    const bool queued = trtmc::run_whisper_queued_decode_step(
        decoder, state, 0, [&decoder](const void* device_logits) {
            check(device_logits == &decoder.device_logits,
                  "the download is handed the decoder's device logits");
            g_issued.emplace_back("logits download");
            return true;
        });
    check(queued, "a step that queued its download says so");
    // Nothing is issued before the enqueue returns, where the module records
    // the stage timer's end event: the timer covers the engine as before.
    check(g_issued == std::vector<std::string>{"uploads", "engine", "logits download", "row copies",
                                               "wait"},
          "a step issues the engine, the download, the row copies, then one wait");
    check(state.position == 1, "a step advances the state once");
}

void check_every_way_out_waits() {
    const auto ends_in_one_wait = [] {
        return !g_issued.empty() && g_issued.back() == "wait" &&
               std::count(g_issued.begin(), g_issued.end(), std::string("wait")) == 1;
    };
    {
        Decoder decoder;
        State state;
        g_issued.clear();
        bool threw = false;
        try {
            trtmc::run_whisper_queued_decode_step(decoder, state, 0, [](const void*) -> bool {
                throw std::runtime_error("download failed");
            });
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check(threw && ends_in_one_wait(), "a download that throws is passed on after the wait");
    }
    {
        Decoder decoder;
        State state;
        state.throw_on_advance = true;
        g_issued.clear();
        bool threw = false;
        try {
            trtmc::run_whisper_queued_decode_step(decoder, state, 0,
                                                  [](const void*) { return true; });
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check(threw && ends_in_one_wait(), "an advance that throws is passed on after the wait");
    }
    {
        // The module has issued the uploads when its enqueue fails.
        Decoder decoder;
        State state;
        decoder.fail_enqueue = true;
        g_issued.clear();
        bool threw = false;
        bool download_called = false;
        try {
            trtmc::run_whisper_queued_decode_step(decoder, state, 0,
                                                  [&download_called](const void*) {
                                                      download_called = true;
                                                      return true;
                                                  });
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check(threw && g_issued == std::vector<std::string>{"uploads", "wait"},
              "a failed enqueue is passed on after the wait");
        check(!download_called && state.position == 0,
              "a failed enqueue issues no download and does not advance the state");
    }
    {
        Decoder decoder;
        State state;
        g_issued.clear();
        const bool queued = trtmc::run_whisper_queued_decode_step(
            decoder, state, 0, [](const void*) { return false; });
        check(!queued && ends_in_one_wait(), "a download that was not queued is reported");
        check(state.position == 1, "the state advances whether or not the download queued");
    }
}

void check_decode_loop() {
    // A scripted step: the logits after the prompt name script[0], after each
    // further step the next entry; the fourth selection is the end-of-text
    // token.
    constexpr int32_t eot = 5;
    const std::vector<int32_t> prompt = {1, 2, 3, 4};
    const std::vector<int32_t> script = {7, 9, 11, eot, 13};
    const auto scripted = [&](std::size_t calls, std::vector<float>& logits) {
        const std::size_t after_prompt = calls > prompt.size() ? calls - prompt.size() : 0;
        logits.assign(kVocab, 0.0F);
        logits[static_cast<std::size_t>(script[std::min(after_prompt, script.size() - 1)])] = 1.0F;
    };
    for (const int32_t limit : {0, 1, 3, 4, 10}) {
        std::size_t calls_a = 0;
        const auto a = trtmc::run_whisper_decode_loop(
            prompt, limit, eot,
            [&](int32_t, std::vector<float>& logits, std::string&) {
                scripted(++calls_a, logits);
                return true;
            },
            [](const std::vector<float>& logits) {
                return trtmc::whisper_select_argmax_token(logits);
            });
        std::size_t calls_b = 0;
        std::vector<float> storage;
        const auto b = trtmc::run_whisper_decode_loop_over<trtmc::WhisperLogitsView>(
            prompt, limit, eot,
            [&](int32_t, trtmc::WhisperLogitsView& logits, std::string&) {
                scripted(++calls_b, storage);
                logits = {storage.data(), storage.size()};
                return true;
            },
            [](const trtmc::WhisperLogitsView& logits) {
                return trtmc::whisper_select_argmax_token(logits.data, logits.size);
            });
        check(a.output_ids == b.output_ids && calls_a == calls_b,
              "the loop gives the same tokens over a vector and over a view");
        check(!a.prefill_failed && !a.decode_failed && !b.prefill_failed && !b.decode_failed,
              "a scripted run does not fail");
        const std::vector<int32_t> expected(script.begin(),
                                            script.begin() + std::min<int32_t>(limit, 4));
        check(b.output_ids == expected, "the loop stops at the limit or at the end-of-text token");
    }
    {
        const auto none = trtmc::run_whisper_decode_loop_over<trtmc::WhisperLogitsView>(
            prompt, 5, eot, [](int32_t, trtmc::WhisperLogitsView&, std::string&) { return true; },
            [](const trtmc::WhisperLogitsView&) { return 0; });
        check(none.output_ids.empty() && !none.prefill_failed && !none.decode_failed,
              "a view with no logits after the prompt selects nothing");
    }
    {
        int calls = 0;
        const auto failed = trtmc::run_whisper_decode_loop_over<trtmc::WhisperLogitsView>(
            prompt, 5, eot,
            [&calls](int32_t, trtmc::WhisperLogitsView&, std::string& error) {
                error = "step failed";
                return ++calls < 2;
            },
            [](const trtmc::WhisperLogitsView&) { return 0; });
        check(failed.prefill_failed && failed.output_ids.empty() && failed.error == "step failed",
              "a failed prompt step is reported");
    }
}

void check_selection() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<std::vector<float>> cases = {
        {1.0F},
        {1.0F, 2.0F, 3.0F},
        {3.0F, 2.0F, 1.0F},
        {2.0F, 5.0F, 5.0F, 1.0F},
        {-0.0F, 0.0F, -0.0F},
        {0.0F, -0.0F},
        {nan, 9.0F, 1.0F},
        {1.0F, nan, 9.0F, nan},
        {nan, nan},
        {-inf, -inf, -inf},
        {-inf, nan, -inf},
        {1.0F, inf, inf, 2.0F},
        {-1.0F, -2.0F, -0.5F, -0.5F},
    };
    for (const auto& logits : cases) {
        const int32_t expected = static_cast<int32_t>(
            std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
        check(trtmc::whisper_select_argmax_token(logits.data(), logits.size()) == expected,
              "the selection over a pointer is std::max_element's");
        check(trtmc::whisper_select_argmax_token(logits) == expected,
              "the selection over a vector is std::max_element's");
    }
    const float one = 1.0F;
    check(trtmc::whisper_select_argmax_token(nullptr, 0) == 0 &&
              trtmc::whisper_select_argmax_token(nullptr, 3) == 0 &&
              trtmc::whisper_select_argmax_token(&one, 0) == 0 &&
              trtmc::whisper_select_argmax_token(std::vector<float>{}) == 0,
          "no logits select token 0");
}

void check_page_locked_buffer() {
    {
        trtmc::whisper::PinnedHostBuffer buffer;
        check(buffer.get() == nullptr, "a new buffer holds nothing");
        check(buffer.allocate(64) == cudaSuccess && buffer.get() != nullptr,
              "an allocation is held");
        const void* first = buffer.get();
        check(buffer.allocate(128) == cudaSuccess && buffer.get() != first &&
                  g_host_outstanding.size() == 1,
              "a second allocation frees the first");
        g_fail_host_allocation = true;
        check(buffer.allocate(256) != cudaSuccess && buffer.get() == nullptr &&
                  g_host_outstanding.empty(),
              "a failed allocation leaves the buffer empty and frees what it held");
        g_fail_host_allocation = false;
        check(buffer.allocate(64) == cudaSuccess && g_host_outstanding.size() == 1,
              "a buffer can be allocated again after a failure");
    }
    check(g_host_outstanding.empty(), "leaving scope frees the buffer");
    const int frees = g_host_frees;
    {
        trtmc::whisper::PinnedHostBuffer never_allocated;
    }
    check(g_host_frees == frees, "a buffer that holds nothing frees nothing");
}

} // namespace

int main() {
    check_conditions();
    check_factory_choice();
    check_step_order();
    check_every_way_out_waits();
    check_decode_loop();
    check_selection();
    check_page_locked_buffer();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("whisper decode-step checks passed\n");
    return 0;
}
