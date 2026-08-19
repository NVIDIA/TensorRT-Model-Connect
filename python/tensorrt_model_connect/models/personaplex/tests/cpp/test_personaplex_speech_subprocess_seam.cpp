/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-15
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-01
// Intent:         Speech subprocess seam: output plan resample, frame rate clamping, token parsing
// Preconditions:  Output plan with various target/max output configurations
// Postconditions: Resample path correct, frame rate clamped, tokens parsed from subprocess output
// =============================================================================

#include "speech_generation_policy.h"
#include "subprocess_runner.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

void check_contains(const std::string& text, const std::string& needle, const char* test_name) {
    check(text.find(needle) != std::string::npos, test_name);
}

void test_output_plan_resample_path() {
    trtmc::SpeechOutputPlanInput input;
    input.sample_rate = 24000;
    input.frame_rate = 12.5F;
    input.num_frames = 50;
    input.num_input_samples = 96000;
    input.input_sample_rate = 48000;
    input.tail_frames = 5;
    input.max_output_frames = 100;
    input.max_delay = 4;

    const auto plan = trtmc::ComputeSpeechOutputPlan(input);

    check(plan.effective_frames == 23, "output plan: resample effective frames");
    check(plan.extra_tail == 5, "output plan: resample extra tail");
    check(plan.output_frames == 28, "output plan: resample output frames");
    check(plan.total_iters == 33, "output plan: resample total iters");
}

void test_output_plan_frame_rate_disabled_and_clamped() {
    trtmc::SpeechOutputPlanInput input;
    input.sample_rate = 24000;
    input.frame_rate = 0.0F;
    input.num_frames = 10;
    input.num_input_samples = 12345;
    input.input_sample_rate = 16000;
    input.tail_frames = -3;
    input.max_output_frames = 4;
    input.max_delay = 2;

    const auto plan = trtmc::ComputeSpeechOutputPlan(input);

    check(plan.effective_frames == 8, "output plan: disabled frame-rate effective frames");
    check(plan.extra_tail == 0, "output plan: negative tail clamps to zero");
    check(plan.output_frames == 4, "output plan: max output clamp");
    check(plan.total_iters == 7, "output plan: disabled frame-rate total iters");
}

void test_output_plan_small_inputs_do_not_go_negative() {
    trtmc::SpeechOutputPlanInput input;
    input.sample_rate = 24000;
    input.frame_rate = 12.5F;
    input.num_frames = 1;
    input.num_input_samples = 0;
    input.input_sample_rate = 24000;
    input.tail_frames = 3;
    input.max_output_frames = 10;
    input.max_delay = 1;

    const auto plan = trtmc::ComputeSpeechOutputPlan(input);

    check(plan.effective_frames == 0, "output plan: effective frames floor");
    check(plan.output_frames == 3, "output plan: small input output frames");
    check(plan.total_iters == 5, "output plan: small input total iters");
}

void test_output_plan_large_target_clamps_to_max_output() {
    trtmc::SpeechOutputPlanInput input;
    input.sample_rate = 24000;
    input.frame_rate = 0.0F;
    input.num_frames = 100;
    input.num_input_samples = 1;
    input.input_sample_rate = 0;
    input.tail_frames = 50;
    input.max_output_frames = 60;
    input.max_delay = 4;

    const auto plan = trtmc::ComputeSpeechOutputPlan(input);

    check(plan.effective_frames == 98, "output plan: large target effective frames");
    check(plan.extra_tail == 50, "output plan: large target tail");
    check(plan.output_frames == 60, "output plan: large target output clamp");
    check(plan.total_iters == 65, "output plan: large target total iters");
}

void test_output_plan_keeps_the_final_partial_codec_frame() {
    trtmc::SpeechOutputPlanInput input;
    input.sample_rate = 24000;
    input.frame_rate = 12.5F;
    input.num_frames = 345;
    input.num_input_samples = 661632;
    input.input_sample_rate = 24000;
    input.max_output_frames = 400;
    input.max_delay = 1;

    const auto plan = trtmc::ComputeSpeechOutputPlan(input);

    check(plan.effective_frames == 343, "output plan: partial frame uses ceil semantics");
    check(plan.output_frames == 343, "output plan: partial frame matches official output length");
}

class FakeSubprocessRunner final : public trtmc::ISubprocessRunner {
  public:
    int rc{0};
    std::vector<char> stdout_data;
    std::string stderr_data;
    std::vector<std::string> last_argv;
    int call_count{0};

    int run(const std::vector<std::string>& argv, const void*, std::size_t,
            std::vector<char>& out_stdout, std::string& out_stderr) override {
        ++call_count;
        last_argv = argv;
        out_stdout = stdout_data;
        out_stderr = stderr_data;
        return rc;
    }
};

std::vector<char> make_token_bytes(std::initializer_list<int32_t> tokens) {
    std::vector<char> bytes(tokens.size() * sizeof(int32_t));
    std::size_t index = 0;
    for (int32_t token : tokens) {
        std::memcpy(bytes.data() + index * sizeof(int32_t), &token, sizeof(int32_t));
        ++index;
    }
    return bytes;
}

void test_tokenize_runtime_success_parses_tokens() {
    FakeSubprocessRunner runner;
    runner.stdout_data = make_token_bytes({11, 22, 33});

    const auto result =
        trtmc::TokenizeSpeechPromptRuntime("/usr/bin/python3", "hello world", runner);

    check(runner.call_count == 1, "success: runner called once");
    check(runner.last_argv.size() == 3, "success: argv size");
    if (runner.last_argv.size() == 3) {
        check(runner.last_argv[0] == "/bin/sh", "success: argv[0]");
        check(runner.last_argv[1] == "-c", "success: argv[1]");
        check_contains(runner.last_argv[2],
                       "/usr/bin/python3 -c \"from transformers import AutoTokenizer; ",
                       "success: command prefix");
        check_contains(runner.last_argv[2],
                       "ids = tok.encode('hello world', add_special_tokens=False); ",
                       "success: prompt embedded in command");
    }
    check(result.rc == 0, "success: rc");
    check(result.tokens == std::vector<int32_t>({11, 22, 33}), "success: token parse");
    check(result.stderr_data.empty(), "success: stderr empty");
}

void test_tokenize_runtime_failure_propagates_rc_and_stderr() {
    FakeSubprocessRunner runner;
    runner.rc = 17;
    runner.stderr_data = "subprocess failed";
    runner.stdout_data = make_token_bytes({101, 202});

    const auto result = trtmc::TokenizeSpeechPromptRuntime("/usr/bin/python3", "ignored", runner);

    check(result.rc == 17, "failure: rc propagated");
    check(result.tokens.empty(), "failure: tokens empty");
    check(result.stderr_data == "subprocess failed", "failure: stderr propagated");
}

void test_tokenize_runtime_empty_stdout_stays_empty() {
    FakeSubprocessRunner runner;
    runner.rc = 0;
    runner.stderr_data = "warnings only";

    const auto result = trtmc::TokenizeSpeechPromptRuntime("/usr/bin/python3", "ignored", runner);

    check(result.rc == 0, "empty stdout: rc preserved");
    check(result.tokens.empty(), "empty stdout: tokens empty");
    check(result.stderr_data == "warnings only", "empty stdout: stderr preserved");
}

} // namespace

int main() {
    test_output_plan_resample_path();
    test_output_plan_frame_rate_disabled_and_clamped();
    test_output_plan_small_inputs_do_not_go_negative();
    test_output_plan_large_target_clamps_to_max_output();
    test_output_plan_keeps_the_final_partial_codec_frame();

    test_tokenize_runtime_success_parses_tokens();
    test_tokenize_runtime_failure_propagates_rc_and_stderr();
    test_tokenize_runtime_empty_stdout_stays_empty();

    if (failures != 0)
        return 1;
    std::cout << "test_speech_subprocess_seam: PASS\n";
    return 0;
}
