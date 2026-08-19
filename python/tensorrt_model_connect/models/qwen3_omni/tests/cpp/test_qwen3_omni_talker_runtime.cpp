/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "talker_runtime.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

constexpr std::uint32_t kWorkerMagic = 0x514f4d4eU;
constexpr std::uint32_t kWorkerReady = 1;
constexpr std::uint32_t kWorkerOk = 2;
constexpr std::uint32_t kWorkerError = 3;

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

bool read_exact(int fd, void* output, std::size_t size) {
    auto* bytes = static_cast<char*>(output);
    std::size_t offset = 0;
    while (offset < size) {
        const auto count = read(fd, bytes + offset, size - offset);
        if (count < 0 && errno == EINTR)
            continue;
        if (count <= 0)
            return false;
        offset += static_cast<std::size_t>(count);
    }
    return true;
}

bool write_all(int fd, const void* input, std::size_t size) {
    const auto* bytes = static_cast<const char*>(input);
    std::size_t offset = 0;
    while (offset < size) {
        const auto count = write(fd, bytes + offset, size - offset);
        if (count < 0 && errno == EINTR)
            continue;
        if (count <= 0)
            return false;
        offset += static_cast<std::size_t>(count);
    }
    return true;
}

std::uint32_t read_u32_le(const char* input) {
    std::uint32_t value = 0;
    for (int shift = 0; shift < 32; shift += 8)
        value |= static_cast<std::uint32_t>(static_cast<unsigned char>(input[shift / 8])) << shift;
    return value;
}

void append_u32_le(std::vector<char>& output, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8)
        output.push_back(static_cast<char>((value >> shift) & 0xffU));
}

void append_f64_le(std::vector<char>& output, double value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(value));
    for (int shift = 0; shift < 64; shift += 8)
        output.push_back(static_cast<char>((bits >> shift) & 0xffU));
}

bool write_response(std::uint32_t status, const std::vector<char>& payload = {}) {
    std::vector<char> header;
    append_u32_le(header, kWorkerMagic);
    append_u32_le(header, status);
    append_u32_le(header, static_cast<std::uint32_t>(payload.size()));
    append_f64_le(header, status == kWorkerReady ? 0.0 : 1.25);
    return write_all(STDOUT_FILENO, header.data(), header.size()) &&
           write_all(STDOUT_FILENO, payload.data(), payload.size());
}

std::string request_prompt(const std::vector<char>& payload) {
    if (payload.size() < 8)
        return {};
    const std::uint32_t prompt_size = read_u32_le(payload.data());
    if (8 + static_cast<std::size_t>(prompt_size) > payload.size())
        return {};
    return std::string(payload.data() + 8, payload.data() + 8 + prompt_size);
}

int read_fake_request(std::vector<char>& payload) {
    char header[8];
    if (!read_exact(STDIN_FILENO, header, sizeof(header)))
        return 0;
    const std::uint32_t magic = read_u32_le(header);
    const std::uint32_t payload_size = read_u32_le(header + 4);
    if (magic != kWorkerMagic)
        return 3;
    if (payload_size == 0)
        return 0;
    payload.resize(payload_size);
    return read_exact(STDIN_FILENO, payload.data(), payload.size()) ? -1 : 4;
}

int respond_to_fake_request(const std::vector<char>& payload, int32_t request_count) {
    const std::string prompt = request_prompt(payload);
    if (prompt == "crash")
        _exit(9);
    if (prompt == "error") {
        const std::string stderr_message = "fixture stderr\n";
        (void)write_all(STDERR_FILENO, stderr_message.data(), stderr_message.size());
        const std::string message = "fixture request failed";
        return write_response(kWorkerError, std::vector<char>(message.begin(), message.end())) ? -1
                                                                                               : 5;
    }
    const int32_t codes[] = {request_count, request_count + 10};
    std::vector<char> bytes(sizeof(codes));
    std::memcpy(bytes.data(), codes, sizeof(codes));
    return write_response(kWorkerOk, bytes) ? -1 : 6;
}

int run_fake_worker() {
    if (!write_response(kWorkerReady))
        return 2;
    int32_t request_count = 0;
    for (;;) {
        std::vector<char> payload;
        const int read_status = read_fake_request(payload);
        if (read_status >= 0)
            return read_status;
        const int response_status = respond_to_fake_request(payload, ++request_count);
        if (response_status >= 0)
            return response_status;
    }
}

std::string self_executable() {
    std::vector<char> path(4096, '\0');
    const auto size = readlink("/proc/self/exe", path.data(), path.size() - 1);
    if (size <= 0)
        return {};
    return std::string(path.data(), static_cast<std::size_t>(size));
}

void test_worker_is_reused_and_recovers_after_failure() {
    trtmc::Qwen3OmniTalkerRuntime runtime("", "fixture", "", 2, 4,
                                          {self_executable(), "--fake-worker"});

    const auto first = runtime.run("first", "assistant");
    if (first.exit_code != 0)
        std::cerr << "first request error: " << first.stderr_data << '\n';
    check(first.exit_code == 0, "first request succeeds");
    check(first.frame_major_codes == std::vector<int32_t>({1, 11}),
          "first request returns fixture codes");
    check(first.talker_ms == 1.25, "first request reports Talker timing");
    check(first.ipc_ms >= 0.0, "first request reports IPC timing");
    check(first.output_materialization_ms >= 0.0,
          "first request reports output materialization timing");

    const auto second = runtime.run("second", "assistant");
    check(second.exit_code == 0, "second request succeeds");
    check(second.frame_major_codes == std::vector<int32_t>({2, 12}),
          "second request uses the same worker state");
    auto stats = runtime.stats();
    check(stats.worker_starts == 1, "two requests initialize one worker");
    check(stats.requests == 2, "two requests are counted");
    check(stats.worker_running, "worker remains running after two requests");

    const auto request_error = runtime.run("error", "assistant");
    check(request_error.exit_code == 1, "request error is explicit");
    check(request_error.stderr_data.find("fixture request failed") != std::string::npos,
          "request error payload is propagated");
    check(request_error.stderr_data.find("fixture stderr") != std::string::npos,
          "worker stderr is propagated");
    const auto after_error = runtime.run("after-error", "assistant");
    check(after_error.exit_code == 0, "worker remains usable after request error");
    check(after_error.frame_major_codes == std::vector<int32_t>({4, 14}),
          "request failure does not tear down a healthy worker");

    const auto crash = runtime.run("crash", "assistant");
    check(crash.exit_code == -1, "worker process failure returns runtime error");
    check(crash.stderr_data.find("exited") != std::string::npos,
          "worker process failure is explicit");
    const auto restarted = runtime.run("after-crash", "assistant");
    check(restarted.exit_code == 0, "request after crash succeeds");
    check(restarted.frame_major_codes == std::vector<int32_t>({1, 11}),
          "next request restarts a failed worker cleanly");
    stats = runtime.stats();
    check(stats.worker_starts == 2, "worker restart count is deterministic");
    check(stats.requests == 6, "request count includes failed request");
    check(stats.worker_running, "restarted worker remains running");

    runtime.shutdown();
    check(!runtime.stats().worker_running, "explicit shutdown reaps the worker");
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--fake-worker")
        return run_fake_worker();
    test_worker_is_reused_and_recovers_after_failure();
    return failures == 0 ? 0 : 1;
}
