/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/qwen3_omni/talker_runtime.h"

#include <array>
#include <cerrno>
#include <cstring>
#include <poll.h>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace trtmc {
namespace {

using Pipe = std::array<int, 2>;

struct ProcessPipes {
    Pipe input{-1, -1};
    Pipe output{-1, -1};
    Pipe error{-1, -1};
};

void close_fd(int& fd) {
    if (fd >= 0) {
        close(fd);
        fd = -1;
    }
}

void close_all(ProcessPipes& pipes) {
    for (Pipe* pipe : {&pipes.input, &pipes.output, &pipes.error}) {
        close_fd((*pipe)[0]);
        close_fd((*pipe)[1]);
    }
}

bool open_all(ProcessPipes& pipes) {
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, pipes.input.data()) != 0)
        return false;
    for (Pipe* pipe : {&pipes.output, &pipes.error}) {
        if (::pipe(pipe->data()) != 0) {
            close_all(pipes);
            return false;
        }
    }
    return true;
}

void append_u32_le(std::vector<char>& output, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8)
        output.push_back(static_cast<char>((value >> shift) & 0xffU));
}

std::vector<char> make_request(const std::string& prompt, const std::string& assistant_text) {
    if (prompt.size() > UINT32_MAX || assistant_text.size() > UINT32_MAX)
        throw std::runtime_error("Qwen3-Omni Talker request text is too large");
    std::vector<char> request;
    request.reserve(8 + prompt.size() + assistant_text.size());
    append_u32_le(request, static_cast<std::uint32_t>(prompt.size()));
    append_u32_le(request, static_cast<std::uint32_t>(assistant_text.size()));
    request.insert(request.end(), prompt.begin(), prompt.end());
    request.insert(request.end(), assistant_text.begin(), assistant_text.end());
    return request;
}

bool send_all(int fd, const std::vector<char>& data) {
    std::size_t offset = 0;
    while (offset < data.size()) {
        const auto count = send(fd, data.data() + offset, data.size() - offset, MSG_NOSIGNAL);
        if (count < 0 && errno == EINTR)
            continue;
        if (count <= 0)
            return false;
        offset += static_cast<std::size_t>(count);
    }
    return true;
}

void read_ready_fd(pollfd& descriptor, std::vector<char>& output) {
    char buffer[65536];
    for (;;) {
        const auto count = read(descriptor.fd, buffer, sizeof(buffer));
        if (count > 0) {
            output.insert(output.end(), buffer, buffer + count);
        } else if (count == 0) {
            close(descriptor.fd);
            descriptor.fd = -1;
        } else if (errno != EINTR && errno != EAGAIN) {
            close(descriptor.fd);
            descriptor.fd = -1;
        }
        return;
    }
}

[[noreturn]] void exec_child(ProcessPipes& pipes, const std::vector<std::string>& argv) {
    if (dup2(pipes.input[0], STDIN_FILENO) < 0 || dup2(pipes.output[1], STDOUT_FILENO) < 0 ||
        dup2(pipes.error[1], STDERR_FILENO) < 0) {
        _exit(127);
    }
    close_all(pipes);

    std::vector<char*> child_argv;
    child_argv.reserve(argv.size() + 1);
    for (const auto& argument : argv)
        child_argv.push_back(const_cast<char*>(argument.c_str()));
    child_argv.push_back(nullptr);
    execvp(child_argv[0], child_argv.data());
    _exit(127);
}

void drain_output(ProcessPipes& pipes, std::vector<char>& stdout_data,
                  std::vector<char>& stderr_data) {
    pollfd descriptors[2] = {
        {pipes.output[0], POLLIN, 0},
        {pipes.error[0], POLLIN, 0},
    };
    while (descriptors[0].fd >= 0 || descriptors[1].fd >= 0) {
        const int ready = poll(descriptors, 2, -1);
        if (ready < 0 && errno == EINTR)
            continue;
        if (ready < 0)
            break;
        for (int index = 0; index < 2; ++index) {
            if (descriptors[index].fd >= 0 &&
                (descriptors[index].revents & (POLLIN | POLLHUP | POLLERR))) {
                read_ready_fd(descriptors[index], index == 0 ? stdout_data : stderr_data);
            }
        }
    }
    close_fd(descriptors[0].fd);
    close_fd(descriptors[1].fd);
    pipes.output[0] = -1;
    pipes.error[0] = -1;
}

int wait_for_child(pid_t pid) {
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR)
            return -1;
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

int run_process(const std::vector<std::string>& argv, const std::vector<char>& input,
                std::vector<char>& stdout_data, std::vector<char>& stderr_data) {
    ProcessPipes pipes;
    if (!open_all(pipes))
        return -1;

    const pid_t pid = fork();
    if (pid < 0) {
        close_all(pipes);
        return -1;
    }
    if (pid == 0)
        exec_child(pipes, argv);

    close_fd(pipes.input[0]);
    close_fd(pipes.output[1]);
    close_fd(pipes.error[1]);
    const bool wrote_request = send_all(pipes.input[1], input);
    close_fd(pipes.input[1]);
    drain_output(pipes, stdout_data, stderr_data);
    const int exit_code = wait_for_child(pid);
    close_all(pipes);
    if (!wrote_request)
        return -1;
    return exit_code;
}

std::vector<std::string> make_talker_argv(const std::string& hf_python, const std::string& model_id,
                                          const std::string& model_revision, int32_t max_frames) {
    std::vector<std::string> argv = {
        hf_python,
        "-m",
        "tensorrt_model_connect.families.qwen3_omni.audio_runtime",
        "--model-id",
        model_id,
        "--max-frames",
        std::to_string(max_frames),
    };
    if (!model_revision.empty()) {
        argv.emplace_back("--revision");
        argv.push_back(model_revision);
    }
    return argv;
}

} // namespace

Qwen3OmniTalkerRuntimeResult
run_qwen3_omni_official_talker(const std::string& hf_python, const std::string& model_id,
                               const std::string& model_revision, const std::string& prompt,
                               const std::string& assistant_text, int32_t n_codebooks,
                               int32_t max_frames) {
    Qwen3OmniTalkerRuntimeResult result;
    if (hf_python.empty()) {
        result.stderr_data = "--hf-python is required for the official Qwen3-Omni Talker";
        return result;
    }
    if (model_id.empty() || n_codebooks <= 0 || max_frames <= 0) {
        result.stderr_data = "invalid Qwen3-Omni official Talker runtime configuration";
        return result;
    }

    const auto argv = make_talker_argv(hf_python, model_id, model_revision, max_frames);
    std::vector<char> stdout_data;
    std::vector<char> stderr_data;
    result.exit_code =
        run_process(argv, make_request(prompt, assistant_text), stdout_data, stderr_data);
    result.stderr_data.assign(stderr_data.begin(), stderr_data.end());
    if (result.exit_code != 0)
        return result;
    if (stdout_data.empty() || stdout_data.size() % sizeof(int32_t) != 0) {
        result.exit_code = -1;
        result.stderr_data += "\nQwen3-Omni official Talker returned an invalid binary payload";
        return result;
    }

    result.frame_major_codes.resize(stdout_data.size() / sizeof(int32_t));
    std::memcpy(result.frame_major_codes.data(), stdout_data.data(), stdout_data.size());
    if (result.frame_major_codes.size() % static_cast<std::size_t>(n_codebooks) != 0 ||
        result.frame_major_codes.size() >
            static_cast<std::size_t>(n_codebooks) * static_cast<std::size_t>(max_frames)) {
        result.exit_code = -1;
        result.frame_major_codes.clear();
        result.stderr_data += "\nQwen3-Omni official Talker returned an invalid codec shape";
    }
    return result;
}

} // namespace trtmc
