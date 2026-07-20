/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/qwen3_omni/talker_runtime.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <poll.h>
#include <pthread.h>
#include <spawn.h>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <utility>

extern char** environ;

namespace trtmc {
namespace {

using Clock = std::chrono::steady_clock;
using Pipe = std::array<int, 2>;

constexpr std::uint32_t kWorkerMagic = 0x514f4d4eU;
constexpr std::uint32_t kWorkerReady = 1;
constexpr std::uint32_t kWorkerOk = 2;
constexpr std::uint32_t kWorkerError = 3;
constexpr std::size_t kResponseHeaderBytes = 20;
constexpr std::uint32_t kMaxResponseBytes = 64U * 1024U * 1024U;

struct ProcessPipes {
    Pipe input{-1, -1};
    Pipe output{-1, -1};
    Pipe error{-1, -1};
};

struct WorkerResponse {
    std::uint32_t status{0};
    double talker_ms{0.0};
    std::vector<char> payload;
};

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void append_error(std::string& destination, const std::string& message) {
    if (message.empty())
        return;
    if (!destination.empty() && destination.back() != '\n')
        destination.push_back('\n');
    destination += message;
}

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

bool set_nonblocking(int fd) {
    const int flags = fcntl(fd, F_GETFL, 0);
    return flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

class ScopedSigpipeBlock {
  public:
    ScopedSigpipeBlock() {
        sigemptyset(&blocked_);
        sigaddset(&blocked_, SIGPIPE);
        active_ = pthread_sigmask(SIG_BLOCK, &blocked_, &previous_) == 0;
        sigset_t pending;
        sigemptyset(&pending);
        was_pending_ = active_ && sigpending(&pending) == 0 && sigismember(&pending, SIGPIPE) == 1;
    }

    ~ScopedSigpipeBlock() {
        if (active_)
            (void)pthread_sigmask(SIG_SETMASK, &previous_, nullptr);
    }

    void consume_if_generated(int write_errno) const {
        if (write_errno != EPIPE || !active_ || was_pending_)
            return;
        timespec timeout{0, 0};
        (void)sigtimedwait(&blocked_, nullptr, &timeout);
    }

  private:
    sigset_t blocked_{};
    sigset_t previous_{};
    bool active_{false};
    bool was_pending_{false};
};

void append_u32_le(std::vector<char>& output, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8)
        output.push_back(static_cast<char>((value >> shift) & 0xffU));
}

std::uint32_t read_u32_le(const char* input) {
    std::uint32_t value = 0;
    for (int shift = 0; shift < 32; shift += 8)
        value |= static_cast<std::uint32_t>(static_cast<unsigned char>(input[shift / 8])) << shift;
    return value;
}

double read_f64_le(const char* input) {
    static_assert(sizeof(double) == 8, "Qwen3-Omni worker protocol requires binary64 doubles");
    std::uint64_t bits = 0;
    for (int shift = 0; shift < 64; shift += 8)
        bits |= static_cast<std::uint64_t>(static_cast<unsigned char>(input[shift / 8])) << shift;
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
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

std::vector<char> frame_request(const std::vector<char>& request) {
    if (request.size() > UINT32_MAX)
        throw std::runtime_error("Qwen3-Omni Talker framed request is too large");
    std::vector<char> frame;
    frame.reserve(8 + request.size());
    append_u32_le(frame, kWorkerMagic);
    append_u32_le(frame, static_cast<std::uint32_t>(request.size()));
    frame.insert(frame.end(), request.begin(), request.end());
    return frame;
}

std::vector<char> shutdown_request() {
    std::vector<char> frame;
    append_u32_le(frame, kWorkerMagic);
    append_u32_le(frame, 0);
    return frame;
}

bool send_all(int fd, const std::vector<char>& data) {
    ScopedSigpipeBlock sigpipe_block;
    std::size_t offset = 0;
    while (offset < data.size()) {
        const auto count = write(fd, data.data() + offset, data.size() - offset);
        if (count < 0 && errno == EINTR)
            continue;
        if (count <= 0) {
            const int write_errno = errno;
            sigpipe_block.consume_if_generated(write_errno);
            errno = write_errno;
            return false;
        }
        offset += static_cast<std::size_t>(count);
    }
    return true;
}

std::vector<std::string> make_talker_argv(const std::string& hf_python, const std::string& model_id,
                                          const std::string& model_revision, int32_t max_frames) {
    std::vector<std::string> argv = {
        hf_python,
        "-m",
        "tensorrt_model_connect.families.qwen3_omni.audio_runtime",
        "--worker",
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

void add_spawn_close(posix_spawn_file_actions_t& actions, int fd) {
    if (fd >= 0 && fd != STDIN_FILENO && fd != STDOUT_FILENO && fd != STDERR_FILENO)
        posix_spawn_file_actions_addclose(&actions, fd);
}

int spawn_worker(const std::vector<std::string>& argv, ProcessPipes& pipes, pid_t& pid) {
    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions) != 0)
        return -1;
    int rc = posix_spawn_file_actions_adddup2(&actions, pipes.input[0], STDIN_FILENO);
    rc = rc == 0 ? posix_spawn_file_actions_adddup2(&actions, pipes.output[1], STDOUT_FILENO) : rc;
    rc = rc == 0 ? posix_spawn_file_actions_adddup2(&actions, pipes.error[1], STDERR_FILENO) : rc;
    for (const Pipe* pipe : {&pipes.input, &pipes.output, &pipes.error}) {
        add_spawn_close(actions, (*pipe)[0]);
        add_spawn_close(actions, (*pipe)[1]);
    }

    std::vector<char*> child_argv;
    child_argv.reserve(argv.size() + 1);
    for (const auto& argument : argv)
        child_argv.push_back(const_cast<char*>(argument.c_str()));
    child_argv.push_back(nullptr);
    if (rc == 0)
        rc = posix_spawnp(&pid, child_argv[0], &actions, nullptr, child_argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    return rc;
}

} // namespace

class Qwen3OmniTalkerRuntime::Impl {
  public:
    Impl(std::string hf_python, std::string model_id, std::string model_revision,
         int32_t n_codebooks, int32_t max_frames, std::vector<std::string> worker_argv)
        : hf_python_(std::move(hf_python)), model_id_(std::move(model_id)),
          model_revision_(std::move(model_revision)), n_codebooks_(n_codebooks),
          max_frames_(max_frames), worker_argv_(std::move(worker_argv)) {}

    ~Impl() { shutdown(); }

    Qwen3OmniTalkerRuntimeResult run(const std::string& prompt, const std::string& assistant_text) {
        std::lock_guard<std::mutex> lock(mutex_);
        Qwen3OmniTalkerRuntimeResult result;
        if (!validate(result.stderr_data))
            return result;

        if (!ensure_started(result.worker_start_ms, result.stderr_data))
            return result;

        const auto framed_request = frame_request(make_request(prompt, assistant_text));
        const auto request_start = Clock::now();
        if (!send_all(input_fd_, framed_request)) {
            append_error(result.stderr_data, "Qwen3-Omni Talker worker request write failed: " +
                                                 std::string(std::strerror(errno)));
            fail_worker(result.stderr_data);
            return result;
        }
        ++requests_;

        WorkerResponse response;
        std::string protocol_error;
        if (!read_response(response, protocol_error)) {
            append_error(result.stderr_data, protocol_error);
            fail_worker(result.stderr_data);
            return result;
        }
        const auto response_end = Clock::now();
        result.talker_ms = response.talker_ms;
        result.ipc_ms = std::max(0.0, elapsed_ms(request_start, response_end) - result.talker_ms);

        if (response.status == kWorkerError) {
            result.exit_code = 1;
            append_error(result.stderr_data,
                         std::string(response.payload.begin(), response.payload.end()));
            append_error(result.stderr_data, take_stderr());
            return result;
        }
        if (response.status != kWorkerOk) {
            append_error(result.stderr_data, "Qwen3-Omni Talker worker returned invalid status");
            fail_worker(result.stderr_data);
            return result;
        }

        const auto materialize_start = Clock::now();
        result.exit_code =
            materialize_codes(response.payload, result.frame_major_codes, result.stderr_data) ? 0
                                                                                              : -1;
        result.output_materialization_ms = elapsed_ms(materialize_start, Clock::now());
        append_error(result.stderr_data, take_stderr());
        if (result.exit_code != 0)
            stop_worker(false);
        return result;
    }

    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_worker(true);
    }

    Qwen3OmniTalkerRuntimeStats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return {worker_starts_, requests_, pid_ > 0};
    }

  private:
    bool validate(std::string& error) const {
        if ((worker_argv_.empty() && hf_python_.empty()) || model_id_.empty() ||
            n_codebooks_ <= 0 || max_frames_ <= 0) {
            error = "invalid Qwen3-Omni official Talker runtime configuration";
            if (worker_argv_.empty() && hf_python_.empty())
                error = "--hf-python is required for the official Qwen3-Omni Talker";
            return false;
        }
        return true;
    }

    bool ensure_started(double& worker_start_ms, std::string& error) {
        if (pid_ > 0)
            return true;
        const auto start = Clock::now();
        if (!launch_worker(error) || !await_ready(error))
            return false;
        worker_start_ms = elapsed_ms(start, Clock::now());
        return true;
    }

    bool launch_worker(std::string& error) {
        ProcessPipes pipes;
        if (!open_all(pipes)) {
            error = "Qwen3-Omni Talker worker pipe creation failed";
            return false;
        }
        const auto argv = worker_argv_.empty() ? make_talker_argv(hf_python_, model_id_,
                                                                  model_revision_, max_frames_)
                                               : worker_argv_;
        pid_t child_pid = -1;
        const int spawn_rc = spawn_worker(argv, pipes, child_pid);
        if (spawn_rc != 0) {
            close_all(pipes);
            error =
                "Qwen3-Omni Talker worker spawn failed: " + std::string(std::strerror(spawn_rc));
            return false;
        }

        pid_ = child_pid;
        ++worker_starts_;
        close_fd(pipes.input[0]);
        close_fd(pipes.output[1]);
        close_fd(pipes.error[1]);
        input_fd_ = pipes.input[1];
        output_fd_ = pipes.output[0];
        error_fd_ = pipes.error[0];
        if (!set_nonblocking(output_fd_) || !set_nonblocking(error_fd_)) {
            error = "Qwen3-Omni Talker worker pipe setup failed";
            stop_worker(false);
            return false;
        }
        return true;
    }

    bool await_ready(std::string& error) {
        WorkerResponse ready;
        std::string protocol_error;
        if (!read_response(ready, protocol_error) || ready.status != kWorkerReady ||
            !ready.payload.empty()) {
            error = protocol_error.empty() ? "Qwen3-Omni Talker worker did not become ready"
                                           : protocol_error;
            fail_worker(error);
            return false;
        }
        return true;
    }

    bool read_response(WorkerResponse& response, std::string& error) {
        if (!ensure_stdout(kResponseHeaderBytes, error))
            return false;
        const auto header = consume_stdout(kResponseHeaderBytes);
        const std::uint32_t magic = read_u32_le(header.data());
        response.status = read_u32_le(header.data() + 4);
        const std::uint32_t payload_size = read_u32_le(header.data() + 8);
        response.talker_ms = read_f64_le(header.data() + 12);
        if (magic != kWorkerMagic) {
            error = "Qwen3-Omni Talker worker returned invalid response magic";
            return false;
        }
        if (payload_size > kMaxResponseBytes) {
            error = "Qwen3-Omni Talker worker response is too large";
            return false;
        }
        if (!ensure_stdout(payload_size, error))
            return false;
        response.payload = consume_stdout(payload_size);
        drain_stderr_now();
        return true;
    }

    bool ensure_stdout(std::size_t size, std::string& error) {
        while (stdout_buffer_.size() < size) {
            if (!poll_worker_io(error))
                return false;
            if (output_fd_ < 0 && stdout_buffer_.size() < size) {
                error = "Qwen3-Omni Talker worker exited before completing its response";
                return false;
            }
        }
        return true;
    }

    bool poll_worker_io(std::string& error) {
        pollfd descriptors[2] = {
            {output_fd_, POLLIN, 0},
            {error_fd_, POLLIN, 0},
        };
        int ready = -1;
        do {
            ready = poll(descriptors, 2, -1);
        } while (ready < 0 && errno == EINTR);
        if (ready < 0) {
            error = "Qwen3-Omni Talker worker poll failed";
            return false;
        }
        drain_poll_descriptor(descriptors[1], error_fd_, stderr_pending_);
        drain_poll_descriptor(descriptors[0], output_fd_, stdout_buffer_);
        return true;
    }

    static void drain_poll_descriptor(const pollfd& descriptor, int& fd,
                                      std::vector<char>& destination) {
        if (descriptor.fd >= 0 && (descriptor.revents & (POLLIN | POLLHUP | POLLERR)))
            drain_fd(fd, destination);
    }

    static void drain_fd(int& fd, std::vector<char>& destination) {
        if (fd < 0)
            return;
        char buffer[65536];
        for (;;) {
            const auto count = read(fd, buffer, sizeof(buffer));
            if (count > 0) {
                destination.insert(destination.end(), buffer, buffer + count);
                continue;
            }
            if (count == 0) {
                close_fd(fd);
                return;
            }
            if (errno == EINTR)
                continue;
            if (errno != EAGAIN && errno != EWOULDBLOCK)
                close_fd(fd);
            return;
        }
    }

    std::vector<char> consume_stdout(std::size_t size) {
        std::vector<char> result(stdout_buffer_.begin(), stdout_buffer_.begin() + size);
        stdout_buffer_.erase(stdout_buffer_.begin(), stdout_buffer_.begin() + size);
        return result;
    }

    void drain_stderr_now() {
        if (error_fd_ < 0)
            return;
        pollfd descriptor{error_fd_, POLLIN, 0};
        if (poll(&descriptor, 1, 0) > 0 && (descriptor.revents & (POLLIN | POLLHUP | POLLERR))) {
            drain_fd(error_fd_, stderr_pending_);
        }
    }

    std::string take_stderr() {
        drain_stderr_now();
        std::string result(stderr_pending_.begin(), stderr_pending_.end());
        stderr_pending_.clear();
        return result;
    }

    bool materialize_codes(const std::vector<char>& payload, std::vector<int32_t>& codes,
                           std::string& error) const {
        if (payload.empty() || payload.size() % sizeof(int32_t) != 0) {
            append_error(error, "Qwen3-Omni official Talker returned an invalid binary payload");
            return false;
        }
        codes.resize(payload.size() / sizeof(int32_t));
        std::memcpy(codes.data(), payload.data(), payload.size());
        if (codes.size() % static_cast<std::size_t>(n_codebooks_) != 0 ||
            codes.size() >
                static_cast<std::size_t>(n_codebooks_) * static_cast<std::size_t>(max_frames_)) {
            codes.clear();
            append_error(error, "Qwen3-Omni official Talker returned an invalid codec shape");
            return false;
        }
        return true;
    }

    bool wait_for_exit(std::chrono::milliseconds timeout) {
        const auto deadline = Clock::now() + timeout;
        while (pid_ > 0) {
            int status = 0;
            const pid_t waited = waitpid(pid_, &status, WNOHANG);
            if (waited == pid_ || (waited < 0 && errno == ECHILD)) {
                pid_ = -1;
                return true;
            }
            if (waited < 0 && errno != EINTR)
                return false;
            if (Clock::now() >= deadline)
                return false;
            drain_stderr_now();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return true;
    }

    void stop_worker(bool graceful) {
        if (pid_ > 0 && graceful && input_fd_ >= 0)
            (void)send_all(input_fd_, shutdown_request());
        close_fd(input_fd_);
        if (pid_ > 0 && !wait_for_exit(std::chrono::milliseconds(graceful ? 2000 : 0))) {
            kill(pid_, SIGTERM);
            if (!wait_for_exit(std::chrono::milliseconds(500))) {
                kill(pid_, SIGKILL);
                (void)wait_for_exit(std::chrono::milliseconds(500));
            }
        }
        drain_stderr_now();
        close_fd(output_fd_);
        close_fd(error_fd_);
        stdout_buffer_.clear();
        pid_ = -1;
    }

    void fail_worker(std::string& error) {
        stop_worker(false);
        append_error(error, take_stderr());
    }

    std::string hf_python_;
    std::string model_id_;
    std::string model_revision_;
    int32_t n_codebooks_{0};
    int32_t max_frames_{0};
    std::vector<std::string> worker_argv_;
    mutable std::mutex mutex_;
    pid_t pid_{-1};
    int input_fd_{-1};
    int output_fd_{-1};
    int error_fd_{-1};
    int64_t worker_starts_{0};
    int64_t requests_{0};
    std::vector<char> stdout_buffer_;
    std::vector<char> stderr_pending_;
};

Qwen3OmniTalkerRuntime::Qwen3OmniTalkerRuntime(std::string hf_python, std::string model_id,
                                               std::string model_revision, int32_t n_codebooks,
                                               int32_t max_frames,
                                               std::vector<std::string> worker_argv)
    : impl_(std::make_unique<Impl>(std::move(hf_python), std::move(model_id),
                                   std::move(model_revision), n_codebooks, max_frames,
                                   std::move(worker_argv))) {}

Qwen3OmniTalkerRuntime::~Qwen3OmniTalkerRuntime() = default;
Qwen3OmniTalkerRuntime::Qwen3OmniTalkerRuntime(Qwen3OmniTalkerRuntime&&) noexcept = default;
Qwen3OmniTalkerRuntime&
Qwen3OmniTalkerRuntime::operator=(Qwen3OmniTalkerRuntime&&) noexcept = default;

Qwen3OmniTalkerRuntimeResult Qwen3OmniTalkerRuntime::run(const std::string& prompt,
                                                         const std::string& assistant_text) {
    return impl_->run(prompt, assistant_text);
}

void Qwen3OmniTalkerRuntime::shutdown() {
    impl_->shutdown();
}

Qwen3OmniTalkerRuntimeStats Qwen3OmniTalkerRuntime::stats() const {
    return impl_->stats();
}

} // namespace trtmc
