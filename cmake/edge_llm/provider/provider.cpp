/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "provider.h"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <poll.h>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

extern char** environ;
extern "C" const trtmc::edge_llm::ProviderV1* trtmc_edge_provider_v1();

namespace {
thread_local std::string last_error;
constexpr std::size_t kMessageLimit = 16 * 1024 * 1024;
constexpr int kChildFd = 198;

/// Child process contains the version-specific Python and CUDA/TRT plugin state.
class Worker {
  public:
    explicit Worker(const char* script, const char* descriptor) {
        std::ifstream input(descriptor);
        const auto config = nlohmann::json::parse(input);
        if (config.at("version") != TRTMC_EDGE_PROVIDER_VERSION)
            throw std::runtime_error("Provider descriptor/DSO version mismatch");
        const auto python = config.at("python").get<std::string>();
        if (!std::filesystem::path(python).is_absolute())
            throw std::runtime_error("Provider Python must be an absolute path");
        if (!std::filesystem::path(script).is_absolute())
            throw std::runtime_error("Provider worker must be an absolute path");
        std::vector<std::string> args{python,      "-I",
                                      script,      "--provider",
                                      descriptor,  "serve",
                                      "--version", TRTMC_EDGE_PROVIDER_VERSION,
                                      "--fd",      std::to_string(kChildFd)};
        std::vector<char*> argv;
        for (auto& arg : args)
            argv.push_back(arg.data());
        argv.push_back(nullptr);
        std::vector<std::string> env;
        std::string libraries;
        for (const auto& path : config.value("library_paths", nlohmann::json::array()))
            libraries += path.get<std::string>() + ":";
        for (char** entry = environ; *entry; ++entry) {
            const std::string value(*entry);
            if (value.rfind("EDGELLM_PLUGIN_PATH=", 0) == 0)
                continue;
            if (value.rfind("LD_LIBRARY_PATH=", 0) == 0) {
                libraries += value.substr(16);
                continue;
            }
            env.push_back(value);
        }
        if (!libraries.empty() && libraries.back() == ':')
            libraries.pop_back(); // Never add the current directory to a loader path.
        if (!libraries.empty())
            env.push_back("LD_LIBRARY_PATH=" + libraries);
        if (config.contains("plugin"))
            env.push_back("EDGELLM_PLUGIN_PATH=" + config.at("plugin").get<std::string>());
        std::vector<char*> envp;
        for (auto& value : env)
            envp.push_back(value.data());
        envp.push_back(nullptr);
        int sockets[2];
        if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets) != 0)
            throw std::runtime_error("Cannot create Edge provider socket");
        posix_spawn_file_actions_t actions;
        int result = posix_spawn_file_actions_init(&actions);
        if (result == 0) {
            result = posix_spawn_file_actions_adddup2(&actions, STDERR_FILENO, STDOUT_FILENO);
            if (result == 0)
                result = posix_spawn_file_actions_adddup2(&actions, sockets[1], kChildFd);
            if (result == 0 && sockets[0] != kChildFd)
                result = posix_spawn_file_actions_addclose(&actions, sockets[0]);
            if (result == 0 && sockets[1] != kChildFd)
                result = posix_spawn_file_actions_addclose(&actions, sockets[1]);
            if (result == 0)
                result =
                    posix_spawn(&pid_, python.c_str(), &actions, nullptr, argv.data(), envp.data());
            posix_spawn_file_actions_destroy(&actions);
        }
        ::close(sockets[1]);
        if (result != 0) {
            ::close(sockets[0]);
            throw std::runtime_error("Cannot start Edge provider: " +
                                     std::string(strerror(result)));
        }
        socket_ = sockets[0];
    }

    ~Worker() {
        if (socket_ >= 0)
            ::close(socket_);
        if (pid_ <= 0)
            return;
        // Bounded shutdown also covers failed native initialization.
        kill(pid_, SIGTERM);
        for (int i = 0; i < 20; ++i) {
            const auto result = waitpid(pid_, nullptr, WNOHANG);
            if (result == pid_ || (result < 0 && errno == ECHILD))
                return;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        kill(pid_, SIGKILL);
        while (waitpid(pid_, nullptr, 0) < 0 && errno == EINTR) {
        }
    }
    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;

    const char* call(const char* message) {
        if (failed_)
            throw std::runtime_error(
                "Edge provider is unavailable after an earlier transport failure");
        const std::string data = std::string(message) + "\n";
        if (data.size() > kMessageLimit)
            throw std::runtime_error("Oversized Edge provider request");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(10);
        for (std::size_t offset = 0; offset < data.size();) {
            wait(POLLOUT, deadline);
            const auto count = send(socket_, data.data() + offset, data.size() - offset,
                                    MSG_NOSIGNAL | MSG_DONTWAIT);
            if (count < 0 && (errno == EINTR || errno == EAGAIN))
                continue;
            if (count <= 0)
                throw std::runtime_error("Edge provider terminated while receiving request");
            offset += count;
        }
        response_.clear();
        while (response_.size() <= kMessageLimit) {
            wait(POLLIN, deadline);
            char buffer[4096];
            const auto count = recv(socket_, buffer, sizeof(buffer), MSG_DONTWAIT);
            if (count < 0 && (errno == EINTR || errno == EAGAIN))
                continue;
            if (count <= 0)
                throw std::runtime_error("Edge provider terminated; see its stderr diagnostics");
            response_.append(buffer, count);
            if (response_.back() == '\n')
                return response_.c_str();
        }
        throw std::runtime_error("Oversized Edge provider response");
    }

    void poison() noexcept { failed_ = true; }

  private:
    void wait(short events, std::chrono::steady_clock::time_point deadline) {
        while (true) {
            auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 deadline - std::chrono::steady_clock::now())
                                 .count();
            if (remaining <= 0)
                throw std::runtime_error("Edge provider timed out");
            pollfd state{socket_, events, 0};
            const int result = poll(&state, 1, static_cast<int>(remaining));
            if (result < 0 && errno == EINTR)
                continue;
            if (result <= 0)
                throw std::runtime_error("Edge provider timeout or socket error");
            if (state.revents & events)
                return;
            throw std::runtime_error("Edge provider process exited");
        }
    }
    bool failed_{false};
    pid_t pid_{-1};
    int socket_{-1};
    std::string response_;
};

void* open_worker(const char* script, const char* descriptor) noexcept {
    try {
        return new Worker(script, descriptor);
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    } catch (...) {
        last_error = "Unknown provider startup failure";
        return nullptr;
    }
}
const char* call_worker(void* handle, const char* message) noexcept {
    try {
        return static_cast<Worker*>(handle)->call(message);
    } catch (const std::exception& error) {
        static_cast<Worker*>(handle)->poison();
        last_error = error.what();
        return nullptr;
    } catch (...) {
        static_cast<Worker*>(handle)->poison();
        last_error = "Unknown provider call failure";
        return nullptr;
    }
}
void close_worker(void* handle) noexcept {
    delete static_cast<Worker*>(handle);
}
const char* error_text() noexcept {
    return last_error.c_str();
}
const trtmc::edge_llm::ProviderV1 api{1,
                                      sizeof(trtmc::edge_llm::ProviderV1),
                                      TRTMC_EDGE_PROVIDER_VERSION,
                                      open_worker,
                                      call_worker,
                                      close_worker,
                                      error_text};
} // namespace

extern "C" const trtmc::edge_llm::ProviderV1* trtmc_edge_provider_v1() {
    return &api;
}
