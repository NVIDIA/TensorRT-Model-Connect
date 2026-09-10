/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "audio_protocol.h"

#include <chrono>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <cerrno>
#include <poll.h>
#include <unistd.h>
#endif

namespace trtmc::examples::windows_voicechat {

// Preserve a dedicated protocol handle before redirecting runtime diagnostics.
// This also captures C printf calls and diagnostics in loaded vendor DLLs.
class ProtocolOutput {
  public:
    ProtocolOutput() {
        std::fflush(stdout);
#ifdef _WIN32
        const int protocol_fd = _dup(_fileno(stdout));
        if (protocol_fd < 0)
            throw std::runtime_error("cannot duplicate protocol stdout");
        file_ = _fdopen(protocol_fd, "wb");
        if (file_ == nullptr) {
            _close(protocol_fd);
            throw std::runtime_error("cannot open protocol stdout");
        }
        _setmode(protocol_fd, _O_BINARY);
        if (_dup2(_fileno(stderr), _fileno(stdout)) != 0) {
            std::fclose(file_);
            throw std::runtime_error("cannot redirect runtime diagnostics to stderr");
        }
#else
        const int protocol_fd = dup(fileno(stdout));
        if (protocol_fd < 0)
            throw std::runtime_error("cannot duplicate protocol stdout");
        file_ = fdopen(protocol_fd, "wb");
        if (file_ == nullptr) {
            close(protocol_fd);
            throw std::runtime_error("cannot open protocol stdout");
        }
        if (dup2(fileno(stderr), fileno(stdout)) < 0) {
            std::fclose(file_);
            throw std::runtime_error("cannot redirect runtime diagnostics to stderr");
        }
#endif
    }

    ~ProtocolOutput() { std::fclose(file_); }
    ProtocolOutput(const ProtocolOutput&) = delete;
    ProtocolOutput& operator=(const ProtocolOutput&) = delete;

    void write(const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (std::fwrite(message.data(), 1, message.size(), file_) != message.size() ||
            std::fputc('\n', file_) == EOF || std::fflush(file_) != 0)
            throw std::runtime_error("desktop application closed the protocol output pipe");
    }

  private:
    std::FILE* file_{nullptr};
    std::mutex mutex_;
};

// Polling the anonymous pipe keeps EOF, Ctrl-C and worker failure observable
// without a detached reader thread or a blocking getline during shutdown.
class CommandInput {
  public:
    CommandInput() {
#ifdef _WIN32
        handle_ = GetStdHandle(STD_INPUT_HANDLE);
        if (handle_ == INVALID_HANDLE_VALUE || GetFileType(handle_) != FILE_TYPE_PIPE)
            throw std::runtime_error("bridge stdin must be a pipe; launch it from the desktop app");
#endif
    }

    bool eof() const noexcept { return eof_ && buffer_.empty(); }

    bool next(std::string& line) {
        if (take_line(line))
            return true;
        if (eof_)
            return false;
        char chunk[16384];
        std::size_t count = 0;
#ifdef _WIN32
        DWORD available = 0;
        if (!PeekNamedPipe(handle_, nullptr, 0, nullptr, &available, nullptr)) {
            const auto error = GetLastError();
            if (error == ERROR_BROKEN_PIPE || error == ERROR_PIPE_NOT_CONNECTED) {
                eof_ = true;
                return take_line(line);
            }
            throw std::runtime_error("cannot poll the desktop input pipe");
        }
        if (available == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            return false;
        }
        DWORD received = 0;
        if (!ReadFile(handle_, chunk, static_cast<DWORD>(sizeof(chunk)), &received, nullptr)) {
            if (GetLastError() == ERROR_BROKEN_PIPE) {
                eof_ = true;
                return take_line(line);
            }
            throw std::runtime_error("cannot read the desktop input pipe");
        }
        count = received;
#else
        pollfd descriptor{STDIN_FILENO, POLLIN, 0};
        const auto status = poll(&descriptor, 1, 20);
        if (status < 0 && errno != EINTR)
            throw std::runtime_error("cannot poll the desktop input pipe");
        if (status <= 0)
            return false;
        const auto received = read(STDIN_FILENO, chunk, sizeof(chunk));
        if (received < 0 && errno == EINTR)
            return false;
        if (received < 0)
            throw std::runtime_error("cannot read the desktop input pipe");
        count = static_cast<std::size_t>(received);
#endif
        if (count == 0)
            eof_ = true;
        buffer_.append(chunk, count);
        if (buffer_.size() > kMaxCommandBytes && buffer_.find('\n') > kMaxCommandBytes)
            throw std::runtime_error("desktop input command exceeds the 1 MiB limit");
        return take_line(line);
    }

  private:
    bool take_line(std::string& line) {
        const auto end = buffer_.find('\n');
        if (end == std::string::npos) {
            if (eof_ && !buffer_.empty())
                throw std::runtime_error("desktop input ended in an incomplete JSON command");
            return false;
        }
        if (end > kMaxCommandBytes)
            throw std::runtime_error("desktop input command exceeds the 1 MiB limit");
        line = buffer_.substr(0, end);
        buffer_.erase(0, end + 1);
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        return true;
    }

    std::string buffer_;
    bool eof_{false};
#ifdef _WIN32
    HANDLE handle_{INVALID_HANDLE_VALUE};
#endif
};

} // namespace trtmc::examples::windows_voicechat
