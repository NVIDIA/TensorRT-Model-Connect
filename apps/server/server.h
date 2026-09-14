/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/task.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <memory>
#include <string>

namespace trtmc::server {

struct Options {
    std::string bundle;
    std::string runtime_root;
    std::string model_name;
    std::string host{"127.0.0.1"};
    std::uint16_t port{8000};
    std::size_t queue_capacity{16};
    std::size_t max_queued_bytes{1024 * 1024};
    std::size_t max_body_bytes{1024 * 1024};
    std::size_t max_prompt_bytes{256 * 1024};
    std::int32_t max_new_tokens{4096};
    std::chrono::milliseconds shutdown_grace{30000};
};

struct Response {
    int status{500};
    std::string body;
    std::string request_id;
    double queue_ms{0.0};
    double inference_ms{0.0};
    double total_ms{0.0};
    double task_setup_ms{0.0};
    double task_prefill_ms{0.0};
    double task_decode_ms{0.0};
};

using Reply = std::function<void(Response)>;

class InferenceService {
  public:
    InferenceService(ITask& task, Options options);
    ~InferenceService();
    InferenceService(const InferenceService&) = delete;
    InferenceService& operator=(const InferenceService&) = delete;

    void start();
    void submit(const std::string& path, const std::string& body, Reply reply);
    Response get(const std::string& path) const;
    void begin_draining();
    void wait_stopped();
    void set_stopped_callback(std::function<void()> callback);

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class HttpServer {
  public:
    HttpServer(ITask& task, Options options);
    ~HttpServer();
    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    void run(std::ostream& log);
    void request_stop();
    std::uint16_t port() const noexcept;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

Options parse_args(int argc, char** argv);
void print_usage(std::ostream& output);
int run(int argc, char** argv, std::ostream& output, std::ostream& error);

} // namespace trtmc::server
