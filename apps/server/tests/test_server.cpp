/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "server/server.h"

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

class FakeText final : public trtmc::ITextGeneration {
  public:
    std::int32_t default_max_new_tokens() const override { return 17; }

    trtmc::TextResult generate(const std::string& prompt,
                               const trtmc::TextGenerationConfig& config) override {
        const int active = ++active_;
        int observed = max_active_.load();
        while (active > observed && !max_active_.compare_exchange_weak(observed, active)) {
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            prompts_.push_back(prompt);
            configs_.push_back(config);
            entered_ = true;
        }
        entered_cv_.notify_all();
        {
            std::unique_lock<std::mutex> lock(gate_mutex_);
            gate_cv_.wait(lock, [this] { return !blocked_; });
        }
        --active_;
        return {"reply:" + prompt, {1, 2, 3}, 1.0, 2.0};
    }

    void block() {
        {
            std::lock_guard<std::mutex> lock(gate_mutex_);
            blocked_ = true;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        entered_ = false;
    }

    void wait_entered() {
        std::unique_lock<std::mutex> lock(mutex_);
        entered_cv_.wait(lock, [this] { return entered_; });
    }

    void release() {
        {
            std::lock_guard<std::mutex> lock(gate_mutex_);
            blocked_ = false;
        }
        gate_cv_.notify_all();
    }

    std::vector<std::string> prompts() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return prompts_;
    }

    std::vector<trtmc::TextGenerationConfig> configs() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return configs_;
    }

    int max_active() const { return max_active_.load(); }

  private:
    mutable std::mutex mutex_;
    std::condition_variable entered_cv_;
    bool entered_{false};
    std::vector<std::string> prompts_;
    std::vector<trtmc::TextGenerationConfig> configs_;
    std::mutex gate_mutex_;
    std::condition_variable gate_cv_;
    bool blocked_{false};
    std::atomic<int> active_{0};
    std::atomic<int> max_active_{0};
};

std::future<trtmc::server::Response> submit(trtmc::server::InferenceService& service,
                                            const std::string& path, Json request) {
    auto promise = std::make_shared<std::promise<trtmc::server::Response>>();
    auto future = promise->get_future();
    service.submit(path, request.dump(), [promise](trtmc::server::Response response) mutable {
        promise->set_value(std::move(response));
    });
    return future;
}

Json completion_request(const std::string& prompt) {
    return {{"model", "test-model"}, {"prompt", prompt}, {"max_tokens", 8}};
}

std::string http_exchange(std::uint16_t port, const std::string& request) {
    const int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd < 0)
        throw std::runtime_error("failed to create test socket");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
    if (connect(socket_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        close(socket_fd);
        throw std::runtime_error("failed to connect to test server");
    }
    std::size_t sent = 0;
    while (sent < request.size()) {
        const auto count = send(socket_fd, request.data() + sent, request.size() - sent, 0);
        if (count <= 0) {
            close(socket_fd);
            throw std::runtime_error("failed to send test request");
        }
        sent += static_cast<std::size_t>(count);
    }
    std::string response;
    char buffer[4096];
    while (true) {
        const auto count = recv(socket_fd, buffer, sizeof(buffer), 0);
        if (count < 0) {
            close(socket_fd);
            throw std::runtime_error("failed to receive test response");
        }
        if (count == 0)
            break;
        response.append(buffer, static_cast<std::size_t>(count));
        const auto header_end = response.find("\r\n\r\n");
        const auto length_header = response.find("\r\nContent-Length: ");
        if (header_end != std::string::npos && length_header != std::string::npos) {
            const auto value_begin = length_header + 18;
            const auto value_end = response.find("\r\n", value_begin);
            if (value_end != std::string::npos) {
                const auto body_size =
                    std::stoull(response.substr(value_begin, value_end - value_begin));
                if (response.size() >= header_end + 4 + body_size)
                    break;
            }
        }
    }
    close(socket_fd);
    return response;
}

void test_http_endpoint() {
    FakeText task;
    trtmc::server::Options options;
    options.model_name = "test-model";
    options.host = "127.0.0.1";
    options.port = 0;
    trtmc::server::HttpServer server(task, options);
    std::ostringstream log;
    std::exception_ptr server_error;
    std::thread thread([&] {
        try {
            server.run(log);
        } catch (...) {
            server_error = std::current_exception();
        }
    });

    const auto health =
        http_exchange(server.port(),
                      "GET /health/ready HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    check(health.find("HTTP/1.1 200 OK") != std::string::npos, "HTTP readiness endpoint");

    const auto body = completion_request("network").dump();
    const auto request = "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
                         "Content-Type: application/json\r\nConnection: close\r\nContent-Length: " +
                         std::to_string(body.size()) + "\r\n\r\n" + body;
    const auto completion = http_exchange(server.port(), request);
    check(completion.find("HTTP/1.1 200 OK") != std::string::npos, "HTTP completion endpoint");
    check(completion.find("reply:network") != std::string::npos, "HTTP completion response body");
    const auto wrong_method = http_exchange(
        server.port(),
        "GET /v1/completions HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    check(wrong_method.find("HTTP/1.1 405 Method Not Allowed") != std::string::npos,
          "HTTP method validation");

    server.request_stop();
    thread.join();
    check(server_error == nullptr, "HTTP server exits cleanly");
}

void test_protocol_mapping() {
    FakeText task;
    trtmc::server::Options options;
    options.model_name = "test-model";
    trtmc::server::InferenceService service(task, options);
    service.start();

    auto completion = submit(service, "/v1/completions",
                             {{"model", "test-model"},
                              {"prompt", "hello"},
                              {"max_tokens", 9},
                              {"temperature", 0.25},
                              {"top_p", 0.75},
                              {"seed", 4},
                              {"n", 1},
                              {"stream", false}})
                          .get();
    check(completion.status == 200, "completion succeeds");
    const auto completion_json = Json::parse(completion.body);
    check(completion_json.at("object") == "text_completion", "completion object kind");
    check(completion_json.at("model") == "test-model", "completion model");
    check(completion_json.at("choices").at(0).at("text") == "reply:hello",
          "completion output text");
    check(!completion.request_id.empty(), "completion request id");

    auto chat = submit(service, "/v1/chat/completions",
                       {{"model", "test-model"},
                        {"messages",
                         {{{"role", "system"}, {"content", "be brief"}},
                          {{"role", "user"}, {"content", "hello chat"}}}},
                        {"max_completion_tokens", 11}})
                    .get();
    check(chat.status == 200, "chat succeeds");
    const auto chat_json = Json::parse(chat.body);
    check(chat_json.at("object") == "chat.completion", "chat object kind");
    check(chat_json.at("choices").at(0).at("message").at("content") == "reply:hello chat",
          "chat output text");

    const auto prompts = task.prompts();
    const auto configs = task.configs();
    check(prompts == std::vector<std::string>({"hello", "hello chat"}), "prompts mapped");
    check(configs.size() == 2, "configs captured");
    if (configs.size() == 2) {
        check(configs[0].max_new_tokens == 9, "completion max tokens mapped");
        check(configs[0].temperature == 0.25F, "temperature mapped");
        check(configs[0].top_p == 0.75F, "top-p mapped");
        check(configs[0].seed == 4, "seed mapped");
        check(configs[0].top_k == 0, "OpenAI sampling uses the full vocabulary");
        check(configs[1].use_chat_template, "chat template enabled");
        check(configs[1].seed >= 0, "missing seed receives a request seed");
        check(configs[1].system_prompt == "be brief", "system prompt mapped");
    }

    check(service.get("/health/live").status == 200, "liveness succeeds");
    check(service.get("/health/ready").status == 200, "readiness succeeds");
    const auto models = Json::parse(service.get("/v1/models").body);
    check(models.at("data").at(0).at("id") == "test-model", "model listing");
    const auto metrics = service.get("/metrics").body;
    check(metrics.find("trtmc_server_inference_duration_seconds_count 2") != std::string::npos,
          "metrics count completions");
    check(metrics.find("trtmc_server_task_prefill_duration_seconds_count 2") != std::string::npos,
          "metrics count family timings");

    service.begin_draining();
    service.wait_stopped();
    check(service.get("/health/ready").status == 503, "draining is not ready");
}

void test_validation() {
    FakeText task;
    trtmc::server::Options options;
    options.model_name = "test-model";
    options.max_body_bytes = 1024;
    options.max_prompt_bytes = 8;
    options.max_new_tokens = 32;
    trtmc::server::InferenceService service(task, options);
    service.start();

    check(submit(service, "/missing", {{"model", "test-model"}}).get().status == 404,
          "unknown route rejected");
    check(submit(service, "/v1/completions", {{"model", "other"}, {"prompt", "x"}}).get().status ==
              404,
          "unknown model rejected");
    check(submit(service, "/v1/completions",
                 {{"model", "test-model"}, {"prompt", "x"}, {"stream", true}})
                  .get()
                  .status == 400,
          "streaming rejected");
    check(submit(service, "/v1/completions",
                 {{"model", "test-model"}, {"prompt", "x"}, {"unknown", 1}})
                  .get()
                  .status == 400,
          "unknown field rejected");
    check(submit(service, "/v1/completions", {{"model", "test-model"}, {"prompt", "123456789"}})
                  .get()
                  .status == 413,
          "large prompt rejected");
    check(submit(service, "/v1/completions",
                 {{"model", "test-model"}, {"prompt", "x"}, {"max_tokens", 33}})
                  .get()
                  .status == 400,
          "token limit enforced");
    check(
        submit(service, "/v1/completions", {{"model", "test-model"}, {"prompt", "x"}, {"seed", -1}})
                .get()
                .status == 400,
        "negative seed rejected");
    check(submit(service, "/v1/chat/completions",
                 {{"model", "test-model"},
                  {"messages",
                   {{{"role", "user"}, {"content", "first"}},
                    {{"role", "user"}, {"content", "second"}}}}})
                  .get()
                  .status == 400,
          "multi-turn chat rejected");

    service.begin_draining();
    service.wait_stopped();
}

void test_bounded_serial_queue() {
    FakeText task;
    task.block();
    trtmc::server::Options options;
    options.model_name = "test-model";
    options.queue_capacity = 1;
    options.max_queued_bytes = 1024;
    trtmc::server::InferenceService service(task, options);
    service.start();

    auto active = submit(service, "/v1/completions", completion_request("active"));
    task.wait_entered();
    auto queued = submit(service, "/v1/completions", completion_request("queued"));
    auto rejected = submit(service, "/v1/completions", completion_request("rejected"));
    check(rejected.get().status == 429, "full queue rejected");
    task.release();
    check(active.get().status == 200, "active request completes");
    check(queued.get().status == 200, "queued request completes");
    check(task.max_active() == 1, "task calls are serialized");

    service.begin_draining();
    auto draining = submit(service, "/v1/completions", completion_request("late"));
    check(draining.get().status == 503, "draining rejects admission");
    service.wait_stopped();
}

void test_shutdown_deadline() {
    FakeText task;
    task.block();
    trtmc::server::Options options;
    options.model_name = "test-model";
    options.shutdown_grace = std::chrono::milliseconds(0);
    trtmc::server::InferenceService service(task, options);
    service.start();

    auto active = submit(service, "/v1/completions", completion_request("active"));
    task.wait_entered();
    auto queued = submit(service, "/v1/completions", completion_request("queued"));
    service.begin_draining();
    task.release();
    check(active.get().status == 200, "active request finishes during shutdown");
    check(queued.get().status == 503, "expired queued request rejected during shutdown");
    service.wait_stopped();
}

void test_argument_parsing() {
    std::vector<std::string> arguments{
        "trtmc-server", "model.bundle", "--runtime-root",   "/lib", "--model-name", "model",
        "--port",       "9000",         "--queue-capacity", "3"};
    std::vector<char*> argv;
    for (auto& argument : arguments)
        argv.push_back(argument.data());
    const auto options = trtmc::server::parse_args(static_cast<int>(argv.size()), argv.data());
    check(options.bundle == "model.bundle", "bundle parsed");
    check(options.runtime_root == "/lib", "runtime root parsed");
    check(options.model_name == "model", "model name parsed");
    check(options.port == 9000, "port parsed");
    check(options.queue_capacity == 3, "queue capacity parsed");

    std::vector<std::string> default_runtime_arguments{"trtmc-server", "model.bundle",
                                                       "--model-name", "model"};
    argv.clear();
    for (auto& argument : default_runtime_arguments)
        argv.push_back(argument.data());
    const auto default_runtime_options =
        trtmc::server::parse_args(static_cast<int>(argv.size()), argv.data());
    check(default_runtime_options.runtime_root.empty(), "runtime root defaults to automatic");

    std::vector<std::string> invalid_arguments{
        "trtmc-server", "model.bundle", "--runtime-root",   "/lib",
        "--model-name", "model",        "--queue-capacity", "-1"};
    argv.clear();
    for (auto& argument : invalid_arguments)
        argv.push_back(argument.data());
    bool rejected = false;
    try {
        (void)trtmc::server::parse_args(static_cast<int>(argv.size()), argv.data());
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "negative size rejected");
}

} // namespace

int main() {
    test_http_endpoint();
    test_protocol_mapping();
    test_validation();
    test_bounded_serial_queue();
    test_shutdown_deadline();
    test_argument_parsing();
    std::cerr << (failures == 0 ? "ALL PASSED\n" : "SOME FAILED\n");
    return failures;
}
