/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "server/server.h"

#include "trtmc/runtime/family_loader.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <deque>
#include <event2/buffer.h>
#include <event2/event.h>
#include <event2/http.h>
#include <event2/keyvalq_struct.h>
#include <event2/thread.h>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <netinet/in.h>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace trtmc::server {
namespace {

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

bool is_generation_route(const std::string& route) {
    return route == "/v1/completions" || route == "/v1/chat/completions";
}

bool is_get_route(const std::string& route) {
    return route == "/health/live" || route == "/health/ready" || route == "/v1/models" ||
           route == "/metrics";
}

std::string json_body(Json value) {
    return value.dump() + "\n";
}

Response error_response(int status, const std::string& request_id, const std::string& message,
                        const std::string& code, const std::string& param = {}) {
    Json error{{"message", message}, {"type", "invalid_request_error"}, {"code", code}};
    error["param"] = param.empty() ? Json(nullptr) : Json(param);
    return {status, json_body({{"error", std::move(error)}}), request_id};
}

double milliseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

std::int64_t unix_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string request_id(std::uint64_t value) {
    std::ostringstream output;
    output << "trtmc-" << std::hex << unix_seconds() << '-' << value;
    return output.str();
}

void require_keys(const Json& object, const std::set<std::string>& allowed) {
    if (!object.is_object())
        throw std::invalid_argument("request body must be a JSON object");
    for (const auto& [key, unused] : object.items()) {
        (void)unused;
        if (allowed.count(key) == 0)
            throw std::invalid_argument("unsupported field: " + key);
    }
}

template <typename T>
T optional_number(const Json& request, const char* name, T fallback) {
    const auto it = request.find(name);
    if (it == request.end() || it->is_null())
        return fallback;
    if constexpr (std::is_integral_v<T>) {
        if (!it->is_number_integer())
            throw std::invalid_argument(std::string(name) + " must be an integer");
    } else if (!it->is_number()) {
        throw std::invalid_argument(std::string(name) + " must be a number");
    }
    return it->get<T>();
}

struct PreparedRequest {
    std::string route;
    std::string prompt;
    TextGenerationConfig config;
};

PreparedRequest prepare_request(const std::string& path, const std::string& body,
                                ITextGeneration& text, const Options& options,
                                std::int32_t default_seed) {
    Json request;
    try {
        request = Json::parse(body);
    } catch (const Json::parse_error&) {
        throw std::invalid_argument("request body is not valid JSON");
    }

    const bool chat = path == "/v1/chat/completions";
    if (!chat && path != "/v1/completions")
        throw std::out_of_range("route not found");
    const std::set<std::string> allowed =
        chat ? std::set<std::string>{"model",       "messages",
                                     "max_tokens",  "max_completion_tokens",
                                     "temperature", "top_p",
                                     "seed",        "n",
                                     "stream"}
             : std::set<std::string>{"model", "prompt", "max_tokens", "temperature",
                                     "top_p", "seed",   "n",          "stream"};
    require_keys(request, allowed);

    if (!request.contains("model") || !request.at("model").is_string())
        throw std::invalid_argument("model must be a string");
    if (request.at("model").get<std::string>() != options.model_name)
        throw std::domain_error("requested model is not served by this process");

    if (request.contains("stream") && !request.at("stream").is_null()) {
        if (!request.at("stream").is_boolean())
            throw std::invalid_argument("stream must be a boolean");
        if (request.at("stream").get<bool>())
            throw std::invalid_argument("streaming is not supported by this server");
    }
    const auto n = optional_number<std::int32_t>(request, "n", 1);
    if (n != 1)
        throw std::invalid_argument("n must be 1");

    TextGenerationConfig config;
    config.max_new_tokens = std::min(text.default_max_new_tokens(), options.max_new_tokens);
    if (chat && request.contains("max_tokens") && request.contains("max_completion_tokens"))
        throw std::invalid_argument("max_tokens and max_completion_tokens are mutually exclusive");
    const char* max_tokens_name =
        chat && request.contains("max_completion_tokens") ? "max_completion_tokens" : "max_tokens";
    config.max_new_tokens =
        optional_number<std::int32_t>(request, max_tokens_name, config.max_new_tokens);
    if (config.max_new_tokens < 1 || config.max_new_tokens > options.max_new_tokens)
        throw std::invalid_argument(std::string(max_tokens_name) + " is outside the server limit");
    config.temperature = optional_number<float>(request, "temperature", config.temperature);
    if (config.temperature < 0.0F || config.temperature > 2.0F)
        throw std::invalid_argument("temperature must be between 0 and 2");
    config.top_p = optional_number<float>(request, "top_p", config.top_p);
    if (config.top_p <= 0.0F || config.top_p > 1.0F)
        throw std::invalid_argument("top_p must be greater than 0 and at most 1");
    config.top_k = 0;
    config.seed = optional_number<std::int32_t>(request, "seed", default_seed);
    if (config.seed < 0)
        throw std::invalid_argument("seed must be non-negative");

    std::string prompt;
    if (!chat) {
        if (!request.contains("prompt") || !request.at("prompt").is_string())
            throw std::invalid_argument("prompt must be one string");
        prompt = request.at("prompt").get<std::string>();
    } else {
        if (!request.contains("messages") || !request.at("messages").is_array())
            throw std::invalid_argument("messages must be an array");
        const auto& messages = request.at("messages");
        if (messages.empty() || messages.size() > 2)
            throw std::invalid_argument(
                "messages must contain one user message and at most one preceding system message");
        std::size_t index = 0;
        if (messages.size() == 2) {
            const auto& system = messages.at(0);
            require_keys(system, {"role", "content"});
            if (system.value("role", "") != "system" || !system.contains("content") ||
                !system.at("content").is_string()) {
                throw std::invalid_argument("the first message must be a text system message");
            }
            config.system_prompt = system.at("content").get<std::string>();
            index = 1;
        }
        const auto& user = messages.at(index);
        require_keys(user, {"role", "content"});
        if (user.value("role", "") != "user" || !user.contains("content") ||
            !user.at("content").is_string()) {
            throw std::invalid_argument("the final message must be a text user message");
        }
        prompt = user.at("content").get<std::string>();
        config.use_chat_template = true;
    }
    if (prompt.size() + config.system_prompt.size() > options.max_prompt_bytes)
        throw std::length_error("prompt exceeds the server byte limit");
    return {path, std::move(prompt), std::move(config)};
}

std::string reason_phrase(int status) {
    switch (status) {
    case 200:
        return "OK";
    case 400:
        return "Bad Request";
    case 404:
        return "Not Found";
    case 405:
        return "Method Not Allowed";
    case 413:
        return "Payload Too Large";
    case 429:
        return "Too Many Requests";
    case 500:
        return "Internal Server Error";
    case 503:
        return "Service Unavailable";
    default:
        return "Error";
    }
}

std::size_t parse_size(const std::string& text, const char* option, std::size_t minimum = 1) {
    if (text.empty() || text.front() == '-' || text.front() == '+')
        throw std::invalid_argument(std::string(option) + " must be an unsigned integer");
    std::size_t consumed = 0;
    unsigned long long value = 0;
    try {
        value = std::stoull(text, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(option) + " must be an unsigned integer");
    }
    if (consumed != text.size() || value < minimum ||
        value > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument(std::string(option) + " is outside the supported range");
    return static_cast<std::size_t>(value);
}

} // namespace

class InferenceService::Impl {
  public:
    struct Work {
        PreparedRequest request;
        std::string id;
        std::size_t bytes{0};
        Clock::time_point admitted;
        Reply reply;
    };

    Impl(ITask& task, Options options)
        : text_(dynamic_cast<ITextGeneration*>(&task)), options_(std::move(options)) {
        if (text_ == nullptr)
            throw std::invalid_argument("bundle task does not implement text generation");
        if (options_.model_name.empty())
            throw std::invalid_argument("model name must not be empty");
    }

    ~Impl() {
        begin_draining();
        if (worker_.joinable())
            worker_.join();
    }

    void start() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (started_)
            throw std::logic_error("inference service is already started");
        started_ = true;
        accepting_ = true;
        worker_ = std::thread([this] { worker_loop(); });
    }

    void record(const std::string& route, int status) {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        const auto metric_route = is_generation_route(route) ? route : "unknown";
        ++request_counts_[metric_route + "\n" + std::to_string(status)];
    }

    void submit(const std::string& path, const std::string& body, Reply reply) {
        const auto sequence = next_request_id_.fetch_add(1, std::memory_order_relaxed);
        const auto id = request_id(sequence);
        if (body.size() > options_.max_body_bytes) {
            record(path, 413);
            reply(error_response(413, id, "request body exceeds the server limit",
                                 "content_too_large"));
            return;
        }

        PreparedRequest prepared;
        try {
            const auto default_seed = static_cast<std::int32_t>(
                sequence % static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()));
            prepared = prepare_request(path, body, *text_, options_, default_seed);
        } catch (const std::out_of_range& error) {
            record(path, 404);
            reply(error_response(404, id, error.what(), "route_not_found"));
            return;
        } catch (const std::domain_error& error) {
            record(path, 404);
            reply(error_response(404, id, error.what(), "model_not_found", "model"));
            return;
        } catch (const std::length_error& error) {
            record(path, 413);
            reply(error_response(413, id, error.what(), "content_too_large", "prompt"));
            return;
        } catch (const std::exception& error) {
            record(path, 400);
            reply(error_response(400, id, error.what(), "invalid_request"));
            return;
        }

        Response rejection;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!accepting_) {
                rejection = error_response(503, id, "server is not accepting inference requests",
                                           "server_draining");
            } else if (queue_.size() >= options_.queue_capacity ||
                       body.size() > options_.max_queued_bytes -
                                         std::min(queued_bytes_, options_.max_queued_bytes)) {
                rejection = error_response(429, id, "inference queue is full", "queue_full");
            } else {
                queued_bytes_ += body.size();
                queue_.push_back(
                    {std::move(prepared), id, body.size(), Clock::now(), std::move(reply)});
            }
        }
        if (rejection.status != 500) {
            record(path, rejection.status);
            reply(std::move(rejection));
            return;
        }
        cv_.notify_one();
    }

    Response get(const std::string& path) const {
        if (path == "/health/live")
            return {200, json_body({{"status", "live"}}), {}};
        if (path == "/health/ready") {
            std::lock_guard<std::mutex> lock(mutex_);
            return accepting_ ? Response{200, json_body({{"status", "ready"}}), {}}
                              : Response{503, json_body({{"status", "not_ready"}}), {}};
        }
        if (path == "/v1/models") {
            return {200,
                    json_body({{"object", "list"},
                               {"data", Json::array({{{"id", options_.model_name},
                                                      {"object", "model"},
                                                      {"created", 0},
                                                      {"owned_by", "tensorrt-model-connect"}}})}}),
                    {}};
        }
        if (path == "/metrics")
            return {200, metrics(), {}};
        return error_response(404, {}, "route not found", "route_not_found");
    }

    void begin_draining() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (draining_)
                return;
            accepting_ = false;
            draining_ = true;
            drain_deadline_ = Clock::now() + options_.shutdown_grace;
        }
        cv_.notify_all();
    }

    void wait_stopped() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!started_)
            return;
        stopped_cv_.wait(lock, [this] { return stopped_; });
        lock.unlock();
        if (worker_.joinable())
            worker_.join();
    }

    void set_stopped_callback(std::function<void()> callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_callback_ = std::move(callback);
    }

  private:
    std::string metrics() const {
        std::ostringstream output;
        output << "# TYPE trtmc_server_ready gauge\ntrtmc_server_ready ";
        {
            std::lock_guard<std::mutex> lock(mutex_);
            output << (accepting_ ? 1 : 0) << '\n';
            output << "# TYPE trtmc_server_queue_depth gauge\ntrtmc_server_queue_depth "
                   << queue_.size() << '\n';
            output << "# TYPE trtmc_server_queue_bytes gauge\ntrtmc_server_queue_bytes "
                   << queued_bytes_ << '\n';
            output << "# TYPE trtmc_server_active_requests gauge\ntrtmc_server_active_requests "
                   << (active_ ? 1 : 0) << '\n';
        }
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        output << "# TYPE trtmc_server_requests_total counter\n";
        for (const auto& [key, count] : request_counts_) {
            const auto split = key.find('\n');
            output << "trtmc_server_requests_total{route=\"" << key.substr(0, split)
                   << "\",status=\"" << key.substr(split + 1) << "\"} " << count << '\n';
        }
        output << "# TYPE trtmc_server_queue_duration_seconds summary\n"
               << "trtmc_server_queue_duration_seconds_sum " << queue_seconds_sum_ << '\n'
               << "trtmc_server_queue_duration_seconds_count " << completed_ << '\n'
               << "# TYPE trtmc_server_inference_duration_seconds summary\n"
               << "trtmc_server_inference_duration_seconds_sum " << inference_seconds_sum_ << '\n'
               << "trtmc_server_inference_duration_seconds_count " << completed_ << '\n'
               << "# TYPE trtmc_server_task_setup_duration_seconds summary\n"
               << "trtmc_server_task_setup_duration_seconds_sum " << task_setup_seconds_sum_ << '\n'
               << "trtmc_server_task_setup_duration_seconds_count " << succeeded_ << '\n'
               << "# TYPE trtmc_server_task_prefill_duration_seconds summary\n"
               << "trtmc_server_task_prefill_duration_seconds_sum " << task_prefill_seconds_sum_
               << '\n'
               << "trtmc_server_task_prefill_duration_seconds_count " << succeeded_ << '\n'
               << "# TYPE trtmc_server_task_decode_duration_seconds summary\n"
               << "trtmc_server_task_decode_duration_seconds_sum " << task_decode_seconds_sum_
               << '\n'
               << "trtmc_server_task_decode_duration_seconds_count " << succeeded_ << '\n';
        return output.str();
    }

    void worker_loop() {
        while (true) {
            Work work;
            bool drain_expired = false;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return draining_ || !queue_.empty(); });
                if (queue_.empty() && draining_)
                    break;
                work = std::move(queue_.front());
                queue_.pop_front();
                queued_bytes_ -= work.bytes;
                active_ = true;
                drain_expired = draining_ && Clock::now() > drain_deadline_;
            }

            const auto started = Clock::now();
            Response response;
            if (drain_expired) {
                response = error_response(503, work.id, "shutdown grace period expired",
                                          "server_draining");
            } else {
                try {
                    const auto result = text_->generate(work.request.prompt, work.request.config);
                    response.task_setup_ms = result.setup_ms;
                    response.task_prefill_ms = result.prefill_ms;
                    response.task_decode_ms = result.decode_ms;
                    const Json choice =
                        work.request.route == "/v1/chat/completions"
                            ? Json{{"index", 0},
                                   {"message", {{"role", "assistant"}, {"content", result.text}}},
                                   {"logprobs", nullptr},
                                   {"finish_reason", nullptr}}
                            : Json{{"index", 0},
                                   {"text", result.text},
                                   {"logprobs", nullptr},
                                   {"finish_reason", nullptr}};
                    response.status = 200;
                    response.request_id = work.id;
                    response.body =
                        json_body({{"id", work.id},
                                   {"object", work.request.route == "/v1/chat/completions"
                                                  ? "chat.completion"
                                                  : "text_completion"},
                                   {"created", unix_seconds()},
                                   {"model", options_.model_name},
                                   {"choices", Json::array({choice})}});
                } catch (const std::invalid_argument& error) {
                    response = error_response(400, work.id, error.what(), "invalid_request");
                } catch (const std::exception&) {
                    response =
                        error_response(500, work.id, "model execution failed", "internal_error");
                }
            }
            const auto finished = Clock::now();
            response.queue_ms = milliseconds(started - work.admitted);
            response.inference_ms = milliseconds(finished - started);
            response.total_ms = milliseconds(finished - work.admitted);
            record(work.request.route, response.status);
            {
                std::lock_guard<std::mutex> lock(metrics_mutex_);
                queue_seconds_sum_ += response.queue_ms / 1000.0;
                inference_seconds_sum_ += response.inference_ms / 1000.0;
                ++completed_;
                if (response.status == 200) {
                    task_setup_seconds_sum_ += response.task_setup_ms / 1000.0;
                    task_prefill_seconds_sum_ += response.task_prefill_ms / 1000.0;
                    task_decode_seconds_sum_ += response.task_decode_ms / 1000.0;
                    ++succeeded_;
                }
            }
            try {
                work.reply(std::move(response));
            } catch (const std::exception&) {
                // A disconnected transport must not terminate the inference worker.
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                active_ = false;
            }
        }

        std::function<void()> callback;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
            callback = stopped_callback_;
        }
        stopped_cv_.notify_all();
        if (callback)
            callback();
    }

    ITextGeneration* text_;
    Options options_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable stopped_cv_;
    std::deque<Work> queue_;
    std::size_t queued_bytes_{0};
    bool accepting_{false};
    bool started_{false};
    bool draining_{false};
    bool stopped_{false};
    bool active_{false};
    Clock::time_point drain_deadline_{};
    std::thread worker_;
    std::function<void()> stopped_callback_;
    std::atomic<std::uint64_t> next_request_id_{1};

    mutable std::mutex metrics_mutex_;
    std::unordered_map<std::string, std::uint64_t> request_counts_;
    std::uint64_t completed_{0};
    std::uint64_t succeeded_{0};
    double queue_seconds_sum_{0.0};
    double inference_seconds_sum_{0.0};
    double task_setup_seconds_sum_{0.0};
    double task_prefill_seconds_sum_{0.0};
    double task_decode_seconds_sum_{0.0};
};

InferenceService::InferenceService(ITask& task, Options options)
    : impl_(std::make_unique<Impl>(task, std::move(options))) {}
InferenceService::~InferenceService() = default;
void InferenceService::start() {
    impl_->start();
}
void InferenceService::submit(const std::string& path, const std::string& body, Reply reply) {
    impl_->submit(path, body, std::move(reply));
}
Response InferenceService::get(const std::string& path) const {
    return impl_->get(path);
}
void InferenceService::begin_draining() {
    impl_->begin_draining();
}
void InferenceService::wait_stopped() {
    impl_->wait_stopped();
}
void InferenceService::set_stopped_callback(std::function<void()> callback) {
    impl_->set_stopped_callback(std::move(callback));
}

namespace {

struct Deferred {
    std::function<void()> function;
};

void run_deferred(evutil_socket_t, short, void* opaque) {
    std::unique_ptr<Deferred> deferred(static_cast<Deferred*>(opaque));
    deferred->function();
}

bool defer(event_base* base, std::function<void()> function) noexcept {
    auto deferred = std::make_unique<Deferred>();
    deferred->function = std::move(function);
    timeval immediate{0, 0};
    if (event_base_once(base, -1, EV_TIMEOUT, run_deferred, deferred.get(), &immediate) != 0)
        return false;
    (void)deferred.release();
    return true;
}

} // namespace

class HttpServer::Impl {
  public:
    Impl(ITask& task, Options options) : options_(std::move(options)), service_(task, options_) {
        static std::once_flag threads_once;
        std::call_once(threads_once, [] {
            if (evthread_use_pthreads() != 0)
                throw std::runtime_error("failed to enable Libevent thread support");
        });
        try {
            base_ = event_base_new();
            if (base_ == nullptr)
                throw std::runtime_error("failed to create Libevent base");
            http_ = evhttp_new(base_);
            if (http_ == nullptr)
                throw std::runtime_error("failed to create HTTP server");
            evhttp_set_max_body_size(http_, static_cast<ev_ssize_t>(options_.max_body_bytes));
            evhttp_set_gencb(http_, request_callback, this);
            bound_ = evhttp_bind_socket_with_handle(http_, options_.host.c_str(), options_.port);
            if (bound_ == nullptr)
                throw std::runtime_error("failed to bind " + options_.host + ':' +
                                         std::to_string(options_.port));
            sockaddr_storage address{};
            socklen_t address_length = sizeof(address);
            if (getsockname(evhttp_bound_socket_get_fd(bound_),
                            reinterpret_cast<sockaddr*>(&address), &address_length) != 0)
                throw std::runtime_error("failed to inspect bound HTTP socket");
            if (address.ss_family == AF_INET)
                options_.port = ntohs(reinterpret_cast<sockaddr_in*>(&address)->sin_port);
            else if (address.ss_family == AF_INET6)
                options_.port = ntohs(reinterpret_cast<sockaddr_in6*>(&address)->sin6_port);
            sigint_ = evsignal_new(base_, SIGINT, signal_callback, this);
            sigterm_ = evsignal_new(base_, SIGTERM, signal_callback, this);
            if (sigint_ == nullptr || sigterm_ == nullptr || event_add(sigint_, nullptr) != 0 ||
                event_add(sigterm_, nullptr) != 0) {
                throw std::runtime_error("failed to install shutdown signal handlers");
            }
        } catch (...) {
            cleanup();
            throw;
        }
        service_.set_stopped_callback([this] {
            if (!defer(base_, [this] { event_base_loopexit(base_, nullptr); }))
                event_base_loopbreak(base_);
        });
    }

    ~Impl() {
        service_.begin_draining();
        service_.wait_stopped();
        cleanup();
    }

    void cleanup() noexcept {
        if (sigint_ != nullptr)
            event_free(sigint_);
        sigint_ = nullptr;
        if (sigterm_ != nullptr)
            event_free(sigterm_);
        sigterm_ = nullptr;
        if (http_ != nullptr)
            evhttp_free(http_);
        http_ = nullptr;
        bound_ = nullptr;
        if (base_ != nullptr)
            event_base_free(base_);
        base_ = nullptr;
    }

    void run(std::ostream& log) {
        log_ = &log;
        service_.start();
        log << json_body({{"event", "server_ready"},
                          {"host", options_.host},
                          {"port", options_.port},
                          {"model", options_.model_name}})
            << std::flush;
        const int result = event_base_dispatch(base_);
        service_.begin_draining();
        service_.wait_stopped();
        if (result == -1)
            throw std::runtime_error("HTTP event loop failed");
        log << json_body({{"event", "server_stopped"}}) << std::flush;
    }

    void request_stop() { service_.begin_draining(); }

    std::uint16_t port() const noexcept { return options_.port; }

  private:
    static void signal_callback(evutil_socket_t, short, void* opaque) {
        static_cast<Impl*>(opaque)->request_stop();
    }

    static void request_callback(evhttp_request* request, void* opaque) {
        static_cast<Impl*>(opaque)->handle(request);
    }

    void send(evhttp_request* request, const std::string& route, Response response) {
        auto* headers = evhttp_request_get_output_headers(request);
        evhttp_add_header(headers, "Content-Type",
                          route == "/metrics" ? "text/plain; version=0.0.4" : "application/json");
        evhttp_add_header(headers, "Cache-Control", "no-store");
        if (!response.request_id.empty())
            evhttp_add_header(headers, "X-Request-ID", response.request_id.c_str());
        if (response.status == 429 || response.status == 503)
            evhttp_add_header(headers, "Retry-After", "1");
        auto* output = evhttp_request_get_output_buffer(request);
        evbuffer_add(output, response.body.data(), response.body.size());
        const auto reason = reason_phrase(response.status);
        evhttp_send_reply(request, response.status, reason.c_str(), output);
        if (log_ != nullptr && !response.request_id.empty()) {
            *log_ << json_body({{"event", "request_complete"},
                                {"request_id", response.request_id},
                                {"route", route},
                                {"status", response.status},
                                {"queue_ms", response.queue_ms},
                                {"inference_ms", response.inference_ms},
                                {"total_ms", response.total_ms},
                                {"task_setup_ms", response.task_setup_ms},
                                {"task_prefill_ms", response.task_prefill_ms},
                                {"task_decode_ms", response.task_decode_ms},
                                {"response_bytes", response.body.size()}})
                  << std::flush;
        }
    }

    void handle(evhttp_request* request) {
        const char* raw_uri = evhttp_request_get_uri(request);
        std::string route = raw_uri == nullptr ? "/" : raw_uri;
        const auto query = route.find('?');
        if (query != std::string::npos)
            route.resize(query);
        const auto method = evhttp_request_get_command(request);
        if (method == EVHTTP_REQ_GET) {
            if (is_generation_route(route)) {
                send(request, route,
                     error_response(405, {}, "HTTP method is not supported", "method_not_allowed"));
                return;
            }
            send(request, route, service_.get(route));
            return;
        }
        if (method != EVHTTP_REQ_POST) {
            send(request, route,
                 error_response(405, {}, "HTTP method is not supported", "method_not_allowed"));
            return;
        }
        if (is_get_route(route)) {
            send(request, route,
                 error_response(405, {}, "HTTP method is not supported", "method_not_allowed"));
            return;
        }
        auto* input = evhttp_request_get_input_buffer(request);
        const auto length = evbuffer_get_length(input);
        std::string body(length, '\0');
        if (length != 0)
            evbuffer_copyout(input, body.data(), length);
        evhttp_request_own(request);
        service_.submit(route, body, [this, request, route](Response response) mutable {
            if (!defer(base_, [this, request, route, response = std::move(response)]() mutable {
                    send(request, route, std::move(response));
                })) {
                evhttp_request_free(request);
                event_base_loopbreak(base_);
            }
        });
    }

    Options options_;
    InferenceService service_;
    event_base* base_{nullptr};
    evhttp* http_{nullptr};
    evhttp_bound_socket* bound_{nullptr};
    event* sigint_{nullptr};
    event* sigterm_{nullptr};
    std::ostream* log_{nullptr};
};

HttpServer::HttpServer(ITask& task, Options options)
    : impl_(std::make_unique<Impl>(task, std::move(options))) {}
HttpServer::~HttpServer() = default;
void HttpServer::run(std::ostream& log) {
    impl_->run(log);
}
void HttpServer::request_stop() {
    impl_->request_stop();
}
std::uint16_t HttpServer::port() const noexcept {
    return impl_->port();
}

Options parse_args(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "help")
        return {};
    if (argc < 2)
        throw std::invalid_argument("a bundle path is required");
    Options options;
    options.bundle = argv[1];
    for (int index = 2; index < argc; ++index) {
        const std::string option = argv[index];
        if (index + 1 >= argc)
            throw std::invalid_argument(option + " requires a value");
        const std::string value = argv[++index];
        if (option == "--runtime-root")
            options.runtime_root = value;
        else if (option == "--model-name")
            options.model_name = value;
        else if (option == "--host")
            options.host = value;
        else if (option == "--port") {
            const auto port = parse_size(value, option.c_str());
            if (port > std::numeric_limits<std::uint16_t>::max())
                throw std::invalid_argument("--port is outside the supported range");
            options.port = static_cast<std::uint16_t>(port);
        } else if (option == "--queue-capacity")
            options.queue_capacity = parse_size(value, option.c_str());
        else if (option == "--max-queued-bytes")
            options.max_queued_bytes = parse_size(value, option.c_str());
        else if (option == "--max-body-bytes")
            options.max_body_bytes = parse_size(value, option.c_str());
        else if (option == "--max-prompt-bytes")
            options.max_prompt_bytes = parse_size(value, option.c_str());
        else if (option == "--max-new-tokens") {
            const auto limit = parse_size(value, option.c_str());
            if (limit > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
                throw std::invalid_argument("--max-new-tokens is outside the supported range");
            options.max_new_tokens = static_cast<std::int32_t>(limit);
        } else if (option == "--shutdown-grace-ms")
            options.shutdown_grace =
                std::chrono::milliseconds(parse_size(value, option.c_str(), 0));
        else
            throw std::invalid_argument("unknown option: " + option);
    }
    if (options.model_name.empty())
        throw std::invalid_argument("--model-name is required");
    return options;
}

void print_usage(std::ostream& output) {
    output << "Usage:\n"
              "  trtmc-server BUNDLE --model-name NAME [OPTIONS]\n\n"
              "Options:\n"
              "  --runtime-root DIR          Override the directory containing runtime DSOs\n"
              "  --host ADDRESS              Listen address (default: 127.0.0.1)\n"
              "  --port PORT                 Listen port (default: 8000)\n"
              "  --queue-capacity COUNT      Maximum waiting requests (default: 16)\n"
              "  --max-queued-bytes BYTES    Maximum queued body bytes (default: 1048576)\n"
              "  --max-body-bytes BYTES      Maximum HTTP body bytes (default: 1048576)\n"
              "  --max-prompt-bytes BYTES    Maximum prompt bytes (default: 262144)\n"
              "  --max-new-tokens COUNT      Per-request output-token ceiling (default: 4096)\n"
              "  --shutdown-grace-ms MS      Grace period for queued work (default: 30000)\n";
}

int run(int argc, char** argv, std::ostream& output, std::ostream& error) {
    try {
        if (argc == 2 && std::string(argv[1]) == "help") {
            print_usage(output);
            return 0;
        }
        auto options = parse_args(argc, argv);
        auto task = load_task(options.bundle, options.runtime_root);
        HttpServer server(*task, std::move(options));
        server.run(output);
        return 0;
    } catch (const std::exception& exception) {
        error << "trtmc-server: " << exception.what() << '\n';
        return 1;
    }
}

} // namespace trtmc::server
