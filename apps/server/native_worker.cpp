/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "server/native_worker.h"

#include "trtmc/task.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace trtmc::server {
namespace {

using Json = nlohmann::json;
constexpr std::size_t kMaxLineBytes = 16U * 1024U * 1024U;

class ProtocolError final : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

template <typename T>
void assign(const Json& config, const char* name, T& destination) {
    const auto value = config.find(name);
    if (value == config.end())
        return;
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
        if (!value->is_number_integer())
            throw ProtocolError(std::string("config.") + name + " must be an integer");
        bool in_range = false;
        if (value->is_number_unsigned()) {
            const auto number = value->get<Json::number_unsigned_t>();
            in_range =
                number <= static_cast<Json::number_unsigned_t>(std::numeric_limits<T>::max());
        } else {
            const auto number = value->get<Json::number_integer_t>();
            in_range =
                number >= static_cast<Json::number_integer_t>(std::numeric_limits<T>::min()) &&
                number <= static_cast<Json::number_integer_t>(std::numeric_limits<T>::max());
        }
        if (!in_range)
            throw ProtocolError(std::string("config.") + name + " is out of range");
    }
    destination = value->get<T>();
}

void validate_fields(const Json& object, std::initializer_list<const char*> allowed) {
    for (auto field = object.begin(); field != object.end(); ++field) {
        const bool found = std::any_of(allowed.begin(), allowed.end(),
                                       [&](const char* name) { return field.key() == name; });
        if (!found)
            throw ProtocolError("config." + field.key() + " is unsupported");
    }
}

TextGenerationConfig parse_config(const Json& request, std::int32_t default_tokens) {
    const auto found = request.find("config");
    const Json config = found == request.end() ? Json::object() : *found;
    if (!config.is_object())
        throw ProtocolError("config must be an object");
    validate_fields(config, {"max_new_tokens", "temperature", "top_k", "top_p", "min_p", "seed",
                             "system_prompt", "use_chat_template", "enable_thinking"});

    TextGenerationConfig result;
    result.max_new_tokens = default_tokens > 0 ? default_tokens : 128;
    try {
        assign(config, "max_new_tokens", result.max_new_tokens);
        assign(config, "temperature", result.temperature);
        assign(config, "top_k", result.top_k);
        assign(config, "top_p", result.top_p);
        assign(config, "min_p", result.min_p);
        assign(config, "seed", result.seed);
        assign(config, "system_prompt", result.system_prompt);
        assign(config, "use_chat_template", result.use_chat_template);
        assign(config, "enable_thinking", result.enable_thinking);
    } catch (const nlohmann::json::exception&) {
        throw ProtocolError("generation config contains an invalid value");
    }
    if (result.max_new_tokens <= 0)
        throw ProtocolError("config.max_new_tokens must be positive");
    if (result.top_k < 0)
        throw ProtocolError("config.top_k must be non-negative");
    if (!std::isfinite(result.temperature) || result.temperature < 0.0F)
        throw ProtocolError("config.temperature must be finite and non-negative");
    if (!std::isfinite(result.top_p) || result.top_p < 0.0F || result.top_p > 1.0F)
        throw ProtocolError("config.top_p must be in [0, 1]");
    if (!std::isfinite(result.min_p) || result.min_p < 0.0F || result.min_p > 1.0F)
        throw ProtocolError("config.min_p must be in [0, 1]");
    return result;
}

std::string string_field(const Json& request, const char* name, bool empty_ok = false) {
    const auto value = request.find(name);
    if (value == request.end() || !value->is_string())
        throw ProtocolError(std::string(name) + " must be a string");
    const auto result = value->get<std::string>();
    if (!empty_ok && result.empty())
        throw ProtocolError(std::string(name) + " must not be empty");
    return result;
}

Json error_response(const Json& id, const char* type, const std::string& message) {
    return {{"id", id}, {"ok", false}, {"error", {{"type", type}, {"message", message}}}};
}

bool write_json(std::ostream& output, const Json& value) {
    output << value.dump(-1, ' ', false, Json::error_handler_t::replace) << '\n';
    output.flush();
    return static_cast<bool>(output);
}

Json request_id(const Json& request) {
    if (!request.is_object())
        return nullptr;
    const auto id = request.find("id");
    if (id == request.end() || !id->is_string() || id->get_ref<const std::string&>().empty())
        return nullptr;
    return *id;
}

} // namespace

int run_text_worker(ITask& task, std::istream& input, std::ostream& output) {
    auto* text = dynamic_cast<ITextGeneration*>(&task);
    if (text == nullptr)
        throw std::invalid_argument("bundle task does not implement text generation");

    if (!write_json(output, {{"event", "ready"},
                             {"protocol_version", 1},
                             {"capabilities", Json::array({ITextGeneration::kTask})},
                             {"default_max_new_tokens", text->default_max_new_tokens()}}))
        return 2;

    std::vector<char> buffer(kMaxLineBytes + 2U);
    while (input.good()) {
        input.getline(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto extracted = input.gcount();
        if (input.eof() && extracted == 0)
            return 0;

        Json id = nullptr;
        bool shutdown = false;
        bool fatal = false;
        Json response;
        try {
            if (input.fail() && !input.eof()) {
                input.clear(input.rdstate() & ~std::ios::failbit);
                input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                throw ProtocolError("request exceeds the 16 MiB JSONL limit");
            }
            std::size_t size = static_cast<std::size_t>(extracted);
            if (!input.eof() && size > 0)
                --size;
            if (size == 0)
                continue;
            Json request;
            try {
                request = Json::parse(std::string_view(buffer.data(), size));
            } catch (const nlohmann::json::parse_error&) {
                throw ProtocolError("request is not valid JSON");
            }
            id = request_id(request);
            if (id.is_null())
                throw ProtocolError("id must be a non-empty string");
            const auto operation = string_field(request, "op");
            if (operation == "shutdown") {
                response = {{"id", id}, {"ok", true}, {"result", {{"status", "shutting_down"}}}};
                shutdown = true;
            } else if (operation == "generate") {
                const auto prompt = string_field(request, "prompt", true);
                const auto result =
                    text->generate(prompt, parse_config(request, text->default_max_new_tokens()));
                response = {{"id", id},
                            {"ok", true},
                            {"result",
                             {{"text", result.text},
                              {"completion_tokens", result.token_ids.size()},
                              {"setup_ms", result.setup_ms},
                              {"prefill_ms", result.prefill_ms},
                              {"decode_ms", result.decode_ms}}}};
            } else {
                throw ProtocolError("unknown operation: " + operation);
            }
        } catch (const ProtocolError& error) {
            response = error_response(id, "invalid_request_error", error.what());
        } catch (const std::exception& error) {
            std::cerr << "[trtmc.server.worker] " << error.what() << '\n';
            response = error_response(id, "runtime_error", "native worker operation failed");
            fatal = true;
        } catch (...) {
            std::cerr << "[trtmc.server.worker] unknown native worker error\n";
            response = error_response(id, "runtime_error", "native worker operation failed");
            fatal = true;
        }
        if (!write_json(output, response))
            return 2;
        if (fatal)
            return 1;
        if (shutdown)
            return 0;
    }
    return input.bad() ? 2 : 0;
}

} // namespace trtmc::server
