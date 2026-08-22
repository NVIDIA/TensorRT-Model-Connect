/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "serve/realtime_worker.h"

#include "trtmc/speech_session.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <initializer_list>
#include <istream>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace trtmc::serve {
namespace {

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

constexpr int32_t kWireSampleRate = 24000;
constexpr std::size_t kMaxEventIdBytes = 128U;

struct ProtocolError final : std::exception {
    ProtocolError(const char* error_code, const char* safe_message)
        : code(error_code), message(safe_message) {}

    const char* what() const noexcept override { return message; }

    const char* code;
    const char* message;
};

enum class InputKind { kLine, kLineTooLong, kReadError, kEof };

struct InputItem {
    InputKind kind{InputKind::kEof};
    std::string line;
};

class BoundedCommandQueue {
  public:
    BoundedCommandQueue(std::size_t max_items, std::size_t max_bytes)
        : max_items_(max_items), max_bytes_(max_bytes) {}

    void push(InputItem item) {
        const std::size_t bytes = item.line.size();
        std::unique_lock<std::mutex> lock(mutex_);
        writable_.wait(lock, [&] {
            return items_.size() < max_items_ && bytes <= max_bytes_ - queued_bytes_;
        });
        queued_bytes_ += bytes;
        items_.push_back(std::move(item));
        lock.unlock();
        readable_.notify_one();
    }

    std::optional<InputItem> pop_for(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!readable_.wait_for(lock, timeout, [&] { return !items_.empty(); }))
            return std::nullopt;
        InputItem item = std::move(items_.front());
        items_.pop_front();
        queued_bytes_ -= item.line.size();
        lock.unlock();
        writable_.notify_one();
        return item;
    }

  private:
    std::size_t max_items_{0};
    std::size_t max_bytes_{0};
    std::size_t queued_bytes_{0};
    std::mutex mutex_;
    std::condition_variable readable_;
    std::condition_variable writable_;
    std::deque<InputItem> items_;
};

struct LineReadResult {
    InputKind kind{InputKind::kEof};
    std::string line;
};

LineReadResult finalize_line(LineReadResult result, bool too_long) {
    if (!too_long && !result.line.empty() && result.line.back() == '\r')
        result.line.pop_back();
    result.kind = too_long ? InputKind::kLineTooLong : InputKind::kLine;
    return result;
}

LineReadResult finalize_eof(std::istream& input, LineReadResult result, bool too_long) {
    if (input.bad())
        return {InputKind::kReadError, {}};
    if (result.line.empty() && !too_long)
        return {InputKind::kEof, {}};
    return finalize_line(std::move(result), too_long);
}

void append_line_character(LineReadResult& result, int value, std::size_t max_bytes,
                           bool& too_long) {
    if (result.line.size() < max_bytes)
        result.line.push_back(static_cast<char>(value));
    else
        too_long = true;
}

LineReadResult read_bounded_line(std::istream& input, std::size_t max_bytes) {
    LineReadResult result;
    result.kind = InputKind::kLine;
    result.line.reserve(std::min<std::size_t>(max_bytes, 4096U));
    bool too_long = false;
    while (true) {
        const int value = input.get();
        if (value == std::char_traits<char>::eof())
            return finalize_eof(input, std::move(result), too_long);
        if (value == '\n')
            return finalize_line(std::move(result), too_long);
        append_line_character(result, value, max_bytes, too_long);
    }
}

bool is_close_line(const std::string& line) {
    try {
        const auto value = json::parse(line);
        return value.is_object() && value.contains("type") && value.at("type").is_string() &&
               value.at("type") == "session.close";
    } catch (const json::exception&) {
        return false;
    }
}

void read_commands(std::istream& input, BoundedCommandQueue& queue, std::size_t max_line_bytes) {
    try {
        while (true) {
            auto item = read_bounded_line(input, max_line_bytes);
            const bool close = item.kind == InputKind::kLine && is_close_line(item.line);
            const bool stop = item.kind == InputKind::kEof || item.kind == InputKind::kReadError;
            queue.push({item.kind, std::move(item.line)});
            if (stop)
                return;
            if (close) {
                queue.push({InputKind::kEof, {}});
                return;
            }
        }
    } catch (...) {
        queue.push({InputKind::kReadError, {}});
    }
}

enum class WriteResult { kOk, kLineTooLong, kStreamError };

class JsonlWriter {
  public:
    JsonlWriter(std::ostream& output, std::size_t max_line_bytes)
        : output_(output), max_line_bytes_(max_line_bytes) {}

    WriteResult write(const json& value) {
        std::string line;
        try {
            line = value.dump();
        } catch (const json::exception&) {
            return WriteResult::kLineTooLong;
        }
        if (line.size() > max_line_bytes_)
            return WriteResult::kLineTooLong;
        output_ << line << '\n';
        output_.flush();
        return output_ ? WriteResult::kOk : WriteResult::kStreamError;
    }

  private:
    std::ostream& output_;
    std::size_t max_line_bytes_{0};
};

bool is_safe_event_id(const std::string& value) {
    return value.size() <= kMaxEventIdBytes &&
           std::all_of(value.begin(), value.end(), [](unsigned char character) {
               return character >= 0x20U && character <= 0x7eU;
           });
}

std::optional<std::string> event_id_from(const json& value) {
    if (!value.contains("event_id"))
        return std::nullopt;
    if (!value.at("event_id").is_string())
        throw ProtocolError("invalid_event_id", "event_id must be a short printable string");
    auto event_id = value.at("event_id").get<std::string>();
    if (!is_safe_event_id(event_id))
        throw ProtocolError("invalid_event_id", "event_id must be a short printable string");
    return event_id;
}

bool contains_only(const json& value, std::initializer_list<std::string_view> allowed) {
    for (const auto& item : value.items()) {
        const bool found = std::find(allowed.begin(), allowed.end(), item.key()) != allowed.end();
        if (!found)
            return false;
    }
    return true;
}

void require_fields(const json& value, std::initializer_list<std::string_view> allowed) {
    if (!contains_only(value, allowed))
        throw ProtocolError("invalid_message", "message contains an unsupported field");
}

const std::string& require_string(const json& value, const char* field) {
    if (!value.contains(field) || !value.at(field).is_string())
        throw ProtocolError("invalid_message", "message contains an invalid string field");
    return value.at(field).get_ref<const std::string&>();
}

std::uint64_t require_uint64(const json& value, const char* field) {
    if (!value.contains(field) || !value.at(field).is_number_unsigned())
        throw ProtocolError("invalid_message", "message contains an invalid integer field");
    return value.at(field).get<std::uint64_t>();
}

std::int64_t require_nonnegative_int64(const json& value, const char* field) {
    if (!value.contains(field) || !value.at(field).is_number_integer())
        throw ProtocolError("invalid_message", "message contains an invalid integer field");
    if (value.at(field).is_number_unsigned()) {
        const auto number = value.at(field).get<std::uint64_t>();
        if (number > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
            throw ProtocolError("invalid_message", "integer field is out of range");
        return static_cast<std::int64_t>(number);
    }
    const auto number = value.at(field).get<std::int64_t>();
    if (number < 0)
        throw ProtocolError("invalid_message", "integer field must be non-negative");
    return number;
}

int base64_digit(unsigned char value) {
    if (value >= 'A' && value <= 'Z')
        return value - 'A';
    if (value >= 'a' && value <= 'z')
        return value - 'a' + 26;
    if (value >= '0' && value <= '9')
        return value - '0' + 52;
    if (value == '+')
        return 62;
    if (value == '/')
        return 63;
    return -1;
}

bool valid_base64_tail(const std::string& text, std::size_t offset, int third, int fourth) {
    const bool last = offset + 4U == text.size();
    if (third < 0)
        return last && text[offset + 2U] == '=' && text[offset + 3U] == '=';
    if (fourth < 0)
        return last && text[offset + 3U] == '=';
    return true;
}

struct Base64Quartet {
    int first{-1};
    int second{-1};
    int third{-1};
    int fourth{-1};
};

Base64Quartet read_base64_quartet(const std::string& text, std::size_t offset) {
    return {base64_digit(static_cast<unsigned char>(text[offset])),
            base64_digit(static_cast<unsigned char>(text[offset + 1U])),
            base64_digit(static_cast<unsigned char>(text[offset + 2U])),
            base64_digit(static_cast<unsigned char>(text[offset + 3U]))};
}

bool has_canonical_base64_bits(const Base64Quartet& quartet) {
    if (quartet.third < 0)
        return (quartet.second & 0x0f) == 0;
    if (quartet.fourth < 0)
        return (quartet.third & 0x03) == 0;
    return true;
}

void validate_base64_quartet(const std::string& text, std::size_t offset,
                             const Base64Quartet& quartet) {
    if (quartet.first < 0 || quartet.second < 0 ||
        !valid_base64_tail(text, offset, quartet.third, quartet.fourth))
        throw ProtocolError("invalid_audio", "audio must be strict base64 PCM16LE");
    if (!has_canonical_base64_bits(quartet))
        throw ProtocolError("invalid_audio", "audio must use canonical base64 padding");
}

void append_base64_quartet(const Base64Quartet& quartet, std::vector<std::uint8_t>& decoded) {
    decoded.push_back(static_cast<std::uint8_t>((quartet.first << 2) | (quartet.second >> 4)));
    if (quartet.third >= 0)
        decoded.push_back(static_cast<std::uint8_t>((quartet.second << 4) | (quartet.third >> 2)));
    if (quartet.fourth >= 0)
        decoded.push_back(static_cast<std::uint8_t>((quartet.third << 6) | quartet.fourth));
}

void validate_base64_size(const std::string& text, std::size_t max_bytes) {
    if (text.empty() || text.size() % 4U != 0U)
        throw ProtocolError("invalid_audio", "audio must be strict base64 PCM16LE");
    const std::size_t max_encoded = 4U * ((max_bytes + 2U) / 3U);
    if (text.size() > max_encoded)
        throw ProtocolError("audio_too_large", "audio chunk exceeds the configured limit");
}

std::vector<std::uint8_t> decode_base64(const std::string& text, std::size_t max_bytes) {
    validate_base64_size(text, max_bytes);
    std::vector<std::uint8_t> decoded;
    decoded.reserve(std::min(max_bytes, text.size() / 4U * 3U));
    for (std::size_t offset = 0; offset < text.size(); offset += 4U) {
        const auto quartet = read_base64_quartet(text, offset);
        validate_base64_quartet(text, offset, quartet);
        append_base64_quartet(quartet, decoded);
    }
    if (decoded.size() > max_bytes)
        throw ProtocolError("audio_too_large", "audio chunk exceeds the configured limit");
    return decoded;
}

std::string encode_base64(const std::vector<std::uint8_t>& bytes) {
    constexpr std::string_view alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    result.reserve(4U * ((bytes.size() + 2U) / 3U));
    for (std::size_t offset = 0; offset < bytes.size(); offset += 3U) {
        const std::uint32_t first = bytes[offset];
        const std::uint32_t second = offset + 1U < bytes.size() ? bytes[offset + 1U] : 0U;
        const std::uint32_t third = offset + 2U < bytes.size() ? bytes[offset + 2U] : 0U;
        const std::uint32_t word = (first << 16U) | (second << 8U) | third;
        result.push_back(alphabet[(word >> 18U) & 0x3fU]);
        result.push_back(alphabet[(word >> 12U) & 0x3fU]);
        result.push_back(offset + 1U < bytes.size() ? alphabet[(word >> 6U) & 0x3fU] : '=');
        result.push_back(offset + 2U < bytes.size() ? alphabet[word & 0x3fU] : '=');
    }
    return result;
}

std::vector<float> pcm16_to_float(const std::vector<std::uint8_t>& bytes) {
    if (bytes.empty() || bytes.size() % 2U != 0U)
        throw ProtocolError("invalid_audio", "audio must contain complete PCM16LE samples");
    std::vector<float> samples(bytes.size() / 2U);
    for (std::size_t index = 0; index < samples.size(); ++index) {
        int32_t value = static_cast<int32_t>(bytes[index * 2U]) |
                        (static_cast<int32_t>(bytes[index * 2U + 1U]) << 8);
        if (value >= 32768)
            value -= 65536;
        samples[index] = static_cast<float>(value) / 32768.0F;
    }
    return samples;
}

int32_t float_to_pcm16(float value) {
    if (value <= -1.0F)
        return -32768;
    if (value >= 1.0F)
        return 32767;
    return static_cast<int32_t>(std::lrint(static_cast<double>(value) * 32768.0));
}

std::string float_to_pcm16_base64(const std::vector<float>& samples) {
    std::vector<std::uint8_t> bytes;
    bytes.reserve(samples.size() * 2U);
    for (const float sample : samples) {
        if (!std::isfinite(sample))
            throw ProtocolError("invalid_event", "native audio event contains an invalid sample");
        const int32_t value = float_to_pcm16(sample);
        const std::uint16_t bits = static_cast<std::uint16_t>(value & 0xffff);
        bytes.push_back(static_cast<std::uint8_t>(bits & 0xffU));
        bytes.push_back(static_cast<std::uint8_t>((bits >> 8U) & 0xffU));
    }
    return encode_base64(bytes);
}

const char* speech_event_name(SpeechSessionEventKind kind) {
    constexpr std::array<const char*, 15> names = {
        "agent_audio",
        "agent_text",
        "user_transcript",
        "turn_started",
        "turn_finished",
        "yielded",
        "cancelled",
        "reset",
        "error",
        "input_finished",
        "user_speech_started",
        "user_speech_stopped",
        "function_call",
        "function_call_started",
        "function_response_finished",
    };
    const auto index = static_cast<std::size_t>(kind);
    return index < names.size() ? names[index] : nullptr;
}

struct WorkerStats {
    std::uint64_t commands{0};
    std::uint64_t audio_samples{0};
    std::uint64_t events{0};
    std::uint64_t errors{0};
    std::uint64_t stale_events{0};
};

struct DrainResult {
    bool input_finished{false};
    bool cancelled{false};
    bool failed{false};
};

class RealtimeWorker {
  public:
    RealtimeWorker(IPipeline& pipeline, std::istream& input, std::ostream& output,
                   RealtimeWorkerLimits limits)
        : pipeline_(pipeline), input_(input), writer_(output, limits.max_line_bytes),
          limits_(std::move(limits)),
          command_queue_(limits_.max_queued_commands, limits_.max_queued_bytes) {
        validate_limits();
    }

    int run() {
        emit_ready();
        std::thread reader(
            [this] { read_commands(input_, command_queue_, limits_.max_line_bytes); });
        const int result = command_loop();
        reader.join();
        emit_end();
        return output_failed_ ? 1 : result;
    }

  private:
    static bool valid_transport_limits(const RealtimeWorkerLimits& limits) {
        if (limits.max_line_bytes < 256U || limits.max_audio_bytes == 0U)
            return false;
        if (limits.max_audio_bytes % 2U != 0U || limits.max_session_audio_samples == 0U)
            return false;
        if (limits.max_session_config_bytes == 0U || limits.max_queued_commands == 0U)
            return false;
        return limits.max_queued_bytes >= limits.max_line_bytes;
    }

    static bool valid_event_limits(const RealtimeWorkerLimits& limits) {
        if (limits.max_native_events_per_drain == 0U ||
            limits.max_native_audio_samples_per_drain == 0U)
            return false;
        return limits.max_event_text_bytes != 0U && limits.max_event_audio_samples != 0U;
    }

    static bool valid_timing_limits(const RealtimeWorkerLimits& limits) {
        return limits.close_drain_timeout_ms >= 0 && limits.idle_poll_ms > 0;
    }

    void validate_limits() const {
        if (!valid_transport_limits(limits_) || !valid_event_limits(limits_) ||
            !valid_timing_limits(limits_))
            throw std::invalid_argument("invalid realtime worker limits");
    }

    void emit_ready() {
        json value = {{"type", "session.ready"},
                      {"input_audio_format", "pcm16"},
                      {"input_sample_rate", kWireSampleRate},
                      {"output_sample_rate", kWireSampleRate}};
        write(value);
    }

    void emit_end() {
        json value = {{"type", "session.end"},
                      {"reason", end_reason_},
                      {"stats",
                       {{"commands", stats_.commands},
                        {"audio_samples", stats_.audio_samples},
                        {"events", stats_.events},
                        {"errors", stats_.errors},
                        {"stale_events", stats_.stale_events}}}};
        if (close_event_id_)
            value["event_id"] = *close_event_id_;
        write(value);
    }

    void write(const json& value) {
        const auto result = writer_.write(value);
        if (result == WriteResult::kStreamError)
            output_failed_ = true;
    }

    void emit_error(const char* code, const char* message,
                    const std::optional<std::string>& event_id = std::nullopt) {
        ++stats_.errors;
        json value = {{"type", "error"}, {"code", code}, {"message", message}};
        if (event_id)
            value["event_id"] = *event_id;
        write(value);
    }

    int command_loop() {
        const auto poll = std::chrono::milliseconds(limits_.idle_poll_ms);
        while (!close_requested_) {
            auto item = command_queue_.pop_for(poll);
            if (item && handle_input_item(*item))
                return input_failed_ ? 1 : 0;
            drain_available_events();
        }
        return 0;
    }

    bool handle_input_item(const InputItem& item) {
        if (item.kind == InputKind::kLineTooLong) {
            emit_error("line_too_large", "JSONL message exceeds the configured limit");
            return false;
        }
        if (item.kind == InputKind::kReadError) {
            emit_error("input_error", "JSONL input could not be read");
            input_failed_ = true;
            end_reason_ = "input_error";
            cancel_session();
            return true;
        }
        if (item.kind == InputKind::kEof) {
            end_reason_ = "eof";
            cancel_session();
            return true;
        }
        ++stats_.commands;
        handle_line(item.line);
        return close_requested_;
    }

    void handle_line(const std::string& line) {
        std::optional<std::string> event_id;
        try {
            const auto value = json::parse(line);
            if (!value.is_object())
                throw ProtocolError("invalid_message", "JSONL message must be one object");
            event_id = event_id_from(value);
            dispatch(value, event_id);
        } catch (const json::parse_error&) {
            emit_error("invalid_json", "JSONL message is not valid JSON");
        } catch (const ProtocolError& error) {
            emit_error(error.code, error.message, event_id);
        } catch (...) {
            emit_error("session_error", "speech session operation failed", event_id);
        }
    }

    void dispatch(const json& value, const std::optional<std::string>& event_id) {
        const auto& type = require_string(value, "type");
        if (type == "session.update")
            return update_session(value, event_id);
        if (type == "input_audio_buffer.append")
            return append_audio(value, event_id);
        if (type == "input_audio_buffer.commit")
            return commit_input(value, event_id);
        if (type == "input_audio_buffer.clear")
            return clear_input(value, event_id);
        if (type == "response.create")
            return create_response(value, event_id);
        if (type == "response.cancel")
            return cancel_response(value, event_id);
        if (type == "conversation.item.truncate")
            return truncate_response(value, event_id);
        if (type == "conversation.item.create")
            return submit_tool_output(value, event_id);
        if (type == "session.close")
            return close_session(value, event_id);
        throw ProtocolError("unknown_event", "JSONL message type is not supported");
    }

    void update_session(const json& value, const std::optional<std::string>& event_id) {
        require_fields(value, {"type", "event_id", "input_sample_rate", "output_sample_rate",
                               "instructions", "tools", "on_hold_messages"});
        if (first_audio_received_)
            throw ProtocolError("invalid_state", "session cannot be updated after audio starts");
        validate_wire_rates(value);
        const auto instructions = optional_instructions(value);
        const json tools = optional_tools(value);
        const json on_hold = optional_on_hold_messages(value);
        validate_session_config_size(instructions, tools, on_hold);
        if (!tools.empty() && !dynamic_cast<ISpeechToolSessionProvider*>(&pipeline_))
            throw ProtocolError("unsupported", "configured tools are not supported");
        if (tools.empty() && !dynamic_cast<ISpeechSessionProvider*>(&pipeline_))
            throw ProtocolError("unsupported", "persistent speech sessions are not supported");
        recreate_session(instructions, tools, on_hold);
        emit_session_updated(event_id, !tools.empty());
    }

    void validate_wire_rates(const json& value) const {
        const auto input_rate = require_nonnegative_int64(value, "input_sample_rate");
        const auto output_rate = require_nonnegative_int64(value, "output_sample_rate");
        if (input_rate != kWireSampleRate || output_rate != kWireSampleRate)
            throw ProtocolError("unsupported", "realtime audio must use 24 kHz PCM16 mono");
    }

    std::string optional_instructions(const json& value) const {
        if (!value.contains("instructions"))
            return {};
        if (!value.at("instructions").is_string())
            throw ProtocolError("invalid_message", "instructions must be a string");
        return value.at("instructions").get<std::string>();
    }

    json optional_tools(const json& value) const {
        if (!value.contains("tools"))
            return json::array();
        if (!value.at("tools").is_array())
            throw ProtocolError("invalid_message", "tools must be an array");
        return value.at("tools");
    }

    json optional_on_hold_messages(const json& value) const {
        if (!value.contains("on_hold_messages"))
            return json::object();
        if (!value.at("on_hold_messages").is_object())
            throw ProtocolError("invalid_message", "on_hold_messages must be an object");
        return value.at("on_hold_messages");
    }

    void validate_session_config_size(const std::string& instructions, const json& tools,
                                      const json& on_hold) const {
        const std::size_t tools_bytes = tools.dump().size();
        const std::size_t on_hold_bytes = on_hold.dump().size();
        const auto available_after_instructions =
            limits_.max_session_config_bytes -
            std::min(instructions.size(), limits_.max_session_config_bytes);
        const auto available_after_tools =
            available_after_instructions - std::min(tools_bytes, available_after_instructions);
        if (instructions.size() > limits_.max_session_config_bytes ||
            tools_bytes > available_after_instructions || on_hold_bytes > available_after_tools)
            throw ProtocolError("session_too_large", "session configuration exceeds its limit");
        if (tools.empty() && !on_hold.empty())
            throw ProtocolError("invalid_message", "on_hold_messages require configured tools");
    }

    void recreate_session(const std::string& instructions, const json& tools, const json& on_hold) {
        if (session_)
            safe_cancel(*session_);
        session_.reset();
        SpeechSessionConfig config;
        config.input_sample_rate = kWireSampleRate;
        config.output_sample_rate = kWireSampleRate;
        config.system_prompt = instructions;
        config.emit_agent_audio = true;
        config.emit_agent_text = true;
        config.emit_user_transcript = true;
        config.enable_barge_in = true;
        try {
            session_ = make_session(config, tools, on_hold);
        } catch (...) {
            throw ProtocolError("session_rejected", "session configuration was rejected");
        }
        if (!session_)
            throw ProtocolError("session_unavailable", "speech session is unavailable");
        have_epoch_ = false;
        current_epoch_ = 0;
        current_sequence_ = 0;
        tools_enabled_ = !tools.empty();
        tool_response_active_ = false;
    }

    std::unique_ptr<ISpeechSession> make_session(const SpeechSessionConfig& config,
                                                 const json& tools, const json& on_hold) {
        if (!tools.empty()) {
            auto* provider = dynamic_cast<ISpeechToolSessionProvider*>(&pipeline_);
            if (!provider)
                throw ProtocolError("unsupported", "configured tools are not supported");
            SpeechToolSessionConfig tool_config;
            tool_config.tools_json = tools.dump();
            tool_config.on_hold_messages_json = on_hold.empty() ? std::string{} : on_hold.dump();
            return provider->create_tool_speech_session(config, tool_config);
        }
        auto* provider = dynamic_cast<ISpeechSessionProvider*>(&pipeline_);
        if (!provider)
            throw ProtocolError("unsupported", "persistent speech sessions are not supported");
        return provider->create_speech_session(config);
    }

    void emit_session_updated(const std::optional<std::string>& event_id, bool tools_enabled) {
        json value = {
            {"type", "session.updated"},
            {"input_sample_rate", kWireSampleRate},
            {"output_sample_rate", kWireSampleRate},
            {"capabilities",
             {{"realtime_controls",
               dynamic_cast<ISpeechRealtimeControl*>(session_.get()) != nullptr},
              {"function_call_output",
               tools_enabled && dynamic_cast<ISpeechToolSession*>(session_.get()) != nullptr}}}};
        if (event_id)
            value["event_id"] = *event_id;
        write(value);
    }

    void append_audio(const json& value, const std::optional<std::string>&) {
        require_fields(value, {"type", "event_id", "audio"});
        require_session();
        const auto& encoded = require_string(value, "audio");
        const auto bytes = decode_base64(encoded, limits_.max_audio_bytes);
        auto samples = pcm16_to_float(bytes);
        if (samples.size() >
            limits_.max_session_audio_samples -
                std::min(session_audio_samples_, limits_.max_session_audio_samples))
            throw ProtocolError("session_too_large", "session audio exceeds its limit");
        try {
            session_->append_audio(samples.data(), static_cast<int32_t>(samples.size()));
        } catch (...) {
            throw ProtocolError("session_error", "speech session rejected an audio chunk");
        }
        session_audio_samples_ += samples.size();
        stats_.audio_samples += samples.size();
        first_audio_received_ = true;
    }

    void commit_input(const json& value, const std::optional<std::string>& event_id) {
        require_fields(value, {"type", "event_id"});
        auto* control = require_realtime_control();
        try {
            control->commit_input_turn(false);
        } catch (...) {
            throw ProtocolError("session_error", "input commit failed");
        }
        emit_ack("input_audio_buffer.committed", event_id);
    }

    void clear_input(const json& value, const std::optional<std::string>& event_id) {
        require_fields(value, {"type", "event_id"});
        auto* control = require_realtime_control();
        try {
            control->clear_pending_input();
        } catch (...) {
            throw ProtocolError("session_error", "input clear failed");
        }
        emit_ack("input_audio_buffer.cleared", event_id);
    }

    void create_response(const json& value, const std::optional<std::string>&) {
        require_fields(value, {"type", "event_id"});
        if (tool_response_active_)
            throw ProtocolError("invalid_state", "function response is already active");
        auto* control = require_realtime_control();
        try {
            control->create_response();
        } catch (...) {
            throw ProtocolError("session_error", "response creation failed");
        }
    }

    void cancel_response(const json& value, const std::optional<std::string>& event_id) {
        require_fields(value, {"type", "event_id"});
        auto* control = require_realtime_control();
        try {
            control->cancel_response();
        } catch (...) {
            throw ProtocolError("session_error", "response cancellation failed");
        }
        emit_ack("response.cancelled", event_id);
    }

    void truncate_response(const json& value, const std::optional<std::string>& event_id) {
        require_fields(value, {"type", "event_id", "epoch", "played_output_samples"});
        auto* control = require_realtime_control();
        const auto epoch = require_uint64(value, "epoch");
        require_current_epoch(epoch);
        const auto samples = require_nonnegative_int64(value, "played_output_samples");
        try {
            control->truncate_response(epoch, samples);
        } catch (...) {
            throw ProtocolError("session_error", "response truncation failed");
        }
        json acknowledgement = {{"type", "conversation.item.truncated"},
                                {"epoch", epoch},
                                {"played_output_samples", samples}};
        if (event_id)
            acknowledgement["event_id"] = *event_id;
        write(acknowledgement);
    }

    void emit_ack(const char* type, const std::optional<std::string>& event_id) {
        json acknowledgement = {{"type", type}};
        if (event_id)
            acknowledgement["event_id"] = *event_id;
        write(acknowledgement);
    }

    void submit_tool_output(const json& value, const std::optional<std::string>&) {
        require_fields(value, {"type", "event_id", "epoch", "call_id", "output"});
        require_session();
        if (!tools_enabled_)
            throw ProtocolError("unsupported", "function call output is not enabled");
        auto* tool_session = dynamic_cast<ISpeechToolSession*>(session_.get());
        if (!tool_session)
            throw ProtocolError("unsupported", "function call output is not supported");
        const auto epoch = require_uint64(value, "epoch");
        require_current_epoch(epoch);
        const auto& call_id = require_string(value, "call_id");
        const auto& output = require_string(value, "output");
        if (!is_safe_event_id(call_id) || output.size() > limits_.max_session_config_bytes)
            throw ProtocolError("invalid_message", "function call output exceeds its limit");
        try {
            tool_session->submit_tool_response(epoch, call_id, output);
        } catch (...) {
            throw ProtocolError("session_error", "function call output was rejected");
        }
        tool_response_active_ = true;
    }

    void close_session(const json& value, const std::optional<std::string>& event_id) {
        require_fields(value, {"type", "event_id", "mode"});
        std::string mode = "finish";
        if (value.contains("mode"))
            mode = require_string(value, "mode");
        if (mode != "finish" && mode != "cancel")
            throw ProtocolError("invalid_message", "session close mode must be finish or cancel");
        close_event_id_ = event_id;
        end_reason_ = mode == "finish" ? "closed" : "cancelled";
        if (mode == "finish")
            finish_session();
        else
            cancel_session();
        close_requested_ = true;
    }

    void finish_session() {
        if (!session_)
            return;
        try {
            session_->finish_input();
        } catch (...) {
            emit_error("session_error", "speech session could not finish");
            cancel_session();
            return;
        }
        const auto deadline =
            Clock::now() + std::chrono::milliseconds(limits_.close_drain_timeout_ms);
        while (Clock::now() <= deadline) {
            const auto drained = wait_and_emit_events();
            if (drained.input_finished || drained.cancelled || drained.failed)
                return;
        }
        emit_error("close_timeout", "speech session did not finish within its limit");
        cancel_session();
    }

    DrainResult wait_and_emit_events() {
        try {
            auto events = session_->wait_events(limits_.idle_poll_ms);
            return emit_native_events(std::move(events));
        } catch (...) {
            emit_error("session_error", "speech session event polling failed");
            DrainResult result;
            result.failed = true;
            return result;
        }
    }

    void cancel_session() {
        if (!session_)
            return;
        safe_cancel(*session_);
        drain_available_events();
    }

    static void safe_cancel(ISpeechSession& session) noexcept {
        try {
            session.cancel();
        } catch (...) {
        }
    }

    ISpeechRealtimeControl* require_realtime_control() {
        require_session();
        auto* control = dynamic_cast<ISpeechRealtimeControl*>(session_.get());
        if (!control)
            throw ProtocolError("unsupported", "realtime control is not supported");
        return control;
    }

    void require_session() const {
        if (!session_)
            throw ProtocolError("invalid_state", "session.update is required before this event");
    }

    void require_current_epoch(std::uint64_t epoch) const {
        if (have_epoch_ && epoch != current_epoch_)
            throw ProtocolError("stale_epoch", "event epoch is no longer current");
    }

    void drain_available_events() {
        if (!session_)
            return;
        try {
            emit_native_events(session_->take_events());
        } catch (...) {
            emit_error("session_error", "speech session event polling failed");
        }
    }

    DrainResult emit_native_events(std::vector<SpeechSessionEvent> events) {
        DrainResult result;
        const std::size_t audio_samples = native_audio_sample_count(events);
        if (events.size() > limits_.max_native_events_per_drain ||
            audio_samples > limits_.max_native_audio_samples_per_drain) {
            emit_error("event_queue_limit", "native event batch exceeds its limit");
            if (session_)
                safe_cancel(*session_);
            result.failed = true;
            return result;
        }
        for (const auto& event : events) {
            if (event_is_stale(event)) {
                ++stats_.stale_events;
                continue;
            }
            update_event_order(event);
            update_tool_response_state(event.kind);
            emit_native_event(event);
            result.input_finished |= event.kind == SpeechSessionEventKind::kInputFinished;
            result.cancelled |= event.kind == SpeechSessionEventKind::kCancelled;
            result.failed |= event.kind == SpeechSessionEventKind::kError;
        }
        return result;
    }

    void update_tool_response_state(SpeechSessionEventKind kind) {
        if (kind == SpeechSessionEventKind::kFunctionResponseFinished ||
            kind == SpeechSessionEventKind::kCancelled || kind == SpeechSessionEventKind::kReset ||
            kind == SpeechSessionEventKind::kYielded || kind == SpeechSessionEventKind::kError)
            tool_response_active_ = false;
    }

    static std::size_t native_audio_sample_count(const std::vector<SpeechSessionEvent>& events) {
        std::size_t samples = 0;
        for (const auto& event : events) {
            if (event.audio_samples.size() > std::numeric_limits<std::size_t>::max() - samples)
                return std::numeric_limits<std::size_t>::max();
            samples += event.audio_samples.size();
        }
        return samples;
    }

    bool event_is_stale(const SpeechSessionEvent& event) const {
        if (!have_epoch_)
            return false;
        return event.epoch < current_epoch_ ||
               (event.epoch == current_epoch_ && event.sequence <= current_sequence_);
    }

    void update_event_order(const SpeechSessionEvent& event) {
        if (!have_epoch_ || event.epoch > current_epoch_) {
            have_epoch_ = true;
            current_epoch_ = event.epoch;
            current_sequence_ = event.sequence;
            return;
        }
        current_sequence_ = event.sequence;
    }

    void emit_native_event(const SpeechSessionEvent& event) {
        const char* kind = speech_event_name(event.kind);
        if (!kind) {
            emit_error("invalid_event", "native speech event kind is invalid");
            return;
        }
        if (event.text.size() > limits_.max_event_text_bytes ||
            event.audio_samples.size() > limits_.max_event_audio_samples) {
            emit_error("event_too_large", "native speech event exceeds its limit");
            return;
        }
        if (event.kind == SpeechSessionEventKind::kAgentAudio &&
            (event.audio_samples.empty() || event.sample_rate != kWireSampleRate)) {
            emit_error("invalid_event", "native agent audio must be non-empty 24 kHz PCM");
            return;
        }
        json value = native_event_json(event, kind);
        const auto status = writer_.write(value);
        if (status == WriteResult::kLineTooLong)
            emit_error("event_too_large", "native speech event exceeds its line limit");
        else if (status == WriteResult::kStreamError)
            output_failed_ = true;
        else
            ++stats_.events;
    }

    json native_event_json(const SpeechSessionEvent& event, const char* kind) const {
        const std::string text =
            event.kind == SpeechSessionEventKind::kError ? "speech session error" : event.text;
        json value = {{"type", "session.event"},
                      {"kind", kind},
                      {"epoch", event.epoch},
                      {"sequence", event.sequence},
                      {"frame_index", event.frame_index},
                      {"media_start_sample", event.media_start_sample},
                      {"media_end_sample", event.media_end_sample},
                      {"sample_rate", event.sample_rate},
                      {"is_final", event.is_final},
                      {"text", text}};
        if (event.kind == SpeechSessionEventKind::kAgentAudio)
            value["audio"] = float_to_pcm16_base64(event.audio_samples);
        return value;
    }

    IPipeline& pipeline_;
    std::istream& input_;
    JsonlWriter writer_;
    RealtimeWorkerLimits limits_;
    BoundedCommandQueue command_queue_;
    std::unique_ptr<ISpeechSession> session_;
    WorkerStats stats_;
    std::optional<std::string> close_event_id_;
    std::string end_reason_{"closed"};
    std::size_t session_audio_samples_{0};
    std::uint64_t current_epoch_{0};
    std::uint64_t current_sequence_{0};
    bool have_epoch_{false};
    bool first_audio_received_{false};
    bool tools_enabled_{false};
    bool tool_response_active_{false};
    bool close_requested_{false};
    bool input_failed_{false};
    bool output_failed_{false};
};

void emit_start_failure(std::ostream& output, const RealtimeWorkerLimits& limits) {
    JsonlWriter writer(output, limits.max_line_bytes);
    writer.write({{"type", "error"},
                  {"code", "pipeline_unavailable"},
                  {"message", "speech pipeline is unavailable"}});
    writer.write({{"type", "session.end"},
                  {"reason", "start_error"},
                  {"stats",
                   {{"commands", 0},
                    {"audio_samples", 0},
                    {"events", 0},
                    {"errors", 1},
                    {"stale_events", 0}}}});
}

} // namespace

int run_realtime_worker(IPipeline& pipeline, std::istream& input, std::ostream& output,
                        const RealtimeWorkerLimits& limits) {
    try {
        RealtimeWorker worker(pipeline, input, output, limits);
        return worker.run();
    } catch (...) {
        emit_start_failure(output, limits);
        return 1;
    }
}

int run_realtime_worker(const RealtimePipelineFactory& factory, std::istream& input,
                        std::ostream& output, const RealtimeWorkerLimits& limits) {
    try {
        auto pipeline = factory ? factory() : nullptr;
        if (!pipeline)
            throw std::runtime_error("pipeline factory returned null");
        return run_realtime_worker(*pipeline, input, output, limits);
    } catch (...) {
        emit_start_failure(output, limits);
        return 1;
    }
}

} // namespace trtmc::serve
