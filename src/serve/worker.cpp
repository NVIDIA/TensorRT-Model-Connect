/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "serve/worker.h"

#include "trtmc/trtmc_io.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <istream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trtmc::serve {
namespace {

using Json = nlohmann::json;

constexpr std::size_t kMaxRequestLineBytes = 16U * 1024U * 1024U;
constexpr const char* kRuntimeErrorMessage = "native worker operation failed";

class ProtocolError final : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

class UnsupportedMediaTypeError final : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

struct DispatchResult {
    Json result;
    bool shutdown{false};
};

template <typename T>
void assign_if_present(const Json& object, const char* name, T& destination) {
    const auto value = object.find(name);
    if (value != object.end())
        destination = value->get<T>();
}

Json request_config(const Json& request) {
    const auto value = request.find("config");
    if (value == request.end())
        return Json::object();
    if (!value->is_object())
        throw ProtocolError("config must be a JSON object");
    return *value;
}

void require_config_fields(const Json& config, std::initializer_list<const char*> allowed) {
    for (auto field = config.begin(); field != config.end(); ++field) {
        bool supported = false;
        for (const char* candidate : allowed) {
            if (field.key() == candidate) {
                supported = true;
                break;
            }
        }
        if (!supported)
            throw ProtocolError("config." + field.key() + " is unsupported");
    }
}

void require_finite(float value, const char* name) {
    if (!std::isfinite(value))
        throw ProtocolError(std::string(name) + " must be finite");
}

void require_non_negative(float value, const char* name) {
    require_finite(value, name);
    if (value < 0.0F)
        throw ProtocolError(std::string(name) + " must be non-negative");
}

void validate_generate_counts(const GenerateConfig& config) {
    if (config.max_new_tokens <= 0)
        throw ProtocolError("config.max_new_tokens must be positive");
    if (config.top_k < 0)
        throw ProtocolError("config.top_k must be non-negative");
}

void validate_generate_sampling(const GenerateConfig& config) {
    require_non_negative(config.temperature, "config.temperature");
    require_finite(config.top_p, "config.top_p");
    require_finite(config.min_p, "config.min_p");
    if (config.top_p < 0.0F || config.top_p > 1.0F)
        throw ProtocolError("config.top_p must be in [0, 1]");
    if (config.min_p < 0.0F || config.min_p > 1.0F)
        throw ProtocolError("config.min_p must be in [0, 1]");
}

GenerateConfig parse_generate_config(const Json& request, int32_t default_max_new_tokens) {
    const Json config = request_config(request);
    require_config_fields(config, {"max_new_tokens", "temperature", "top_p", "min_p", "top_k",
                                   "seed", "use_chat_template", "enable_thinking"});
    GenerateConfig result;
    result.max_new_tokens = default_max_new_tokens > 0 ? default_max_new_tokens : 128;

    assign_if_present(config, "max_new_tokens", result.max_new_tokens);
    assign_if_present(config, "temperature", result.temperature);
    assign_if_present(config, "top_p", result.top_p);
    assign_if_present(config, "min_p", result.min_p);
    assign_if_present(config, "top_k", result.top_k);
    assign_if_present(config, "seed", result.seed);
    assign_if_present(config, "use_chat_template", result.use_chat_template);
    assign_if_present(config, "enable_thinking", result.enable_thinking);

    validate_generate_counts(result);
    validate_generate_sampling(result);
    return result;
}

TranscriptionConfig parse_transcription_config(const Json& request, int32_t input_sample_rate) {
    const Json config = request_config(request);
    require_config_fields(config, {"language"});
    TranscriptionConfig result;
    result.input_sample_rate = input_sample_rate;
    assign_if_present(config, "language", result.source_language);
    return result;
}

TranscriptionStreamConfig parse_stream_config(const Json& request) {
    const Json config = request_config(request);
    require_config_fields(config, {"sample_rate_hz", "channels", "audio_format", "language"});
    TranscriptionStreamConfig result;
    assign_if_present(config, "sample_rate_hz", result.input_sample_rate);
    assign_if_present(config, "language", result.language);

    const int32_t channels = config.value("channels", 1);
    if (channels != 1)
        throw ProtocolError("config.channels must be 1 because streaming input is mono");
    const std::string audio_format = config.value("audio_format", std::string{"pcm16le"});
    if (audio_format != "pcm16le")
        throw ProtocolError("config.audio_format must be 'pcm16le'");

    if (result.input_sample_rate <= 0)
        throw ProtocolError("config.sample_rate_hz must be positive");
    return result;
}

Json transcription_segments_json(const std::vector<TranscriptionSegment>& segments) {
    Json result = Json::array();
    for (const auto& segment : segments) {
        result.push_back({
            {"start_seconds", segment.start_seconds},
            {"end_seconds", segment.end_seconds},
            {"text", segment.text},
            {"token_ids", segment.token_ids},
        });
    }
    return result;
}

Json text_result_json(const TextResult& result) {
    return {
        {"text", result.text},
        {"token_ids", result.token_ids},
        {"completion_tokens", result.token_ids.size()},
        {"segments", transcription_segments_json(result.segments)},
        {"setup_ms", result.setup_ms},
        {"prefill_ms", result.prefill_ms},
        {"decode_ms", result.decode_ms},
    };
}

Json stream_result_json(const TranscriptionStreamResult& result) {
    return {{"text", result.text}};
}

int decode_base64_character(unsigned char value) {
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

int decode_required_base64_character(char value) {
    const int decoded = decode_base64_character(static_cast<unsigned char>(value));
    if (decoded < 0)
        throw ProtocolError("audio is not valid base64");
    return decoded;
}

int decode_optional_base64_character(char value) {
    return value == '=' ? 0 : decode_required_base64_character(value);
}

void validate_base64_padding(char third, char fourth, bool last) {
    if (third == '=' && fourth != '=')
        throw ProtocolError("audio is not valid base64");
    if (!last && third == '=')
        throw ProtocolError("audio is not valid base64");
    if (!last && fourth == '=')
        throw ProtocolError("audio is not valid base64");
}

struct Base64Quartet {
    std::uint32_t bits{0};
    bool has_second_byte{false};
    bool has_third_byte{false};
};

Base64Quartet parse_base64_quartet(const std::string& encoded, std::size_t offset, bool last) {
    const char third = encoded[offset + 2U];
    const char fourth = encoded[offset + 3U];
    validate_base64_padding(third, fourth, last);
    const int first_value = decode_required_base64_character(encoded[offset]);
    const int second_value = decode_required_base64_character(encoded[offset + 1U]);
    const int third_value = decode_optional_base64_character(third);
    const int fourth_value = decode_optional_base64_character(fourth);
    const std::uint32_t bits = (static_cast<std::uint32_t>(first_value) << 18U) |
                               (static_cast<std::uint32_t>(second_value) << 12U) |
                               (static_cast<std::uint32_t>(third_value) << 6U) |
                               static_cast<std::uint32_t>(fourth_value);
    return {bits, third != '=', fourth != '='};
}

void append_base64_quartet(const Base64Quartet& quartet, std::vector<std::uint8_t>& decoded) {
    decoded.push_back(static_cast<std::uint8_t>((quartet.bits >> 16U) & 0xFFU));
    if (quartet.has_second_byte)
        decoded.push_back(static_cast<std::uint8_t>((quartet.bits >> 8U) & 0xFFU));
    if (quartet.has_third_byte)
        decoded.push_back(static_cast<std::uint8_t>(quartet.bits & 0xFFU));
}

std::vector<std::uint8_t> decode_base64(const std::string& encoded) {
    if (encoded.empty())
        return {};
    if (encoded.size() % 4U != 0U)
        throw ProtocolError("audio has invalid base64 length");

    std::vector<std::uint8_t> decoded;
    decoded.reserve(encoded.size() / 4U * 3U);
    for (std::size_t offset = 0; offset < encoded.size(); offset += 4U) {
        const bool last = offset + 4U == encoded.size();
        append_base64_quartet(parse_base64_quartet(encoded, offset, last), decoded);
    }
    return decoded;
}

std::vector<float> decode_pcm16_base64(const std::string& encoded) {
    const auto bytes = decode_base64(encoded);
    if (bytes.size() % 2U != 0U)
        throw ProtocolError("audio must contain complete little-endian int16 samples");
    if (bytes.size() / 2U > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
        throw ProtocolError("PCM chunk contains too many samples");

    std::vector<float> samples(bytes.size() / 2U);
    for (std::size_t index = 0; index < samples.size(); ++index) {
        const std::uint16_t raw = static_cast<std::uint16_t>(bytes[index * 2U]) |
                                  (static_cast<std::uint16_t>(bytes[index * 2U + 1U]) << 8U);
        const int32_t value =
            raw >= 0x8000U ? static_cast<int32_t>(raw) - 0x10000 : static_cast<int32_t>(raw);
        samples[index] = static_cast<float>(value) / 32768.0F;
    }
    return samples;
}

std::string required_string(const Json& request, const char* field, bool allow_empty = false) {
    const auto value = request.find(field);
    if (value == request.end() || !value->is_string())
        throw ProtocolError(std::string(field) + " must be a string");
    const std::string result = value->get<std::string>();
    if (result.empty() && !allow_empty)
        throw ProtocolError(std::string(field) + " must not be empty");
    return result;
}

class Worker final {
  public:
    Worker(IPipeline& pipeline, const BundleInfo& bundle_info)
        : pipeline_(pipeline), runtime_strategy_(bundle_info.runtime_strategy),
          max_cache_length_(bundle_info.max_cache_length) {}

    Json ready_event() const {
        return {
            {"event", "ready"},
            {"protocol_version", 2},
            {"model_id", pipeline_.model_id()},
            {"pipeline_type", pipeline_.pipeline_type()},
            {"default_max_new_tokens", pipeline_.default_max_new_tokens()},
            {"runtime_strategy", runtime_strategy_},
            {"max_cache_length", max_cache_length_},
        };
    }

    DispatchResult dispatch(const Json& request) {
        if (!request.is_object())
            throw ProtocolError("request must be a JSON object");
        const std::string operation = required_string(request, "op");
        if (operation == "shutdown")
            return {{{"status", "shutting_down"}}, true};
        if (operation == "generate")
            return {generate(request), false};
        if (operation == "transcribe")
            return {transcribe(request), false};
        if (operation == "probe_transcription_stream")
            return {probe_transcription_stream(request), false};
        if (operation == "stream_start")
            return {stream_start(request), false};
        if (operation == "stream_chunk")
            return {stream_chunk(request), false};
        if (operation == "stream_finish")
            return {stream_finish(), false};
        if (operation == "stream_reset")
            return {stream_reset(), false};
        throw ProtocolError("unknown operation: " + operation);
    }

  private:
    void require_no_active_stream(const char* operation) const {
        if (active_stream_) {
            throw ProtocolError(std::string(operation) +
                                " is unavailable while a transcription stream is active");
        }
    }

    Json generate(const Json& request) {
        require_no_active_stream("generate");
        const std::string prompt = required_string(request, "prompt", true);
        const auto config = parse_generate_config(request, pipeline_.default_max_new_tokens());
        return text_result_json(pipeline_.generate(prompt, config));
    }

    Json transcribe(const Json& request) {
        require_no_active_stream("transcribe");
        const std::string audio_path = required_string(request, "audio_path");
        AudioResult audio;
        try {
            audio = io::read_wav(audio_path);
        } catch (const io::WavFormatError&) {
            throw UnsupportedMediaTypeError(
                "uploaded audio must be a supported PCM16 or IEEE float32 WAV file");
        }
        if (audio.samples.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
            throw ProtocolError("audio file contains too many samples");
        }
        const auto config = parse_transcription_config(request, audio.sample_rate);
        return text_result_json(pipeline_.transcribe(
            audio.samples.data(), static_cast<int32_t>(audio.samples.size()), config));
    }

    Json stream_start(const Json& request) {
        if (active_stream_)
            throw ProtocolError("this worker already has an active transcription stream");

        auto stream = pipeline_.create_transcription_stream(parse_stream_config(request));
        if (!stream)
            throw std::runtime_error("pipeline returned a null transcription stream");
        active_stream_ = std::move(stream);
        return Json::object();
    }

    Json probe_transcription_stream(const Json& request) {
        require_no_active_stream("probe_transcription_stream");
        auto stream = pipeline_.create_transcription_stream(parse_stream_config(request));
        if (!stream)
            throw std::runtime_error("pipeline returned a null transcription stream");
        stream->reset();
        return {{"supported", true}};
    }

    ITranscriptionStream& active_stream() {
        if (!active_stream_)
            throw ProtocolError("no active transcription stream");
        return *active_stream_;
    }

    std::unique_ptr<ITranscriptionStream> take_stream() {
        if (!active_stream_)
            throw ProtocolError("no active transcription stream");
        auto result = std::move(active_stream_);
        return result;
    }

    Json stream_chunk(const Json& request) {
        const auto encoded = request.find("audio");
        if (encoded == request.end() || !encoded->is_string())
            throw ProtocolError("audio must be a base64 PCM16 string");
        auto samples = decode_pcm16_base64(encoded->get<std::string>());
        const auto result =
            active_stream().accept_audio(samples.empty() ? nullptr : samples.data(),
                                         static_cast<int32_t>(samples.size()), false);
        return stream_result_json(result);
    }

    Json stream_finish() {
        auto stream = take_stream();
        return stream_result_json(stream->finish());
    }

    Json stream_reset() {
        auto stream = take_stream();
        stream->reset();
        return Json::object();
    }

    IPipeline& pipeline_;
    std::string runtime_strategy_;
    int32_t max_cache_length_{0};
    std::unique_ptr<ITranscriptionStream> active_stream_;
};

bool valid_request_id(const Json& id) {
    return id.is_string() && !id.get_ref<const std::string&>().empty();
}

Json success_response(const Json& id, Json result) {
    return {
        {"id", id},
        {"ok", true},
        {"result", std::move(result)},
    };
}

Json error_response(const Json& id, const char* type, const std::string& message,
                    const char* code = nullptr, const char* param = nullptr) {
    Json error = {{"type", type}, {"message", message}};
    if (code != nullptr)
        error["code"] = code;
    if (param != nullptr)
        error["param"] = param;
    return {{"id", id}, {"ok", false}, {"error", std::move(error)}};
}

bool write_message(std::ostream& output, const Json& message) {
    output << message.dump(-1, ' ', false, Json::error_handler_t::replace) << '\n';
    output.flush();
    return static_cast<bool>(output);
}

struct ProcessedRequest {
    Json response;
    bool shutdown{false};
};

enum class RequestLineRead { kRecord, kEnd, kError };

RequestLineRead read_request_line(std::istream& input, std::vector<char>& buffer,
                                  std::size_t& line_size) {
    input.getline(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize extracted = input.gcount();
    if (input.bad())
        return RequestLineRead::kError;
    if (input.eof() && extracted == 0)
        return RequestLineRead::kEnd;

    if (input.fail()) {
        input.clear(input.rdstate() & ~std::ios::failbit);
        input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        if (input.bad())
            return RequestLineRead::kError;
        line_size = kMaxRequestLineBytes + 1U;
        return RequestLineRead::kRecord;
    }

    line_size = static_cast<std::size_t>(extracted);
    if (!input.eof())
        --line_size; // getline() counts, but does not store, the delimiter.
    return RequestLineRead::kRecord;
}

Json extract_request_id(const Json& request) {
    if (!request.is_object())
        return nullptr;
    const auto id = request.find("id");
    if (id == request.end() || !valid_request_id(*id))
        return nullptr;
    return *id;
}

bool is_invalid_request_exception(const std::exception& error) {
    return dynamic_cast<const ProtocolError*>(&error) != nullptr ||
           dynamic_cast<const nlohmann::json::exception*>(&error) != nullptr;
}

ProcessedRequest process_request_line(Worker& worker, std::string_view line) {
    Json request_id = nullptr;
    try {
        if (line.size() > kMaxRequestLineBytes)
            throw ProtocolError("request exceeds the 16 MiB JSONL limit");
        const Json request = Json::parse(line.begin(), line.end());
        request_id = extract_request_id(request);
        if (request_id.is_null())
            throw ProtocolError("id must be a non-empty string");

        auto dispatched = worker.dispatch(request);
        return {success_response(request_id, std::move(dispatched.result)), dispatched.shutdown};
    } catch (const UnsupportedMediaTypeError& error) {
        return {error_response(request_id, "invalid_request_error", error.what(),
                               "unsupported_media_type", "file"),
                false};
    } catch (const std::exception& error) {
        if (is_invalid_request_exception(error))
            return {error_response(request_id, "invalid_request_error", error.what()), false};
        std::cerr << "[trtmc.serve.worker] " << error.what() << '\n';
        return {error_response(request_id, "runtime_error", kRuntimeErrorMessage), false};
    } catch (...) {
        std::cerr << "[trtmc.serve.worker] unknown native worker error\n";
        return {error_response(request_id, "runtime_error", kRuntimeErrorMessage), false};
    }
}

} // namespace

int run_worker_protocol(IPipeline& pipeline, const BundleInfo& bundle_info, std::istream& input,
                        std::ostream& output) {
    Worker worker(pipeline, bundle_info);
    if (!write_message(output, worker.ready_event()))
        return 2;

    std::vector<char> line_buffer(kMaxRequestLineBytes + 2U);
    while (true) {
        std::size_t line_size = 0;
        const RequestLineRead read = read_request_line(input, line_buffer, line_size);
        if (read == RequestLineRead::kEnd)
            return 0;
        if (read == RequestLineRead::kError)
            return 2;
        if (line_size == 0)
            continue;

        auto processed =
            process_request_line(worker, std::string_view(line_buffer.data(), line_size));
        if (!write_message(output, processed.response))
            return 2;
        if (processed.shutdown)
            return 0;
    }
}

} // namespace trtmc::serve
