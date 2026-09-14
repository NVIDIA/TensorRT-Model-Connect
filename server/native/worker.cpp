/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "native/worker.h"

#include "trtmc/task.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
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
constexpr std::size_t kWavReadBufferBytes = 64U * 1024U;
constexpr const char* kRuntimeErrorMessage = "native worker operation failed";

class ProtocolError final : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

class WavFormatError final : public std::runtime_error {
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
        const bool supported = std::any_of(allowed.begin(), allowed.end(),
                                           [&](const char* name) { return field.key() == name; });
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

void validate_generate_config(const TextGenerationConfig& config) {
    if (config.max_new_tokens <= 0)
        throw ProtocolError("config.max_new_tokens must be positive");
    if (config.top_k < 0)
        throw ProtocolError("config.top_k must be non-negative");
    require_non_negative(config.temperature, "config.temperature");
    require_finite(config.top_p, "config.top_p");
    require_finite(config.min_p, "config.min_p");
    if (config.top_p < 0.0F || config.top_p > 1.0F)
        throw ProtocolError("config.top_p must be in [0, 1]");
    if (config.min_p < 0.0F || config.min_p > 1.0F)
        throw ProtocolError("config.min_p must be in [0, 1]");
}

TextGenerationConfig parse_generate_config(const Json& request, std::int32_t default_tokens) {
    const Json config = request_config(request);
    require_config_fields(config, {"max_new_tokens", "temperature", "top_p", "min_p", "top_k",
                                   "seed", "use_chat_template", "enable_thinking"});
    TextGenerationConfig result;
    result.max_new_tokens = default_tokens > 0 ? default_tokens : 128;
    try {
        assign_if_present(config, "max_new_tokens", result.max_new_tokens);
        assign_if_present(config, "temperature", result.temperature);
        assign_if_present(config, "top_p", result.top_p);
        assign_if_present(config, "min_p", result.min_p);
        assign_if_present(config, "top_k", result.top_k);
        assign_if_present(config, "seed", result.seed);
        assign_if_present(config, "use_chat_template", result.use_chat_template);
        assign_if_present(config, "enable_thinking", result.enable_thinking);
    } catch (const nlohmann::json::exception&) {
        throw ProtocolError("generation config contains an invalid value");
    }
    validate_generate_config(result);
    return result;
}

TranscriptionConfig parse_transcription_config(const Json& request, std::int32_t sample_rate) {
    const Json config = request_config(request);
    require_config_fields(config, {"language"});
    TranscriptionConfig result;
    result.input_sample_rate = sample_rate;
    try {
        assign_if_present(config, "language", result.source_language);
    } catch (const nlohmann::json::exception&) {
        throw ProtocolError("transcription config contains an invalid value");
    }
    return result;
}

TranscriptionStreamConfig parse_stream_config(const Json& request) {
    const Json config = request_config(request);
    require_config_fields(config, {"sample_rate_hz", "channels", "audio_format", "language"});
    TranscriptionStreamConfig result;
    std::int32_t channels = 1;
    std::string audio_format{"pcm16le"};
    try {
        assign_if_present(config, "sample_rate_hz", result.input_sample_rate);
        assign_if_present(config, "language", result.language);
        channels = config.value("channels", 1);
        audio_format = config.value("audio_format", std::string{"pcm16le"});
    } catch (const nlohmann::json::exception&) {
        throw ProtocolError("stream config contains an invalid value");
    }
    if (channels != 1)
        throw ProtocolError("config.channels must be 1 because streaming input is mono");
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

std::uint16_t read_u16_le(const char* data) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(data);
    return static_cast<std::uint16_t>(bytes[0]) | (static_cast<std::uint16_t>(bytes[1]) << 8U);
}

std::uint32_t read_u32_le(const char* data) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(data);
    return static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8U) |
           (static_cast<std::uint32_t>(bytes[2]) << 16U) |
           (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

void read_exact(std::ifstream& input, char* destination, std::size_t size, const char* message) {
    if (size > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
        throw WavFormatError(message);
    input.read(destination, static_cast<std::streamsize>(size));
    if (input.gcount() != static_cast<std::streamsize>(size))
        throw WavFormatError(message);
}

void seek_exact(std::ifstream& input, std::uint64_t position) {
    if (position > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()))
        throw WavFormatError("WAV file is too large");
    input.clear();
    input.seekg(static_cast<std::streamoff>(position), std::ios::beg);
    if (!input)
        throw WavFormatError("WAV seek failed");
}

struct WavLayout {
    std::uint16_t format{0};
    std::uint16_t channels{0};
    std::uint32_t sample_rate{0};
    std::uint16_t bits_per_sample{0};
    std::uint64_t data_offset{0};
    std::uint32_t data_size{0};
    bool have_format{false};
    bool have_data{false};
};

std::uint64_t read_wav_container_end(std::ifstream& input, std::uint64_t file_size) {
    if (file_size < 12U)
        throw WavFormatError("WAV file is too small");
    std::array<char, 12> header{};
    seek_exact(input, 0);
    read_exact(input, header.data(), header.size(), "WAV header is truncated");
    if (std::memcmp(header.data(), "RIFF", 4) != 0)
        throw WavFormatError("file is not a RIFF container");
    if (std::memcmp(header.data() + 8, "WAVE", 4) != 0)
        throw WavFormatError("RIFF container is not WAVE audio");
    const std::uint32_t declared_size = read_u32_le(header.data() + 4);
    if (declared_size < 4U)
        throw WavFormatError("WAV RIFF chunk is too small");
    const std::uint64_t container_end = 8U + static_cast<std::uint64_t>(declared_size);
    if (container_end > file_size)
        throw WavFormatError("WAV contains a truncated RIFF chunk");
    return container_end;
}

struct WavChunk {
    std::array<char, 4> id{};
    std::uint32_t size{0};
    std::uint64_t data_offset{0};
    std::uint64_t padded_size{0};
};

WavChunk read_wav_chunk(std::ifstream& input, std::uint64_t position, std::uint64_t container_end) {
    if (container_end - position < 8U)
        throw WavFormatError("WAV contains a truncated chunk header");
    std::array<char, 8> header{};
    seek_exact(input, position);
    read_exact(input, header.data(), header.size(), "WAV chunk header is truncated");

    WavChunk chunk;
    std::copy_n(header.data(), chunk.id.size(), chunk.id.data());
    chunk.size = read_u32_le(header.data() + 4);
    chunk.data_offset = position + 8U;
    chunk.padded_size =
        static_cast<std::uint64_t>(chunk.size) + static_cast<std::uint64_t>(chunk.size & 1U);
    if (chunk.padded_size > container_end - chunk.data_offset)
        throw WavFormatError("WAV contains a truncated chunk");
    return chunk;
}

void read_wav_format(std::ifstream& input, const WavChunk& chunk, WavLayout& layout) {
    if (chunk.size < 16U)
        throw WavFormatError("WAV fmt chunk is too small");
    std::array<char, 16> format{};
    seek_exact(input, chunk.data_offset);
    read_exact(input, format.data(), format.size(), "WAV fmt chunk is truncated");
    layout.format = read_u16_le(format.data());
    layout.channels = read_u16_le(format.data() + 2);
    layout.sample_rate = read_u32_le(format.data() + 4);
    layout.bits_per_sample = read_u16_le(format.data() + 14);
    layout.have_format = true;
}

void apply_wav_chunk(std::ifstream& input, const WavChunk& chunk, WavLayout& layout) {
    if (std::memcmp(chunk.id.data(), "fmt ", 4) == 0) {
        read_wav_format(input, chunk, layout);
        return;
    }
    if (std::memcmp(chunk.id.data(), "data", 4) != 0 || chunk.size == 0U || layout.have_data)
        return;
    layout.data_offset = chunk.data_offset;
    layout.data_size = chunk.size;
    layout.have_data = true;
}

void validate_wav_layout(const WavLayout& layout) {
    if (!layout.have_format || !layout.have_data)
        throw WavFormatError("WAV must contain non-empty fmt and data chunks");
    if (layout.channels == 0U || layout.sample_rate == 0U)
        throw WavFormatError("WAV channels and sample rate must be positive");
    if (layout.sample_rate > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
        throw WavFormatError("WAV sample rate exceeds the supported range");
    const bool pcm16 = layout.format == 1U && layout.bits_per_sample == 16U;
    const bool float32 = layout.format == 3U && layout.bits_per_sample == 32U;
    if (!pcm16 && !float32)
        throw WavFormatError("WAV samples must be PCM16 or IEEE float32");
}

WavLayout parse_wav_layout(std::ifstream& input, std::uint64_t file_size) {
    const std::uint64_t container_end = read_wav_container_end(input, file_size);

    WavLayout layout;
    std::uint64_t position = 12U;
    while (position < container_end) {
        const WavChunk chunk = read_wav_chunk(input, position, container_end);
        apply_wav_chunk(input, chunk, layout);
        position = chunk.data_offset + chunk.padded_size;
    }
    validate_wav_layout(layout);
    return layout;
}

float decode_wav_sample(const char* data, std::uint16_t format) {
    if (format == 3U) {
        const std::uint32_t bits = read_u32_le(data);
        float value = 0.0F;
        std::memcpy(&value, &bits, sizeof(value));
        if (!std::isfinite(value))
            throw WavFormatError("WAV contains a non-finite float sample");
        return value;
    }
    const std::uint16_t raw = read_u16_le(data);
    const std::int32_t value =
        raw >= 0x8000U ? static_cast<std::int32_t>(raw) - 0x10000 : static_cast<std::int32_t>(raw);
    return static_cast<float>(value) / 32768.0F;
}

struct DecodedAudio {
    std::vector<float> samples;
    std::int32_t sample_rate{0};
};

DecodedAudio read_wav(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        throw std::runtime_error("cannot open uploaded audio file");
    const std::streampos end = input.tellg();
    if (end < 0)
        throw std::runtime_error("cannot determine uploaded audio file size");
    const auto file_size = static_cast<std::uint64_t>(end);
    const WavLayout layout = parse_wav_layout(input, file_size);
    const std::size_t sample_width = layout.bits_per_sample / 8U;
    const std::size_t frame_width = sample_width * static_cast<std::size_t>(layout.channels);
    if (frame_width == 0U || layout.data_size % frame_width != 0U)
        throw WavFormatError("WAV data does not contain complete audio frames");
    const std::size_t frame_count = layout.data_size / frame_width;
    if (frame_count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw WavFormatError("WAV contains too many audio frames");

    DecodedAudio result;
    result.samples.resize(frame_count);
    result.sample_rate = static_cast<std::int32_t>(layout.sample_rate);
    seek_exact(input, layout.data_offset);
    const std::size_t frames_per_read =
        std::max<std::size_t>(1U, kWavReadBufferBytes / frame_width);
    std::vector<char> buffer(frames_per_read * frame_width);
    for (std::size_t first = 0; first < frame_count; first += frames_per_read) {
        const std::size_t count = std::min(frames_per_read, frame_count - first);
        const std::size_t bytes = count * frame_width;
        read_exact(input, buffer.data(), bytes, "WAV audio data is truncated");
        for (std::size_t frame = 0; frame < count; ++frame) {
            float sum = 0.0F;
            for (std::uint16_t channel = 0; channel < layout.channels; ++channel) {
                const std::size_t offset =
                    frame * frame_width + static_cast<std::size_t>(channel) * sample_width;
                sum += decode_wav_sample(buffer.data() + offset, layout.format);
            }
            result.samples[first + frame] = sum / static_cast<float>(layout.channels);
        }
    }
    return result;
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
    if ((third == '=' && fourth != '=') || (!last && (third == '=' || fourth == '=')))
        throw ProtocolError("audio is not valid base64");
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
        const char third = encoded[offset + 2U];
        const char fourth = encoded[offset + 3U];
        validate_base64_padding(third, fourth, last);
        const std::uint32_t bits =
            (static_cast<std::uint32_t>(decode_required_base64_character(encoded[offset])) << 18U) |
            (static_cast<std::uint32_t>(decode_required_base64_character(encoded[offset + 1U]))
             << 12U) |
            (static_cast<std::uint32_t>(decode_optional_base64_character(third)) << 6U) |
            static_cast<std::uint32_t>(decode_optional_base64_character(fourth));
        decoded.push_back(static_cast<std::uint8_t>((bits >> 16U) & 0xFFU));
        if (third != '=')
            decoded.push_back(static_cast<std::uint8_t>((bits >> 8U) & 0xFFU));
        if (fourth != '=')
            decoded.push_back(static_cast<std::uint8_t>(bits & 0xFFU));
    }
    return decoded;
}

std::vector<float> decode_pcm16_base64(const std::string& encoded) {
    const auto bytes = decode_base64(encoded);
    if (bytes.size() % 2U != 0U)
        throw ProtocolError("audio must contain complete little-endian int16 samples");
    if (bytes.size() / 2U > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw ProtocolError("PCM chunk contains too many samples");

    std::vector<float> samples(bytes.size() / 2U);
    for (std::size_t index = 0; index < samples.size(); ++index) {
        const std::uint16_t raw = static_cast<std::uint16_t>(bytes[index * 2U]) |
                                  (static_cast<std::uint16_t>(bytes[index * 2U + 1U]) << 8U);
        const std::int32_t value = raw >= 0x8000U ? static_cast<std::int32_t>(raw) - 0x10000
                                                  : static_cast<std::int32_t>(raw);
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
    explicit Worker(ITask& task)
        : text_(dynamic_cast<ITextGeneration*>(&task)),
          transcription_(dynamic_cast<ITranscription*>(&task)),
          streaming_(dynamic_cast<IStreamingTranscription*>(&task)) {}

    Json ready_event() const {
        Json capabilities = Json::array();
        if (text_ != nullptr)
            capabilities.push_back(ITextGeneration::kTask);
        if (transcription_ != nullptr)
            capabilities.push_back(ITranscription::kTask);
        if (streaming_ != nullptr)
            capabilities.push_back(IStreamingTranscription::kTask);
        Json result = {
            {"event", "ready"},
            {"protocol_version", 3},
            {"capabilities", std::move(capabilities)},
        };
        if (text_ != nullptr)
            result["default_max_new_tokens"] = text_->default_max_new_tokens();
        return result;
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
        if (active_stream_)
            throw ProtocolError(std::string(operation) +
                                " is unavailable while a transcription stream is active");
    }

    ITextGeneration& text() const {
        if (text_ == nullptr)
            throw ProtocolError("loaded task does not support text generation");
        return *text_;
    }

    ITranscription& transcription() const {
        if (transcription_ == nullptr)
            throw ProtocolError("loaded task does not support transcription");
        return *transcription_;
    }

    IStreamingTranscription& streaming() const {
        if (streaming_ == nullptr)
            throw ProtocolError("loaded task does not support streaming transcription");
        return *streaming_;
    }

    Json generate(const Json& request) {
        require_no_active_stream("generate");
        auto& interface = text();
        const std::string prompt = required_string(request, "prompt", true);
        const auto config = parse_generate_config(request, interface.default_max_new_tokens());
        return text_result_json(interface.generate(prompt, config));
    }

    Json transcribe(const Json& request) {
        require_no_active_stream("transcribe");
        const std::string audio_path = required_string(request, "audio_path");
        DecodedAudio audio;
        try {
            audio = read_wav(audio_path);
        } catch (const WavFormatError&) {
            throw UnsupportedMediaTypeError(
                "uploaded audio must be a supported PCM16 or IEEE float32 WAV file");
        }
        const auto config = parse_transcription_config(request, audio.sample_rate);
        return text_result_json(transcription().transcribe(
            audio.samples.data(), static_cast<std::int32_t>(audio.samples.size()), config));
    }

    Json probe_transcription_stream(const Json& request) {
        require_no_active_stream("probe_transcription_stream");
        auto stream = streaming().create_transcription_stream(parse_stream_config(request));
        if (!stream)
            throw std::runtime_error("streaming transcription task returned a null stream");
        stream->reset();
        return {{"supported", true}};
    }

    Json stream_start(const Json& request) {
        if (active_stream_)
            throw ProtocolError("this worker already has an active transcription stream");
        auto stream = streaming().create_transcription_stream(parse_stream_config(request));
        if (!stream)
            throw std::runtime_error("streaming transcription task returned a null stream");
        active_stream_ = std::move(stream);
        return Json::object();
    }

    ITranscriptionStream& active_stream() {
        if (!active_stream_)
            throw ProtocolError("no active transcription stream");
        return *active_stream_;
    }

    std::unique_ptr<ITranscriptionStream> take_stream() {
        if (!active_stream_)
            throw ProtocolError("no active transcription stream");
        return std::move(active_stream_);
    }

    Json stream_chunk(const Json& request) {
        const auto encoded = request.find("audio");
        if (encoded == request.end() || !encoded->is_string())
            throw ProtocolError("audio must be a base64 PCM16 string");
        auto samples = decode_pcm16_base64(encoded->get<std::string>());
        const auto result =
            active_stream().accept_audio(samples.empty() ? nullptr : samples.data(),
                                         static_cast<std::int32_t>(samples.size()), false);
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

    ITextGeneration* text_{nullptr};
    ITranscription* transcription_{nullptr};
    IStreamingTranscription* streaming_{nullptr};
    std::unique_ptr<ITranscriptionStream> active_stream_;
};

bool valid_request_id(const Json& id) {
    return id.is_string() && !id.get_ref<const std::string&>().empty();
}

Json success_response(const Json& id, Json result) {
    return {{"id", id}, {"ok", true}, {"result", std::move(result)}};
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
        --line_size;
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

ProcessedRequest process_request_line(Worker& worker, std::string_view line) {
    Json request_id = nullptr;
    try {
        if (line.size() > kMaxRequestLineBytes)
            throw ProtocolError("request exceeds the 16 MiB JSONL limit");
        Json request;
        try {
            request = Json::parse(line.begin(), line.end());
        } catch (const nlohmann::json::parse_error&) {
            throw ProtocolError("request is not valid JSON");
        }
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
        if (dynamic_cast<const ProtocolError*>(&error) != nullptr)
            return {error_response(request_id, "invalid_request_error", error.what()), false};
        std::cerr << "[trtmc.serve.worker] " << error.what() << '\n';
        return {error_response(request_id, "runtime_error", kRuntimeErrorMessage), false};
    } catch (...) {
        std::cerr << "[trtmc.serve.worker] unknown native worker error\n";
        return {error_response(request_id, "runtime_error", kRuntimeErrorMessage), false};
    }
}

} // namespace

int run_worker_protocol(ITask& task, std::istream& input, std::ostream& output) {
    Worker worker(task);
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
