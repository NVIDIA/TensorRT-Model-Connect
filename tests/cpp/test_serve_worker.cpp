/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Native serve-worker protocol tests. These use an injected fake IPipeline, so
// they cover the long-lived JSONL dispatch and audio conversion without a GPU,
// TensorRT engine, or bundle artifact.

#include "serve/worker.h"
#include "test_helpers.h"
#include "trtmc/pipeline.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;

int failures = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

struct StreamObservations {
    int accept_calls{0};
    int finish_calls{0};
    int reset_calls{0};
    bool last_is_final{false};
    std::vector<float> last_samples;
};

class FakeTranscriptionStream final : public trtmc::ITranscriptionStream {
  public:
    FakeTranscriptionStream(trtmc::TranscriptionStreamConfig config,
                            std::shared_ptr<StreamObservations> observations)
        : config_(std::move(config)), observations_(std::move(observations)) {}

    trtmc::TranscriptionStreamResult accept_audio(const float* audio_samples, int32_t num_samples,
                                                  bool is_final) override {
        ++observations_->accept_calls;
        observations_->last_is_final = is_final;
        observations_->last_samples.assign(audio_samples, audio_samples + num_samples);
        trtmc::TranscriptionStreamResult result;
        result.text = "partial transcript";
        result.token_ids = {4, 5};
        result.is_final = is_final;
        result.chunk_index = observations_->accept_calls;
        result.accepted_samples = num_samples;
        result.sample_rate = config_.input_sample_rate;
        return result;
    }

    trtmc::TranscriptionStreamResult finish() override {
        ++observations_->finish_calls;
        trtmc::TranscriptionStreamResult result;
        result.text = "final transcript";
        result.token_ids = {4, 5, 6};
        result.is_final = true;
        result.chunk_index = observations_->accept_calls;
        result.accepted_samples = static_cast<int64_t>(observations_->last_samples.size());
        result.sample_rate = config_.input_sample_rate;
        return result;
    }

    void reset() override { ++observations_->reset_calls; }

    trtmc::TranscriptionStreamConfig config() const override { return config_; }

  private:
    trtmc::TranscriptionStreamConfig config_;
    std::shared_ptr<StreamObservations> observations_;
};

class FakePipeline final : public trtmc::IPipeline {
  public:
    const char* model_id() const override { return "fake/streaming-asr"; }
    const char* pipeline_type() const override { return "FakeStreamingAsrPipeline"; }
    int32_t default_max_new_tokens() const override { return 77; }

    trtmc::TextResult generate(const std::string& prompt,
                               const trtmc::GenerateConfig& config) override {
        ++generate_calls;
        last_prompt = prompt;
        last_generate_config = config;
        if (generate_invalid_argument)
            throw std::invalid_argument(
                "sensitive provider invalid argument at /tmp/private-provider.bundle");
        if (generate_runtime_error)
            throw std::runtime_error("sensitive native runtime detail at /tmp/private.bundle");
        trtmc::TextResult result{"generated: " + prompt, {8, 9}, 1.25, 2.5};
        if (generate_invalid_utf8)
            result.text.push_back(static_cast<char>(0xFF));
        result.setup_ms = 0.5;
        return result;
    }

    trtmc::TextResult transcribe(const float* audio_samples, int32_t num_samples,
                                 const trtmc::TranscriptionConfig& config) override {
        ++transcribe_calls;
        last_transcription_config = config;
        last_transcription_samples.assign(audio_samples, audio_samples + num_samples);
        trtmc::TranscriptionSegment segment;
        segment.start_seconds = 0.0;
        segment.end_seconds = 0.25;
        segment.text = "hello";
        segment.token_ids = {10};
        return {"hello from wav", {10, 11}, 0.0, 3.0, {segment}};
    }

    std::unique_ptr<trtmc::ITranscriptionStream>
    create_transcription_stream(const trtmc::TranscriptionStreamConfig& config) override {
        if (!stream_supported)
            throw std::runtime_error("streaming transcription unsupported");
        ++create_stream_calls;
        last_stream_config = config;
        stream_config_history.push_back(config);
        stream_observations = std::make_shared<StreamObservations>();
        stream_history.push_back(stream_observations);
        return std::make_unique<FakeTranscriptionStream>(config, stream_observations);
    }

    int generate_calls{0};
    int transcribe_calls{0};
    int create_stream_calls{0};
    bool stream_supported{true};
    bool generate_invalid_argument{false};
    bool generate_invalid_utf8{false};
    bool generate_runtime_error{false};
    std::string last_prompt;
    trtmc::GenerateConfig last_generate_config;
    trtmc::TranscriptionConfig last_transcription_config;
    trtmc::TranscriptionStreamConfig last_stream_config;
    std::vector<trtmc::TranscriptionStreamConfig> stream_config_history;
    std::vector<float> last_transcription_samples;
    std::shared_ptr<StreamObservations> stream_observations;
    std::vector<std::shared_ptr<StreamObservations>> stream_history;
};

std::filesystem::path make_temp_dir() {
    char pattern[] = "/tmp/trtmc_serve_worker_test_XXXXXX";
    char* directory = mkdtemp(pattern);
    if (directory == nullptr)
        throw std::runtime_error("mkdtemp failed");
    return std::filesystem::path(directory);
}

std::vector<Json> parse_output_lines(const std::string& output) {
    std::vector<Json> messages;
    std::istringstream stream(output);
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty())
            messages.push_back(Json::parse(line));
    }
    return messages;
}

void append_request(std::ostringstream& input, const Json& request) {
    input << request.dump() << '\n';
}

void write_pcm16_wav(const std::filesystem::path& path, const std::vector<int16_t>& samples,
                     uint16_t channels, uint32_t sample_rate) {
    std::ofstream output(path, std::ios::binary);
    const uint16_t format = 1;
    const uint16_t bits_per_sample = 16;
    const uint16_t block_align = channels * sizeof(int16_t);
    const uint32_t byte_rate = sample_rate * block_align;
    const uint32_t data_size = static_cast<uint32_t>(samples.size() * sizeof(int16_t));
    const uint32_t file_size = 36 + data_size;
    const uint32_t fmt_size = 16;

    output.write("RIFF", 4);
    output.write(reinterpret_cast<const char*>(&file_size), 4);
    output.write("WAVE", 4);
    output.write("fmt ", 4);
    output.write(reinterpret_cast<const char*>(&fmt_size), 4);
    output.write(reinterpret_cast<const char*>(&format), 2);
    output.write(reinterpret_cast<const char*>(&channels), 2);
    output.write(reinterpret_cast<const char*>(&sample_rate), 4);
    output.write(reinterpret_cast<const char*>(&byte_rate), 4);
    output.write(reinterpret_cast<const char*>(&block_align), 2);
    output.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
    output.write("data", 4);
    output.write(reinterpret_cast<const char*>(&data_size), 4);
    output.write(reinterpret_cast<const char*>(samples.data()), data_size);
}

void test_ready_metadata_and_stream_probe() {
    std::ostringstream requests;
    append_request(requests, {{"id", "probe-1"},
                              {"op", "probe_transcription_stream"},
                              {"config",
                               {{"sample_rate_hz", 24000},
                                {"channels", 1},
                                {"audio_format", "pcm16le"},
                                {"language", "en-US"}}}});
    append_request(
        requests,
        {{"id", "probe-2"},
         {"op", "probe_transcription_stream"},
         {"config", {{"sample_rate_hz", 16000}, {"channels", 1}, {"audio_format", "pcm16le"}}}});
    append_request(requests, {{"id", "stop"}, {"op", "shutdown"}});

    trtmc::BundleInfo bundle_info;
    bundle_info.runtime_strategy = "fake_streaming_asr";
    bundle_info.max_cache_length = 512;
    FakePipeline pipeline;
    std::istringstream input(requests.str());
    std::ostringstream output;
    const int status = trtmc::serve::run_worker_protocol(pipeline, bundle_info, input, output);
    check(status == 0, "ready/probe worker exits successfully");

    const auto messages = parse_output_lines(output.str());
    check(messages.size() == 4, "ready plus two probes and shutdown");
    if (messages.size() == 4) {
        check(messages[0].value("runtime_strategy", "") == "fake_streaming_asr",
              "ready includes runtime strategy");
        check(messages[0].value("max_cache_length", 0) == 512, "ready includes max cache length");
        check(messages[1]["result"] == Json{{"supported", true}},
              "first stream probe returns only supported=true");
        check(messages[2]["result"] == Json{{"supported", true}},
              "second probe returns only supported=true");
    }
    check(pipeline.create_stream_calls == 2 && pipeline.stream_history.size() == 2,
          "each probe creates an isolated stream");
    check(pipeline.stream_config_history.size() == 2 &&
              pipeline.stream_config_history[0].input_sample_rate == 24000 &&
              pipeline.stream_config_history[0].language == "en-US" &&
              pipeline.stream_config_history[1].input_sample_rate == 16000,
          "probe maps the canonical stream config fields");
    if (pipeline.stream_history.size() == 2) {
        check(pipeline.stream_history[0]->reset_calls == 1 &&
                  pipeline.stream_history[1]->reset_calls == 1,
              "probe resets streams before destruction");
    }
}

void test_failed_stream_probe_keeps_worker_usable() {
    std::ostringstream requests;
    append_request(
        requests,
        {{"id", "probe"}, {"op", "probe_transcription_stream"}, {"config", Json::object()}});
    append_request(requests, {{"id", "generate"}, {"op", "generate"}, {"prompt", "still alive"}});
    append_request(requests, {{"id", "stop"}, {"op", "shutdown"}});

    FakePipeline pipeline;
    pipeline.stream_supported = false;
    std::istringstream input(requests.str());
    std::ostringstream output;
    const int status =
        trtmc::serve::run_worker_protocol(pipeline, trtmc::BundleInfo{}, input, output);
    check(status == 0, "worker survives unsupported stream probe");

    const auto messages = parse_output_lines(output.str());
    check(messages.size() == 4, "failed probe still emits later generation and shutdown");
    if (messages.size() == 4) {
        check(!messages[1].value("ok", true) &&
                  messages[1]["error"].value("type", "") == "runtime_error",
              "unsupported probe is a structured runtime error");
        check(messages[2].value("ok", false) && pipeline.generate_calls == 1,
              "generation succeeds after failed probe");
    }
}

void test_invalid_utf8_result_is_replaced_in_jsonl() {
    std::ostringstream requests;
    append_request(requests, {{"id", "generate"}, {"op", "generate"}, {"prompt", "text"}});
    append_request(requests, {{"id", "stop"}, {"op", "shutdown"}});

    FakePipeline pipeline;
    pipeline.generate_invalid_utf8 = true;
    std::istringstream input(requests.str());
    std::ostringstream output;
    const int status =
        trtmc::serve::run_worker_protocol(pipeline, trtmc::BundleInfo{}, input, output);
    check(status == 0, "worker survives invalid UTF-8 in native output");

    const auto messages = parse_output_lines(output.str());
    check(messages.size() == 3, "invalid UTF-8 result still emits response and shutdown");
    if (messages.size() == 3) {
        const std::string text = messages[1]["result"].value("text", "");
        check(messages[1].value("ok", false), "invalid UTF-8 result remains successful");
        check(text.find("\xEF\xBF\xBD") != std::string::npos,
              "invalid UTF-8 is replaced with the Unicode replacement character");
        check(messages[2].value("ok", false), "worker remains usable after UTF-8 replacement");
    }
    check(output.str().find(static_cast<char>(0xFF)) == std::string::npos,
          "invalid UTF-8 byte is absent from JSONL output");
}

void test_runtime_error_is_generic_on_stdout_and_detailed_on_stderr() {
    std::ostringstream requests;
    append_request(requests, {{"id", "generate"}, {"op", "generate"}, {"prompt", "private"}});
    append_request(requests, {{"id", "stop"}, {"op", "shutdown"}});

    FakePipeline pipeline;
    pipeline.generate_runtime_error = true;
    std::istringstream input(requests.str());
    std::ostringstream output;
    std::ostringstream diagnostics;
    auto* previous_stderr = std::cerr.rdbuf(diagnostics.rdbuf());
    const int status =
        trtmc::serve::run_worker_protocol(pipeline, trtmc::BundleInfo{}, input, output);
    std::cerr.rdbuf(previous_stderr);

    check(status == 0, "worker survives native runtime error");
    const auto messages = parse_output_lines(output.str());
    check(messages.size() == 3, "runtime error produces one response before shutdown");
    if (messages.size() == 3) {
        const auto& error = messages[1]["error"];
        check(!messages[1].value("ok", true) && error.value("type", "") == "runtime_error",
              "native runtime failure keeps the runtime error type");
        check(error.value("message", "") == "native worker operation failed",
              "native runtime stdout uses a fixed generic message");
        check(output.str().find("sensitive native runtime detail") == std::string::npos &&
                  output.str().find("/tmp/private.bundle") == std::string::npos,
              "native runtime stdout hides diagnostics and paths");
    }
    check(diagnostics.str().find("sensitive native runtime detail") != std::string::npos &&
              diagnostics.str().find("/tmp/private.bundle") != std::string::npos,
          "native runtime detail remains on stderr");
}

void test_provider_invalid_argument_is_not_a_public_client_error() {
    std::ostringstream requests;
    append_request(requests, {{"id", "generate"}, {"op", "generate"}, {"prompt", "private"}});
    append_request(requests, {{"id", "stop"}, {"op", "shutdown"}});

    FakePipeline pipeline;
    pipeline.generate_invalid_argument = true;
    std::istringstream input(requests.str());
    std::ostringstream output;
    std::ostringstream diagnostics;
    auto* previous_stderr = std::cerr.rdbuf(diagnostics.rdbuf());
    const int status =
        trtmc::serve::run_worker_protocol(pipeline, trtmc::BundleInfo{}, input, output);
    std::cerr.rdbuf(previous_stderr);

    check(status == 0, "worker survives provider invalid_argument");
    const auto messages = parse_output_lines(output.str());
    check(messages.size() == 3, "provider invalid_argument produces one response");
    if (messages.size() == 3) {
        const auto& error = messages[1]["error"];
        check(!messages[1].value("ok", true) && error.value("type", "") == "runtime_error",
              "provider invalid_argument is a runtime failure");
        check(error.value("message", "") == "native worker operation failed",
              "provider invalid_argument uses the generic public message");
        check(output.str().find("sensitive provider invalid argument") == std::string::npos &&
                  output.str().find("/tmp/private-provider.bundle") == std::string::npos,
              "provider invalid_argument details stay off stdout");
    }
    check(diagnostics.str().find("sensitive provider invalid argument") != std::string::npos,
          "provider invalid_argument detail remains on stderr");
}

void test_full_worker_lifecycle() {
    const auto temporary = make_temp_dir();
    const auto wav_path = temporary / "request.wav";
    write_pcm16_wav(wav_path,
                    {
                        16384,
                        8192,
                        0,
                        -8192, // 0.125 after four-channel averaging
                        0,
                        8192,
                        16384,
                        24576, // 0.375 after four-channel averaging
                        -16384,
                        0,
                        16384,
                        0, // 0.0 after four-channel averaging
                    },
                    4, 22050);

    std::ostringstream requests;
    append_request(requests, {{"id", "generate"},
                              {"op", "generate"},
                              {"prompt", "summarize this"},
                              {"config",
                               {{"max_new_tokens", 12},
                                {"temperature", 0.25},
                                {"top_p", 0.8},
                                {"min_p", 0.05},
                                {"top_k", 7},
                                {"seed", 42},
                                {"use_chat_template", true},
                                {"enable_thinking", false}}}});
    append_request(requests, {{"id", "asr"},
                              {"op", "transcribe"},
                              {"audio_path", wav_path.string()},
                              {"config", {{"language", "en"}}}});
    append_request(requests, {{"id", "start"},
                              {"op", "stream_start"},
                              {"config",
                               {{"sample_rate_hz", 16000},
                                {"channels", 1},
                                {"audio_format", "pcm16le"},
                                {"language", "en-US"}}}});
    // PCM16 little-endian samples: 0, 16384, -32768.
    append_request(requests, {{"id", "chunk"}, {"op", "stream_chunk"}, {"audio", "AAAAQACA"}});
    append_request(requests, {{"id", "reset"}, {"op", "stream_reset"}});
    append_request(
        requests,
        {{"id", "restart"},
         {"op", "stream_start"},
         {"config", {{"sample_rate_hz", 16000}, {"channels", 1}, {"audio_format", "pcm16le"}}}});
    append_request(requests, {{"id", "finish"}, {"op", "stream_finish"}});
    append_request(requests, {{"id", "stop"}, {"op", "shutdown"}});

    FakePipeline pipeline;
    std::istringstream input(requests.str());
    std::ostringstream output;
    const int status =
        trtmc::serve::run_worker_protocol(pipeline, trtmc::BundleInfo{}, input, output);
    check(status == 0, "worker exits successfully after shutdown");

    const auto messages = parse_output_lines(output.str());
    check(messages.size() == 9, "ready plus eight request responses");
    if (messages.size() != 9) {
        trtmc_test::remove_all_safe(temporary);
        return;
    }
    check(messages[0].value("event", "") == "ready", "first message is ready event");
    check(messages[0].value("protocol_version", 0) == 2, "ready protocol version");
    check(messages[0].value("model_id", "") == "fake/streaming-asr", "ready model id");
    check(messages[0].value("pipeline_type", "") == "FakeStreamingAsrPipeline",
          "ready pipeline type");
    check(messages[0].value("default_max_new_tokens", 0) == 77, "ready returns default max tokens");

    for (std::size_t index = 1; index < messages.size(); ++index)
        check(messages[index].value("ok", false), "happy-path response has ok=true");

    check(messages[1]["id"] == "generate", "string request id is preserved");
    check(messages[1]["result"].value("text", "") == "generated: summarize this",
          "generate response text");
    check(messages[1]["result"]["token_ids"] == Json::array({8, 9}), "generate response token ids");
    check(messages[1]["result"].value("completion_tokens", 0) == 2,
          "generate response completion token count");
    check(pipeline.generate_calls == 1 && pipeline.last_prompt == "summarize this",
          "generate dispatches once to persistent pipeline");
    check(pipeline.last_generate_config.max_new_tokens == 12 &&
              std::abs(pipeline.last_generate_config.temperature - 0.25F) < 1e-6F &&
              std::abs(pipeline.last_generate_config.top_p - 0.8F) < 1e-6F &&
              std::abs(pipeline.last_generate_config.min_p - 0.05F) < 1e-6F &&
              pipeline.last_generate_config.top_k == 7 &&
              pipeline.last_generate_config.seed == 42 &&
              pipeline.last_generate_config.use_chat_template &&
              !pipeline.last_generate_config.enable_thinking,
          "generate maps exactly the Python app config fields");

    check(messages[2]["result"].value("text", "") == "hello from wav",
          "transcription response text");
    check(messages[2]["result"]["segments"].size() == 1, "transcription response segments");
    check(pipeline.transcribe_calls == 1 && pipeline.last_transcription_samples.size() == 3,
          "transcribe decodes WAV and dispatches once");
    if (pipeline.last_transcription_samples.size() == 3) {
        check(std::abs(pipeline.last_transcription_samples[0] - 0.125F) < 1e-6F,
              "transcribe averages all input channels");
        check(std::abs(pipeline.last_transcription_samples[1] - 0.375F) < 1e-6F,
              "transcribe preserves the second downmixed frame");
        check(std::abs(pipeline.last_transcription_samples[2]) < 1e-6F,
              "transcribe preserves channel cancellation");
    }
    check(pipeline.last_transcription_config.input_sample_rate == 22050 &&
              pipeline.last_transcription_config.source_language == "en" &&
              pipeline.last_transcription_config.max_output_tokens == 224 &&
              pipeline.last_transcription_config.beam_size == 1 &&
              !pipeline.last_transcription_config.timestamps,
          "transcription maps language and native WAV sample rate only");

    check(messages[3]["result"] == Json::object(), "stream start returns an empty object");
    check(pipeline.create_stream_calls == 2 && pipeline.stream_history.size() == 2 &&
              pipeline.stream_config_history.size() == 2 &&
              pipeline.stream_config_history[0].input_sample_rate == 16000 &&
              pipeline.stream_config_history[0].language == "en-US",
          "stream maps only canonical sample rate and language fields");
    check(messages[4]["result"] == Json{{"text", "partial transcript"}},
          "stream chunk returns only text");
    check(messages[5]["result"] == Json::object(), "stream reset returns an empty object");
    check(messages[6]["result"] == Json::object(), "restarted stream returns an empty object");
    const auto first_stream =
        pipeline.stream_history.empty() ? nullptr : pipeline.stream_history[0];
    check(first_stream && first_stream->last_samples.size() == 3,
          "PCM16 base64 chunk sample count");
    if (first_stream && first_stream->last_samples.size() == 3) {
        check(std::abs(first_stream->last_samples[0]) < 1e-6F, "PCM16 zero sample decoded");
        check(std::abs(first_stream->last_samples[1] - 0.5F) < 1e-6F,
              "PCM16 positive sample decoded");
        check(std::abs(first_stream->last_samples[2] + 1.0F) < 1e-6F,
              "PCM16 negative sample decoded");
    }
    check(first_stream && first_stream->reset_calls == 1, "stream reset dispatches");
    const auto second_stream =
        pipeline.stream_history.size() < 2 ? nullptr : pipeline.stream_history[1];
    check(second_stream && second_stream->finish_calls == 1, "stream finish dispatches");
    check(messages[7]["result"] == Json{{"text", "final transcript"}},
          "stream finish returns only text");
    check(messages[8]["result"].value("status", "") == "shutting_down",
          "shutdown response emitted");

    trtmc_test::remove_all_safe(temporary);
}

void test_protocol_v2_rejects_noncanonical_inputs() {
    const auto temporary = make_temp_dir();
    const auto wav_path = temporary / "request.wav";
    write_pcm16_wav(wav_path, {0}, 1, 16000);

    std::ostringstream requests;
    append_request(requests, {{"id", 7}, {"op", "generate"}, {"prompt", "integer id"}});
    append_request(requests, {{"id", ""}, {"op", "generate"}, {"prompt", "empty id"}});
    append_request(requests, {{"id", "generate-extra"},
                              {"op", "generate"},
                              {"prompt", "extra config"},
                              {"config", {{"num_samples", 2}}}});
    append_request(requests, {{"id", "transcribe-alias"},
                              {"op", "transcribe"},
                              {"audio_path", wav_path.string()},
                              {"config", {{"max_output_tokens", 12}}}});
    append_request(requests, {{"id", "stream-rate-alias"},
                              {"op", "stream_start"},
                              {"config", {{"input_sample_rate", 16000}}}});
    append_request(
        requests,
        {{"id", "stream-format-alias"},
         {"op", "stream_start"},
         {"config", {{"sample_rate_hz", 16000}, {"channels", 1}, {"audio_format", "pcm16"}}}});
    append_request(requests, {{"id", "stream-extra"},
                              {"op", "stream_start"},
                              {"config", {{"sample_rate_hz", 16000}, {"att_context_left", 56}}}});
    append_request(requests, {{"id", "healthy"}, {"op", "generate"}, {"prompt", "still alive"}});
    append_request(requests, {{"id", "stop"}, {"op", "shutdown"}});

    FakePipeline pipeline;
    std::istringstream input(requests.str());
    std::ostringstream output;
    const int status =
        trtmc::serve::run_worker_protocol(pipeline, trtmc::BundleInfo{}, input, output);
    check(status == 0, "worker survives noncanonical protocol inputs");

    const auto messages = parse_output_lines(output.str());
    check(messages.size() == 10, "noncanonical inputs each receive one structured response");
    if (messages.size() == 10) {
        check(messages[1]["id"].is_null() && !messages[1].value("ok", true) &&
                  messages[1]["error"].value("message", "").find("non-empty string") !=
                      std::string::npos,
              "protocol v2 rejects integer request ids");
        check(messages[2]["id"].is_null() && !messages[2].value("ok", true) &&
                  messages[2]["error"].value("message", "").find("non-empty string") !=
                      std::string::npos,
              "protocol v2 rejects empty request ids");
        check(messages[3]["error"].value("message", "").find("config.num_samples") !=
                  std::string::npos,
              "generate rejects fields not sent by the Python app");
        check(messages[4]["error"].value("message", "").find("config.max_output_tokens") !=
                  std::string::npos,
              "offline transcription rejects legacy decoding controls");
        check(messages[5]["error"].value("message", "").find("config.input_sample_rate") !=
                  std::string::npos,
              "stream config rejects the input_sample_rate alias");
        check(messages[6]["error"].value("message", "").find("must be 'pcm16le'") !=
                  std::string::npos,
              "stream config rejects the pcm16 format alias");
        check(messages[7]["error"].value("message", "").find("config.att_context_left") !=
                  std::string::npos,
              "stream config rejects extra tuning fields");
        for (std::size_t index = 1; index <= 7; ++index) {
            check(!messages[index].value("ok", true) &&
                      messages[index]["error"].value("type", "") == "invalid_request_error",
                  "noncanonical protocol input is an invalid request");
        }
        check(messages[8].value("ok", false), "canonical request succeeds after rejected inputs");
    }
    check(pipeline.generate_calls == 1 && pipeline.transcribe_calls == 0 &&
              pipeline.create_stream_calls == 0,
          "rejected inputs never reach IPipeline state");
    trtmc_test::remove_all_safe(temporary);
}

void test_unsupported_audio_error_is_structured() {
    const auto temporary = make_temp_dir();
    const auto audio_path = temporary / "invalid.wav";
    const auto missing_path = temporary / "missing.wav";
    {
        std::ofstream output(audio_path, std::ios::binary);
        output << "not a wav";
    }

    std::ostringstream requests;
    append_request(requests,
                   {{"id", "bad-wav"}, {"op", "transcribe"}, {"audio_path", audio_path.string()}});
    append_request(
        requests,
        {{"id", "missing-wav"}, {"op", "transcribe"}, {"audio_path", missing_path.string()}});
    append_request(requests, {{"id", "generate"}, {"op", "generate"}, {"prompt", "still alive"}});
    append_request(requests, {{"id", "stop"}, {"op", "shutdown"}});

    FakePipeline pipeline;
    std::istringstream input(requests.str());
    std::ostringstream output;
    const int status =
        trtmc::serve::run_worker_protocol(pipeline, trtmc::BundleInfo{}, input, output);
    check(status == 0, "worker survives unsupported audio");

    const auto messages = parse_output_lines(output.str());
    check(messages.size() == 5, "audio failures emit errors then later responses");
    if (messages.size() == 5) {
        const auto& format_error = messages[1]["error"];
        check(!messages[1].value("ok", true), "unsupported audio response has ok=false");
        check(format_error.value("type", "") == "invalid_request_error",
              "unsupported audio is a client request error");
        check(format_error.value("code", "") == "unsupported_media_type",
              "unsupported audio has stable error code");
        check(format_error.value("param", "") == "file",
              "unsupported audio identifies file parameter");
        check(format_error.value("message", "") ==
                  "uploaded audio must be a supported PCM16 or IEEE float32 WAV file",
              "unsupported audio returns a safe fixed message");

        const auto& io_error = messages[2]["error"];
        check(!messages[2].value("ok", true) && io_error.value("type", "") == "runtime_error",
              "missing audio path remains a runtime error");
        check(!io_error.contains("code"), "missing audio path has no client media error code");
        check(io_error.value("message", "").find(missing_path.string()) == std::string::npos,
              "missing audio error does not expose the temporary path");
        check(messages[3].value("ok", false) && pipeline.generate_calls == 1,
              "worker accepts a request after unsupported audio");
    }
    check(pipeline.transcribe_calls == 0, "unsupported audio never reaches IPipeline");
    trtmc_test::remove_all_safe(temporary);
}

void test_protocol_errors_remain_structured() {
    std::ostringstream requests;
    requests << "not-json\n";
    append_request(requests, {{"id", "metadata"}, {"op", "metadata"}});
    append_request(requests, {{"id", "legacy-start"}, {"op", "transcription_stream_start"}});
    append_request(requests, {{"id", "start"}, {"op", "stream_start"}});
    append_request(requests, {{"id", "duplicate"}, {"op", "stream_start"}});
    append_request(requests,
                   {{"id", "legacy-audio"}, {"op", "stream_chunk"}, {"pcm16_base64", "AAAA"}});
    append_request(requests, {{"id", "bad-audio"}, {"op", "stream_chunk"}, {"audio", "%%%"}});
    append_request(requests, {{"id", "reset"}, {"op", "stream_reset"}});
    append_request(requests, {{"id", "finish"}, {"op", "stream_finish"}});
    append_request(requests, {{"id", "stop"}, {"op", "shutdown"}});

    FakePipeline pipeline;
    std::istringstream input(requests.str());
    std::ostringstream output;
    const int status =
        trtmc::serve::run_worker_protocol(pipeline, trtmc::BundleInfo{}, input, output);
    check(status == 0, "worker continues after malformed requests");

    const auto messages = parse_output_lines(output.str());
    check(messages.size() == 11, "ready plus ten error/lifecycle responses");
    if (messages.size() != 11)
        return;
    check(messages[1]["id"].is_null() && !messages[1].value("ok", true),
          "malformed JSON gets id=null error response");
    check(messages[1]["error"].value("type", "") == "invalid_request_error",
          "malformed JSON error type");
    check(messages[2]["id"] == "metadata" && !messages[2].value("ok", true),
          "metadata RPC is not part of the protocol");
    check(messages[3]["id"] == "legacy-start" && !messages[3].value("ok", true),
          "legacy stream operation is rejected");
    check(messages[4].value("ok", false), "canonical stream start succeeds");
    check(!messages[5].value("ok", true), "a worker permits only one active stream");
    check(!messages[6].value("ok", true), "audio has no legacy field alias");
    check(messages[7]["id"] == "bad-audio" && !messages[7].value("ok", true),
          "invalid base64 gets structured error");
    check(messages[7]["error"].value("type", "") == "invalid_request_error",
          "invalid base64 error type");
    check(messages[8].value("ok", false), "canonical reset clears the active stream");
    check(!messages[9].value("ok", true), "finish without an active stream is rejected");
    check(messages[10].value("ok", false), "shutdown still succeeds after errors");
}

} // namespace

int main() {
    try {
        test_ready_metadata_and_stream_probe();
        test_failed_stream_probe_keeps_worker_usable();
        test_invalid_utf8_result_is_replaced_in_jsonl();
        test_runtime_error_is_generic_on_stdout_and_detailed_on_stderr();
        test_provider_invalid_argument_is_not_a_public_client_error();
        test_full_worker_lifecycle();
        test_protocol_v2_rejects_noncanonical_inputs();
        test_unsupported_audio_error_is_structured();
        test_protocol_errors_remain_structured();
    } catch (const std::exception& error) {
        std::cerr << "Unhandled test exception: " << error.what() << '\n';
        return 1;
    }

    if (failures != 0)
        std::cerr << failures << " serve worker test(s) failed\n";
    return failures == 0 ? 0 : 1;
}
