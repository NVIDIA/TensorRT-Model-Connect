/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The native protocol is tested with an injected ITask, so every case is CPU-only.

#include "native/entrypoint.h"
#include "native/worker.h"
#include "trtmc/task.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

struct StreamObservations {
    int accept_calls{0};
    int finish_calls{0};
    int reset_calls{0};
    std::vector<float> last_samples;
};

class FakeStream final : public trtmc::ITranscriptionStream {
  public:
    FakeStream(trtmc::TranscriptionStreamConfig config,
               std::shared_ptr<StreamObservations> observations)
        : config_(std::move(config)), observations_(std::move(observations)) {}

    trtmc::TranscriptionStreamResult accept_audio(const float* samples, std::int32_t count,
                                                  bool is_final) override {
        ++observations_->accept_calls;
        observations_->last_samples.assign(samples, samples + count);
        return {"partial transcript",        {4, 5}, is_final,
                observations_->accept_calls, count,  config_.input_sample_rate};
    }

    trtmc::TranscriptionStreamResult finish() override {
        ++observations_->finish_calls;
        return {"final transcript",
                {4, 5, 6},
                true,
                observations_->accept_calls,
                static_cast<std::int64_t>(observations_->last_samples.size()),
                config_.input_sample_rate};
    }

    void reset() override { ++observations_->reset_calls; }
    trtmc::TranscriptionStreamConfig config() const override { return config_; }

  private:
    trtmc::TranscriptionStreamConfig config_;
    std::shared_ptr<StreamObservations> observations_;
};

class FakeTask final : public trtmc::ITextGeneration,
                       public trtmc::ITranscription,
                       public trtmc::IStreamingTranscription {
  public:
    const char* task() const noexcept override { return "test_multi_capability"; }
    std::int32_t default_max_new_tokens() const override { return 77; }

    trtmc::TextResult generate(const std::string& prompt,
                               const trtmc::TextGenerationConfig& config) override {
        ++generate_calls;
        last_prompt = prompt;
        last_generate_config = config;
        if (generate_invalid_argument)
            throw std::invalid_argument(
                "sensitive provider invalid argument at /tmp/private-provider.bundle");
        if (generate_runtime_error)
            throw std::runtime_error("sensitive native runtime detail at /tmp/private.bundle");
        if (generate_json_exception) {
            const auto ignored = Json::parse("sensitive-provider-json-detail");
            (void)ignored;
        }
        trtmc::TextResult result{"generated: " + prompt, {8, 9}, 1.25, 2.5};
        if (generate_invalid_utf8)
            result.text.push_back(static_cast<char>(0xFF));
        result.setup_ms = 0.5;
        return result;
    }

    trtmc::TextResult transcribe(const float* samples, std::int32_t count,
                                 const trtmc::TranscriptionConfig& config) override {
        ++transcribe_calls;
        last_transcription_config = config;
        last_transcription_samples.assign(samples, samples + count);
        trtmc::TranscriptionSegment segment;
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
        stream_configs.push_back(config);
        auto observations = std::make_shared<StreamObservations>();
        stream_history.push_back(observations);
        return std::make_unique<FakeStream>(config, std::move(observations));
    }

    int generate_calls{0};
    int transcribe_calls{0};
    int create_stream_calls{0};
    bool stream_supported{true};
    bool generate_invalid_argument{false};
    bool generate_invalid_utf8{false};
    bool generate_runtime_error{false};
    bool generate_json_exception{false};
    std::string last_prompt;
    trtmc::TextGenerationConfig last_generate_config;
    trtmc::TranscriptionConfig last_transcription_config;
    std::vector<float> last_transcription_samples;
    std::vector<trtmc::TranscriptionStreamConfig> stream_configs;
    std::vector<std::shared_ptr<StreamObservations>> stream_history;
};

class TextOnlyTask final : public trtmc::ITextGeneration {
  public:
    std::int32_t default_max_new_tokens() const override { return 5; }
    trtmc::TextResult generate(const std::string& prompt,
                               const trtmc::TextGenerationConfig&) override {
        return {prompt, {1}};
    }
};

std::filesystem::path make_temp_dir() {
    char pattern[] = "/tmp/trtmc_serve_worker_test_XXXXXX";
    char* directory = mkdtemp(pattern);
    if (directory == nullptr)
        throw std::runtime_error("mkdtemp failed");
    return std::filesystem::path(directory);
}

struct TempDir {
    std::filesystem::path path{make_temp_dir()};
    ~TempDir() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

std::vector<Json> parse_output_lines(const std::string& output) {
    std::vector<Json> messages;
    std::istringstream stream(output);
    for (std::string line; std::getline(stream, line);) {
        if (!line.empty())
            messages.push_back(Json::parse(line));
    }
    return messages;
}

void append_request(std::ostringstream& input, const Json& request) {
    input << request.dump() << '\n';
}

template <typename Sample>
void write_wav(const std::filesystem::path& path, const std::vector<Sample>& samples,
               std::uint16_t format, std::uint16_t channels, std::uint32_t sample_rate) {
    std::ofstream output(path, std::ios::binary);
    const std::uint16_t bits_per_sample = static_cast<std::uint16_t>(sizeof(Sample) * 8U);
    const std::uint16_t block_align = static_cast<std::uint16_t>(channels * sizeof(Sample));
    const std::uint32_t byte_rate = sample_rate * block_align;
    const std::uint32_t data_size = static_cast<std::uint32_t>(samples.size() * sizeof(Sample));
    const std::uint32_t file_size = 36U + data_size;
    const std::uint32_t fmt_size = 16;
    output.write("RIFF", 4);
    output.write(reinterpret_cast<const char*>(&file_size), 4);
    output.write("WAVEfmt ", 8);
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

std::vector<Json> run(FakeTask& task, const std::string& input, int* status = nullptr) {
    std::istringstream requests(input);
    std::ostringstream output;
    const int result = trtmc::serve::run_worker_protocol(task, requests, output);
    if (status != nullptr)
        *status = result;
    return parse_output_lines(output.str());
}

void test_ready_schema_uses_actual_task_capabilities() {
    std::ostringstream requests;
    append_request(requests, {{"id", "stop"}, {"op", "shutdown"}});
    FakeTask task;
    const auto messages = run(task, requests.str());
    check(messages.size() == 2, "ready plus shutdown response");
    if (messages.size() == 2) {
        const auto& ready = messages[0];
        check(ready == Json{{"event", "ready"},
                            {"protocol_version", 3},
                            {"capabilities", Json::array({"text_generation", "transcription",
                                                          "transcription_streaming"})},
                            {"default_max_new_tokens", 77}},
              "ready schema is minimal and capability based");
    }

    TextOnlyTask text;
    std::istringstream input(requests.str());
    std::ostringstream output;
    check(trtmc::serve::run_worker_protocol(text, input, output) == 0,
          "text-only task runs protocol");
    const auto text_messages = parse_output_lines(output.str());
    check(text_messages.size() == 2 &&
              text_messages[0]["capabilities"] == Json::array({"text_generation"}),
          "ready does not infer unsupported task capabilities");
}

void test_generation_and_stream_lifecycle() {
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
    append_request(requests, {{"id", "probe"},
                              {"op", "probe_transcription_stream"},
                              {"config",
                               {{"sample_rate_hz", 16000},
                                {"channels", 1},
                                {"audio_format", "pcm16le"},
                                {"language", "en-US"}}}});
    append_request(
        requests,
        {{"id", "start"},
         {"op", "stream_start"},
         {"config", {{"sample_rate_hz", 16000}, {"channels", 1}, {"audio_format", "pcm16le"}}}});
    append_request(requests, {{"id", "chunk"}, {"op", "stream_chunk"}, {"audio", "AAAAQACA"}});
    append_request(requests, {{"id", "reset"}, {"op", "stream_reset"}});
    append_request(
        requests,
        {{"id", "restart"}, {"op", "stream_start"}, {"config", {{"sample_rate_hz", 16000}}}});
    append_request(requests, {{"id", "finish"}, {"op", "stream_finish"}});
    append_request(requests, {{"id", "stop"}, {"op", "shutdown"}});

    FakeTask task;
    int status = -1;
    const auto messages = run(task, requests.str(), &status);
    check(status == 0 && messages.size() == 9, "full worker lifecycle completes");
    if (messages.size() != 9)
        return;
    check(messages[1]["result"].value("text", "") == "generated: summarize this" &&
              messages[1]["result"]["token_ids"] == Json::array({8, 9}) &&
              messages[1]["result"].value("completion_tokens", 0) == 2,
          "generation result preserves public fields");
    check(task.generate_calls == 1 && task.last_generate_config.max_new_tokens == 12 &&
              std::abs(task.last_generate_config.temperature - 0.25F) < 1e-6F &&
              task.last_generate_config.top_k == 7 && task.last_generate_config.use_chat_template &&
              !task.last_generate_config.enable_thinking,
          "generation config maps to ITextGeneration");
    check(messages[2]["result"] == Json{{"supported", true}}, "stream capability probe succeeds");
    check(messages[3]["result"] == Json::object() &&
              messages[4]["result"] == Json{{"text", "partial transcript"}} &&
              messages[5]["result"] == Json::object() &&
              messages[7]["result"] == Json{{"text", "final transcript"}},
          "stream lifecycle returns canonical payloads");
    check(task.create_stream_calls == 3 && task.stream_history.size() == 3,
          "probe and each stream use independent native state");
    if (task.stream_history.size() == 3) {
        const auto& samples = task.stream_history[1]->last_samples;
        check(samples.size() == 3 && std::abs(samples[0]) < 1e-6F &&
                  std::abs(samples[1] - 0.5F) < 1e-6F && std::abs(samples[2] + 1.0F) < 1e-6F,
              "stream PCM16 base64 is decoded exactly once");
        check(task.stream_history[0]->reset_calls == 1 &&
                  task.stream_history[1]->reset_calls == 1 &&
                  task.stream_history[2]->finish_calls == 1,
              "stream ownership terminates by probe, reset, or finish");
    }
}

void test_server_owned_wav_decode() {
    TempDir temporary;
    const auto pcm_path = temporary.path / "pcm.wav";
    write_wav<std::int16_t>(
        pcm_path, {16384, 8192, 0, -8192, 0, 8192, 16384, 24576, -16384, 0, 16384, 0}, 1, 4, 22050);
    const auto float_path = temporary.path / "float.wav";
    write_wav<float>(float_path, {0.25F, -0.5F}, 3, 1, 16000);

    std::ostringstream requests;
    append_request(requests, {{"id", "pcm"},
                              {"op", "transcribe"},
                              {"audio_path", pcm_path.string()},
                              {"config", {{"language", "fr"}}}});
    append_request(requests,
                   {{"id", "float"}, {"op", "transcribe"}, {"audio_path", float_path.string()}});
    append_request(requests, {{"id", "stop"}, {"op", "shutdown"}});
    FakeTask task;
    const auto messages = run(task, requests.str());
    check(messages.size() == 4 && messages[1].value("ok", false) && messages[2].value("ok", false),
          "PCM16 and float32 WAV requests succeed");
    check(task.transcribe_calls == 2 && task.last_transcription_samples.size() == 2 &&
              std::abs(task.last_transcription_samples[0] - 0.25F) < 1e-6F &&
              std::abs(task.last_transcription_samples[1] + 0.5F) < 1e-6F,
          "server-owned float32 WAV decoder preserves samples");

    std::ostringstream pcm_only;
    append_request(pcm_only,
                   {{"id", "pcm"}, {"op", "transcribe"}, {"audio_path", pcm_path.string()}});
    append_request(pcm_only, {{"id", "stop"}, {"op", "shutdown"}});
    FakeTask pcm_task;
    (void)run(pcm_task, pcm_only.str());
    check(pcm_task.last_transcription_samples.size() == 3 &&
              std::abs(pcm_task.last_transcription_samples[0] - 0.125F) < 1e-6F &&
              std::abs(pcm_task.last_transcription_samples[1] - 0.375F) < 1e-6F &&
              std::abs(pcm_task.last_transcription_samples[2]) < 1e-6F &&
              pcm_task.last_transcription_config.input_sample_rate == 22050,
          "server-owned PCM16 WAV decoder downmixes and preserves sample rate");
}

void test_protocol_rejects_edges_without_poisoning_task() {
    TempDir temporary;
    const auto invalid_path = temporary.path / "invalid.wav";
    {
        std::ofstream output(invalid_path, std::ios::binary);
        output << "not a wav";
    }
    const auto missing_path = temporary.path / "missing.wav";

    std::ostringstream requests;
    requests << "not-json\n";
    append_request(requests, {{"id", 7}, {"op", "generate"}, {"prompt", "bad id"}});
    append_request(requests, {{"id", "bad-config"},
                              {"op", "generate"},
                              {"prompt", "extra"},
                              {"config", {{"num_samples", 2}}}});
    append_request(
        requests, {{"id", "bad-wav"}, {"op", "transcribe"}, {"audio_path", invalid_path.string()}});
    append_request(
        requests,
        {{"id", "missing-wav"}, {"op", "transcribe"}, {"audio_path", missing_path.string()}});
    append_request(requests, {{"id", "start"}, {"op", "stream_start"}});
    append_request(requests, {{"id", "duplicate"}, {"op", "stream_start"}});
    append_request(requests, {{"id", "bad-audio"}, {"op", "stream_chunk"}, {"audio", "%%%"}});
    append_request(requests, {{"id", "reset"}, {"op", "stream_reset"}});
    append_request(requests, {{"id", "healthy"}, {"op", "generate"}, {"prompt", "alive"}});
    append_request(requests, {{"id", "stop"}, {"op", "shutdown"}});

    FakeTask task;
    std::ostringstream diagnostics;
    auto* previous = std::cerr.rdbuf(diagnostics.rdbuf());
    const auto messages = run(task, requests.str());
    std::cerr.rdbuf(previous);
    check(messages.size() == 12, "each malformed request receives one response");
    if (messages.size() != 12)
        return;
    check(messages[1]["id"].is_null() &&
              messages[1]["error"].value("type", "") == "invalid_request_error" &&
              messages[2]["id"].is_null(),
          "malformed JSON and ids stay structured");
    check(messages[3]["error"].value("message", "").find("config.num_samples") != std::string::npos,
          "noncanonical generation fields fail closed");
    check(messages[4]["error"].value("code", "") == "unsupported_media_type" &&
              messages[4]["error"].value("param", "") == "file",
          "malformed WAV is a stable media error");
    check(messages[5]["error"].value("type", "") == "runtime_error" &&
              messages[5]["error"].value("message", "") == "native worker operation failed" &&
              messages[5].dump().find(missing_path.string()) == std::string::npos,
          "I/O paths remain private runtime details");
    check(!messages[7].value("ok", true) && !messages[8].value("ok", true) &&
              messages[9].value("ok", false) && messages[10].value("ok", false),
          "state and base64 errors do not poison later canonical requests");
    check(task.generate_calls == 1, "only canonical generation reaches the task");
    check(diagnostics.str().find("cannot open uploaded audio file") != std::string::npos,
          "private I/O diagnostic remains on stderr");
}

void test_runtime_failures_are_redacted_and_worker_survives() {
    for (int failure_kind = 0; failure_kind < 3; ++failure_kind) {
        std::ostringstream requests;
        append_request(requests, {{"id", "failure"}, {"op", "generate"}, {"prompt", "private"}});
        append_request(requests, {{"id", "stop"}, {"op", "shutdown"}});
        FakeTask task;
        task.generate_runtime_error = failure_kind == 0;
        task.generate_invalid_argument = failure_kind == 1;
        task.generate_json_exception = failure_kind == 2;
        std::istringstream input(requests.str());
        std::ostringstream output;
        std::ostringstream diagnostics;
        auto* previous = std::cerr.rdbuf(diagnostics.rdbuf());
        const int status = trtmc::serve::run_worker_protocol(task, input, output);
        std::cerr.rdbuf(previous);
        const auto messages = parse_output_lines(output.str());
        check(status == 0 && messages.size() == 3 &&
                  messages[1]["error"].value("type", "") == "runtime_error" &&
                  messages[1]["error"].value("message", "") == "native worker operation failed",
              "provider exception is a generic runtime error");
        const bool private_detail_logged =
            failure_kind == 2 ? diagnostics.str().find("json.exception") != std::string::npos
                              : diagnostics.str().find("sensitive") != std::string::npos;
        check(output.str().find("sensitive") == std::string::npos && private_detail_logged,
              "provider diagnostics remain stderr-only");
    }
}

void test_invalid_utf8_and_oversized_records_are_bounded() {
    std::ostringstream utf8_requests;
    append_request(utf8_requests, {{"id", "generate"}, {"op", "generate"}, {"prompt", "text"}});
    append_request(utf8_requests, {{"id", "stop"}, {"op", "shutdown"}});
    FakeTask utf8_task;
    utf8_task.generate_invalid_utf8 = true;
    std::istringstream utf8_input(utf8_requests.str());
    std::ostringstream utf8_output;
    check(trtmc::serve::run_worker_protocol(utf8_task, utf8_input, utf8_output) == 0,
          "invalid UTF-8 result does not stop worker");
    check(utf8_output.str().find(static_cast<char>(0xFF)) == std::string::npos &&
              utf8_output.str().find("\xEF\xBF\xBD") != std::string::npos,
          "invalid UTF-8 is replaced on JSONL output");

    constexpr std::size_t limit = 16U * 1024U * 1024U;
    std::ostringstream oversized;
    oversized << std::string(limit + 4096U, 'x') << '\n';
    append_request(oversized, {{"id", "healthy"}, {"op", "generate"}, {"prompt", "alive"}});
    append_request(oversized, {{"id", "stop"}, {"op", "shutdown"}});
    FakeTask task;
    const auto messages = run(task, oversized.str());
    check(messages.size() == 4 && messages[1]["id"].is_null() &&
              messages[1]["error"].value("message", "") ==
                  "request exceeds the 16 MiB JSONL limit" &&
              messages[2].value("ok", false) && task.generate_calls == 1,
          "oversized record is discarded before the next request");
}

void test_entrypoint_requires_explicit_runtime_and_keeps_stdout_clean() {
    TempDir temporary;
    const std::string missing = (temporary.path / "private.bundle").string();
    trtmc::server::NativeWorkerOptions options;
    options.bundle_path = missing;
    options.runtime_root = temporary.path.string();
    std::ostringstream protocol;
    std::ostringstream diagnostics;
    auto* previous_stdout = std::cout.rdbuf(protocol.rdbuf());
    auto* previous_stderr = std::cerr.rdbuf(diagnostics.rdbuf());
    const int status = trtmc::server::run_native_worker(options);
    std::cout.rdbuf(previous_stdout);
    std::cerr.rdbuf(previous_stderr);
    check(status == EXIT_FAILURE && protocol.str().empty(),
          "startup failure never enters protocol stdout");
    check(diagnostics.str().find("Error: native worker failed:") != std::string::npos &&
              diagnostics.str().find(missing) != std::string::npos,
          "startup detail remains on private stderr");

    char command[] = "bundle.bundle";
    char* argv[] = {command};
    std::ostringstream parse_diagnostics;
    previous_stderr = std::cerr.rdbuf(parse_diagnostics.rdbuf());
    const int parse_status = trtmc::server::run_native_worker(1, argv);
    std::cerr.rdbuf(previous_stderr);
    check(parse_status == EXIT_FAILURE &&
              parse_diagnostics.str().find("requires --runtime-root") != std::string::npos,
          "worker CLI requires explicit runtime placement");
}

} // namespace

int main() {
    try {
        test_ready_schema_uses_actual_task_capabilities();
        test_generation_and_stream_lifecycle();
        test_server_owned_wav_decode();
        test_protocol_rejects_edges_without_poisoning_task();
        test_runtime_failures_are_redacted_and_worker_survives();
        test_invalid_utf8_and_oversized_records_are_bounded();
        test_entrypoint_requires_explicit_runtime_and_keeps_stdout_clean();
    } catch (const std::exception& error) {
        std::cerr << "Unhandled test exception: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0)
        std::cerr << failures << " serve worker test(s) failed\n";
    return failures == 0 ? 0 : 1;
}
