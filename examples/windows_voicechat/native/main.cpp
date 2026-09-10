/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "audio_protocol.h"
#include "stdio_transport.h"
#include "trtmc/bundle.h"
#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <trtmc/nemotron_voicechat/live_control.h>

namespace {

using namespace trtmc::examples::windows_voicechat;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

volatile std::sig_atomic_t signal_requested = 0;
void on_signal(int) {
    signal_requested = 1;
}

struct Options {
    std::string bundle;
    std::string runtime_root;
    std::string runtime_cache;
    std::string system_prompt;
    int seed{0};
    bool help{false};
};

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " --bundle MODEL.bundle --runtime-root DIR [OPTIONS]\n\n"
              << "Native Nemotron VoiceChat bridge for the Windows desktop app.\n"
              << "Requires a nemotron_voicechat bundle built with backend trt_rtx.\n"
              << "Reads newline JSON commands on stdin; writes newline JSON events on stdout.\n"
              << "Runtime diagnostics go to stderr. Audio is mono float32 PCM.\n\n"
              << "  --bundle PATH         TensorRT-RTX VoiceChat bundle (required)\n"
              << "  --runtime-root DIR    Directory containing TRTMC runtime DLLs (required)\n"
              << "  --runtime-cache PATH  Optional TensorRT-RTX runtime cache\n"
              << "  --system-prompt TEXT  Optional conversation system prompt\n"
              << "  --seed N              Nonnegative deterministic seed (default: 0)\n"
              << "  -h, --help            Show this help without loading a model\n";
}

Options parse_options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--help" || option == "-h") {
            result.help = true;
            continue;
        }
        if (index + 1 == argc)
            throw CommandError(option + " requires a value");
        const std::string value = argv[++index];
        if (option == "--bundle")
            result.bundle = value;
        else if (option == "--runtime-root")
            result.runtime_root = value;
        else if (option == "--runtime-cache")
            result.runtime_cache = value;
        else if (option == "--system-prompt")
            result.system_prompt = value;
        else if (option == "--seed") {
            try {
                std::size_t consumed = 0;
                const auto seed = std::stoll(value, &consumed);
                if (consumed != value.size() || seed < 0 || seed > std::numeric_limits<int>::max())
                    throw CommandError("--seed requires an integer between 0 and 2147483647");
                result.seed = static_cast<int>(seed);
            } catch (const std::exception&) {
                throw CommandError("--seed requires an integer between 0 and 2147483647");
            }
        } else
            throw CommandError("unknown option: " + option);
    }
    if (!result.help && (result.bundle.empty() || result.runtime_root.empty()))
        throw CommandError("--bundle and --runtime-root are required");
    return result;
}

void emit(ProtocolOutput& output, const Json& value) {
    output.write(value.dump());
}

void emit_error(ProtocolOutput& output, const std::string& message, bool fatal) {
    emit(output, {{"type", "error"}, {"message", message}, {"fatal", fatal}});
}

const char* event_kind(trtmc::SpeechSessionEventKind kind) {
    using Kind = trtmc::SpeechSessionEventKind;
    switch (kind) {
    case Kind::kAgentAudio:
        return "agent_audio";
    case Kind::kAgentText:
        return "agent_text";
    case Kind::kUserTranscript:
        return "user_transcript";
    case Kind::kTurnStarted:
        return "turn_started";
    case Kind::kTurnFinished:
        return "turn_finished";
    case Kind::kYielded:
        return "yielded";
    case Kind::kCancelled:
        return "cancelled";
    case Kind::kReset:
        return "reset";
    case Kind::kError:
        return "error";
    case Kind::kInputFinished:
        return "input_finished";
    case Kind::kUserSpeechStarted:
        return "user_speech_started";
    case Kind::kUserSpeechStopped:
        return "user_speech_stopped";
    case Kind::kFunctionCall:
        return "function_call";
    case Kind::kFunctionCallStarted:
        return "function_call_started";
    case Kind::kFunctionResponseFinished:
        return "function_response_finished";
    case Kind::kInputCleared:
        return "input_cleared";
    case Kind::kContextRolled:
        return "context_rolled";
    }
    throw std::runtime_error("speech session produced an unsupported event kind");
}

Json serialize_event(const trtmc::SpeechSessionEvent& event) {
    Json result{{"type", "event"},
                {"kind", event_kind(event.kind)},
                {"epoch", event.epoch},
                {"sequence", event.sequence},
                {"text", event.text},
                {"isFinal", event.is_final},
                {"sampleRate", event.sample_rate},
                {"mediaStartSample", event.media_start_sample},
                {"mediaEndSample", event.media_end_sample},
                {"frameIndex", event.frame_index}};
    if (event.kind == trtmc::SpeechSessionEventKind::kAgentAudio) {
        if (event.sample_rate != kOutputSampleRate)
            throw std::runtime_error("speech session changed its 48000 Hz output sample rate");
        result["encoding"] = "f32le";
        result["audio"] = encode_audio(event.audio_samples);
        result["sampleCount"] = event.audio_samples.size();
    }
    return result;
}

Json parse_command(const std::string& line) {
    try {
        auto command = Json::parse(line);
        if (!command.is_object() || !command.contains("type") || !command["type"].is_string())
            throw CommandError("command requires a string type field");
        return command;
    } catch (const Json::exception& error) {
        throw CommandError(std::string("invalid JSON command: ") + error.what());
    }
}

std::vector<float> command_audio(const Json& command) {
    if (!command.contains("sampleRate") || !command["sampleRate"].is_number_integer() ||
        command["sampleRate"] != kInputSampleRate)
        throw CommandError(
            "audio sampleRate must be 16000; resample microphone audio before sending");
    const bool has_samples = command.contains("samples");
    const bool has_audio = command.contains("audio");
    if (has_samples == has_audio)
        throw CommandError("audio requires exactly one of samples or audio");
    if (has_audio) {
        if (!command["audio"].is_string())
            throw CommandError("audio must be a base64 string of little-endian float32 samples");
        if (command.contains("encoding") && command["encoding"] != "f32le")
            throw CommandError("audio encoding must be f32le");
        return decode_audio(command["audio"].get_ref<const std::string&>());
    }
    const auto& values = command["samples"];
    if (!values.is_array() || values.empty() || values.size() > kMaxInputSamples)
        throw CommandError("samples must contain 1 to 16000 mono microphone samples");
    std::vector<float> samples;
    samples.reserve(values.size());
    for (const auto& value : values) {
        if (!value.is_number())
            throw CommandError("microphone samples must be numbers");
        const auto sample = value.get<double>();
        if (!std::isfinite(sample) || std::abs(sample) > 1.0)
            throw CommandError("microphone samples must be finite and between -1 and 1");
        samples.push_back(static_cast<float>(sample));
    }
    return samples;
}

class SessionRunner {
  public:
    SessionRunner(trtmc::ISpeechSession& session, ProtocolOutput& output)
        : session_(session), output_(output), worker_([this] { events_loop(); }) {}

    ~SessionRunner() {
        stopping_.store(true);
        try {
            session_.cancel();
        } catch (...) {
            // A primary inference failure is already reported by run().
        }
        if (worker_.joinable())
            worker_.join();
    }

    void run(CommandInput& input) {
        std::string line;
        while (!stopping_.load() && signal_requested == 0 && !input.eof()) {
            if (!input.next(line) || line.empty())
                continue;
            try {
                const auto command = parse_command(line);
                const auto type = command["type"].get<std::string>();
                if (type == "audio") {
                    if (input_finished_)
                        throw CommandError("audio input is finished; start a new session");
                    const auto samples = command_audio(command);
                    session_.append_audio(samples.data(),
                                          static_cast<std::int32_t>(samples.size()));
                } else if (type == "reset") {
                    // A barrier prevents a batch taken before reset from reaching
                    // the desktop after the new conversation starts.
                    std::lock_guard<std::mutex> lock(event_gate_);
                    emit(output_, {{"type", "flush"}, {"reason", "reset"}});
                    session_.reset();
                    input_finished_ = false;
                } else if (type == "interrupt") {
                    std::lock_guard<std::mutex> lock(event_gate_);
                    auto* live_control =
                        dynamic_cast<trtmc::INemotronVoiceChatLiveControl*>(&session_);
                    if (live_control == nullptr)
                        throw CommandError("update the Nemotron runtime to support continuous "
                                           "conversation refresh");
                    // Forget old dialogue while retaining the live acoustic
                    // frontend. Generic cancellation replays old generation;
                    // full session reset also reloads perception engines.
                    // Acknowledge only after the family worker's reset barrier.
                    live_control->reset_conversation_context();
                    input_finished_ = false;
                    emit(output_, {{"type", "flush"},
                                   {"reason", "interrupt"},
                                   {"interruptStatus", "context_reset"}});
                } else if (type == "finish") {
                    session_.finish_input();
                    input_finished_ = true;
                } else if (type == "stop") {
                    stopping_.store(true);
                } else if (type == "ping") {
                    emit(output_, {{"type", "pong"}});
                } else {
                    throw CommandError("unknown command type: " + type);
                }
            } catch (const CommandError& error) {
                emit_error(output_, error.what(), false);
            }
        }
        std::lock_guard<std::mutex> lock(failure_mutex_);
        if (failure_)
            std::rethrow_exception(failure_);
    }

  private:
    void events_loop() noexcept {
        try {
            while (!stopping_.load()) {
                // Keep event dequeue and publication on the same side of a
                // reset barrier. Poll with zero timeout while holding the gate.
                bool had_events = false;
                {
                    std::lock_guard<std::mutex> lock(event_gate_);
                    for (const auto& event : session_.take_events()) {
                        had_events = true;
                        emit(output_, serialize_event(event));
                        if (event.kind == trtmc::SpeechSessionEventKind::kError)
                            throw std::runtime_error(event.text.empty() ? "speech session failed"
                                                                        : event.text);
                        if (event.kind == trtmc::SpeechSessionEventKind::kCancelled ||
                            event.kind == trtmc::SpeechSessionEventKind::kInputFinished)
                            stopping_.store(true);
                    }
                }
                if (!had_events)
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(failure_mutex_);
                failure_ = std::current_exception();
            }
            stopping_.store(true);
        }
    }

    trtmc::ISpeechSession& session_;
    ProtocolOutput& output_;
    std::atomic<bool> stopping_{false};
    bool input_finished_{false};
    std::mutex event_gate_;
    std::mutex failure_mutex_;
    std::exception_ptr failure_;
    std::thread worker_;
};

int run(const Options& options, ProtocolOutput& output) {
    // Validate metadata before loading any backend DLL: there is deliberately
    // no TensorRT fallback and no backend selection inferred from the host.
    const auto info = trtmc::InspectBundle(options.bundle);
    if (info.family != "nemotron_voicechat")
        throw std::runtime_error("selected bundle family must be nemotron_voicechat");
    if (info.backend != "trt_rtx")
        throw std::runtime_error(
            "selected bundle must use TensorRT-RTX (trt_rtx); rebuild it with --backend trt_rtx");
    if (!std::filesystem::is_directory(std::filesystem::u8path(options.runtime_root)))
        throw std::runtime_error("runtime root is not an existing directory");
    CommandInput input;
    emit(output, {{"type", "loading"}, {"backend", "trt_rtx"}, {"family", info.family}});
    const auto started = Clock::now();
    auto task = trtmc::load_task(options.bundle, options.runtime_root, 0, options.runtime_cache);
    auto* provider = dynamic_cast<trtmc::ISpeechSessionProvider*>(task.get());
    if (provider == nullptr)
        throw std::runtime_error("selected bundle does not provide persistent speech sessions");
    trtmc::SpeechSessionConfig config;
    config.input_sample_rate = kInputSampleRate;
    config.output_sample_rate = kOutputSampleRate;
    config.system_prompt = options.system_prompt;
    config.emit_agent_audio = true;
    config.emit_agent_text = true;
    config.emit_user_transcript = true;
    config.enable_barge_in = true;
    config.seed = options.seed;
    auto session = provider->create_speech_session(config);
    if (!session)
        throw std::runtime_error("speech provider returned no session");
    const auto actual = session->config();
    if (actual.input_sample_rate != kInputSampleRate ||
        actual.output_sample_rate != kOutputSampleRate)
        throw std::runtime_error("speech session must support 16000 Hz input and 48000 Hz output");
    const auto load_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - started).count();
    emit(output, {{"type", "ready"},
                  {"backend", "trt_rtx"},
                  {"family", info.family},
                  {"inputSampleRate", actual.input_sample_rate},
                  {"outputSampleRate", actual.output_sample_rate},
                  {"loadTimeMs", load_ms},
                  {"protocolVersion", 1}});
    {
        SessionRunner runner(*session, output);
        runner.run(input);
    }
    session.reset();
    task.reset();
    emit(output, {{"type", "stopped"}});
    return EXIT_SUCCESS;
}

} // namespace

int bridge_main(int argc, char** argv) {
    std::unique_ptr<ProtocolOutput> output;
    try {
        const auto options = parse_options(argc, argv);
        if (options.help) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
#ifndef _WIN32
        std::signal(SIGPIPE, SIG_IGN);
#endif
        output = std::make_unique<ProtocolOutput>();
        return run(options, *output);
    } catch (const std::exception& error) {
        if (output) {
            try {
                emit_error(*output, error.what(), true);
            } catch (...) {
                // The parent may already have closed stdout during disconnect.
            }
        }
        std::cerr << "VoiceChat bridge: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    try {
        std::vector<std::string> arguments;
        arguments.reserve(static_cast<std::size_t>(argc));
        for (int index = 0; index < argc; ++index) {
            const int count = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[index], -1,
                                                  nullptr, 0, nullptr, nullptr);
            if (count <= 0)
                throw std::runtime_error("command line contains an invalid Unicode argument");
            std::string argument(static_cast<std::size_t>(count), '\0');
            if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[index], -1, argument.data(),
                                    count, nullptr, nullptr) != count)
                throw std::runtime_error("cannot decode a Unicode command line argument");
            argument.pop_back();
            arguments.push_back(std::move(argument));
        }
        std::vector<char*> pointers;
        pointers.reserve(arguments.size());
        for (auto& argument : arguments)
            pointers.push_back(argument.data());
        return bridge_main(argc, pointers.data());
    } catch (const std::exception& error) {
        std::cerr << "VoiceChat bridge: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
#else
int main(int argc, char** argv) {
    return bridge_main(argc, argv);
}
#endif
