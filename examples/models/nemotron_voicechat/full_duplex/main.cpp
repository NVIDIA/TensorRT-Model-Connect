/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "playback_queue.h"
#include "trtmc/speech_session.h"

#include <algorithm>
#include <alsa/asoundlib.h>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using trtmc::SpeechSessionEvent;
using trtmc::SpeechSessionEventKind;
using trtmc::examples::voicechat::float_to_pcm16;
using trtmc::examples::voicechat::pcm16_to_float;
using trtmc::examples::voicechat::PlaybackQueue;
using trtmc::examples::voicechat::PlaybackQueueItemKind;

constexpr int kCaptureChunkMs = 20;
constexpr int kCaptureWaitMs = 50;
constexpr int kPlaybackQueueSeconds = 4;
constexpr int kEventWaitMs = 50;

volatile std::sig_atomic_t g_signal_requested = 0;

void signal_handler(int) {
    g_signal_requested = 1;
}

struct Options {
    std::string bundle_path;
    std::string capture_device{"default"};
    std::string playback_device{"default"};
    std::string system_prompt;
    int input_rate{16000};
    int output_rate{48000};
    int latency_ms{80};
    int seed{0};
    bool help{false};
    bool list_devices{false};
};

class CliError : public std::invalid_argument {
  public:
    using std::invalid_argument::invalid_argument;
};

void print_usage(std::ostream& output, const char* program) {
    output << "Usage: " << program << " [OPTIONS] MODEL.bundle\n\n"
           << "Run a local full-duplex Nemotron VoiceChat session with ALSA.\n\n"
           << "Options:\n"
           << "  --capture-device NAME   ALSA capture PCM (default: default)\n"
           << "  --playback-device NAME  ALSA playback PCM (default: default)\n"
           << "  --input-rate HZ         Capture/session rate (default: 16000)\n"
           << "  --output-rate HZ        Session/playback rate (default: 48000)\n"
           << "  --latency-ms MS         ALSA target latency (default: 80)\n"
           << "  --seed N                Deterministic speech seed (default: 0)\n"
           << "  --system-prompt TEXT    Optional system prompt\n"
           << "  --list-devices          List ALSA PCM names and exit\n"
           << "  -h, --help              Show this help and exit\n\n"
           << "Use a headset to prevent speaker echo from triggering barge-in.\n";
}

int parse_integer(const std::string& value, const char* option, int minimum, int maximum) {
    std::size_t consumed = 0;
    long long parsed = 0;
    try {
        parsed = std::stoll(value, &consumed, 10);
    } catch (const std::exception&) {
        throw CliError(std::string(option) + " requires an integer");
    }
    if (consumed != value.size() || parsed < minimum || parsed > maximum)
        throw CliError(std::string(option) + " is outside its supported range");
    return static_cast<int>(parsed);
}

std::string take_option_value(int& index, int argc, char** argv, const char* option) {
    if (++index >= argc)
        throw CliError(std::string(option) + " requires a value");
    return argv[index];
}

bool parse_value_option(const std::string& argument, int& index, int argc, char** argv,
                        Options& options) {
    if (argument == "--capture-device") {
        options.capture_device = take_option_value(index, argc, argv, "--capture-device");
        return true;
    }
    if (argument == "--playback-device") {
        options.playback_device = take_option_value(index, argc, argv, "--playback-device");
        return true;
    }
    if (argument == "--input-rate") {
        options.input_rate = parse_integer(take_option_value(index, argc, argv, "--input-rate"),
                                           "--input-rate", 8000, 192000);
        return true;
    }
    if (argument == "--output-rate") {
        options.output_rate = parse_integer(take_option_value(index, argc, argv, "--output-rate"),
                                            "--output-rate", 8000, 192000);
        return true;
    }
    if (argument == "--latency-ms") {
        options.latency_ms = parse_integer(take_option_value(index, argc, argv, "--latency-ms"),
                                           "--latency-ms", 10, 1000);
        return true;
    }
    if (argument == "--seed") {
        options.seed = parse_integer(take_option_value(index, argc, argv, "--seed"), "--seed", 0,
                                     std::numeric_limits<int>::max());
        return true;
    }
    if (argument == "--system-prompt") {
        options.system_prompt = take_option_value(index, argc, argv, "--system-prompt");
        return true;
    }
    return false;
}

void set_bundle_path(const std::string& argument, Options& options) {
    if (!options.bundle_path.empty())
        throw CliError("only one model bundle may be specified");
    options.bundle_path = argument;
}

bool parse_flag(const std::string& argument, Options& options) {
    if (argument == "-h" || argument == "--help") {
        options.help = true;
        return true;
    }
    if (argument == "--list-devices") {
        options.list_devices = true;
        return true;
    }
    return false;
}

void validate_options(const Options& options) {
    if (!options.help && !options.list_devices && options.bundle_path.empty())
        throw CliError("MODEL.bundle is required");
}

Options parse_options(int argc, char** argv) {
    Options options;
    bool positional_only = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (positional_only) {
            set_bundle_path(argument, options);
            continue;
        }
        if (argument == "--") {
            positional_only = true;
            continue;
        }
        if (parse_flag(argument, options))
            continue;
        if (parse_value_option(argument, index, argc, argv, options))
            continue;
        if (!argument.empty() && argument.front() == '-')
            throw CliError("unknown option: " + argument);
        set_bundle_path(argument, options);
    }
    validate_options(options);
    return options;
}

[[noreturn]] void throw_alsa_error(const std::string& operation, int error) {
    throw std::runtime_error(operation + ": " + snd_strerror(error));
}

class AlsaPcm {
  public:
    AlsaPcm(const std::string& device, snd_pcm_stream_t stream, unsigned int sample_rate,
            unsigned int latency_ms)
        : stream_(stream) {
        const int open_mode = stream == SND_PCM_STREAM_CAPTURE ? SND_PCM_NONBLOCK : 0;
        int status = snd_pcm_open(&handle_, device.c_str(), stream, open_mode);
        if (status < 0)
            throw_alsa_error("cannot open ALSA device '" + device + "'", status);
        status = snd_pcm_set_params(handle_, SND_PCM_FORMAT_S16_LE, SND_PCM_ACCESS_RW_INTERLEAVED,
                                    1, sample_rate, 1, latency_ms * 1000U);
        if (status < 0) {
            snd_pcm_close(handle_);
            handle_ = nullptr;
            throw_alsa_error("cannot configure ALSA device '" + device + "'", status);
        }
    }

    ~AlsaPcm() {
        if (handle_ != nullptr)
            snd_pcm_close(handle_);
    }

    AlsaPcm(const AlsaPcm&) = delete;
    AlsaPcm& operator=(const AlsaPcm&) = delete;

    std::size_t read_frames(std::int16_t* samples, std::size_t capacity) {
        while (true) {
            const auto result = snd_pcm_readi(handle_, samples, capacity);
            if (result >= 0)
                return static_cast<std::size_t>(result);
            if (result == -EINTR)
                return 0;
            if (result == -EAGAIN) {
                const int wait_status = snd_pcm_wait(handle_, kCaptureWaitMs);
                if (wait_status > 0)
                    continue;
                if (wait_status == 0 || wait_status == -EINTR)
                    return 0;
                recover_or_throw(wait_status, "ALSA capture wait failed");
                continue;
            }
            recover_or_throw(static_cast<int>(result), "ALSA capture failed");
        }
    }

    std::size_t write_frames(const std::int16_t* samples, std::size_t count) {
        while (true) {
            const auto result = snd_pcm_writei(handle_, samples, count);
            if (result >= 0)
                return static_cast<std::size_t>(result);
            if (result == -EINTR)
                return 0;
            recover_or_throw(static_cast<int>(result), "ALSA playback failed");
        }
    }

    void flush_playback() {
        if (stream_ != SND_PCM_STREAM_PLAYBACK)
            throw std::logic_error("cannot flush a capture PCM");
        const int drop_status = snd_pcm_drop(handle_);
        if (drop_status < 0 && drop_status != -EBADFD)
            throw_alsa_error("cannot drop stale ALSA playback", drop_status);
        const int prepare_status = snd_pcm_prepare(handle_);
        if (prepare_status < 0)
            throw_alsa_error("cannot prepare ALSA playback after flush", prepare_status);
    }

    void discard_noexcept() noexcept {
        if (handle_ != nullptr && stream_ == SND_PCM_STREAM_PLAYBACK)
            (void)snd_pcm_drop(handle_);
    }

  private:
    void recover_or_throw(int error, const char* operation) {
        // alsa-lib may wait indefinitely while recovering a suspended device.
        // Fail the session so Ctrl-C/docker stop keeps a bounded shutdown path.
        if (error == -ESTRPIPE)
            throw_alsa_error(operation, error);
        const int recovered = snd_pcm_recover(handle_, error, 1);
        if (recovered < 0)
            throw_alsa_error(operation, recovered);
    }

    snd_pcm_t* handle_{nullptr};
    snd_pcm_stream_t stream_;
};

class RunState {
  public:
    bool stopping() const noexcept { return stopping_.load(); }

    void request_stop() noexcept { stopping_.store(true); }

    void fail(std::exception_ptr failure) noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!failure_)
                failure_ = std::move(failure);
        }
        request_stop();
    }

    void rethrow_if_failed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (failure_)
            std::rethrow_exception(failure_);
    }

  private:
    std::atomic<bool> stopping_{false};
    mutable std::mutex mutex_;
    std::exception_ptr failure_;
};

void list_alsa_devices() {
    void** hints = nullptr;
    const int status = snd_device_name_hint(-1, "pcm", &hints);
    if (status < 0)
        throw_alsa_error("cannot enumerate ALSA PCM devices", status);
    std::cout << "NAME\tDIRECTION\tDESCRIPTION\n";
    for (void** current = hints; current != nullptr && *current != nullptr; ++current) {
        char* name = snd_device_name_get_hint(*current, "NAME");
        char* direction = snd_device_name_get_hint(*current, "IOID");
        char* description = snd_device_name_get_hint(*current, "DESC");
        if (name != nullptr) {
            std::string clean_description = description == nullptr ? "" : description;
            std::replace(clean_description.begin(), clean_description.end(), '\n', ' ');
            std::cout << name << '\t' << (direction == nullptr ? "Input/Output" : direction) << '\t'
                      << clean_description << '\n';
        }
        std::free(name);
        std::free(direction);
        std::free(description);
    }
    snd_device_name_free_hint(hints);
}

void capture_loop(AlsaPcm& capture, trtmc::ISpeechSession& session, int sample_rate,
                  RunState& state, PlaybackQueue& playback_queue) noexcept {
    try {
        const auto chunk_samples =
            static_cast<std::size_t>(std::max(1, sample_rate * kCaptureChunkMs / 1000));
        std::vector<std::int16_t> pcm(chunk_samples);
        std::vector<float> audio(chunk_samples);
        while (!state.stopping()) {
            const auto count = capture.read_frames(pcm.data(), pcm.size());
            if (count == 0 || state.stopping())
                continue;
            std::transform(pcm.begin(), pcm.begin() + static_cast<std::ptrdiff_t>(count),
                           audio.begin(), pcm16_to_float);
            session.append_audio(audio.data(), static_cast<std::int32_t>(count));
        }
    } catch (...) {
        if (!state.stopping())
            state.fail(std::current_exception());
        playback_queue.stop();
    }
}

void playback_loop(AlsaPcm& playback, PlaybackQueue& queue, int sample_rate,
                   RunState& state) noexcept {
    try {
        const auto write_chunk =
            static_cast<std::size_t>(std::max(1, sample_rate * kCaptureChunkMs / 1000));
        while (true) {
            auto item = queue.wait_pop();
            if (item.kind == PlaybackQueueItemKind::kStopped) {
                playback.discard_noexcept();
                return;
            }
            if (item.kind == PlaybackQueueItemKind::kFlush) {
                playback.flush_playback();
                continue;
            }

            std::size_t offset = 0;
            while (offset < item.samples.size() && !state.stopping() &&
                   queue.generation_is_current(item.generation)) {
                const auto count = std::min(write_chunk, item.samples.size() - offset);
                const auto written = playback.write_frames(item.samples.data() + offset, count);
                offset += written;
            }
        }
    } catch (...) {
        if (!state.stopping())
            state.fail(std::current_exception());
        queue.stop();
    }
}

struct SessionThreads {
    std::thread playback;
    std::thread capture;
};

SessionThreads start_session_threads(AlsaPcm& playback, AlsaPcm& capture,
                                     PlaybackQueue& playback_queue, trtmc::ISpeechSession& session,
                                     int output_rate, int input_rate, RunState& state) {
    SessionThreads threads;
    try {
        threads.playback = std::thread(playback_loop, std::ref(playback), std::ref(playback_queue),
                                       output_rate, std::ref(state));
        threads.capture = std::thread(capture_loop, std::ref(capture), std::ref(session),
                                      input_rate, std::ref(state), std::ref(playback_queue));
    } catch (...) {
        state.request_stop();
        playback_queue.stop();
        if (threads.capture.joinable())
            threads.capture.join();
        if (threads.playback.joinable())
            threads.playback.join();
        throw;
    }
    return threads;
}

class TranscriptPrinter {
  public:
    void agent_text(const SpeechSessionEvent& event) {
        if (event.epoch != agent_epoch_) {
            finish_agent_line();
            agent_epoch_ = event.epoch;
            saw_agent_delta_ = false;
        }
        if (!agent_line_open_) {
            std::cout << "agent> " << std::flush;
            agent_line_open_ = true;
        }
        if (!event.is_final) {
            std::cout << event.text << std::flush;
            saw_agent_delta_ = true;
        } else {
            if (!saw_agent_delta_)
                std::cout << event.text;
            finish_agent_line();
        }
    }

    void user_text(const SpeechSessionEvent& event) {
        if (!event.is_final || event.text.empty())
            return;
        finish_agent_line();
        std::cout << "user> " << event.text << '\n';
    }

    void status(const std::string& text) {
        finish_agent_line();
        std::cout << '[' << text << "]\n";
    }

    void finish_agent_line() {
        if (agent_line_open_)
            std::cout << '\n';
        agent_line_open_ = false;
        saw_agent_delta_ = false;
    }

  private:
    std::uint64_t agent_epoch_{0};
    bool agent_line_open_{false};
    bool saw_agent_delta_{false};
};

void enqueue_agent_audio(const SpeechSessionEvent& event, int expected_sample_rate,
                         PlaybackQueue& queue) {
    if (event.sample_rate != expected_sample_rate)
        throw std::runtime_error("speech session changed its output sample rate");
    std::vector<std::int16_t> pcm;
    pcm.reserve(event.audio_samples.size());
    std::transform(event.audio_samples.begin(), event.audio_samples.end(), std::back_inserter(pcm),
                   float_to_pcm16);
    if (!queue.try_push(std::move(pcm)))
        throw std::runtime_error("playback queue exceeded its four-second bound");
}

bool consume_payload_event(const SpeechSessionEvent& event, int output_rate, PlaybackQueue& queue,
                           TranscriptPrinter& printer) {
    switch (event.kind) {
    case SpeechSessionEventKind::kAgentAudio:
        enqueue_agent_audio(event, output_rate, queue);
        return true;
    case SpeechSessionEventKind::kAgentText:
        printer.agent_text(event);
        return true;
    case SpeechSessionEventKind::kUserTranscript:
        printer.user_text(event);
        return true;
    default:
        return false;
    }
}

void consume_lifecycle_event(const SpeechSessionEvent& event, PlaybackQueue& queue,
                             TranscriptPrinter& printer, RunState& state) {
    switch (event.kind) {
    case SpeechSessionEventKind::kYielded:
        (void)queue.request_flush();
        printer.status(event.text.empty() ? "yielded" : "yielded: " + event.text);
        break;
    case SpeechSessionEventKind::kCancelled:
        (void)queue.request_flush();
        printer.status("cancelled");
        state.request_stop();
        break;
    case SpeechSessionEventKind::kReset:
        (void)queue.request_flush();
        printer.status("reset");
        break;
    case SpeechSessionEventKind::kError:
        (void)queue.request_flush();
        throw std::runtime_error(event.text.empty() ? "speech session failed" : event.text);
    case SpeechSessionEventKind::kInputFinished:
        state.request_stop();
        break;
    case SpeechSessionEventKind::kTurnFinished:
        printer.finish_agent_line();
        break;
    default:
        break;
    }
}

void consume_event(const SpeechSessionEvent& event, int output_rate, PlaybackQueue& queue,
                   TranscriptPrinter& printer, RunState& state) {
    if (!consume_payload_event(event, output_rate, queue, printer))
        consume_lifecycle_event(event, queue, printer, state);
}

int run(const Options& options) {
    // Fail on an unavailable host audio device before loading the large model.
    AlsaPcm capture(options.capture_device, SND_PCM_STREAM_CAPTURE,
                    static_cast<unsigned int>(options.input_rate),
                    static_cast<unsigned int>(options.latency_ms));
    AlsaPcm playback(options.playback_device, SND_PCM_STREAM_PLAYBACK,
                     static_cast<unsigned int>(options.output_rate),
                     static_cast<unsigned int>(options.latency_ms));

    auto pipeline = trtmc::load(options.bundle_path);
    auto* provider = dynamic_cast<trtmc::ISpeechSessionProvider*>(pipeline.get());
    if (provider == nullptr)
        throw std::runtime_error("bundle does not support persistent speech sessions");

    trtmc::SpeechSessionConfig config;
    config.input_sample_rate = options.input_rate;
    config.output_sample_rate = options.output_rate;
    config.system_prompt = options.system_prompt;
    config.emit_agent_audio = true;
    config.emit_agent_text = true;
    config.emit_user_transcript = true;
    config.enable_barge_in = true;
    config.seed = options.seed;
    auto session = provider->create_speech_session(config);
    const auto actual_config = session->config();
    if (actual_config.output_sample_rate <= 0)
        throw std::runtime_error("speech session returned an invalid output sample rate");

    const auto playback_capacity = static_cast<std::size_t>(actual_config.output_sample_rate) *
                                   static_cast<std::size_t>(kPlaybackQueueSeconds);
    PlaybackQueue playback_queue(playback_capacity);
    RunState state;
    TranscriptPrinter printer;

    auto threads = start_session_threads(playback, capture, playback_queue, *session,
                                         actual_config.output_sample_rate,
                                         actual_config.input_sample_rate, state);

    std::cout << "Listening on '" << options.capture_device << "'; playing on '"
              << options.playback_device << "'. Press Ctrl-C to stop.\n";
    try {
        while (!state.stopping() && g_signal_requested == 0) {
            for (const auto& event : session->wait_events(kEventWaitMs))
                consume_event(event, actual_config.output_sample_rate, playback_queue, printer,
                              state);
        }
    } catch (...) {
        state.fail(std::current_exception());
    }

    state.request_stop();
    playback_queue.stop();
    try {
        session->cancel();
    } catch (...) {
        state.fail(std::current_exception());
    }
    threads.capture.join();
    threads.playback.join();
    printer.finish_agent_line();
    state.rethrow_if_failed();
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        if (options.help) {
            print_usage(std::cout, argv[0]);
            return EXIT_SUCCESS;
        }
        if (options.list_devices) {
            list_alsa_devices();
            return EXIT_SUCCESS;
        }
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);
        return run(options);
    } catch (const CliError& error) {
        std::cerr << "Error: " << error.what() << "\n\n";
        print_usage(std::cerr, argv[0]);
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
