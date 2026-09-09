/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/sdk_dispatch.h"
#include "trtmc/audio.hpp"
#include "trtmc/speech.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <nlohmann/json.hpp>

namespace trtmc::cli {
namespace {
using detail::has_option;
using detail::require_option;

std::optional<std::string> option(const Command& command, const char* key) {
    return has_option(command, key) ? std::optional<std::string>{command.options.at(key)}
                                    : std::nullopt;
}
AudioView audio_view(const io::LoadedAudio& audio) {
    return {{audio.samples.data(), audio.samples.size()},
            static_cast<std::uint32_t>(audio.sample_rate),
            static_cast<std::uint32_t>(audio.channels)};
}
bool translate(const Command& command) {
    return has_option(command, "--translate") &&
           detail::parse_bool(command.options.at("--translate"), "--translate");
}
bool translation_intent(const Command& command, const Model& model) {
    if (has_option(command, "--translate"))
        return translate(command);
    const auto primary = model.info().bundle_task;
    return has_option(command, "--target-language") || primary == SpeechTranslation::kTask ||
           primary == BatchSpeechTranslation::kTask;
}
void validate_translation(const Command& command, bool translation) {
    if (has_option(command, "--translate") && translate(command) != translation)
        throw std::invalid_argument("--translate conflicts with the selected speech Task");
    if (!translation && has_option(command, "--target-language"))
        throw std::invalid_argument("--target-language requires a speech translation Task");
}
Config transcription_config(const Command& command, const std::vector<ConfigField>& fields) {
    // Existing CLI spellings map to their existing option names, without
    // injecting model defaults or defining another parameter parser.
    Command normalized = command;
    for (const auto& alias :
         {std::pair{"--max-input-seconds", "--max-input-duration-seconds"},
          std::pair{"--segment-length-seconds", "--segment-duration-seconds"},
          std::pair{"--segment-min-seconds", "--segment-min-duration-seconds"}}) {
        const auto old = normalized.options.find(alias.first);
        if (old != normalized.options.end()) {
            const auto value = old->second;
            normalized.options.erase(old);
            if (!normalized.options.emplace(alias.second, value).second)
                throw std::invalid_argument("duplicate transcription option");
        }
    }
    return detail::task_config(
        normalized, fields, {"--input", "--source-language", "--target-language", "--translate"});
}

void write_audio(const AudioGenerationResult& result, const std::string& path,
                 std::ostream& output) {
    io::write_wav_interleaved(result.samples(), result.sample_rate(), result.channels(), path);
    detail::write_json(output, {{"output", path},
                                {"sample_rate", result.sample_rate()},
                                {"channels", result.channels()},
                                {"num_samples", result.samples().size()},
                                {"num_frames", result.frame_count()},
                                {"setup_ms", result.setup_ms()},
                                {"inference_ms", result.inference_ms()}});
}

void generate_audio(const Command& command, const Model& model, std::string_view id,
                    std::ostream& output) {
    const bool streaming = id == StreamingTextToSpeech::kTask;
    if (has_option(command, "--stream") &&
        detail::parse_bool(command.options.at("--stream"), "--stream") != streaming)
        throw std::invalid_argument("--stream conflicts with the selected audio Task");
    if (!streaming && has_option(command, "--chunk-frames"))
        throw std::invalid_argument("--chunk-frames requires streaming text-to-speech");
    const auto prompt = require_option(command, "--prompt");
    const auto path = require_option(command, "--output");
    if (streaming) {
        const auto task = model.task<StreamingTextToSpeech>();
        const auto config = detail::task_config(command, task.config_fields(),
                                                {"--prompt", "--output", "--stream", "--language"});
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        file.exceptions(std::ios::badbit | std::ios::failbit);
        std::uint64_t written = 0;
        SpeechAudioFormat format{};
        const auto summary = task.run(
            {prompt, option(command, "--language")},
            [&](const AudioView& chunk) {
                if (!chunk.sample_rate || chunk.channels == 0 || chunk.samples.empty())
                    throw std::runtime_error("streaming audio omitted its actual format/data");
                if (format.sample_rate &&
                    (format.sample_rate != *chunk.sample_rate || format.channels != chunk.channels))
                    throw std::runtime_error("streaming audio changed output format");
                format = {*chunk.sample_rate, chunk.channels};
                if (chunk.samples.size() >
                    static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) /
                        sizeof(float))
                    throw std::runtime_error("audio chunk is too large to write");
                for (const auto sample : chunk.samples)
                    if (!std::isfinite(sample))
                        throw std::runtime_error("generated audio contains a non-finite sample");
                file.write(reinterpret_cast<const char*>(chunk.samples.data()),
                           static_cast<std::streamsize>(chunk.samples.size() * sizeof(float)));
                file.flush(); // Direct callback backpressure; no worker or accumulating queue.
                if (written > std::numeric_limits<std::uint64_t>::max() - chunk.samples.size())
                    throw std::runtime_error("audio sample count overflow");
                written += chunk.samples.size();
            },
            config);
        file.close();
        if (summary.outcome != AudioDeliveryOutcome::Complete || !written ||
            written != summary.emitted_sample_count ||
            summary.output.sample_rate != format.sample_rate ||
            summary.output.channels != format.channels)
            throw std::runtime_error("streaming audio did not complete its reported output");
        detail::write_json(output, {{"output", path},
                                    {"format", "float32le"},
                                    {"sample_rate", format.sample_rate},
                                    {"channels", format.channels},
                                    {"num_samples", written},
                                    {"num_frames", summary.emitted_frame_count},
                                    {"setup_ms", summary.setup_ms},
                                    {"inference_ms", summary.inference_ms}});
    } else if (id == TextToSpeech::kTask) {
        const auto task = model.task<TextToSpeech>();
        const auto config = detail::task_config(command, task.config_fields(),
                                                {"--prompt", "--output", "--stream", "--language"});
        write_audio(task.run({prompt, option(command, "--language")}, config), path, output);
    } else {
        if (has_option(command, "--language"))
            throw std::invalid_argument("--language requires a text-to-speech Task");
        const auto task = model.task<TextToAudio>();
        const auto config = detail::task_config(command, task.config_fields(),
                                                {"--prompt", "--output", "--stream"});
        write_audio(task.run({prompt}, config), path, output);
    }
}

void transcribe_batch(const Command& command, const Model& model, std::string_view id,
                      std::ostream& output) {
    if (command.inputs.empty())
        throw std::invalid_argument("transcribe-batch requires at least one --input");
    const bool translation =
        id == BatchSpeechTranslation::kTask ||
        (id == MixedBatchSpeechToText::kTask && translation_intent(command, model));
    validate_translation(command, translation);
    std::vector<io::LoadedAudio> audio;
    audio.reserve(command.inputs.size());
    for (const auto& path : command.inputs)
        audio.push_back(io::read_wav_interleaved(path));
    const auto source = option(command, "--source-language");
    const auto target = option(command, "--target-language");
    auto emit = [&](const auto& results) {
        nlohmann::json items = nlohmann::json::array();
        for (std::size_t i = 0; i < results.size(); ++i)
            items.push_back(detail::text_json(results[i]));
        detail::write_json(output, {{"results", std::move(items)}});
    };
    if (id == MixedBatchSpeechToText::kTask) {
        const auto task = model.task<MixedBatchSpeechToText>();
        const auto config = transcription_config(command, task.config_fields());
        MixedBatchSpeechToTextRequest request;
        for (const auto& input : audio) {
            if (translation)
                request.items.push_back(
                    {SpeechTranslationRequest{audio_view(input), target, source}, config});
            else
                request.items.push_back(
                    {SpeechTranscriptionRequest{audio_view(input), source}, config});
        }
        emit(task.run(request));
    } else if (translation) {
        const auto task = model.task<BatchSpeechTranslation>();
        const auto config = transcription_config(command, task.config_fields());
        BatchSpeechTranslationRequest request;
        for (const auto& input : audio)
            request.items.push_back({{audio_view(input), target, source}, config});
        emit(task.run(request));
    } else {
        const auto task = model.task<BatchSpeechTranscription>();
        const auto config = transcription_config(command, task.config_fields());
        BatchSpeechTranscriptionRequest request;
        for (const auto& input : audio)
            request.items.push_back({{audio_view(input), source}, config});
        emit(task.run(request));
    }
}

nlohmann::json transcript_update(const SpeechTranscriptUpdate& update) {
    auto result = detail::text_json(update.transcript());
    result["is_final"] = update.is_final();
    result["chunk_index"] = update.chunk_index();
    result["accepted_samples"] = update.accepted_samples(); // Per-channel sample frames.
    result["sample_rate"] = update.input_format().sample_rate;
    result["channels"] = update.input_format().channels;
    return result;
}
void transcribe_stream(const Command& command, const Model& model, std::ostream& output) {
    const auto audio = io::read_wav_interleaved(require_option(command, "--input"));
    const auto task = model.task<StreamingSpeechTranscription>();
    const auto config = detail::task_config(command, task.config_fields(),
                                            {"--input", "--chunk-samples", "--language"});
    auto stream = task.create({{static_cast<std::uint32_t>(audio.sample_rate),
                                static_cast<std::uint32_t>(audio.channels)},
                               option(command, "--language")},
                              config);
    // --chunk-samples counts per-channel frames. One second is an application
    // IO default, not a model/ASR context-size default. Never split a PCM frame.
    const auto frames = detail::int_option(command, "--chunk-samples", audio.sample_rate, 1);
    if (static_cast<std::size_t>(frames) >
        std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(audio.channels))
        throw std::invalid_argument("audio chunk size overflows host storage");
    const auto scalars =
        static_cast<std::size_t>(frames) * static_cast<std::size_t>(audio.channels);
    nlohmann::json chunks = nlohmann::json::array();
    for (std::size_t offset = 0; offset < audio.samples.size();) {
        const auto count = std::min(scalars, audio.samples.size() - offset);
        chunks.push_back(
            transcript_update(stream.accept_audio({audio.samples.data() + offset, count})));
        offset += count;
    }
    auto final = transcript_update(stream.finish());
    const auto info = stream.info();
    final["source_language"] =
        info.source_language ? nlohmann::json(*info.source_language) : nlohmann::json(nullptr);
    detail::write_json(output, {{"chunks", std::move(chunks)}, {"final", std::move(final)}});
}

const char* speech_event_name(SpeechEventKind kind) {
    static constexpr const char* names[] = {"agent_audio",
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
                                            "input_cleared"};
    const auto index = static_cast<std::uint32_t>(kind);
    if (index == 0 || index > sizeof(names) / sizeof(*names))
        throw std::runtime_error("unknown speech event kind");
    return names[index - 1];
}
void speech_session(const Command& command, const Model& model, std::string_view id,
                    std::ostream& output) {
    const auto audio = io::read_wav_interleaved(require_option(command, "--input"));
    const SpeechDialogueRequest request{
        {static_cast<std::uint32_t>(audio.sample_rate), static_cast<std::uint32_t>(audio.channels)},
        option(command, "--system-prompt")};
    auto create = [&](const auto& task) {
        const auto config =
            detail::task_config(command, task.config_fields(),
                                {"--input", "--output", "--system-prompt", "--timeout-ms"});
        return task.create(request, config);
    };
    auto session = id == OfflineSpeechDialogue::kTask ? create(model.task<OfflineSpeechDialogue>())
                                                      : create(model.task<DuplexSpeechDialogue>());
    const auto info = session.info();
    const auto timeout = detail::int_option(command, "--timeout-ms", 30000, 0);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout);
    nlohmann::json events = nlohmann::json::array();
    std::vector<float> agent_audio;
    SpeechAudioFormat format{};
    bool input_finished = false, ended = false, cancelled = false;
    auto consume = [&](SpeechReadResult batch) {
        bool epoch_ended = batch.status == SpeechPollStatus::EpochEnd;
        if (batch.events) {
            if (batch.events->state() == SpeechReadState::Failed)
                throw std::runtime_error("speech session failed");
            epoch_ended = batch.events->state() == SpeechReadState::EpochEnded;
            for (std::size_t i = 0; i < batch.events->size(); ++i) {
                const auto event = batch.events->at(i);
                nlohmann::json item{{"kind", speech_event_name(event.kind)},
                                    {"epoch", event.epoch},
                                    {"sequence", event.sequence},
                                    {"text", std::string(event.text)},
                                    {"is_final", event.is_final},
                                    {"audio_samples", event.audio.samples.size()},
                                    {"channels", event.audio.channels},
                                    {"media_start_sample", event.media_start_sample},
                                    {"media_end_sample", event.media_end_sample},
                                    {"frame_index", event.frame_index}};
                item["sample_rate"] = event.audio.sample_rate
                                          ? nlohmann::json(*event.audio.sample_rate)
                                          : nlohmann::json(nullptr);
                if (event.tool_call) {
                    static constexpr const char* states[] = {"unknown", "complete", "incomplete",
                                                             "malformed"};
                    const auto state = static_cast<std::uint32_t>(event.tool_call->state);
                    if (state >= sizeof(states) / sizeof(*states))
                        throw std::runtime_error("unknown speech tool-call state");
                    item["tool_call"] = {{"call_id", event.tool_call->call_id},
                                         {"name", event.tool_call->name},
                                         {"arguments_json", event.tool_call->arguments_json},
                                         {"state", states[state]}};
                }
                events.push_back(std::move(item));
                if (event.kind == SpeechEventKind::Error)
                    throw std::runtime_error(event.text.empty() ? "speech session failed"
                                                                : std::string(event.text));
                if (event.kind == SpeechEventKind::Cancelled)
                    cancelled = true;
                if (event.kind == SpeechEventKind::AgentAudio && !event.audio.samples.empty()) {
                    if (!event.audio.sample_rate || event.audio.channels == 0)
                        throw std::runtime_error("agent audio omitted its format");
                    if (format.sample_rate && (format.sample_rate != *event.audio.sample_rate ||
                                               format.channels != event.audio.channels))
                        throw std::runtime_error("speech session changed output format");
                    format = {*event.audio.sample_rate, event.audio.channels};
                    for (const auto sample : event.audio.samples) {
                        if (!std::isfinite(sample))
                            throw std::runtime_error("agent audio contains a non-finite sample");
                        agent_audio.push_back(sample);
                    }
                }
            }
        }
        if (cancelled)
            throw std::runtime_error("speech session was cancelled before normal completion");
        if (epoch_ended) {
            if (!input_finished)
                throw std::runtime_error("speech epoch ended before all input was submitted");
            ended = true;
        }
    };
    auto read = [&] {
        const auto now = std::chrono::steady_clock::now();
        const auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        const auto wait_ms = std::max<std::int64_t>(0, std::min<std::int64_t>(remaining, 500));
        consume(session.wait_events(wait_ms));
        if (!ended && std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error("speech session timed out before normal epoch completion");
    };
    // A false return accepts no input. Drain events and retry the same buffer;
    // do not drop samples or create a shared producer/consumer worker.
    while (!session.append_audio({audio.samples.data(), audio.samples.size()}))
        read();
    session.finish_input();
    input_finished = true;
    while (!ended)
        read();
    // Epoch completion finishes this file command, not the reusable session
    // type. TurnFinished alone never ends the loop; no separate permanent EOF.
    nlohmann::json result{{"events", std::move(events)},
                          {"sample_rate", format.sample_rate},
                          {"channels", format.channels},
                          {"num_samples", agent_audio.size()},
                          {"input_sample_rate", info.input.sample_rate},
                          {"input_channels", info.input.channels}};
    result["system_prompt"] =
        info.system_prompt ? nlohmann::json(*info.system_prompt) : nlohmann::json(nullptr);
    if (has_option(command, "--output")) {
        if (agent_audio.empty())
            throw std::runtime_error("speech session produced no agent audio");
        const auto path = command.options.at("--output");
        io::write_wav_interleaved({agent_audio.data(), agent_audio.size()}, format.sample_rate,
                                  format.channels, path);
        result["output"] = path;
    }
    detail::write_json(output, result);
}
} // namespace

std::string_view audio_task_for_command(const Command& command, const Model& model) {
    if (!command.selected_task.empty())
        return {};
    if (command.kind == CommandKind::kTranscribe) {
        if (has_option(command, "--translate"))
            return translate(command) ? SpeechTranslation::kTask : SpeechTranscription::kTask;
        if (has_option(command, "--target-language") ||
            model.info().bundle_task == SpeechTranslation::kTask)
            return SpeechTranslation::kTask;
        return SpeechTranscription::kTask;
    }
    if (command.kind == CommandKind::kTranscribeBatch) {
        const auto primary = model.info().bundle_task;
        if (primary == MixedBatchSpeechToText::kTask)
            return MixedBatchSpeechToText::kTask;
        if (translation_intent(command, model)) {
            if (!model.supports<BatchSpeechTranslation>() &&
                model.supports<MixedBatchSpeechToText>())
                return MixedBatchSpeechToText::kTask;
            return BatchSpeechTranslation::kTask;
        }
        if (!model.supports<BatchSpeechTranscription>() && model.supports<MixedBatchSpeechToText>())
            return MixedBatchSpeechToText::kTask;
        return BatchSpeechTranscription::kTask;
    }
    if (command.kind == CommandKind::kGenerateAudio) {
        if (has_option(command, "--stream") &&
            detail::parse_bool(command.options.at("--stream"), "--stream"))
            return StreamingTextToSpeech::kTask;
        if (model.info().bundle_task == TextToSpeech::kTask ||
            (!model.supports<TextToAudio>() && model.supports<TextToSpeech>()))
            return TextToSpeech::kTask;
        return TextToAudio::kTask;
    }
    if (command.kind == CommandKind::kSpeak)
        return SpeechToSpeechResponse::kTask;
    if (command.kind == CommandKind::kTranscribeStreaming)
        return StreamingSpeechTranscription::kTask;
    if (command.kind == CommandKind::kSpeechSession) {
        if (model.info().bundle_task == OfflineSpeechDialogue::kTask ||
            (!model.supports<DuplexSpeechDialogue>() && model.supports<OfflineSpeechDialogue>()))
            return OfflineSpeechDialogue::kTask;
        return DuplexSpeechDialogue::kTask;
    }
    return {};
}

bool dispatch_sdk_audio(const Command& command, const Model& model, std::string_view id,
                        std::ostream& output) {
    if (command.kind == CommandKind::kGenerateAudio &&
        (id == TextToAudio::kTask || id == TextToSpeech::kTask ||
         id == StreamingTextToSpeech::kTask)) {
        generate_audio(command, model, id, output);
        return true;
    }
    if (command.kind == CommandKind::kTranscribeBatch &&
        (id == BatchSpeechTranscription::kTask || id == BatchSpeechTranslation::kTask ||
         id == MixedBatchSpeechToText::kTask)) {
        transcribe_batch(command, model, id, output);
        return true;
    }
    if (command.kind == CommandKind::kTranscribeStreaming &&
        id == StreamingSpeechTranscription::kTask) {
        transcribe_stream(command, model, output);
        return true;
    }
    if (command.kind == CommandKind::kSpeak && id == SpeechToSpeechResponse::kTask) {
        const auto audio = io::read_wav_interleaved(require_option(command, "--input"));
        const auto task = model.task<SpeechToSpeechResponse>();
        const auto config =
            detail::task_config(command, task.config_fields(), {"--input", "--output"});
        write_audio(task.run({audio_view(audio)}, config), require_option(command, "--output"),
                    output);
        return true;
    }
    if (command.kind == CommandKind::kSpeechSession &&
        (id == DuplexSpeechDialogue::kTask || id == OfflineSpeechDialogue::kTask)) {
        speech_session(command, model, id, output);
        return true;
    }
    if (command.kind != CommandKind::kTranscribe ||
        (id != SpeechTranscription::kTask && id != SpeechTranslation::kTask))
        return false;
    const bool translation = id == SpeechTranslation::kTask;
    validate_translation(command, translation);
    const auto audio = io::read_wav_interleaved(require_option(command, "--input"));
    if (translation) {
        const auto task = model.task<SpeechTranslation>();
        const auto config = transcription_config(command, task.config_fields());
        const auto result = task.run({audio_view(audio), option(command, "--target-language"),
                                      option(command, "--source-language")},
                                     config);
        detail::write_json(output, detail::text_json(result));
    } else {
        const auto task = model.task<SpeechTranscription>();
        const auto config = transcription_config(command, task.config_fields());
        const auto result =
            task.run({audio_view(audio), option(command, "--source-language")}, config);
        detail::write_json(output, detail::text_json(result));
    }
    return true;
}

} // namespace trtmc::cli
