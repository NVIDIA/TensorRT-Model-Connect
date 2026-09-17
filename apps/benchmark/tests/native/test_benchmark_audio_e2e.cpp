/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/io.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>

namespace {
using Json = nlohmann::json;
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
std::string quote(const std::string& value) {
    std::string output{"'"};
    for (char c : value)
        output += c == '\'' ? "'\\''" : std::string(1, c);
    return output + "'";
}
void bundle(const std::filesystem::path& path, const std::string& task,
            const std::string& family = "audio_fixture") {
    const auto header = Json{
        {"format", 1},
        {"family", family},
        {"task", task},
        {"backend", "fake"},
        {"sections",
         Json::object()}}.dump();
    std::ofstream output(path, std::ios::binary);
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.write("BUNDLE\x01\x00", 8);
    for (unsigned shift = 0; shift < 64; shift += 8)
        output.put(static_cast<char>((static_cast<std::uint64_t>(header.size()) >> shift) & 255U));
    output << header;
}

void audio_artifacts(const std::string& worker, const std::filesystem::path& root) {
    const auto directory = root / "benchmark_audio_artifacts";
    std::filesystem::create_directories(directory);
    const auto wav = directory / "input.wav";
    const std::vector<float> input_samples{0.25F, -0.25F, 0.5F, -0.5F};
    trtmc::cli::io::write_wav_interleaved({input_samples.data(), input_samples.size()}, 8000, 2,
                                          wav.string());
    for (const std::string name :
         {"text_to_audio", "text_to_speech", "empty_last", "speech_to_speech_response",
          "streaming_text_to_speech", "empty_stream"}) {
        const bool response = name == "speech_to_speech_response";
        const bool streaming = name == "streaming_text_to_speech" || name == "empty_stream";
        const bool empty_last = name == "empty_last", empty_stream = name == "empty_stream";
        const auto selected = streaming    ? std::string("streaming_text_to_speech")
                              : empty_last ? std::string("text_to_audio")
                                           : name;
        const auto mode = streaming || response ? selected
                          : empty_last          ? "artifact_audio_empty_last"
                                                : "artifact_audio";
        const auto model = directory / (name + ".bundle");
        const auto request_path = directory / (name + ".request.json");
        const auto output = directory / (name + ".json");
        bundle(model, mode, streaming ? "speech_fixture" : "audio_fixture");
        const Json request{
            {"schema_version", 2},
            {"case_name", name},
            {"bundle", model.string()},
            {"runtime_root", root.string()},
            {"selected_task", selected},
            {"operation", response ? "speak" : "generate_audio"},
            {"request", response ? Json{{"audio_path", wav.string()}}
                                 : Json{{"prompt", empty_stream ? "benchmark-empty" : "Hello"}}},
            {"measurement", {{"warmup", 1}, {"iterations", 2}}}};
        {
            std::ofstream file(request_path);
            file << request;
        }
        const auto command = quote(worker) + " --request " + quote(request_path.string()) +
                             " --output " + quote(output.string());
        check(std::system(command.c_str()) == 0, "audio artifact worker completes");
        std::ifstream file(output);
        Json result;
        file >> result;
        check(result.at("status") == "completed" && result.at("observations").size() == 2 &&
                  result.at("observation_serialization_included") == false,
              "audio artifacts preserve the number of measured calls and timing boundary");
        check(!result.at("output_summary").contains("runtime_e2e_wall_ms"),
              "cached summary retains the original summary fields without per-call wall time");
        std::vector<std::string> artifacts;
        for (std::size_t i = 0; i < 2; ++i) {
            const auto& observation = result.at("observations").at(i);
            check(observation.contains("audio_artifact"),
                  "measured audio has explicit artifact state");
            if (!observation.contains("audio_artifact"))
                continue;
            if (empty_stream || (empty_last && i == 1)) {
                check(observation.at("audio_artifact").is_null() &&
                          observation.at("output_samples") == 0,
                      "valid empty output has no fabricated or stale waveform");
                continue;
            }
            const auto path = observation.at("audio_artifact").get<std::string>();
            const auto artifact = directory / path;
            artifacts.push_back(artifact.string());
            check(path == name + ".audio." + std::to_string(i + 1) + ".wav",
                  "each measured waveform has its own portable case-relative artifact path");
            const auto audio = trtmc::cli::io::read_wav_interleaved(artifact.string());
            const float first = static_cast<float>(i + 2) / 8;
            const auto expected =
                response    ? std::vector<float>{-0.5F, 0.5F, -0.25F, 0.25F}
                : streaming ? std::vector<float>{0,    0.25F, 0.5F, 0.75F, 1,    0.25F,
                                                 0.5F, 0.75F, 2,    0.25F, 0.5F, 0.75F}
                : name == "text_to_speech" ? std::vector<float>{first, 0, 0.1F, 0.25F}
                                           : std::vector<float>{first, 0, 0.25F, -0.25F};
            check(audio.channels == 2 && audio.sample_rate == (response ? 8000 : 24000) &&
                      audio.samples == expected,
                  "FLOAT32 WAV preserves every interleaved sample including unclipped values");
            check(observation.at("output_samples") == expected.size(),
                  "waveform length equals the actual measured result");
            if (streaming)
                check(observation.at("streaming_pcm_copy_included") == true,
                      "streaming receipt explicitly includes borrowed-PCM copying in the call");
        }
        const auto& summary = result.at("output_summary");
        check(summary.contains("audio_artifact"), "summary identifies its actual audio output");
        if (summary.contains("audio_artifact"))
            check(summary.at("audio_artifact") ==
                      result.at("observations").back().at("audio_artifact"),
                  "summary references the final measurement without another observer write");
        std::size_t written = 0;
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            const auto filename = entry.path().filename().string();
            if (filename.rfind(name + ".audio.", 0) == 0 && entry.path().extension() == ".wav")
                ++written;
        }
        check(written == (empty_stream ? 0U
                          : empty_last ? 1U
                                       : 2U),
              "warmup and summary do not create extra waveform artifacts");
        for (const auto& path : artifacts)
            std::filesystem::remove(path);
        for (const auto& path : {model, request_path, output})
            std::filesystem::remove(path);
    }
    std::filesystem::remove(wav);
    std::filesystem::remove(directory);
}
void dialogue_artifacts(const std::string& worker, const std::filesystem::path& root) {
    const auto directory = root / "benchmark_dialogue_artifacts";
    std::filesystem::create_directories(directory);
    const auto input_audio = directory / "input.wav";
    const std::vector<float> input_samples{2.0F, -0.25F, 0.5F, -0.5F, 0.75F, -0.75F};
    trtmc::cli::io::write_wav_interleaved({input_samples.data(), input_samples.size()}, 8000, 2,
                                          input_audio.string());
    for (const std::string task :
         {"duplex_speech_dialogue", "offline_speech_dialogue", "tool_speech_dialogue"}) {
        for (const bool assets : {false, true}) {
            const auto name = task + (assets ? "-loaded" : "-cached");
            const auto model = directory / (name + ".bundle");
            const auto request_path = directory / (name + ".request.json");
            const auto output = directory / (name + ".json");
            bundle(model, task, "speech_fixture");
            Json values{{"audio_path", input_audio.string()}, {"chunk_frames", 1}};
            if (task == "tool_speech_dialogue") {
                values["tools"] = {{{"name", "lookup"}, {"parameters_schema_json", "{}"}},
                                   {{"name", "other"}, {"parameters_schema_json", "{}"}}};
                values["tool_replies"] = {{{"name", "lookup"}, {"content_text", "first"}},
                                          {{"name", "other"}, {"content_text", "second"}}};
            }
            const Json request{
                {"schema_version", 2},
                {"case_name", name},
                {"bundle", model.string()},
                {"runtime_root", root.string()},
                {"operation", "speech_dialogue"},
                {"request", values},
                {"measurement",
                 {{"warmup", 1}, {"iterations", 2}, {"asset_loading_included", assets}}}};
            {
                std::ofstream file(request_path);
                file << request;
            }
            const auto command = quote(worker) + " --request " + quote(request_path.string()) +
                                 " --output " + quote(output.string());
            const int status = std::system(command.c_str());
            check(status == 0, "dialogue media worker completes");
            if (status != 0)
                continue;
            Json result;
            std::ifstream(output) >> result;
            check(result.at("observations").size() == 2, "dialogue keeps two measured calls");
            std::set<std::string> paths;
            for (const auto& observation : result.at("observations")) {
                check(observation.contains("input_audio_artifact") &&
                          observation.contains("event_audio_artifacts"),
                      "dialogue retains actual input and per-event audio artifacts");
                if (!observation.contains("input_audio_artifact") ||
                    !observation.contains("event_audio_artifacts"))
                    continue;
                const auto source = observation.at("input_audio_artifact").get<std::string>();
                check(std::filesystem::path(source).filename() == source &&
                          paths.insert(source).second,
                      "each measurement owns a portable input-audio artifact");
                const auto input =
                    trtmc::cli::io::read_wav_interleaved((directory / source).string());
                check(
                    input.samples == input_samples && input.channels == 2 &&
                        input.sample_rate == 8000,
                    "input artifact preserves every consumed sample, rate and interleaved channel");
                const auto& events = observation.at("events");
                const auto& audio = observation.at("event_audio_artifacts");
                check(audio.size() == events.size(), "event audio references retain event order");
                for (std::size_t index = 0; index < std::min(events.size(), audio.size());
                     ++index) {
                    const auto& event = events[index];
                    if (event.at("audio").empty()) {
                        check(audio[index].is_null(), "non-audio events have no fabricated WAV");
                        continue;
                    }
                    const auto path = audio[index].get<std::string>();
                    check(std::filesystem::path(path).filename() == path &&
                              paths.insert(path).second,
                          "each measured audio event owns a separate portable WAV");
                    const auto actual =
                        trtmc::cli::io::read_wav_interleaved((directory / path).string());
                    check(actual.samples == event.at("audio").get<std::vector<float>>() &&
                              actual.channels == event.at("channels") &&
                              actual.sample_rate == event.at("sample_rate"),
                          "event WAV preserves all PCM and its own format without merging epochs");
                }
            }
            const auto& summary = result.at("output_summary");
            if (summary.contains("input_audio_artifact") &&
                summary.contains("event_audio_artifacts"))
                check(summary.at("input_audio_artifact") ==
                              result.at("observations").back().at("input_audio_artifact") &&
                          summary.at("event_audio_artifacts") ==
                              result.at("observations").back().at("event_audio_artifacts"),
                      "dialogue summary references the last measurement without extra files");
            std::size_t written = 0;
            for (const auto& entry : std::filesystem::directory_iterator(directory))
                if (entry.path().filename().string().rfind(name + ".audio.", 0) == 0 &&
                    entry.path().extension() == ".wav")
                    ++written;
            check(written == paths.size(), "dialogue warmup and summary write no extra media");
            for (const auto& path : paths)
                std::filesystem::remove(directory / path);
            for (const auto& path : {model, request_path, output})
                std::filesystem::remove(path);
        }
    }
    std::filesystem::remove(input_audio);
    std::filesystem::remove(directory);
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: test_benchmark_audio_e2e WORKER SDK_RUNTIME_ROOT\n";
        return 2;
    }
    try {
        const std::filesystem::path root(argv[2]);
        audio_artifacts(argv[1], root);
        const auto model = root / "benchmark_audio.bundle";
        const auto wav = root / "benchmark_audio_stereo.wav";
        const auto input = root / "benchmark_audio_request.json";
        const auto output = root / "benchmark_audio_result.json";
        const float samples[] = {0.25F, -0.25F, 0.5F, -0.5F, 0.75F, -0.75F};
        trtmc::cli::io::write_wav_interleaved({samples, 6}, 8000, 2, wav.string());
        Json request{
            {"schema_version", 2},
            {"case_name", "audio-sdk"},
            {"bundle", model.string()},
            {"runtime_root", root.string()},
            {"operation", "transcribe"},
            {"request", {{"audio_path", wav.string()}}},
            {"measurement",
             {{"warmup", 1}, {"iterations", 2}, {"timing_scope", "public_task_call_wall"}}}};
        const auto command = quote(argv[1]) + " --request " + quote(input.string()) + " --output " +
                             quote(output.string());
        auto run = [&](bool success = true) {
            {
                std::ofstream file(input);
                file << request;
            }
            const int status = std::system(command.c_str());
            check(success ? status == 0 : status != 0, "worker process status");
            std::ifstream file(output);
            Json result;
            file >> result;
            check(result.at("status") == (success ? "completed" : "failed"), "receipt status");
            if (!success)
                check(!result.contains("observations"),
                      "failed call has no completed observations");
            return result;
        };
        bundle(model, "speech_transcription");
        for (bool include_assets : {false, true}) {
            request["measurement"]["asset_loading_included"] = include_assets;
            const auto result = run();
            check(result.at("task") == "speech_transcription" &&
                      result.at("observations").size() == 2 &&
                      result.at("observation_serialization_included") == false,
                  "semantic identity and unchanged measurement boundary");
            const auto& summary = result.at("output_summary");
            check(summary.at("text") == "asr:auto!" && summary.at("input_channels") == 2 &&
                      summary.at("input_samples") == 6 && summary.at("input_frames") == 3 &&
                      summary.at("token_ids") == Json::array({3, 8000, 2}) &&
                      std::abs(summary.at("input_audio_seconds").get<double>() - 3.0 / 8000) <
                          1e-12,
                  "stereo PCM, absent language/default config and warmup count preserved");
            check(summary.at("segments").size() == 1 &&
                      summary.at("segments").at(0).at("end_seconds") == 3.0 / 8000,
                  "real segment timing retained after result observation");
        }
        const Json audio_input{{"audio_path", wav.string()}};
        request["request"] = audio_input;
        request["request"]["language"] = "fr";
        request["request"]["max_new_tokens"] = 17;
        auto result = run();
        check(result.at("output_summary").at("text") == "asr:fr!" &&
                  result.at("output_summary").at("decode_ms") == 17,
              "ASR wire token limit maps to declared max_output_tokens");
        request["request"]["max_new_tokens"] = 0;
        check(run().at("output_summary").at("decode_ms") == 0, "explicit zero is not a default");
        request["request"]["config"] = {{"max_output_tokens", 17}};
        run(false); // The mapped top-level and nested spelling must remain duplicated.
        request["request"] = audio_input;
        request["request"]["max_new_tokens"] = 0.5;
        run(false);
        for (const Json& bad : std::vector<Json>{"", nullptr, 17, false}) {
            request["request"] = audio_input;
            request["request"]["language"] = bad;
            run(false);
        }
        for (const Json& extras :
             std::vector<Json>{{{"target_language", "fr"}},
                               {{"streaming", true}},
                               {{"streaming", "false"}},
                               {{"chunk_ms", 160}},
                               {{"config", {{"unknown", 1}}}},
                               {{"suffix", "a"}, {"config", {{"suffix", "b"}}}}}) {
            request["request"] = audio_input;
            request["request"].update(extras);
            run(false);
        }
        bundle(model, "speech_translation");
        request["request"] = audio_input;
        check(run().at("output_summary").at("text") == "translate:auto->en!",
              "translation keeps fixed family default target language");
        request["request"]["language"] = "en";
        request["request"]["target_language"] = "fr";
        request["request"]["max_new_tokens"] = 3;
        result = run();
        check(result.at("output_summary").at("text") == "translate:en->fr!" &&
                  result.at("output_summary").at("decode_ms") == 3,
              "source and target languages remain independent typed operands");
        request["request"]["target_language"] = "";
        run(false);

        const auto mono_wav = root / "benchmark_audio_mono.wav";
        const float mono_samples[] = {0.125F, -0.25F, 0.5F, -0.75F};
        trtmc::cli::io::write_wav_interleaved({mono_samples, 4}, 16000, 1, mono_wav.string());
        for (const auto* task : {"batch_speech_transcription", "batch_speech_translation",
                                 "mixed_batch_speech_to_text"}) {
            bundle(model, task);
            Json items = Json::array({{{"audio_path", wav.string()}, {"config", {{"suffix", ""}}}},
                                      {{"audio_path", mono_wav.string()},
                                       {"source_language", "fr"},
                                       {"config", {{"suffix", "?"}}}}});
            const bool mixed = std::string(task) == "mixed_batch_speech_to_text";
            const bool translation = std::string(task) == "batch_speech_translation";
            if (translation || mixed)
                items[1]["target_language"] = "de";
            if (mixed) {
                items[0]["kind"] = "transcription";
                items[1]["kind"] = "translation";
            }
            for (const bool include_assets : {false, true}) {
                request["request"] = {{"items", items}};
                request["measurement"]["asset_loading_included"] = include_assets;
                result = run();
                const auto& summary = result.at("output_summary");
                const auto& first = summary.at("items").at(0);
                const auto& second = summary.at("items").at(1);
                check(summary.at("transcribed_items") == 2 &&
                          std::abs(summary.at("input_audio_seconds").get<double>() -
                                   (3.0 / 8000 + 4.0 / 16000)) < 1e-12 &&
                          first.at("input_channels") == 2 &&
                          first.at("input_sample_rate") == 8000 &&
                          second.at("input_channels") == 1 &&
                          second.at("input_sample_rate") == 16000 &&
                          first.at("input_frames") == 3 && second.at("input_frames") == 4,
                      "native speech batch preserves unequal item rates/channels and sums real "
                      "duration");
                check(first.at("token_ids") ==
                              (mixed ? Json::array({3, 0, 0, 0, 0, 0}) : Json::array({3, 0, 0})) &&
                          second.at("token_ids") ==
                              (mixed ? Json::array({3, 1, 0, 0, 0, 0}) : Json::array({3, 1, 0})),
                      "one native batch call per warmup/iteration; no scalar or alternate-batch "
                      "calls");
                check(first.at("text") == (mixed         ? "mixed_asr:auto"
                                           : translation ? "batch_translate:auto->en"
                                                         : "batch_asr:auto") &&
                          second.at("text") == (mixed         ? "mixed_translate:fr->de?"
                                                : translation ? "batch_translate:fr->de?"
                                                              : "batch_asr:fr?"),
                      "per-item language presence, variant and Config survive batch transport");
                check(first.at("segments").at(0).at("end_seconds") == 3.0 / 8000 &&
                          second.at("segments").at(0).at("end_seconds") == 4.0 / 16000 &&
                          result.at("asset_loading_included") == include_assets,
                      "batch transcript segments and asset timing policy remain explicit");
            }
            request["request"] = {{"items", items}, {"config", {{"suffix", "global"}}}};
            run(false);
            request["request"] = {{"items", items}};
            request["request"]["items"][1]["config"] = {{"suffix", 3}};
            run(false);
            request["request"] = {{"items", items}};
            request["request"]["items"][1]["source_language"] = nullptr;
            run(false);
            request["request"] = {{"items", items}};
            if (mixed)
                request["request"]["items"][1]["kind"] = "guess";
            else
                request["request"]["items"][1]["kind"] = "transcription";
            run(false);
        }
        std::filesystem::remove(mono_wav);

        const auto original_audio = trtmc::cli::io::read_wav_interleaved(wav.string());
        Json waveform = Json::array();
        for (const float sample : original_audio.samples)
            waveform.push_back(sample);
        request["operation"] = "speech_dialogue";
        for (const auto* task : {"duplex_speech_dialogue", "offline_speech_dialogue"}) {
            bundle(model, task, "speech_fixture");
            request["request"] = {
                {"audio_path", wav.string()}, {"chunk_frames", 1}, {"system_prompt", ""}};
            for (const bool include_assets : {false, true}) {
                request["measurement"]["asset_loading_included"] = include_assets;
                result = run();
                for (const auto& observation : result.at("observations")) {
                    const auto& events = observation.at("events");
                    check(observation.at("lifecycle_scope") ==
                                  "fresh_create_append_finish_drain_close" &&
                              observation.at("input_chunks") == 3 &&
                              observation.at("append_attempts") == 3 &&
                              observation.at("system_prompt") == "" &&
                              events.at(0).at("sequence") == 0 && events.at(0).at("epoch") == 1 &&
                              events.at(0).at("kind") == "user_speech_started",
                          "every dialogue iteration starts a fresh session and preserves explicit "
                          "empty prompt/chunking");
                    bool audio_seen = false, finished = false, reply_seen = false;
                    for (const auto& event : events) {
                        if (event.at("kind") == "agent_audio") {
                            audio_seen =
                                event.at("audio") == waveform && event.at("channels") == 2 &&
                                event.at("sample_rate") == 24000 &&
                                event.at("media_start_sample") == 0 &&
                                event.at("media_end_sample") == 3 && event.at("frame_index") == 3;
                        }
                        if (event.at("kind") == "input_finished")
                            finished = event.at("is_final").get<bool>();
                        if (event.at("kind") == "agent_text")
                            reply_seen |=
                                event.at("text") == (std::string(task) == "offline_speech_dialogue"
                                                         ? "offline reply"
                                                         : "live reply");
                    }
                    check(audio_seen && finished && reply_seen &&
                              observation.at("read_states").back() == 2 &&
                              std::abs(observation.at("output_audio_seconds").get<double>() -
                                       3.0 / 24000) < 1e-12 &&
                              observation.at("input_sample_rate") == 8000 &&
                              !observation.contains("output_tokens"),
                          "dialogue retains typed PCM/event timeline, drains normal epoch end, and "
                          "does not guess text tokens");
                }
            }
        }
        bundle(model, "tool_speech_dialogue", "speech_fixture");
        const std::string preset_error("preset\0error", 12);
        const Json tool_input{
            {"audio_path", wav.string()},
            {"chunk_frames", 1},
            {"tools",
             Json::array(
                 {{{"name", "lookup"}, {"description", "Lookup"}, {"parameters_schema_json", "{}"}},
                  {{"name", "other"},
                   {"description", "Other"},
                   {"parameters_schema_json", "{}"}}})},
            {"tool_replies",
             Json::array(
                 {{{"name", "lookup"}, {"content_text", preset_error}, {"is_error", true}},
                  {{"name", "other"}, {"content_text", "preset-ok"}, {"is_error", false}}})},
            {"acknowledgements",
             Json::array({{{"tool_name", "lookup"}, {"messages", {"first ack", "chosen ack"}}}})},
            {"default_acknowledgements", {"first default", "chosen default"}}};
        request["request"] = tool_input;
        result = run();
        for (const auto& observation : result.at("observations")) {
            bool acknowledged = false, default_acknowledged = false, error_reply = false,
                 ok_reply = false;
            std::size_t calls = 0;
            for (const auto& event : observation.at("events")) {
                if (event.contains("tool_call")) {
                    ++calls;
                    check(event.at("tool_call").at("state") == "unknown" &&
                              event.at("tool_call").at("arguments_json") == "{}" &&
                              event.at("epoch") == 2,
                          "tool-call arguments, state and fresh epoch are preserved without "
                          "reinterpretation");
                }
                acknowledged |= event.at("text") == "chosen ack";
                default_acknowledged |= event.at("text") == "chosen default";
                error_reply |= event.at("kind") == "error" && event.at("text") == preset_error;
                ok_reply |= event.at("kind") == "agent_text" && event.at("text") == "preset-ok";
            }
            check(calls == 2 && observation.at("submitted_tool_replies") == 2 && acknowledged &&
                      default_acknowledged && error_reply && ok_reply &&
                      observation.at("system_prompt") == "fixture prompt",
                  "tool dialogue submits only ordered preset replies, preserves acknowledgements, "
                  "and treats error replies as recoverable events");
        }
        request["request"]["tool_replies"][0]["name"] = "wrong";
        run(false);
        request["request"] = tool_input;
        request["request"]["tool_replies"].erase(1);
        run(false);
        request["request"] = tool_input;
        request["request"].erase("tool_replies");
        run(false);
        request["request"] = tool_input;
        request["request"]["default_acknowledgements"] = Json::array();
        run(false);
        bundle(model, "duplex_speech_dialogue", "speech_fixture");
        request["request"] = {{"audio_path", wav.string()}, {"chunk_frames", 0}};
        run(false);
        const auto oversized_wav = root / "benchmark_dialogue_oversized.wav";
        std::vector<float> oversized(20, 0);
        trtmc::cli::io::write_wav_interleaved({oversized.data(), oversized.size()}, 8000, 2,
                                              oversized_wav.string());
        request["request"] = {{"audio_path", oversized_wav.string()}, {"timeout_ms", 5}};
        check(
            run(false).at("error").get<std::string>().find("timed out") != std::string::npos,
            "unaccepted audio is not dropped or completed; bounded backpressure fails explicitly");
        std::filesystem::remove(oversized_wav);

        request["operation"] = "generate_audio";
        request["request"] = {{"prompt", "Hello"}};
        for (const auto* task : {"text_to_audio", "text_to_speech"}) {
            bundle(model, task);
            result = run();
            const auto& summary = result.at("output_summary");
            check(summary.at("output_samples") == 4 && summary.at("num_samples") == 4 &&
                      summary.at("output_frames") == 2 && summary.at("channels") == 2 &&
                      summary.at("sample_rate") == 24000 &&
                      summary.at("output_audio_seconds") == 2.0 / 24000,
                  "PCM generation uses real frame counts, rates and channels");
        }
        request["request"] = {{"prompt", "Hello"},
                              {"language", "fr"},
                              {"speaker", 1},
                              {"config", {{"normalize", false}, {"gain", 0.0}}}};
        run();
        request["request"]["speaker"] = "1";
        run(false);
        request["request"] = {{"prompt", "Hello"}, {"config", {{"gain", -1.0}}}};
        run(false); // Provider validation must not fall back or produce a completed receipt.
        request["request"] = {{"prompt", "Hello"}, {"language", "fr"}};
        bundle(model, "text_to_audio");
        run(false);

        bundle(model, "speech_to_speech_response");
        request["operation"] = "speak";
        request["request"] = audio_input;
        for (bool include_assets : {false, true}) {
            request["measurement"]["asset_loading_included"] = include_assets;
            result = run();
            const auto& summary = result.at("output_summary");
            check(summary.at("input_channels") == 2 && summary.at("channels") == 2 &&
                      summary.at("sample_rate") == 8000 && summary.at("output_frames") == 3 &&
                      summary.at("input_audio_seconds") == summary.at("output_audio_seconds"),
                  "one-shot speech response preserves stereo physical duration and caching mode");
        }

        bundle(model, "streaming_speech_transcription", "speech_fixture");
        trtmc::cli::io::write_wav_interleaved({samples, 6}, 1000, 2, wav.string());
        request["operation"] = "transcribe";
        request["request"] = {{"audio_path", wav.string()},
                              {"chunk_ms", 1},
                              {"streaming", true},
                              {"config", {{"text", "prefix:"}}}};
        result = run();
        for (const auto& observation : result.at("observations"))
            check(observation.at("text") == "prefix:frames:3" &&
                      observation.at("is_final") == true && observation.at("chunk_index") == 3 &&
                      observation.at("input_frames") == 3 &&
                      observation.at("input_audio_seconds") == 0.003 &&
                      observation.at("first_partial_ms").is_number(),
                  "each streaming iteration gets a fresh epoch and complete interleaved frames");
        request["request"] = audio_input;
        check(run().at("output_summary").at("chunk_index") == 1,
              "absent packetization uses the existing 160 ms benchmark default");
        for (const Json& bad : std::vector<Json>{0, -1, 0.5, true, "160",
                                                 std::numeric_limits<std::uint64_t>::max()}) {
            request["request"] = audio_input;
            request["request"]["chunk_ms"] = bad;
            run(false);
        }
        for (const Json& extras :
             std::vector<Json>{{{"streaming", false}},
                               {{"target_language", "fr"}},
                               {{"max_new_tokens", 1}},
                               {{"config", {{"text", "benchmark-fail"}}}},
                               {{"config", {{"text", "benchmark-unfinished"}}}}}) {
            request["request"] = audio_input;
            request["request"].update(extras);
            run(false);
        }

        bundle(model, "streaming_text_to_speech", "speech_fixture");
        request["operation"] = "generate_audio";
        request["request"] = {{"prompt", "Hello"}};
        result = run();
        check(result.at("output_summary").at("output_samples") == 12 &&
                  result.at("output_summary").at("output_frames") == 6 &&
                  result.at("output_summary").at("channels") == 2 &&
                  result.at("output_summary").at("output_audio_seconds") == 6.0 / 24000,
              "direct synchronous callbacks retain the actual PCM frame and sample counts");
        request["request"]["streaming"] = true;
        run();
        request["request"]["streaming"] = false;
        run(false);
        for (const auto* prompt :
             {"benchmark-fail", "benchmark-stopped", "benchmark-count-mismatch"}) {
            request["request"] = {{"prompt", prompt}};
            run(false);
        }
        request["request"] = {{"prompt", "Hello"}, {"config", {{"missing", 1}}}};
        run(false);
        request["operation"] = "speak";
        request["request"] = audio_input;
        run(false); // Wrong semantic Task is not adapted or retried.

        bundle(model, "speech_transcription");
        request["operation"] = "transcribe";
        request["request"] = audio_input;
        {
            std::ofstream truncated(wav, std::ios::binary);
            truncated << "RIFF";
        }
        run(false);
        dialogue_artifacts(argv[1], root);
        std::cout << (failures ? "FAILED\n" : "ALL PASSED\n");
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
