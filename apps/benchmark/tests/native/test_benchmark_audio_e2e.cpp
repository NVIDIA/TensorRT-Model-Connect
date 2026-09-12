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
} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: test_benchmark_audio_e2e WORKER SDK_RUNTIME_ROOT\n";
        return 2;
    }
    try {
        const std::filesystem::path root(argv[2]);
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
              "direct synchronous callbacks count real PCM without accumulating or writing audio");
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
        std::cout << (failures ? "FAILED\n" : "ALL PASSED\n");
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
