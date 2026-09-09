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
