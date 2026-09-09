/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/cli.h"
#include "cli/io.h"
#include "cli/sdk_dispatch.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace {
using nlohmann::json;
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
void bundle(const std::filesystem::path& path, std::string mode,
            const char* family = "audio_fixture") {
    const auto header = json{
        {"format", 1},
        {"family", family},
        {"task", std::move(mode)},
        {"backend", "fake"},
        {"sections",
         json::object()}}.dump();
    std::ofstream out(path, std::ios::binary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.write("BUNDLE\x01\x00", 8);
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.put(static_cast<char>((static_cast<std::uint64_t>(header.size()) >> shift) & 255U));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
}
struct Run {
    int status;
    std::string output, error;
};
Run run(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    for (auto& argument : arguments)
        argv.push_back(argument.data());
    std::ostringstream output, error;
    const auto status = trtmc::cli::run(static_cast<int>(argv.size()), argv.data(), output, error);
    return {status, output.str(), error.str()};
}
void generated_and_batch(const std::filesystem::path& root, const std::filesystem::path& model,
                         const std::filesystem::path& input) {
    const auto tts = root / "audio_cli_tts.bundle", mixed = root / "audio_cli_mixed.bundle",
               output = root / "audio_cli_generated.wav", second = root / "audio_cli_mono.wav";
    bundle(tts, "text_to_speech");
    bundle(mixed, "mixed_batch_speech_to_text");
    const float mono[] = {0.1F, 0.2F, 0.3F, 0.4F};
    trtmc::cli::io::write_wav_interleaved({mono, 4}, 16000, 1, second.string());
    const auto generated = run({"trtmc", "generate-audio", model.string(), "--runtime-root",
                                root.string(), "--prompt", "sound", "--output", output.string()});
    check(generated.status == 0, "prompt-conditioned audio generation selects its actual Task");
    if (generated.status == 0) {
        const auto json_output = json::parse(generated.output);
        const auto saved = trtmc::cli::io::read_wav_interleaved(output.string());
        check(json_output.at("channels") == 2 && json_output.at("num_samples") == 4 &&
                  saved.channels == 2 && saved.sample_rate == 24000 && saved.samples.size() == 4 &&
                  std::abs(saved.samples[0] - 0.05F) < 1e-6 && saved.samples[3] == -0.25F,
              "generated WAV preserves stereo ordering, actual rate and signed PCM");
    }
    const auto speech = run({"trtmc", "generate-audio", tts.string(), "--runtime-root",
                             root.string(), "--prompt", "Hello", "--output", output.string(),
                             "--language", "fr", "--set", "speaker=1", "--set", "normalize=false"});
    check(speech.status == 0, "TTS primary and typed language work without a Task selector");
    if (speech.status == 0) {
        const auto saved = trtmc::cli::io::read_wav_interleaved(output.string());
        check(saved.samples[1] == 0.1F && saved.samples[2] == 0.2F && saved.samples[3] == 0,
              "family speaker/normalization config and typed TTS language are not conflated");
    }
    const auto spoken = run({"trtmc", "speak", model.string(), "--runtime-root", root.string(),
                             "--input", input.string(), "--output", output.string()});
    check(spoken.status == 0, "one-shot speech response is not implemented as a dialogue session");
    if (spoken.status == 0) {
        const auto saved = trtmc::cli::io::read_wav_interleaved(output.string());
        check(saved.sample_rate == 8000 && saved.channels == 2 && saved.samples.size() == 6 &&
                  saved.samples.front() == -0.75F && saved.samples.back() == 0.25F,
              "speech response preserves actual input and output channel metadata");
    }
    const auto batch = run({"trtmc", "transcribe-batch", model.string(), "--runtime-root",
                            root.string(), "--input", input.string(), "--input", second.string()});
    check(batch.status == 0, "independent WAV inputs use one native batch Task");
    if (batch.status == 0) {
        const auto items = json::parse(batch.output).at("results");
        check(items.size() == 2 && items[0].at("text") == "batch_asr:auto!" &&
                  items[1].at("token_ids") == json::array({1, 1, 0}) &&
                  items[0].at("setup_ms") == 2 && items[1].at("setup_ms") == 1 &&
                  items[0].at("decode_ms") == 6 && items[1].at("decode_ms") == 4,
              "batch preserves item order, independent formats and never loops over single ASR");
    }
    const auto translated =
        run({"trtmc", "transcribe-batch", model.string(), "--runtime-root", root.string(),
             "--input", input.string(), "--target-language", "fr"});
    check(translated.status == 0 && json::parse(translated.output).at("results").at(0).at("text") ==
                                        "batch_translate:auto->fr!",
          "batch target language selects native translation before execution");
    const auto mixed_result =
        run({"trtmc", "transcribe-batch", mixed.string(), "--runtime-root", root.string(),
             "--input", input.string(), "--input", second.string(), "--translate", "true"});
    check(mixed_result.status == 0,
          "mixed batch primary remains a distinct native family operation");
    if (mixed_result.status == 0) {
        const auto items = json::parse(mixed_result.output).at("results");
        check(items[0].at("text") == "mixed_translate:auto->en!" &&
                  items[1].at("token_ids") == json::array({1, 1, 0, 0, 0, 0}),
              "mixed native batch does not invoke homogeneous batches or scalar methods");
    }
    for (const auto& args : std::vector<std::vector<std::string>>{
             {"trtmc", "generate-audio", model.string(), "--runtime-root", root.string(),
              "--prompt", "p", "--output", output.string(), "--language", "fr"},
             {"trtmc", "generate-audio", tts.string(), "--runtime-root", root.string(), "--prompt",
              "p", "--output", output.string(), "--chunk-frames", "2"},
             {"trtmc", "generate-audio", tts.string(), "--runtime-root", root.string(), "--prompt",
              "p", "--output", output.string(), "--task", "text_to_speech", "--stream", "true"},
             {"trtmc", "transcribe-batch", model.string(), "--runtime-root", root.string()}}) {
        const auto bad = run(args);
        check(bad.status != 0 && bad.output.empty(),
              "unsupported audio input/lifecycle combinations reject");
    }
    for (const auto& path : {tts, mixed, output, second})
        std::filesystem::remove(path);
}

void streamed_and_session(const std::filesystem::path& root, const std::filesystem::path& input) {
    const auto live = root / "audio_cli_live.bundle", offline = root / "audio_cli_offline.bundle",
               bad = root / "audio_cli_bad_session.bundle", raw = root / "audio_cli_stream.f32",
               output = root / "audio_cli_session.wav",
               too_large = root / "audio_cli_backpressure.wav";
    bundle(live, "duplex_speech_dialogue", "speech_fixture");
    bundle(offline, "offline_speech_dialogue", "speech_fixture");
    bundle(bad, "bad_event", "speech_fixture");
    const auto generated =
        run({"trtmc", "generate-audio", live.string(), "--runtime-root", root.string(), "--prompt",
             "Hello", "--output", raw.string(), "--stream", "true"});
    check(generated.status == 0, "streaming TTS uses the synchronous public callback contract");
    if (generated.status == 0) {
        const auto metadata = json::parse(generated.output);
        const auto values = trtmc::cli::detail::read_float32_file(raw.string());
        check(metadata.at("format") == "float32le" && metadata.at("channels") == 2 &&
                  metadata.at("sample_rate") == 24000 && metadata.at("num_samples") == 12 &&
                  metadata.at("num_frames") == 6 && values.size() == 12 && values[8] == 2,
              "streamed bytes preserve every callback chunk, real channels and final counts");
    }
    if (std::filesystem::exists("/dev/full")) {
        const auto failed_write =
            run({"trtmc", "generate-audio", live.string(), "--runtime-root", root.string(),
                 "--prompt", "Hello", "--output", "/dev/full", "--stream", "true"});
        check(failed_write.status != 0 && failed_write.output.empty(),
              "callback write/flush failure stops generation and propagates without success JSON");
    }
    const auto asr = run({"trtmc", "transcribe-streaming", live.string(), "--runtime-root",
                          root.string(), "--input", input.string(), "--chunk-samples", "1",
                          "--language", "fr", "--set", "text=kept:"});
    check(asr.status == 0, "input-stream ASR uses existing file/chunk flags");
    if (asr.status == 0) {
        const auto value = json::parse(asr.output);
        check(value.at("chunks").size() == 3 &&
                  value.at("chunks").at(1).at("accepted_samples") == 2 &&
                  value.at("chunks").at(1).at("is_final") == false &&
                  value.at("final").at("text") == "kept:frames:3" &&
                  value.at("final").at("is_final") == true &&
                  value.at("final").at("channels") == 2 &&
                  value.at("final").at("source_language") == "fr" &&
                  value.at("final").at("segments").at(0).at("end_seconds") == 3.0 / 8000,
              "ASR chunking counts per-channel frames, preserves language and final timed "
              "transcript");
    }
    for (const bool is_offline : {false, true}) {
        const auto result =
            run({"trtmc", "speech-session", (is_offline ? offline : live).string(),
                 "--runtime-root", root.string(), "--input", input.string(), "--output",
                 output.string(), "--system-prompt", "kept system prompt"});
        check(result.status == 0, "file speech-session completes its selected live/offline epoch");
        if (result.status == 0) {
            const auto value = json::parse(result.output);
            const auto saved = trtmc::cli::io::read_wav_interleaved(output.string());
            bool reply = false, finished = false, audio_metadata = false;
            for (const auto& event : value.at("events")) {
                if (event.at("kind") == "agent_text")
                    reply = event.at("text") == (is_offline ? "offline reply" : "live reply");
                if (event.at("kind") == "input_finished")
                    finished = true;
                if (event.at("kind") == "agent_audio")
                    audio_metadata =
                        event.at("channels") == 2 && event.at("media_start_sample") == 0 &&
                        event.at("media_end_sample") == 3 && event.at("frame_index") == 1 &&
                        event.at("epoch").get<std::uint64_t>() > 0;
            }
            check(reply && finished && audio_metadata &&
                      value.at("system_prompt") == "kept system prompt" && saved.channels == 2 &&
                      saved.sample_rate == 24000 && saved.samples.size() == 6 &&
                      saved.samples[0] == 0.25F && saved.samples[5] == -0.75F,
                  "session retains text, full event coordinates, prompt and all actual output PCM");
        }
    }
    const float many[20] = {};
    trtmc::cli::io::write_wav_interleaved({many, 20}, 8000, 2, too_large.string());
    const auto pressure = run({"trtmc", "speech-session", live.string(), "--runtime-root",
                               root.string(), "--input", too_large.string(), "--timeout-ms", "1"});
    check(pressure.status != 0 && pressure.output.empty() &&
              pressure.error.find("timed out") != std::string::npos,
          "unaccepted input is not dropped or reported as successful under backpressure");
    const auto invalid = run({"trtmc", "speech-session", bad.string(), "--runtime-root",
                              root.string(), "--input", input.string()});
    check(invalid.status != 0 && invalid.output.empty(),
          "invalid family event cannot become successful EOF");
    const auto tool_protocol =
        run({"trtmc", "speech-session", live.string(), "--runtime-root", root.string(), "--input",
             input.string(), "--task", "tool_speech_dialogue"});
    check(tool_protocol.status != 0 && tool_protocol.output.empty(),
          "CLI does not fabricate tool declarations or a persistent interaction protocol");
    for (const auto& path : {live, offline, bad, raw, output, too_large})
        std::filesystem::remove(path);
}
void exercise(const std::filesystem::path& root) {
    const auto all = root / "audio_cli.bundle", restricted = root / "audio_cli_asr.bundle",
               translation_default = root / "audio_cli_translation.bundle",
               input = root / "audio_cli_stereo.wav";
    bundle(all, "speech_transcription");
    bundle(restricted, "asr_only");
    bundle(translation_default, "speech_translation");
    const float values[] = {0.25F, -0.25F, 0.5F, -0.5F, 0.75F, -0.75F};
    trtmc::cli::io::write_wav_interleaved({values, 6}, 8000, 2, input.string());
    const std::vector<std::string> base{"trtmc",          "transcribe",  all.string(),
                                        "--runtime-root", root.string(), "--input",
                                        input.string()};
    auto invoke = [&](std::initializer_list<std::string> extra) {
        auto args = base;
        args.insert(args.end(), extra.begin(), extra.end());
        return run(std::move(args));
    };
    const auto basic = invoke({});
    check(basic.status == 0, "minimal offline ASR CLI selects the typed Task without extra flags");
    if (basic.status != 0)
        throw std::runtime_error(basic.error);
    const auto result = json::parse(basic.output);
    check(result.at("text") == "asr:auto!" && result.at("token_ids") == json::array({1, 8000, 2}) &&
              result.at("setup_ms") == 2 && result.at("decode_ms") == 6,
          "actual WAV channels/sample rate and family defaults survive the CLI");
    check(result.at("segments").at(0).at("text") == "asr:auto!" &&
              std::abs(result.at("segments").at(0).at("end_seconds").get<double>() - 3.0 / 8000) <
                  1e-12,
          "transcription keeps human-readable text, token IDs and segment timestamps");
    const auto explicit_config = invoke({"--source-language", "fr", "--set", "suffix="});
    check(explicit_config.status == 0 && json::parse(explicit_config.output).at("text") == "asr:fr",
          "source language is typed input and an explicit empty config remains supplied");
    const auto translation = invoke({"--translate", "true"});
    check(translation.status == 0 &&
              json::parse(translation.output).at("text") == "translate:auto->en!",
          "translation is a separate Task using its family-declared target default");
    const auto targeted = invoke(
        {"--task", "speech_translation", "--source-language", "de", "--target-language", "fr"});
    check(targeted.status == 0 && json::parse(targeted.output).at("text") == "translate:de->fr!",
          "explicit source and target roles cross the public ABI");
    const auto target_only = invoke({"--target-language", "fr"});
    check(target_only.status == 0 &&
              json::parse(target_only.output).at("text") == "translate:auto->fr!",
          "a target language selects translation without another flag");
    const auto fixed_translation =
        run({"trtmc", "transcribe", translation_default.string(), "--runtime-root", root.string(),
             "--input", input.string()});
    check(fixed_translation.status == 0 &&
              json::parse(fixed_translation.output).at("text") == "translate:auto->en!",
          "a fixed translation primary retains its default without extra flags");
    for (const auto& extra : std::vector<std::vector<std::string>>{
             {"--task", "speech_transcription", "--target-language", "fr"},
             {"--source-language", ""},
             {"--task", "speech_transcription", "--translate", "true"},
             {"--task", "speech_translation", "--translate", "false"},
             {"--translate", "invalid"},
             {"--set", "unknown=1"},
             {"--set", "suffix=a", "--set", "suffix=b"}}) {
        auto args = base;
        args.insert(args.end(), extra.begin(), extra.end());
        const auto invalid = run(std::move(args));
        check(invalid.status != 0 && invalid.output.empty(),
              "invalid language/Task/config inputs fail without silently changing the request");
    }
    const auto unsupported = run({"trtmc", "transcribe", restricted.string(), "--runtime-root",
                                  root.string(), "--input", input.string(), "--translate", "true"});
    check(unsupported.status != 0 && unsupported.output.empty() &&
              unsupported.error.find("does not support") != std::string::npos,
          "missing translation Task does not retry transcription");
    generated_and_batch(root, all, input);
    streamed_and_session(root, input);
    for (const auto& path : {all, restricted, translation_default, input})
        std::filesystem::remove(path);
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    try {
        exercise(argv[1]);
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        ++failures;
    }
    if (!failures)
        std::cout << "ALL PASSED\n";
    return failures ? 1 : 0;
}
