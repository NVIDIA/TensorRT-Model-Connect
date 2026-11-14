/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lfm2_audio/runtime/pipeline.h"
#include "trtmc/internal/audio.h"
#include "trtmc/internal/language.h"
#include "trtmc/internal/speech.h"
#include "trtmc/runtime/family_loader.h"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

using json = nlohmann::json;
std::vector<float> read(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        throw std::runtime_error("cannot read waveform");
    const auto n = f.tellg();
    if (n < 0 || n % 4)
        throw std::runtime_error("invalid waveform bytes");
    std::vector<float> v(static_cast<std::size_t>(n) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), n);
    return v;
}
void save(const std::string& path, const std::vector<float>& audio) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(audio.data()), audio.size() * 4);
    if (!f)
        throw std::runtime_error("cannot save waveform");
}
json trace_json(trtmc::lfm2_audio::Pipeline& pipeline) {
    const auto trace = pipeline.last_generation_trace();
    std::vector<std::vector<int32_t>> codes(8);
    for (const auto& frame : trace.audio_codes)
        for (int i = 0; i < 8; ++i)
            codes[i].push_back(frame[i]);
    const bool ended = !trace.audio_codes.empty() && trace.audio_codes.back()[0] == 2048;
    return {{"tokens", trace.text_tokens},
            {"codes", codes},
            {"modalities", trace.modalities},
            {"input_audio_encodings", trace.input_audio_encodings},
            {"prior_audio_code_frames", trace.prior_audio_code_frames},
            {"prompt_tokens", trace.prompt_tokens},
            {"prompt_text_tokens", trace.prompt_text_tokens},
            {"ended_audio", ended}};
}
int main(int argc, char** argv) {
    try {
        if (argc != 4)
            throw std::invalid_argument("usage: audio_probe BUNDLE RUNTIME_ROOT CASES.json");
        auto owner = trtmc::load_task(argv[1], argv[2]);
        auto* pipeline = dynamic_cast<trtmc::lfm2_audio::Pipeline*>(owner.get());
        if (!pipeline)
            throw std::runtime_error("not an LFM audio family instance");
        json cases;
        std::ifstream(argv[3]) >> cases;
        for (const auto& c : cases) {
            using namespace trtmc::internal;
            const int limit = c.value("max_new_tokens", 128);
            const std::string voice = c.value("voice", std::string("us_male"));
            const std::string mode = c.at("mode");
            std::vector<ConfigEntry> options{
                {"max_new_tokens", int64_t{limit}},
                {"text_temperature", c.value("text_temperature", 0.0)},
                {"text_top_k", int64_t{c.value("text_top_k", 1)}},
                {"audio_temperature", c.value("audio_temperature", 0.0)},
                {"audio_top_k", int64_t{c.value("audio_top_k", 1)}},
                {"seed", int64_t{c.value("seed", 0)}}};
            if (mode == "tts" || mode == "tts_stream")
                options.push_back({"voice", std::string_view{voice}});
            const ConfigView cfg{options.data(), options.size()};
            json result = {{"name", c.at("name")}};
            if (mode == "asr") {
                const auto audio = read(c.at("audio"));
                const auto out = dynamic_cast<ISpeechTranscription&>(*owner).run(
                    {{{audio.data(), audio.size()}, c.value("sample_rate", 16000), 1},
                     std::string_view{"en"}},
                    cfg);
                result["text"] = out.text;
                result["tokens"] = out.token_ids;
            } else if (mode == "mixed") {
                const auto audio = read(c.at("audio"));
                const std::string text = c.at("text");
                const std::array<AudioTextPart, 1> system{TextPartView{"Respond with text."}};
                const std::array<AudioTextPart, 2> user{
                    AudioView{{audio.data(), audio.size()}, c.value("sample_rate", 16000), 1},
                    TextPartView{text}};
                const std::array<AudioTextMessage, 2> messages{
                    {{MessageRole::System, {system.data(), system.size()}},
                     {MessageRole::User, {user.data(), user.size()}}}};
                const auto out = dynamic_cast<IAudioTextToText&>(*owner).run(
                    {{messages.data(), messages.size()}}, cfg);
                result["text"] = out.text;
                result["tokens"] = out.token_ids;
            } else if (mode == "text") {
                const std::string text = c.at("text");
                const std::array<ConversationPartView, 1> system{
                    TextPartView{"Respond with text."}};
                const std::array<ConversationPartView, 1> user{TextPartView{text}};
                const std::array<ConversationMessageView, 2> messages{
                    {{MessageRole::System, {system.data(), system.size()}},
                     {MessageRole::User, {user.data(), user.size()}}}};
                const TextConversationRequest request{{messages.data(), messages.size()}, {}};
                const auto rendered = pipeline->text_conversation_prompt_tokens(request);
                if (rendered != c.at("prompt_text_tokens").get<std::vector<std::int32_t>>())
                    throw std::runtime_error(
                        "native text prompt differs from original ChatState before inference");
                // The public API client executes text generation after this
                // exact renderer check. Do not run a second private text path.
                result["prompt_text_tokens"] = rendered;
                result["prompt_tokens"] = rendered.size();
            } else if (mode == "speech_response") {
                const auto audio = read(c.at("audio"));
                const auto out = dynamic_cast<ISpeechToSpeechResponse&>(*owner).run(
                    {{{audio.data(), audio.size()}, c.value("sample_rate", 16000), 1}}, cfg);
                save(c.at("output"), out.samples);
                result["samples"] = out.samples.size();
            } else if (mode == "tts") {
                const std::string text = c.at("text");
                const auto out =
                    dynamic_cast<ITextToSpeech&>(*owner).run({text, std::string_view{"en"}}, cfg);
                save(c.at("output"), out.samples);
                result["samples"] = out.samples.size();
            } else if (mode == "tts_stream") {
                const std::string text = c.at("text");
                std::vector<float> audio;
                std::vector<std::size_t> chunks;
                const auto summary = dynamic_cast<IStreamingTextToSpeech&>(*owner).run(
                    {text, std::string_view{"en"}}, cfg, [&](const AudioView& chunk) {
                        audio.insert(audio.end(), chunk.samples.begin(), chunk.samples.end());
                        chunks.push_back(chunk.samples.size());
                        return true;
                    });
                save(c.at("output"), audio);
                result["samples"] = audio.size();
                result["chunks"] = chunks;
                if (summary.emitted_sample_count != audio.size())
                    throw std::runtime_error("stream summary count mismatch");
            } else if (mode == "chat") {
                auto session = dynamic_cast<IOfflineSpeechDialogue&>(*owner).create_offline_session(
                    {{c.value("sample_rate", 16000), 1}, std::nullopt}, cfg);
                json turns = json::array();
                int index = 0;
                for (const auto& turn : c.at("turns")) {
                    if (turn.contains("audio")) {
                        const auto audio = read(turn.at("audio"));
                        if (!session->append_audio({audio.data(), audio.size()}))
                            throw std::runtime_error("unexpected backpressure");
                    } else {
                        const std::string text = turn.at("text");
                        if (!session->append_text(text))
                            throw std::runtime_error("unexpected text backpressure");
                    }
                    session->realtime_control()->commit_input_turn();
                    auto batch = session->read_events(0);
                    std::string text;
                    std::vector<float> audio;
                    std::vector<std::size_t> chunks;
                    for (const auto& event : batch.events) {
                        if (event.kind == SpeechEventKind::AgentText)
                            text += event.text;
                        if (event.kind == SpeechEventKind::AgentAudio) {
                            audio.insert(audio.end(), event.audio.samples.begin(),
                                         event.audio.samples.end());
                            chunks.push_back(event.audio.samples.size());
                        }
                    }
                    const std::string output =
                        c.at("output_prefix").get<std::string>() + std::to_string(index++) + ".f32";
                    save(output, audio);
                    turns.push_back({{"text", text},
                                     {"samples", audio.size()},
                                     {"chunks", chunks},
                                     {"trace", trace_json(*pipeline)}});
                }
                result["turns"] = turns;
            } else
                throw std::invalid_argument("unknown mode");
            if (mode != "text")
                result["trace"] = trace_json(*pipeline);
            std::cout << result.dump() << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
