/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/language.hpp"
#include "trtmc/speech.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

using json = nlohmann::json;

std::vector<float> read_audio(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() < 0 || input.tellg() % 4)
        throw std::runtime_error("invalid PCM fixture");
    std::vector<float> samples(static_cast<std::size_t>(input.tellg()) / 4);
    input.seekg(0);
    input.read(reinterpret_cast<char*>(samples.data()), samples.size() * 4);
    return samples;
}

void save_audio(const std::string& path, const std::vector<float>& samples) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(samples.data()), samples.size() * 4);
    if (!output)
        throw std::runtime_error("could not write PCM result");
}

int main(int argc, char** argv) {
    try {
        if (argc != 4)
            throw std::invalid_argument("usage: api_probe BUNDLE RUNTIME_ROOT CASES.json");
        trtmc::LoadOptions options;
        options.runtime_root = argv[2];
        auto model = trtmc::Model::load(argv[1], options);
        json cases;
        std::ifstream(argv[3]) >> cases;
        for (const auto& item : cases) {
            const std::string mode = item.at("mode");
            if (mode != "chat" && mode != "text" && mode != "mixed")
                continue;
            const trtmc::Config config{
                {"max_new_tokens", item.value("max_new_tokens", std::int64_t{512})},
                {"text_temperature", item.value("text_temperature", 0.0)},
                {"text_top_k", item.value("text_top_k", std::int64_t{1})},
                {"audio_temperature", item.value("audio_temperature", 0.0)},
                {"audio_top_k", item.value("audio_top_k", std::int64_t{1})},
                {"seed", item.value("seed", std::int64_t{0})}};
            if (mode == "mixed") {
                const auto samples = read_audio(item.at("audio"));
                const trtmc::AudioView audio{{samples.data(), samples.size()},
                                             item.value("sample_rate", std::uint32_t{16000}),
                                             1};
                const trtmc::AudioTextToTextRequest request{
                    {{trtmc::MessageRole::System, {trtmc::TextPart{"Respond with text."}}},
                     {trtmc::MessageRole::User,
                      {audio, trtmc::TextPart{item.at("text").get<std::string>()}}}}};
                const auto result = model.task<trtmc::AudioTextToText>().run(request, config);
                const auto tokens = result.token_ids();
                std::cout << json{{"name", item.at("name")},
                                  {"text", result.text()},
                                  {"tokens",
                                   std::vector<std::int32_t>(tokens.begin(), tokens.end())}}
                                 .dump()
                          << '\n';
                continue;
            }
            if (mode == "text") {
                const trtmc::TextConversationRequest request{
                    {{trtmc::MessageRole::System, {trtmc::TextPart{"Respond with text."}}},
                     {trtmc::MessageRole::User,
                      {trtmc::TextPart{item.at("text").get<std::string>()}}}},
                    {}};
                if (!item.contains("context_tokens")) {
                    for (bool tools : {false, true}) {
                        auto unsupported = request;
                        if (tools)
                            unsupported.tools.push_back(
                                {"echo", "Return text", R"({"type":"object","properties":{}})"});
                        else
                            unsupported.messages.insert(unsupported.messages.end() - 1,
                                                        {trtmc::MessageRole::Assistant,
                                                         {trtmc::ReasoningPart{"reasoning"}}});
                        bool rejected = false;
                        std::string rejection;
                        try {
                            (void)model.task<trtmc::TextConversation>().run(unsupported, config);
                        } catch (const trtmc::Error& error) {
                            rejected = error.code() == TRTMC_UNSUPPORTED;
                            rejection = error.what();
                        }
                        if (!rejected)
                            throw std::runtime_error(
                                "unsupported text conversation parts must fail closed: " +
                                rejection);
                    }
                }
                const auto result = model.task<trtmc::TextConversation>().run(request, config);
                const auto tokens = result.result().token_ids();
                const auto message = result.message();
                if (message.parts.size() != 1 ||
                    !std::holds_alternative<trtmc::TextPart>(message.parts[0]))
                    throw std::runtime_error("expected one text response part");
                std::cout << json{{"name", item.at("name")},
                                  {"text", std::get<trtmc::TextPart>(message.parts[0]).text},
                                  {"tokens",
                                   std::vector<std::int32_t>(tokens.begin(), tokens.end())},
                                  {"input_tokens", result.usage().input_tokens.value()}}
                                 .dump()
                          << '\n';
                continue;
            }
            auto session = model.task<trtmc::OfflineSpeechDialogue>().create(
                {{item.value("sample_rate", std::uint32_t{16000}), 1}, std::nullopt}, config);
            const auto info = session.info();
            if (!info.output || info.output->sample_rate != 24000 || info.output->channels != 1)
                throw std::runtime_error("incorrect session PCM contract");
            json turns = json::array();
            std::size_t index = 0;
            for (const auto& turn : item.at("turns")) {
                bool accepted;
                if (turn.contains("audio")) {
                    const auto samples = read_audio(turn.at("audio"));
                    accepted = session.append_audio({samples.data(), samples.size()});
                } else {
                    accepted = session.append_text(turn.at("text").get<std::string>());
                }
                if (!accepted)
                    throw std::runtime_error("unexpected session input backpressure");
                session.realtime().commit_input_turn();
                auto events = session.take_events();
                if (!events.events)
                    throw std::runtime_error("committed turn produced no events");
                std::string text;
                std::vector<float> audio;
                bool started = false, finished = false;
                for (std::size_t i = 0; i < events.events->size(); ++i) {
                    const auto event = events.events->at(i);
                    started |= event.kind == trtmc::SpeechEventKind::TurnStarted;
                    finished |= event.kind == trtmc::SpeechEventKind::TurnFinished;
                    if (event.kind == trtmc::SpeechEventKind::AgentText)
                        text.append(event.text);
                    if (event.kind == trtmc::SpeechEventKind::AgentAudio)
                        audio.insert(audio.end(), event.audio.samples.begin(),
                                     event.audio.samples.end());
                }
                if (!started || !finished || audio.empty())
                    throw std::runtime_error("missing dialogue turn or audio events");
                const auto path = item.at("output_prefix").get<std::string>() +
                                  std::to_string(index++) + ".api.f32";
                save_audio(path, audio);
                turns.push_back({{"text", text}, {"samples", audio.size()}, {"waveform", path}});
            }
            session.finish_input();
            const auto closed = session.take_events();
            if (!closed.events || closed.events->size() != 1 ||
                closed.events->at(0).kind != trtmc::SpeechEventKind::InputFinished)
                throw std::runtime_error(
                    "finish_input after completed turns must only close input");
            session.finish_input();
            if (session.take_events().status != trtmc::SpeechPollStatus::EpochEnd)
                throw std::runtime_error("repeated finish_input must be idempotent");
            session.close();
            std::cout << json{{"name", item.at("name")}, {"turns", turns}}.dump() << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
