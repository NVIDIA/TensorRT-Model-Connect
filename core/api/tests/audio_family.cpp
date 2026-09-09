/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/internal/audio.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/family_factory.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace {
using namespace trtmc::internal;

struct Options {
    double gain{0};
    std::int64_t speaker{0};
    bool normalize{false};
    std::string preset;
    std::string suffix;
    std::int64_t max_output_tokens{224};
};

TextResult transcript(std::string text, const AudioView& audio, std::vector<std::int32_t> ids) {
    TextResult result;
    result.text = std::move(text);
    result.token_ids = std::move(ids);
    result.setup_ms = audio.channels;
    result.prefill_ms = static_cast<double>(audio.sample_rate.value()) / 1000;
    result.decode_ms = audio.samples.size();
    result.segments.push_back(
        {0, static_cast<double>(audio.samples.size() / audio.channels) / audio.sample_rate.value(),
         result.text, result.token_ids});
    return result;
}

class AudioFixture final : public IModel,
                           public ITextToAudio,
                           public ITextAudioTokenHistoryToAudio,
                           public IBatchTextAudioTokenHistoryToAudio,
                           public ITextToSpeech,
                           public ISpeechTranscription,
                           public ISpeechTranslation,
                           public IAudioLanguageIdentification,
                           public ISpeechToSpeechResponse,
                           public IBatchSpeechTranscription,
                           public IBatchSpeechTranslation,
                           public IMixedBatchSpeechToText,
                           public IBatchTextToAudio,
                           public IBatchTextToSpeech {
  public:
    explicit AudioFixture(std::string mode) : mode_(std::move(mode)) {}
    const char* task() const noexcept override { return mode_.c_str(); }
    std::vector<TaskInfo> task_info() const override {
        if (mode_ == "asr_only")
            return {{ISpeechTranscription::kTask, 1, 0}};
        if (mode_ == "history_single_only")
            return {{ITextAudioTokenHistoryToAudio::kTask, 1, 0}};
        if (mode_ == "audio_single_only")
            return {{ITextToAudio::kTask, 1, 0}, {ITextToSpeech::kTask, 1, 0}};
        if (mode_ == "batch_audio_only")
            return {{IBatchTextToAudio::kTask, 1, 0}};
        if (mode_ == "batch_speech_only")
            return {{IBatchTextToSpeech::kTask, 1, 0}};
        return {{ITextToAudio::kTask, 1, 0},
                {ITextAudioTokenHistoryToAudio::kTask, 1, 0},
                {IBatchTextAudioTokenHistoryToAudio::kTask, 1, 0},
                {ITextToSpeech::kTask, 1, 0},
                {ISpeechTranscription::kTask, 1, 0},
                {ISpeechTranslation::kTask, 1, 0},
                {IAudioLanguageIdentification::kTask, 1, 0},
                {ISpeechToSpeechResponse::kTask, 1, 0},
                {IBatchSpeechTranscription::kTask, 1, 0},
                {IBatchSpeechTranslation::kTask, 1, 0},
                {IMixedBatchSpeechToText::kTask, 1, 0},
                {IBatchTextToAudio::kTask, 1, 0},
                {IBatchTextToSpeech::kTask, 1, 0}};
    }
    std::vector<ConfigField> config_fields(std::string_view task) const override {
        const auto tasks = task_info();
        if (std::none_of(tasks.begin(), tasks.end(),
                         [&](const TaskInfo& info) { return info.id == task; }))
            throw UnsupportedTask("audio fixture Task is disabled");
        if (task == IAudioLanguageIdentification::kTask)
            return {};
        if (task == ITextAudioTokenHistoryToAudio::kTask ||
            task == IBatchTextAudioTokenHistoryToAudio::kTask)
            return {{"gain", ConfigKind::F64, ConfigValue{1.0}, "Output gain"},
                    {"normalize", ConfigKind::Bool, ConfigValue{true}, "Boolean marker"}};
        if (task == ITextToAudio::kTask || task == IBatchTextToAudio::kTask)
            return {{"gain", ConfigKind::F64, ConfigValue{1.0}, "Output gain"},
                    {"preset", ConfigKind::String, ConfigValue{std::string_view{}}, "Preset"}};
        if (task == ITextToSpeech::kTask || task == IBatchTextToSpeech::kTask)
            return {{"gain", ConfigKind::F64, ConfigValue{1.0}, "Output gain"},
                    {"speaker", ConfigKind::I64, ConfigValue{std::int64_t{0}}, "Speaker index"},
                    {"normalize", ConfigKind::Bool, ConfigValue{true}, "Text normalization"}};
        if (task == ISpeechToSpeechResponse::kTask)
            return {{"gain", ConfigKind::F64, ConfigValue{1.0}, "Response gain"}};
        if (task == ISpeechTranscription::kTask || task == ISpeechTranslation::kTask)
            return {{"suffix", ConfigKind::String, ConfigValue{std::string_view{"!"}},
                     "Transcript suffix"},
                    {"max_output_tokens", ConfigKind::I64, ConfigValue{std::int64_t{224}},
                     "Output token limit"}};
        return {{"suffix", ConfigKind::String, ConfigValue{std::string_view{"!"}},
                 "Transcript suffix"}};
    }

    AudioResult run(const TextToAudioRequest& input, ConfigView config) override {
        const auto options = parse(config, ITextToAudio::kTask);
        ++single_audio_;
        return generated({static_cast<float>(input.prompt.size()) / 100,
                          static_cast<float>(options.preset.size()) / 100, 0.25F, -0.25F},
                         options);
    }
    AudioResult run(const TextToSpeechRequest& input, ConfigView config) override {
        const auto lang = resolve(input.language);
        const auto options = parse(config, ITextToSpeech::kTask);
        ++single_speech_;
        return generated({static_cast<float>(input.text.size()) / 100,
                          static_cast<float>(options.speaker) / 10, lang == "en" ? 0.1F : 0.2F,
                          options.normalize ? 0.25F : 0.0F},
                         options);
    }
    AudioResult run(const TextAudioTokenHistoryToAudioRequest& input, ConfigView config) override {
        validate_history(input.history);
        const auto options = history_options(config, ITextAudioTokenHistoryToAudio::kTask);
        auto result = history_output(input.prompt, input.history, options);
        result.setup_ms = ++history_single_;
        result.inference_ms = single_audio_;
        return result;
    }
    TextResult run(const SpeechTranscriptionRequest& input, ConfigView config) override {
        const auto options = parse(config, ISpeechTranscription::kTask);
        const auto audio = resolve_audio(input.audio);
        ++single_asr_;
        auto result = transcript(
            "asr:" + std::string(input.source_language.value_or("auto")) + options.suffix, audio,
            {single_asr_, static_cast<std::int32_t>(*audio.sample_rate),
             static_cast<std::int32_t>(audio.channels)});
        for (const auto& entry : config)
            if (entry.name == "max_output_tokens")
                result.decode_ms = static_cast<double>(options.max_output_tokens);
        return result;
    }
    TextResult run(const SpeechTranslationRequest& input, ConfigView config) override {
        const auto target = resolve(input.target_language);
        const auto options = parse(config, ISpeechTranslation::kTask);
        const auto audio = resolve_audio(input.audio);
        ++single_translation_;
        auto result =
            transcript("translate:" + std::string(input.source_language.value_or("auto")) + "->" +
                           target + options.suffix,
                       audio, {single_translation_, 2, 3});
        for (const auto& entry : config)
            if (entry.name == "max_output_tokens")
                result.decode_ms = static_cast<double>(options.max_output_tokens);
        return result;
    }
    LabelScoresResult run(const AudioLanguageIdentificationRequest& input,
                          ConfigView config) override {
        (void)resolve_audio(input.audio);
        (void)parse(config, IAudioLanguageIdentification::kTask);
        LabelScoresResult result{
            {0.75F, 0.25F}, {"en", "fr"}, ScoreKind::Probability, "fixture_languages"};
        if (mode_ == "bad_labels")
            result.labels.clear();
        return result;
    }
    AudioResult run(const SpeechToSpeechResponseRequest& input, ConfigView config) override {
        const auto options = parse(config, ISpeechToSpeechResponse::kTask);
        const auto audio = resolve_audio(input.audio);
        AudioResult output;
        for (auto i = audio.samples.size(); i > 0; --i)
            output.samples.push_back(audio.samples[i - 1] * static_cast<float>(options.gain));
        output.sample_rate = *audio.sample_rate;
        output.channels = audio.channels;
        output.inference_ms = 4;
        return output;
    }
    BatchTextResult run_batch(const BatchSpeechTranscriptionRequest& input) override {
        std::vector<Options> options;
        std::vector<AudioView> audios;
        for (const auto& item : input.items) {
            options.push_back(parse(item.config, IBatchSpeechTranscription::kTask));
            audios.push_back(resolve_audio(item.input.audio));
        }
        ++batch_asr_;
        BatchTextResult output;
        for (std::size_t i = 0; i < input.items.size(); ++i) {
            const auto& request = input.items[i].input;
            output.push_back(
                transcript("batch_asr:" + std::string(request.source_language.value_or("auto")) +
                               options[i].suffix,
                           audios[i], {batch_asr_, static_cast<std::int32_t>(i), single_asr_}));
        }
        return output;
    }
    BatchTextResult run_batch(const BatchSpeechTranslationRequest& input) override {
        std::vector<Options> options;
        std::vector<std::string> targets;
        std::vector<AudioView> audios;
        for (const auto& item : input.items) {
            options.push_back(parse(item.config, IBatchSpeechTranslation::kTask));
            targets.push_back(resolve(item.input.target_language));
            audios.push_back(resolve_audio(item.input.audio));
        }
        ++batch_translation_;
        BatchTextResult output;
        for (std::size_t i = 0; i < input.items.size(); ++i) {
            const auto& request = input.items[i].input;
            output.push_back(transcript(
                "batch_translate:" + std::string(request.source_language.value_or("auto")) + "->" +
                    targets[i] + options[i].suffix,
                audios[i],
                {batch_translation_, static_cast<std::int32_t>(i), single_translation_}));
        }
        return output;
    }

    BatchTextResult run_batch(const MixedBatchSpeechToTextRequest& input) override {
        std::vector<AudioView> audios;
        std::vector<std::string> prepared;
        for (const auto& item : input.items) {
            const auto options = parse(item.config, IMixedBatchSpeechToText::kTask);
            if (const auto* asr = std::get_if<SpeechTranscriptionRequest>(&item.input)) {
                audios.push_back(resolve_audio(asr->audio));
                prepared.push_back(
                    "mixed_asr:" + std::string(asr->source_language.value_or("auto")) +
                    options.suffix);
            } else {
                const auto& translation = std::get<SpeechTranslationRequest>(item.input);
                audios.push_back(resolve_audio(translation.audio));
                prepared.push_back(
                    "mixed_translate:" + std::string(translation.source_language.value_or("auto")) +
                    "->" + resolve(translation.target_language) + options.suffix);
            }
        }
        ++mixed_calls_;
        BatchTextResult output;
        for (std::size_t i = 0; i < prepared.size(); ++i)
            output.push_back(transcript(std::move(prepared[i]), audios[i],
                                        {mixed_calls_, static_cast<std::int32_t>(i), single_asr_,
                                         single_translation_, batch_asr_, batch_translation_}));
        return output;
    }

    BatchAudioResult run_batch(const BatchTextAudioTokenHistoryToAudioRequest& input) override {
        validate_history(input.history);
        std::vector<Options> options;
        for (size_t index = 0; index < input.items.size(); ++index) {
            try {
                options.push_back(history_options(input.items[index].config,
                                                  IBatchTextAudioTokenHistoryToAudio::kTask));
            } catch (const ConfigError& error) {
                throw ConfigError("batch item[" + std::to_string(index) + "]: " + error.what());
            }
        }
        if (mode_ == "history_uniform_gain")
            for (size_t index = 1; index < options.size(); ++index)
                if (options[index].gain != options.front().gain)
                    throw ConfigError("history fixture batch requires uniform gain");
        ++history_batch_;
        if (mode_ == "history_execution_error")
            throw std::runtime_error("history fixture execution failed");
        BatchAudioResult output;
        for (size_t index = 0; index < input.items.size(); ++index) {
            auto result =
                history_output(input.items[index].input.prompt, input.history, options[index]);
            result.samples.resize(10 + index, 0); // Actual unequal lengths, not batch padding.
            result.setup_ms = history_batch_;
            result.inference_ms = history_single_;
            if (mode_ == "history_bad_pcm" && index == 1)
                result.channels = 0;
            output.push_back(std::move(result));
        }
        if (mode_ == "history_bad_count")
            output.pop_back();
        return output;
    }

    BatchAudioResult run_batch(const BatchTextToAudioRequest& input) override {
        std::vector<Options> options;
        for (const auto& item : input.items)
            options.push_back(parse(item.config, IBatchTextToAudio::kTask));
        if (mode_ == "uniform_preset") {
            for (std::size_t i = 1; i < options.size(); ++i)
                if (options[i].preset != options[0].preset)
                    throw ConfigError("fixture native batch requires one resolved preset");
        }
        ++batch_audio_;
        BatchAudioResult result;
        for (std::size_t i = 0; i < input.items.size(); ++i) {
            auto audio = batch_output(i, batch_audio_, single_audio_);
            audio.samples[0] = static_cast<float>(input.items[i].input.prompt.size()) / 100;
            audio.samples[1] = static_cast<float>(options[i].preset.size()) / 100;
            for (auto& sample : audio.samples)
                sample *= static_cast<float>(options[i].gain);
            result.push_back(std::move(audio));
        }
        return finish_generated_batch(std::move(result));
    }
    BatchAudioResult run_batch(const BatchTextToSpeechRequest& input) override {
        std::vector<Options> options;
        std::vector<std::string> languages;
        for (const auto& item : input.items) {
            options.push_back(parse(item.config, IBatchTextToSpeech::kTask));
            languages.push_back(resolve(item.input.language));
        }
        ++batch_speech_;
        BatchAudioResult result;
        for (std::size_t i = 0; i < input.items.size(); ++i) {
            auto audio = batch_output(i, batch_speech_, single_speech_);
            audio.samples[0] = static_cast<float>(input.items[i].input.text.size()) / 100;
            audio.samples[1] = static_cast<float>(options[i].speaker) / 10;
            audio.samples[2] = languages[i] == "en" ? 0.1F : 0.2F;
            audio.samples[3] = options[i].normalize ? 0.25F : 0;
            for (auto& sample : audio.samples)
                sample *= static_cast<float>(options[i].gain);
            result.push_back(std::move(audio));
        }
        return finish_generated_batch(std::move(result));
    }

  private:
    BatchAudioResult finish_generated_batch(BatchAudioResult result) const {
        if (mode_ == "batch_execution_error")
            throw std::runtime_error("fixture native batch execution failed");
        if (mode_ == "bad_generated_count" && !result.empty())
            result.pop_back();
        if (mode_ == "bad_generated_metadata" && !result.empty())
            result.back().channels = 0;
        if (mode_ == "bad_generated_rate" && !result.empty())
            result.back().sample_rate = 0;
        if (mode_ == "bad_generated_frame" && !result.empty()) {
            result.back().channels = 2;
            if (result.back().samples.size() % 2 == 0)
                result.back().samples.pop_back();
        }
        if (mode_ == "empty_generated_item" && !result.empty())
            result.back().samples.clear();
        return result;
    }
    static AudioResult batch_output(std::size_t index, std::int32_t batch_calls,
                                    std::int32_t single_calls) {
        AudioResult result;
        result.channels = index % 2 ? 2 : 1;
        result.sample_rate = index % 2 ? 16000 : 24000;
        result.samples.assign((index + 4) * result.channels, -0.25F);
        // Protocol-only observable counters, not measurements or model audio.
        result.setup_ms = batch_calls;
        result.inference_ms = single_calls;
        return result;
    }
    static void validate_history(const SemanticAcousticHistoryView& history) {
        if (history.semantic_tokens.empty() || history.coarse_tokens.codebooks != 2 ||
            history.fine_tokens.codebooks != 8 || history.coarse_tokens.frames == 0 ||
            history.fine_tokens.frames == 0)
            throw std::invalid_argument(
                "fixture requires nonempty semantic and 2/8 acoustic history");
        const auto vocabulary = [](trtmc::Span<const std::int32_t> values, std::int32_t limit) {
            for (const auto id : values)
                if (id < 0 || id >= limit)
                    throw std::invalid_argument("token history is outside the fixture vocabulary");
        };
        vocabulary(history.semantic_tokens, 10000);
        vocabulary(history.coarse_tokens.tokens, 1024);
        vocabulary(history.fine_tokens.tokens, 1024);
    }
    Options history_options(ConfigView config, std::string_view task) const {
        for (const auto& entry : config)
            if (entry.name == "voice_preset" || entry.name == "preset")
                throw ConfigError("preset conflicts with explicit token history");
        return parse(config, task);
    }
    static AudioResult history_output(std::string_view prompt,
                                      const SemanticAcousticHistoryView& history,
                                      const Options& options) {
        const auto semantic = history.semantic_tokens;
        const auto coarse = history.coarse_tokens;
        const auto fine = history.fine_tokens;
        std::vector<float> samples{
            static_cast<float>(semantic[0]),
            static_cast<float>(semantic[semantic.size() - 1]),
            static_cast<float>(coarse.tokens[0]),
            static_cast<float>(coarse.tokens[coarse.frames]),
            static_cast<float>(coarse.tokens[coarse.tokens.size() - 1]),
            static_cast<float>(fine.tokens[0]),
            static_cast<float>(fine.tokens[(fine.codebooks - 1) * fine.frames]),
            static_cast<float>(fine.tokens[fine.tokens.size() - 1]),
            static_cast<float>(prompt.size()),
            options.normalize ? 1.0F : 0.0F};
        for (auto& sample : samples)
            sample *= static_cast<float>(options.gain);
        return {std::move(samples), 24000, 1, 0, 0};
    }
    AudioView resolve_audio(AudioView input) const {
        if (!input.sample_rate) {
            if (mode_ == "explicit_sample_rate")
                throw std::invalid_argument("fixture bundle has no input sample-rate default");
            input.sample_rate = 16000;
        }
        return input;
    }
    std::string resolve(const std::optional<std::string_view>& language) const {
        if (language)
            return std::string(*language);
        if (mode_ == "explicit_languages")
            throw std::invalid_argument("fixture bundle has no language default");
        return "en";
    }
    AudioResult generated(std::vector<float> samples, const Options& options) const {
        for (auto& sample : samples)
            sample *= static_cast<float>(options.gain);
        return {std::move(samples), 24000, mode_ == "bad_audio" ? 0U : 2U, 1, 2};
    }
    Options parse(ConfigView config, std::string_view task) const {
        const auto fields = config_fields(task);
        Options options;
        const auto assign = [&](std::string_view name, const ConfigValue& value) {
            if (name == "gain")
                options.gain = std::get<double>(value);
            if (name == "speaker")
                options.speaker = std::get<std::int64_t>(value);
            if (name == "normalize")
                options.normalize = std::get<bool>(value);
            if (name == "preset")
                options.preset = std::get<std::string_view>(value);
            if (name == "suffix")
                options.suffix = std::get<std::string_view>(value);
            if (name == "max_output_tokens")
                options.max_output_tokens = std::get<std::int64_t>(value);
        };
        for (const auto& field : fields)
            assign(field.name, *field.default_value);
        std::set<std::string_view> seen;
        for (const auto& entry : config) {
            const auto field =
                std::find_if(fields.begin(), fields.end(),
                             [&](const ConfigField& field) { return field.name == entry.name; });
            if (field == fields.end())
                throw ConfigError("unknown audio config");
            if (!seen.insert(entry.name).second)
                throw ConfigError("duplicate audio config");
            if (config_kind(entry.value) != field->kind)
                throw ConfigError("wrong audio config type");
            assign(entry.name, entry.value);
        }
        if (!std::isfinite(options.gain) || options.gain < 0)
            throw ConfigError("gain must be finite and nonnegative");
        if (options.speaker < 0 || options.speaker > 1)
            throw ConfigError("fixture supports two speaker indices");
        return options;
    }
    std::string mode_;
    std::int32_t single_asr_{0}, single_translation_{0}, batch_asr_{0}, batch_translation_{0},
        mixed_calls_{0};
    std::int32_t single_audio_{0}, single_speech_{0}, batch_audio_{0}, batch_speech_{0};
    std::int32_t history_single_{0}, history_batch_{0};
};
class DeclaredHistoryOnly final : public IModel {
  public:
    const char* task() const noexcept override { return "history_declared_only"; }
    std::vector<TaskInfo> task_info() const override {
        return {{ITextAudioTokenHistoryToAudio::kTask, 1, 0},
                {IBatchTextAudioTokenHistoryToAudio::kTask, 1, 0}};
    }
    std::vector<ConfigField> config_fields(std::string_view) const override { return {}; }
};
} // namespace

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.reader.info().family != "audio_fixture")
        throw std::invalid_argument("unexpected audio fixture family");
    if (context.reader.info().task == "history_declared_only")
        return new DeclaredHistoryOnly;
    return new AudioFixture(context.reader.info().task);
}
