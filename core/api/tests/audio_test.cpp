/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/audio.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
int failures;
void check(bool passed, const char* label) {
    if (!passed) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}
template <class Function>
bool rejects(Function function, trtmc_status code) {
    try {
        function();
    } catch (const trtmc::Error& error) {
        return error.code() == code;
    }
    return false;
}
void write_bundle(const std::filesystem::path& path, const std::string& mode) {
    const unsigned char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    const std::string header =
        "{\"format\":1,\"family\":\"audio_fixture\",\"task\":\"" + mode +
        "\",\"backend\":\"fake\",\"sections\":{\"engine.plan\":{\"offset\":0,\"length\":4}}}";
    std::ofstream output(path, std::ios::binary);
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.write(reinterpret_cast<const char*>(magic), sizeof(magic));
    for (unsigned shift = 0; shift < 64; shift += 8)
        output.put(static_cast<char>((static_cast<std::uint64_t>(header.size()) >> shift) & 255U));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write("PLAN", 4);
}

void test_stateless(const trtmc::Model& model, trtmc::AudioView audio) {
    check(model.tasks().size() == 13, "all stateless audio contracts are discoverable");
    auto generated = model.task<trtmc::TextToAudio>().run({"wind"}, {{"preset", "calm"}});
    check(generated.channels() == 2 && generated.sample_rate() == 24000 &&
              generated.frame_count() == 2,
          "audio output retains stereo rate and scalar sample count");
    check(generated.samples()[0] == 0.04F && generated.samples()[1] == 0.04F,
          "text-to-audio prompt and preset reach family");
    auto spoken = model.task<trtmc::TextToSpeech>().run({"Hello", std::string{"fr"}},
                                                        {{"speaker", 1}, {"normalize", false}});
    check(spoken.samples()[0] == 0.05F && spoken.samples()[1] == 0.1F &&
              spoken.samples()[2] == 0.2F && spoken.samples()[3] == 0,
          "speech text language speaker and normalization remain distinct inputs/config");
    auto default_speech = model.task<trtmc::TextToSpeech>().run({"Hello"});
    check(default_speech.samples()[2] == 0.1F,
          "family supplies declared synthesis language default");
    check(rejects([&] { (void)model.task<trtmc::TextToSpeech>().run({"Hello", std::string{}}); },
                  TRTMC_INVALID_ARGUMENT),
          "empty explicit synthesis language rejected");

    auto transcript =
        model.task<trtmc::SpeechTranscription>().run({audio, std::string{"fr"}}, {{"suffix", "?"}});
    check(transcript.text() == "asr:fr?" && transcript.token_ids()[1] == 16000 &&
              transcript.token_ids()[2] == 2,
          "known source language and stereo input pass through unchanged");
    check(transcript.segments().size() == 1 && transcript.segments()[0].text == "asr:fr?" &&
              transcript.setup_ms() == 2 && transcript.prefill_ms() == 16 &&
              transcript.decode_ms() == 4,
          "transcript segments token IDs and timings preserved");
    auto translated =
        model.task<trtmc::SpeechTranslation>().run({audio, std::string{"de"}, std::string{"fr"}});
    check(translated.text() == "translate:fr->de!", "speech translation keeps both language roles");
    check(model.task<trtmc::SpeechTranslation>().run({audio}).text() == "translate:auto->en!",
          "speech translation family defaults work when languages omitted");
    auto language = model.task<trtmc::AudioLanguageIdentification>().run({audio});
    check(language.labels() == std::vector<std::string_view>{"en", "fr"} &&
              language.scores()[0] == 0.75F && language.kind() == TRTMC_SCORE_PROBABILITY &&
              language.vocabulary_id() == "fixture_languages",
          "language IDs scores and score semantics remain human-interpretable");
    auto response = model.task<trtmc::SpeechToSpeechResponse>().run({audio}, {{"gain", 2.0}});
    check(response.channels() == 2 && response.sample_rate() == 16000 &&
              response.samples()[0] == 0.8F,
          "one-shot response owns its actual PCM rate and channels");
    auto moved = std::move(response);
    check(moved.samples()[0] == 0.8F && response.samples().empty(),
          "audio owner move clears source views");
    auto unspecified = audio;
    unspecified.sample_rate.reset();
    check(model.task<trtmc::SpeechTranscription>().run({unspecified}).token_ids()[1] == 16000,
          "omitted input rate uses the family's declared fixed rate");
    check(model.task<trtmc::SpeechToSpeechResponse>().run({unspecified}).sample_rate() == 16000,
          "one-shot speech response preserves the family input-rate default");
    auto explicit_rate = audio;
    explicit_rate.sample_rate = 22050;
    check(model.task<trtmc::SpeechTranscription>().run({explicit_rate}).token_ids()[1] == 22050,
          "explicit input rate is not replaced by the shared layer");
    explicit_rate.sample_rate = 0;
    check(rejects([&] { (void)model.task<trtmc::SpeechTranscription>().run({explicit_rate}); },
                  TRTMC_INVALID_ARGUMENT),
          "explicit zero rate is not an omitted rate");
}

void test_history(const std::string& path, const trtmc::LoadOptions& options) {
    using namespace trtmc;
    auto model = Model::load(path, options);
    std::int32_t semantic[]{11, 12, 13}, coarse[]{21, 22, 23, 24};
    std::vector<std::int32_t> fine(24);
    for (size_t i = 0; i < fine.size(); ++i)
        fine[i] = static_cast<int32_t>(100 + i);
    const SemanticAcousticHistoryView history{
        {semantic, 3}, {{coarse, 4}, 2, 2}, {{fine.data(), fine.size()}, 8, 3}};
    const TextAudioTokenHistoryToAudioRequest request{"hello", history};
    auto task = model.task<TextAudioTokenHistoryToAudio>();
    auto result = task.run(request);
    const std::vector<float> expected{11, 13, 21, 23, 24, 100, 121, 123, 5, 1};
    check(
        std::vector<float>(result.samples().begin(), result.samples().end()) == expected &&
            result.sample_rate() == 24000 && result.channels() == 1 && result.setup_ms() == 1 &&
            result.inference_ms() == 0,
        "typed history preserves three roles, codebook-major order and independent frame lengths");
    check(semantic[0] == 11 && coarse[3] == 24 && fine[23] == 123,
          "generation must not offset, transpose or mutate caller history");
    check(task.config_fields().size() == 2,
          "history Task discovers its own family options rather than preset-only config");
    check(rejects([&] { task.run(request, {{"voice_preset", "speaker"}}); }, TRTMC_INVALID_CONFIG),
          "family rejects named preset conflict instead of silently dropping typed history");
    check(rejects([&] { task.run(request, {{"gain", 1.0}, {"gain", 0.0}}); }, TRTMC_INVALID_CONFIG),
          "history config duplicates reach family validation");
    auto bad = request;
    bad.history.coarse_tokens.frames = 3;
    check(rejects([&] { task.run(bad); }, TRTMC_INVALID_ARGUMENT),
          "history count/shape mismatch fails");
    bad = request;
    bad.history.fine_tokens = {{}, 8, 0};
    check(rejects([&] { task.run(bad); }, TRTMC_INVALID_ARGUMENT),
          "representable empty history reaches family policy, not an unconditioned fallback");
    coarse[0] = -1;
    check(rejects([&] { task.run(request); }, TRTMC_INVALID_ARGUMENT),
          "history vocabulary is family-owned");
    coarse[0] = 21;
    auto zero = task.run(request, {{"gain", 0.0}, {"normalize", false}});
    check(zero.setup_ms() == 2 && std::all_of(zero.samples().begin(), zero.samples().end(),
                                              [](float value) { return value == 0; }),
          "explicit zero/false is preserved and invalid history/config did not execute");
    auto owned = [&] {
        return Model::load(path, options).task<TextAudioTokenHistoryToAudio>().run(request);
    }();
    semantic[0] = 999;
    coarse[0] = 999;
    fine[0] = 999;
    check(owned.samples()[0] == 11 && owned.samples()[2] == 21 && owned.samples()[5] == 100,
          "PCM result owns values after request mutation and model/proxy release");
}

void test_history_batch(const std::string& path, const trtmc::LoadOptions& options,
                        const std::filesystem::path& root) {
    using namespace trtmc;
    auto model = Model::load(path, options);
    std::int32_t semantic[]{11, 12, 13}, coarse[]{21, 22, 23, 24};
    std::vector<std::int32_t> fine(24);
    for (size_t i = 0; i < fine.size(); ++i)
        fine[i] = static_cast<std::int32_t>(100 + i);
    const SemanticAcousticHistoryView history{
        {semantic, 3}, {{coarse, 4}, 2, 2}, {{fine.data(), fine.size()}, 8, 3}};
    const BatchTextAudioTokenHistoryToAudioRequest request{
        history, {{{"first"}, {}}, {{"second!"}, {{"gain", 0.5}, {"normalize", false}}}}};
    auto task = model.task<BatchTextAudioTokenHistoryToAudio>();
    auto result = task.run(request);
    check(result.size() == 2 && result[0].samples.size() == 10 && result[1].samples.size() == 11 &&
              result[0].sample_rate == 24000 && result[1].channels == 1 &&
              result[0].samples[0] == 11 && result[1].samples[0] == 5.5F &&
              result[0].samples[6] == 121 && result[1].samples[6] == 60.5F &&
              result[0].samples[8] == 5 && result[1].samples[8] == 3.5F &&
              result[1].samples[9] == 0,
          "batch broadcasts exactly one typed history while preserving prompt/config and actual "
          "audio lengths");
    check(result[0].setup_ms == 1 && result[1].setup_ms == 1 && result[0].inference_ms == 0 &&
              result[1].inference_ms == 0,
          "history batch executes one family method with no scalar history calls");
    check(semantic[0] == 11 && coarse[3] == 24 && fine[23] == 123,
          "batch never offsets or modifies caller token histories");
    check(task.config_fields().size() == 2, "batch metadata describes per-item family config");
    auto bad = request;
    bad.items[1].config = {{"voice_preset", "speaker"}};
    check(rejects([&] { task.run(bad); }, TRTMC_INVALID_CONFIG),
          "preset conflict rejects a later batch item");
    bad = request;
    bad.items[1].config = {{"normalize", false}, {"normalize", true}};
    check(rejects([&] { task.run(bad); }, TRTMC_INVALID_CONFIG),
          "duplicate item options are not merged");
    bad = request;
    bad.items.clear();
    check(rejects([&] { task.run(bad); }, TRTMC_INVALID_ARGUMENT),
          "empty history batch is not scalar generation");
    fine[0] = 1024;
    check(rejects([&] { task.run(request); }, TRTMC_INVALID_ARGUMENT),
          "shared history vocabulary failure aborts the batch");
    fine[0] = 100;
    check(task.run(request)[0].setup_ms == 2,
          "preflight failures never start a partial history batch");
    for (const auto& mode : {"history_uniform_gain", "history_bad_count", "history_bad_pcm",
                             "history_execution_error"}) {
        const auto file = root / (std::string(mode) + ".bundle");
        write_bundle(file, mode);
        auto fixture = Model::load(file.string(), options);
        check(rejects([&] { fixture.task<BatchTextAudioTokenHistoryToAudio>().run(request); },
                      std::string(mode) == "history_uniform_gain" ? TRTMC_INVALID_CONFIG
                                                                  : TRTMC_INTERNAL_ERROR),
              "unsupported option combination or provider output/execution failure has no partial "
              "success");
    }
    const auto single_path = root / "history_single_only.bundle";
    write_bundle(single_path, "history_single_only");
    auto single = Model::load(single_path.string(), options);
    check(single.supports<TextAudioTokenHistoryToAudio>() &&
              !single.supports<BatchTextAudioTokenHistoryToAudio>(),
          "scalar history support does not imply batch support");
    const auto declared_path = root / "history_declared_only.bundle";
    write_bundle(declared_path, "history_declared_only");
    check(Model::load(declared_path.string(), options).tasks().empty(),
          "history tasks without a bound implementation are not exposed");
    auto owned = Model::load(path, options).task<BatchTextAudioTokenHistoryToAudio>().run(request);
    semantic[0] = coarse[0] = fine[0] = 999;
    check(owned[0].samples[0] == 11 && owned[1].samples[2] == 10.5F && owned[1].samples[5] == 50,
          "batch PCM owns data after caller history changes and model/proxy release");
}

void test_batches(const std::string& path, const trtmc::LoadOptions& options,
                  trtmc::AudioView audio) {
    const auto model = trtmc::Model::load(path, options);
    const auto asr = model.task<trtmc::BatchSpeechTranscription>();
    auto output = asr.run(
        {{{{audio, std::string{"en"}}, {{"suffix", "?"}}}, {{audio, std::string{"fr"}}, {}}}});
    check(output.size() == 2 && output[0].text == "batch_asr:en?" &&
              output[1].text == "batch_asr:fr!",
          "native ASR batch keeps per-item inputs configs and order");
    check(output[0].token_ids[0] == 1 && output[1].token_ids[2] == 0,
          "native ASR batch executes once without single-transcribe fallback");
    check(output[1].segments.size() == 1 && output[1].segments[0].text == "batch_asr:fr!",
          "ASR batch shares owned text/segment storage");
    check(rejects([&] { (void)asr.run({{{{audio}, {}}, {{audio}, {{"suffix", false}}}}}); },
                  TRTMC_INVALID_CONFIG),
          "invalid ASR item rejects entire batch before execution");
    check(asr.run({{{{audio}, {}}}})[0].token_ids[0] == 2, "invalid ASR batch did not execute");

    const auto translation = model.task<trtmc::BatchSpeechTranslation>();
    auto translated =
        translation.run({{{{audio, std::string{"de"}, std::string{"fr"}}, {{"suffix", "?"}}},
                          {{audio, std::string{"en"}, std::string{"de"}}, {}}}});
    check(translated[0].text == "batch_translate:fr->de?" &&
              translated[1].text == "batch_translate:de->en!" && translated[1].token_ids[2] == 0,
          "native translation batch preserves independent language pairs");
    check(asr.run({{}}).empty() && translation.run({{}}).empty(),
          "empty batches retain distinct typed contracts");
}
void test_mixed(const std::string& path, const trtmc::LoadOptions& options,
                trtmc::AudioView audio) {
    const auto model = trtmc::Model::load(path, options);
    const auto mixed = model.task<trtmc::MixedBatchSpeechToText>();
    const trtmc::MixedBatchSpeechToTextRequest request{{
        {trtmc::SpeechTranscriptionRequest{audio, std::string{"en"}}, {{"suffix", "?"}}},
        {trtmc::SpeechTranslationRequest{audio, std::string{"de"}, std::string{"fr"}}, {}},
    }};
    auto output = mixed.run(request);
    check(output.size() == 2 && output[0].text == "mixed_asr:en?" &&
              output[1].text == "mixed_translate:fr->de!",
          "mixed native batch preserves per-item operation language and config");
    const auto counts = output[1].token_ids;
    check(counts[0] == 1 && counts[1] == 1 && counts[2] == 0 && counts[3] == 0 && counts[4] == 0 &&
              counts[5] == 0,
          "mixed native batch calls neither single nor homogeneous batch interfaces");
    auto invalid = request;
    invalid.items[1].config = trtmc::Config{{"suffix", false}};
    check(rejects([&] { (void)mixed.run(invalid); }, TRTMC_INVALID_CONFIG),
          "mixed batch prevalidates every variant before execution");
    check(mixed.run(request)[0].token_ids[0] == 2,
          "rejected mixed batch leaves execution state untouched");
    check(output[1].segments[0].text == "mixed_translate:fr->de!",
          "mixed batch preserves owned segment views");
    check(mixed.run({{}}).empty(), "empty mixed batch is representable");
}

void test_generated_batches(const std::string& path, const trtmc::LoadOptions& options) {
    const auto model = trtmc::Model::load(path, options);
    const auto audio = model.task<trtmc::BatchTextToAudio>();
    auto result = audio.run(
        {{{{"wind"}, {{"preset", "calm"}}}, {{"rain now"}, {{"gain", 0.5}, {"preset", "quiet"}}}}});
    check(result.size() == 2 && result[0].samples.size() == 4 && result[1].samples.size() == 10 &&
              result[0].channels == 1 && result[1].channels == 2 &&
              result[0].sample_rate == 24000 && result[1].sample_rate == 16000 &&
              result[1].frame_count() == 5,
          "native audio batch retains unequal actual PCM lengths, sample rates and channels");
    check(result[0].samples[0] == 0.04F && result[0].samples[1] == 0.04F &&
              result[1].samples[0] == 0.04F && result[1].samples[1] == 0.025F &&
              result[0].setup_ms == 1 && result[1].inference_ms == 0,
          "one family batch call receives item text/config without calling single generation");
    const auto speech = model.task<trtmc::BatchTextToSpeech>();
    auto spoken =
        speech.run({{{{"Hello", std::string{"en"}}, {{"speaker", 1}, {"normalize", false}}},
                     {{"Bonjour", std::string{"fr"}}, {}}}});
    check(spoken.size() == 2 && spoken[0].samples.size() == 4 && spoken[1].samples.size() == 10 &&
              spoken[0].samples[1] == 0.1F && spoken[0].samples[2] == 0.1F &&
              spoken[0].samples[3] == 0 && spoken[1].samples[1] == 0 &&
              spoken[1].samples[2] == 0.2F && spoken[1].samples[3] == 0.25F &&
              spoken[0].setup_ms == 1 && spoken[1].inference_ms == 0,
          "native speech batch preserves per-item language/speaker/normalization and family "
          "defaults");
    check(audio.config_fields().size() == 2 && speech.config_fields().size() == 3,
          "each synthesis batch has its own family-declared config metadata");
    const trtmc::BatchTextToAudioRequest good{{{{"a"}, {}}, {{"b"}, {}}}};
    const auto before = audio.run(good)[0].setup_ms;
    for (const auto& config : std::vector<trtmc::Config>{{{"missing", 1}},
                                                         {{"gain", "bad"}},
                                                         {{"gain", -1.0}},
                                                         {{"preset", "one"}, {"preset", "two"}}}) {
        check(rejects([&] { (void)audio.run({{{{"a"}, {}}, {{"b"}, config}}}); },
                      TRTMC_INVALID_CONFIG),
              "invalid later item rejects the whole synthesis batch during family preflight");
    }
    check(audio.run(good)[0].setup_ms == before + 1,
          "bad item configs do not start partial native execution");
    check(rejects([&] { (void)audio.run({{}}); }, TRTMC_INVALID_ARGUMENT) &&
              rejects([&] { (void)speech.run({{}}); }, TRTMC_INVALID_ARGUMENT),
          "new synthesis batch contracts reject empty submissions without changing ASR batches");
    check(
        rejects([&] { (void)speech.run({{{{"x", std::string{}}, {}}}}); }, TRTMC_INVALID_ARGUMENT),
        "explicit empty per-item speech language is not omission");
    auto zero = audio.run({{{{""}, {{"gain", 0.0}, {"preset", ""}}}}});
    check(zero[0].samples.size() == 4 && zero[0].samples[0] == 0 && zero[0].samples[3] == 0,
          "zero/empty config values remain explicit and trailing PCM zeros are not trimmed");
    auto moved = std::move(result);
    check(result.empty() && moved.size() == 2 && moved[1].sample_rate == 16000,
          "batch result move clears source count and preserves owned PCM");
    check(rejects([&] { (void)result.at(0); }, TRTMC_INVALID_ARGUMENT) &&
              rejects([&] { (void)moved.at(2); }, TRTMC_INVALID_ARGUMENT),
          "moved-from and out-of-range audio item views fail without stale owner access");
    auto assigned = audio.run(good);
    assigned = std::move(moved);
    check(moved.empty() && assigned[1].samples.size() == 10,
          "audio batch move assignment releases its former owner");
}

void test_generated_modes(const std::filesystem::path& root, const trtmc::LoadOptions& options) {
    for (const auto* mode :
         {"audio_single_only", "batch_audio_only", "batch_speech_only", "uniform_preset",
          "bad_generated_count", "bad_generated_metadata", "bad_generated_rate",
          "bad_generated_frame", "batch_execution_error", "empty_generated_item"})
        write_bundle(root / (std::string{"audio-generated-"} + mode + ".bundle"), mode);
    auto load = [&](const char* mode) {
        return trtmc::Model::load(
            (root / (std::string{"audio-generated-"} + mode + ".bundle")).string(), options);
    };
    const auto single = load("audio_single_only");
    check(single.supports<trtmc::TextToAudio>() && single.supports<trtmc::TextToSpeech>() &&
              !single.supports<trtmc::BatchTextToAudio>() &&
              !single.supports<trtmc::BatchTextToSpeech>(),
          "single synthesis does not advertise either native batch");
    const auto only_audio = load("batch_audio_only");
    check(only_audio.supports<trtmc::BatchTextToAudio>() &&
              !only_audio.supports<trtmc::TextToAudio>() &&
              !only_audio.supports<trtmc::BatchTextToSpeech>(),
          "batch-audio support is independent of single and speech contracts");
    const auto only_speech = load("batch_speech_only");
    check(only_speech.supports<trtmc::BatchTextToSpeech>() &&
              !only_speech.supports<trtmc::TextToSpeech>() &&
              !only_speech.supports<trtmc::BatchTextToAudio>(),
          "batch-speech has an independent loaded-model declaration");
    const auto uniform = load("uniform_preset").task<trtmc::BatchTextToAudio>();
    auto defaults = uniform.run({{{{"one"}, {}}, {{"two"}, {{"preset", ""}}}}});
    check(defaults.size() == 2 && defaults[0].setup_ms == 1,
          "omitted preset and explicit equivalent default remain compatible");
    check(rejects([&] { (void)uniform.run({{{{"one"}, {{"preset", "calm"}}}, {{"two"}, {}}}}); },
                  TRTMC_INVALID_CONFIG),
          "family rejects incompatible native-batch voices after default resolution");
    auto same = uniform.run({{{{"one"}, {{"preset", "calm"}}}, {{"two"}, {{"preset", "calm"}}}}});
    check(same[0].setup_ms == 2 && defaults[1].samples[1] == 0,
          "uniformity preflight precedes execution and leaves earlier result storage intact");
    const trtmc::BatchTextToAudioRequest request{{{{"one"}, {}}, {{"two"}, {}}}};
    for (const auto* mode : {"bad_generated_count", "bad_generated_metadata", "bad_generated_rate",
                             "bad_generated_frame", "batch_execution_error"}) {
        const auto model = load(mode);
        check(
            rejects([&] { (void)model.task<trtmc::BatchTextToAudio>().run(request); },
                    TRTMC_INTERNAL_ERROR),
            "wrong result count, malformed PCM or execution failure cannot return partial success");
        check(rejects(
                  [&] {
                      (void)model.task<trtmc::BatchTextToSpeech>().run(
                          {{{{"one"}, {}}, {{"two"}, {}}}});
                  },
                  TRTMC_INTERNAL_ERROR),
              "speech synthesis batch uses the same all-or-error result contract");
    }
    auto empty_item = load("empty_generated_item").task<trtmc::BatchTextToAudio>().run(request);
    check(empty_item.size() == 2 && empty_item[1].samples.empty() && empty_item[1].channels == 2 &&
              empty_item[1].sample_rate == 16000,
          "a family-declared valid empty waveform is not confused with a failed batch");
    auto retained = [&] {
        auto model = load("batch_speech_only");
        trtmc::BatchTextToSpeechRequest input{
            {{{"owned", std::string{"fr"}}, {{"normalize", false}}}}};
        return model.task<trtmc::BatchTextToSpeech>().run(input);
    }();
    check(retained[0].samples[0] == 0.05F && retained[0].samples[2] == 0.2F &&
              retained[0].samples[3] == 0,
          "PCM survives temporary model/proxy/input/config destruction");
    for (const auto* mode :
         {"audio_single_only", "batch_audio_only", "batch_speech_only", "uniform_preset",
          "bad_generated_count", "bad_generated_metadata", "bad_generated_rate",
          "bad_generated_frame", "batch_execution_error", "empty_generated_item"})
        std::filesystem::remove(root / (std::string{"audio-generated-"} + mode + ".bundle"));
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    const std::filesystem::path root(argv[1]);
    const auto full = root / "audio-full.bundle";
    const auto restricted = root / "audio-asr.bundle";
    const auto explicit_language = root / "audio-explicit.bundle";
    const auto bad_audio = root / "audio-bad-pcm.bundle";
    const auto bad_labels = root / "audio-bad-labels.bundle";
    const auto explicit_rate = root / "audio-explicit-rate.bundle";
    write_bundle(full, "audio_all");
    write_bundle(restricted, "asr_only");
    write_bundle(explicit_language, "explicit_languages");
    write_bundle(bad_audio, "bad_audio");
    write_bundle(bad_labels, "bad_labels");
    write_bundle(explicit_rate, "explicit_sample_rate");
    trtmc::LoadOptions options;
    options.runtime_root = root.string();
    const float pcm[] = {0.1F, 0.2F, 0.3F, 0.4F};
    const trtmc::AudioView audio{{pcm, 4}, 16000, 2};
    const auto model = trtmc::Model::load(full.string(), options);
    test_stateless(model, audio);
    test_history(full.string(), options);
    test_history_batch(full.string(), options, root);
    test_batches(full.string(), options, audio);
    test_mixed(full.string(), options, audio);
    test_generated_batches(full.string(), options);
    test_generated_modes(root, options);
    const auto limited = trtmc::Model::load(restricted.string(), options);
    check(limited.supports<trtmc::SpeechTranscription>() &&
              !limited.supports<trtmc::SpeechTranslation>() &&
              !limited.supports<trtmc::BatchSpeechTranscription>(),
          "loaded mode limits supported audio contracts");
    check(!limited.supports<trtmc::MixedBatchSpeechToText>(),
          "single ASR does not imply mixed native batch support");
    const auto explicit_model = trtmc::Model::load(explicit_language.string(), options);
    check(rejects([&] { (void)explicit_model.task<trtmc::TextToSpeech>().run({"Hello"}); },
                  TRTMC_INVALID_ARGUMENT),
          "family with no synthesis language default rejects omission");
    check(rejects([&] { (void)explicit_model.task<trtmc::SpeechTranslation>().run({audio}); },
                  TRTMC_INVALID_ARGUMENT),
          "family with no translation target default rejects omission");
    const auto rate_model = trtmc::Model::load(explicit_rate.string(), options);
    auto unspecified = audio;
    unspecified.sample_rate.reset();
    check(rejects([&] { (void)rate_model.task<trtmc::SpeechTranscription>().run({unspecified}); },
                  TRTMC_INVALID_ARGUMENT),
          "family without input-rate default rejects omission");
    check(rate_model.task<trtmc::SpeechTranscription>().run({audio}).token_ids()[1] == 16000,
          "same family accepts a known explicit rate");
    check(rejects(
              [&] {
                  (void)trtmc::Model::load(bad_audio.string(), options)
                      .task<trtmc::TextToAudio>()
                      .run({"wind"});
              },
              TRTMC_INTERNAL_ERROR),
          "invalid family output channel metadata rejected");
    check(rejects(
              [&] {
                  (void)trtmc::Model::load(bad_labels.string(), options)
                      .task<trtmc::AudioLanguageIdentification>()
                      .run({audio});
              },
              TRTMC_INTERNAL_ERROR),
          "language identification cannot return unlabeled ordinal scores");
    auto retained = [&] {
        const auto temporary = trtmc::Model::load(full.string(), options);
        return temporary.task<trtmc::TextToAudio>().run({"wind"});
    }();
    check(retained.samples()[0] == 0.04F, "audio result survives temporary model/proxy owners");
    for (const auto& file :
         {full, restricted, explicit_language, bad_audio, bad_labels, explicit_rate})
        std::filesystem::remove(file);
    std::cerr << (failures == 0 ? "ALL PASSED\n" : "SOME FAILED\n");
    return failures;
}
