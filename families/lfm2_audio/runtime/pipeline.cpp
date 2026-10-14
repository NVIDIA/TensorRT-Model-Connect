/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lfm2_audio/runtime/pipeline.h"

#include "families/lfm2_audio/runtime/audio_dsp.h"
#include "families/lfm2_audio/runtime/conv_state.h"
#include "families/lfm2_audio/runtime/hybrid_state.h"
#include "families/lfm2_audio/runtime/plugin_helpers.h"
#include "families/lfm2_audio/runtime/sampling.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <variant>

namespace trtmc::lfm2_audio {
namespace {
using namespace internal;
const ConfigField kFields[] = {
    {"max_new_tokens", ConfigKind::I64, ConfigValue{int64_t{512}},
     "Maximum generated text tokens and audio frames combined"},
    {"text_temperature", ConfigKind::F64, ConfigValue{0.0},
     "Text temperature; nonpositive values select greedy decoding"},
    {"text_top_k", ConfigKind::I64, ConfigValue{int64_t{1}},
     "Text top-k; zero leaves the vocabulary unfiltered"},
    {"audio_temperature", ConfigKind::F64, ConfigValue{0.0},
     "Audio-codebook temperature; nonpositive values select greedy decoding"},
    {"audio_top_k", ConfigKind::I64, ConfigValue{int64_t{1}},
     "Audio top-k; zero leaves the vocabulary unfiltered"},
    {"seed", ConfigKind::I64, ConfigValue{int64_t{0}}, "Native sampling seed"},
    {"voice", ConfigKind::String, ConfigValue{std::string_view{"us_male"}},
     "TTS voice: us_male, us_female, uk_male, uk_female"},
};
struct Options {
    int max_tokens{512}, text_top_k{1}, audio_top_k{1};
    float text_temperature{0}, audio_temperature{0};
    std::uint32_t seed{0};
    std::string voice{"us_male"};
};
Options options(ConfigView values, bool tts = false) {
    validate_config(tts ? Span<const ConfigField>{kFields, 7} : Span<const ConfigField>{kFields, 6},
                    values);
    Options out;
    const auto max = config_get<int64_t>(values, kFields, "max_new_tokens").value();
    const auto text_k = config_get<int64_t>(values, kFields, "text_top_k").value();
    const auto audio_k = config_get<int64_t>(values, kFields, "audio_top_k").value();
    const auto seed = config_get<int64_t>(values, kFields, "seed").value();
    const double text_t = config_get<double>(values, kFields, "text_temperature").value();
    const double audio_t = config_get<double>(values, kFields, "audio_temperature").value();
    if (max <= 0 || max > 32768 || text_k < 0 || text_k > 65536 || audio_k < 0 || audio_k > 2049 ||
        seed < 0 || seed > std::numeric_limits<std::uint32_t>::max() || !std::isfinite(text_t) ||
        !std::isfinite(audio_t) || text_t > 1e6 || audio_t > 1e6)
        throw ConfigError("invalid LFM audio generation options");
    out.max_tokens = static_cast<int>(max);
    out.text_top_k = static_cast<int>(text_k);
    out.audio_top_k = static_cast<int>(audio_k);
    out.seed = static_cast<std::uint32_t>(seed);
    out.text_temperature = static_cast<float>(text_t);
    out.audio_temperature = static_cast<float>(audio_t);
    out.voice = std::string(config_get<std::string_view>(values, kFields, "voice").value());
    return out;
}
std::string voice_prompt(const std::string& voice) {
    if (voice == "us_male")
        return "Perform TTS. Use the US male voice.";
    if (voice == "us_female")
        return "Perform TTS. Use the US female voice.";
    if (voice == "uk_male")
        return "Perform TTS. Use the UK male voice.";
    if (voice == "uk_female")
        return "Perform TTS. Use the UK female voice.";
    throw ConfigError("LFM audio voice must be us_male, us_female, uk_male, or uk_female");
}
std::vector<float> floats(const BundleReader& bundle, const char* name, std::size_t count) {
    const auto bytes = require_section(bundle, name);
    if (bytes.size() != count * sizeof(float))
        throw std::runtime_error(std::string("invalid LFM audio section: ") + name);
    std::vector<float> result(count);
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return result;
}
std::vector<float> output(const TensorMap& tensors, const char* name, std::size_t count) {
    const auto& tensor = tensors.at(name);
    if (tensor.dtype != DType::kFloat32 || tensor.data == nullptr || tensor.numel() != count)
        throw std::runtime_error(std::string("invalid LFM audio engine output: ") + name);
    const auto* data = static_cast<const float*>(tensor.data);
    return {data, data + count};
}
void language(const std::optional<std::string_view>& value) {
    if (value && *value != "en" && *value != "English")
        throw std::invalid_argument("LFM2.5 audio supports English");
}
struct AudioSegment {
    std::size_t embedding_offset;
    MelFeatures mel;
};
struct History {
    std::vector<float> embeddings;
    std::vector<int32_t> text_tokens;
    std::vector<AudioSegment> audio_segments;
    std::size_t audio_code_frames{0};
};
struct Generation {
    std::vector<std::int32_t> text;
    std::vector<std::array<std::int32_t, 8>> codes;
    std::vector<std::int32_t> modalities;
    std::vector<float> waveform;
    bool stopped{false};
};
using Emit = std::function<bool(const Generation&, bool)>;
} // namespace

struct Engines {
    std::unique_ptr<ITrtModule> backbone, encoder, depth, detokenizer;
    std::unique_ptr<Lfm2AudioInferenceState> backbone_state, depth_state, detokenizer_state;
    std::shared_ptr<ITokenizer> tokenizer;
    std::vector<float> text_embeddings, audio_embeddings, depth_embeddings, detokenizer_embeddings,
        filters;
    std::mutex mutex;
    int max_context{0}, interleaved_text{6}, interleaved_audio{12};
    std::string task_name;
    GenerationTrace trace;

    std::unique_ptr<Lfm2AudioInferenceState> state(ITrtModule& module, const nlohmann::json& cfg,
                                                   int capacity, DType dtype) {
        auto kv = std::make_unique<Lfm2AudioKvCache>(
            cfg.at("num_attention_layers").get<int>(), capacity,
            cfg.at("num_key_value_heads").get<int>(), cfg.at("head_dim").get<int>(),
            module.stream(), dtype);
        auto conv = std::make_unique<Lfm2AudioConvState>(
            cfg.at("num_conv_layers").get<int>(), cfg.at("hidden_size").get<int>(),
            cfg.at("conv_L_cache").get<int>(), module.stream(), dtype);
        auto result = std::make_unique<Lfm2AudioHybridState>(std::move(kv), std::move(conv));
        result->bind_to(module);
        if (!result->ok())
            throw std::runtime_error("could not allocate LFM audio state");
        return result;
    }
    explicit Engines(const FamilyContext& context) {
        const auto cfg =
            nlohmann::json::parse(require_text_section(context.reader, "runtime.json"));
        if (cfg.at("version") != 1 || cfg.at("precision") != "bf16")
            throw std::runtime_error("invalid LFM audio runtime contract");
        max_context = cfg.at("max_context").get<int>();
        if (max_context <= 0 || max_context > 32768)
            throw std::runtime_error("invalid LFM audio context capacity");
        interleaved_text = cfg.at("interleaved_n_text").get<int>();
        interleaved_audio = cfg.at("interleaved_n_audio").get<int>();
        if (interleaved_text != 6 || interleaved_audio != 12)
            throw std::runtime_error("invalid LFM audio modality schedule");
        backbone = load_module(context.backend, require_section(context.reader, "backbone.plan"));
        encoder = load_module(context.backend, require_section(context.reader, "encoder.plan"));
        depth = load_module(context.backend, require_section(context.reader, "depthformer.plan"));
        detokenizer =
            load_module(context.backend, require_section(context.reader, "detokenizer.plan"));
        backbone->set_timing_label("lfm2_audio.backbone");
        encoder->set_timing_label("lfm2_audio.encoder");
        depth->set_timing_label("lfm2_audio.depthformer");
        detokenizer->set_timing_label("lfm2_audio.detokenizer");
        backbone_state = state(*backbone, cfg.at("lfm"), max_context, DType::kBFloat16);
        detokenizer_state =
            state(*detokenizer, cfg.at("detokenizer"), max_context * 6, DType::kFloat32);
        depth_state =
            std::make_unique<Lfm2AudioKvCache>(6, 8, 8, 32, depth->stream(), DType::kBFloat16);
        depth_state->bind_to(*depth);
        tokenizer = create_tokenizer(context.reader);
        text_embeddings = floats(context.reader, "text_embeddings.f32", 65536 * 2048);
        audio_embeddings = floats(context.reader, "audio_embeddings.f32", 2049 * 8 * 2048);
        depth_embeddings = floats(context.reader, "depth_embeddings.f32", 2049 * 8 * 1024);
        detokenizer_embeddings =
            floats(context.reader, "detokenizer_embeddings.f32", 2048 * 8 * 512);
        filters = floats(context.reader, "mel_filters.f32", 128 * 257);
        task_name = context.reader.info().task;
    }
    void append_token(History& history, int id) const {
        if (id < 0 || id >= 65536)
            throw std::invalid_argument("invalid LFM text token");
        const auto begin = text_embeddings.begin() + static_cast<std::ptrdiff_t>(id) * 2048;
        history.embeddings.insert(history.embeddings.end(), begin, begin + 2048);
        history.text_tokens.push_back(id);
    }
    void append_text(History& history, const std::string& text) const {
        for (int id : tokenizer->encode(text))
            append_token(history, id);
    }
    History start(const std::string& system) const {
        History history;
        append_text(history, "<|startoftext|><|im_start|>system\n" + system + "<|im_end|>\n");
        return history;
    }
    History text_history(const TextConversationRequest& request) const {
        if (request.messages.empty())
            throw std::invalid_argument("LFM audio text conversation must not be empty");
        if (!request.tools.empty())
            throw UnsupportedTask("LFM audio text conversation does not support tools");
        History history;
        append_text(history, "<|startoftext|>");
        for (const auto& message : request.messages) {
            std::string role;
            switch (message.role) {
            case MessageRole::System:
                role = "system";
                break;
            case MessageRole::User:
                role = "user";
                break;
            case MessageRole::Assistant:
                role = "assistant";
                break;
            default:
                throw std::invalid_argument("LFM audio supports system, user, and assistant turns");
            }
            append_text(history, "<|im_start|>" + role + "\n");
            for (const auto& part : message.parts) {
                const auto* text = std::get_if<TextPartView>(&part);
                if (!text)
                    throw UnsupportedTask("LFM audio text conversation accepts text parts only");
                append_text(history, std::string(text->text));
            }
            append_text(history, "<|im_end|>\n");
        }
        append_text(history, "<|im_start|>assistant\n");
        return history;
    }
    void append_codes(History& history, const std::array<int32_t, 8>& codes) const {
        const auto offset = history.embeddings.size();
        history.embeddings.resize(offset + 2048);
        for (int d = 0; d < 2048; ++d) {
            float sum = 0;
            for (int c = 0; c < 8; ++c)
                sum += audio_embeddings[(c * 2049 + codes[c]) * 2048 + d];
            history.embeddings[offset + d] = bf16(sum);
        }
        ++history.audio_code_frames;
    }
    void append_audio(History& history, const internal::AudioView& audio) {
        if (audio.channels != 1)
            throw std::invalid_argument("LFM audio requires mono input");
        auto mel = extract_mel(audio.samples.data(), audio.samples.size(),
                               static_cast<int>(audio.sample_rate.value_or(16000)), filters);
        const auto tokens = (mel.frames + 7) / 8;
        if (history.embeddings.size() / 2048 + tokens > static_cast<std::size_t>(max_context))
            throw std::invalid_argument("LFM audio input exceeds context capacity");
        const auto offset = history.embeddings.size();
        history.embeddings.resize(offset + static_cast<std::size_t>(tokens) * 2048);
        history.audio_segments.push_back({offset, std::move(mel)});
    }
    std::size_t encode_audio_history(History& history) {
        if (history.audio_segments.empty())
            return 0;
        if (!encoder->has_input("valid_frames"))
            throw std::runtime_error("LFM audio encoder is missing its input-length contract");
        int max_frames = 0;
        for (const auto& part : history.audio_segments)
            max_frames = std::max(max_frames, part.mel.frames);
        const auto padded_tokens = (max_frames + 7) / 8;
        // Original _prefill pads all historical input segments before the
        // shared Conformer call. Recompute earlier inputs when that maximum
        // changes, while preserving generated audio-code embeddings verbatim.
        std::size_t encoded = 0;
        for (const auto& part : history.audio_segments) {
            std::vector<float> padded(static_cast<std::size_t>(max_frames) * 128);
            std::copy(part.mel.values.begin(), part.mel.values.end(), padded.begin());
            std::int32_t valid_tokens = (part.mel.frames + 7) / 8;
            TensorMap inputs{
                {"mel", Tensor{padded.data(), {1, 1, max_frames, 128}, DType::kFloat32}},
                {"valid_frames", Tensor{&valid_tokens, {1}, DType::kInt32}}};
            auto outputs = encoder->forward(inputs);
            ++encoded;
            auto embeddings =
                output(outputs, "audio_embeddings", static_cast<std::size_t>(padded_tokens) * 2048);
            std::copy_n(embeddings.begin(), static_cast<std::size_t>(valid_tokens) * 2048,
                        history.embeddings.begin() +
                            static_cast<std::ptrdiff_t>(part.embedding_offset));
        }
        return encoded;
    }
    struct StepOutput {
        std::vector<float> logits, hidden;
    };
    StepOutput step(ITrtModule& module, Lfm2AudioInferenceState& state, const float* embedding,
                    int width, const float* previous = nullptr, bool decoding = false) {
        TensorMap inputs{{"input_embedding",
                          Tensor{const_cast<float*>(embedding), {1, width}, DType::kFloat32}}};
        if (previous)
            inputs.emplace("previous_embedding",
                           Tensor{const_cast<float*>(previous), {1, 1024}, DType::kFloat32});
        std::int32_t phase = decoding ? 1 : 0;
        if (module.has_input("decode_phase"))
            inputs.emplace("decode_phase", Tensor{&phase, {1, 1}, DType::kInt32});
        state.prepare_step(inputs);
        module.forward_async(inputs);
        state.advance();
        StepOutput result;
        const auto read = [&](const char* name, std::vector<float>& values) {
            if (!module.has_output(name))
                return;
            const auto shape = module.tensor_shape(name);
            std::size_t count = 1;
            for (auto dim : shape) {
                if (dim <= 0)
                    throw std::runtime_error("LFM audio requires fixed decoder output geometry");
                count *= static_cast<std::size_t>(dim);
            }
            if (module.tensor_dtype(name) != DType::kFloat32)
                throw std::runtime_error("LFM audio requires FP32 exposed decoder outputs");
            values.resize(count);
            if (cudaMemcpyAsync(values.data(), module.device_ptr(name), count * sizeof(float),
                                cudaMemcpyDeviceToHost, module.stream()) != cudaSuccess)
                throw std::runtime_error("LFM audio decoder output transfer failed");
        };
        read("logits", result.logits);
        read("hidden_state", result.hidden);
        module.sync();
        return result;
    }
    std::array<int32_t, 8> audio_frame(const std::vector<float>& hidden, const Options& cfg,
                                       std::mt19937& rng) {
        depth_state->reset();
        std::array<int32_t, 8> codes;
        std::vector<float> previous(1024);
        for (int i = 0; i < 8; ++i) {
            const auto outs = step(*depth, *depth_state, hidden.data(), 2048, previous.data());
            codes[i] = sample(outs.logits, cfg.audio_temperature, cfg.audio_top_k, rng);
            const auto begin = depth_embeddings.begin() + (i * 2049 + codes[i]) * 1024;
            previous.assign(begin, begin + 1024);
        }
        if (codes[0] == 2048)
            codes.fill(2048);
        return codes;
    }
    void synthesize(const std::array<int32_t, 8>& codes, std::vector<float>& spectra) {
        std::array<float, 512> embedding{};
        for (int c = 0; c < 8; ++c) {
            if (codes[c] < 0 || codes[c] >= 2048)
                throw std::runtime_error("invalid LFM audio detokenizer code");
            for (int d = 0; d < 512; ++d)
                embedding[d] += detokenizer_embeddings[(c * 2048 + codes[c]) * 512 + d];
        }
        for (float& value : embedding)
            value /= 8;
        for (int frame = 0; frame < 6; ++frame) {
            auto outputs = step(*detokenizer, *detokenizer_state, embedding.data(), 512);
            auto spectrum = std::move(outputs.logits);
            spectra.insert(spectra.end(), spectrum.begin(), spectrum.end());
        }
    }
    Generation generate(History& history, const Options& cfg, bool interleaved, Emit emit = {}) {
        if (history.embeddings.empty() || history.embeddings.size() % 2048)
            throw std::invalid_argument("invalid LFM audio history");
        if (history.embeddings.size() / 2048 + cfg.max_tokens >
            static_cast<std::size_t>(max_context))
            throw std::invalid_argument(
                "LFM audio prompt and requested generation exceed context capacity");
        const auto prompt_tokens = history.embeddings.size() / 2048;
        const auto prompt_text_tokens = history.text_tokens;
        const auto prior_audio_code_frames = history.audio_code_frames;
        const auto input_audio_encodings = encode_audio_history(history);
        backbone_state->reset();
        detokenizer_state->reset();
        StepOutput outputs;
        for (std::size_t token = 0; token < history.embeddings.size() / 2048; ++token)
            outputs =
                step(*backbone, *backbone_state, history.embeddings.data() + token * 2048, 2048);
        std::mt19937 rng(cfg.seed);
        Generation generated;
        StreamingIstft synthesis;
        bool audio = false, text_done = false;
        int left = interleaved_text;
        for (int i = 0; i < cfg.max_tokens; ++i) {
            if (interleaved)
                --left;
            if (!audio) {
                const auto id = sample(outputs.logits, cfg.text_temperature, cfg.text_top_k, rng);
                if (interleaved && id == 7)
                    break;
                generated.text.push_back(id);
                generated.modalities.push_back(1);
                append_token(history, id);
                if (!interleaved && id == 7)
                    break;
                if (interleaved) {
                    if (id == 130)
                        text_done = true;
                    if (left == 0 || text_done) {
                        audio = true;
                        left = interleaved_audio;
                    }
                } else if (id == 128)
                    audio = true;
            } else {
                auto codes = audio_frame(outputs.hidden, cfg, rng);
                if (interleaved && left == 0 && !text_done) {
                    audio = false;
                    left = interleaved_text;
                }
                if (codes[0] == 2048)
                    audio = false;
                generated.codes.push_back(codes);
                generated.modalities.push_back(3);
                append_codes(history, codes);
                if (codes[0] != 2048) {
                    std::vector<float> spectra;
                    synthesize(codes, spectra);
                    auto chunk = synthesis.append(spectra.data(), spectra.size() / 1282);
                    generated.waveform.insert(generated.waveform.end(), chunk.begin(), chunk.end());
                }
            }
            if (emit && !emit(generated, false)) {
                generated.stopped = true;
                break;
            }
            if (i + 1 < cfg.max_tokens)
                outputs = step(*backbone, *backbone_state,
                               history.embeddings.data() + history.embeddings.size() - 2048, 2048,
                               nullptr, true);
        }
        auto tail = synthesis.finish();
        generated.waveform.insert(generated.waveform.end(), tail.begin(), tail.end());
        if (emit && !generated.stopped && !emit(generated, true))
            generated.stopped = true;
        trace = {generated.text,        generated.codes,         generated.modalities,
                 input_audio_encodings, prior_audio_code_frames, prompt_tokens,
                 prompt_text_tokens};
        return generated;
    }
};

Pipeline::Pipeline(const FamilyContext& context) : engines_(std::make_shared<Engines>(context)) {}
Pipeline::~Pipeline() = default;
GenerationTrace Pipeline::last_generation_trace() const {
    std::lock_guard<std::mutex> guard(engines_->mutex);
    return engines_->trace;
}
std::vector<std::int32_t>
Pipeline::text_conversation_prompt_tokens(const TextConversationRequest& request) const {
    std::lock_guard<std::mutex> guard(engines_->mutex);
    return engines_->text_history(request).text_tokens;
}
const char* Pipeline::task() const noexcept {
    return engines_->task_name.c_str();
}
std::vector<internal::TaskInstance> Pipeline::task_bindings() {
    return {bind<ISpeechTranscription>(*this, {kFields, 6}),
            bind<ISpeechToSpeechResponse>(*this, {kFields, 6}),
            bind<ITextToSpeech>(*this, kFields),
            bind<IStreamingTextToSpeech>(*this, kFields),
            bind<IAudioTextToText>(*this, {kFields, 6}),
            bind<IOfflineSpeechDialogue>(*this, {kFields, 6}),
            bind<ITextConversation>(*this, {kFields, 6})};
}
TextResult Pipeline::run(const SpeechTranscriptionRequest& request, ConfigView values) {
    language(request.source_language);
    const auto cfg = options(values);
    std::lock_guard<std::mutex> guard(engines_->mutex);
    auto history = engines_->start("Perform ASR.");
    engines_->append_text(history, "<|im_start|>user\n");
    engines_->append_audio(history, request.audio);
    engines_->append_text(history, "<|im_end|>\n<|im_start|>assistant\n");
    auto result = engines_->generate(history, cfg, false);
    return {engines_->tokenizer->decode(result.text), std::move(result.text)};
}
internal::AudioResult Pipeline::run(const SpeechToSpeechResponseRequest& request,
                                    ConfigView values) {
    const auto cfg = options(values);
    std::lock_guard<std::mutex> guard(engines_->mutex);
    auto history = engines_->start("Respond with interleaved text and audio.");
    engines_->append_text(history, "<|im_start|>user\n");
    engines_->append_audio(history, request.audio);
    engines_->append_text(history, "<|im_end|>\n<|im_start|>assistant\n");
    auto generated = engines_->generate(history, cfg, true);
    return {std::move(generated.waveform), 24000, 1};
}
internal::AudioResult Pipeline::run(const TextToSpeechRequest& request, ConfigView values) {
    language(request.language);
    const auto cfg = options(values, true);
    const auto system = voice_prompt(cfg.voice);
    if (request.text.empty())
        throw std::invalid_argument("LFM TTS text must not be empty");
    std::lock_guard<std::mutex> guard(engines_->mutex);
    auto history = engines_->start(system);
    engines_->append_text(history, "<|im_start|>user\n" + std::string(request.text) +
                                       "<|im_end|>\n<|im_start|>assistant\n");
    auto generated = engines_->generate(history, cfg, false);
    return {std::move(generated.waveform), 24000, 1};
}
StreamingAudioSummary Pipeline::run(const TextToSpeechRequest& request, ConfigView values,
                                    const AudioChunkSink& sink) {
    language(request.language);
    const auto cfg = options(values, true);
    const auto system = voice_prompt(cfg.voice);
    if (request.text.empty() || !sink)
        throw std::invalid_argument("LFM streaming TTS requires text and a sink");
    std::lock_guard<std::mutex> guard(engines_->mutex);
    auto history = engines_->start(system);
    engines_->append_text(history, "<|im_start|>user\n" + std::string(request.text) +
                                       "<|im_end|>\n<|im_start|>assistant\n");
    std::size_t emitted = 0;
    auto generated = engines_->generate(history, cfg, false, [&](const Generation& result, bool) {
        const auto ready = result.waveform.size();
        if (ready <= emitted)
            return true;
        AudioView view{{result.waveform.data() + emitted, ready - emitted}, 24000, 1};
        const bool keep = sink(view);
        emitted = ready;
        return keep;
    });
    StreamingAudioSummary summary;
    summary.emitted_sample_count = summary.emitted_frame_count = emitted;
    summary.output = {24000, 1};
    summary.outcome =
        generated.stopped ? AudioDeliveryOutcome::Stopped : AudioDeliveryOutcome::Complete;
    return summary;
}
TextResult Pipeline::run(const AudioTextToTextRequest& request, ConfigView values) {
    const auto cfg = options(values);
    if (request.messages.empty())
        throw std::invalid_argument("LFM audio conversation must not be empty");
    bool has_audio = false, has_text = false;
    for (const auto& message : request.messages)
        for (const auto& part : message.parts) {
            has_audio |= std::holds_alternative<AudioView>(part);
            has_text |= std::holds_alternative<TextPartView>(part);
        }
    if (!has_audio || !has_text)
        throw std::invalid_argument("LFM audio/text conversation requires audio and text parts");
    std::lock_guard<std::mutex> guard(engines_->mutex);
    History history;
    engines_->append_text(history, "<|startoftext|>");
    for (const auto& message : request.messages) {
        std::string role;
        switch (message.role) {
        case MessageRole::System:
            role = "system";
            break;
        case MessageRole::User:
            role = "user";
            break;
        case MessageRole::Assistant:
            role = "assistant";
            break;
        default:
            throw std::invalid_argument("LFM audio supports system, user, and assistant turns");
        }
        engines_->append_text(history, "<|im_start|>" + role + "\n");
        for (const auto& part : message.parts) {
            if (const auto* text = std::get_if<TextPartView>(&part))
                engines_->append_text(history, std::string(text->text));
            else {
                if (message.role != MessageRole::User)
                    throw std::invalid_argument(
                        "LFM audio waveform input must be a user turn; use a dialogue session to "
                        "retain generated audio codes");
                engines_->append_audio(history, std::get<AudioView>(part));
            }
        }
        engines_->append_text(history, "<|im_end|>\n");
    }
    engines_->append_text(history, "<|im_start|>assistant\n");
    auto result = engines_->generate(history, cfg, false);
    return {engines_->tokenizer->decode(result.text), std::move(result.text)};
}

ConversationResult Pipeline::run(const TextConversationRequest& request, ConfigView values) {
    const auto cfg = options(values);
    std::lock_guard<std::mutex> guard(engines_->mutex);
    auto history = engines_->text_history(request);
    const auto input_tokens = history.embeddings.size() / 2048;
    auto generated = engines_->generate(history, cfg, false);
    ConversationResult result;
    result.parts.push_back(TextPart{engines_->tokenizer->decode(generated.text)});
    result.finish_reason = !generated.text.empty() && generated.text.back() == 7
                               ? FinishReason::Stop
                               : FinishReason::Length;
    result.usage.input_tokens = input_tokens;
    result.usage.output_tokens = generated.text.size() + generated.codes.size();
    result.usage.total_tokens = input_tokens + *result.usage.output_tokens;
    result.token_ids = std::move(generated.text);
    return result;
}

namespace {
class Session final : public ISpeechDialogueSession, public ISpeechRealtimeControl {
  public:
    Session(std::shared_ptr<Engines> engines, const SpeechDialogueRequest& request,
            ConfigView values)
        : engines_(std::move(engines)), cfg_(options(values)),
          sample_rate_(request.input.sample_rate.value_or(16000)),
          system_(request.system_prompt ? std::string(*request.system_prompt)
                                        : "Respond with interleaved text and audio.") {
        if (request.input.channels != 1 || sample_rate_ == 0 || sample_rate_ > 384000)
            throw std::invalid_argument("LFM dialogue requires mono PCM and a valid input rate");
        history_ = engines_->start(system_);
        effective_ = {
            {"max_new_tokens", int64_t{cfg_.max_tokens}},
            {"text_temperature", double{cfg_.text_temperature}},
            {"text_top_k", int64_t{cfg_.text_top_k}},
            {"audio_temperature", double{cfg_.audio_temperature}},
            {"audio_top_k", int64_t{cfg_.audio_top_k}},
            {"seed", int64_t{cfg_.seed}},
        };
    }
    ~Session() override { stop(); }
    void stop() noexcept override {
        stopped_.store(true);
        cancelled_.store(true);
        cv_.notify_all();
    }
    bool append_audio(Span<const float> samples) override {
        if (samples.empty() ||
            !std::all_of(samples.begin(), samples.end(), [](float v) { return std::isfinite(v); }))
            throw std::invalid_argument("LFM dialogue requires finite, nonempty PCM input");
        std::lock_guard<std::mutex> guard(state_mutex_);
        require_input();
        if (pending_samples_ + samples.size() >
            static_cast<std::size_t>(sample_rate_) * engines_->max_context * 8 / 100)
            return false;
        if (pending_.empty() || !std::holds_alternative<std::vector<float>>(pending_.back()))
            pending_.emplace_back(std::vector<float>{});
        auto& audio = std::get<std::vector<float>>(pending_.back());
        audio.insert(audio.end(), samples.begin(), samples.end());
        pending_samples_ += samples.size();
        return true;
    }
    bool append_text(std::string_view text) override {
        if (text.empty())
            throw std::invalid_argument("LFM dialogue text must not be empty");
        std::lock_guard<std::mutex> guard(state_mutex_);
        require_input();
        if (pending_text_bytes_ + text.size() >
            static_cast<std::size_t>(engines_->max_context) * 32)
            return false;
        if (!pending_.empty() && std::holds_alternative<std::string>(pending_.back()))
            std::get<std::string>(pending_.back()).append(text);
        else
            pending_.emplace_back(std::string(text));
        pending_text_bytes_ += text.size();
        return true;
    }
    void finish_input() override {
        std::lock_guard<std::mutex> request(request_mutex_);
        {
            std::lock_guard<std::mutex> guard(state_mutex_);
            if (finished_)
                return;
        }
        if (prepare_input_turn(true))
            respond();
        event(SpeechEventKind::InputFinished);
    }
    SpeechEventBatch read_events(std::int64_t timeout_ms) override {
        if (timeout_ms < -1)
            throw std::invalid_argument("invalid speech event timeout");
        std::unique_lock<std::mutex> guard(event_mutex_);
        auto ready = [&] { return !events_.empty() || stopped_ || epoch_ended_; };
        if (timeout_ms < 0)
            cv_.wait(guard, ready);
        else if (timeout_ms > 0)
            cv_.wait_for(guard, std::chrono::milliseconds(timeout_ms), ready);
        SpeechEventBatch result;
        result.events.swap(events_);
        result.state =
            stopped_ || epoch_ended_ ? SpeechReadState::EpochEnded : SpeechReadState::Active;
        return result;
    }
    void cancel() override {
        std::lock_guard<std::mutex> state(state_mutex_);
        epoch_cancelled_.store(true);
        cancelled_.store(true);
        event(SpeechEventKind::Cancelled);
    }
    void reset() override {
        cancelled_.store(true);
        std::lock_guard<std::mutex> request(request_mutex_);
        std::lock_guard<std::mutex> state(state_mutex_);
        if (stopped_)
            throw std::invalid_argument("LFM dialogue is stopped");
        history_ = engines_->start(system_);
        pending_.clear();
        pending_samples_ = 0;
        pending_text_bytes_ = 0;
        waiting_response_ = false;
        finished_ = false;
        cancelled_.store(false);
        epoch_cancelled_.store(false);
        {
            std::lock_guard<std::mutex> events(event_mutex_);
            ++epoch_;
            epoch_ended_ = false;
        }
        event(SpeechEventKind::Reset);
    }
    SpeechSessionInfo info() const override {
        SpeechSessionInfo info;
        info.input = {sample_rate_, 1};
        info.output = SpeechAudioFormat{24000, 1};
        info.source_language = "en";
        info.system_prompt = system_;
        return info;
    }
    ConfigView effective_config() const override { return {effective_.data(), effective_.size()}; }
    ISpeechRealtimeControl* realtime_control() noexcept override { return this; }
    void commit_input_turn(bool create = true) override {
        std::lock_guard<std::mutex> request(request_mutex_);
        prepare_input_turn(false);
        if (create)
            respond();
    }
    void create_response() override {
        std::lock_guard<std::mutex> request(request_mutex_);
        respond();
    }
    void clear_pending_input() override {
        std::lock_guard<std::mutex> state(state_mutex_);
        require_input();
        pending_.clear();
        pending_samples_ = 0;
        pending_text_bytes_ = 0;
        event(SpeechEventKind::InputCleared);
    }
    void cancel_response() override { cancelled_.store(true); }
    void truncate_response(std::uint64_t, std::int64_t) override {
        throw UnsupportedTask("LFM offline dialogue does not support playback truncation");
    }

  private:
    bool prepare_input_turn(bool finishing) {
        std::lock_guard<std::mutex> state(state_mutex_);
        if (stopped_ || epoch_cancelled_)
            throw std::invalid_argument("LFM dialogue input is closed");
        if (finishing && pending_.empty()) {
            finished_ = true;
            return waiting_response_;
        }
        require_input();
        if (pending_.empty())
            throw std::invalid_argument("LFM dialogue has no pending input");
        auto candidate = history_;
        engines_->append_text(candidate, "<|im_start|>user\n");
        std::lock_guard<std::mutex> execute(engines_->mutex);
        for (const auto& part : pending_) {
            if (const auto* text = std::get_if<std::string>(&part))
                engines_->append_text(candidate, *text);
            else {
                const auto& audio = std::get<std::vector<float>>(part);
                engines_->append_audio(candidate,
                                       AudioView{{audio.data(), audio.size()}, sample_rate_, 1});
            }
        }
        engines_->append_text(candidate, "<|im_end|>\n");
        const auto assistant_tokens = engines_->tokenizer->encode("<|im_start|>assistant\n").size();
        if (candidate.embeddings.size() / 2048 + cfg_.max_tokens + assistant_tokens >
            static_cast<std::size_t>(engines_->max_context))
            throw std::invalid_argument(
                "LFM dialogue input and generation exceed context capacity");
        std::swap(history_, candidate);
        pending_.clear();
        pending_samples_ = 0;
        pending_text_bytes_ = 0;
        waiting_response_ = true;
        finished_ = finishing;
        return true;
    }
    void require_input() const {
        if (stopped_ || finished_ || epoch_cancelled_)
            throw std::invalid_argument("LFM dialogue input is closed");
        if (waiting_response_)
            throw std::invalid_argument("LFM dialogue has a committed turn awaiting its response");
    }
    void event(SpeechEventKind kind, std::string text = {}, std::vector<float> audio = {},
               std::size_t start = 0, bool final = false) {
        std::lock_guard<std::mutex> guard(event_mutex_);
        SpeechDialogueEvent e;
        e.kind = kind;
        e.epoch = epoch_;
        e.sequence = ++sequence_;
        e.text = std::move(text);
        e.is_final = final;
        if (!audio.empty()) {
            e.media_start_sample = static_cast<std::int64_t>(start);
            e.media_end_sample = static_cast<std::int64_t>(start + audio.size());
            e.audio = {std::move(audio), 24000, 1};
        }
        events_.push_back(std::move(e));
        if (kind == SpeechEventKind::TurnStarted)
            epoch_ended_ = false;
        if (kind == SpeechEventKind::TurnFinished || kind == SpeechEventKind::Cancelled ||
            kind == SpeechEventKind::InputFinished)
            epoch_ended_ = true;
        cv_.notify_all();
    }
    static std::string complete_text(std::string_view bytes, bool final, std::size_t& consumed) {
        std::string text;
        std::size_t offset = 0;
        while (offset < bytes.size()) {
            const auto first = static_cast<unsigned char>(bytes[offset]);
            const std::size_t width = first < 0x80                     ? 1
                                      : first >= 0xc2 && first <= 0xdf ? 2
                                      : first >= 0xe0 && first <= 0xef ? 3
                                      : first >= 0xf0 && first <= 0xf4 ? 4
                                                                       : 0;
            if (width == 0) {
                text += "\xef\xbf\xbd";
                ++offset;
                continue;
            }
            std::size_t valid = 1;
            while (valid < width && offset + valid < bytes.size()) {
                const auto next = static_cast<unsigned char>(bytes[offset + valid]);
                if (next < 0x80 || next > 0xbf ||
                    (valid == 1 &&
                     ((first == 0xe0 && next < 0xa0) || (first == 0xed && next > 0x9f) ||
                      (first == 0xf0 && next < 0x90) || (first == 0xf4 && next > 0x8f))))
                    break;
                ++valid;
            }
            if (valid == width) {
                text.append(bytes.substr(offset, width));
                offset += width;
            } else if (offset + valid == bytes.size() && !final) {
                break;
            } else {
                // Match loss-tolerant ByteLevel decoding only once a sequence
                // is known to be invalid, or generation has ended mid-character.
                text += "\xef\xbf\xbd";
                offset += valid;
            }
        }
        consumed = offset;
        return text;
    }
    void finish_response() {
        {
            std::lock_guard<std::mutex> state(state_mutex_);
            waiting_response_ = false;
        }
        try {
            event(SpeechEventKind::TurnFinished, {}, {}, 0, true);
        } catch (...) {
            // Even an allocation failure while enqueueing the terminal event
            // must release a direct-session reader and preserve its exception.
            {
                std::lock_guard<std::mutex> events(event_mutex_);
                epoch_ended_ = true;
            }
            cv_.notify_all();
            throw;
        }
    }
    void respond() {
        History candidate;
        {
            std::lock_guard<std::mutex> state(state_mutex_);
            if (stopped_ || epoch_cancelled_ || !waiting_response_)
                throw std::invalid_argument("LFM dialogue has no committed user turn");
            candidate = history_;
            cancelled_.store(false);
            event(SpeechEventKind::TurnStarted);
        }
        try {
            engines_->append_text(candidate, "<|im_start|>assistant\n");
            std::size_t emitted_samples = 0, emitted_text_bytes = 0, decoded_tokens = 0;
            std::string decoded_text;
            std::lock_guard<std::mutex> execute(engines_->mutex);
            auto result = engines_->generate(
                candidate, cfg_, true, [&](const Generation& result, bool final) {
                    std::lock_guard<std::mutex> state(state_mutex_);
                    if (cancelled_ || epoch_cancelled_ || stopped_)
                        return false;
                    if (result.text.size() > decoded_tokens) {
                        decoded_text = engines_->tokenizer->decode(result.text);
                        if (decoded_text.size() < emitted_text_bytes)
                            throw std::runtime_error(
                                "LFM audio text decoding changed an emitted prefix");
                        decoded_tokens = result.text.size();
                    }
                    if (decoded_text.size() > emitted_text_bytes) {
                        std::size_t consumed = 0;
                        auto chunk =
                            complete_text(std::string_view(decoded_text).substr(emitted_text_bytes),
                                          final, consumed);
                        if (!chunk.empty())
                            event(SpeechEventKind::AgentText, std::move(chunk), {}, 0, final);
                        emitted_text_bytes += consumed;
                    }
                    const auto ready = result.waveform.size();
                    if (ready > emitted_samples) {
                        std::vector<float> chunk(result.waveform.begin() + emitted_samples,
                                                 result.waveform.begin() + ready);
                        event(SpeechEventKind::AgentAudio, {}, std::move(chunk), emitted_samples,
                              final);
                        emitted_samples = ready;
                    }
                    return true;
                });
            if (!result.stopped)
                engines_->append_text(candidate, "<|im_end|>\n");
            {
                std::lock_guard<std::mutex> state(state_mutex_);
                if (!result.stopped && !cancelled_ && !epoch_cancelled_ && !stopped_)
                    std::swap(history_, candidate);
            }
        } catch (...) {
            try {
                finish_response();
            } catch (...) {
                // finish_response already marked the epoch terminal. Retain
                // the original generation/callback failure for the caller.
            }
            throw;
        }
        finish_response();
    }
    std::shared_ptr<Engines> engines_;
    Options cfg_;
    std::uint32_t sample_rate_;
    std::string system_;
    std::vector<ConfigEntry> effective_;
    History history_;
    std::vector<std::variant<std::string, std::vector<float>>> pending_;
    std::size_t pending_samples_{0}, pending_text_bytes_{0};
    bool waiting_response_{false}, finished_{false}, epoch_ended_{false};
    std::atomic<bool> stopped_{false}, cancelled_{false}, epoch_cancelled_{false};
    std::mutex request_mutex_, state_mutex_, event_mutex_;
    std::condition_variable cv_;
    std::vector<SpeechDialogueEvent> events_;
    std::uint64_t epoch_{1}, sequence_{0};
};
} // namespace

std::unique_ptr<ISpeechDialogueSession>
Pipeline::create_offline_session(const SpeechDialogueRequest& request, ConfigView values) {
    return std::make_unique<Session>(engines_, request, values);
}
} // namespace trtmc::lfm2_audio
