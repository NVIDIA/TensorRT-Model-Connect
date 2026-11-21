/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "chat.h"
#include "media_geometry.h"
#include "native_decoder.h"
#include "native_media.h"
#include "request.h"
#include "resampler.h"
#include "sampler.h"
#include "tokenizer.h"
#include "trtmc/runtime/family_factory.h"
#include "visual_frontend.h"
#ifdef TRTMC_HAS_EDGE_LLM
#include "edge_llm/adapter.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <locale>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace trtmc::nemotron_omni {
namespace {
using namespace internal;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using Part = std::variant<TextPartView, ImageView, VideoView, AudioView, AlignedAudioVideoView>;
using InputMessage = MediaMessage<Part>;
std::string text_asset(const BundleReader& reader, const char* name) {
    auto bytes = reader.read_section(name);
    return {bytes.begin(), bytes.end()};
}
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
struct EncodedMedia {
    std::string text;
    std::vector<float> visual, audio;
};
std::string repeat(const char* token, size_t count) {
    std::string result;
    for (size_t i = 0; i < count; ++i)
        result += token;
    return result;
}
std::vector<float> rgb(ImageView image) {
    if (!image.data || !image.height || !image.width || image.channels != 3 ||
        image.height > uint32_t(std::numeric_limits<int>::max()) ||
        image.width > uint32_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Omni requires nonempty RGB image input");
    const size_t count = size_t(image.height) * image.width * 3;
    const bool integer = image.format == ImageFormat::UInt8;
    if ((!integer && image.format != ImageFormat::Float32) ||
        count > std::numeric_limits<size_t>::max() / sizeof(float) ||
        image.byte_size != count * (integer ? 1 : sizeof(float)))
        throw std::invalid_argument("Omni image byte size or format is invalid");
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i) {
        const float value = integer ? static_cast<const uint8_t*>(image.data)[i] / 255.f
                                    : static_cast<const float*>(image.data)[i];
        if (!std::isfinite(value) || value < 0 || value > 1)
            throw std::invalid_argument("Omni float RGB pixels must be finite within [0, 1]");
        result[i] = value;
    }
    return result;
}

class Pipeline final : public ITextGeneration,
                       public IVisionLanguageGeneration,
                       public IModel,
                       public ITextContinuation,
                       public IImagesTextToText,
                       public IVideoTextToText,
                       public IAudioTextToText,
                       public IImageAudioTextToText,
                       public IAudioVideoTextToText {
  public:
    explicit Pipeline(const FamilyContext& context)
        : task_(context.reader.info().task),
          metadata_(Json::parse(text_asset(context.reader, "runtime.json"))),
          config_(metadata_.at("config")), decoder_(context), media_(context),
          tokenizer_(text_asset(context.reader, "tokenizer.json")),
          embeddings_(context.reader.read_section("embeddings.f32")) {
        if (embeddings_.size() != size_t(decoder_.hidden()) * decoder_.vocab() * sizeof(float))
            throw std::invalid_argument("Omni embedding table differs from decoder dimensions");
        auto generation = Json::parse(text_asset(context.reader, "generation_config.json"));
        const auto stops = generation.at("eos_token_id");
        eos_ =
            stops.is_array() ? stops.get<std::vector<int>>() : std::vector<int>{stops.get<int>()};
        for (const auto id : eos_)
            if (id < 0 || id >= decoder_.vocab())
                throw std::invalid_argument("Omni EOS token is outside vocabulary");
    }
    const char* task() const noexcept override { return task_.c_str(); }
    int32_t default_max_new_tokens() const override { return 128; }
    std::vector<TaskInstance> task_bindings() override {
        return {bind<ITextContinuation>(*this, field_view(false)),
                bind<IImagesTextToText>(*this, field_view()),
                bind<IVideoTextToText>(*this, field_view()),
                bind<IAudioTextToText>(*this, field_view()),
                bind<IImageAudioTextToText>(*this, field_view()),
                bind<IAudioVideoTextToText>(*this, field_view())};
    }
    TextResult generate(const std::string& text, const TextGenerationConfig& options) override {
        std::vector<Part> parts{TextPartView{text}};
        InputMessage message{MessageRole::User, {parts.data(), parts.size()}};
        return execute({&message, 1}, options);
    }
    TextResult generate(const std::string& text, const float* pixels, int32_t h, int32_t w,
                        const TextGenerationConfig& options) override {
        if (h <= 0 || w <= 0)
            throw std::invalid_argument("Omni image dimensions must be positive");
        std::vector<Part> parts{ImageView{pixels, size_t(h) * w * 3 * sizeof(float), uint32_t(h),
                                          uint32_t(w), 3, ImageFormat::Float32},
                                TextPartView{text}};
        InputMessage message{MessageRole::User, {parts.data(), parts.size()}};
        return execute({&message, 1}, options);
    }
    TextResult run(const TextContinuationRequest& input, ConfigView options) override {
        const auto* text = std::get_if<std::string_view>(&input.prefix);
        if (!text)
            throw std::invalid_argument("Omni continuation accepts text rather than raw token IDs");
        return generate(std::string(*text), read_config(options, false));
    }
    TextResult run(const ImagesTextToTextRequest& input, ConfigView options) override {
        if (!input.tools.empty())
            throw std::invalid_argument("Omni adapter does not expose tool execution");
        return messages(input.messages, options);
    }
    TextResult run(const VideoTextToTextRequest& input, ConfigView options) override {
        if (!input.tools.empty())
            throw std::invalid_argument("Omni adapter does not expose tool execution");
        return messages(input.messages, options);
    }
    TextResult run(const AudioTextToTextRequest& input, ConfigView options) override {
        return messages(input.messages, options);
    }
    TextResult run(const ImageAudioTextToTextRequest& input, ConfigView options) override {
        return messages(input.messages, options);
    }
    TextResult run(const AudioVideoTextToTextRequest& input, ConfigView options) override {
        return messages(input.messages, options);
    }

  private:
    std::string task_;
    Json metadata_, config_;
    NativeDecoder decoder_;
    NativeMedia media_;
    Tokenizer tokenizer_;
    std::vector<char> embeddings_;
    std::vector<int> eos_;
    std::mutex mutex_;
    template <class P>
    TextResult messages(Span<const MediaMessage<P>> input, ConfigView options) {
        std::vector<std::vector<Part>> storage;
        std::vector<InputMessage> converted;
        storage.reserve(input.size());
        converted.reserve(input.size());
        for (const auto& source : input) {
            storage.emplace_back();
            for (const auto& part : source.parts)
                std::visit(
                    [&](const auto& value) {
                        using V = std::decay_t<decltype(value)>;
                        if constexpr (std::is_same_v<V, TextPartView> ||
                                      std::is_same_v<V, ImageView> ||
                                      std::is_same_v<V, VideoView> ||
                                      std::is_same_v<V, AudioView> ||
                                      std::is_same_v<V, AlignedAudioVideoView>)
                            storage.back().emplace_back(value);
                        else
                            throw std::invalid_argument("Unsupported Omni message content");
                    },
                    part);
            converted.push_back({source.role, {storage.back().data(), storage.back().size()}});
        }
        return execute({converted.data(), converted.size()}, read_config(options));
    }
    std::vector<float> audio(AudioView source) {
        const auto rate = source.sample_rate.value_or(16000);
        if (!source.channels || source.samples.empty() || source.samples.size() % source.channels ||
            rate == 0 || rate > uint32_t(std::numeric_limits<int32_t>::max()) ||
            source.samples.size() / source.channels > size_t(std::numeric_limits<int32_t>::max()))
            throw std::invalid_argument(
                "Omni PCM requires a valid sample rate, channels and complete frames");
        const size_t count = source.samples.size() / source.channels;
        if (count * uint64_t(16000) / rate >= uint64_t(media_.max_audio_frames() + 1) * 160)
            throw std::invalid_argument(
                "Omni audio exceeds built acoustic capacity; it is not truncated");
        std::vector<float> mono(count);
        for (size_t i = 0; i < count; ++i) {
            double value = 0;
            for (size_t channel = 0; channel < source.channels; ++channel) {
                const auto sample = source.samples[i * source.channels + channel];
                if (!std::isfinite(sample))
                    throw std::invalid_argument("Omni PCM must be finite");
                value += sample;
            }
            mono[i] = float(value / source.channels);
        }
        if (rate != 16000)
            mono = resample_pcm(mono.data(), int32_t(mono.size()), int32_t(rate), 16000);
        return media_.audio(mono);
    }
    EncodedMedia video(VideoView source) {
        if (source.frames.empty() || source.frames.size() > size_t(decoder_.capacity()) ||
            source.timestamps_seconds.size() != source.frames.size())
            throw std::invalid_argument(
                "Omni video requires actual frames and one explicit timestamp per frame");
        for (size_t i = 0; i < source.frames.size(); ++i)
            if (!std::isfinite(source.timestamps_seconds[i]) || source.timestamps_seconds[i] < 0 ||
                (i && source.timestamps_seconds[i] < source.timestamps_seconds[i - 1]))
                throw std::invalid_argument(
                    "Omni video timestamps must be finite, nonnegative and ordered");
        const auto& vision = config_.at("vision_config");
        const int target = vision.at("video_target_num_patches");
        const auto grid =
            video_grid(source.frames[0].height, source.frames[0].width, media_.patch(), target,
                       vision.value("video_maintain_aspect_ratio", false));
        if (grid.count() > media_.max_patches())
            throw std::invalid_argument("Omni video exceeds visual engine capacity");
        const size_t groups = (source.frames.size() + 1) / 2;
        const size_t features = grid.count() / 4;
        const double pruning = config_.value("video_pruning_rate", 0.);
        if (!std::isfinite(pruning) || pruning < 0 || pruning >= 1)
            throw std::invalid_argument("Omni video pruning must lie within [0, 1)");
        const size_t keep = std::max(features, size_t(groups * features * (1. - pruning)));
        if (keep >= size_t(decoder_.capacity()))
            throw std::invalid_argument("Omni video tokens exceed decoder capacity");
        std::vector<float> all;
        for (size_t i = 0; i < source.frames.size(); i += 2) {
            std::vector<std::vector<float>> frames;
            for (size_t j = 0; j < 2; ++j) {
                const auto frame = source.frames[std::min(i + j, source.frames.size() - 1)];
                if (frame.width != source.frames[0].width ||
                    frame.height != source.frames[0].height)
                    throw std::invalid_argument("Omni video frames must share dimensions");
                frames.push_back(resize_rgb(rgb(frame), frame.height, frame.width,
                                            grid.rows * media_.patch(),
                                            grid.columns * media_.patch()));
            }
            auto values =
                media_.visual(frames, grid.rows * media_.patch(), grid.columns * media_.patch());
            all.insert(all.end(), values.begin(), values.end());
        }
        const auto retained =
            video_retained_indices(all, int(features), decoder_.hidden(), pruning);
        EncodedMedia result;
        size_t cursor = 0;
        for (size_t group = 0; group < groups; ++group) {
            std::ostringstream prefix;
            prefix.imbue(std::locale::classic());
            prefix << std::fixed << std::setprecision(2);
            if (group)
                prefix << "\n";
            prefix << "Frame " << group * 2 + 1 << " sampled at "
                   << source.timestamps_seconds[group * 2] << " seconds";
            if (group * 2 + 1 < source.frames.size())
                prefix << " and frame " << group * 2 + 2 << " sampled at "
                       << source.timestamps_seconds[group * 2 + 1] << " seconds";
            result.text += prefix.str() + ": <img>";
            while (cursor < retained.size() && size_t(retained[cursor]) / features == group) {
                const auto offset = size_t(retained[cursor++]) * decoder_.hidden();
                result.visual.insert(result.visual.end(), all.begin() + offset,
                                     all.begin() + offset + decoder_.hidden());
                result.text += "<image>";
            }
            result.text += "</img>";
        }
        return result;
    }
    EncodedMedia content(const InputMessage& message, int remaining) {
        std::string text;
        std::vector<ImageView> images;
        std::vector<VideoView> videos;
        std::vector<AudioView> sounds;
        for (const auto& part : message.parts) {
            std::visit(
                [&](const auto& value) {
                    using V = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<V, TextPartView>)
                        text += value.text;
                    else if constexpr (std::is_same_v<V, ImageView>)
                        images.push_back(value);
                    else if constexpr (std::is_same_v<V, VideoView>)
                        videos.push_back(value);
                    else if constexpr (std::is_same_v<V, AudioView>)
                        sounds.push_back(value);
                    else {
                        if (!value.audio_start_seconds || *value.audio_start_seconds != 0 ||
                            value.video.timestamps_seconds.empty() ||
                            value.video.timestamps_seconds[0] != 0)
                            throw std::invalid_argument(
                                "Omni aligned audio-video requires an explicit shared zero origin");
                        videos.push_back(value.video);
                        sounds.push_back(value.audio);
                    }
                },
                part);
        }
        if (message.role != MessageRole::User &&
            (!images.empty() || !videos.empty() || !sounds.empty()))
            throw std::invalid_argument("Omni native media is accepted in user messages");
        EncodedMedia result, other;
        for (const auto& source : videos) {
            auto encoded = video(source);
            other.text += encoded.text + "\n";
            other.visual.insert(other.visual.end(), encoded.visual.begin(), encoded.visual.end());
        }
        for (const auto& source : sounds) {
            auto encoded = audio(source);
            other.text += "<so_start>" +
                          repeat("<so_embedding>", encoded.size() / decoder_.hidden()) +
                          "<so_end>\n";
            other.audio.insert(other.audio.end(), encoded.begin(), encoded.end());
        }
        const int budget = remaining - int(tokenizer_.encode(text + other.text).size()) - 32;
        if (!images.empty()) {
            std::vector<Grid> dimensions;
            for (const auto& image : images)
                dimensions.push_back({int(image.height), int(image.width)});
            const auto& vision = config_.at("vision_config").at("args");
            const int minimum = vision.at("min_num_patches");
            const int maximum =
                std::min(vision.at("max_num_patches").get<int>(), media_.max_patches());
            auto grids = image_grids(dimensions, media_.patch(), minimum, maximum, budget);
            for (size_t i = 0; i < images.size(); ++i) {
                const auto& source = images[i];
                const auto grid = grids[i];
                auto pixels = resize_rgb(rgb(source), source.height, source.width,
                                         grid.rows * media_.patch(), grid.columns * media_.patch());
                auto encoded = media_.visual({pixels}, grid.rows * media_.patch(),
                                             grid.columns * media_.patch());
                if (images.size() > 1) {
                    if (i)
                        result.text += " ";
                    result.text += "<image " + std::to_string(i + 1) + ">";
                }
                result.text +=
                    "<img>" + repeat("<image>", encoded.size() / decoder_.hidden()) + "</img>";
                result.visual.insert(result.visual.end(), encoded.begin(), encoded.end());
            }
            result.text += "\n";
        }
        result.text += other.text + text;
        result.visual.insert(result.visual.end(), other.visual.begin(), other.visual.end());
        result.audio = std::move(other.audio);
        return result;
    }
    TextResult execute(Span<const InputMessage> input, const TextGenerationConfig& options) {
        const std::lock_guard<std::mutex> guard(mutex_);
        validate_request(options);
        if (input.empty() || options.max_new_tokens >= decoder_.capacity())
            throw std::invalid_argument("Omni request requires messages and generation capacity");
        const auto start = Clock::now();
        std::vector<ChatMessage> history;
        if (!options.system_prompt.empty())
            history.push_back({"system", options.system_prompt});
        std::vector<float> visual, audio_values;
        int remaining = decoder_.capacity() - options.max_new_tokens;
        for (const auto& message : input) {
            auto encoded = content(message, remaining);
            remaining -= int(tokenizer_.encode(encoded.text).size()) + 8;
            history.push_back({role_name(message.role), encoded.text});
            visual.insert(visual.end(), encoded.visual.begin(), encoded.visual.end());
            audio_values.insert(audio_values.end(), encoded.audio.begin(), encoded.audio.end());
        }
        if (!options.use_chat_template &&
            (history.size() != 1 || !visual.empty() || !audio_values.empty()))
            throw std::invalid_argument(
                "Omni multimodal and multi-message requests require the checkpoint chat template");
        const auto prompt = options.use_chat_template
                                ? render_chat(history, options.enable_thinking)
                                : history[0].content;
        const auto ids = tokenizer_.encode(prompt);
        if (ids.empty() || ids.size() + options.max_new_tokens > size_t(decoder_.capacity()))
            throw std::invalid_argument("Omni prompt and requested output exceed built context "
                                        "capacity; nothing is truncated");
        const int image_id = config_.at("img_context_token_id"),
                  sound_id = config_.at("sound_context_token_id");
        if (size_t(std::count(ids.begin(), ids.end(), image_id)) * decoder_.hidden() !=
                visual.size() ||
            size_t(std::count(ids.begin(), ids.end(), sound_id)) * decoder_.hidden() !=
                audio_values.size())
            throw std::invalid_argument(
                "Omni prompt placeholders do not match actual media features");
        for (const auto id : ids)
            if (id < 0 || id >= decoder_.vocab())
                throw std::invalid_argument("Omni prompt token is outside model vocabulary");
        TextResult result;
        result.setup_ms = elapsed(start);
        decoder_.reset();
        auto embedding = [&](int id) {
            std::vector<float> values(decoder_.hidden());
            std::memcpy(values.data(),
                        embeddings_.data() + size_t(id) * decoder_.hidden() * sizeof(float),
                        values.size() * sizeof(float));
            return values;
        };
        const auto prefill = Clock::now();
        std::vector<float> logits;
        size_t v = 0, a = 0;
        for (const auto id : ids) {
            auto values = embedding(id);
            if (id == image_id) {
                std::copy_n(visual.data() + v, values.size(), values.data());
                v += values.size();
            }
            if (id == sound_id) {
                std::copy_n(audio_values.data() + a, values.size(), values.data());
                a += values.size();
            }
            logits = decoder_.step(values);
        }
        result.prefill_ms = elapsed(prefill);
        const auto params =
            nemotron_omni_sampling_params_from_config(options, eos_.empty() ? -1 : eos_[0]);
        auto sampler = create_nemotron_omni_sampler(params);
        std::vector<bool> seen(decoder_.vocab(), false);
        for (const auto id : ids)
            seen[id] = true;
        const auto decode = Clock::now();
        for (int step = 0; step < options.max_new_tokens; ++step) {
            if (options.repetition_penalty != 1)
                for (size_t i = 0; i < logits.size(); ++i)
                    if (seen[i])
                        logits[i] = logits[i] < 0 ? logits[i] * options.repetition_penalty
                                                  : logits[i] / options.repetition_penalty;
            const auto selected = sampler->sample(logits.data(), decoder_.vocab(), params).token_id;
            if (selected < 0 || selected >= decoder_.vocab())
                throw std::runtime_error("Omni sampler produced an invalid token");
            result.token_ids.push_back(selected);
            seen[selected] = true;
            if (std::find(eos_.begin(), eos_.end(), selected) != eos_.end() ||
                step + 1 == options.max_new_tokens)
                break;
            logits = decoder_.step(embedding(selected));
        }
        result.decode_ms = elapsed(decode);
        result.text = tokenizer_.decode(result.token_ids);
        return result;
    }
};
} // namespace
} // namespace trtmc::nemotron_omni
extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("nemotron_omni does not support --kv-cache-size");
    if (context.reader.find_section("edge_llm.json")) {
#ifdef TRTMC_HAS_EDGE_LLM
        return trtmc::nemotron_omni::edge_llm::create(context.reader);
#else
        throw std::runtime_error(
            "This Omni runtime was built without the optional Edge SDK; "
            "build a native bundle or install the matching Edge-enabled runtime");
#endif
    }
    return new trtmc::nemotron_omni::Pipeline(context);
}
