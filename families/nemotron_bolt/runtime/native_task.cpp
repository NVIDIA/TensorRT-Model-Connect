/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "native_task.h"

#include "chat.h"
#include "media_geometry.h"
#include "native_language.h"
#include "native_lilicorr.h"
#include "native_vision.h"
#include "request.h"
#include "sampler.h"
#include "tokenizer.h"
#include "visual_frontend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_bf16.h>
#include <iomanip>
#include <limits>
#include <locale>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream>
#include <type_traits>
namespace trtmc::nemotron_bolt {
namespace {
using namespace internal;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using Part = std::variant<TextPartView, ImageView, VideoView>;
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
    std::vector<float> visual;
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
        throw std::invalid_argument("Bolt requires nonempty RGB image input");
    const size_t count = size_t(image.height) * image.width * 3;
    const bool integer = image.format == ImageFormat::UInt8;
    if ((!integer && image.format != ImageFormat::Float32) ||
        count > std::numeric_limits<size_t>::max() / sizeof(float) ||
        image.byte_size != count * (integer ? 1 : sizeof(float)))
        throw std::invalid_argument("Bolt image byte size or format is invalid");
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i) {
        const float value = integer ? static_cast<const uint8_t*>(image.data)[i] / 255.f
                                    : static_cast<const float*>(image.data)[i];
        if (!std::isfinite(value) || value < 0 || value > 1)
            throw std::invalid_argument("Bolt float RGB pixels must be finite within [0, 1]");
        result[i] = value;
    }
    return result;
}

class TextTask : public ITextGeneration, public IModel, public ITextContinuation {
  public:
    explicit TextTask(const FamilyContext& c)
        : task_(c.reader.info().task), meta_(Json::parse(text_asset(c.reader, "native.json"))),
          width_(meta_.at("hidden")), vocab_(meta_.at("vocab")), capacity_(meta_.at("capacity")),
          language_(
              c.backend,
              [&](const std::string& k) {
                  return c.reader.read_section("language/" + k + ".plan");
              },
              meta_.at("block_types").get<std::vector<std::string>>(), width_, vocab_, capacity_),
          tokenizer_(text_asset(c.reader, "tokenizer.json")),
          embedding_(c.reader.read_section("embeddings.bf16")) {
        if (meta_.at("version") != 1 || meta_.at("model_type") != "nemotron_h_puzzle" ||
            (meta_.at("execution_variant") != "autoregressive" &&
             meta_.at("execution_variant") != "lilicorr") ||
            meta_.at("precision") != "bf16" || width_ != 4096 || vocab_ != 131072 ||
            meta_.at("block_types").size() != 88 ||
            embedding_.size() != size_t(width_) * vocab_ * 2)
            throw std::invalid_argument("Invalid Bolt native language assets");
        eos_ = Json::parse(text_asset(c.reader, "generation.json"))
                   .at("eos_token_id")
                   .get<std::vector<int>>();
        if (eos_ != std::vector<int>{2, 11})
            throw std::invalid_argument("Unexpected Bolt EOS set");
        if (meta_.at("execution_variant") == "lilicorr") {
            draft_ = std::make_unique<NativeLiLiCorr>(c.backend, [&](const std::string& name) {
                return c.reader.read_section("draft/" + name + ".plan");
            });
        }
        if (task_ == "vision_language_generation") {
            const auto bytes = c.reader.read_section("positions.f32");
            if (bytes.size() != size_t(128) * 128 * 1280 * sizeof(float))
                throw std::invalid_argument("Invalid visual positions");
            std::vector<float> pos(bytes.size() / sizeof(float));
            std::memcpy(pos.data(), bytes.data(), bytes.size());
            vision_ = std::make_unique<NativeVision>(
                c.backend,
                [&](const std::string& k) {
                    return c.reader.read_section("vision/" + k + ".plan");
                },
                std::move(pos), 128, meta_.at("max_patches"));
        } else if (task_ != "text_generation")
            throw std::invalid_argument("Unknown native Bolt task");
    }
    const char* task() const noexcept override { return task_.c_str(); }
    int32_t default_max_new_tokens() const override { return 128; }
    std::vector<TaskInstance> task_bindings() override {
        return {bind<ITextContinuation>(*this, field_view(false))};
    }
    TextResult generate(const std::string& text, const TextGenerationConfig& options) override {
        std::vector<Part> parts{TextPartView{text}};
        InputMessage msg{MessageRole::User, {parts.data(), 1}};
        return execute({&msg, 1}, options);
    }
    TextResult run(const TextContinuationRequest& req, ConfigView options) override {
        const auto* text = std::get_if<std::string_view>(&req.prefix);
        if (!text)
            throw std::invalid_argument("Bolt expects tokenizable text");
        return generate(std::string(*text), read_config(options, false));
    }

  protected:
    std::string task_;
    Json meta_;
    int width_, vocab_, capacity_;
    NativeLanguage language_;
    std::unique_ptr<NativeLiLiCorr> draft_;
    Tokenizer tokenizer_;
    std::vector<char> embedding_;
    std::vector<int> eos_;
    std::unique_ptr<NativeVision> vision_;
    std::mutex mutex_;
    template <class P>
    TextResult messages(Span<const MediaMessage<P>> input, ConfigView options) {
        std::vector<std::vector<Part>> storage;
        std::vector<InputMessage> converted;
        storage.reserve(input.size());
        converted.reserve(input.size());
        for (const auto& msg : input) {
            auto& parts = storage.emplace_back();
            for (const auto& p : msg.parts)
                std::visit(
                    [&](const auto& v) {
                        using V = std::decay_t<decltype(v)>;
                        if constexpr (std::is_same_v<V, TextPartView> ||
                                      std::is_same_v<V, ImageView> || std::is_same_v<V, VideoView>)
                            parts.emplace_back(v);
                        else
                            throw std::invalid_argument("Unsupported Bolt content");
                    },
                    p);
            converted.push_back({msg.role, {parts.data(), parts.size()}});
        }
        return execute({converted.data(), converted.size()}, read_config(options));
    }
    EncodedMedia video(VideoView source) {
        if (source.frames.empty() || source.frames.size() > size_t(capacity_) ||
            source.timestamps_seconds.size() != source.frames.size())
            throw std::invalid_argument(
                "Bolt video requires actual frames and one explicit timestamp per frame");
        for (size_t i = 0; i < source.frames.size(); ++i)
            if (!std::isfinite(source.timestamps_seconds[i]) || source.timestamps_seconds[i] < 0 ||
                (i && source.timestamps_seconds[i] < source.timestamps_seconds[i - 1]))
                throw std::invalid_argument(
                    "Bolt video timestamps must be finite, nonnegative and ordered");
        const auto& vision = meta_;
        const int target = vision.at("video_patches");
        const auto grid =
            video_grid(source.frames[0].height, source.frames[0].width, 16, target, true);
        if (grid.count() > meta_.at("max_patches").get<int>())
            throw std::invalid_argument("Bolt video exceeds visual engine capacity");
        const size_t groups = (source.frames.size() + 1) / 2;
        const size_t features = grid.count() / 4;
        const double pruning = meta_.at("pruning").get<double>();
        if (!std::isfinite(pruning) || pruning < 0 || pruning >= 1)
            throw std::invalid_argument("Bolt video pruning must lie within [0, 1)");
        const size_t keep = std::max(features, size_t(groups * features * (1. - pruning)));
        if (keep >= size_t(capacity_))
            throw std::invalid_argument("Bolt video tokens exceed decoder capacity");
        std::vector<float> all;
        for (size_t i = 0; i < source.frames.size(); i += 2) {
            std::vector<std::vector<float>> frames;
            for (size_t j = 0; j < 2; ++j) {
                const auto frame = source.frames[std::min(i + j, source.frames.size() - 1)];
                if (frame.width != source.frames[0].width ||
                    frame.height != source.frames[0].height)
                    throw std::invalid_argument("Bolt video frames must share dimensions");
                frames.push_back(resize_rgb(rgb(frame), frame.height, frame.width, grid.rows * 16,
                                            grid.columns * 16));
            }
            auto values = vision_->encode(frames, grid.rows * 16, grid.columns * 16);
            all.insert(all.end(), values.begin(), values.end());
        }
        const auto retained = video_retained_indices(all, int(features), width_, pruning);
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
                const auto offset = size_t(retained[cursor++]) * width_;
                result.visual.insert(result.visual.end(), all.begin() + offset,
                                     all.begin() + offset + width_);
                result.text += "<image>";
            }
            result.text += "</img>";
        }
        return result;
    }
    EncodedMedia content(const InputMessage& msg, int available) {
        std::vector<ImageView> images;
        std::vector<VideoView> videos;
        std::string plain;
        for (const auto& p : msg.parts) {
            if (auto* t = std::get_if<TextPartView>(&p))
                plain += t->text;
            else if (auto* i = std::get_if<ImageView>(&p))
                images.push_back(*i);
            else
                videos.push_back(std::get<VideoView>(p));
        }
        if (images.empty() && videos.empty())
            return {plain, {}};
        if (!vision_ || msg.role != MessageRole::User)
            throw std::invalid_argument("Bolt media requires a user vision request");
        const bool explicit_images = plain.find("<image>") != plain.npos,
                   explicit_video = plain.find("<video>") != plain.npos;
        std::string formatted;
        std::size_t i = 0;
        for (const auto& p : msg.parts) {
            if (auto* t = std::get_if<TextPartView>(&p))
                formatted += t->text;
            else if (std::holds_alternative<ImageView>(p)) {
                ++i;
                if (!explicit_images) {
                    if (images.size() > 1)
                        formatted += "<image " + std::to_string(i) + ">";
                    formatted += "<image>";
                }
            } else if (!explicit_video)
                formatted += "<video>";
        }
        std::vector<Grid> dims;
        for (auto im : images)
            dims.push_back({int(im.height), int(im.width)});
        std::vector<Grid> grids;
        if (!dims.empty())
            grids =
                image_grids(dims, 16, meta_.at("min_patches"), meta_.at("max_patches"), available);
        EncodedMedia out;
        std::size_t vi = 0, ii = 0, pos = 0;
        for (;;) {
            auto ip = formatted.find("<image>", pos), vp = formatted.find("<video>", pos),
                 next = std::min(ip, vp);
            out.text += formatted.substr(pos, next == formatted.npos ? formatted.npos : next - pos);
            if (next == formatted.npos)
                break;
            EncodedMedia media;
            if (next == ip) {
                if (ii >= images.size())
                    throw std::invalid_argument("Missing Bolt image data");
                const auto im = images[ii];
                const auto grid = grids[ii++];
                auto tile =
                    resize_rgb(rgb(im), im.height, im.width, 16 * grid.rows, 16 * grid.columns);
                media.visual =
                    vision_->encode({std::move(tile)}, 16 * grid.rows, 16 * grid.columns);
                media.text = "<img>" + repeat("<image>", media.visual.size() / width_) + "</img>";
            } else {
                if (vi >= videos.size())
                    throw std::invalid_argument("Missing Bolt video data");
                media = video(videos[vi++]);
            }
            out.text += media.text;
            out.visual.insert(out.visual.end(), media.visual.begin(), media.visual.end());
            pos = next + 7;
        }
        if (ii != images.size() || vi != videos.size())
            throw std::invalid_argument("Unmatched Bolt media");
        return out;
    }
    TextResult execute(Span<const InputMessage> msgs, const TextGenerationConfig& opt) {
        std::lock_guard<std::mutex> guard(mutex_);
        validate_request(opt);
        const int budget = opt.max_new_tokens;
        if (msgs.empty() || budget >= capacity_)
            throw std::invalid_argument("Bolt request exceeds capacity");
        auto start = Clock::now();
        std::vector<ChatMessage> hist;
        std::vector<float> feats;
        if (!opt.system_prompt.empty())
            hist.push_back({"system", opt.system_prompt});
        int available = capacity_ - budget;
        for (const auto& m : msgs) {
            auto part = content(m, available);
            available -= int(tokenizer_.encode(part.text).size());
            hist.push_back({role_name(m.role), std::move(part.text)});
            feats.insert(feats.end(), part.visual.begin(), part.visual.end());
        }
        if (!opt.use_chat_template && (hist.size() != 1 || !feats.empty()))
            throw std::invalid_argument("Bolt raw input must be one text prompt");
        auto ids = tokenizer_.encode(opt.use_chat_template ? render_chat(hist, opt.enable_thinking)
                                                           : hist.front().content);
        const int image_id = meta_.at("image_token");
        if (ids.empty() || ids.size() + budget > size_t(capacity_) ||
            size_t(std::count(ids.begin(), ids.end(), image_id)) * width_ != feats.size())
            throw std::invalid_argument("Bolt capacity or visual-token count mismatch");
        TextResult result;
        result.setup_ms = elapsed(start);
        auto pre = Clock::now();
        language_.reset();
        // Earlier positions cannot enter a future 1024-wide draft window.
        const size_t first = ids.size() > 1024 ? ids.size() - 1024 : 0;
        if (draft_)
            draft_->reset(int(first));
        std::vector<uint16_t> row(width_), captured;

        std::vector<float> logits;
        std::size_t index = 0;
        auto fetch = [&](int id) {
            if (id < 0 || id >= vocab_)
                throw std::invalid_argument("Bolt token outside vocabulary");
            std::memcpy(row.data(), embedding_.data() + std::size_t(id) * width_ * 2, width_ * 2);
        };
        for (size_t i = 0; i < ids.size(); ++i) {
            fetch(ids[i]);
            if (ids[i] == image_id) {
                for (auto& v : row) {
                    auto f = __float2bfloat16_rn(feats[index++]);
                    std::memcpy(&v, &f, sizeof(v));
                }
            }
            bool retain = draft_ && i >= first;
            logits = language_.step(row.data(), row.size(), i + 1 == ids.size(),
                                    retain ? &captured : nullptr);
            if (retain)
                draft_->accepted(captured);
        }
        result.prefill_ms = elapsed(pre);
        auto dec = Clock::now();
        auto params = nemotron_bolt_sampling_params_from_config(opt, eos_.front());
        auto sampler = create_nemotron_bolt_sampler(params);
        std::vector<bool> seen(vocab_, false);
        for (auto id : ids)
            seen[id] = true;
        auto penalize = [&] {
            if (opt.repetition_penalty != 1)
                for (int j = 0; j < vocab_; ++j)
                    if (seen[j])
                        logits[j] = logits[j] < 0 ? logits[j] * opt.repetition_penalty
                                                  : logits[j] / opt.repetition_penalty;
        };
        auto emit = [&](int id) {
            if (id < 0 || id >= vocab_)
                throw std::runtime_error("Bolt sampler emitted invalid token");
            result.token_ids.push_back(id);
            seen[id] = true;
            return result.token_ids.size() == size_t(budget) ||
                   std::find(eos_.begin(), eos_.end(), id) != eos_.end();
        };
        if (!draft_) {
            for (int i = 0; i < budget; ++i) {
                penalize();
                int id = sampler->sample(logits.data(), vocab_, params).token_id;
                if (emit(id))
                    break;
                fetch(id);
                logits = language_.step(row.data(), row.size(), true);
            }
        } else if (budget > 0) {
            BoltVerificationSampler verification(params);
            auto advance = [&](int id) {
                fetch(id);
                logits = language_.step(row.data(), row.size(), true, &captured);
                draft_->accepted(captured);
                penalize();
            };
            penalize();
            int root = verification.target(logits);
            for (;;) {
                if (emit(root))
                    break;
                // The root is the new token after the committed history. It must not yet
                // appear in the draft KV cache when its seven successors are proposed.
                const auto chain = draft_->propose(root, embedding_, verification);
                advance(root);
                bool rejected = false, done = false;
                for (const auto& choice : chain) {
                    const auto [id, accepted] = verification.verify(logits, choice);
                    if (!accepted) {
                        root = id;
                        rejected = true;
                        break;
                    }
                    if (emit(id)) {
                        done = true;
                        break;
                    }
                    advance(id);
                }
                if (done)
                    break;
                if (!rejected)
                    root = verification.target(logits);
            }
        }
        result.decode_ms = elapsed(dec);
        result.text = tokenizer_.decode(result.token_ids);
        return result;
    }
};
class VisionTask final : public TextTask,
                         public IVisionLanguageGeneration,
                         public IImagesTextToText,
                         public IVideoTextToText {
  public:
    using TextTask::TextTask;
    const char* task() const noexcept override { return TextTask::task(); }
    std::vector<TaskInstance> task_bindings() override {
        return {bind<ITextContinuation>(*this, field_view(false)),
                bind<IImagesTextToText>(*this, field_view()),
                bind<IVideoTextToText>(*this, field_view())};
    }
    TextResult generate(const std::string& text, const float* pixels, int32_t h, int32_t w,
                        const TextGenerationConfig& opt) override {
        if (h <= 0 || w <= 0)
            throw std::invalid_argument("Invalid Bolt image dimensions");
        std::vector<Part> parts{ImageView{pixels, size_t(h) * w * 3 * sizeof(float), uint32_t(h),
                                          uint32_t(w), 3, ImageFormat::Float32},
                                TextPartView{text}};
        InputMessage msg{MessageRole::User, {parts.data(), parts.size()}};
        return execute({&msg, 1}, opt);
    }
    TextResult run(const ImagesTextToTextRequest& r, ConfigView c) override {
        if (!r.tools.empty())
            throw std::invalid_argument("Bolt does not implement tools");
        return messages(r.messages, c);
    }
    TextResult run(const VideoTextToTextRequest& r, ConfigView c) override {
        if (!r.tools.empty())
            throw std::invalid_argument("Bolt does not implement tools");
        return messages(r.messages, c);
    }
};
} // namespace
ITask* create_native(const FamilyContext& c) {
    if (c.reader.info().task == "vision_language_generation")
        return new VisionTask(c);
    return new TextTask(c);
}
} // namespace trtmc::nemotron_bolt
