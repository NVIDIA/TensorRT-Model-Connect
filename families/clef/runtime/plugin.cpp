/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/clef/runtime/media.h"
#include "families/clef/runtime/record.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace trtmc::clef {
namespace {
float to_float(std::uint16_t value) {
    std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
std::uint16_t to_bf16(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>(bits >> 16);
}
void cuda_check(cudaError_t code) {
    if (code != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(code));
}

class Pipeline final : public IStructuredDecision {
  public:
    explicit Pipeline(const FamilyContext& context) {
        const auto bytes = context.reader.read_section("runtime.json");
        config_ = Json::parse(bytes.begin(), bytes.end());
        const auto& text = config_.at("text_config");
        hidden_ = text.at("hidden_size").get<int>();
        vocab_ = text.at("vocab_size").get<int>();
        max_length_ = config_.at("max_sequence_length").get<int>();
        if (hidden_ <= 0 || vocab_ <= 0 || max_length_ < 1 || config_.at("precision") != "bf16")
            throw std::invalid_argument("invalid Clef runtime geometry or precision");
        const auto tokenizer = context.reader.read_section("tokenizer.json");
        tokenizer_ = CreateBpeTokenizer(tokenizer.data(), tokenizer.size(), false);
        embedding_ = context.reader.read_section("embedding.bin");
        lexical_ = context.reader.read_section("lexical_embedding.bin");
        const auto embedding_bytes = static_cast<std::size_t>(hidden_) * vocab_ * 2;
        if (embedding_.size() != embedding_bytes || lexical_.size() != embedding_bytes)
            throw std::invalid_argument("Clef embedding shape mismatch");
        auto load = [&](const std::string& section) {
            const auto data = context.reader.read_section(section);
            ModuleCreateOptions options;
            options.stream = stream_;
            auto module = context.backend.create_module(data.data(), data.size(), options);
            if (!module || !module->ok())
                throw std::runtime_error("cannot load Clef " + section);
            if (!stream_)
                stream_ = module->stream();
            module->set_timing_label("clef " + section);
            return module;
        };
        if (context.reader.find_section("backbone.plan")) {
            layers_.push_back(load("backbone.plan"));
        } else {
            for (int i = 0; i < text.at("num_hidden_layers").get<int>(); ++i)
                layers_.push_back(load("layer." + std::to_string(i) + ".plan"));
        }
        norm_ = load("norm.plan");
        head_ = load("head.plan");
        if (context.reader.find_section("vision.plan")) {
            vision_ = load("vision.plan");
            vision_positions_ = context.reader.read_section("vision_positions.bin");
        }
    }

    StructuredDecisionResult decide(const StructuredDecisionRequest& request) override {
        std::lock_guard<std::mutex> guard(mutex_);
        const auto document = Json::parse(request.document);
        if (document.contains("model") && !document["model"].is_string())
            throw std::invalid_argument("model must be a string when supplied");
        if ((document.contains("images") && !document["images"].empty()) ||
            (document.contains("videos") && !document["videos"].empty()))
            throw std::invalid_argument(
                "pass decoded media in StructuredDecisionRequest images/videos");
        if ((!request.images.empty() || !request.videos.empty()) && !vision_)
            throw std::invalid_argument("this Clef bundle does not contain the vision encoder");
        const auto media =
            request.images.empty() && request.videos.empty()
                ? MediaRecord{}
                : preprocess_media(*tokenizer_, request, document, config_.at("processor_config"));
        const auto encoded = encode_record(*tokenizer_, document, max_length_,
                                           request.max_state_tokens, media.tokens);
        const auto sequence = static_cast<std::int64_t>(encoded.input_ids.size());
        const auto alignment = config_.value("backbone_sequence_alignment", 64);
        if (alignment != 1 && alignment != 64)
            throw std::invalid_argument("invalid backbone sequence alignment");
        const auto padded = ((sequence + alignment - 1) / alignment) * alignment;
        const auto q_count = static_cast<std::int64_t>(encoded.questions.size());
        std::int64_t o_count = 0;
        for (const auto& q : encoded.questions)
            o_count += q.option_ids.size();
        if (q_count > config_.at("max_questions").get<int>() ||
            o_count > config_.at("max_options").get<int>())
            throw std::invalid_argument("record exceeds bundle question or option capacity");
        std::vector<std::uint16_t> embedded(padded * hidden_, 0);
        const auto* embedding = reinterpret_cast<const std::uint16_t*>(embedding_.data());
        for (std::int64_t i = 0; i < sequence; ++i) {
            const auto id = encoded.input_ids[i];
            if (id < 0 || id >= vocab_)
                throw std::invalid_argument("token ID outside Clef vocabulary");
            std::copy_n(embedding + static_cast<std::size_t>(id) * hidden_, hidden_,
                        embedded.data() + i * hidden_);
        }
        std::vector<std::array<int, 3>> positions;
        if (!media.frames.empty()) {
            const int image_token = tokenizer_->id_for_token("<|image_pad|>");
            const int video_token = tokenizer_->id_for_token("<|video_pad|>");
            positions = media_positions(encoded, media, image_token, video_token);
            std::size_t input_offset = 0;
            for (std::size_t begin = 0; begin < media.frames.size();) {
                std::size_t end = begin + 1;
                if (vision_->has_input("frame_ids")) {
                    // The original runs all images together, and all video
                    // frames together. Attention stays within each frame.
                    while (end < media.frames.size() &&
                           media.frames[end].video == media.frames[begin].video)
                        ++end;
                }
                const auto features = encode_media(media.frames, begin, end);
                std::size_t feature_offset = 0;
                for (std::size_t i = begin; i < end; ++i) {
                    const auto& frame = media.frames[i];
                    const auto tokens =
                        static_cast<std::size_t>(frame.grid_height) * frame.grid_width / 4;
                    const int token = frame.video ? video_token : image_token;
                    while (input_offset < encoded.input_ids.size() &&
                           encoded.input_ids[input_offset] != token)
                        ++input_offset;
                    std::copy_n(features.data() + feature_offset * hidden_, tokens * hidden_,
                                embedded.data() + input_offset * hidden_);
                    input_offset += tokens;
                    feature_offset += tokens;
                }
                begin = end;
            }
        }
        const auto& text = config_.at("text_config");
        const auto rotary =
            static_cast<int>(text.at("head_dim").get<int>() *
                             text.at("rope_parameters").at("partial_rotary_factor").get<double>());
        const auto theta = text.at("rope_parameters").at("rope_theta").get<float>();
        std::vector<std::uint16_t> cos(padded * rotary), sin(padded * rotary);
        for (std::int64_t i = 0; i < padded; ++i) {
            for (int j = 0; j < rotary / 2; ++j) {
                const float inverse = 1.0F / std::pow(theta, static_cast<float>(2 * j) / rotary);
                int coordinate = static_cast<int>(i);
                if (!positions.empty()) {
                    if (i < sequence) {
                        const auto& sections = text.at("rope_parameters").at("mrope_section");
                        int axis = 0;
                        if (j % 3 == 1 && j < sections.at(1).get<int>() * 3)
                            axis = 1;
                        if (j % 3 == 2 && j < sections.at(2).get<int>() * 3)
                            axis = 2;
                        coordinate = positions[i][axis];
                    } else
                        coordinate =
                            *std::max_element(positions.back().begin(), positions.back().end()) +
                            1 + i - sequence;
                }
                const float angle = static_cast<float>(coordinate) * inverse;
                const auto c = to_bf16(std::cos(angle)), s = to_bf16(std::sin(angle));
                cos[i * rotary + j] = cos[i * rotary + j + rotary / 2] = c;
                sin[i * rotary + j] = sin[i * rotary + j + rotary / 2] = s;
            }
        }
        void* previous = nullptr;
        for (auto& layer : layers_) {
            TensorMap inputs;
            if (!previous)
                inputs["hidden_states"] = {embedded.data(), {padded, hidden_}, DType::kBFloat16};
            else
                layer->bind_external("hidden_states", previous, {padded, hidden_});
            if (layer->has_input("rope_cos")) {
                inputs["rope_cos"] = {cos.data(), {padded, 1, rotary}, DType::kBFloat16};
                inputs["rope_sin"] = {sin.data(), {padded, 1, rotary}, DType::kBFloat16};
            }
            layer->forward_async(inputs);
            previous = layer->device_ptr("output");
        }
        norm_->bind_external("hidden_states", previous, {sequence, hidden_});
        norm_->forward_async({});
        head_->bind_external("hidden_states", norm_->device_ptr("output"), {sequence, hidden_});
        std::vector<float> q_pool(q_count * sequence, 0), o_pool(o_count * sequence, 0);
        std::vector<std::int32_t> option_fields, types;
        std::vector<std::uint16_t> lexical_options(o_count * hidden_);
        std::vector<std::uint16_t> mask(q_count * o_count,
                                        to_bf16(-std::numeric_limits<float>::infinity()));
        const auto* lexical = reinterpret_cast<const std::uint16_t*>(lexical_.data());
        std::int64_t option = 0;
        for (std::int64_t qi = 0; qi < q_count; ++qi) {
            const auto& q = encoded.questions[qi];
            types.push_back(q.type);
            for (int i = q.span.first; i < q.span.second; ++i)
                q_pool[qi * sequence + i] = 1.0F / (q.span.second - q.span.first);
            for (const auto& [begin, end] : q.option_spans) {
                option_fields.push_back(qi);
                mask[qi * o_count + option] = to_bf16(0.0F);
                std::vector<float> sum(hidden_, 0);
                for (int i = begin; i < end; ++i) {
                    o_pool[option * sequence + i] = 1.0F / (end - begin);
                    const auto* row =
                        lexical + static_cast<std::size_t>(encoded.input_ids[i]) * hidden_;
                    for (int j = 0; j < hidden_; ++j)
                        sum[j] += to_float(row[j]);
                }
                for (int j = 0; j < hidden_; ++j)
                    lexical_options[option * hidden_ + j] = to_bf16(sum[j] / (end - begin));
                ++option;
            }
        }
        auto last = static_cast<std::int32_t>(sequence - 1);
        head_->forward_async({
            {"lexical_options", {lexical_options.data(), {o_count, hidden_}, DType::kBFloat16}},
            {"question_pool", {q_pool.data(), {q_count, sequence}, DType::kFloat32}},
            {"option_pool", {o_pool.data(), {o_count, sequence}, DType::kFloat32}},
            {"option_fields", {option_fields.data(), {o_count}, DType::kInt32}},
            {"type_ids", {types.data(), {q_count}, DType::kInt32}},
            {"group_mask", {mask.data(), {q_count, o_count}, DType::kBFloat16}},
            {"last_index", {&last, {1}, DType::kInt32}},
        });
        std::vector<float> logits(o_count);
        cuda_check(cudaMemcpyAsync(logits.data(), head_->device_ptr("logits"),
                                   logits.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_));
        cuda_check(cudaStreamSynchronize(stream_));
        StructuredDecisionResult result;
        result.input_tokens = sequence;
        Json answers = Json::object();
        option = 0;
        for (const auto& question : encoded.questions) {
            DecisionScores scores;
            scores.question_id = question.id;
            scores.option_ids = question.option_ids;
            scores.logits.assign(logits.begin() + option,
                                 logits.begin() + option + question.option_ids.size());
            const auto max = *std::max_element(scores.logits.begin(), scores.logits.end());
            float sum = 0;
            for (const auto logit : scores.logits) {
                if (!std::isfinite(logit))
                    throw std::runtime_error("Clef returned non-finite logits");
                scores.probabilities.push_back(std::exp(logit - max));
                sum += scores.probabilities.back();
            }
            for (auto& p : scores.probabilities)
                p /= sum;
            answers[question.id] = systemone_answer(document["questions"][question.id],
                                                    question.option_ids, scores.probabilities);
            option += question.option_ids.size();
            result.scores.push_back(std::move(scores));
        }
        result.document = Json({{"model", document.value("model", Json("clef"))},
                                {"answers", answers},
                                {"usage", {{"input_tokens", sequence}, {"output_tokens", 0}}}})
                              .dump();
        return result;
    }

  private:
    std::vector<std::uint16_t> encode_media(const std::vector<VisionFrame>& frames,
                                            std::size_t begin, std::size_t end) {
        const auto& geometry = config_.at("vision_config");
        const int width = geometry.at("hidden_size").get<int>();
        const int heads = geometry.at("num_heads").get<int>();
        const int side =
            static_cast<int>(std::sqrt(geometry.at("num_position_embeddings").get<int>()));
        std::vector<float> patches, positions, cos, sin;
        std::vector<std::int32_t> groups;
        for (std::size_t i = begin; i < end; ++i) {
            const auto& frame = frames[i];
            const auto count = static_cast<std::size_t>(frame.grid_height) * frame.grid_width;
            patches.insert(patches.end(), frame.patches.begin(), frame.patches.end());
            std::vector<float> p, c, s;
            vision_positions(frame, vision_positions_, width, heads, side, p, c, s,
                             config_.value("round_vision_products", false));
            positions.insert(positions.end(), p.begin(), p.end());
            cos.insert(cos.end(), c.begin(), c.end());
            sin.insert(sin.end(), s.begin(), s.end());
            groups.insert(groups.end(), count, static_cast<std::int32_t>(i - begin));
        }
        const auto count = static_cast<std::int64_t>(groups.size());
        const auto capacity =
            vision_->input_profile_shape("patches", 0, ProfileShapeSelector::kMax);
        if (capacity.empty() || count > capacity[0])
            throw std::invalid_argument("media exceeds the bundle vision capacity");
        TensorMap inputs = {
            {"patches", {patches.data(), {count, 1536}, DType::kFloat32}},
            {"positions", {positions.data(), {count, width}, DType::kFloat32}},
            {"rope_cos", {cos.data(), {count, 1, width / heads}, DType::kFloat32}},
            {"rope_sin", {sin.data(), {count, 1, width / heads}, DType::kFloat32}},
        };
        if (vision_->has_input("frame_ids"))
            inputs["frame_ids"] = {groups.data(), {count}, DType::kInt32};
        const auto outputs = vision_->forward(inputs);
        const auto& features = outputs.at("visual_embeddings");
        if (features.dtype != DType::kBFloat16 ||
            features.numel() != static_cast<std::size_t>(count / 4) * hidden_)
            throw std::runtime_error("vision encoder returned incompatible embeddings");
        std::vector<std::uint16_t> result(features.numel());
        std::memcpy(result.data(), features.data, features.nbytes());
        return result;
    }
    Json config_;
    int hidden_{0}, vocab_{0}, max_length_{0};
    std::vector<char> embedding_, lexical_, vision_positions_;
    std::unique_ptr<ITokenizer> tokenizer_;
    std::vector<std::unique_ptr<ITrtModule>> layers_;
    std::unique_ptr<ITrtModule> norm_, head_, vision_;
    cudaStream_t stream_{nullptr};
    std::mutex mutex_;
};
} // namespace
} // namespace trtmc::clef

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("clef does not support --kv-cache-size");
    return new trtmc::clef::Pipeline(context);
}
