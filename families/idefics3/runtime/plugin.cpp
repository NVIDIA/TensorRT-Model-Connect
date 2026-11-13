/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/idefics3/runtime/preprocess.h"
#include "families/idefics3/runtime/tokenizer.h"
#include "trtmc/internal/language.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::idefics3 {
namespace {
std::vector<char> section(const BundleReader& reader, const char* name) {
    const auto* found = reader.find_section(name);
    if (!found || found->length == 0) throw std::runtime_error(std::string("Idefics3 bundle lacks ") + name);
    return reader.read_section(name);
}
Tensor tensor(void* data, std::initializer_list<int64_t> shape, DType dtype) {
    Tensor result;
    result.data = data; result.shape = shape; result.dtype = dtype;
    return result;
}
const Tensor& output(const TensorMap& outputs, const char* name, std::size_t count) {
    const auto found = outputs.find(name);
    if (found == outputs.end() || !found->second.data || found->second.dtype != DType::kFloat32 ||
        found->second.numel() != count) throw std::runtime_error(std::string("Invalid Idefics3 output ") + name);
    return found->second;
}

class Pipeline final : public internal::IModel, public internal::IImagesTextToText {
  public:
    Pipeline(std::unique_ptr<ITrtModule> vision, std::unique_ptr<ITrtModule> decoder,
             std::unique_ptr<ITokenizer> tokenizer, nlohmann::json cfg)
        : vision_(std::move(vision)), decoder_(std::move(decoder)), tokenizer_(std::move(tokenizer)),
          cfg_(std::move(cfg)), fields_{
            {"max_new_tokens", internal::ConfigKind::I64, internal::ConfigValue{int64_t{64}}, "Maximum generated tokens"},
            {"do_sample", internal::ConfigKind::Bool, internal::ConfigValue{false}, "Greedy decoding only"}} {}
    const char* task() const noexcept override { return "images_text_to_text"; }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::IImagesTextToText>(*this, {fields_.data(), fields_.size()})};
    }
    internal::TextResult run(const internal::ImagesTextToTextRequest& request,
                            internal::ConfigView config) override {
        const Span<const internal::ConfigField> fields{fields_.data(), fields_.size()};
        internal::validate_config(fields, config);
        const auto max_new = *internal::config_get<int64_t>(config, fields, "max_new_tokens");
        if (*internal::config_get<bool>(config, fields, "do_sample"))
            throw std::invalid_argument("Idefics3 currently supports greedy decoding");
        const int32_t length = cfg_.at("sequence_length"), width = cfg_.at("hidden_size");
        const int32_t vocab = cfg_.at("vocab_size"), image_size = cfg_.at("image_size");
        const int32_t image_sequence = cfg_.at("image_sequence"), image_id = cfg_.at("image_id");
        if (max_new <= 0 || max_new > length) throw std::invalid_argument("Invalid Idefics3 generation length");
        if (request.messages.size() != 1 || !request.tools.empty() ||
            request.messages[0].role != internal::MessageRole::User || request.messages[0].parts.size() != 2)
            throw std::invalid_argument("Idefics3 requires one user message with an image followed by a question");
        const auto* image = std::get_if<internal::ImageView>(&request.messages[0].parts[0]);
        const auto* question = std::get_if<internal::TextPartView>(&request.messages[0].parts[1]);
        if (!image || !question || question->text.empty())
            throw std::invalid_argument("Idefics3 requires an image followed by a nonempty question");
        if (!image->data || image->format != internal::ImageFormat::Float32 || image->channels != 3 ||
            image->height == 0 || image->width == 0 ||
            image->height > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
            image->width > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
            static_cast<uint64_t>(image->height) > std::numeric_limits<std::size_t>::max() / image->width / 3 / sizeof(float) ||
            image->byte_size != static_cast<std::size_t>(image->height) * image->width * 3 * sizeof(float))
            throw std::invalid_argument("Idefics3 requires contiguous float32 RGB input");
        auto prepared = preprocess_idefics3_image(static_cast<const float*>(image->data),
            image->height, image->width, cfg_.at("longest_edge"), image_size);
        const std::string prompt = cfg_.at("chat_prefix").get<std::string>() +
            idefics3_image_prompt(prepared.rows, prepared.columns, image_sequence) +
            cfg_.at("between_image_text").get<std::string>() + std::string(question->text) +
            cfg_.at("chat_suffix").get<std::string>();
        auto ids = tokenizer_->encode(prompt);
        const auto initial_count = ids.size();
        if (initial_count == 0 || initial_count + max_new > static_cast<std::size_t>(length))
            throw std::invalid_argument("Idefics3 image/question and generation exceed the built sequence profile");
        const auto expected_images = prepared.views.size() * image_sequence;
        if (static_cast<std::size_t>(std::count(ids.begin(), ids.end(), image_id)) != expected_images)
            throw std::invalid_argument("Idefics3 image placeholder count does not match its tiled views");
        std::vector<float> image_features;
        image_features.reserve(expected_images * width);
        for (auto& pixels : prepared.views) {
            const auto outputs = vision_->forward({{"pixel_values", tensor(pixels.data(),
                {1, 3, image_size, image_size}, DType::kFloat32)}});
            const auto& features = output(outputs, "image_features", image_sequence * width);
            const auto* values = static_cast<const float*>(features.data);
            image_features.insert(image_features.end(), values, values + features.numel());
        }
        std::vector<float> image_rows(static_cast<std::size_t>(length) * width, 0), image_mask(length, 0);
        std::size_t seen = 0;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] == image_id) {
                std::copy_n(image_features.data() + seen * width, width, image_rows.data() + i * width);
                image_mask[i] = 1; ++seen;
            }
        }
        std::vector<int32_t> valid(length, 0);
        std::fill_n(valid.begin(), initial_count, 1);
        ids.resize(length, cfg_.at("pad_id").get<int32_t>());
        const auto eos = cfg_.at("eos_ids").get<std::vector<int32_t>>();
        internal::TextResult result;
        for (int64_t i = 0; i < max_new; ++i) {
            int32_t last = static_cast<int32_t>(initial_count + i - 1);
            const auto predicted = decoder_->forward({
                {"input_ids", tensor(ids.data(), {length}, DType::kInt32)},
                {"attention_mask", tensor(valid.data(), {length}, DType::kInt32)},
                {"image_rows", tensor(image_rows.data(), {length, width}, DType::kFloat32)},
                {"image_mask", tensor(image_mask.data(), {length, 1}, DType::kFloat32)},
                {"last_index", tensor(&last, {1}, DType::kInt32)}});
            const auto& logits = output(predicted, "logits", vocab);
            const auto* values = static_cast<const float*>(logits.data);
            for (int32_t j = 0; j < vocab; ++j)
                if (!std::isfinite(values[j])) throw std::runtime_error("Idefics3 logit is nonfinite");
            const auto next = static_cast<int32_t>(std::max_element(values, values + vocab) - values);
            result.token_ids.push_back(next);
            if (std::find(eos.begin(), eos.end(), next) != eos.end()) break;
            ids[initial_count + i] = next; valid[initial_count + i] = 1;
        }
        result.text = tokenizer_->decode(result.token_ids);
        return result;
    }
  private:
    std::unique_ptr<ITrtModule> vision_, decoder_;
    std::unique_ptr<ITokenizer> tokenizer_;
    nlohmann::json cfg_;
    std::vector<internal::ConfigField> fields_;
};
} // namespace
} // namespace trtmc::idefics3

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    if (context.kv_cache_size_bytes || context.reader.info().task != "images_text_to_text")
        throw std::invalid_argument("Idefics3 requires image/text generation without an external KV cache");
    const auto raw = idefics3::section(context.reader, "runtime.json");
    const auto cfg = nlohmann::json::parse(raw.begin(), raw.end());
    const int32_t length = cfg.at("sequence_length"), width = cfg.at("hidden_size");
    const int32_t vocab = cfg.at("vocab_size"), size = cfg.at("image_size"), edge = cfg.at("longest_edge");
    const int32_t image_sequence = cfg.at("image_sequence"), image_id = cfg.at("image_id"), pad = cfg.at("pad_id");
    if (length <= 0 || width <= 0 || vocab <= 0 || size <= 0 || size > 2048 ||
        edge <= 0 || edge > 4 * size || image_sequence <= 0 || image_id < 0 || image_id >= vocab || pad < 0 || pad >= vocab)
        throw std::runtime_error("Invalid Idefics3 runtime dimensions");
    auto load = [&](const char* name) {
        const auto plan = idefics3::section(context.reader, name);
        auto engine = context.backend.create_module(plan.data(), plan.size(), {});
        if (!engine || !engine->ok()) throw std::runtime_error(std::string("Idefics3 failed to load ") + name);
        return engine;
    };
    const auto raw_tokenizer = idefics3::section(context.reader, "tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(raw_tokenizer.data(), raw_tokenizer.size(), false);
    return new idefics3::Pipeline(load("vision.plan"), load("decoder.plan"), std::move(tokenizer), cfg);
}
