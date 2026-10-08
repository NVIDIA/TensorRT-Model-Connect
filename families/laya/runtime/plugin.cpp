/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/laya/runtime/record.h"
#include "families/laya/runtime/routing.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <numeric>
#include <stdexcept>

namespace trtmc::laya {
namespace {
std::vector<float> softmax(const float* values, std::size_t count) {
    if (count == 0)
        throw std::invalid_argument("empty probability distribution");
    const float maximum = *std::max_element(values, values + count);
    float total = 0;
    std::vector<float> result;
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i]))
            throw std::runtime_error("Laya returned non-finite logits");
        result.push_back(std::exp(values[i] - maximum));
        total += result.back();
    }
    for (auto& value : result)
        value /= total;
    return result;
}

class Pipeline final : public IStructuredDecision {
  public:
    explicit Pipeline(const FamilyContext& context, const std::string& prefix = "") {
        const auto metadata = context.reader.read_section(prefix + "runtime.json");
        config_ = Json::parse(metadata.begin(), metadata.end());
        if (config_.at("precision") != "bf16")
            throw std::invalid_argument("Laya requires BF16 matrix precision");
        max_batch_ = config_.at("max_batch_size").get<std::size_t>();
        max_options_ = config_.at("max_options").get<std::size_t>();
        actions_ = config_.at("num_actions").get<std::size_t>();
        if (!max_batch_ || !max_options_ || !actions_)
            throw std::invalid_argument("invalid Laya bundle geometry");
        const auto tokenizer = context.reader.read_section(prefix + "tokenizer.json");
        tokenizer_ = CreateBpeTokenizer(tokenizer.data(), tokenizer.size(), false);
        const auto plan = context.reader.read_section(prefix + "model.plan");
        engine_ = context.backend.create_module(plan.data(), plan.size(), {});
        if (!engine_ || !engine_->ok())
            throw std::runtime_error("cannot load Laya TensorRT engine");
    }

    StructuredDecisionResult decide(const StructuredDecisionRequest& request) override {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!request.images.empty() || !request.videos.empty())
            throw std::invalid_argument(
                "Laya accepts text and JSON state, not image or video tensors");
        if (request.max_state_tokens != -1)
            throw std::invalid_argument(
                "Laya uses its total sequence and question-head token budgets");
        const auto document = Json::parse(request.document);
        const auto rows = encode_record(*tokenizer_, document, config_);
        StructuredDecisionResult result;
        Json answers = Json::object();
        for (std::size_t begin = 0; begin < rows.size(); begin += max_batch_) {
            const auto count = std::min(max_batch_, rows.size() - begin);
            std::size_t sequence = 1, options = 1;
            for (std::size_t i = begin; i < begin + count; ++i) {
                sequence = std::max(sequence, rows[i].tokens.size());
                options = std::max(options, rows[i].options.size());
            }
            if (options > max_options_)
                throw std::invalid_argument("question exceeds bundle option capacity");
            std::vector<std::int32_t> ids(count * sequence,
                                          config_.at("pad_token_id").get<std::int32_t>());
            std::vector<std::int32_t> attention(count * sequence, 0), markers(count * options, 0),
                mask(count * options, 0);
            std::vector<std::int32_t> types(count), positions(sequence);
            std::iota(positions.begin(), positions.end(), 0);
            for (std::size_t i = 0; i < count; ++i) {
                const auto& row = rows[begin + i];
                std::copy(row.tokens.begin(), row.tokens.end(), ids.begin() + i * sequence);
                std::fill_n(attention.begin() + i * sequence, row.tokens.size(), 1);
                std::copy(row.markers.begin(), row.markers.end(), markers.begin() + i * options);
                std::fill_n(mask.begin() + i * options, row.markers.size(), 1);
                types[i] = row.type;
                result.input_tokens += row.tokens.size();
            }
            const std::vector<std::int64_t> token_shape{static_cast<std::int64_t>(count),
                                                        static_cast<std::int64_t>(sequence)};
            const std::vector<std::int64_t> option_shape{static_cast<std::int64_t>(count),
                                                         static_cast<std::int64_t>(options)};
            const auto output = engine_->forward({
                {"input_ids", {ids.data(), token_shape, DType::kInt32}},
                {"attention_mask", {attention.data(), token_shape, DType::kInt32}},
                {"marker_pos", {markers.data(), option_shape, DType::kInt32}},
                {"marker_mask", {mask.data(), option_shape, DType::kInt32}},
                {"qtype", {types.data(), {static_cast<std::int64_t>(count)}, DType::kInt32}},
                {"position_ids",
                 {positions.data(), {static_cast<std::int64_t>(sequence)}, DType::kInt32}},
            });
            const auto& logits = output.at("logits");
            const auto& act = output.at("act_logits");
            if (logits.dtype != DType::kFloat32 || logits.numel() != count * options ||
                act.dtype != DType::kFloat32 || act.numel() != count * actions_)
                throw std::runtime_error("Laya engine returned incompatible decision tensors");
            const auto* values = static_cast<const float*>(logits.data);
            const auto* action_values = static_cast<const float*>(act.data);
            for (std::size_t i = 0; i < count; ++i) {
                const auto& row = rows[begin + i];
                DecisionScores score;
                score.question_id = row.id;
                score.option_ids = row.options;
                const auto tau =
                    static_cast<float>(temperature(config_, row.type, row.options.size()));
                for (std::size_t j = 0; j < row.options.size(); ++j)
                    score.logits.push_back(values[i * options + j] / tau);
                score.probabilities = softmax(score.logits.data(), score.logits.size());
                const auto action = softmax(action_values + i * actions_, actions_);
                answers[row.id] = format_answer(document["questions"][row.id], row.options,
                                                score.probabilities, action[0]);
                result.scores.push_back(std::move(score));
            }
        }
        result.document =
            Json{{"model", "laya-rl-agent"},
                 {"answers", answers},
                 {"usage", {{"input_tokens", result.input_tokens}, {"output_tokens", 0}}}}
                .dump();
        return result;
    }

  private:
    Json config_;
    std::unique_ptr<ITokenizer> tokenizer_;
    std::unique_ptr<ITrtModule> engine_;
    std::size_t max_batch_, max_options_, actions_;
    std::mutex mutex_;
};

class Router final : public IStructuredDecision {
  public:
    explicit Router(const FamilyContext& context)
        : routing_([&]() {
              const auto data = context.reader.read_section("router.json");
              return Json::parse(data.begin(), data.end());
          }()) {
        for (const auto& name : {"english", "multilingual", "typed-decisions"})
            models_[name] = std::make_unique<Pipeline>(context, std::string(name) + "/");
    }

    StructuredDecisionResult decide(const StructuredDecisionRequest& request) override {
        const auto record = Json::parse(request.document);
        const auto decision = routing_.route(record);
        auto result = models_.at(decision.at("model").get<std::string>())->decide(request);
        auto document = Json::parse(result.document);
        document["routing"] = decision;
        result.document = document.dump();
        return result;
    }

  private:
    Routing routing_;
    std::unordered_map<std::string, std::unique_ptr<Pipeline>> models_;
};
} // namespace
} // namespace trtmc::laya

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("laya does not support --kv-cache-size");
    if (context.reader.find_section("router.json"))
        return new trtmc::laya::Router(context);
    return new trtmc::laya::Pipeline(context);
}
