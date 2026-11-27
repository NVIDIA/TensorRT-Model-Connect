/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// One HSTU ranking request per decide(): the record's ids go to the
// embedding engine, whose outputs ARE the HSTU block engine's inputs, whose
// candidate embeddings ARE the ranking head's input (bound in place, one CUDA
// stream, no host round trip); the logits come back once per request.

#include "families/hstu/runtime/record.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::hstu {
namespace {

void cuda_check(cudaError_t code, const char* what) {
    if (code != cudaSuccess)
        throw std::runtime_error(std::string("HSTU ") + what + ": " + cudaGetErrorString(code));
}

enum class Direction { kInput, kOutput };

void require_tensor(const ITrtModule& module, const std::string& section, const std::string& name,
                    Direction direction, DType dtype, const std::vector<int64_t>& shape) {
    const bool input = direction == Direction::kInput;
    if (!(input ? module.has_input(name) : module.has_output(name)))
        throw std::runtime_error("HSTU " + section + " has no " + (input ? "input" : "output") +
                                 " '" + name + "'");
    if (module.tensor_dtype(name) != dtype || module.tensor_shape(name) != shape)
        throw std::runtime_error("HSTU " + section + " tensor '" + name +
                                 "' does not match the bundle's geometry");
}

// The hidden width the embedding engine writes, read only from a rank-2 output.
int64_t hidden_width(const ITrtModule& embedding) {
    if (!embedding.has_output("item_embeddings"))
        throw std::runtime_error("HSTU embedding.plan has no output 'item_embeddings'");
    const auto shape = embedding.tensor_shape("item_embeddings");
    if (shape.size() != 2 || shape.back() < 1)
        throw std::runtime_error("HSTU embedding.plan item_embeddings is not [tokens, hidden]");
    return shape.back();
}

class Pipeline final : public IStructuredDecision {
  public:
    explicit Pipeline(const FamilyContext& context) {
        const auto runtime = context.reader.read_section("runtime.json");
        geometry_ = parse_geometry(Json::parse(runtime.begin(), runtime.end()));
        embedding_ = load(context, "embedding.plan");
        hstu_ = load(context, "hstu.plan");
        head_ = load(context, "ranking_head.plan");

        const auto h = geometry_.history_length, c = geometry_.num_candidates;
        const auto contextual = static_cast<int64_t>(geometry_.contextual.size());
        constexpr auto in = Direction::kInput;
        constexpr auto out = Direction::kOutput;
        require_tensor(*embedding_, "embedding.plan", "contextual_ids", in, DType::kInt32,
                       {contextual});
        require_tensor(*embedding_, "embedding.plan", "history_item_ids", in, DType::kInt32, {h});
        require_tensor(*embedding_, "embedding.plan", "history_action_ids", in, DType::kInt32, {h});
        require_tensor(*embedding_, "embedding.plan", "candidate_item_ids", in, DType::kInt32, {c});
        // The embeddings between engines stay on the device in the work dtype.
        const auto work = geometry_.precision == "bf16" ? DType::kBFloat16 : DType::kFloat32;
        const auto hidden = hidden_width(*embedding_);
        const std::vector<std::pair<std::string, std::vector<int64_t>>> handoff{
            {"contextual_embeddings", {contextual, hidden}},
            {"item_embeddings", {h + c, hidden}},
            {"action_embeddings", {h, hidden}}};
        for (const auto& [name, shape] : handoff) {
            require_tensor(*embedding_, "embedding.plan", name, out, work, shape);
            require_tensor(*hstu_, "hstu.plan", name, in, work, shape);
            hstu_->bind_external(name, embedding_->device_ptr(name));
        }
        require_tensor(*hstu_, "hstu.plan", "candidate_embeddings", out, work, {c, hidden});
        require_tensor(*head_, "ranking_head.plan", "candidate_embeddings", in, work, {c, hidden});
        head_->bind_external("candidate_embeddings", hstu_->device_ptr("candidate_embeddings"));
        require_tensor(*head_, "ranking_head.plan", "logits", out, DType::kFloat32,
                       {c, geometry_.num_tasks});
        logits_.resize(static_cast<std::size_t>(c * geometry_.num_tasks));
    }

    StructuredDecisionResult decide(const StructuredDecisionRequest& request) override {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!request.images.empty() || !request.videos.empty())
            throw std::invalid_argument("an HSTU ranking request takes no media");
        if (request.max_state_tokens != -1)
            throw std::invalid_argument(
                "an HSTU ranking request takes no max_state_tokens: its geometry is the bundle's");
        auto ids = parse_request(request.document, geometry_);
        auto host = [](std::vector<std::int32_t>& values) {
            return Tensor{values.data(), {static_cast<int64_t>(values.size())}, DType::kInt32};
        };
        embedding_->forward_async({{"contextual_ids", host(ids.contextual_ids)},
                                   {"history_item_ids", host(ids.history_item_ids)},
                                   {"history_action_ids", host(ids.history_action_ids)},
                                   {"candidate_item_ids", host(ids.candidate_item_ids)}});
        hstu_->forward_device_async({});
        head_->forward_device_async({});
        cuda_check(cudaMemcpyAsync(logits_.data(), head_->device_ptr("logits"),
                                   logits_.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                   head_->stream()),
                   "logits copy");
        cuda_check(cudaStreamSynchronize(head_->stream()), "request");
        return respond(ids);
    }

  private:
    std::unique_ptr<ITrtModule> load(const FamilyContext& context, const std::string& section) {
        const auto* entry = context.reader.find_section(section.c_str());
        if (entry == nullptr || entry->length == 0)
            throw std::runtime_error("HSTU bundle section is missing or empty: " + section);
        const auto plan = context.reader.read_section(section.c_str());
        ModuleCreateOptions options;
        options.stream = stream_;
        auto module = context.backend.create_module(plan.data(), plan.size(), options);
        if (!module || !module->ok())
            throw std::runtime_error("HSTU " + section + " failed to load");
        if (stream_ == nullptr)
            stream_ = module->stream();
        module->set_timing_label(section);
        return module;
    }

    StructuredDecisionResult respond(const Request& ids) const {
        if (!std::all_of(logits_.begin(), logits_.end(), [](float v) { return std::isfinite(v); }))
            throw std::runtime_error("HSTU logits are not finite");
        const auto candidates = static_cast<std::size_t>(geometry_.num_candidates);
        const auto tasks = static_cast<std::size_t>(geometry_.num_tasks);
        std::vector<std::string> option_ids;
        option_ids.reserve(candidates);
        for (const auto id : ids.candidate_item_ids)
            option_ids.push_back(std::to_string(id));
        StructuredDecisionResult result;
        Json answers = Json::object();
        for (std::size_t task = 0; task < tasks; ++task) {
            DecisionScores scores;
            scores.question_id = "task_" + std::to_string(task);
            scores.option_ids = option_ids;
            for (std::size_t candidate = 0; candidate < candidates; ++candidate) {
                const float logit = logits_[candidate * tasks + task];
                scores.logits.push_back(logit);
                scores.probabilities.push_back(1.0F / (1.0F + std::exp(-logit)));
            }
            std::vector<std::size_t> order(candidates);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                return scores.logits[a] > scores.logits[b];
            });
            Json ranked = Json::array();
            for (const auto index : order)
                ranked.push_back(option_ids[index]);
            answers[scores.question_id] = std::move(ranked);
            result.scores.push_back(std::move(scores));
        }
        Json document{{"model", ids.model},
                      {"answers", std::move(answers)},
                      {"usage",
                       {{"history_items", geometry_.history_length},
                        {"candidates", geometry_.num_candidates},
                        {"tokens", geometry_.tokens}}}};
        result.document = document.dump();
        result.input_tokens = static_cast<std::int32_t>(geometry_.tokens);
        return result;
    }

    Geometry geometry_;
    cudaStream_t stream_{nullptr};
    // Declared in load order: the embedding module owns the shared stream and
    // is destroyed last.
    std::unique_ptr<ITrtModule> embedding_, hstu_, head_;
    std::vector<float> logits_;
    std::mutex mutex_;
};

} // namespace
} // namespace trtmc::hstu

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("hstu does not support --kv-cache-size");
    return new trtmc::hstu::Pipeline(context);
}
