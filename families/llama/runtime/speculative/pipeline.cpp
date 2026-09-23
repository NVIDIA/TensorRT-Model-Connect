/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/llama/runtime/speculative/pipeline.h"

#include "families/llama/runtime/chat_templates.h"
#include "families/llama/runtime/plugin_helpers.h"
#include "families/llama/runtime/speculative/eagle3.h"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <stdexcept>

namespace trtmc::llama::speculative {
namespace {
void checked(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
std::vector<std::int32_t> chain(int rows) {
    std::vector<std::int32_t> parents(rows);
    std::iota(parents.begin(), parents.end(), -1);
    return parents;
}

template <typename T>
std::vector<T> slice(const std::vector<T>& source, std::size_t start, std::size_t end) {
    if (start > end || end > source.size())
        throw std::out_of_range("speculative feature slice");
    return {source.begin() + start, source.begin() + end};
}

double elapsed(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}
} // namespace

Pipeline::Pipeline(const FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("speculative prototype uses the bundle's fixed KV capacity");
    manifest_ = nlohmann::json::parse(require_text_section(context.reader, "speculative.json"));
    if (manifest_.at("version") != 1 || manifest_.at("method") != "eagle3")
        throw std::invalid_argument("unsupported speculative bundle version or method");
    auto target = Contract::parse(manifest_.at("target"));
    auto draft = Contract::parse(manifest_.at("draft"));
    if (target.draft || !draft.draft || target.capacity != draft.capacity ||
        target.feature_width != draft.feature_width || target.hidden != draft.hidden)
        throw std::invalid_argument("incompatible target/draft contracts");
    depth_ = manifest_.at("draft_depth");
    if (depth_ < 1 || depth_ >= target.query_limit(Phase::kDecode) ||
        depth_ >= draft.query_limit(Phase::kDecode))
        throw std::invalid_argument("draft depth exceeds compiled query profile");
    mapping_ = manifest_.at("d2t").get<std::vector<std::int32_t>>();
    if (mapping_.size() != static_cast<std::size_t>(draft.vocab))
        throw std::invalid_argument("draft vocabulary mapping has the wrong size");
    for (auto token : mapping_) {
        if (token < 0 || token >= target.vocab)
            throw std::invalid_argument("draft vocabulary mapping is out of range");
    }
    const auto stop = manifest_.at("target_config").at("eos_token_id");
    eos_ = stop.is_array() ? stop.get<std::vector<std::int32_t>>()
                           : std::vector<std::int32_t>{stop.get<std::int32_t>()};
    if (manifest_.contains("stop_token_ids"))
        eos_ = manifest_.at("stop_token_ids").get<std::vector<std::int32_t>>();
    auto load = [&](const char* section, Contract contract, const std::string& label) {
        const auto plan = require_section(context.reader, section);
        std::unique_ptr<ITrtModule> selection;
        if (contract.device_selection) {
            const auto selection_plan = require_section(
                context.reader, contract.draft ? "draft_selection.plan" : "target_selection.plan");
            selection =
                load_engine(context.backend, selection_plan, (label + " selection").c_str());
        }
        if (contract.profiles.empty())
            return std::make_unique<Engine>(load_engine(context.backend, plan, label.c_str()),
                                            contract, nullptr, std::move(selection));
        auto dual = context.backend.create_dual_profile_modules(plan.data(), plan.size(), {});
        if (!dual.prefill || !dual.decode || !dual.prefill->ok() || !dual.decode->ok())
            throw std::runtime_error("could not create prefill/decode contexts for " + label);
        dual.prefill->set_timing_label(label + " prefill");
        dual.decode->set_timing_label(label + " decode");
        return std::make_unique<Engine>(std::move(dual.decode), contract, std::move(dual.prefill),
                                        std::move(selection));
    };
    target_ = load("target.plan", target, "llama target");
    draft_ = load("draft.plan", draft, "eagle3 draft");
    if (manifest_.contains("device_policy")) {
        const auto& policy = manifest_.at("device_policy");
        if (policy.at("version") != 1 || !target.device_selection || !draft.device_selection)
            throw std::invalid_argument("unsupported device policy contract");
        const auto mode = policy.at("default").get<std::string>();
        if (mode != "host" && mode != "device_draft" && mode != "device_full")
            throw std::invalid_argument("unsupported default execution policy");
        default_policy_ = mode == "host"           ? ExecutionPolicy::kHost
                          : mode == "device_draft" ? ExecutionPolicy::kDeviceDraft
                                                   : ExecutionPolicy::kDeviceFull;
        device_ = std::make_unique<DeviceEagle3>(context, *target_, *draft_, depth_);
    }
    prompt_features_ =
        DeviceTensor({target.capacity, target.feature_width}, DType::kFloat16, target_->stream());
    if (!prompt_features_.ok())
        throw std::runtime_error("could not allocate prompt conditioning storage");
    tokenizer_ = create_tokenizer(context.reader);
    if (context.reader.find_section("chat_template.jinja"))
        template_format_ = llama_detect_chat_template_format(
            require_text_section(context.reader, "chat_template.jinja"));
}

TextResult Pipeline::generate(const std::string& prompt, const TextGenerationConfig& config) {
    if ((config.temperature > 0 && config.top_k != 1) || config.repetition_penalty != 1.0F)
        throw std::invalid_argument("EAGLE3 prototype supports greedy unpenalized decoding only");
    const auto& mode = config.text_generation_mode;
    if (mode != "auto" && mode != "eagle3" && mode != "autoregressive")
        throw std::invalid_argument("expected auto, eagle3, or autoregressive generation mode");
    const auto text =
        config.use_chat_template && !template_format_.empty()
            ? llama_apply_chat_template(template_format_, prompt, config.enable_thinking)
            : prompt;
    auto ids = tokenizer_->encode(text);
    const int bos = manifest_.at("target_config").at("bos_token_id");
    if (ids.size() > 1 && ids[0] == bos && ids[1] == bos)
        ids.erase(ids.begin());
    return generate_ids(ids, config.max_new_tokens, mode != "autoregressive");
}

TextResult Pipeline::generate_ids(const std::vector<std::int32_t>& prompt, int count,
                                  bool speculative, bool ignore_eos, int draft_width,
                                  ExecutionPolicy policy) {
    if (policy == ExecutionPolicy::kDefault)
        policy = default_policy_;
    if (policy != ExecutionPolicy::kHost && policy != ExecutionPolicy::kDeviceDraft &&
        policy != ExecutionPolicy::kDeviceFull)
        throw std::invalid_argument("unsupported execution policy");
    const bool device = speculative && policy != ExecutionPolicy::kHost;
    if (device && !device_)
        throw std::invalid_argument("bundle has no device policy plans");
    const auto& tc = target_->contract();
    Eagle3 method(*draft_, mapping_, depth_, draft_width);
    if (draft_width < 1 || draft_width > 2 ||
        depth_ * draft_width + 1 > tc.query_limit(Phase::kDecode))
        throw std::invalid_argument(
            "prototype supports a chain or a two-child tree within the query profile");
    if (prompt.empty() || count < 0 || prompt.size() > static_cast<std::size_t>(tc.capacity) ||
        static_cast<std::size_t>(count) > static_cast<std::size_t>(tc.capacity) - prompt.size())
        throw std::invalid_argument("prompt and output exceed speculative cache capacity");
    for (auto id : prompt) {
        if (id < 0 || id >= tc.vocab)
            throw std::invalid_argument("prompt token is outside the target vocabulary");
    }
    accepted_lengths_.clear();
    TextResult result;
    if (count == 0)
        return result;
    target_->reset();
    draft_->reset();
    const auto prefill_start = std::chrono::steady_clock::now();
    StepResult target_result;
    for (int start = 0; start < static_cast<int>(prompt.size());) {
        const int rows =
            std::min(tc.query_limit(Phase::kPrefill), static_cast<int>(prompt.size()) - start);
        target_result = target_->run(Phase::kPrefill, slice(prompt, start, start + rows), start,
                                     chain(rows), false);
        if (speculative)
            checked(cudaMemcpyAsync(static_cast<std::uint16_t*>(prompt_features_.data()) +
                                        std::size_t(start) * tc.feature_width,
                                    target_result.features.data,
                                    std::size_t(rows) * tc.feature_width * 2,
                                    cudaMemcpyDeviceToDevice, target_->stream()));
        start += rows;
    }
    int root = target_result.token();
    auto emit = [&](int token) {
        if (static_cast<int>(result.token_ids.size()) >= count)
            return false;
        result.token_ids.push_back(token);
        return static_cast<int>(result.token_ids.size()) < count &&
               (ignore_eos || std::find(eos_.begin(), eos_.end(), token) == eos_.end());
    };
    bool more = emit(root);
    if (speculative) {
        // Prompt accumulation follows each target call on its own stream.
        // Complete it before a potentially different draft stream reads it.
        checked(cudaStreamSynchronize(target_->stream()));
        if (more)
            method.prefill(prompt, root,
                           {static_cast<const std::uint16_t*>(prompt_features_.data()),
                            static_cast<int>(prompt.size()), tc.feature_width});
    }
    result.prefill_ms = elapsed(prefill_start);
    const auto decode_start = std::chrono::steady_clock::now();
    if (device && more)
        device_->begin(method.features());
    int committed = static_cast<int>(prompt.size());
    while (more) {
        if (!speculative) {
            target_result = target_->run(Phase::kDecode, {root}, committed++, {-1}, false);
            root = target_result.token();
            more = emit(root);
            continue;
        }
        CandidateTree proposal;
        const auto remaining = count - static_cast<int>(result.token_ids.size());
        if (device) {
            device_->propose_and_verify(root, committed, remaining, draft_width, ignore_eos);
            if (policy == ExecutionPolicy::kDeviceFull) {
                const auto accepted = device_->accept(committed);
                accepted_lengths_.push_back(accepted.length - 1);
                result.token_ids.insert(result.token_ids.end(), accepted.emitted.begin(),
                                        accepted.emitted.end());
                more = !accepted.stopped;
                if (!more)
                    break;
                root = accepted.emitted.back();
                device_->feedback(accepted.length);
                committed += accepted.length;
                continue;
            }
            auto verified = device_->read_verification();
            proposal = std::move(verified.first);
            target_result = std::move(verified.second);
        } else {
            proposal = method.propose(root, committed, remaining);
            target_result =
                target_->run(Phase::kDecode, proposal.tokens, committed, proposal.parents, true);
        }
        const auto path = greedy_path(proposal.tokens, proposal.parents, target_result);
        accepted_lengths_.push_back(static_cast<int>(path.size()) - 1);
        if (draft_width > 1)
            target_->commit(committed, path);
        for (std::size_t index = 1; index < path.size() && more; ++index)
            more = emit(proposal.tokens[path[index]]);
        root = target_result.token(path.back());
        if (more)
            more = emit(root);
        if (!more)
            break;
        method.feedback(proposal, path, target_result.features, root, committed);
        if (device)
            device_->adopt(method.features());
        committed += static_cast<int>(path.size());
    }
    if (device)
        device_->finish();
    result.decode_ms = elapsed(decode_start);
    result.text = tokenizer_->decode(result.token_ids);
    return result;
}
} // namespace trtmc::llama::speculative
