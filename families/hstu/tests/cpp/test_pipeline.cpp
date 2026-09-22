/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/pipeline.h"
#include "families/hstu/runtime/request.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void rejects(const std::function<void()>& action, const char* message) {
    bool rejected = false;
    try {
        action();
    } catch (const std::exception&) {
        rejected = true;
    }
    check(rejected, message);
}

class FakeModule final : public trtmc::ITrtModule {
  public:
    bool malformed{false};
    bool nonfinite{false};
    bool retrieval{false};
    bool dense_attention{false};
    bool forbidden_cache_input{false};
    bool transposed_attention{false};
    bool dynamic_key_profile{false};
    trtmc::DType attention_dtype{trtmc::DType::kFloat32};
    std::int64_t allocated_key_width{-1};
    std::int64_t mask_keys{0};
    std::int64_t declared_output_width{2};
    int calls{0};
    float scaling{0};
    std::int64_t batch{0};
    std::int64_t length{0};
    std::int64_t candidates{0};
    std::vector<std::int32_t> tokens, positions, times, dense_metadata;
    trtmc::DType metadata_dtype{trtmc::DType::kInt32};
    std::vector<std::int64_t> metadata_shape{5, -1, 8};
    std::vector<float> mask, embeddings, logits, items;
    std::vector<std::uint16_t> packed_mask;

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        const auto& token = inputs.at("token_ids");
        batch = token.shape.at(0);
        length = token.shape.at(1);
        candidates = retrieval ? inputs.at("candidate_token_ids").shape.at(1) : length;
        const auto logit_rows = length;
        const auto* ids = static_cast<const std::int32_t*>(token.data);
        tokens.assign(ids, ids + token.numel());
        for (const auto& name : {"position_ids", "time_ids"}) {
            const auto found = inputs.find(name);
            if (found != inputs.end()) {
                const auto* data = static_cast<const std::int32_t*>(found->second.data);
                auto& target = std::string(name) == "position_ids" ? positions : times;
                target.assign(data, data + found->second.numel());
            }
        }
        if (dense_attention) {
            const auto& metadata = inputs.at("attention_metadata");
            check(metadata.dtype == trtmc::DType::kInt32 &&
                      metadata.shape == std::vector<std::int64_t>({5, batch + 1, 8}),
                  "dense metadata input shape and dtype");
            check(inputs.count("attention_mask") == 0 &&
                      inputs.count("attention_weights_transposed") == 0 &&
                      inputs.count("scaling_seqlen") == 0,
                  "dense path has no unused dense mask or second scaling input");
            const auto* data = static_cast<const std::int32_t*>(metadata.data);
            dense_metadata.assign(data, data + metadata.numel());
        } else {
            const auto& attention =
                inputs.at(transposed_attention ? "attention_weights_transposed" : "attention_mask");
            mask_keys = attention.shape.at(transposed_attention ? 2 : 3);
            const auto mask_shape = transposed_attention
                                        ? std::vector<std::int64_t>{batch, 1, mask_keys, length}
                                        : std::vector<std::int64_t>{batch, 1, length, mask_keys};
            check(attention.shape == mask_shape, "mask shape");
            check(attention.dtype == attention_dtype, "mask dtype matches engine contract");
            if (attention_dtype == trtmc::DType::kFloat32) {
                const auto* data = static_cast<const float*>(attention.data);
                mask.assign(data, data + attention.numel());
            } else {
                const auto* data = static_cast<const std::uint16_t*>(attention.data);
                packed_mask.assign(data, data + attention.numel());
            }
            if (transposed_attention) {
                check(inputs.count("scaling_seqlen") == 0,
                      "prepared weights are scaled exactly once");
            } else {
                check(mask_keys == length, "legacy mask shape");
                scaling = *static_cast<const float*>(inputs.at("scaling_seqlen").data);
            }
        }
        embeddings.resize(batch * length * 2);
        logits.resize(batch * logit_rows * 2);
        items.resize(batch * candidates * 2);
        for (std::int64_t index = 0; index < batch * length; ++index) {
            embeddings[index * 2] =
                retrieval ? static_cast<float>(index % 2 == 0) : static_cast<float>(index);
            embeddings[index * 2 + 1] =
                retrieval ? static_cast<float>(index % 2 != 0) : static_cast<float>(index) + 0.5F;
        }
        for (std::int64_t index = 0; index < batch * logit_rows; ++index) {
            const auto source = index;
            logits[index * 2] = static_cast<float>(source) * 10;
            logits[index * 2 + 1] = static_cast<float>(source) * 10 + 1;
        }
        for (std::int64_t index = 0; index < batch * candidates; ++index) {
            items[index * 2] = index % 2 == 0 ? 1.0F : 0.0F;
            items[index * 2 + 1] = index % 2 == 0 ? 0.0F : 1.0F;
        }
        if (nonfinite)
            embeddings.back() = std::numeric_limits<float>::quiet_NaN();
        return {
            {"embeddings",
             {embeddings.data(), {batch, length, malformed ? 3 : 2}, trtmc::DType::kFloat32}},
            {"logits", {logits.data(), {batch, logit_rows, 2}, trtmc::DType::kFloat32}},
            {"item_embeddings", {items.data(), {batch, candidates, 2}, trtmc::DType::kFloat32}},
        };
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override {
        if (name == "attention_metadata")
            return dense_attention;
        if (name == "cache_0_pages")
            return forbidden_cache_input;
        if (dense_attention && (name == "attention_mask" ||
                                name == "attention_weights_transposed" || name == "scaling_seqlen"))
            return false;
        if (name == "attention_weights_transposed")
            return transposed_attention;
        if (name == "attention_mask" || name == "scaling_seqlen")
            return !transposed_attention;
        return name == "token_ids" || name == "candidate_token_ids" || name == "position_ids" ||
               name == "time_ids";
    }
    bool has_output(const std::string& name) const override {
        return name == "embeddings" || name == "logits" || name == "item_embeddings";
    }
    trtmc::DType tensor_dtype(const std::string& name) const override {
        if (name == "attention_metadata")
            return metadata_dtype;
        if (dense_attention && (name == "attention_mask" || name == "attention_weights_transposed"))
            throw std::runtime_error("Dense engine must never be queried for mask allocation");
        if (name == "attention_weights_transposed" || name == "attention_mask")
            return attention_dtype;
        return name == "token_ids" || name == "candidate_token_ids" || name == "position_ids" ||
                       name == "time_ids"
                   ? trtmc::DType::kInt32
                   : trtmc::DType::kFloat32;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (name == "attention_metadata")
            return metadata_shape;
        if (name == "attention_weights_transposed")
            return {-1, 1, allocated_key_width, -1};
        if (name == "attention_mask")
            return {-1, 1, -1, allocated_key_width};
        if (name == "scaling_seqlen")
            return {1};
        return has_output(name) ? std::vector<int64_t>{-1, -1, declared_output_width}
                                : std::vector<int64_t>{-1, -1};
    }
    std::vector<int64_t> input_profile_shape(const std::string& name, int32_t,
                                             trtmc::ProfileShapeSelector selector) const override {
        if (name == "attention_weights_transposed")
            return {1, 1,
                    dynamic_key_profile && selector == trtmc::ProfileShapeSelector::kMin
                        ? 1
                        : allocated_key_width,
                    selector == trtmc::ProfileShapeSelector::kMin ? 1 : 16};
        if (dynamic_key_profile && name == "attention_mask")
            return {1, 1, 1,
                    selector == trtmc::ProfileShapeSelector::kMin ? 1 : allocated_key_width};
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string& name) const override {
        return static_cast<int32_t>(tensor_shape(name).size());
    }
    bool input_is_dynamic(const std::string&) const override { return true; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}
};

trtmc::hstu::RuntimeConfig configuration() {
    trtmc::hstu::RuntimeConfig config;
    config.hidden_size = 2;
    config.output_dim = 2;
    config.max_sequence_length = 16;
    config.max_batch_size = 2;
    config.position_buckets = 20;
    config.embedding_tables = {
        {"item", "item", 20, 0, {}},
        {"action", "action", 4, 20, {100, 200, 300, 400}},
        {"context", "context", 3, 24, {-10, 1000, 9000}},
    };
    return config;
}

trtmc::RecommendationRequest request() {
    return {{{{1, 2}, {100, 300}, {{"context", {-10, 9000}}}, {3, 4}, {}, {}},
             {{7}, {200}, {}, {8}, {}, {}}}};
}

void test_assembly() {
    auto module = std::make_unique<FakeModule>();
    auto* fake = module.get();
    trtmc::hstu::Pipeline pipeline(std::move(module), configuration());
    const auto result = pipeline.recommend(request());
    check(fake->tokens ==
              std::vector<std::int32_t>{24, 26, 1, 20, 2, 22, 3, 4, 7, 21, 8, 0, 0, 0, 0, 0},
          "token assembly and sparse IDs");
    check(fake->positions ==
              std::vector<std::int32_t>{0, 1, 2, 3, 4, 5, 6, 6, 0, 1, 2, 0, 0, 0, 0, 0},
          "position saturation");
    check(fake->scaling == 8.0F, "dynamic attention scaling");
    check(result.sequences[0].logits == std::vector<float>{60, 61, 70, 71},
          "ranking target slicing");
    check(result.sequences[1].logits == std::vector<float>{100, 101}, "batched target slicing");
    check(result.sequences[1].sequence_embeddings.size() == 6, "unpad sequence embeddings");
    check(fake->mask[0 * 8 + 5] == 1.0F, "context sees history");
    check(fake->mask[0 * 8 + 6] == 0.0F, "context cannot see candidates");
    check(fake->mask[3 * 8 + 4] == 0.0F, "causal history");
    check(fake->mask[7 * 8 + 6] == 0.0F, "candidate isolation");
    check(fake->mask[7 * 8 + 7] == 1.0F, "candidate self attention");
    for (std::size_t row = 0; row < 8; ++row) {
        for (std::size_t column = 0; column < 8; ++column) {
            if (row >= 3 || column >= 3)
                check(fake->mask[64 + row * 8 + column] == 0.0F,
                      "padding cannot attend or be attended");
        }
    }
}

void test_mask_modes_and_time() {
    auto config = configuration();
    config.target_group_size = 2;
    config.disable_contextual_mask = true;
    config.time_buckets = 2048;
    config.scaling_seqlen = 32;
    auto input = request();
    input.sequences.resize(1);
    input.sequences[0].token_timestamps = {0, 60, 120, 180, 240, 300, 360, 540};
    auto module = std::make_unique<FakeModule>();
    auto* fake = module.get();
    trtmc::hstu::Pipeline pipeline(std::move(module), config);
    (void)pipeline.recommend(input);
    check(fake->mask[5] == 0.0F, "disabled contextual attention");
    check(fake->mask[7 * 8 + 6] == 1.0F, "intra target group attention");
    check(fake->positions == std::vector<std::int32_t>{6, 5, 4, 3, 2, 1, 0, 0},
          "reverse time positions");
    check(fake->times == std::vector<std::int32_t>{3, 2, 2, 2, 2, 2, 1, 0}, "sqrt time buckets");
    check(fake->scaling == 32.0F, "fixed attention scaling");
    input.sequences[0].token_timestamps.front() = 540 - (60 * 2048 * 2048 - 1);
    (void)pipeline.recommend(input);
    check(fake->times.front() == 2048, "timestamp boundary uses reference FP32 rounding");
    config.time_buckets = 0;
    config.is_causal = false;
    input.sequences[0].token_timestamps.clear();
    auto noncausal = std::make_unique<FakeModule>();
    auto* noncausal_ptr = noncausal.get();
    trtmc::hstu::Pipeline bidirectional(std::move(noncausal), config);
    (void)bidirectional.recommend(input);
    check(noncausal_ptr->mask[2 * 8 + 5] == 1.0F, "noncausal history attention");
}

void test_retrieval_and_empty_targets() {
    auto config = configuration();
    config.mode = "retrieval";
    config.output_dim = 1;
    auto module = std::make_unique<FakeModule>();
    module->retrieval = true;
    trtmc::hstu::Pipeline pipeline(std::move(module), config);
    auto input = request();
    const auto result = pipeline.recommend(input);
    check(result.sequences[0].scores == std::vector<float>{1.0F, 0.0F},
          "retrieval queries last item rather than following action, with context prefix");
    check(result.sequences[0].logits.empty(), "retrieval has separate scores");
    input.sequences[0].candidate_item_ids.clear();
    const auto empty = pipeline.recommend(input);
    check(empty.sequences[0].scores.empty() && empty.sequences[0].embeddings.empty(),
          "empty candidates");
    check(empty.sequences[0].sequence_length == 6, "empty candidate history remains represented");
    input.sequences[0].history_item_ids.clear();
    input.sequences[0].history_action_ids.clear();
    rejects([&] { pipeline.recommend(input); },
            "retrieval context alone accepted without history items");
}

void test_rejections() {
    auto malformed_schema = std::make_unique<FakeModule>();
    malformed_schema->declared_output_width = 3;
    rejects([&] { trtmc::hstu::Pipeline invalid(std::move(malformed_schema), configuration()); },
            "incorrect engine output width accepted during initialization");
    auto module = std::make_unique<FakeModule>();
    auto* fake = module.get();
    trtmc::hstu::Pipeline pipeline(std::move(module), configuration());
    rejects([&] { pipeline.recommend({}); }, "empty batch accepted");
    auto input = request();
    input.sequences[0].history_action_ids[0] = 101;
    rejects([&] { pipeline.recommend(input); }, "unknown sparse ID accepted");
    input = request();
    input.sequences[0].history_action_ids.clear();
    rejects([&] { pipeline.recommend(input); }, "missing action accepted");
    input = request();
    input.sequences[0].candidate_item_ids[0] = 20;
    rejects([&] { pipeline.recommend(input); }, "out of range item accepted");
    input = request();
    input.sequences[0].contextual_features.push_back(
        input.sequences[0].contextual_features.front());
    rejects([&] { pipeline.recommend(input); }, "duplicate context accepted");
    input = request();
    input.sequences[0].candidate_item_ids.resize(20, 0);
    rejects([&] { pipeline.recommend(input); }, "oversize sequence accepted");
    check(fake->calls == 0, "invalid requests reached engine");
    fake->malformed = true;
    rejects([&] { pipeline.recommend(request()); }, "malformed engine output accepted");
    fake->malformed = false;
    fake->nonfinite = true;
    input = request();
    input.sequences.resize(1);
    rejects([&] { pipeline.recommend(input); }, "nonfinite engine output accepted");
}

void test_public_candidate_outputs() {
    auto old_module = std::make_unique<FakeModule>();
    trtmc::hstu::Pipeline old_pipeline(std::move(old_module), configuration());
    auto module = std::make_unique<FakeModule>();
    module->transposed_attention = true;
    auto* fake = module.get();
    trtmc::hstu::Pipeline prepared_pipeline(std::move(module), configuration());
    auto input = request();
    for (int probe = 0; probe < 3; ++probe) {
        if (probe == 1)
            input.sequences[0].candidate_item_ids.clear();
        if (probe == 2)
            input.sequences[1].candidate_item_ids.clear();
        const auto expected = old_pipeline.recommend(input);
        const auto actual = prepared_pipeline.recommend(input);
        for (std::size_t sample = 0; sample < input.sequences.size(); ++sample) {
            const auto& left = actual.sequences[sample];
            const auto& right = expected.sequences[sample];
            check(left.candidate_item_ids == input.sequences[sample].candidate_item_ids &&
                      left.logits == right.logits,
                  "prepared inputs preserve ordered candidate logits and empty targets");
            check(left.embeddings == right.embeddings &&
                      left.sequence_embeddings == right.sequence_embeddings &&
                      left.sequence_length == right.sequence_length,
                  "prepared inputs preserve full unpadded public embeddings");
        }
    }
    input = request();
    for (auto& sequence : input.sequences) {
        sequence.history_item_ids.clear();
        sequence.history_action_ids.clear();
        sequence.contextual_features.clear();
    }
    const auto empty_history = prepared_pipeline.recommend(input);
    check(empty_history.sequences[0].logits.size() == 4 &&
              empty_history.sequences[1].logits.size() == 2,
          "historyless users still receive all requested candidate logits");
    check(fake->mask_keys == 2 && fake->mask == std::vector<float>{0.5F, 0, 0, 0.5F, 0.5F, 0, 0, 0},
          "historyless mixed candidate counts preserve diagonal isolation and batch scaling");
}

void test_transposed_weight_precision() {
    for (const auto dtype :
         {trtmc::DType::kFloat32, trtmc::DType::kFloat16, trtmc::DType::kBFloat16}) {
        for (const auto divisor : {3, -1}) {
            auto config = configuration();
            config.scaling_seqlen = divisor;
            auto module = std::make_unique<FakeModule>();
            module->transposed_attention = true;
            module->attention_dtype = dtype;
            module->allocated_key_width = 16;
            module->dynamic_key_profile = true;
            auto* fake = module.get();
            trtmc::hstu::Pipeline pipeline(std::move(module), config);
            const auto result = pipeline.recommend(request());
            check(fake->mask_keys == 8,
                  "dynamic key width uses logical batch length, not allocation");
            check(result.sequences[0].logits == std::vector<float>{60, 61, 70, 71},
                  "prepared weights preserve public candidate slicing");
            const auto allowed = divisor == 3 ? 1.0F / 3.0F : 1.0F / 8.0F;
            if (dtype == trtmc::DType::kFloat32) {
                check(fake->mask[5 * 8] == allowed && fake->mask[6 * 8] == 0.0F,
                      "FP32 prepared weights retain scale and context visibility");
                for (std::size_t row = 0; row < 8; ++row)
                    for (std::size_t column = 0; column < 8; ++column)
                        if (row >= 3 || column >= 3)
                            check(fake->mask[64 + row * 8 + column] == 0.0F,
                                  "prepared FP32 padding remains zero");
            } else {
                // Independent IEEE round-to-nearest bit patterns for 1/3 and 1/8.
                const std::uint16_t bits = dtype == trtmc::DType::kFloat16
                                               ? (divisor == 3 ? 0x3555 : 0x3000)
                                               : (divisor == 3 ? 0x3EAB : 0x3E00);
                check(fake->packed_mask[5 * 8] == bits && fake->packed_mask[6 * 8] == 0,
                      "prepared weights use the required low-precision rounded value");
                for (std::size_t row = 0; row < 8; ++row)
                    for (std::size_t column = 0; column < 8; ++column)
                        if (row >= 3 || column >= 3)
                            check(fake->packed_mask[64 + row * 8 + column] == 0,
                                  "prepared low-precision padding remains zero");
            }
            auto empty_targets = request();
            for (auto& sequence : empty_targets.sequences)
                sequence.candidate_item_ids.clear();
            const auto empty = pipeline.recommend(empty_targets);
            check(fake->mask_keys == 6 && empty.sequences[0].logits.empty() &&
                      empty.sequences[1].logits.empty(),
                  "prepared weights support empty targets with mixed history lengths");
        }
    }
}

void test_attention_mask_query_offsets() {
    const auto config = configuration();
    const auto input = request();
    const auto first = trtmc::hstu::assemble(input.sequences[0], config);
    const auto second = trtmc::hstu::assemble(input.sequences[1], config);
    auto mask = trtmc::hstu::make_attention_mask(2, 2, 8, trtmc::DType::kBFloat16, 1.0F / 3.0F);
    trtmc::hstu::fill_attention_mask(mask, 0, first, config, 6);
    trtmc::hstu::fill_attention_mask(mask, 1, second, config, second.tokens.size());
    check(mask.packed[6] == 0x3EAB && mask.packed[7] == 0 && mask.packed[14] == 0 &&
              mask.packed[15] == 0x3EAB,
          "cached query offsets preserve candidate isolation and diagonal visibility");
    for (std::size_t index = 16; index < mask.packed.size(); ++index)
        check(mask.packed[index] == 0, "zero-query sample has no visible padded rows");
    auto empty = trtmc::hstu::make_attention_mask(1, 0, 8, trtmc::DType::kFloat16, 1.0F);
    trtmc::hstu::fill_attention_mask(empty, 0, first, config, first.tokens.size());
    check(empty.packed.empty(), "all-zero-query batch has an empty mask buffer");
    rejects([&] { trtmc::hstu::make_attention_mask(1, 1, 1, trtmc::DType::kInt32, 1.0F); },
            "integer attention weights accepted");
    FakeModule dynamic;
    dynamic.allocated_key_width = 16;
    check(trtmc::hstu::attention_key_width(dynamic, 8) == 16,
          "legacy fixed-capacity key dimension is preserved");
    dynamic.dynamic_key_profile = true;
    check(trtmc::hstu::attention_key_width(dynamic, 8) == 8,
          "profile exposes dynamic key width when allocated shape is positive");
}

void test_transposed_prepared_attention() {
    for (const auto dtype :
         {trtmc::DType::kFloat32, trtmc::DType::kFloat16, trtmc::DType::kBFloat16}) {
        auto module = std::make_unique<FakeModule>();
        module->transposed_attention = true;
        module->attention_dtype = dtype;
        module->allocated_key_width = 16;
        module->dynamic_key_profile = true;
        auto* fake = module.get();
        trtmc::hstu::Pipeline pipeline(std::move(module), configuration());
        const auto result = pipeline.recommend(request());
        check(result.sequences[0].logits == std::vector<float>{60, 61, 70, 71},
              "transposed weights preserve full-head candidate slicing");
        check(fake->mask_keys == 8, "transposed key width comes from profile axis two");
        const auto visible = [&](std::size_t index, bool allowed) {
            if (dtype == trtmc::DType::kFloat32)
                check(fake->mask.at(index) == (allowed ? 0.125F : 0.0F),
                      "transposed FP32 weights use full padded batch scaling");
            else {
                const std::uint16_t bits = dtype == trtmc::DType::kFloat16 ? 0x3000 : 0x3E00;
                check(fake->packed_mask.at(index) == (allowed ? bits : 0),
                      "transposed packed weights use correctly rounded default scale");
            }
        };
        // These unequal causal positions distinguish direct [K,Q] from [Q,K].
        visible(4 * 8 + 3, false);
        visible(3 * 8 + 4, true);
        visible(5 * 8, true);  // Context query zero can read the full history.
        visible(6 * 8, false); // Context must never read candidates.
        for (std::size_t key = 0; key < 8; ++key)
            for (std::size_t query = 0; query < 8; ++query)
                if (key >= 3 || query >= 3)
                    visible(64 + key * 8 + query, false);
    }
    FakeModule profile;
    profile.transposed_attention = true;
    profile.allocated_key_width = 16;
    check(trtmc::hstu::attention_key_width(profile, 8) == 16,
          "dynamic queries must not make a static transposed key dimension dynamic");
    profile.dynamic_key_profile = true;
    check(trtmc::hstu::attention_key_width(profile, 8) == 8,
          "transposed dynamic key profiles use axis two despite allocated dimensions");
    profile.dynamic_key_profile = false;
    profile.allocated_key_width = 4;
    rejects([&] { trtmc::hstu::attention_key_width(profile, 8); },
            "transposed static key axis smaller than history was accepted");
}

void test_transposed_cached_query_offsets() {
    const auto config = configuration();
    const auto input = request();
    const auto first = trtmc::hstu::assemble(input.sequences[0], config);
    const auto second = trtmc::hstu::assemble(input.sequences[1], config);
    auto mask =
        trtmc::hstu::make_attention_mask(2, 2, 8, trtmc::DType::kBFloat16, 1.0F / 8.0F, true);
    trtmc::hstu::fill_attention_mask(mask, 0, first, config, 6);
    trtmc::hstu::fill_attention_mask(mask, 1, second, config, 2);
    check(mask.tensor().shape == std::vector<std::int64_t>{2, 1, 8, 2},
          "cached transposed masks expose key-major nonsquare tensor shape");
    check(mask.packed[12] == 0x3E00 && mask.packed[13] == 0 && mask.packed[14] == 0 &&
              mask.packed[15] == 0x3E00,
          "absolute candidate positions retain diagonal isolation after prefix removal");
    for (std::size_t key = 0; key < 8; ++key) {
        check(mask.packed[16 + key * 2] == (key < 3 ? 0x3E00 : 0),
              "mixed cached histories retain visible keys and zero key padding");
        check(mask.packed[17 + key * 2] == 0, "mixed cached query padding stays zero");
    }
    auto empty = trtmc::hstu::make_attention_mask(1, 0, 8, trtmc::DType::kBFloat16, 0.125F, true);
    trtmc::hstu::fill_attention_mask(empty, 0, first, config, first.tokens.size());
    check(empty.packed.empty() && empty.tensor().shape == std::vector<std::int64_t>{1, 1, 8, 0},
          "zero-query transposed masks retain an empty query dimension");
}

void check_interval_mask(std::size_t length, std::size_t history, std::size_t contextual,
                         std::size_t first_query, const trtmc::hstu::RuntimeConfig& config,
                         trtmc::DType dtype, std::size_t query_padding, std::size_t key_padding) {
    trtmc::hstu::Sequence sequence;
    sequence.tokens.resize(length);
    sequence.history_end = static_cast<std::int32_t>(history);
    sequence.contextual_length = static_cast<std::int32_t>(contextual);
    sequence.candidates = static_cast<std::int32_t>(length - history);
    const auto rows = length - first_query + query_padding;
    const auto keys = length + key_padding;
    const auto scale = config.scaling_seqlen > 0 ? static_cast<float>(config.scaling_seqlen)
                                                 : static_cast<float>(std::max(length, keys + 1));
    auto mask = trtmc::hstu::make_attention_mask(2, rows, keys, dtype, 1.0F / scale, true);
    trtmc::hstu::fill_attention_mask(mask, 1, sequence, config, first_query);
    for (std::size_t sample = 0; sample < 2; ++sample) {
        for (std::size_t key = 0; key < keys; ++key) {
            for (std::size_t query = 0; query < rows; ++query) {
                const auto absolute_query = first_query + query;
                const bool allowed = sample == 1 && key < length && absolute_query < length &&
                                     trtmc::hstu::attention_allowed(
                                         static_cast<std::int32_t>(absolute_query),
                                         static_cast<std::int32_t>(key), sequence, config);
                const auto index = (sample * keys + key) * rows + query;
                const bool equal = dtype == trtmc::DType::kFloat32
                                       ? mask.floats[index] == (allowed ? mask.allowed_float : 0.0F)
                                       : mask.packed[index] == (allowed ? mask.allowed_packed : 0);
                if (!equal)
                    throw std::runtime_error(
                        "interval/scalar mask mismatch: length=" + std::to_string(length) +
                        " history=" + std::to_string(history) + " context=" +
                        std::to_string(contextual) + " first_query=" + std::to_string(first_query) +
                        " key=" + std::to_string(key) + " query=" + std::to_string(query) +
                        " group=" + std::to_string(config.target_group_size) +
                        " causal=" + std::to_string(config.is_causal) +
                        " disable_context=" + std::to_string(config.disable_contextual_mask));
            }
        }
    }
}

void check_interval_policies(std::size_t length, std::size_t history, std::size_t contextual,
                             std::size_t first_query) {
    auto config = configuration();
    for (const auto causal : {false, true}) {
        config.is_causal = causal;
        for (const auto disable_context : {false, true}) {
            config.disable_contextual_mask = disable_context;
            for (const auto group : {1, 2, 3, 4, std::numeric_limits<std::int32_t>::max()}) {
                config.target_group_size = group;
                config.scaling_seqlen = group == 1 ? -1 : 7;
                for (const auto dtype :
                     {trtmc::DType::kFloat32, trtmc::DType::kFloat16, trtmc::DType::kBFloat16})
                    check_interval_mask(length, history, contextual, first_query, config, dtype,
                                        (history + first_query) % 3, (contextual + group % 3) % 3);
            }
        }
    }
}

void test_interval_mask_exhaustive() {
    // All short valid histories, including context-only and candidate-only inputs,
    // all query suffixes, empty queries, groups larger than the sequence, and padding.
    for (std::size_t length = 0; length <= 6; ++length)
        for (std::size_t history = 0; history <= length; ++history)
            for (std::size_t contextual = 0; contextual <= history; ++contextual)
                for (std::size_t first_query = 0; first_query <= length; ++first_query)
                    check_interval_policies(length, history, contextual, first_query);
}

void test_interval_mask_randomized() {
    std::mt19937 random(0x48535455);
    auto config = configuration();
    for (int sample = 0; sample < 256; ++sample) {
        const auto length = std::size_t{1} + random() % 256;
        const auto history = random() % (length + 1);
        const auto contextual = random() % (history + 1);
        const auto first_query = random() % (length + 1);
        config.is_causal = random() % 2;
        config.disable_contextual_mask = random() % 2;
        config.target_group_size = sample % 8 == 0 ? std::numeric_limits<std::int32_t>::max()
                                                   : 1 + random() % (length + 8);
        config.scaling_seqlen = sample % 2 == 0 ? -1 : 1024;
        for (const auto dtype :
             {trtmc::DType::kFloat32, trtmc::DType::kFloat16, trtmc::DType::kBFloat16})
            check_interval_mask(length, history, contextual, first_query, config, dtype,
                                random() % 5, random() % 7);
    }
}

void test_finite_classification() {
    const auto scalar = [](const std::vector<float>& values) {
        return std::all_of(values.begin(), values.end(),
                           [](float value) { return std::isfinite(value); });
    };
    const auto store = [](float& value, std::uint32_t bits) {
        std::memcpy(&value, &bits, sizeof(bits));
    };
    const std::vector<std::pair<std::uint32_t, bool>> edges = {
        {0x00000000, true},  {0x80000000, true},  {0x00000001, true},  {0x80000001, true},
        {0x007FFFFF, true},  {0x807FFFFF, true},  {0x00800000, true},  {0x80800000, true},
        {0x7F7FFFFF, true},  {0xFF7FFFFF, true},  {0x7F800000, false}, {0xFF800000, false},
        {0x7FC00000, false}, {0xFFC00000, false}, {0x7F800001, false}, {0xFF800001, false},
        {0x7FFFFFFF, false}, {0xFFFFFFFF, false},
    };
    check(trtmc::hstu::all_finite({}), "empty outputs must remain finite");
    for (const auto size : {1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65}) {
        std::vector<float> values(size, 0.25F);
        for (std::size_t position = 0; position < values.size(); ++position) {
            for (const auto& [bits, expected] : edges) {
                store(values[position], bits);
                check(trtmc::hstu::all_finite(values) == expected,
                      "IEEE finite classification or vector tail changed");
            }
            values[position] = 0.25F;
        }
    }
    std::mt19937 random(20260916);
    std::vector<float> single(1);
    for (int index = 0; index < 65536; ++index) {
        store(single.front(), random());
        check(trtmc::hstu::all_finite(single) == scalar(single),
              "seeded raw float bits differ from scalar finite predicate");
    }
    for (int trial = 0; trial < 256; ++trial) {
        std::vector<float> values(random() % 1025);
        for (auto& value : values)
            store(value, random());
        check(trtmc::hstu::all_finite(values) == scalar(values),
              "seeded output vector differs from scalar finite predicate");
    }
    for (const auto& bits : std::vector<std::vector<std::uint32_t>>{{0x7F800001},
                                                                    {0x7F800000, 0x7F800001},
                                                                    {0x7F800001, 0x7F800000},
                                                                    {0x7FC00000, 0x7F800001},
                                                                    {0x00000001, 0x7F800001}}) {
        std::vector<float> values(bits.size());
        for (std::size_t index = 0; index < bits.size(); ++index)
            store(values[index], bits[index]);
        std::feclearexcept(FE_ALL_EXCEPT);
        const auto expected = scalar(values);
        const auto flags = std::fetestexcept(FE_ALL_EXCEPT);
        std::feclearexcept(FE_ALL_EXCEPT);
        check(trtmc::hstu::all_finite(values) == expected, "first nonfinite result changed");
        check(std::fetestexcept(FE_ALL_EXCEPT) == flags,
              "signaling NaN or first-error floating exception flags changed");
    }
    std::feclearexcept(FE_ALL_EXCEPT);
}

void test_runtime_config() {
    const std::string source = R"({"schema_version":1,"mode":"ranking","hidden_size":2,
      "max_sequence_length":16,"max_batch_size":2,"position_buckets":20,"time_buckets":0,
      "target_group_size":1,"scaling_seqlen":-1,"is_causal":true,"disable_contextual_mask":false,
      "prediction_head":[4,2],"embedding_tables":[{"name":"item","role":"item",
      "num_embeddings":2,"offset":0,"keys_offset":0}]})";
    const std::vector<char> data(source.begin(), source.end());
    std::vector<char> keys(16, 0);
    keys[0] = 10;
    keys[8] = 20;
    const auto config = trtmc::hstu::parse_runtime_config(data, keys);
    check(config.embedding_tables[0].keys == std::vector<std::int64_t>{10, 20},
          "sparse key decoding");
    check(config.output_dim == 2, "prediction head output dimension");
    check(!config.enable_history_cache, "old bundles must default to no history cache");
    auto invalid_cache = source;
    invalid_cache.insert(1, "\"enable_history_cache\":1,");
    rejects(
        [&] {
            trtmc::hstu::parse_runtime_config(
                std::vector<char>(invalid_cache.begin(), invalid_cache.end()), keys);
        },
        "nonboolean cache flag accepted");
    auto cached_source = source;
    cached_source.insert(1, "\"enable_history_cache\":true,\"num_layers\":2,\"num_heads\":1,"
                            "\"head_dim\":2,\"cache_artifact_id\":\"artifact-one\",");
    const auto cached_config = trtmc::hstu::parse_runtime_config(
        std::vector<char>(cached_source.begin(), cached_source.end()), keys);
    check(cached_config.enable_history_cache && cached_config.num_layers == 2 &&
              cached_config.cache_artifact_id == "artifact-one",
          "native cache contract parsing");
    auto zero_scaling = source;
    const std::string scale_field = "\"scaling_seqlen\":-1";
    zero_scaling.replace(zero_scaling.find(scale_field), scale_field.size(),
                         "\"scaling_seqlen\":0");
    const std::vector<char> invalid_scaling(zero_scaling.begin(), zero_scaling.end());
    rejects([&] { trtmc::hstu::parse_runtime_config(invalid_scaling, keys); },
            "zero attention scaling accepted");
    rejects([&] { trtmc::hstu::parse_runtime_config(data, {}); }, "missing sparse keys accepted");
    keys[8] = 10;
    rejects([&] { trtmc::hstu::parse_runtime_config(data, keys); },
            "duplicate sparse keys accepted");
}

void test_runtime_config_requires_item_table() {
    for (const bool cached : {false, true}) {
        for (const auto* mode : {"ranking", "retrieval"}) {
            auto parse = [&](const std::string& tables, int group_size = 1) {
                const std::string source =
                    R"({"schema_version":1,"hidden_size":2,"max_sequence_length":16,)"
                    R"("max_batch_size":2,"position_buckets":0,"time_buckets":0,)"
                    R"("scaling_seqlen":16,"is_causal":true,"disable_contextual_mask":false,)"
                    R"("prediction_head":[2],"num_layers":2,"num_heads":1,"head_dim":2,)"
                    R"("cache_artifact_id":"item-table-test","mode":")" +
                    std::string(mode) + R"(","enable_history_cache":)" +
                    (cached ? "true" : "false") + R"(,"target_group_size":)" +
                    std::to_string(group_size) + R"(,"embedding_tables":)" + tables + "}";
                return trtmc::hstu::parse_runtime_config({source.begin(), source.end()}, {});
            };
            const std::string valid =
                R"([{"name":"products","role":"item","num_embeddings":2,"offset":0}])";
            const auto config = parse(valid);
            check(config.enable_history_cache == cached && config.mode == mode,
                  "valid item role must be accepted for both cache modes and tasks");
            check(trtmc::hstu::find_role(config, "item")->name == "products",
                  "item table admission must use role rather than table name");
            for (const auto* invalid :
                 {"[]", R"([{"name":"actions","role":"action","num_embeddings":2,"offset":0}])",
                  R"([{"name":"context","role":"context","num_embeddings":2,"offset":0}])",
                  R"([{"name":"item","role":"context","num_embeddings":2,"offset":0}])"}) {
                bool rejected = false;
                try {
                    (void)parse(invalid);
                } catch (const std::invalid_argument& error) {
                    check(std::string(error.what()) == "hstu requires one item embedding table",
                          "missing item role must fail through configuration validation");
                    rejected = true;
                }
                check(rejected, "runtime configuration without an item role was accepted");
            }
            rejects([&] { (void)parse(valid, 0); }, "zero target group size must remain rejected");
        }
    }
}

trtmc::hstu::RuntimeConfig dense_configuration() {
    auto config = configuration();
    config.embedding_tables.pop_back();
    config.scaling_seqlen = 1024;
    config.enable_history_cache = false;
    return config;
}

trtmc::RecommendationRequest dense_request() {
    auto result = request();
    for (auto& sequence : result.sequences)
        sequence.contextual_features.clear();
    return result;
}

void test_dense_metadata_contract() {
    const auto config = dense_configuration();
    for (const std::size_t batch : {1U, 2U, 4U, 8U}) {
        std::vector<trtmc::hstu::Sequence> sequences(batch);
        for (std::size_t user = 0; user < batch; ++user) {
            sequences[user].history_end = static_cast<std::int32_t>(user % 4);
            sequences[user].candidates = 1;
            sequences[user].tokens.resize(user % 4 + 1);
        }
        const auto metadata = trtmc::hstu::make_dense_attention_metadata(sequences, 5);
        const auto stride = 8 * (batch + 1);
        check(metadata.size() == 5 * stride &&
                  metadata[batch] == static_cast<std::int32_t>(batch * 5) &&
                  metadata[stride + batch] == static_cast<std::int32_t>(batch * 5),
              "one generic metadata layout covers every admitted batch size");
    }
    for (int first_history = 0; first_history <= 4; ++first_history)
        for (int second_history = 0; second_history <= 4; ++second_history)
            for (int first_candidates = 0; first_candidates <= 3; ++first_candidates)
                for (int second_candidates = 0; second_candidates <= 3; ++second_candidates) {
                    if (first_history + first_candidates == 0 ||
                        second_history + second_candidates == 0)
                        continue;
                    std::vector<trtmc::hstu::Sequence> sequences(2);
                    sequences[0].history_end = first_history;
                    sequences[0].candidates = first_candidates;
                    sequences[0].tokens.resize(first_history + first_candidates);
                    sequences[1].history_end = second_history;
                    sequences[1].candidates = second_candidates;
                    sequences[1].tokens.resize(second_history + second_candidates);
                    const auto width =
                        std::max(sequences[0].tokens.size(), sequences[1].tokens.size());
                    const auto metadata =
                        trtmc::hstu::make_dense_attention_metadata(sequences, width);
                    const std::size_t stride = 8 * (sequences.size() + 1);
                    check(metadata.size() == 5 * stride, "dense transport has five padded planes");
                    for (std::size_t user = 0; user <= sequences.size(); ++user)
                        check(metadata[user] == static_cast<std::int32_t>(user * width) &&
                                  metadata[stride + user] ==
                                      static_cast<std::int32_t>(user * width),
                              "dense query/key offsets include the same physical padded rows");
                    for (std::size_t user = 0; user < sequences.size(); ++user) {
                        const auto history =
                            static_cast<std::int32_t>(width) - metadata[2 * stride + user];
                        check(history == sequences[user].history_end,
                              "padding never becomes history");
                        for (std::int32_t row = 0;
                             row < static_cast<std::int32_t>(sequences[user].tokens.size()); ++row)
                            for (std::int32_t column = 0; column < static_cast<std::int32_t>(width);
                                 ++column) {
                                const bool dense_allowed =
                                    column <= row && (column < history || row == column);
                                const bool original_allowed =
                                    column <
                                        static_cast<std::int32_t>(sequences[user].tokens.size()) &&
                                    trtmc::hstu::attention_allowed(row, column, sequences[user],
                                                                   config);
                                check(dense_allowed == original_allowed,
                                      "every actual history/candidate row preserves the original "
                                      "mask and rejects padding");
                            }
                    }
                    for (std::size_t plane = 0; plane < 5; ++plane) {
                        const auto used = plane < 2 ? 3U : plane == 2 ? 2U : 0U;
                        check(std::all_of(metadata.begin() + plane * stride + used,
                                          metadata.begin() + (plane + 1) * stride,
                                          [](auto value) { return value == 0; }),
                              "unused metadata words and both page planes remain zero");
                    }
                }
    trtmc::hstu::Sequence valid;
    valid.tokens = {1};
    valid.candidates = 1;
    rejects([&] { trtmc::hstu::make_dense_attention_metadata({}, 1); },
            "empty metadata batch rejected");
    rejects([&] { trtmc::hstu::make_dense_attention_metadata({valid}, 0); },
            "zero physical width rejected");
    rejects(
        [&] {
            trtmc::hstu::make_dense_attention_metadata(
                {valid, valid}, static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()));
        },
        "query/key offset overflow rejected before allocation");
    rejects(
        [&] {
            trtmc::hstu::make_dense_attention_metadata({valid},
                                                       std::numeric_limits<std::size_t>::max());
        },
        "oversized physical width rejected without arithmetic wrap");
    auto bad = valid;
    bad.contextual_length = 1;
    rejects([&] { trtmc::hstu::make_dense_attention_metadata({bad}, 1); }, "context rows rejected");
    bad = valid;
    bad.history_end = -1;
    rejects([&] { trtmc::hstu::make_dense_attention_metadata({bad}, 1); },
            "negative history rejected");
    bad = valid;
    bad.history_end = 2;
    rejects([&] { trtmc::hstu::make_dense_attention_metadata({bad}, 2); },
            "history beyond sequence rejected");
    bad = valid;
    bad.candidates = 0;
    rejects([&] { trtmc::hstu::make_dense_attention_metadata({bad}, 1); },
            "candidate mismatch rejected");
    bad = valid;
    bad.tokens.push_back(2);
    bad.candidates = 2;
    rejects([&] { trtmc::hstu::make_dense_attention_metadata({bad}, 1); },
            "unpadded rows beyond width rejected");
    rejects([&] { trtmc::hstu::make_dense_attention_metadata({{}}, 1); },
            "empty actual user remains invalid");
}

void test_dense_pipeline_outputs_and_admission() {
    const auto config = dense_configuration();
    auto module = std::make_unique<FakeModule>();
    auto* fake = module.get();
    fake->dense_attention = true;
    trtmc::hstu::Pipeline pipeline(std::move(module), config);
    auto ordinary_module = std::make_unique<FakeModule>();
    auto* ordinary = ordinary_module.get();
    trtmc::hstu::Pipeline old_pipeline(std::move(ordinary_module), config);
    auto input = dense_request();
    for (int variation = 0; variation < 3; ++variation) {
        if (variation == 1) {
            for (auto& sequence : input.sequences)
                sequence.candidate_item_ids.clear();
        } else if (variation == 2) {
            input = dense_request();
            input.sequences[1].history_item_ids.clear();
            input.sequences[1].history_action_ids.clear();
        }
        const auto actual = pipeline.recommend(input);
        const auto expected = old_pipeline.recommend(input);
        check(fake->tokens == ordinary->tokens && fake->positions == ordinary->positions,
              "dense metadata preserves every actual/padded token and position ID");
        check(fake->mask.empty() && fake->packed_mask.empty(),
              "dense path constructs no unused mask");
        const auto stride = 8 * (input.sequences.size() + 1);
        for (std::size_t user = 0; user < input.sequences.size(); ++user) {
            const auto history = input.sequences[user].history_item_ids.size() +
                                 input.sequences[user].history_action_ids.size();
            check(
                fake->dense_metadata[user + 1] ==
                        static_cast<std::int32_t>((user + 1) * fake->length) &&
                    fake->dense_metadata[stride + user + 1] == fake->dense_metadata[user + 1] &&
                    fake->dense_metadata[2 * stride + user] ==
                        fake->length - static_cast<std::int64_t>(history),
                "Pipeline emits offsets and padded target counts from each actual public request");
        }
        for (std::size_t user = 0; user < actual.sequences.size(); ++user) {
            const auto& a = actual.sequences[user];
            const auto& e = expected.sequences[user];
            check(a.logits == e.logits && a.embeddings == e.embeddings &&
                      a.sequence_embeddings == e.sequence_embeddings &&
                      a.candidate_item_ids == e.candidate_item_ids &&
                      a.sequence_length == e.sequence_length &&
                      a.num_candidates == e.num_candidates,
                  "dense path returns only actual user rows and preserves candidate order/empty "
                  "targets");
            check(a.cache.source == e.cache.source && a.cache.reason == e.cache.reason &&
                      a.cache.reused_history_tokens == 0 && !a.cache.published,
                  "ordinary cache-disabled report defaults remain unchanged");
        }
    }
    trtmc::IRecommendation* public_task = &pipeline;
    check(dynamic_cast<trtmc::IRecommendationSessionFactory*>(public_task) == nullptr,
          "ordinary dense Pipeline does not expose persistent session state");
    const auto prior_calls = fake->calls;
    input.sequences[1].candidate_item_ids.clear();
    rejects([&] { pipeline.recommend(input); }, "completely empty user remains rejected");
    check(fake->calls == prior_calls, "invalid empty user never reaches engine");
    auto reject_config = [&](trtmc::hstu::RuntimeConfig wrong) {
        rejects(
            [&] {
                auto engine = std::make_unique<FakeModule>();
                engine->dense_attention = true;
                trtmc::hstu::Pipeline invalid(std::move(engine), wrong);
            },
            "unsupported dense metadata semantics rejected");
    };
    auto wrong = config;
    wrong.enable_history_cache = true;
    reject_config(wrong);
    wrong = config;
    wrong.is_causal = false;
    reject_config(wrong);
    wrong = config;
    wrong.target_group_size = 2;
    reject_config(wrong);
    wrong = config;
    wrong.scaling_seqlen = -1;
    reject_config(wrong);
    wrong = config;
    wrong.time_buckets = 2048;
    reject_config(wrong);
    wrong = config;
    wrong.max_sequence_length = 1025;
    reject_config(wrong);
    wrong = config;
    wrong.mode = "retrieval";
    reject_config(wrong);
    wrong = config;
    wrong.embedding_tables.push_back({"context", "context", 1, 24, {}});
    reject_config(wrong);
    rejects(
        [&] {
            auto engine = std::make_unique<FakeModule>();
            engine->dense_attention = true;
            engine->metadata_shape[0] = 4;
            trtmc::hstu::Pipeline invalid(std::move(engine), config);
        },
        "incorrect metadata plane count rejected");
    rejects(
        [&] {
            auto engine = std::make_unique<FakeModule>();
            engine->dense_attention = true;
            engine->metadata_dtype = trtmc::DType::kFloat32;
            trtmc::hstu::Pipeline invalid(std::move(engine), config);
        },
        "incorrect metadata dtype rejected");
    rejects(
        [&] {
            auto engine = std::make_unique<FakeModule>();
            engine->dense_attention = true;
            engine->forbidden_cache_input = true;
            trtmc::hstu::Pipeline invalid(std::move(engine), config);
        },
        "paged state cannot enter ordinary dense Pipeline");
}
} // namespace

int main() {
    try {
        test_assembly();
        test_mask_modes_and_time();
        test_retrieval_and_empty_targets();
        test_rejections();
        test_public_candidate_outputs();
        test_transposed_weight_precision();
        test_attention_mask_query_offsets();
        test_transposed_prepared_attention();
        test_transposed_cached_query_offsets();
        test_interval_mask_exhaustive();
        test_interval_mask_randomized();
        test_finite_classification();
        test_runtime_config();
        test_runtime_config_requires_item_table();
        test_dense_metadata_contract();
        test_dense_pipeline_outputs_and_admission();
        std::cout << "HSTU runtime contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
