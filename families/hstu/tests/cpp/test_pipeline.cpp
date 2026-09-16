/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/pipeline.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
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
    std::int64_t declared_output_width{2};
    int calls{0};
    float scaling{0};
    std::int64_t batch{0};
    std::int64_t length{0};
    std::int64_t candidates{0};
    std::vector<std::int32_t> tokens, positions, times;
    std::vector<float> mask, embeddings, logits, items;

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        const auto& token = inputs.at("token_ids");
        batch = token.shape.at(0);
        length = token.shape.at(1);
        candidates = retrieval ? inputs.at("candidate_token_ids").shape.at(1) : length;
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
        const auto& attention = inputs.at("attention_mask");
        check(attention.shape == std::vector<std::int64_t>{batch, 1, length, length}, "mask shape");
        const auto* attention_data = static_cast<const float*>(attention.data);
        mask.assign(attention_data, attention_data + attention.numel());
        scaling = *static_cast<const float*>(inputs.at("scaling_seqlen").data);
        embeddings.resize(batch * length * 2);
        logits.resize(batch * length * 2);
        items.resize(batch * candidates * 2);
        for (std::int64_t index = 0; index < batch * length; ++index) {
            embeddings[index * 2] =
                retrieval ? static_cast<float>(index % 2 == 0) : static_cast<float>(index);
            embeddings[index * 2 + 1] =
                retrieval ? static_cast<float>(index % 2 != 0) : static_cast<float>(index) + 0.5F;
            logits[index * 2] = static_cast<float>(index) * 10;
            logits[index * 2 + 1] = static_cast<float>(index) * 10 + 1;
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
            {"logits", {logits.data(), {batch, length, 2}, trtmc::DType::kFloat32}},
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
        return name == "token_ids" || name == "candidate_token_ids" || name == "position_ids" ||
               name == "time_ids" || name == "attention_mask" || name == "scaling_seqlen";
    }
    bool has_output(const std::string& name) const override {
        return name == "embeddings" || name == "logits" || name == "item_embeddings";
    }
    trtmc::DType tensor_dtype(const std::string& name) const override {
        return name == "token_ids" || name == "candidate_token_ids" || name == "position_ids" ||
                       name == "time_ids"
                   ? trtmc::DType::kInt32
                   : trtmc::DType::kFloat32;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (name == "attention_mask")
            return {-1, 1, -1, -1};
        if (name == "scaling_seqlen")
            return {1};
        return has_output(name) ? std::vector<int64_t>{-1, -1, declared_output_width}
                                : std::vector<int64_t>{-1, -1};
    }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
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

} // namespace

int main() {
    try {
        test_assembly();
        test_mask_modes_and_time();
        test_retrieval_and_empty_targets();
        test_rejections();
        test_runtime_config();
        std::cout << "HSTU runtime contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
