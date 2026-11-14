/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/roberta/runtime/pipeline.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
using namespace trtmc;

class Tokenizer final : public ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string& text) const override {
        if (text == "short")
            return {0, 2};
        if (text == "limit")
            return {0, 3, 4, 2};
        return {0, 3, 4, 5, 2};
    }
    std::string decode(const std::vector<int32_t>&) const override { return {}; }
    int32_t id_for_token(std::string_view) const override { return -1; }
    std::string token_for_id(int32_t) const override { return {}; }
};

class Engine final : public ITrtModule {
  public:
    DType mask_dtype{DType::kInt32};
    bool dynamic{false};
    int calls{0};
    std::vector<float> states;

    TensorMap forward(const TensorMap& inputs) override {
        ++calls;
        const auto& ids = inputs.at("input_ids");
        const auto& mask = inputs.at("attention_mask");
        if (ids.dtype != DType::kInt32 || mask.dtype != mask_dtype || ids.shape != mask.shape)
            throw std::runtime_error("wrong input bindings");
        const auto rows = dynamic ? ids.numel() : 4;
        states.assign(rows * 2, 1000.0f);
        for (std::size_t i = 0; i < std::min(rows, ids.numel()); ++i) {
            const auto valid = mask_dtype == DType::kInt32
                                   ? static_cast<const int32_t*>(mask.data)[i]
                                   : static_cast<const float*>(mask.data)[i];
            if (valid != 1)
                throw std::runtime_error("incorrect attention mask");
            const auto token = static_cast<const int32_t*>(ids.data)[i];
            states[i * 2] = static_cast<float>(token + 1);
            states[i * 2 + 1] = static_cast<float>(9 - token);
        }
        return {
            {"hidden_states", {states.data(), {static_cast<int64_t>(rows), 2}, DType::kFloat32}}};
    }
    DeviceTensorMap forward_device(const DeviceTensorMap&) override {
        throw std::logic_error("unused");
    }
    void forward_device_async(const DeviceTensorMap&) override { throw std::logic_error("unused"); }
    void forward_async(const TensorMap&) override { throw std::logic_error("unused"); }
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<TensorInfo> input_info() const override {
        return {{"input_ids", {dynamic ? -1 : 4}, DType::kInt32, true},
                {"attention_mask", {dynamic ? -1 : 4}, mask_dtype, true}};
    }
    std::vector<TensorInfo> output_info() const override {
        return {{"hidden_states", {dynamic ? -1 : 4, 2}, DType::kFloat32, false}};
    }
    bool has_input(const std::string&) const override { return true; }
    bool has_output(const std::string&) const override { return true; }
    DType tensor_dtype(const std::string&) const override { return DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {4}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             ProfileShapeSelector) const override {
        return {8};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return 1; }
    bool input_is_dynamic(const std::string&) const override { return dynamic; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}
};

void check(bool condition) {
    if (!condition)
        throw std::runtime_error("RoBERTa capacity contract failed");
}

void valid_requests(EncoderPipeline& pipeline, Engine& engine) {
    const auto before = engine.calls;
    for (const auto* text : {"short", "limit", "short"}) {
        const auto ids = Tokenizer().encode(text);
        const auto result = pipeline.embed(text);
        float x = 0.0f, y = 0.0f;
        for (const auto id : ids) {
            x += id + 1;
            y += 9 - id;
        }
        const auto norm = std::sqrt(x * x + y * y);
        check(result.dim == 2 && result.data.size() == 2);
        check(std::abs(result.data[0] - x / norm) < 1e-6f);
        check(std::abs(result.data[1] - y / norm) < 1e-6f);
        const auto encoded = pipeline.encode(text);
        check(encoded.dim == 2 && encoded.data == std::vector<float>({1.0f, 9.0f}));
    }
    check(engine.calls == before + 6);
}

int oversized_requests(EncoderPipeline& pipeline, Engine& engine) {
    int failures = 0;
    for (int route = 0; route < 4; ++route) {
        const auto before = engine.calls;
        bool rejected = false;
        try {
            if (route == 0)
                pipeline.embed("oversize");
            else if (route == 1)
                pipeline.encode("oversize");
            else if (route == 2)
                pipeline.rerank("query", "document");
            else
                pipeline.encode_ids({0, 3, 4, 5, 2});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected || engine.calls != before) {
            std::cerr << "oversize route " << route << " reached forward or was accepted\n";
            ++failures;
        }
    }
    return failures;
}
} // namespace

int main() {
    int failures = 0;
    for (const auto dtype : {DType::kInt32, DType::kFloat32}) {
        auto engine = std::make_unique<Engine>();
        auto* observed = engine.get();
        observed->mask_dtype = dtype;
        EncoderPipeline pipeline(std::move(engine), "embedding", std::make_shared<Tokenizer>());
        valid_requests(pipeline, *observed);
        failures += oversized_requests(pipeline, *observed);
        valid_requests(pipeline, *observed);
    }
    auto engine = std::make_unique<Engine>();
    engine->dynamic = true;
    auto* observed = engine.get();
    EncoderPipeline dynamic(std::move(engine), "embedding", std::make_shared<Tokenizer>());
    try {
        check(dynamic.embed("oversize").dim == 2 && observed->calls == 1);
    } catch (const std::invalid_argument&) {
        std::cerr << "dynamic OPT=4/MAX=8 rejected valid five-token request\n";
        ++failures;
    }
    std::cout << "capacity failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
