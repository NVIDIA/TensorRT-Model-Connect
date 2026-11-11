/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/diffusion_gemma/runtime/pipeline.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace {
using namespace trtmc;
using namespace trtmc::diffusion_gemma;

void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

struct Probe {
    // Module destruction and subsequent creation overwrite these borrowed
    // outputs. The pipeline must retain values independently of module lifetime.
    std::vector<std::uint16_t> staging = std::vector<std::uint16_t>(512);
    OwnedTensor raw_hidden;
    std::vector<std::int32_t> positions;
    bool qkv_live{false};
    bool layer_live{false};
    int qkv_calls{0};
    int layer_calls{0};
    int initial_encoder_layers{0};
    int decoder_layers{0};
    int appended_encoder_layers{0};
    bool invalid_query_shape{false};
};

class Module final : public ITrtModule {
  public:
    Module(std::string name, Probe& probe) : name_(std::move(name)), probe_(probe) {
        const bool qkv = name_.rfind("qkv_", 0) == 0;
        const bool layer = name_.rfind("layer_", 0) == 0;
        check(!(qkv || layer) || (!probe_.qkv_live && !probe_.layer_live),
              "only one projection/layer context may remain live");
        if (qkv || layer)
            std::fill(probe_.staging.begin(), probe_.staging.end(), 0xdead);
        probe_.qkv_live = probe_.qkv_live || qkv;
        probe_.layer_live = probe_.layer_live || layer;
    }
    ~Module() override {
        if (name_.rfind("qkv_", 0) == 0 || name_.rfind("layer_", 0) == 0) {
            std::fill(probe_.staging.begin(), probe_.staging.end(), 0xbeef);
            probe_.qkv_live = probe_.layer_live = false;
        }
    }
    TensorMap forward(const TensorMap& inputs) override {
        if (name_ == "embedding.plan") {
            const auto& ids = inputs.at("input_ids");
            const auto length = ids.shape.at(0);
            local_.assign(static_cast<std::size_t>(length * 4), 0x3f80);
            for (std::int64_t row = 0; row < length; ++row)
                local_[static_cast<std::size_t>(row * 4)] +=
                    static_cast<const std::int32_t*>(ids.data)[row];
            const Tensor result{local_.data(), {length, 4}, DType::kBFloat16};
            return {{"encoder_embeddings", result}, {"decoder_embeddings", result}};
        }
        if (name_ == "output.plan") {
            const auto length = inputs.at("hidden_states").shape.at(0);
            logits_.assign(static_cast<std::size_t>(length * 8), 0.0F);
            local_.assign(static_cast<std::size_t>(length * 4), 0x3f80);
            return {{"raw_logits", {logits_.data(), {length, 8}, DType::kFloat32}},
                    {"self_conditioning", {local_.data(), {length, 4}, DType::kBFloat16}}};
        }
        const auto& hidden = inputs.at("hidden_states");
        const auto length = hidden.shape.at(0);
        const auto first = static_cast<const std::uint16_t*>(hidden.data)[0];
        if (name_.rfind("qkv_", 0) == 0) {
            ++probe_.qkv_calls;
            check(inputs.size() == 2, "projection consumes only raw hidden states and positions");
            probe_.raw_hidden = OwnedTensor(hidden);
            const auto* positions =
                static_cast<const std::int32_t*>(inputs.at("position_ids").data);
            probe_.positions.assign(positions, positions + length);
            std::fill_n(probe_.staging.data(), length * 4, first + 0x100);
            std::fill_n(probe_.staging.data() + 128, length * 2, first + 0x200);
            std::fill_n(probe_.staging.data() + 256, length * 2, first + 0x300);
            return {{"query",
                     {probe_.staging.data(),
                      {probe_.invalid_query_shape ? 1 : 2, length, 2},
                      DType::kBFloat16}},
                    {"key", {probe_.staging.data() + 128, {1, length, 2}, DType::kBFloat16}},
                    {"value", {probe_.staging.data() + 256, {1, length, 2}, DType::kBFloat16}}};
        }
        ++probe_.layer_calls;
        check(inputs.size() == 7 && inputs.find("position_ids") == inputs.end(),
              "layer consumes the materialized QKV contract");
        check(!probe_.qkv_live, "projection context must be released before the main layer");
        check(hidden.shape == probe_.raw_hidden.shape &&
                  std::memcmp(hidden.data, probe_.raw_hidden.bytes.data(), hidden.nbytes()) == 0,
              "main layer must receive the original residual input");
        for (const auto& [name, offset] : std::vector<std::pair<std::string, std::uint16_t>>{
                 {"query", 0x100}, {"key", 0x200}, {"value", 0x300}}) {
            const auto& tensor = inputs.at(name);
            check(tensor.dtype == DType::kBFloat16, "QKV handoff must retain BF16 bits");
            const auto* data = static_cast<const std::uint16_t*>(tensor.data);
            check(std::all_of(data, data + tensor.numel(),
                              [&](auto bits) { return bits == first + offset; }),
                  "QKV values must survive projection destruction and staging reuse");
        }
        const auto& past_key = inputs.at("past_key");
        const auto& mask = inputs.at("attention_mask");
        check(mask.shape == std::vector<std::int64_t>{length, past_key.shape[1] + length},
              "attention mask must include precisely the retained prefix");
        const auto* mask_values = static_cast<const float*>(mask.data);
        if (probe_.positions.front() == 0) {
            ++probe_.initial_encoder_layers;
            check(past_key.shape[1] == 0, "each initial encoder layer starts with an empty cache");
        } else if (length == 3) {
            ++probe_.decoder_layers;
            check(probe_.positions == std::vector<std::int32_t>{4, 5, 6},
                  "decoder uses absolute positions after the full prefix");
            check(std::all_of(mask_values, mask_values + mask.numel(),
                              [](float value) { return value == 0.0F; }),
                  "decoder attends to retained prefix and complete canvas");
        } else {
            ++probe_.appended_encoder_layers;
            check(probe_.positions == std::vector<std::int32_t>{4},
                  "encoder append advances absolute positions");
        }
        std::copy_n(static_cast<const std::uint16_t*>(hidden.data), hidden.numel(),
                    probe_.staging.data());
        for (std::size_t i = 0; i < hidden.numel(); ++i)
            ++probe_.staging[i];
        const auto total = past_key.shape[1] + length;
        for (const auto& [name, output_offset] :
             std::vector<std::pair<std::string, std::size_t>>{{"key", 128}, {"value", 256}}) {
            const auto& past = inputs.at("past_" + name);
            auto* target = probe_.staging.data() + output_offset;
            if (past.nbytes() != 0)
                std::memcpy(target, past.data, past.nbytes());
            std::memcpy(target + past.numel(), inputs.at(name).data, inputs.at(name).nbytes());
        }
        return {{"hidden_states_out", {probe_.staging.data(), hidden.shape, DType::kBFloat16}},
                {"present_key", {probe_.staging.data() + 128, {1, total, 2}, DType::kBFloat16}},
                {"present_value", {probe_.staging.data() + 256, {1, total, 2}, DType::kBFloat16}}};
    }
    DeviceTensorMap forward_device(const DeviceTensorMap&) override { return {}; }
    void forward_device_async(const DeviceTensorMap&) override {}
    void forward_async(const TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<TensorInfo> input_info() const override { return {}; }
    std::vector<TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string&) const override { return false; }
    bool has_output(const std::string&) const override { return false; }
    DType tensor_dtype(const std::string&) const override { return DType::kBFloat16; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return 0; }
    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    std::string name_;
    Probe& probe_;
    std::vector<std::uint16_t> local_;
    std::vector<float> logits_;
};

class Backend final : public IBackend {
  public:
    Probe probe;
    std::unique_ptr<ITrtModule> create_module(const void* plan, std::size_t size,
                                              const ModuleCreateOptions&) override {
        return std::make_unique<Module>(std::string(static_cast<const char*>(plan), size), probe);
    }
    std::unique_ptr<ITrtModule>
    create_module_prebound(const void*, std::size_t, const ModuleCreateOptions&,
                           const std::vector<ModuleExternalBinding>&) override {
        return {};
    }
    BackendDualProfileModules create_dual_profile_modules(const void*, std::size_t,
                                                          const ModuleCreateOptions&) override {
        return {};
    }
    const char* name() const override { return "test"; }
};

class Tokenizer final : public ITokenizer {
  public:
    std::vector<std::int32_t> encode(const std::string&) const override { return {2}; }
    std::string decode(const std::vector<std::int32_t>&) const override { return {}; }
    std::int32_t id_for_token(std::string_view) const override { return 0; }
    std::string token_for_id(std::int32_t) const override { return {}; }
};

class TestBundle {
  public:
    TestBundle() {
        path = std::filesystem::temp_directory_path() /
               ("diffusion-gemma-pipeline-" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                ".bundle");
        nlohmann::json header{{"format", 1},
                              {"family", "diffusion_gemma"},
                              {"task", "text_continuation"},
                              {"backend", "test"},
                              {"sections", nlohmann::json::object()}};
        std::string payload;
        for (const std::string name : {"embedding.plan", "qkv_0.plan", "layer_0.plan", "qkv_1.plan",
                                       "layer_1.plan", "output.plan"}) {
            header["sections"][name] = {{"offset", payload.size()}, {"length", name.size()}};
            payload += name;
        }
        const auto text = header.dump();
        std::ofstream stream(path, std::ios::binary);
        stream.write("BUNDLE\x01\x00", 8);
        const std::uint64_t size = text.size();
        stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
        stream << text << payload;
    }
    ~TestBundle() { std::filesystem::remove(path); }
    std::filesystem::path path;
};
} // namespace

int main() {
    try {
        TestBundle bundle;
        Backend backend;
        RuntimeConfig config;
        config.hidden_size = 4;
        config.vocab_size = 8;
        config.num_layers = 2;
        config.num_attention_heads = 2;
        config.num_key_value_heads = config.num_global_key_value_heads = 1;
        config.head_dim = config.global_head_dim = 2;
        config.sliding_window = 3;
        config.max_sequence_length = 12;
        config.canvas_length = 3;
        config.layer_types = {"sliding_attention", "full_attention"};
        Pipeline pipeline(BundleReader(bundle.path.string()), backend, config,
                          std::make_unique<Tokenizer>());
        pipeline.encode({2, 3, 4, 5});
        check(backend.probe.initial_encoder_layers == 2, "both encoder stages execute");
        check(pipeline.keys()[0].shape == std::vector<std::int64_t>{1, 2, 2} &&
                  pipeline.keys()[1].shape == std::vector<std::int64_t>{1, 4, 2},
              "sliding cache retains its tail while full cache retains the prefix");
        const auto keys = pipeline.keys(), values = pipeline.values();
        auto conditioning = pipeline.zero_conditioning(3);
        for (int step = 0; step < 2; ++step) {
            auto observation = pipeline.denoise({3, 4, 5}, conditioning, 0.8F);
            conditioning = std::move(observation.self_conditioning);
            for (std::size_t layer = 0; layer < keys.size(); ++layer)
                check(pipeline.keys()[layer].bytes == keys[layer].bytes &&
                          pipeline.values()[layer].bytes == values[layer].bytes,
                      "decoder steps must preserve the exact encoder cache bytes");
        }
        check(backend.probe.decoder_layers == 4, "both decoder stages execute twice");
        pipeline.encode({6});
        check(backend.probe.appended_encoder_layers == 2, "paired encoder append executes");
        check(pipeline.keys()[1].shape == std::vector<std::int64_t>{1, 5, 2},
              "encoder append commits the newly projected cache");
        pipeline.reset();
        backend.probe.invalid_query_shape = true;
        bool rejected = false;
        try {
            pipeline.encode({2});
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected, "invalid QKV output shapes must fail before the main layer");
        check(backend.probe.qkv_calls == backend.probe.layer_calls + 1,
              "shape failure cannot reach a main layer");
        std::cout << "diffusion_gemma paired pipeline contracts passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
