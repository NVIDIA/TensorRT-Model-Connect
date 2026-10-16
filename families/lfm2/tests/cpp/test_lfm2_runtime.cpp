/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lfm2/runtime/conv_state.h"
#include "families/lfm2/runtime/hybrid_state.h"
#include "families/lfm2/runtime/kv_cache.h"
#include "families/lfm2/runtime/pipeline.h"
#include "trtmc/runtime/trt_module.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

class FakeLfm2Module final : public trtmc::ITrtModule {
  public:
    struct Call {
        int32_t token{0};
        int32_t position{0};
        int32_t write_index{0};
        int32_t kv_length{0};
        std::vector<int32_t> tokens;
        std::vector<int32_t> positions;
    };
    struct PhaseCall {
        int32_t value;
        trtmc::DType dtype;
        std::vector<int64_t> shape;
    };

    FakeLfm2Module(cudaStream_t stream, trtmc::DType state_dtype, int32_t vocab_size = 6,
                   bool explicit_phase = false)
        : stream_(stream), state_dtype_(state_dtype), vocab_size_(vocab_size),
          device_logits_({vocab_size}, trtmc::DType::kFloat32, stream) {
        add("token_id", {1}, trtmc::DType::kInt32, true);
        if (explicit_phase)
            add("decode_phase", {1, 1}, trtmc::DType::kInt32, true);
        add("position_id", {1}, trtmc::DType::kInt32, true);
        add("cache_write_indices", {1}, trtmc::DType::kInt32, true);
        add("key_value_lengths", {1}, trtmc::DType::kInt32, true);
        add("cache_k_0", {1, 2, 8, 4}, state_dtype_, true);
        add("cache_v_0", {1, 2, 8, 4}, state_dtype_, true);
        add("present_k_0", {1, 2, 8, 4}, state_dtype_, false);
        add("present_v_0", {1, 2, 8, 4}, state_dtype_, false);
        add("conv_state_0", {4, 3}, state_dtype_, true);
        add("present_conv_0", {4, 3}, state_dtype_, false);
        add("logits", {vocab_size}, trtmc::DType::kFloat32, false);
        bindings_["logits"] = device_logits_.data();
    }

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        forward_async(inputs);
        sync();
        return {};
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}

    void forward_async(const trtmc::TensorMap& inputs) override {
        const auto& token_input = inputs.at("token_id");
        const auto& positions = inputs.at("position_id");
        const auto count = token_input.shape.at(0);
        check(token_input.dtype == trtmc::DType::kInt32 && token_input.shape.size() == 1 &&
                  positions.shape == token_input.shape && count > 0,
              "token and position vectors have equal positive length");
        const auto* token_data = static_cast<const int32_t*>(token_input.data);
        const auto* position_data = static_cast<const int32_t*>(positions.data);
        if (has_input("decode_phase")) {
            const auto& phase = inputs.at("decode_phase");
            phase_calls.push_back({scalar(inputs, "decode_phase"), phase.dtype, phase.shape});
        } else {
            check(inputs.count("decode_phase") == 0,
                  "legacy engines receive no undeclared decode-phase input");
        }
        int32_t length = 0;
        if (has_input("active_length_extent")) {
            const auto& extent = inputs.at("active_length_extent");
            check(extent.data == nullptr && extent.dtype == trtmc::DType::kInt32 &&
                      extent.shape.size() == 1,
                  "extent input transports host dimensions without values");
            check(inputs.count("key_value_lengths") == 0,
                  "extent engines receive no scalar length");
            length = static_cast<int32_t>(extent.shape.at(0));
        } else {
            length = scalar(inputs, "key_value_lengths");
        }
        calls.push_back(Call{scalar(inputs, "token_id"),
                             scalar(inputs, "position_id"),
                             scalar(inputs, "cache_write_indices"),
                             length,
                             {token_data, token_data + count},
                             {position_data, position_data + count}});
        std::vector<float> logits(static_cast<std::size_t>(vocab_size_), -10.0F);
        logits[emitted_token] = 10.0F;
        device_logits_.copy_from_host(logits.data());
        const auto present = bindings_.find("present_conv_0");
        if (present != bindings_.end()) {
            cudaMemsetAsync(present->second, 0x2A, 4U * 3U * trtmc::dtype_size(state_dtype_),
                            stream_);
        }
    }

    void sync() override { cudaStreamSynchronize(stream_); }
    cudaStream_t stream() const override { return stream_; }
    void enable_cuda_graph() override { graph_enabled_ = true; }
    bool cuda_graph_active() const override { return graph_enabled_; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return selected_profile; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override { return has(name, true); }
    bool has_output(const std::string& name) const override { return has(name, false); }
    trtmc::DType tensor_dtype(const std::string& name) const override {
        return entries_.at(name).dtype;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        const auto found = entries_.find(name);
        return found == entries_.end() ? std::vector<int64_t>{} : found->second.shape;
    }
    std::vector<int64_t> input_profile_shape(const std::string& name, int32_t profile,
                                             trtmc::ProfileShapeSelector selector) const override {
        if (batched_profiles && (name == "token_id" || name == "position_id")) {
            if (profile == 0)
                return prefill_rows.at(static_cast<std::size_t>(selector));
            return decode_rows.at(static_cast<std::size_t>(selector));
        }
        if (name == "active_length_extent")
            return extent_profile.at(static_cast<std::size_t>(selector));
        return tensor_shape(name);
    }
    int32_t optimization_profile_count() const override { return profile_count; }
    void* device_ptr(const std::string& name) const override {
        const auto found = bindings_.find(name);
        return found == bindings_.end() ? nullptr : found->second;
    }
    void bind_external(const std::string& name, void* pointer) override {
        bindings_[name] = pointer;
        if (name == "cache_k_0")
            bindings_["present_k_0"] = pointer;
        if (name == "cache_v_0")
            bindings_["present_v_0"] = pointer;
    }
    void bind_external(const std::string& name, void* pointer,
                       const std::vector<int64_t>&) override {
        bind_external(name, pointer);
    }
    int32_t input_rank(const std::string& name) const override {
        return has_input(name) ? static_cast<int32_t>(tensor_shape(name).size()) : 0;
    }
    bool input_is_dynamic(const std::string& name) const override {
        return (name == "active_length_extent" && extent_dynamic) ||
               (batched_profiles && (name == "token_id" || name == "position_id"));
    }
    void reset_execution_context() override { ++reset_calls; }
    void set_timing_label(std::string) override {}
    bool ok() const override { return stream_ != nullptr && device_logits_.ok(); }
    void keep_alive(std::shared_ptr<void> resource) override {
        resources_.push_back(std::move(resource));
    }

    void set_tensor(const std::string& name, std::vector<int64_t> shape, trtmc::DType dtype) {
        auto& entry = entries_.at(name);
        entry.shape = std::move(shape);
        entry.dtype = dtype;
    }

    void use_extent(bool keep_scalar = false) {
        if (!keep_scalar)
            entries_.erase("key_value_lengths");
        add("active_length_extent", {8}, trtmc::DType::kInt32, true);
    }

    void remove_tensor(const std::string& name) { entries_.erase(name); }

    void use_batched_profiles(int profile) {
        use_extent();
        if (!has_input("decode_phase"))
            add("decode_phase", {1, 1}, trtmc::DType::kInt32, true);
        batched_profiles = true;
        profile_count = 2;
        selected_profile = profile;
        entries_.at("token_id").shape = {profile == 0 ? 8 : 1};
        entries_.at("position_id").shape = {profile == 0 ? 8 : 1};
    }

    std::vector<Call> calls;
    std::vector<PhaseCall> phase_calls;
    std::array<std::vector<int64_t>, 3> extent_profile{{{1}, {8}, {8}}};
    int32_t selected_profile{0};
    int32_t profile_count{1};
    bool extent_dynamic{true};
    bool batched_profiles{false};
    std::array<std::vector<int64_t>, 3> prefill_rows{{{1}, {8}, {8}}};
    std::array<std::vector<int64_t>, 3> decode_rows{{{1}, {1}, {1}}};
    int emitted_token{2};
    int reset_calls{0};

  private:
    struct Entry {
        std::vector<int64_t> shape;
        trtmc::DType dtype;
        bool input;
    };

    void add(std::string name, std::vector<int64_t> shape, trtmc::DType dtype, bool input) {
        entries_.emplace(std::move(name), Entry{std::move(shape), dtype, input});
    }
    bool has(const std::string& name, bool input) const {
        const auto found = entries_.find(name);
        return found != entries_.end() && found->second.input == input;
    }
    static int32_t scalar(const trtmc::TensorMap& inputs, const std::string& name) {
        return *static_cast<const int32_t*>(inputs.at(name).data);
    }

    cudaStream_t stream_{nullptr};
    trtmc::DType state_dtype_;
    int32_t vocab_size_{0};
    trtmc::DeviceTensor device_logits_;
    bool graph_enabled_{false};
    std::unordered_map<std::string, Entry> entries_;
    std::unordered_map<std::string, void*> bindings_;
    std::vector<std::shared_ptr<void>> resources_;
};

class RecordingTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string& text) const override {
        last_text = text;
        // Simulate add_bos_token=true applied to text that already begins with
        // the chat template's literal BOS token.
        return {1, 1, 4};
    }
    std::string decode(const std::vector<int32_t>&) const override { return "decoded"; }
    int32_t id_for_token(std::string_view token) const override {
        return token == "<|startoftext|>" ? 1 : -1;
    }
    std::string token_for_id(int32_t id) const override { return id == 1 ? "<|startoftext|>" : ""; }

    mutable std::string last_text;
};

trtmc::Lfm2KvCacheNames extent_names() {
    trtmc::Lfm2KvCacheNames names;
    names.key_value_lengths.clear();
    names.active_length_extent = "active_length_extent";
    return names;
}

void test_exactly_one_length_contract() {
    for (bool both : {false, true}) {
        auto names = extent_names();
        if (both)
            names.key_value_lengths = "key_value_lengths";
        else
            names.active_length_extent.clear();
        bool rejected = false;
        try {
            trtmc::Lfm2KvCache cache(1, 8, 2, 4, nullptr, trtmc::DType::kBFloat16, names);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "cache names require exactly one length contract before allocation");
    }
    trtmc::Lfm2KvCacheNames scalar;
    scalar.batched_prefill = true;
    bool rejected = false;
    try {
        trtmc::Lfm2KvCache cache(1, 8, 2, 4, nullptr, trtmc::DType::kBFloat16, scalar);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "batched prefill cannot use the legacy scalar length contract");
}

void test_batched_prefill_and_decode_handoff(cudaStream_t stream) {
    auto prefill = std::make_unique<FakeLfm2Module>(stream, trtmc::DType::kBFloat16, 6, true);
    auto decode = std::make_unique<FakeLfm2Module>(stream, trtmc::DType::kBFloat16, 6, true);
    prefill->use_batched_profiles(0);
    decode->use_batched_profiles(1);
    prefill->set_tensor("logits", {1, 6}, trtmc::DType::kFloat32);
    decode->set_tensor("logits", {1, 6}, trtmc::DType::kFloat32);
    prefill->emitted_token = 2;
    decode->emitted_token = 3;
    auto* prefill_trace = prefill.get();
    auto* decode_trace = decode.get();
    auto names = extent_names();
    names.batched_prefill = true;
    auto kv =
        std::make_unique<trtmc::Lfm2KvCache>(1, 8, 2, 4, stream, trtmc::DType::kBFloat16, names);
    auto conv = std::make_unique<trtmc::Lfm2ConvState>(1, 4, 3, stream, trtmc::DType::kBFloat16);
    auto* kv_state = kv.get();
    auto* conv_state = conv.get();
    auto state = std::make_unique<trtmc::Lfm2HybridState>(std::move(kv), std::move(conv));
    trtmc::Lfm2TextGenConfig config;
    config.vocab_size = 6;
    config.id_eos = 5;
    config.id_eos_ids = {5};
    trtmc::Lfm2TextGenerationPipeline pipeline(std::move(decode), std::move(state), config, stream,
                                               nullptr, "", nullptr, std::move(prefill));
    for (const char* name :
         {"cache_k_0", "cache_v_0", "present_k_0", "present_v_0", "conv_state_0", "present_conv_0"})
        check(prefill_trace->device_ptr(name) == decode_trace->device_ptr(name),
              "prefill and decode contexts bind identical persistent buffers");
    check(prefill_trace->device_ptr("logits") != decode_trace->device_ptr("logits"),
          "handoff fixture has distinct logits allocations");
    trtmc::TextGenerationConfig request;
    request.max_new_tokens = 3;
    request.temperature = 0;
    const auto result = pipeline.generate_ids({4, 1, 4}, request);
    check(result.token_ids == std::vector<int32_t>({4, 1, 4, 2, 3, 3}),
          "generation reads logits from the active prefill then decode module");
    check(prefill_trace->calls.size() == 1 && decode_trace->calls.size() == 2,
          "one full prefill is followed by exactly the required decode steps");
    if (prefill_trace->calls.size() == 1) {
        const auto& call = prefill_trace->calls[0];
        check(call.tokens == std::vector<int32_t>({4, 1, 4}) &&
                  call.positions == std::vector<int32_t>({0, 1, 2}) && call.write_index == 0 &&
                  call.kv_length == 3 && prefill_trace->phase_calls[0].value == 0,
              "full prompt uses position-zero prefill and active length R");
    }
    if (decode_trace->calls.size() == 2) {
        for (int index = 0; index < 2; ++index) {
            const auto& call = decode_trace->calls[index];
            check(call.tokens.size() == 1 && call.positions == std::vector<int32_t>{3 + index} &&
                      call.write_index == 3 + index && call.kv_length == 4 + index &&
                      decode_trace->phase_calls[index].value == 1,
                  "decode starts immediately after all prefill rows and keeps single-token phase");
        }
    }
    check(kv_state->position() == 5 && conv_state->position() == 5,
          "KV and convolution advance by full prefill rows then decode rows");
    std::vector<unsigned char> tail(4 * 3 * 2);
    cudaMemcpy(tail.data(), conv_state->state(0).data(), tail.size(), cudaMemcpyDeviceToHost);
    check(std::all_of(tail.begin(), tail.end(), [](unsigned char byte) { return byte == 0x2a; }),
          "the final convolution tail is committed to the shared stable state");
    request.max_new_tokens = 2;
    const auto second = pipeline.generate_ids({4}, request);
    check(second.token_ids == std::vector<int32_t>({4, 2, 3}) && prefill_trace->calls.size() == 2 &&
              prefill_trace->calls.back().positions == std::vector<int32_t>{0} &&
              decode_trace->calls.size() == 3 && decode_trace->calls.back().position == 1 &&
              prefill_trace->reset_calls == 2 && decode_trace->reset_calls == 2,
          "same pipeline resets both contexts and supports a one-token prompt");
}

void test_batched_profile_rejections(cudaStream_t stream) {
    auto names = extent_names();
    names.batched_prefill = true;
    trtmc::Lfm2KvCache cache(1, 8, 2, 4, stream, trtmc::DType::kBFloat16, names);
    for (int profile : {0, 1}) {
        FakeLfm2Module module(stream, trtmc::DType::kBFloat16);
        module.use_batched_profiles(profile);
        cache.bind_to(module);
    }
    for (int fault = 0; fault < 11; ++fault) {
        FakeLfm2Module module(stream, trtmc::DType::kBFloat16);
        module.use_batched_profiles(0);
        switch (fault) {
        case 0:
            module.profile_count = 1;
            break;
        case 1:
            module.selected_profile = 2;
            break;
        case 2:
            module.prefill_rows[2] = {7};
            break;
        case 3:
            module.decode_rows[1] = {2};
            break;
        case 4:
            module.set_tensor("position_id", {8}, trtmc::DType::kFloat32);
            break;
        case 5:
            module.set_tensor("token_id", {1, 8}, trtmc::DType::kInt32);
            break;
        case 6:
            module.extent_profile[1] = {4};
            break;
        case 7:
            module.enable_cuda_graph();
            break;
        case 8:
            module.remove_tensor("decode_phase");
            break;
        case 9:
            module.set_tensor("decode_phase", {1, 1}, trtmc::DType::kFloat32);
            break;
        case 10:
            module.set_tensor("decode_phase", {1}, trtmc::DType::kInt32);
            break;
        }
        bool rejected = false;
        try {
            cache.bind_to(module);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected && module.device_ptr("cache_k_0") == nullptr,
              "batched metadata faults fail before persistent buffer binding");
    }
    trtmc::TensorMap inputs;
    cache.prepare_step(inputs, 3);
    check(inputs.at("position_id").shape == std::vector<int64_t>{3} &&
              inputs.at("active_length_extent").shape == std::vector<int64_t>{3} &&
              inputs.at("active_length_extent").data == nullptr,
          "batched prefill emits R positions and null extent payload T=R");
    cache.advance(3);
    inputs.clear();
    bool rejected = false;
    try {
        cache.prepare_step(inputs, 2);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected && inputs.empty() && cache.position() == 3,
          "chunked prefill and multirow decode are rejected without state mutation");
}

void test_extent_contract_and_single_token_progression(cudaStream_t stream) {
    FakeLfm2Module module(stream, trtmc::DType::kBFloat16);
    module.use_extent();
    trtmc::Lfm2KvCache cache(1, 8, 2, 4, stream, trtmc::DType::kBFloat16, extent_names());
    cache.bind_to(module);
    trtmc::TensorMap inputs;
    bool rejected = false;
    try {
        cache.prepare_step(inputs, 2);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected && inputs.empty() && cache.position() == 0,
          "extent mode rejects multi-token steps without modifying inputs or state");
    for (int32_t position = 0; position < 8; ++position) {
        inputs.clear();
        cache.prepare_step(inputs);
        const auto& extent = inputs.at("active_length_extent");
        check(extent.data == nullptr && extent.dtype == trtmc::DType::kInt32 &&
                  extent.shape == std::vector<int64_t>{position + 1} &&
                  inputs.count("key_value_lengths") == 0 &&
                  *static_cast<const int32_t*>(inputs.at("cache_write_indices").data) == position &&
                  *static_cast<const int32_t*>(inputs.at("position_id").data) == position,
              "extent, write index and position describe the same host-owned single-token step");
        cache.advance();
    }
    rejected = false;
    try {
        inputs.clear();
        cache.prepare_step(inputs);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    check(rejected && inputs.empty() && cache.position() == 8,
          "extent contract cannot exceed its profile capacity");
    cache.reset();
    cache.prepare_step(inputs);
    check(inputs.at("active_length_extent").shape == std::vector<int64_t>{1} &&
              inputs.at("active_length_extent").data == nullptr,
          "reset restarts extent metadata at one without scalar upload");

    for (int fault = 0; fault < 11; ++fault) {
        FakeLfm2Module malformed(stream, trtmc::DType::kBFloat16);
        malformed.use_extent(fault == 0);
        switch (fault) {
        case 1:
            malformed.remove_tensor("active_length_extent");
            break;
        case 2:
            malformed.set_tensor("active_length_extent", {8}, trtmc::DType::kFloat32);
            break;
        case 3:
            malformed.set_tensor("active_length_extent", {1, 8}, trtmc::DType::kInt32);
            break;
        case 4:
            malformed.extent_dynamic = false;
            break;
        case 5:
            malformed.extent_profile[0] = {2};
            break;
        case 6:
            malformed.extent_profile[1] = {4};
            break;
        case 7:
            malformed.extent_profile[2] = {9};
            break;
        case 8:
            malformed.profile_count = 2;
            break;
        case 9:
            malformed.selected_profile = 1;
            break;
        case 10:
            malformed.enable_cuda_graph();
            break;
        }
        rejected = false;
        try {
            cache.bind_to(malformed);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected && malformed.device_ptr("cache_k_0") == nullptr,
              "extent signature/profile/capture faults reject before binding KV buffers");
    }
}

void test_pipeline_uses_extent_and_resets(cudaStream_t stream) {
    auto module = std::make_unique<FakeLfm2Module>(stream, trtmc::DType::kBFloat16);
    module->use_extent();
    auto* trace = module.get();
    auto kv = std::make_unique<trtmc::Lfm2KvCache>(1, 8, 2, 4, stream, trtmc::DType::kBFloat16,
                                                   extent_names());
    auto conv = std::make_unique<trtmc::Lfm2ConvState>(1, 4, 3, stream, trtmc::DType::kBFloat16);
    auto state = std::make_unique<trtmc::Lfm2HybridState>(std::move(kv), std::move(conv));
    trtmc::Lfm2TextGenConfig config;
    config.vocab_size = 6;
    config.id_eos = 5;
    config.id_eos_ids = {5};
    trtmc::Lfm2TextGenerationPipeline pipeline(std::move(module), std::move(state), config, stream);
    trtmc::TextGenerationConfig request;
    request.temperature = 0.0F;
    request.max_new_tokens = 2;
    const auto first = pipeline.generate_ids({4, 1}, request);
    check(first.token_ids == std::vector<int32_t>({4, 1, 2, 2}) && trace->calls.size() == 3,
          "extent pipeline preserves deterministic generation");
    request.max_new_tokens = 1;
    (void)pipeline.generate_ids({4}, request);
    check(trace->calls.size() == 4 && trace->calls.back().position == 0 &&
              trace->calls.back().kv_length == 1,
          "same pipeline resets host extent state for the next request");
    for (std::size_t index = 0; index < 3 && index < trace->calls.size(); ++index) {
        const auto& call = trace->calls[index];
        check(call.kv_length == static_cast<int32_t>(index + 1) &&
                  call.position == static_cast<int32_t>(index) && call.write_index == call.position,
              "pipeline forwards matching extent, position and KV write index");
    }
}

void test_native_kv_and_double_buffered_conv(cudaStream_t stream) {
    FakeLfm2Module module(stream, trtmc::DType::kFloat16);
    auto kv = std::make_unique<trtmc::Lfm2KvCache>(1, 8, 2, 4, stream, trtmc::DType::kFloat16);
    auto conv = std::make_unique<trtmc::Lfm2ConvState>(1, 4, 3, stream, trtmc::DType::kFloat16);
    auto* kv_ptr = kv.get();
    auto* conv_ptr = conv.get();
    trtmc::Lfm2HybridState state(std::move(kv), std::move(conv));
    state.bind_to(module);

    check(module.device_ptr("cache_k_0") == kv_ptr->cache_k(0).data() &&
              module.device_ptr("present_k_0") == kv_ptr->cache_k(0).data(),
          "native KV input/output aliases one allocation");
    check(module.device_ptr("conv_state_0") == conv_ptr->state(0).data() &&
              module.device_ptr("present_conv_0") == conv_ptr->present(0).data() &&
              conv_ptr->state(0).data() != conv_ptr->present(0).data(),
          "convolution state keeps stable double buffers");

    trtmc::TensorMap inputs;
    state.prepare_step(inputs);
    int32_t token = 4;
    inputs["token_id"] = trtmc::Tensor{&token, {1}, trtmc::DType::kInt32};
    check(inputs.count("attention_mask") == 0, "native path has no dense attention mask");
    check(*static_cast<int32_t*>(inputs.at("position_id").data) == 0 &&
              *static_cast<int32_t*>(inputs.at("cache_write_indices").data) == 0 &&
              *static_cast<int32_t*>(inputs.at("key_value_lengths").data) == 1,
          "first-step native KV metadata is correct");

    module.forward_async(inputs);
    module.sync();
    state.advance();
    std::vector<unsigned char> copied(conv_ptr->state(0).nbytes());
    conv_ptr->state(0).copy_to_host(copied.data());
    check(std::all_of(copied.begin(), copied.end(),
                      [](unsigned char value) { return value == 0x2A; }),
          "advance copies present convolution state into stable input storage");
    check(state.position() == 1, "hybrid state advances in lockstep");

    state.reset();
    std::fill(copied.begin(), copied.end(), 0xFF);
    conv_ptr->state(0).copy_to_host(copied.data());
    check(std::all_of(copied.begin(), copied.end(), [](unsigned char value) { return value == 0; }),
          "reset clears recurrent convolution history");
}

void test_native_contract_rejection_and_capacity(cudaStream_t stream) {
    {
        FakeLfm2Module malformed(stream, trtmc::DType::kFloat16);
        malformed.set_tensor("key_value_lengths", {2}, trtmc::DType::kInt32);
        trtmc::Lfm2KvCache cache(1, 8, 2, 4, stream, trtmc::DType::kFloat16);
        bool rejected = false;
        try {
            cache.bind_to(malformed);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected, "native KV rejects malformed scalar metadata");
    }

    {
        FakeLfm2Module module(stream, trtmc::DType::kFloat16);
        trtmc::Lfm2KvCache cache(1, 8, 2, 4, stream, trtmc::DType::kFloat16);
        cache.bind_to(module);
        for (int32_t position = 0; position < 8; ++position) {
            trtmc::TensorMap inputs;
            cache.prepare_step(inputs);
            cache.advance();
        }
        bool rejected = false;
        try {
            trtmc::TensorMap inputs;
            cache.prepare_step(inputs);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected && cache.position() == 8,
              "native KV rejects writes beyond fixed capacity without progression");
    }

    {
        FakeLfm2Module module(stream, trtmc::DType::kBFloat16);
        auto kv = std::make_unique<trtmc::Lfm2KvCache>(1, 8, 2, 4, stream, trtmc::DType::kBFloat16);
        auto conv =
            std::make_unique<trtmc::Lfm2ConvState>(1, 4, 3, stream, trtmc::DType::kBFloat16);
        trtmc::Lfm2HybridState state(std::move(kv), std::move(conv));
        state.bind_to(module);
        check(state.ok(), "hybrid state preserves BF16 engine storage precision");
    }
}

void test_pipeline_uses_safe_one_token_progression(cudaStream_t stream) {
    auto module = std::make_unique<FakeLfm2Module>(stream, trtmc::DType::kFloat16);
    module->set_tensor("logits", {1, 6}, trtmc::DType::kFloat32);
    auto* trace = module.get();
    auto kv = std::make_unique<trtmc::Lfm2KvCache>(1, 8, 2, 4, stream, trtmc::DType::kFloat16);
    auto conv = std::make_unique<trtmc::Lfm2ConvState>(1, 4, 3, stream, trtmc::DType::kFloat16);
    auto state = std::make_unique<trtmc::Lfm2HybridState>(std::move(kv), std::move(conv));

    trtmc::Lfm2TextGenConfig config;
    config.vocab_size = 6;
    config.id_eos = 5;
    config.id_eos_ids = {5};
    trtmc::Lfm2TextGenerationPipeline pipeline(std::move(module), std::move(state), config, stream);

    trtmc::TextGenerationConfig request;
    request.max_new_tokens = 2;
    request.temperature = 0.0F;
    const auto result = pipeline.generate_ids({4, 1}, request);
    check(result.token_ids == std::vector<int32_t>({4, 1, 2, 2}),
          "pipeline generates expected deterministic continuation");
    check(trace->calls.size() == 3, "two prompt tokens plus one decode token execute");
    if (trace->calls.size() == 3) {
        for (std::size_t index = 0; index < trace->calls.size(); ++index) {
            const auto& call = trace->calls[index];
            check(call.position == static_cast<int32_t>(index) &&
                      call.write_index == static_cast<int32_t>(index) &&
                      call.kv_length == static_cast<int32_t>(index + 1),
                  "pipeline progresses native KV one token at a time");
        }
    }
}

void test_pipeline_selects_decode_phase_and_resets(cudaStream_t stream) {
    auto module = std::make_unique<FakeLfm2Module>(stream, trtmc::DType::kBFloat16, 6, true);
    auto* trace = module.get();
    auto kv = std::make_unique<trtmc::Lfm2KvCache>(1, 8, 2, 4, stream, trtmc::DType::kBFloat16);
    auto conv = std::make_unique<trtmc::Lfm2ConvState>(1, 4, 3, stream, trtmc::DType::kBFloat16);
    auto state = std::make_unique<trtmc::Lfm2HybridState>(std::move(kv), std::move(conv));
    trtmc::Lfm2TextGenConfig config;
    config.vocab_size = 6;
    config.id_eos = 5;
    config.id_eos_ids = {5}; // The fake emits 2, so all requested steps execute.
    trtmc::Lfm2TextGenerationPipeline pipeline(std::move(module), std::move(state), config, stream);
    trtmc::TextGenerationConfig request;
    request.max_new_tokens = 3;
    request.temperature = 0.0F;
    const auto first = pipeline.generate_ids({1, 4, 5}, request);
    check(first.token_ids == std::vector<int32_t>({1, 4, 5, 2, 2, 2}),
          "phase test executes a complete three-token continuation");
    std::vector<int32_t> phases;
    for (const auto& call : trace->phase_calls) {
        check(call.dtype == trtmc::DType::kInt32 && call.shape == std::vector<int64_t>({1, 1}),
              "native phase input uses INT32[1,1] on every step");
        phases.push_back(call.value);
    }
    check(phases == std::vector<int32_t>({0, 0, 0, 1, 1}),
          "all prompt tokens use prefill rounding and only generated tokens use cached rounding");

    trace->phase_calls.clear();
    request.max_new_tokens = 2;
    const auto second = pipeline.generate_ids({4}, request);
    check(second.token_ids == std::vector<int32_t>({4, 2, 2}),
          "second request executes after the previous cached decode");
    phases.clear();
    for (const auto& call : trace->phase_calls) {
        check(call.dtype == trtmc::DType::kInt32 && call.shape == std::vector<int64_t>({1, 1}),
              "reset request keeps the phase input contract");
        phases.push_back(call.value);
    }
    check(phases == std::vector<int32_t>({0, 1}),
          "new request resets to prefill before entering cached decode");
}

void test_pipeline_chat_and_raw_bos_dedup(cudaStream_t stream) {
    auto module = std::make_unique<FakeLfm2Module>(stream, trtmc::DType::kFloat16);
    auto* trace = module.get();
    auto kv = std::make_unique<trtmc::Lfm2KvCache>(1, 8, 2, 4, stream, trtmc::DType::kFloat16);
    auto conv = std::make_unique<trtmc::Lfm2ConvState>(1, 4, 3, stream, trtmc::DType::kFloat16);
    auto state = std::make_unique<trtmc::Lfm2HybridState>(std::move(kv), std::move(conv));
    auto tokenizer = std::make_shared<RecordingTokenizer>();

    trtmc::Lfm2TextGenConfig config;
    config.vocab_size = 6;
    config.id_bos = 1;
    config.id_eos = 5;
    config.id_eos_ids = {5};
    config.chat_template_format = "chatml";
    trtmc::Lfm2TextGenerationPipeline pipeline(std::move(module), std::move(state), config, stream,
                                               tokenizer);

    trtmc::TextGenerationConfig request;
    request.max_new_tokens = 1;
    request.temperature = 0.0F;
    request.use_chat_template = true;
    (void)pipeline.generate("Who are you?", request);
    const std::string expected = "<|startoftext|><|im_start|>user\nWho are you?<|im_end|>\n"
                                 "<|im_start|>assistant\n";
    check(tokenizer->last_text == expected, "pipeline renders exact LFM2 ChatML text");
    check(trace->calls.size() == 2 && trace->calls[0].token == 1 && trace->calls[1].token == 4,
          "chat path removes one duplicated BOS before prefill");

    request.use_chat_template = false;
    (void)pipeline.generate("<|startoftext|><|im_start|>user\nraw tool prompt", request);
    check(trace->calls.size() == 4 && trace->calls[2].token == 1 && trace->calls[3].token == 4,
          "already-rendered raw prompt also removes one duplicated BOS");
}

void test_pipeline_rejects_logits_contract_mismatch(cudaStream_t stream) {
    const auto make_state = [&] {
        auto kv = std::make_unique<trtmc::Lfm2KvCache>(1, 8, 2, 4, stream, trtmc::DType::kFloat16);
        auto conv = std::make_unique<trtmc::Lfm2ConvState>(1, 4, 3, stream, trtmc::DType::kFloat16);
        return std::make_unique<trtmc::Lfm2HybridState>(std::move(kv), std::move(conv));
    };
    const auto make_config = [] {
        trtmc::Lfm2TextGenConfig config;
        config.vocab_size = 6;
        config.id_eos = 5;
        config.id_eos_ids = {5};
        return config;
    };
    trtmc::TextGenerationConfig request;
    request.max_new_tokens = 1;
    request.temperature = 0.0F;

    {
        auto module = std::make_unique<FakeLfm2Module>(stream, trtmc::DType::kFloat16);
        module->set_tensor("logits", {5}, trtmc::DType::kFloat32);
        trtmc::Lfm2TextGenerationPipeline pipeline(std::move(module), make_state(), make_config(),
                                                   stream);
        bool rejected = false;
        try {
            (void)pipeline.generate_ids({1}, request);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected, "pipeline rejects logits shape smaller than configured vocabulary");
    }

    {
        auto module = std::make_unique<FakeLfm2Module>(stream, trtmc::DType::kFloat16);
        module->set_tensor("logits", {6}, trtmc::DType::kFloat16);
        trtmc::Lfm2TextGenerationPipeline pipeline(std::move(module), make_state(), make_config(),
                                                   stream);
        bool rejected = false;
        try {
            (void)pipeline.generate_ids({1}, request);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected, "pipeline rejects non-FP32 logits before device copy");
    }
}

} // namespace

int main(int argc, char** argv) {
    test_exactly_one_length_contract();
    if (failures || (argc == 2 && std::string(argv[1]) == "--cpu-contract"))
        return failures;
    if (!std::filesystem::exists("/dev/nvidiactl")) {
        std::cout << "SKIP: CUDA device is unavailable\n";
        return 77;
    }
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::cerr << "FAIL: unable to create CUDA stream\n";
        return 1;
    }
    test_native_kv_and_double_buffered_conv(stream);
    test_native_contract_rejection_and_capacity(stream);
    test_extent_contract_and_single_token_progression(stream);
    test_batched_prefill_and_decode_handoff(stream);
    test_batched_profile_rejections(stream);
    test_pipeline_uses_extent_and_resets(stream);
    test_pipeline_uses_safe_one_token_progression(stream);
    test_pipeline_selects_decode_phase_and_resets(stream);
    test_pipeline_chat_and_raw_bos_dedup(stream);
    test_pipeline_rejects_logits_contract_mismatch(stream);
    cudaStreamDestroy(stream);
    return failures;
}
