/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Debug windows (debug_hidden_<i> engine outputs) in the Qwen text pipeline:
// discovery, binding before the decoder CUDA graph is enabled, and the
// trace_debug_windows() seam against stub modules that write known rows.

#include "families/qwen/runtime/debug_windows.h"
#include "families/qwen/runtime/kv_cache.h"
#include "families/qwen/runtime/pipeline.h"

#include <cuda_runtime_api.h>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using trtmc::DType;

constexpr int32_t kCapacity = 16;
constexpr int32_t kHidden = 3;
constexpr int32_t kVocab = 40;
constexpr int32_t kEos = 9;

class Tokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string&) const override {
        return {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    }
    std::string decode(const std::vector<int32_t>&) const override { return {}; }
    int32_t id_for_token(std::string_view) const override { return -1; }
    std::string token_for_id(int32_t) const override { return {}; }
};

// Value a stub engine writes into window `layer` for the token at `position`.
float window_value(int32_t layer, int32_t position, int32_t column) {
    return static_cast<float>(1000 * layer + 10 * position + column);
}

// Native-KV stub: one layer, `rows` tokens per call, optional debug outputs that
// it fills on device through the bound pointer, as a TensorRT engine would.
class Module final : public trtmc::ITrtModule {
  public:
    Module(cudaStream_t stream, int32_t rows, std::vector<int32_t> layers, int32_t eos_position)
        : stream_(stream), rows_(rows), layers_(std::move(layers)), eos_position_(eos_position) {
        add("cache_write_indices", {1}, DType::kInt32, true);
        add("key_value_lengths", {1}, DType::kInt32, true);
        add("position_id", {rows}, DType::kInt32, true);
        add("token_id", {rows}, DType::kInt32, true);
        add("cache_k_0", {1, 1, kCapacity, 2}, DType::kFloat16, true);
        add("cache_v_0", {1, 1, kCapacity, 2}, DType::kFloat16, true);
        add("present_k_0", {1, 1, kCapacity, 2}, DType::kFloat16, false);
        add("present_v_0", {1, 1, kCapacity, 2}, DType::kFloat16, false);
        add("logits", {rows, kVocab}, DType::kFloat32, false);
        for (const int32_t layer : layers_)
            add("debug_hidden_" + std::to_string(layer), {rows, kHidden}, DType::kFloat32, false);
    }

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        const auto& positions = inputs.at("position_id");
        const auto* pos = static_cast<const int32_t*>(positions.data);
        const auto n = static_cast<int32_t>(positions.numel());
        for (const int32_t layer : layers_) {
            std::vector<float> rows(static_cast<std::size_t>(n * kHidden));
            for (int32_t r = 0; r < n; ++r)
                for (int32_t c = 0; c < kHidden; ++c)
                    rows[static_cast<std::size_t>(r * kHidden + c)] =
                        window_value(layer, pos[r], c);
            void* target = device_ptr("debug_hidden_" + std::to_string(layer));
            if (target == nullptr || cudaMemcpy(target, rows.data(), rows.size() * sizeof(float),
                                                cudaMemcpyHostToDevice) != cudaSuccess) {
                throw std::runtime_error("stub debug output is not bound");
            }
        }
        logits_.assign(static_cast<std::size_t>(n * kVocab), -10.0F);
        for (int32_t r = 0; r < n; ++r) {
            const int32_t next = pos[r] + 1 == eos_position_ ? kEos : 20 + pos[r];
            logits_[static_cast<std::size_t>(r * kVocab + next)] = 10.0F;
        }
        ++forwards;
        return {{"logits", trtmc::Tensor{logits_.data(), {n, kVocab}, DType::kFloat32}}};
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap& inputs) override { (void)forward(inputs); }
    void sync() override {}
    cudaStream_t stream() const override { return stream_; }
    void enable_cuda_graph() override { windows_bound_at_graph_enable = debug_binds; }
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return info(true); }
    std::vector<trtmc::TensorInfo> output_info() const override { return info(false); }
    bool has_input(const std::string& name) const override { return has(name, true); }
    bool has_output(const std::string& name) const override { return has(name, false); }
    DType tensor_dtype(const std::string& name) const override { return tensors_.at(name).dtype; }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        const auto it = tensors_.find(name);
        return it == tensors_.end() ? std::vector<int64_t>{} : it->second.shape;
    }
    std::vector<int64_t> input_profile_shape(const std::string& name, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return tensor_shape(name);
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string& name) const override {
        const auto it = bindings_.find(name);
        return it == bindings_.end() ? nullptr : it->second;
    }
    void bind_external(const std::string& name, void* ptr) override {
        if (tensors_.count(name) == 0)
            return;
        bindings_[name] = ptr;
        if (name.rfind("cache_", 0) == 0)
            bindings_["present_" + name.substr(6)] = ptr;
        if (name.rfind("debug_hidden_", 0) == 0)
            ++debug_binds;
    }
    void bind_external(const std::string& name, void* ptr, const std::vector<int64_t>&) override {
        bind_external(name, ptr);
    }
    int32_t input_rank(const std::string& name) const override {
        return has_input(name) ? static_cast<int32_t>(tensor_shape(name).size()) : 0;
    }
    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return stream_ != nullptr; }
    void keep_alive(std::shared_ptr<void>) override {}

    int32_t debug_binds{0};
    int32_t windows_bound_at_graph_enable{-1};
    int32_t forwards{0};

  private:
    struct Entry {
        std::vector<int64_t> shape;
        DType dtype;
        bool input;
    };
    void add(const std::string& name, std::vector<int64_t> shape, DType dtype, bool input) {
        tensors_.emplace(name, Entry{std::move(shape), dtype, input});
    }
    bool has(const std::string& name, bool input) const {
        const auto it = tensors_.find(name);
        return it != tensors_.end() && it->second.input == input;
    }
    std::vector<trtmc::TensorInfo> info(bool input) const {
        std::vector<trtmc::TensorInfo> result;
        for (const auto& [name, entry] : tensors_) {
            if (entry.input == input)
                result.push_back(trtmc::TensorInfo{name, entry.shape, entry.dtype, input});
        }
        return result;
    }

    cudaStream_t stream_;
    int32_t rows_;
    std::vector<int32_t> layers_;
    int32_t eos_position_;
    std::vector<float> logits_;
    std::map<std::string, Entry> tensors_;
    std::map<std::string, void*> bindings_;
};

int failures = 0;
void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL [Qwen debug windows]: " << message << '\n';
        ++failures;
    }
}

template <typename Function>
bool throws(Function function) {
    try {
        function();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

trtmc::QwenTextGenConfig make_config() {
    trtmc::QwenTextGenConfig config;
    config.vocab_size = kVocab;
    config.id_eos = kEos;
    config.prefill_max_length = 4;
    config.num_layers = 1;
    return config;
}

trtmc::TextGenerationConfig make_request(int32_t max_new_tokens) {
    trtmc::TextGenerationConfig request;
    request.max_new_tokens = max_new_tokens;
    request.temperature = 0.0F;
    request.top_k = 1;
    return request;
}

struct Rig {
    Module* prefill{nullptr};
    Module* decoder{nullptr};
    std::unique_ptr<trtmc::QwenTextGenerationPipeline> pipeline;
};

Rig make_rig(cudaStream_t stream, std::vector<int32_t> layers, int32_t eos_position) {
    auto prefill = std::make_unique<Module>(stream, 4, layers, eos_position);
    auto decoder = std::make_unique<Module>(stream, 1, layers, eos_position);
    Rig rig{prefill.get(), decoder.get(), nullptr};
    rig.pipeline = std::make_unique<trtmc::QwenTextGenerationPipeline>(
        std::move(decoder),
        std::make_unique<trtmc::QwenKvCache>(1, kCapacity, 2, stream, DType::kFloat16),
        make_config(), std::make_shared<Tokenizer>(), std::move(prefill));
    return rig;
}

bool rows_match(const trtmc::QwenTextGenerationPipeline::DebugWindowValues& window,
                int32_t first_position, int32_t rows) {
    if (window.shape != std::vector<int64_t>{rows, kHidden} ||
        window.values.size() != static_cast<std::size_t>(rows * kHidden)) {
        return false;
    }
    for (int32_t r = 0; r < rows; ++r)
        for (int32_t c = 0; c < kHidden; ++c)
            if (window.values[static_cast<std::size_t>(r * kHidden + c)] !=
                window_value(window.layer, first_position + r, c)) {
                return false;
            }
    return true;
}

void check_discovery(cudaStream_t stream) {
    // Name parsing and contract checks run before any device allocation.
    Module plain(stream, 4, {}, -1);
    trtmc::QwenDebugWindows none;
    none.bind(plain, 4);
    check(none.empty() && plain.debug_binds == 0, "an engine without debug outputs binds nothing");

    Module windows(stream, 4, {7, 2}, -1);
    trtmc::QwenDebugWindows bound;
    bound.bind(windows, 4);
    check(bound.windows().size() == 2 && bound.windows()[0].layer == 2 &&
              bound.windows()[1].layer == 7,
          "debug_hidden_<i> outputs are discovered and sorted by layer");
    bool owned = true;
    for (const auto& window : bound.windows())
        owned = owned && windows.device_ptr(window.name) == window.buffer.data() &&
                window.buffer.shape() == std::vector<int64_t>{4, kHidden};
    check(owned, "each window is bound to its own maximum-shape buffer");
    check(throws([&] {
              trtmc::QwenDebugWindows short_rows;
              short_rows.bind(windows, 5);
          }),
          "a window with fewer rows than the chunk limit is rejected");
}

void check_trace(cudaStream_t stream) {
    Rig rig = make_rig(stream, {0, 5}, -1);
    check(rig.prefill->debug_binds == 2 && rig.decoder->debug_binds == 2,
          "the pipeline binds every prefill and decoder window");
    check(rig.decoder->windows_bound_at_graph_enable == 2,
          "decoder windows are bound before the CUDA graph is enabled");

    const auto trace = rig.pipeline->trace_debug_windows("prompt", make_request(3));
    check(trace.prompt_ids.size() == 10 && rig.prefill->forwards == 3,
          "the trace prefill runs 4/4/2 chunks");
    check(trace.prefill.size() == 2 && rows_match(trace.prefill[0], 0, 10) &&
              rows_match(trace.prefill[1], 0, 10),
          "prefill windows concatenate the chunk rows in prompt order");
    check(trace.generated_ids == std::vector<int32_t>{29, 30, 31},
          "the trace samples the same greedy tokens as generation");
    bool steps_ok = trace.steps.size() == 3;
    for (std::size_t k = 0; steps_ok && k < trace.steps.size(); ++k) {
        steps_ok = trace.steps[k].size() == 2 &&
                   rows_match(trace.steps[k][0], 10 + static_cast<int32_t>(k), 1) &&
                   rows_match(trace.steps[k][1], 10 + static_cast<int32_t>(k), 1);
    }
    check(steps_ok, "decoder pass k consumes generated_ids[k] at position prompt + k");

    const auto generated = rig.pipeline->generate_ids(trace.prompt_ids, make_request(3));
    check(generated.token_ids ==
              std::vector<int32_t>{10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 29, 30, 31},
          "generation after a trace is unchanged");

    Rig eos = make_rig(stream, {1}, 11);
    const auto stopped = eos.pipeline->trace_debug_windows("prompt", make_request(4));
    check(stopped.generated_ids == std::vector<int32_t>{29, kEos} && stopped.steps.size() == 1,
          "a sampled EOS is recorded without a decoder pass");

    Rig plain = make_rig(stream, {}, -1);
    const auto empty = plain.pipeline->trace_debug_windows("prompt", make_request(3));
    check(empty.prefill.empty() && empty.steps.empty() && plain.prefill->forwards == 0,
          "an engine without debug outputs returns an empty trace without running");
}

} // namespace

int main() {
    if (!std::filesystem::exists("/dev/nvidiactl")) {
        std::cout << "SKIP: CUDA device is unavailable\n";
        return 77;
    }
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess)
        return 1;
    check_discovery(stream);
    check_trace(stream);
    cudaStreamDestroy(stream);
    if (failures == 0)
        std::cout << "All Qwen debug window tests passed.\n";
    return failures == 0 ? 0 : 1;
}
