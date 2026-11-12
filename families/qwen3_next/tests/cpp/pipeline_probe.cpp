/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen3_next/runtime/diagnostics.h"
#include "families/qwen3_next/runtime/pipeline.h"
#include "trtmc/runtime/trt_backend.h"

#include <array>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace {
using namespace trtmc;

void check(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}

struct Probe {
    int mode;
    int forwards{0};
    int completed{0};
    int forced{0};
    std::array<int32_t, 2> selected{};
};

class Module final : public ITrtModule {
  public:
    Module(bool head, Probe& probe) : head_(head), probe_(probe) {}
    TensorMap forward(const TensorMap& inputs) override {
        if (head_) {
            check(*static_cast<const uint16_t*>(inputs.at("hidden").data) == 0x4000,
                  "head must receive only the accepted hidden output");
            return {{"logits", {logits_.data(), {1, 2}, DType::kFloat32}}};
        }
        ++probe_.forwards;
        check(*static_cast<const uint16_t*>(inputs.at("hidden").data) == 0x3f80,
              "retry must retain the original hidden input");
        check(*static_cast<const uint16_t*>(inputs.at("conv_state").data) == probe_.completed,
              "only the final forward may commit convolution state");
        check(*static_cast<const float*>(inputs.at("recurrent_state").data) == probe_.completed,
              "only the final forward may commit recurrent state");
        const auto force = *static_cast<const int32_t*>(inputs.at("force_routing").data);
        if (force) {
            ++probe_.forced;
            const auto* expected = static_cast<const int32_t*>(inputs.at("forced_experts").data);
            check(expected[0] == 0 && expected[1] == 1, "retry must use the reference IDs");
        }
        const bool accepted = probe_.mode == 0 || (force && probe_.mode == 1);
        selected_ = accepted ? std::array<int32_t, 2>{0, 1} : std::array<int32_t, 2>{2, 3};
        probe_.selected = selected_;
        if (accepted)
            ++probe_.completed;
        hidden_[0] = accepted ? 0x4000 : 0x7777;
        conv_.fill(accepted ? static_cast<uint16_t>(probe_.completed) : 999);
        recurrent_[0] = accepted ? static_cast<float>(probe_.completed) : 999.0F;
        return {{"router_probabilities", {probabilities_.data(), {1, 4}, DType::kFloat32}},
                {"selected_experts", {selected_.data(), {1, 2}, DType::kInt32}},
                {"hidden_out", {hidden_.data(), {1, 1}, DType::kBFloat16}},
                {"present_conv", {conv_.data(), {3, 1}, DType::kBFloat16}},
                {"present_recurrent", {recurrent_.data(), {1, 1, 1}, DType::kFloat32}}};
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
    bool head_;
    Probe& probe_;
    std::array<float, 4> probabilities_{0.4F, 0.3F, 0.2F, 0.1F};
    std::array<int32_t, 2> selected_{};
    std::array<uint16_t, 1> hidden_{0x3f80};
    std::array<uint16_t, 3> conv_{};
    std::array<float, 1> recurrent_{};
    std::array<float, 2> logits_{1.0F, 0.0F};
};

class Backend final : public IBackend {
  public:
    explicit Backend(Probe& probe) : probe_(probe) {}
    std::unique_ptr<ITrtModule> create_module(const void* data, size_t size,
                                              const ModuleCreateOptions&) override {
        const std::string label(static_cast<const char*>(data), size);
        check(label == "layer" || label == "head", "unexpected mock plan");
        return std::make_unique<Module>(label == "head", probe_);
    }
    std::unique_ptr<ITrtModule>
    create_module_prebound(const void*, size_t, const ModuleCreateOptions&,
                           const std::vector<ModuleExternalBinding>&) override {
        throw std::logic_error("unexpected prebound module");
    }
    BackendDualProfileModules create_dual_profile_modules(const void*, size_t,
                                                          const ModuleCreateOptions&) override {
        throw std::logic_error("unexpected dual profile");
    }
    const char* name() const override { return "mock"; }

  private:
    Probe& probe_;
};
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3)
            throw std::invalid_argument("usage: qwen3_next_pipeline_probe BUNDLE MODE");
        const int mode = std::stoi(argv[2]);
        Probe probe{mode};
        Backend backend(probe);
        BundleReader reader(argv[1]);
        std::unique_ptr<ITask> pipeline(trtmc::qwen3_next::create_pipeline({reader, backend}));
        auto* diagnostics = dynamic_cast<trtmc::qwen3_next::Diagnostics*>(pipeline.get());
        check(diagnostics != nullptr, "missing pipeline diagnostics");
        const auto trace = diagnostics->trace_logits("a", 2);
        check(probe.completed == 2, "both token steps must complete");
        check(probe.selected == std::array<int32_t, 2>{0, 1},
              "final expert IDs must match the reference");
        std::cout << nlohmann::json{{"forwards", probe.forwards},
                                    {"forced", probe.forced},
                                    {"routing_retries", trace.routing_retries},
                                    {"generated_ids", trace.generated_ids},
                                    {"selected_experts", probe.selected}}
                  << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
