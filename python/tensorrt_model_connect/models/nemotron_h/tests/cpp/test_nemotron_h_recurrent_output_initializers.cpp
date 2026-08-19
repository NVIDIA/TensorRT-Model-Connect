/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Unit tests for recurrent-owned output initializers.

#include "recurrent_generation_plan.h"
#include "recurrent_output_initializers.h"

#include <array>
#include <iostream>
#include <vector>

namespace {

namespace under_test = trtmc::nemotron_h_recurrent;

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

void test_initialize_rwkv_outputs() {
    std::vector<float> logits;
    std::vector<std::vector<float>> attn;
    std::vector<std::vector<float>> ff;
    std::vector<std::vector<float>> num;
    std::vector<std::vector<float>> den;
    std::vector<std::vector<float>> maxv;

    under_test::initialize_rwkv_outputs(3, 11, 5, logits, attn, ff, num, den, maxv);

    check(logits.size() == 11, "rwkv outputs allocate logits");
    check(attn.size() == 3 && attn[0].size() == 5, "rwkv outputs allocate attn");
    check(ff.size() == 3 && ff[1].size() == 5, "rwkv outputs allocate ff");
    check(num.size() == 3 && num[2].size() == 5, "rwkv outputs allocate num");
    check(den.size() == 3 && den[0].size() == 5, "rwkv outputs allocate den");
    check(maxv.size() == 3 && maxv[1].size() == 5, "rwkv outputs allocate max");
}

void test_initialize_mamba_outputs() {
    std::vector<float> logits;
    std::vector<std::vector<float>> conv;
    std::vector<std::vector<float>> ssm;

    under_test::initialize_mamba_outputs(2, 13, 6, 7, logits, conv, ssm);

    check(logits.size() == 13, "mamba outputs allocate logits");
    check(conv.size() == 2 && conv[0].size() == 6, "mamba outputs allocate conv");
    check(ssm.size() == 2 && ssm[1].size() == 7, "mamba outputs allocate ssm");
}

void test_nemotron_h_recurrent_contracts() {
    std::vector<std::vector<float>> a(2, std::vector<float>(4, 1.0F));
    std::vector<std::vector<float>> b(2, std::vector<float>(4, 2.0F));
    std::vector<std::vector<float>> bad_layers(1, std::vector<float>(4, 0.0F));
    std::vector<std::vector<float>> bad_sizes = a;
    bad_sizes[1].resize(3);

    const auto ok_states = std::array<const std::vector<std::vector<float>>*, 2>{&a, &b};
    const auto bad_layer_states =
        std::array<const std::vector<std::vector<float>>*, 2>{&a, &bad_layers};

    check(under_test::validate_state_layer_count(ok_states, 2),
          "nemotron-h contract accepts matching layer count");
    check(!under_test::validate_state_layer_count(bad_layer_states, 2),
          "nemotron-h contract rejects layer count mismatch");

    const auto ok_specs = std::array<under_test::StateTensorView, 2>{
        under_test::StateTensorView{&a, 4}, under_test::StateTensorView{&b, 4}};
    const auto bad_specs = std::array<under_test::StateTensorView, 2>{
        under_test::StateTensorView{&a, 4}, under_test::StateTensorView{&bad_sizes, 4}};

    check(under_test::validate_state_tensor_sizes(ok_specs, 2),
          "nemotron-h contract accepts matching tensor sizes");
    check(!under_test::validate_state_tensor_sizes(bad_specs, 2),
          "nemotron-h contract rejects tensor size mismatch");

    std::vector<std::vector<float>> outputs;
    under_test::initialize_layer_outputs(3, 2, outputs);
    check(outputs.size() == 3, "nemotron-h contract allocates layer outputs");
    check(outputs[0] == std::vector<float>({0.0F, 0.0F}),
          "nemotron-h contract initializes outputs to zero");
}

void test_recurrent_host_transfer_plan() {
    check(!under_test::prefill_step_needs_logits(0, 27),
          "nemotron-h skips unused first-prefill logits");
    check(!under_test::prefill_step_needs_logits(25, 27),
          "nemotron-h skips unused intermediate-prefill logits");
    check(under_test::prefill_step_needs_logits(26, 27),
          "nemotron-h copies final-prefill logits for sampling");

    const std::vector<trtmc::TensorInfo> outputs = {
        {"logits", {128000}, trtmc::DType::kFloat32, false},
        {"present_ssm_0", {36864, 128}, trtmc::DType::kFloat32, false},
        {"present_conv_0", {12352, 4}, trtmc::DType::kFloat32, false},
    };
    const auto summary = under_test::host_visible_output_summary(outputs);
    check(summary.tensor_count == 1, "nemotron-h exposes only logits to the host");
    check(summary.bytes == 128000 * sizeof(float),
          "nemotron-h host-visible bytes exclude recurrent state");
}

} // namespace

int main() {
    test_initialize_rwkv_outputs();
    test_initialize_mamba_outputs();
    test_nemotron_h_recurrent_contracts();
    test_recurrent_host_transfer_plan();

    if (g_failures != 0) {
        std::cerr << g_failures << " recurrent output initializer test(s) failed\n";
        return 1;
    }
    return 0;
}
