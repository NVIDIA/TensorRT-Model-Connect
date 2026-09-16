/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen3_8/runtime/edge_llm/contract.h"

#include <functional>
#include <iostream>
#include <limits>
#include <vector>

namespace edge = trtmc::qwen3_8::edge_llm;
namespace {
int failures = 0;

/// Record a failed behavioral assertion without depending on NDEBUG.
void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << std::endl;
        ++failures;
    }
}

/// Verify unsupported input is rejected by the public argument validation boundary.
void rejects(const std::function<void()>& operation) {
    try {
        operation();
        check(false, "Expected invalid_argument");
    } catch (const std::invalid_argument&) {
    }
}
} // namespace

int main() {
    trtmc::TextGenerationConfig config;
    edge::validate_generation(config);
    config.temperature = 0;
    config.use_chat_template = true;
    config.enable_thinking = false;
    config.text_generation_mode = "autoregressive";
    edge::validate_generation(config);
    using Change = std::function<void(trtmc::TextGenerationConfig&)>;
    const std::vector<Change> unsupported{
        [](auto& c) { c.temperature = -1; },
        [](auto& c) { c.temperature = std::numeric_limits<float>::quiet_NaN(); },
        [](auto& c) { c.top_p = 0; },
        [](auto& c) { c.top_p = 1.01F; },
        [](auto& c) { c.top_p = std::numeric_limits<float>::infinity(); },
        [](auto& c) { c.top_k = -1; },
        [](auto& c) { c.min_p = 0.1F; },
        [](auto& c) { c.seed = 42; },
        [](auto& c) { c.eos_token_id = 9; },
        [](auto& c) { c.repetition_penalty = 1.1F; },
        [](auto& c) { c.lora_adapter_id = "adapter"; },
        [](auto& c) { c.stop_on_boxed_answer = true; },
        [](auto& c) { c.text_generation_mode = "diffusion"; },
        [](auto& c) { c.source_language_token_id = 0; },
        [](auto& c) { c.forced_bos_token_id = 0; },
        [](auto& c) { c.guidance_scale = 0; },
        [](auto& c) { c.cfg_scale = 0; },
        [](auto& c) { c.num_steps = 0; },
        [](auto& c) { c.sde_gamma = 0; },
        [](auto& c) { c.initial_latents = {1}; },
        [](auto& c) { c.condition_latents = {1}; },
        [](auto& c) { c.condition_mask = {1}; },
        [](auto& c) { c.sampling_steps = {1}; },
        [](auto& c) { c.sde_noises = {1}; },
        [](auto& c) { c.block_length = 1; },
        [](auto& c) { c.confidence_threshold = 0.5F; },
    };
    for (const auto& change : unsupported) {
        auto invalid = config;
        change(invalid);
        rejects([&] { edge::validate_generation(invalid); });
    }
    edge::validate_capacity(5, 1024, 1024, 1019);
    rejects([] { edge::validate_capacity(5, 1024, 1024, 1020); });
    rejects([] { edge::validate_capacity(1024, 512, 2048, 1); });
    rejects([] { edge::validate_capacity(0, 1024, 1024, 1); });
    rejects([] { edge::validate_capacity(5, 1024, 1024, 0); });
    rejects([] { edge::validate_capacity(5, 1024, 1024, INT64_MAX); });
    for (const auto* name : {"edge_llm/engine/llm.engine", "edge_llm/checkpoint/model.safetensors",
                             "edge_llm/engine/rank0/config.json"})
        check(edge::safe_artifact_path(name), "Valid artifact path rejected");
    for (const auto* name :
         {"/tmp/engine", "edge_llm/engine/../../outside", "edge_llm/engine/./file",
          "edge_llm/engine//file", "edge_llm/engine/", "other/file", "edge_llm/engine/a\\b"})
        check(!edge::safe_artifact_path(name), "Unsafe artifact path accepted");
    check(!edge::safe_artifact_path(std::string("edge_llm/engine/a\0b", 19)),
          "Embedded null artifact accepted");
    return failures == 0 ? 0 : 1;
}
