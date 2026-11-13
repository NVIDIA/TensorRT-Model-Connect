/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen3_5_moe/runtime/edge_llm/contract.h"

#include <functional>
#include <iostream>
#include <limits>
#include <vector>

namespace edge = trtmc::qwen3_5_moe::edge_llm;
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
    const float pixels[] = {-1, 0, 0.5F, 1, 2, 0.1F};
    check(edge::image_bytes(pixels, 1, 2) == std::vector<std::uint8_t>({0, 0, 128, 255, 255, 26}),
          "Normalized HWC conversion differs");
    check(edge::image_bytes(nullptr, 0, 0).empty(), "Absent image must remain text");
    rejects([] { edge::image_bytes(nullptr, 1, 1); });
    rejects([&] { edge::image_bytes(pixels, -1, 1); });
    rejects([&] { edge::image_elements(pixels, INT32_MAX, INT32_MAX); });
    const float invalid[] = {0, 1, std::numeric_limits<float>::quiet_NaN()};
    rejects([&] { edge::image_bytes(invalid, 1, 1); });
    const nlohmann::json marker{
        {"version", 1},
        {"edge_revision", edge::kRevision},
        {"task", "vision_language_generation"},
        {"max_sequence_length", 1024},
        {"max_input_length", 1024},
        {"max_batch_size", 1},
        {"precision", "fp16"},
        {"weight_format", "int4"},
        {"artifacts", nlohmann::json::array()},
        {"eos_token_ids", {248044, 248046}},
        {"source_policy",
         {{"weights", "int4"},
          {"llm", "gptq_experts_original_shared_attention"},
          {"visual", "fp16"},
          {"kv", "fp16"},
          {"group_size", 128}}},
        {"checkpoint_layout", {{"layout", "int4"}, {"tensor_count", 1}, {"shard_count", 1}}},
        {"visual_profile", {{"min", 4}, {"max", 1024}, {"per_image", 512}}}};
    edge::validate_marker(marker, "qwen3_5_moe", "vision_language_generation", "trt");
    auto text_marker = marker;
    text_marker["task"] = "text_generation";
    text_marker["visual_profile"] = nullptr;
    edge::validate_marker(text_marker, "qwen3_5_moe", "text_generation", "trt");
    for (const auto* variant : {"dflash", "dflash_ddtree"}) {
        auto speculative = marker;
        speculative["execution_variant"] = variant;
        speculative["dflash_block_size"] = 16;
        edge::validate_marker(speculative, "qwen3_5_moe", "vision_language_generation", "trt");
        speculative["dflash_block_size"] = 8;
        try {
            edge::validate_marker(speculative, "qwen3_5_moe", "vision_language_generation", "trt");
            check(false, "Invalid DFlash block accepted");
        } catch (const std::exception&) {
        }
    }
    auto mtp = marker;
    mtp["execution_variant"] = "mtp";
    mtp["mtp_draft_steps"] = 3;
    mtp["mtp_verify_size"] = 4;
    mtp["weight_format"] = "nvfp4";
    mtp["target"] = {{"sm", 120}};
    mtp["source_policy"] = {{"weights", "nvfp4"},
                            {"visual", "fp16"},
                            {"kv", "fp8"},
                            {"group_size", 16},
                            {"llm", "mixed_nvfp4_experts_shared_head_fp8_attention_gdn"}};
    mtp["checkpoint_layout"]["layout"] = "nvfp4";
    edge::validate_marker(mtp, "qwen3_5_moe", "vision_language_generation", "trt");
    for (const auto* field : {"mtp_draft_steps", "mtp_verify_size", "max_sequence_length"}) {
        auto invalid = mtp;
        invalid[field] = 0;
        try {
            edge::validate_marker(invalid, "qwen3_5_moe", "vision_language_generation", "trt");
            check(false, "Invalid MTP marker accepted");
        } catch (const std::exception&) {
        }
    }
    auto undocumented = marker;
    undocumented["execution_variant"] = "mtp";
    undocumented["mtp_draft_steps"] = 3;
    undocumented["mtp_verify_size"] = 4;
    try {
        edge::validate_marker(undocumented, "qwen3_5_moe", "vision_language_generation", "trt");
        check(false, "Undocumented GPTQ MTP accepted");
    } catch (const std::exception&) {
    }
    for (const int fault : {0, 1, 2, 3, 4, 5, 6}) {
        auto invalid = marker;
        if (fault == 0)
            invalid["task"] = "text_generation";
        if (fault == 1)
            invalid["weight_format"] = "fp16";
        if (fault == 2)
            invalid["source_policy"]["kv"] = "fp8";
        if (fault == 3)
            invalid["visual_profile"]["per_image"] = 1024;
        if (fault == 4)
            invalid["eos_token_ids"] = {248046};
        if (fault == 5)
            invalid["max_input_length"] = 2048;
        if (fault == 6)
            invalid["checkpoint_layout"]["layout"] = "nvfp4";
        try {
            edge::validate_marker(invalid, "qwen3_5_moe", "vision_language_generation", "trt");
            check(false, "Invalid marker accepted");
        } catch (const std::exception&) {
        }
    }
    return failures == 0 ? 0 : 1;
}
