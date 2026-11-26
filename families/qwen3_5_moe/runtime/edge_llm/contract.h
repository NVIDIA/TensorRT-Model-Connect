/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/task.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::qwen3_5_moe::edge_llm {

#ifdef TRTMC_QWEN35_MOE_EDGE_012
inline constexpr const char* kRelease = "0.12.0";
inline constexpr const char* kRevision = "513dcd18291e587c80928911a480761352f3c2d8";
#else
inline constexpr const char* kRelease = "0.11.0";
inline constexpr const char* kRevision = "95515c2f87fba8982db5a519f9022277667b3cc9";
#endif

/// Reject inconsistent task, source precision and complete-network resource markers.
inline void validate_marker(const nlohmann::json& m, const std::string& family,
                            const std::string& task, const std::string& backend) {
    const auto& revision = m.at("edge_revision");
#ifdef TRTMC_QWEN35_MOE_EDGE_012
    const bool compatible =
        m.at("edge_release") == kRelease && (revision.is_null() || revision == kRevision);
#else
    const bool compatible = revision == kRevision;
#endif
    const bool vision = task == IVisionLanguageGeneration::kTask;
    if (family != "qwen3_5_moe" || backend != "trt" ||
        (!vision && task != ITextGeneration::kTask) || m.at("task") != task ||
        m.at("version") != 1 || !compatible || m.at("max_sequence_length").get<int>() <= 1 ||
        m.at("max_input_length").get<int>() <= 0 ||
        m.at("max_input_length").get<int>() > m.at("max_sequence_length").get<int>() ||
        m.at("max_batch_size") != 1 || m.at("precision") != "fp16" ||
        !m.at("artifacts").is_array() || m.at("eos_token_ids") != nlohmann::json({248044, 248046}))
        throw std::runtime_error("Invalid Qwen3.5-MoE Edge bundle contract");
    const auto variant = m.value("execution_variant", "autoregressive");
    const bool mtp = variant == "mtp";
#ifdef TRTMC_QWEN35_MOE_EDGE_012
    const bool jetspec = variant == "jetspec" && m.value("jet_draft_top_k", 0) == 7;
#else
    constexpr bool jetspec = false;
#endif
    if (mtp ? (m.at("weight_format") != "nvfp4" || m.value("mtp_draft_steps", 0) != 3 ||
               m.value("mtp_verify_size", 0) != 4 || m.at("max_sequence_length").get<int>() <= 4)
            : (variant != "autoregressive" &&
               ((variant != "dflash" && variant != "dflash_ddtree" && !jetspec) ||
                m.value("dflash_block_size", 0) != 16 ||
                m.at("max_sequence_length").get<int>() <= 16)))
        throw std::runtime_error("Invalid Qwen3.5-MoE speculative bundle contract");
    const auto weight = m.at("weight_format").get<std::string>();
    const auto& policy = m.at("source_policy");
    const bool nvfp4 = weight == "nvfp4";
#ifdef TRTMC_QWEN35_MOE_EDGE_012
    const bool original = weight == "none";
    const bool routed_only =
        nvfp4 && policy.at("llm") == "nvfp4_routed_experts_original_shared_head_attention";
#else
    constexpr bool original = false;
    constexpr bool routed_only = false;
#endif
    const auto flow = m.value("edge_build_flow", "direct");
    const bool original_onnx =
        original && (variant == "autoregressive" || variant == "dflash" || jetspec);
    if (flow != "direct" && !(flow == "onnx" && original_onnx))
        throw std::runtime_error("Invalid Qwen3.5-MoE Edge build flow");
    std::string expected_kv = nvfp4 ? "fp8" : "fp16";
#ifdef TRTMC_QWEN35_MOE_EDGE_012
    if (nvfp4) {
        expected_kv = m.at("checkpoint_layout").at("text_kv_dtype").get<std::string>();
        if ((expected_kv != "fp8" && expected_kv != "fp16") ||
            (routed_only && expected_kv != "fp16") || policy.at("sidecar_kv") != "fp8")
            throw std::runtime_error("Invalid NVFP4 checkpoint KV precision");
    }
#endif
    if ((!original && !nvfp4 && weight != "int4") || policy.at("weights") != weight ||
        policy.at("visual") != "fp16" || policy.at("kv") != expected_kv ||
        policy.at("group_size") != (original ? 0
                                    : nvfp4  ? 16
                                             : 128) ||
        policy.at("llm") != (routed_only ? "nvfp4_routed_experts_original_shared_head_attention"
                             : original  ? "original_bf16_experts"
                             : nvfp4     ? "mixed_nvfp4_experts_shared_head_fp8_attention_gdn"
                                         : "gptq_experts_original_shared_attention") ||
        m.at("checkpoint_layout").at("layout") != weight ||
        m.at("checkpoint_layout").at("tensor_count").get<int>() <= 0 ||
        m.at("checkpoint_layout").at("shard_count").get<int>() <= 0)
        throw std::runtime_error("Invalid Qwen3.5-MoE component precision contract");
    if (jetspec && !original && (!nvfp4 || routed_only))
        throw std::runtime_error("Unsupported Qwen3.6-MoE JetSpec target precision");
    if (routed_only && variant != "autoregressive")
        throw std::runtime_error("No speculative contract for routed-only 122B NVFP4");
    if (nvfp4) {
        const int sm = m.at("target").at("sm").get<int>();
#ifdef TRTMC_QWEN35_MOE_EDGE_012
        const bool gb300 = sm == 103;
#else
        constexpr bool gb300 = false;
#endif
        if (sm != 100 && sm != 110 && sm != 120 && sm != 121 && !gb300)
            throw std::runtime_error(
                "NVFP4 and FP8 components require a supported Blackwell target");
    }
    if (vision ? m.at("visual_profile") !=
                     nlohmann::json({{"min", 4}, {"max", 1024}, {"per_image", 512}})
               : !m.at("visual_profile").is_null())
        throw std::runtime_error("Invalid Qwen3.5-MoE visual profile/task contract");
}

/// Return whether an artifact is a normalized file below one of the two Edge roots.
inline bool safe_artifact_path(const std::string& name) {
    if (name.find('\\') != std::string::npos || name.find('\0') != std::string::npos)
        return false;
    const std::filesystem::path path(name);
    if (path.is_absolute() || path.filename().empty())
        return false;
    for (const auto& part : path)
        if (part == "." || part == "..")
            return false;
    return path.generic_string() == name &&
           (name.rfind("edge_llm/engine/", 0) == 0 || name.rfind("edge_llm/checkpoint/", 0) == 0);
}

/// Reject invalid sampling settings and controls with no equivalent Edge request API.
inline void validate_generation(const TextGenerationConfig& c) {
    if (!std::isfinite(c.temperature) || c.temperature < 0 || !std::isfinite(c.top_p) ||
        c.top_p <= 0 || c.top_p > 1 || c.top_k < 0)
        throw std::invalid_argument("Invalid Qwen3.5-MoE Edge sampling parameters");
    if (c.min_p != 0 || c.seed != -1 || c.eos_token_id != -1 || c.repetition_penalty != 1 ||
        !c.lora_adapter_id.empty() || c.stop_on_boxed_answer ||
        (c.text_generation_mode != "auto" && c.text_generation_mode != "autoregressive") ||
        c.source_language_token_id != -1 || c.forced_bos_token_id != -1 || c.guidance_scale != -1 ||
        c.cfg_scale != -1 || c.num_steps != -1 || c.sde_gamma != -1 || !c.initial_latents.empty() ||
        !c.condition_latents.empty() || !c.condition_mask.empty() || !c.sampling_steps.empty() ||
        !c.sde_noises.empty() || c.block_length != 0 || c.confidence_threshold != -1)
        throw std::invalid_argument(
            "Requested generation controls are unsupported by Qwen3.5-MoE Edge");
}

/// Enforce prompt and total capacity without allowing Edge to silently clip generation.
inline void validate_capacity(int prompt_tokens, int input_limit, int capacity,
                              std::int64_t generated_tokens) {
    if (prompt_tokens <= 0 || prompt_tokens > input_limit || generated_tokens <= 0 ||
        generated_tokens > static_cast<std::int64_t>(capacity) - prompt_tokens)
        throw std::invalid_argument(
            "Qwen3.5-MoE Edge prompt and generation exceed bundle capacity");
}

/// Validate external normalized-float HWC images before multiplying dimensions or allocating.
inline std::size_t image_elements(const float* pixels, std::int32_t height, std::int32_t width) {
    if (!pixels && height == 0 && width == 0)
        return 0;
    if (!pixels || height <= 0 || width <= 0)
        throw std::invalid_argument(
            "Qwen3.5-MoE Edge image requires pixels and positive dimensions");
    const auto h = static_cast<std::size_t>(height);
    const auto w = static_cast<std::size_t>(width);
    const auto maximum =
        std::min(std::numeric_limits<std::size_t>::max() / sizeof(float),
                 static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()));
    if (w > maximum / 3 || h > maximum / (w * 3))
        throw std::invalid_argument("Qwen3.5-MoE Edge image dimensions overflow");
    return h * w * 3;
}

/// Map normalized public HWC input to bytes; checkpoint preprocessing belongs to Edge.
inline std::vector<std::uint8_t> image_bytes(const float* pixels, std::int32_t height,
                                             std::int32_t width) {
    const auto size = image_elements(pixels, height, width);
    std::vector<std::uint8_t> result(size);
    for (std::size_t i = 0; i < size; ++i) {
        if (!std::isfinite(pixels[i]))
            throw std::invalid_argument("Qwen3.5-MoE Edge image contains non-finite pixels");
        result[i] =
            static_cast<std::uint8_t>(std::round(std::clamp(pixels[i], 0.0F, 1.0F) * 255.0F));
    }
    return result;
}

} // namespace trtmc::qwen3_5_moe::edge_llm
