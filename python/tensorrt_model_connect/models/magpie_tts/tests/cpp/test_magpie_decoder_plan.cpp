/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-06
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-01
// Intent:         Magpie decoder plan: CFG enablement and GPU greedy fallback estimation
// Preconditions:  MagpieConfig with valid decoder parameters
// Postconditions: CFG flag set correctly, fallback estimate used without cross attention
// =============================================================================

#include "magpie_decoder_plan.h"

#include <cstdint>
#include <iostream>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

void test_magpie_decoder_plan_enables_cfg_and_gpu_greedy() {
    trtmc::MagpieTTSConfig cfg;
    cfg.hidden_size = 1024;
    cfg.num_codebooks = 8;
    cfg.codebook_size = 2018;
    cfg.cfg_scale = 1.5F;
    cfg.greedy = true;
    cfg.finished_limit_with_eot = 12;
    cfg.max_source_positions = 256;

    const auto plan = trtmc::make_magpie_decoder_plan(cfg, true, true, true, true, true, true, 20);

    check(plan.hidden == 1024, "magpie decoder plan forwards hidden size");
    check(plan.total_logits == 8 * 2018, "magpie decoder plan computes total logits");
    check(plan.use_cfg, "magpie decoder plan enables cfg when all unconditional resources exist");
    check(plan.use_gpu_greedy,
          "magpie decoder plan enables gpu greedy when kernels and greedy mode are on");
    check(plan.use_cross_attn_tracking,
          "magpie decoder plan enables cross-attention tracking when output exists");
    check(plan.finished_limit == 12 && plan.max_source_positions == 256,
          "magpie decoder plan forwards stop tracking configuration");
    check(plan.text_consumed_threshold == 18,
          "magpie decoder plan derives text consumed threshold at 90 percent");
}

void test_magpie_decoder_plan_uses_fallback_estimate_without_cross_attention() {
    trtmc::MagpieTTSConfig cfg;
    cfg.num_codebooks = 4;
    cfg.codebook_size = 128;
    cfg.cfg_scale = 0.9F;
    cfg.greedy = false;

    const auto plan =
        trtmc::make_magpie_decoder_plan(cfg, false, false, false, true, false, false, 7);

    check(!plan.use_cfg, "magpie decoder plan disables cfg when scale is not above one");
    check(plan.use_gpu_kernels, "magpie decoder plan forwards gpu kernel availability");
    check(!plan.use_gpu_greedy,
          "magpie decoder plan disables gpu greedy when generation is not greedy");
    check(!plan.use_cross_attn_tracking,
          "magpie decoder plan disables cross-attention tracking without output tensor");
    check(plan.estimated_frames == 21,
          "magpie decoder plan falls back to heuristic frame estimate");
    check(trtmc::should_enable_magpie_cfg(cfg, true, true, true) == false,
          "magpie cfg helper requires cfg_scale above one");
    check(trtmc::should_enable_magpie_gpu_greedy(true, false) == false,
          "magpie gpu greedy helper requires greedy mode");
}

} // namespace

int main() {
    test_magpie_decoder_plan_enables_cfg_and_gpu_greedy();
    test_magpie_decoder_plan_uses_fallback_estimate_without_cross_attention();

    if (g_failures != 0) {
        std::cerr << g_failures << " magpie decoder plan test(s) failed\n";
        return 1;
    }
    return 0;
}
