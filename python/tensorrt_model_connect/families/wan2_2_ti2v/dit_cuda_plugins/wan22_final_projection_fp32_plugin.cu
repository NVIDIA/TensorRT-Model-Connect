/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Source-exact fixed-shape FP32 output projection.  All target-local strict-
 * FP32 cuBLASLt heuristics passed the saved call92 bitwise gate; production
 * selects the first target-local usable tactic rather than serializing a
 * GB300-specific algorithm into the Thor plan.
 */

#define WAN22_TIME_LINEAR_NAMESPACE final_projection_fp32
#define WAN22_TIME_LINEAR_PLUGIN_CLASS FinalProjectionFp32Plugin
#define WAN22_TIME_LINEAR_CREATOR_CLASS FinalProjectionFp32Creator
#define WAN22_TIME_LINEAR_PLUGIN_NAME "Wan22DitFinalProjectionFp32"
#define WAN22_TIME_LINEAR_INSTANCE_NAME "Wan22DitFinalProjectionFp32"
#define WAN22_TIME_LINEAR_M 27'280
#define WAN22_TIME_LINEAR_N 192
#define WAN22_TIME_LINEAR_K 3'072
#define WAN22_TIME_LINEAR_REGISTRAR plugin_registrar_wan22_dit_final_projection_fp32
#define WAN22_TIME_LINEAR_PLAN_INFO_TYPE Wan22DitFinalProjectionFp32PlanInfo
#define WAN22_TIME_LINEAR_PLAN_INFO_FUNCTION trtmc_wan22_dit_final_projection_fp32_plan_info

#include "wan22_time_linear1_plugin.cu"
