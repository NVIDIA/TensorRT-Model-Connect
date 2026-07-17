/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Source-exact FP32 time projection for the fixed TI2V profile.  This is an
 * independently gated specialization of the same target-local cuBLASLt path
 * used by the two time-embedding linears.
 */

#define WAN22_TIME_LINEAR_NAMESPACE time_projection
#define WAN22_TIME_LINEAR_PLUGIN_CLASS TimeProjectionPlugin
#define WAN22_TIME_LINEAR_CREATOR_CLASS TimeProjectionCreator
#define WAN22_TIME_LINEAR_PLUGIN_NAME "Wan22DitTimeProjection"
#define WAN22_TIME_LINEAR_INSTANCE_NAME "Wan22DitTimeProjection"
#define WAN22_TIME_LINEAR_M 27'280
#define WAN22_TIME_LINEAR_N 18'432
#define WAN22_TIME_LINEAR_K 3'072
#define WAN22_TIME_LINEAR_REGISTRAR plugin_registrar_wan22_dit_time_projection
#define WAN22_TIME_LINEAR_PLAN_INFO_TYPE Wan22DitTimeProjectionPlanInfo
#define WAN22_TIME_LINEAR_PLAN_INFO_FUNCTION trtmc_wan22_dit_time_projection_plan_info

#include "wan22_time_linear1_plugin.cu"
