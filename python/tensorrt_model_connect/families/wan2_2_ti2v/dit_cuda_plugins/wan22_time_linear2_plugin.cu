/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Source-exact second FP32 time-embedding linear for the fixed TI2V profile.
 * Keep the target-local cuBLASLt selection and lifecycle identical to the
 * independently qualified first linear; only the fixed K dimension changes.
 */

#define WAN22_TIME_LINEAR_NAMESPACE time_linear2
#define WAN22_TIME_LINEAR_PLUGIN_CLASS TimeLinear2Plugin
#define WAN22_TIME_LINEAR_CREATOR_CLASS TimeLinear2Creator
#define WAN22_TIME_LINEAR_PLUGIN_NAME "Wan22DitTimeLinear2"
#define WAN22_TIME_LINEAR_INSTANCE_NAME "Wan22DitTimeLinear2"
#define WAN22_TIME_LINEAR_M 27'280
#define WAN22_TIME_LINEAR_N 3'072
#define WAN22_TIME_LINEAR_K 3'072
#define WAN22_TIME_LINEAR_REGISTRAR plugin_registrar_wan22_dit_time_linear2
#define WAN22_TIME_LINEAR_PLAN_INFO_TYPE Wan22DitTimeLinear2PlanInfo
#define WAN22_TIME_LINEAR_PLAN_INFO_FUNCTION trtmc_wan22_dit_time_linear2_plan_info

#include "wan22_time_linear1_plugin.cu"
