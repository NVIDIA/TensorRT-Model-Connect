/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pipeline_wrapper.h"

#include <napi.h>

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
    PipelineWrapper::Init(env, exports);
    exports.Set("load", Napi::Function::New(env, PipelineWrapper::Load));
    return exports;
}

NODE_API_MODULE(trtmc_node, InitAll)
