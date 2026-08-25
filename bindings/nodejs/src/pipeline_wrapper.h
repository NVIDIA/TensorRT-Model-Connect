/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <napi.h>
#include <string>
#include <trtmc/pipeline.h>

class PipelineWrapper : public Napi::ObjectWrap<PipelineWrapper> {
  public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    static Napi::Value Load(const Napi::CallbackInfo& info);
    PipelineWrapper(const Napi::CallbackInfo& info);

  private:
    Napi::Value Generate(const Napi::CallbackInfo& info);

    std::unique_ptr<trtmc::Pipeline> pipeline_;
};
