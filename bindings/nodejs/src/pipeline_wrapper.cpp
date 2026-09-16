/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pipeline_wrapper.h"

#include <trtmc/runtime/family_loader.h>

Napi::Object PipelineWrapper::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "Pipeline",
                                      {
                                          InstanceMethod("generate", &PipelineWrapper::Generate),
                                      });

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("Pipeline", func);
    return exports;
}

PipelineWrapper::PipelineWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<PipelineWrapper>(info) {
    // Constructor logic if instantiated from JS directly.
    // Usually, we instantiate it from C++ via Load().
}

Napi::Value PipelineWrapper::Load(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "String expected for bundle path").ThrowAsJavaScriptException();
        return env.Null();
    }

    std::string path = info[0].As<Napi::String>().Utf8Value();
    std::string runtime_root = "";
    if (info.Length() > 1 && info[1].IsString()) {
        runtime_root = info[1].As<Napi::String>().Utf8Value();
    }

    try {
        // Load the actual TRTMC Task
        std::unique_ptr<trtmc::ITask> task = trtmc::load_task(path, runtime_root);

        Napi::FunctionReference* constructor = env.GetInstanceData<Napi::FunctionReference>();
        Napi::Object obj = constructor->New({});

        // Unwrap and set the native task pointer
        PipelineWrapper* wrapper = Napi::ObjectWrap<PipelineWrapper>::Unwrap(obj);
        wrapper->task_ = std::move(task);

        return obj;
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Null();
    }
}

Napi::Value PipelineWrapper::Generate(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "String expected for prompt").ThrowAsJavaScriptException();
        return env.Null();
    }

    std::string prompt = info[0].As<Napi::String>().Utf8Value();
    trtmc::TextGenerationConfig config;

    // Optional Config argument parsing (basic)
    if (info.Length() > 1 && info[1].IsObject()) {
        Napi::Object configObj = info[1].As<Napi::Object>();
        if (configObj.Has("max_new_tokens")) {
            config.max_new_tokens = configObj.Get("max_new_tokens").As<Napi::Number>().Int32Value();
        }
        if (configObj.Has("temperature")) {
            config.temperature = configObj.Get("temperature").As<Napi::Number>().FloatValue();
        }
        if (configObj.Has("top_k")) {
            config.top_k = configObj.Get("top_k").As<Napi::Number>().Int32Value();
        }
    }

    try {
        // Call the C++ engine
        auto* text_gen = dynamic_cast<trtmc::ITextGeneration*>(task_.get());
        if (!text_gen) {
            throw std::runtime_error("Loaded task does not support text generation");
        }
        trtmc::TextResult result = text_gen->generate(prompt, config);

        // Construct the JS return object
        Napi::Object ret = Napi::Object::New(env);
        ret.Set("text", Napi::String::New(env, result.text));

        // Token IDs array
        Napi::Array tokenIds = Napi::Array::New(env, result.token_ids.size());
        for (size_t i = 0; i < result.token_ids.size(); ++i) {
            tokenIds.Set(i, Napi::Number::New(env, result.token_ids[i]));
        }
        ret.Set("token_ids", tokenIds);

        ret.Set("setup_ms", Napi::Number::New(env, result.setup_ms));
        ret.Set("prefill_ms", Napi::Number::New(env, result.prefill_ms));
        ret.Set("decode_ms", Napi::Number::New(env, result.decode_ms));

        return ret;
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Null();
    }
}
