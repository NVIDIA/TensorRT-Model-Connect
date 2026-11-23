/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pipeline_wrapper.h"

#include "task_runtime.h"
#include "trtmc/control.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <trtmc/runtime/family_loader.h>

namespace {
Napi::Object text_result(Napi::Env env, std::string_view text,
                         trtmc::Span<const std::int32_t> tokens, double setup_ms, double prefill_ms,
                         double decode_ms) {
    Napi::Object ret = Napi::Object::New(env);
    ret.Set("text", Napi::String::New(env, text.data(), text.size()));
    Napi::Array token_ids = Napi::Array::New(env, tokens.size());
    for (std::size_t i = 0; i < tokens.size(); ++i)
        token_ids.Set(i, Napi::Number::New(env, tokens[i]));
    ret.Set("token_ids", token_ids);
    ret.Set("setup_ms", Napi::Number::New(env, setup_ms));
    ret.Set("prefill_ms", Napi::Number::New(env, prefill_ms));
    ret.Set("decode_ms", Napi::Number::New(env, decode_ms));
    return ret;
}
} // namespace

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
        std::unique_ptr<trtmc::ITask> task;
        std::optional<trtmc::TextContinuation> continuation;
        if (trtmc::app::uses_existing_task_runtime(trtmc::Bundle::open(path).info().task)) {
            task = trtmc::load_task(path, runtime_root);
        } else {
            trtmc::LoadOptions options;
            options.runtime_root = runtime_root;
            continuation = trtmc::Model::load(path, options).task<trtmc::TextContinuation>();
        }

        Napi::FunctionReference* constructor = env.GetInstanceData<Napi::FunctionReference>();
        Napi::Object obj = constructor->New({});

        PipelineWrapper* wrapper = Napi::ObjectWrap<PipelineWrapper>::Unwrap(obj);
        wrapper->task_ = std::move(task);
        wrapper->continuation_ = std::move(continuation);
        if (wrapper->continuation_)
            wrapper->config_fields_ = wrapper->continuation_->config_fields();

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
    try {
        if (continuation_) {
            trtmc::Config controls;
            if (info.Length() > 1 && info[1].IsObject()) {
                const auto options = info[1].As<Napi::Object>();
                for (const char* name : {"max_new_tokens", "temperature", "top_k"}) {
                    if (!options.Has(name))
                        continue;
                    const auto field =
                        std::find_if(config_fields_.begin(), config_fields_.end(),
                                     [&](const auto& value) { return value.name == name; });
                    if (field == config_fields_.end())
                        throw std::invalid_argument(std::string("Task does not declare config: ") +
                                                    name);
                    const auto value = options.Get(name);
                    if (!value.IsNumber())
                        throw std::invalid_argument(std::string(name) + " must be a number");
                    const auto number = value.As<Napi::Number>().DoubleValue();
                    if (!std::isfinite(number))
                        throw std::invalid_argument(std::string(name) + " must be finite");
                    if (std::string_view(name) == "temperature") {
                        controls.add(name, number);
                    } else {
                        if (std::floor(number) != number || std::abs(number) > 9007199254740991.0)
                            throw std::invalid_argument(std::string(name) +
                                                        " must be a safe integer");
                        controls.add(name, static_cast<std::int64_t>(number));
                    }
                }
            }
            const auto result =
                continuation_->run(trtmc::TextContinuationRequest{prompt}, controls);
            return text_result(env, result.text(), result.token_ids(), result.setup_ms(),
                               result.prefill_ms(), result.decode_ms());
        }
        trtmc::TextGenerationConfig config;
        if (info.Length() > 1 && info[1].IsObject()) {
            const auto options = info[1].As<Napi::Object>();
            if (options.Has("max_new_tokens"))
                config.max_new_tokens =
                    options.Get("max_new_tokens").As<Napi::Number>().Int32Value();
            if (options.Has("temperature"))
                config.temperature = options.Get("temperature").As<Napi::Number>().FloatValue();
            if (options.Has("top_k"))
                config.top_k = options.Get("top_k").As<Napi::Number>().Int32Value();
        }
        auto* text_gen = dynamic_cast<trtmc::ITextGeneration*>(task_.get());
        if (!text_gen) {
            throw std::runtime_error("Loaded task does not support text generation");
        }
        trtmc::TextResult result = text_gen->generate(prompt, config);

        return text_result(env, result.text, {result.token_ids.data(), result.token_ids.size()},
                           result.setup_ms, result.prefill_ms, result.decode_ms);
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Null();
    }
}
