/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../support/mock_trt_engines.h"
#include "runtime/backend/trt_module_impl.h"
#include "runtime/models/whisper/kv_cache.h"
#include "runtime/models/whisper/pipeline.h"
#include "runtime/models/whisper/plugin_helpers.h"
#include "runtime/models/whisper/whisper_config.h"
#include "runtime/models/whisper/whisper_cross_kv_apply.h"
#include "runtime/models/whisper/whisper_cross_kv_plan.h"

#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void test_whisper_transcribe() {
    auto enc_engine = trtmc::test::build_mock_encoder(80, 4, 20);
    if (!enc_engine) {
        std::cerr << "WARNING: Could not build mock encoder engine, skipping\n";
        return;
    }
    const std::vector<float> dec_logits = {0.1F, 0.2F, 0.9F};
    auto dec_engine = trtmc::test::build_mock_step_engine(9, 3, dec_logits);
    if (!dec_engine) {
        std::cerr << "WARNING: Could not build mock decoder engine, skipping\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto encoder = std::make_unique<trtmc::TrtModuleImpl>(
        enc_engine.get(), enc_engine->createExecutionContext(), stream);
    auto decoder = std::make_unique<trtmc::TrtModuleImpl>(
        dec_engine.get(), dec_engine->createExecutionContext(), stream);
    auto cache = std::make_unique<trtmc::WhisperKvCache>(0, 8, 0, stream);

    check(encoder->ok(), "whisper encoder ok");
    check(decoder->ok(), "whisper decoder ok");
    check(cache->ok(), "whisper cache ok");

    trtmc::WhisperConfig wcfg;
    wcfg.mel_length = 4;
    wcfg.max_source_positions = 5;
    wcfg.eot_token_id = 2;

    trtmc::MelFilterbank mel_fb;
    mel_fb.n_freq_bins = 201;
    mel_fb.n_mel_bins = 80;
    mel_fb.data.assign(201 * 80, 0.1F);

    trtmc::WhisperPipeline pipeline(std::move(encoder), std::move(decoder), std::move(cache), wcfg,
                                    4, 0, std::move(mel_fb), 400, 160, 1, 16000, stream);

    check(std::string(pipeline.pipeline_type()) == "WhisperPipeline", "whisper pipeline_type");

    std::vector<float> audio(100, 0.0F);
    auto result = pipeline.transcribe(audio.data(), static_cast<int32_t>(audio.size()), 5);
    check(result.token_ids.size() == 1, "whisper transcribe produces 1 token");
    check(result.token_ids[0] == 2, "whisper transcribe token is eot=2");

    cudaStreamDestroy(stream);
}

void test_whisper_constructor_validates_encoder() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto cache = std::make_unique<trtmc::WhisperKvCache>(0, 8, 0, stream);
    trtmc::WhisperConfig wcfg;
    wcfg.mel_length = 4;
    trtmc::MelFilterbank mel_fb;
    mel_fb.n_freq_bins = 201;
    mel_fb.n_mel_bins = 80;
    mel_fb.data.assign(201 * 80, 0.1F);

    const std::vector<float> dec_logits = {0.1F, 0.2F, 0.9F};
    auto dec_engine = trtmc::test::build_mock_step_engine(9, 3, dec_logits);
    if (!dec_engine) {
        cudaStreamDestroy(stream);
        return;
    }
    auto decoder = std::make_unique<trtmc::TrtModuleImpl>(
        dec_engine.get(), dec_engine->createExecutionContext(), stream);

    bool threw = false;
    try {
        trtmc::WhisperPipeline pipeline(nullptr, std::move(decoder), std::move(cache), wcfg, 4, 0,
                                        std::move(mel_fb), 400, 160, 1, 16000, stream);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "whisper constructor rejects null encoder");

    cudaStreamDestroy(stream);
}

void test_whisper_with_cross_kv() {
    auto enc_engine = trtmc::test::build_mock_encoder(80, 4, 20);
    if (!enc_engine) {
        std::cerr << "WARNING: Could not build encoder for cross-kv test, skipping\n";
        return;
    }
    const std::vector<float> dec_logits = {0.1F, 0.2F, 0.9F};
    auto dec_engine = trtmc::test::build_mock_step_engine(9, 3, dec_logits, 1, 5, 4);
    if (!dec_engine) {
        std::cerr << "WARNING: Could not build decoder for cross-kv test, skipping\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto encoder = std::make_unique<trtmc::TrtModuleImpl>(
        enc_engine.get(), enc_engine->createExecutionContext(), stream);
    auto decoder = std::make_unique<trtmc::TrtModuleImpl>(
        dec_engine.get(), dec_engine->createExecutionContext(), stream);
    check(decoder->has_input("cross_k_0"), "whisper cross-kv: decoder has cross_k_0");
    check(decoder->has_input("cross_v_0"), "whisper cross-kv: decoder has cross_v_0");
    auto cache = std::make_unique<trtmc::WhisperKvCache>(0, 8, 0, stream);

    trtmc::WhisperConfig wcfg;
    wcfg.mel_length = 4;
    wcfg.max_source_positions = 5;
    wcfg.eot_token_id = 2;

    trtmc::MelFilterbank mel_fb;
    mel_fb.n_freq_bins = 201;
    mel_fb.n_mel_bins = 80;
    mel_fb.data.assign(201 * 80, 0.1F);

    trtmc::WhisperPipeline pipeline(std::move(encoder), std::move(decoder), std::move(cache), wcfg,
                                    4, 1, std::move(mel_fb), 400, 160, 1, 16000, stream);

    check(std::string(pipeline.pipeline_type()) == "WhisperPipeline",
          "whisper cross-kv: pipeline_type");

    std::vector<float> audio(100, 0.0F);
    auto result = pipeline.transcribe(audio.data(), static_cast<int32_t>(audio.size()), 5);
    check(result.token_ids.size() == 1, "whisper cross-kv: produces 1 token");
    check(result.token_ids[0] == 2, "whisper cross-kv: token is eot=2");

    cudaStreamDestroy(stream);
}

void test_whisper_cross_kv_stats() {
    trtmc::WhisperCrossKvPlan plan;
    plan.buffer_bytes = 16;
    plan.zero_pad_encoder_output = true;
    plan.valid_bytes = 8;
    plan.pad_bytes = 8;

    trtmc::WhisperCrossKvApplyStats stats;
    std::string error;
    bool ok = trtmc::apply_whisper_cross_kv_plan(
        plan, 2, [](std::size_t, std::size_t) { return true; },
        [](std::size_t, trtmc::WhisperCrossKvBufferKind, std::size_t) { return true; }, error,
        &stats);

    check(ok, "cross_kv_stats: plan succeeds");
    check(stats.zero_ops == 1, "cross_kv_stats: 1 zero op");
    check(stats.copy_ops == 4, "cross_kv_stats: 4 copy ops");
}

void test_whisper_cross_kv_invalid_plan() {
    trtmc::WhisperCrossKvPlan plan;
    plan.buffer_bytes = 0;

    trtmc::WhisperCrossKvApplyStats stats;
    std::string error;
    bool ok = trtmc::apply_whisper_cross_kv_plan(
        plan, 0, [](std::size_t, std::size_t) { return true; },
        [](std::size_t, trtmc::WhisperCrossKvBufferKind, std::size_t) { return true; }, error,
        &stats);

    check(!ok, "cross_kv_invalid: buffer_bytes=0 returns false");
    check(!error.empty(), "cross_kv_invalid: error message set");
}

} // namespace

int main() {
    test_whisper_transcribe();
    test_whisper_constructor_validates_encoder();
    test_whisper_with_cross_kv();
    test_whisper_cross_kv_stats();
    test_whisper_cross_kv_invalid_plan();
    if (failures > 0) {
        std::cerr << failures << " whisper pipeline test(s) FAILED\n";
    }
    return failures;
}
