/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../support/mock_trt_engines.h"
#include "runtime/backend/trt_module_impl.h"
#include "runtime/models/canary/canary_config.h"
#include "runtime/models/canary/canary_cross_kv_apply.h"
#include "runtime/models/canary/canary_cross_kv_plan.h"
#include "runtime/models/canary/kv_cache.h"
#include "runtime/models/canary/pipeline.h"
#include "runtime/models/canary/plugin_helpers.h"

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

void test_canary_transcribe() {
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
    auto* decoder_ptr = decoder.get();
    auto cache = std::make_unique<trtmc::CanaryKvCache>(0, 8, 0, stream);

    check(encoder->ok(), "canary encoder ok");
    check(decoder->ok(), "canary decoder ok");
    check(cache->ok(), "canary cache ok");

    trtmc::CanaryConfig wcfg;
    wcfg.mel_length = 4;
    wcfg.max_source_positions = 5;
    wcfg.eot_token_id = 2;
    wcfg.decoder_start_token_ids = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    wcfg.supported_languages = {"en"};
    wcfg.language_token_ids = {0};
    wcfg.punctuation_token_id = 0;
    wcfg.no_punctuation_token_id = 1;
    wcfg.timestamp_token_id = 1;
    wcfg.no_timestamp_token_id = 2;

    trtmc::MelFilterbank mel_fb;
    mel_fb.n_freq_bins = 257;
    mel_fb.n_mel_bins = 80;
    mel_fb.data.assign(257 * 80, 0.1F);

    trtmc::CanaryPipeline pipeline(std::move(encoder), std::move(decoder), std::move(cache), wcfg,
                                   4, 0, std::move(mel_fb), 512, 400, 160, 1, 16000, 0.97F, true,
                                   stream);

    check(std::string(pipeline.pipeline_type()) == "CanaryPipeline", "canary pipeline_type");
    check(decoder_ptr->cuda_graph_active(), "canary enables decoder CUDA graph by default");

    std::vector<float> audio(100, 0.0F);
    auto result = pipeline.transcribe(audio.data(), static_cast<int32_t>(audio.size()), 5);
    check(result.token_ids.size() == 1, "canary transcribe produces 1 token");
    check(result.token_ids[0] == 2, "canary transcribe token is eot=2");

    trtmc::TranscriptionConfig request;
    request.max_output_tokens = 5;
    request.input_sample_rate = 16000;
    request.timestamps = true;
    const auto timed =
        pipeline.transcribe(audio.data(), static_cast<int32_t>(audio.size()), request);
    check(timed.segments.size() == 1, "canary timestamp request returns one segment");
    check(timed.segments[0].start_seconds == 0.0 && timed.segments[0].end_seconds > 0.0,
          "canary timestamp segment reports input interval in seconds");

    request.timestamps = false;
    request.beam_size = 2;
    const auto beam =
        pipeline.transcribe(audio.data(), static_cast<int32_t>(audio.size()), request);
    check(beam.token_ids == std::vector<int32_t>({2}),
          "canary beam search runs with branchable inference states");

    cudaStreamDestroy(stream);
}

void test_canary_kv_cache_branch_copy() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    trtmc::CanaryKvCache source(1, 4, 2, stream);
    const std::vector<float> source_k = {1.0F, 2.0F, 3.0F, 4.0F, 9.0F, 9.0F, 9.0F, 9.0F};
    const std::vector<float> source_v = {5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 9.0F, 9.0F, 9.0F};
    check(source.cache_k(0).copy_from_host(source_k.data()), "canary branch source K upload");
    check(source.cache_v(0).copy_from_host(source_v.data()), "canary branch source V upload");
    source.set_position(2);

    auto branch = source.create_empty();
    branch->copy_from(source);
    auto* copied = dynamic_cast<trtmc::CanaryKvCache*>(branch.get());
    check(copied != nullptr, "canary branch retains concrete state type");
    check(branch->position() == 2, "canary branch preserves logical cache position");

    std::vector<float> copied_k(source_k.size(), 0.0F);
    std::vector<float> copied_v(source_v.size(), 0.0F);
    if (copied != nullptr) {
        check(copied->cache_k(0).copy_to_host(copied_k.data()), "canary branch K download");
        check(copied->cache_v(0).copy_to_host(copied_v.data()), "canary branch V download");
    }
    check(std::vector<float>(copied_k.begin(), copied_k.begin() + 4) ==
              std::vector<float>(source_k.begin(), source_k.begin() + 4),
          "canary branch copies valid K rows");
    check(std::vector<float>(copied_v.begin(), copied_v.begin() + 4) ==
              std::vector<float>(source_v.begin(), source_v.begin() + 4),
          "canary branch copies valid V rows");

    cudaStreamDestroy(stream);
}

void test_canary_kv_cache_batch_lane_copy() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    trtmc::CanaryKvCache source(1, 3, 1, stream, trtmc::DType::kFloat32, 4);
    source.set_batch_size(3);
    const std::vector<float> source_k = {
        1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 0.0F, 0.0F, 0.0F,
    };
    const std::vector<float> source_v = {
        11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F, 17.0F, 18.0F, 19.0F, 0.0F, 0.0F, 0.0F,
    };
    check(source.cache_k(0).copy_from_host(source_k.data()), "canary batch source K upload");
    check(source.cache_v(0).copy_from_host(source_v.data()), "canary batch source V upload");
    source.set_position(2);

    auto gathered_state = source.create_empty();
    auto* gathered = dynamic_cast<trtmc::CanaryKvCache*>(gathered_state.get());
    check(gathered != nullptr, "canary batch gather retains concrete state type");
    if (gathered != nullptr) {
        gathered->copy_lanes_from(source, {2, 0, 2, 1});
        check(gathered->batch_size() == 4, "canary batch gather updates active batch");
        check(gathered->position() == 2, "canary batch gather preserves cache position");

        std::vector<float> gathered_k(12, 0.0F);
        std::vector<float> gathered_v(12, 0.0F);
        check(gathered->cache_k(0).copy_to_host(gathered_k.data()),
              "canary batch gathered K download");
        check(gathered->cache_v(0).copy_to_host(gathered_v.data()),
              "canary batch gathered V download");
        check(gathered_k == std::vector<float>({
                                7.0F,
                                8.0F,
                                0.0F,
                                1.0F,
                                2.0F,
                                0.0F,
                                7.0F,
                                8.0F,
                                0.0F,
                                4.0F,
                                5.0F,
                                0.0F,
                            }),
              "canary batch gather reorders K lanes");
        check(gathered_v == std::vector<float>({
                                17.0F,
                                18.0F,
                                0.0F,
                                11.0F,
                                12.0F,
                                0.0F,
                                17.0F,
                                18.0F,
                                0.0F,
                                14.0F,
                                15.0F,
                                0.0F,
                            }),
              "canary batch gather reorders V lanes");
    }

    cudaStreamDestroy(stream);
}

void test_canary_constructor_validates_encoder() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto cache = std::make_unique<trtmc::CanaryKvCache>(0, 8, 0, stream);
    trtmc::CanaryConfig wcfg;
    wcfg.mel_length = 4;
    trtmc::MelFilterbank mel_fb;
    mel_fb.n_freq_bins = 257;
    mel_fb.n_mel_bins = 80;
    mel_fb.data.assign(257 * 80, 0.1F);

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
        trtmc::CanaryPipeline pipeline(nullptr, std::move(decoder), std::move(cache), wcfg, 4, 0,
                                       std::move(mel_fb), 512, 400, 160, 1, 16000, 0.97F, true,
                                       stream);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "canary constructor rejects null encoder");

    cudaStreamDestroy(stream);
}

void test_canary_with_cross_kv() {
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
    check(decoder->has_input("cross_k_0"), "canary cross-kv: decoder has cross_k_0");
    check(decoder->has_input("cross_v_0"), "canary cross-kv: decoder has cross_v_0");
    auto* decoder_ptr = decoder.get();
    auto cache = std::make_unique<trtmc::CanaryKvCache>(0, 8, 0, stream);

    trtmc::CanaryConfig wcfg;
    wcfg.mel_length = 4;
    wcfg.max_source_positions = 5;
    wcfg.eot_token_id = 2;
    wcfg.disable_cuda_graph = true;

    trtmc::MelFilterbank mel_fb;
    mel_fb.n_freq_bins = 257;
    mel_fb.n_mel_bins = 80;
    mel_fb.data.assign(257 * 80, 0.1F);

    trtmc::CanaryPipeline pipeline(std::move(encoder), std::move(decoder), std::move(cache), wcfg,
                                   4, 1, std::move(mel_fb), 512, 400, 160, 1, 16000, 0.97F, true,
                                   stream);

    check(std::string(pipeline.pipeline_type()) == "CanaryPipeline",
          "canary cross-kv: pipeline_type");
    check(!decoder_ptr->cuda_graph_active(), "canary honors CUDA graph opt-out");

    std::vector<float> audio(100, 0.0F);
    auto result = pipeline.transcribe(audio.data(), static_cast<int32_t>(audio.size()), 5);
    check(result.token_ids.size() == 1, "canary cross-kv: produces 1 token");
    check(result.token_ids[0] == 2, "canary cross-kv: token is eot=2");

    cudaStreamDestroy(stream);
}

void test_canary_cross_kv_stats() {
    trtmc::CanaryCrossKvPlan plan;
    plan.buffer_bytes = 16;
    plan.zero_pad_encoder_output = true;
    plan.valid_bytes = 8;
    plan.pad_bytes = 8;

    trtmc::CanaryCrossKvApplyStats stats;
    std::string error;
    bool ok = trtmc::apply_canary_cross_kv_plan(
        plan, 2, [](std::size_t, std::size_t) { return true; },
        [](std::size_t, trtmc::CanaryCrossKvBufferKind, std::size_t) { return true; }, error,
        &stats);

    check(ok, "cross_kv_stats: plan succeeds");
    check(stats.zero_ops == 1, "cross_kv_stats: 1 zero op");
    check(stats.copy_ops == 4, "cross_kv_stats: 4 copy ops");
}

void test_canary_cross_kv_invalid_plan() {
    trtmc::CanaryCrossKvPlan plan;
    plan.buffer_bytes = 0;

    trtmc::CanaryCrossKvApplyStats stats;
    std::string error;
    bool ok = trtmc::apply_canary_cross_kv_plan(
        plan, 0, [](std::size_t, std::size_t) { return true; },
        [](std::size_t, trtmc::CanaryCrossKvBufferKind, std::size_t) { return true; }, error,
        &stats);

    check(!ok, "cross_kv_invalid: buffer_bytes=0 returns false");
    check(!error.empty(), "cross_kv_invalid: error message set");
}

} // namespace

int main() {
    test_canary_transcribe();
    test_canary_kv_cache_branch_copy();
    test_canary_kv_cache_batch_lane_copy();
    test_canary_constructor_validates_encoder();
    test_canary_with_cross_kv();
    test_canary_cross_kv_stats();
    test_canary_cross_kv_invalid_plan();
    if (failures > 0) {
        std::cerr << failures << " canary pipeline test(s) FAILED\n";
    }
    return failures;
}
