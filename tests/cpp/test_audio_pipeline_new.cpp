// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-02
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-FAC-01
// Intent:         WhisperPipeline and BarkPipeline construction, transcription,
//                 and audio generation with mock TRT engines; constructor
//                 validation of required modules and embed tables;
//                 SpeechPipeline construction and null-temporal validation;
//                 OmniPipeline construction, generate_audio with inline tokenizer,
//                 and null-thinker validation; MagpiePipeline null-decoder validation;
//                 WhisperPipeline with num_decoder_layers=1 to exercise
//                 whisper_cross_kv_apply.h copy loop;
//                 direct tests of apply_whisper_cross_kv_plan with non-null stats
//                 (covers zero_ops/copy_ops tracking) and invalid plan (buffer_bytes=0)
// Preconditions:  TRT + CUDA GPU available, mock engines built in-process
// Postconditions: Pipelines construct correctly, transcribe/generate_audio
//                 execute without error, constructors reject invalid inputs;
//                 cross-KV copy loop and stats tracking in apply_whisper_cross_kv_plan
//                 exercised; invalid plan returns false with error message
// =============================================================================

// =============================================================================
// Test suite: Audio pipeline unit tests with mock TRT engines
//
// Covers WhisperPipeline and BarkPipeline using constant-output TRT engines:
//
//   WhisperPipeline:
//     - Encoder: mel_features[80,4] -> encoder_output[20] (constant zeros)
//     - Decoder: token_id[1] + attention_mask[9] -> logits[3]=[0.1,0.2,0.9]
//       (argmax=2 = eot_token_id, terminates after first generated token)
//     - Exercises: constructor, transcribe(), run_encoder(), setup_cross_attention(),
//       run_decoder(), run_decoder_step()
//
//   BarkPipeline:
//     - Semantic: attention_mask[513] -> logits[5]=[0.9,0.8,0.7,0.1,0.0]
//       (greedy argmax=0, semantic stage generates 1 token then exits)
//     - Coarse: attention_mask[17] -> logits[12]=all 0.1
//       (after codebook masking selects valid tokens for each codebook)
//     - No codec engine: synthesize_simple_waveform() path exercised
//     - Exercises: constructor, generate_audio(), run_semantic(),
//       run_step_with_embed(), run_step_with_token(), run_coarse(),
//       mask_coarse_logits_for_codebook(), run_fine(), run_codec()
//
//   SpeechPipeline:
//     - Construction test with valid temporal module
//     - Null temporal validation test
//
//   OmniPipeline:
//     - Construction test with valid thinker module
//     - generate_audio with inline tokenizer (token=0 -> thinker breaks)
//     - Null thinker validation test
//
//   MagpiePipeline:
//     - Null decoder validation test
//
//   WhisperPipeline with cross-KV layers:
//     - num_decoder_layers=1 exercises whisper_cross_kv_apply.h copy loop
//
// For full E2E validation with real models, see tests/test_e2e.py.
// =============================================================================

#include "runtime/backend/trt_module_impl.h"
#include "runtime/core/trt_common.h"
#include "runtime/domains/audio/audio_configs.h"
#include "runtime/domains/audio/bark_config.h"
#include "runtime/domains/audio/whisper_config.h"
#include "runtime/domains/audio/whisper_cross_kv_apply.h"
#include "runtime/domains/audio/whisper_cross_kv_plan.h"
#include "runtime/models/bark/pipeline.h"
#include "runtime/models/magpie/pipeline.h"
#include "runtime/models/omni/pipeline.h"
#include "runtime/models/speech/pipeline.h"
#include "runtime/models/whisper/pipeline.h"
#include "trtmc/runtime/kv_cache.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <NvInfer.h>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

static int failures = 0;
static void check(bool c, const char* n) {
    if (!c) {
        std::cerr << "FAIL: " << n << '\n';
        ++failures;
    }
}

static trtmc::TrtLogger g_logger;

// ---------------------------------------------------------------------------
// Inline tokenizer for OmniPipeline tests
// ---------------------------------------------------------------------------
class OmniFixedTokenizer : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string&) const override { return {1, 2}; }
    std::string decode(const std::vector<int32_t>&) const override { return ""; }
    int32_t id_for_token(std::string_view) const override { return 0; }
    std::string token_for_id(int32_t) const override { return ""; }
};

// ---------------------------------------------------------------------------
// Engine builders
// ---------------------------------------------------------------------------

// Encoder: mel_features[mel_bins, mel_len] float32 → encoder_output[enc_out] float32 constant
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>
build_mock_encoder(int32_t mel_bins, int32_t mel_len, int32_t enc_out_size) {
    auto builder = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    if (!builder)
        return nullptr;
    auto network = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* mel_inp = network->addInput("mel_features", nvinfer1::DataType::kFLOAT,
                                      nvinfer1::Dims{2, {mel_bins, mel_len}});

    // Constant encoder output (all zeros)
    std::vector<float> enc_const(enc_out_size, 0.0f);
    auto* const_enc = network->addConstant(
        nvinfer1::Dims{1, {enc_out_size}},
        nvinfer1::Weights{nvinfer1::DataType::kFLOAT, enc_const.data(), enc_out_size});
    if (!const_enc)
        return nullptr;

    auto* enc_out = const_enc->getOutput(0);
    enc_out->setName("encoder_output");
    network->markOutput(*enc_out);

    // Reference mel_inp to avoid TRT optimizer dropping it
    auto* id_mel = network->addIdentity(*mel_inp);
    id_mel->getOutput(0)->setName("_unused_mel");

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan)
        return nullptr;
    auto runtime = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
}

// Decoder/step engine: token_id[1] int32 + attention_mask[mask_size] float32
// → logits[vocab_size] float32 constant
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>
build_mock_step_engine(int32_t mask_size, int32_t vocab_size,
                       const std::vector<float>& const_logits) {
    auto builder = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    if (!builder)
        return nullptr;
    auto network = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* token_inp =
        network->addInput("token_id", nvinfer1::DataType::kINT32, nvinfer1::Dims{1, {1}});
    auto* mask_inp = network->addInput("attention_mask", nvinfer1::DataType::kFLOAT,
                                       nvinfer1::Dims{1, {mask_size}});

    auto* const_w = network->addConstant(
        nvinfer1::Dims{1, {vocab_size}},
        nvinfer1::Weights{nvinfer1::DataType::kFLOAT, const_logits.data(), vocab_size});
    if (!const_w)
        return nullptr;

    auto* out = const_w->getOutput(0);
    out->setName("logits");
    network->markOutput(*out);

    auto* id_tok = network->addIdentity(*token_inp);
    id_tok->getOutput(0)->setName("_unused_token");
    auto* id_mask = network->addIdentity(*mask_inp);
    id_mask->getOutput(0)->setName("_unused_mask");

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan)
        return nullptr;
    auto runtime = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
}

// Attention-only step engine: only attention_mask[mask_size] float32 → logits[vocab_size] constant
// Used for Bark semantic/coarse (token_id is optional, checked with has_input)
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>
build_mock_mask_only_engine(int32_t mask_size, int32_t vocab_size,
                            const std::vector<float>& const_logits) {
    auto builder = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    if (!builder)
        return nullptr;
    auto network = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* mask_inp = network->addInput("attention_mask", nvinfer1::DataType::kFLOAT,
                                       nvinfer1::Dims{1, {mask_size}});

    auto* const_w = network->addConstant(
        nvinfer1::Dims{1, {vocab_size}},
        nvinfer1::Weights{nvinfer1::DataType::kFLOAT, const_logits.data(), vocab_size});
    if (!const_w)
        return nullptr;

    auto* out = const_w->getOutput(0);
    out->setName("logits");
    network->markOutput(*out);

    auto* id_mask = network->addIdentity(*mask_inp);
    id_mask->getOutput(0)->setName("_unused_mask");

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan)
        return nullptr;
    auto runtime = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
}

// ---------------------------------------------------------------------------
// WhisperPipeline tests
// ---------------------------------------------------------------------------

static void test_whisper_transcribe() {
    // Mock encoder: mel_features[80,4] -> encoder_output[20]
    auto enc_engine = build_mock_encoder(80, 4, 20);
    if (!enc_engine) {
        std::cerr << "WARNING: Could not build mock encoder engine, skipping\n";
        return;
    }

    // Mock decoder: token_id[1] + attention_mask[9] -> logits[3]=[0.1,0.2,0.9]
    // argmax=2 = eot_token_id → terminates after first generated token
    const std::vector<float> dec_logits = {0.1f, 0.2f, 0.9f};
    auto dec_engine = build_mock_step_engine(9, 3, dec_logits);
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
    auto cache = std::make_unique<trtmc::KvCache>(0, 8, 0, stream);

    check(encoder->ok(), "whisper encoder ok");
    check(decoder->ok(), "whisper decoder ok");
    check(cache->ok(), "whisper cache ok");

    trtmc::WhisperConfig wcfg;
    wcfg.mel_length = 4;           // expected encoder input = [80, 4]
    wcfg.max_source_positions = 5; // small cross-kv plan
    wcfg.eot_token_id = 2;         // decoder stops on argmax=2

    // MelFilterbank: n_freq_bins=201 (= n_fft/2+1), n_mel_bins=80
    trtmc::MelFilterbank mel_fb;
    mel_fb.n_freq_bins = 201;
    mel_fb.n_mel_bins = 80;
    mel_fb.data.assign(201 * 80, 0.1f);

    trtmc::WhisperPipeline pipeline(std::move(encoder), std::move(decoder), std::move(cache), wcfg,
                                    /*hidden_size=*/4, /*num_decoder_layers=*/0, std::move(mel_fb),
                                    /*mel_n_fft=*/400, /*mel_hop_length=*/160,
                                    /*mel_chunk_length=*/1, /*mel_sampling_rate=*/16000, stream);

    check(std::string(pipeline.pipeline_type()) == "WhisperPipeline", "whisper pipeline_type");

    // Provide 100 zero audio samples — mel extraction pads to 16000 then
    // produces ~101 frames, which run_encoder trims/pads to expected_length=4.
    std::vector<float> audio(100, 0.0f);
    auto result = pipeline.transcribe(audio.data(), static_cast<int32_t>(audio.size()), 5);

    // The decoder produces eot (argmax=2) on the first generation step,
    // so output_ids has 1 element = 2.
    check(result.token_ids.size() == 1, "whisper transcribe produces 1 token");
    check(result.token_ids[0] == 2, "whisper transcribe token is eot=2");

    cudaStreamDestroy(stream);
}

static void test_whisper_constructor_validates_encoder() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto cache = std::make_unique<trtmc::KvCache>(0, 8, 0, stream);
    trtmc::WhisperConfig wcfg;
    wcfg.mel_length = 4;
    trtmc::MelFilterbank mel_fb;
    mel_fb.n_freq_bins = 201;
    mel_fb.n_mel_bins = 80;
    mel_fb.data.assign(201 * 80, 0.1f);

    // Build a valid decoder so that only encoder validation is tested
    const std::vector<float> dec_logits = {0.1f, 0.2f, 0.9f};
    auto dec_engine = build_mock_step_engine(9, 3, dec_logits);
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

// ---------------------------------------------------------------------------
// WhisperPipeline with cross-KV layers (num_decoder_layers=1)
// Exercises the copy loop in whisper_cross_kv_apply.h
// ---------------------------------------------------------------------------

static void test_whisper_with_cross_kv() {
    // Encoder: mel_features[80,4] -> encoder_output[20]
    // 20 floats = 80 bytes = max_source_positions(5) * hidden_size(4) * sizeof(float)
    auto enc_engine = build_mock_encoder(80, 4, 20);
    if (!enc_engine) {
        std::cerr << "WARNING: Could not build encoder for cross-kv test, skipping\n";
        return;
    }

    // Decoder: same as existing test; cross_k_0/cross_v_0 are not actual engine inputs
    // but bind_external silently ignores unknown names
    const std::vector<float> dec_logits = {0.1f, 0.2f, 0.9f};
    auto dec_engine = build_mock_step_engine(9, 3, dec_logits);
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
    // KvCache with num_layers=0 so ok()=true (no layer buffers needed)
    auto cache = std::make_unique<trtmc::KvCache>(0, 8, 0, stream);

    trtmc::WhisperConfig wcfg;
    wcfg.mel_length = 4;
    wcfg.max_source_positions = 5; // cross_kv_bytes = 5*4*4 = 80 bytes
    wcfg.eot_token_id = 2;         // stops after 1 generated token

    trtmc::MelFilterbank mel_fb;
    mel_fb.n_freq_bins = 201;
    mel_fb.n_mel_bins = 80;
    mel_fb.data.assign(201 * 80, 0.1f);

    // num_decoder_layers=1: constructor allocates cross_k_ptrs_[0] and cross_v_ptrs_[0]
    // setup_cross_attention calls apply_whisper_cross_kv_plan with layer_count=1
    trtmc::WhisperPipeline pipeline(std::move(encoder), std::move(decoder), std::move(cache), wcfg,
                                    /*hidden_size=*/4, /*num_decoder_layers=*/1, std::move(mel_fb),
                                    /*mel_n_fft=*/400, /*mel_hop_length=*/160,
                                    /*mel_chunk_length=*/1, /*mel_sampling_rate=*/16000, stream);

    check(std::string(pipeline.pipeline_type()) == "WhisperPipeline",
          "whisper cross-kv: pipeline_type");

    // Transcribe: exercises run_encoder -> setup_cross_attention (copy loop) -> run_decoder
    std::vector<float> audio(100, 0.0f);
    auto result = pipeline.transcribe(audio.data(), static_cast<int32_t>(audio.size()), 5);

    check(result.token_ids.size() == 1, "whisper cross-kv: produces 1 token");
    check(result.token_ids[0] == 2, "whisper cross-kv: token is eot=2");

    cudaStreamDestroy(stream);
}

// ---------------------------------------------------------------------------
// BarkPipeline tests
// ---------------------------------------------------------------------------

static void test_bark_generate_audio() {
    // BarkConfig with small token IDs to keep embed tables tiny.
    // semantic_pad_token=3: with greedy argmax=0 (from constant logits),
    // semantic generates 1 token (token 0) then exits (max_new_tokens=1).
    trtmc::BarkConfig bcfg;
    bcfg.hidden_size = 4;
    bcfg.text_pad_token = 5;
    bcfg.semantic_pad_token = 3;
    bcfg.semantic_infer_token = 4;
    bcfg.semantic_input_vocab = 6;
    bcfg.semantic_output_vocab = 10048; // unused by mock, just informational
    bcfg.semantic_vocab_size = 4;
    bcfg.n_coarse_codebooks = 2;
    bcfg.codebook_size = 4;
    bcfg.coarse_semantic_pad_token = 10;
    bcfg.coarse_infer_token = 9;
    bcfg.max_coarse_input_length = 4;
    bcfg.max_coarse_history = 4;
    bcfg.sliding_window_len = 10;
    bcfg.greedy = true;

    // Semantic cache: max_length=512 handles 256 prefill + 1 infer + 1 gen step
    // Mask size = max_length + 1 = 513
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto sem_cache = std::make_unique<trtmc::KvCache>(0, 512, 0, stream);
    auto coarse_cache = std::make_unique<trtmc::KvCache>(0, 16, 0, stream);

    check(sem_cache->ok(), "bark semantic cache ok");
    check(coarse_cache->ok(), "bark coarse cache ok");

    // Semantic engine: attention_mask[513] -> logits[5]=[0.9,0.8,0.7,0.1,0.0]
    // With greedy and suppress_logits(pad=3), argmax of [0..3] = 0 ≠ pad=3 → token 0
    const std::vector<float> sem_logits = {0.9f, 0.8f, 0.7f, 0.1f, 0.0f};
    auto sem_engine = build_mock_mask_only_engine(513, 5, sem_logits);
    if (!sem_engine) {
        std::cerr << "WARNING: Could not build mock semantic engine, skipping\n";
        cudaStreamDestroy(stream);
        return;
    }

    // Coarse engine: attention_mask[17] -> logits[12]=all 0.1
    // After codebook masking: cb0 selects [4..7] → argmax=4; cb1 → [8..11] → argmax=8
    const std::vector<float> coarse_logits(12, 0.1f);
    auto coarse_engine = build_mock_mask_only_engine(17, 12, coarse_logits);
    if (!coarse_engine) {
        std::cerr << "WARNING: Could not build mock coarse engine, skipping\n";
        cudaStreamDestroy(stream);
        return;
    }

    auto semantic = std::make_unique<trtmc::TrtModuleImpl>(
        sem_engine.get(), sem_engine->createExecutionContext(), stream);
    auto coarse = std::make_unique<trtmc::TrtModuleImpl>(
        coarse_engine.get(), coarse_engine->createExecutionContext(), stream);

    // Embed tables: semantic covers tokens 0..5 (6 rows × 4 floats)
    // coarse covers tokens 0..10 (11 rows × 4 floats)
    std::vector<float> sem_embed(6 * 4, 0.1f);
    std::vector<float> coarse_embed(11 * 4, 0.1f);

    trtmc::BarkPipeline pipeline(std::move(semantic), std::move(coarse), std::move(sem_cache),
                                 std::move(coarse_cache), sem_embed, coarse_embed, bcfg, stream);

    check(std::string(pipeline.pipeline_type()) == "BarkPipeline", "bark pipeline_type");

    // Run generate_audio with max_new_tokens=1 so semantic loop exits quickly.
    // No codec engine: synthesize_simple_waveform produces audio from coarse codes.
    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 1;
    auto out = pipeline.generate_audio("", gen_cfg);

    check(out.num_samples > 0, "bark generate_audio produces samples");
    check(out.sample_rate == 24000, "bark generate_audio sample_rate");

    cudaStreamDestroy(stream);
}

static void test_bark_constructor_validates_semantic() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto coarse_cache = std::make_unique<trtmc::KvCache>(0, 16, 0, stream);
    auto sem_cache = std::make_unique<trtmc::KvCache>(0, 512, 0, stream);

    const std::vector<float> coarse_logits(12, 0.1f);
    auto coarse_engine = build_mock_mask_only_engine(17, 12, coarse_logits);
    if (!coarse_engine) {
        cudaStreamDestroy(stream);
        return;
    }
    auto coarse = std::make_unique<trtmc::TrtModuleImpl>(
        coarse_engine.get(), coarse_engine->createExecutionContext(), stream);

    std::vector<float> sem_embed(24, 0.1f);
    std::vector<float> coarse_embed(44, 0.1f);

    bool threw = false;
    try {
        trtmc::BarkPipeline pipeline(nullptr, std::move(coarse), std::move(sem_cache),
                                     std::move(coarse_cache), sem_embed, coarse_embed,
                                     trtmc::BarkConfig{}, stream);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "bark constructor rejects null semantic module");

    cudaStreamDestroy(stream);
}

static void test_bark_constructor_validates_embed() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const std::vector<float> sem_logits = {0.9f, 0.8f, 0.7f, 0.1f, 0.0f};
    auto sem_engine = build_mock_mask_only_engine(513, 5, sem_logits);
    const std::vector<float> coarse_logits(12, 0.1f);
    auto coarse_engine = build_mock_mask_only_engine(17, 12, coarse_logits);

    if (!sem_engine || !coarse_engine) {
        cudaStreamDestroy(stream);
        return;
    }

    auto semantic = std::make_unique<trtmc::TrtModuleImpl>(
        sem_engine.get(), sem_engine->createExecutionContext(), stream);
    auto coarse = std::make_unique<trtmc::TrtModuleImpl>(
        coarse_engine.get(), coarse_engine->createExecutionContext(), stream);
    auto sem_cache = std::make_unique<trtmc::KvCache>(0, 512, 0, stream);
    auto coarse_cache = std::make_unique<trtmc::KvCache>(0, 16, 0, stream);

    bool threw = false;
    try {
        std::vector<float> empty_embed;
        std::vector<float> coarse_embed(44, 0.1f);
        trtmc::BarkPipeline pipeline(std::move(semantic), std::move(coarse), std::move(sem_cache),
                                     std::move(coarse_cache), empty_embed, coarse_embed,
                                     trtmc::BarkConfig{}, stream);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "bark constructor rejects empty semantic embed");

    cudaStreamDestroy(stream);
}

// ---------------------------------------------------------------------------
// SpeechPipeline tests
// ---------------------------------------------------------------------------

static void test_speech_pipeline_construction() {
    // SpeechPipeline requires temporal_ (non-null) and temporal_cache_ (non-null, ok).
    // All other modules are optional (nullptr is fine).
    const std::vector<float> step_logits = {0.1f, 0.9f, 0.0f};
    auto temporal_engine = build_mock_step_engine(9, 3, step_logits);
    if (!temporal_engine) {
        std::cerr << "WARNING: Could not build temporal engine for SpeechPipeline, skipping\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto temporal = std::make_unique<trtmc::TrtModuleImpl>(
        temporal_engine.get(), temporal_engine->createExecutionContext(), stream);
    auto temporal_cache = std::make_unique<trtmc::KvCache>(0, 8, 0, stream);

    check(temporal->ok(), "speech temporal module ok");
    check(temporal_cache->ok(), "speech temporal cache ok");

    trtmc::SpeechConfig cfg;

    trtmc::SpeechPipeline pipeline(
        /*mimi_encoder=*/nullptr, std::move(temporal), std::move(temporal_cache),
        /*depth_engines=*/{},
        /*depth_cache=*/nullptr,
        /*mimi_decoder=*/nullptr, cfg, stream,
        /*subprocess_runner=*/nullptr, "test-speech");

    check(std::string(pipeline.pipeline_type()) == "SpeechPipeline",
          "SpeechPipeline: pipeline_type");
    check(std::string(pipeline.model_id()) == "test-speech", "SpeechPipeline: model_id");

    cudaStreamDestroy(stream);
}

static void test_speech_validates_temporal() {
    // Null temporal -> constructor throws
    bool threw = false;
    try {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        trtmc::SpeechConfig cfg;
        trtmc::SpeechPipeline p(nullptr, nullptr, nullptr, {}, nullptr, nullptr, cfg, stream,
                                nullptr, "x");
        check(false, "null temporal should throw");
        cudaStreamDestroy(stream);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "speech: null temporal throws");
}

// ---------------------------------------------------------------------------
// OmniPipeline tests
// ---------------------------------------------------------------------------

static void test_omni_pipeline_construction() {
    // Thinker engine: token_id[1] + attention_mask[9] -> logits[4]
    // logits[0]=1.0 (highest) -> omni_argmax returns 0 -> break immediately in run_thinker
    const std::vector<float> thinker_logits = {1.0f, 0.1f, 0.1f, 0.1f};
    auto thinker_engine = build_mock_step_engine(9, 4, thinker_logits);
    if (!thinker_engine) {
        std::cerr << "WARNING: Could not build thinker engine for OmniPipeline, skipping\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto thinker = std::make_unique<trtmc::TrtModuleImpl>(
        thinker_engine.get(), thinker_engine->createExecutionContext(), stream);
    auto thinker_cache = std::make_unique<trtmc::KvCache>(0, 8, 0, stream);

    check(thinker->ok(), "omni thinker module ok");
    check(thinker_cache->ok(), "omni thinker cache ok");

    trtmc::OmniConfig cfg;

    trtmc::OmniPipeline pipeline(std::move(thinker), std::move(thinker_cache),
                                 /*talker=*/nullptr,
                                 /*talker_cache=*/nullptr,
                                 /*code2wav=*/nullptr, cfg, stream,
                                 /*tokenizer=*/nullptr, "test-omni");

    check(std::string(pipeline.pipeline_type()) == "OmniPipeline", "OmniPipeline: pipeline_type");
    check(std::string(pipeline.model_id()) == "test-omni", "OmniPipeline: model_id");

    cudaStreamDestroy(stream);
}

static void test_omni_generate_audio() {
    // Thinker engine returns logits[0]=1.0 -> argmax=0 -> run_thinker breaks after first step
    // text_tokens will be empty -> generate_audio returns early with num_samples=0
    const std::vector<float> thinker_logits = {1.0f, 0.1f, 0.1f, 0.1f};
    auto thinker_engine = build_mock_step_engine(9, 4, thinker_logits);
    if (!thinker_engine) {
        std::cerr << "WARNING: Could not build thinker engine for omni_generate, skipping\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto thinker = std::make_unique<trtmc::TrtModuleImpl>(
        thinker_engine.get(), thinker_engine->createExecutionContext(), stream);
    auto thinker_cache = std::make_unique<trtmc::KvCache>(0, 8, 0, stream);

    trtmc::OmniConfig cfg;

    trtmc::OmniPipeline pipeline(std::move(thinker), std::move(thinker_cache), nullptr, nullptr,
                                 nullptr, cfg, stream, std::make_shared<OmniFixedTokenizer>(),
                                 "test-omni-gen");

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 1;

    // tokenizer->encode("hello") = {1,2} -> run_thinker processes them
    // first generated token = argmax=0 -> break -> empty text_tokens -> early return
    auto result = pipeline.generate_audio("hello", gen_cfg);
    check(result.num_samples == 0,
          "omni generate_audio: no audio when thinker returns empty text tokens");
    check(result.sample_rate == 24000, "omni generate_audio: sample_rate = 24000");

    cudaStreamDestroy(stream);
}

static void test_omni_validates_thinker() {
    // Null thinker -> constructor throws
    bool threw = false;
    try {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        trtmc::OmniConfig cfg;
        trtmc::OmniPipeline p(nullptr, nullptr, nullptr, nullptr, nullptr, cfg, stream, nullptr,
                              "x");
        check(false, "null thinker should throw");
        cudaStreamDestroy(stream);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "omni: null thinker throws");
}

// ---------------------------------------------------------------------------
// MagpiePipeline validation tests
// ---------------------------------------------------------------------------

static void test_magpie_validates_modules() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // null decoder -> constructor throws (checked before encoder)
    bool threw = false;
    try {
        trtmc::MagpieTTSConfig cfg;
        trtmc::MagpiePipeline p(
            /*encoder=*/nullptr,
            /*decoder=*/nullptr, // null decoder -> throws
            /*decoder_cache=*/nullptr,
            /*codec=*/nullptr,
            /*lt_module=*/nullptr,
            /*prefill_module=*/nullptr,
            /*decoder_cache_uncond=*/nullptr,
            /*cross_k=*/{},
            /*cross_v=*/{},
            /*cross_k_uncond=*/{},
            /*cross_v_uncond=*/{}, trtmc::CudaBuffer(0), trtmc::CudaBuffer(0),
            /*audio_embed=*/{},
            /*text_embed=*/{},
            /*context_embed=*/{},
            /*context_lengths=*/{}, cfg, stream, nullptr, "x");
        check(false, "null magpie decoder should throw");
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "magpie: null decoder throws");

    cudaStreamDestroy(stream);
}

static void test_whisper_cross_kv_stats() {
    // Direct test of apply_whisper_cross_kv_plan with non-null WhisperCrossKvApplyStats.
    // Covers: stats->zero_ops increment (lines 46-48) and stats->copy_ops increments
    //         (lines 59-62 for K and lines 69-72 for V, per layer).
    trtmc::WhisperCrossKvPlan plan;
    plan.buffer_bytes = 16; // non-zero: valid plan
    plan.zero_pad_encoder_output = true;
    plan.valid_bytes = 8;
    plan.pad_bytes = 8;

    trtmc::WhisperCrossKvApplyStats stats;
    std::string error;

    bool ok = trtmc::apply_whisper_cross_kv_plan(
        plan,
        /*layer_count=*/2,
        [](std::size_t /*valid*/, std::size_t /*pad*/) -> bool {
            return true; // zero-padding succeeds
        },
        [](std::size_t /*layer*/, trtmc::WhisperCrossKvBufferKind /*kind*/,
           std::size_t /*bytes*/) -> bool {
            return true; // K/V copy succeeds
        },
        error, &stats);

    check(ok, "cross_kv_stats: plan succeeds");
    check(stats.zero_ops == 1, "cross_kv_stats: 1 zero op (zero_pad=true)");
    // 2 layers x (K + V) = 4 copy ops
    check(stats.copy_ops == 4, "cross_kv_stats: 4 copy ops (2 layers x K+V)");
}

static void test_whisper_cross_kv_invalid_plan() {
    // Plan with buffer_bytes=0 covers the early-return error path (lines 35-36)
    trtmc::WhisperCrossKvPlan plan;
    plan.buffer_bytes = 0; // invalid: triggers "invalid whisper cross-kv plan" error

    trtmc::WhisperCrossKvApplyStats stats;
    std::string error;

    bool ok = trtmc::apply_whisper_cross_kv_plan(
        plan, 0, [](std::size_t, std::size_t) { return true; },
        [](std::size_t, trtmc::WhisperCrossKvBufferKind, std::size_t) { return true; }, error,
        &stats);

    check(!ok, "cross_kv_invalid: buffer_bytes=0 returns false");
    check(!error.empty(), "cross_kv_invalid: error message set");
}

int main() {
    test_whisper_transcribe();
    test_whisper_constructor_validates_encoder();
    test_whisper_with_cross_kv();
    test_bark_generate_audio();
    test_bark_constructor_validates_semantic();
    test_bark_constructor_validates_embed();
    test_speech_pipeline_construction();
    test_speech_validates_temporal();
    test_omni_pipeline_construction();
    test_omni_generate_audio();
    test_omni_validates_thinker();
    test_magpie_validates_modules();
    test_whisper_cross_kv_stats();
    test_whisper_cross_kv_invalid_plan();
    if (failures > 0)
        std::cerr << failures << " test(s) FAILED\n";
    return failures;
}
