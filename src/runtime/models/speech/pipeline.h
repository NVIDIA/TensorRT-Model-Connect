#pragma once

// SpeechPipeline: speech-to-speech pipeline with temporal + depth engines.
// Uses TrtModule(mimi_encoder) + TrtModule(temporal) + KvCache +
// TrtModule(depth)[] + TrtModule(mimi_decoder).

#include "runtime/models/speech/speech_config.h"
#include "runtime/models/speech/speech_delay_cache.h"
#include "runtime/models/speech/speech_generation_policy.h"
#include "runtime/models/speech/speech_runtime_plan.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/inference_state.h"
#include "trtmc/runtime/kv_cache.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class ISubprocessRunner;

class SpeechPipeline final : public IPipeline {
  public:
    SpeechPipeline(std::unique_ptr<TrtModule> mimi_encoder, std::unique_ptr<TrtModule> temporal,
                   std::unique_ptr<IInferenceState> temporal_state,
                   std::vector<std::unique_ptr<TrtModule>> depth_engines,
                   std::unique_ptr<IInferenceState> depth_state,
                   std::unique_ptr<TrtModule> mimi_decoder, SpeechConfig config,
                   cudaStream_t stream,
                   std::shared_ptr<ISubprocessRunner> subprocess_runner = nullptr,
                   std::string model_id_str = "");

    ~SpeechPipeline() override;

    AudioResult speak(const float* audio_in, int32_t num_samples, const GenerateConfig& cfg = {},
                      int32_t input_sample_rate = 0) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "SpeechPipeline"; }

  private:
    std::vector<int32_t> run_mimi_encode(const float* samples, int32_t num_samples);

    void run_temporal_embed_step(const float* embed_ptr, int32_t embed_size,
                                 std::vector<float>& logits, std::vector<float>& hidden_out);

    std::vector<int32_t> run_depth(const float* temporal_hidden, int32_t hidden_dim,
                                   int32_t text_token, const int32_t* forced_audio_tokens = nullptr,
                                   const uint8_t* forced_audio_provided = nullptr);

    std::vector<float> run_mimi_decode(const std::vector<int32_t>& codec_tokens,
                                       int32_t num_frames);

    void run_text_prompt();

    bool speak_validate_dual_stream() const;
    void speak_run_generation_loop(const SpeechGenerationSettings& settings,
                                   const SpeechOutputPlan& plan, DelayCacheState& delay_state,
                                   const std::vector<int32_t>& codec_tokens,
                                   std::vector<int32_t>& output_codes, int32_t& frames_collected);
    void speak_postprocess_waveform(std::vector<float>& waveform, int32_t generated_frames) const;

    std::unique_ptr<TrtModule> mimi_encoder_;
    std::unique_ptr<TrtModule> temporal_;
    std::unique_ptr<IInferenceState> temporal_state_;
    std::vector<std::unique_ptr<TrtModule>> depth_engines_;
    std::unique_ptr<IInferenceState> depth_state_;
    std::unique_ptr<TrtModule> mimi_decoder_;

    cudaStream_t stream_;
    SpeechConfig config_;
    std::shared_ptr<ISubprocessRunner> subprocess_runner_;
    std::string model_id_;
    uint64_t rng_state_{0};

    int32_t last_encode_frames_{0};
    int32_t last_encode_codebooks_{0};

    int32_t depth_debug_call_count_{0};
};

} // namespace trtmc
