#pragma once

// WhisperPipeline: encoder-decoder speech-to-text pipeline.
// Uses TrtModule(encoder) + TrtModule(decoder) + IInferenceState.

#include "runtime/core/trt_common.h"
#include "runtime/domains/audio/whisper_config.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/inference_state.h"
#include "trtmc/runtime/kv_cache.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct MelFilterbank;

class WhisperPipeline final : public IPipeline {
  public:
    WhisperPipeline(std::unique_ptr<TrtModule> encoder, std::unique_ptr<TrtModule> decoder,
                    std::unique_ptr<IInferenceState> state, WhisperConfig whisper_config,
                    int32_t hidden_size, int32_t num_decoder_layers, MelFilterbank mel_fb,
                    int32_t mel_n_fft, int32_t mel_hop_length, int32_t mel_chunk_length,
                    int32_t mel_sampling_rate, int32_t mel_win_length, float mel_preemph,
                    bool mel_normalize_per_feature, std::string mel_frontend, cudaStream_t stream,
                    std::shared_ptr<ITokenizer> tokenizer = nullptr, std::string model_id_str = "");

    WhisperPipeline(std::unique_ptr<TrtModule> encoder, std::unique_ptr<TrtModule> decoder,
                    std::unique_ptr<IInferenceState> state, WhisperConfig whisper_config,
                    int32_t hidden_size, int32_t num_decoder_layers, MelFilterbank mel_fb,
                    int32_t mel_n_fft, int32_t mel_hop_length, int32_t mel_chunk_length,
                    int32_t mel_sampling_rate, cudaStream_t stream,
                    std::shared_ptr<ITokenizer> tokenizer = nullptr, std::string model_id_str = "");

    ~WhisperPipeline() override;

    TextResult transcribe(const float* audio_data, int32_t num_samples, int32_t max_new_tokens,
                          int32_t input_sample_rate = 0) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "WhisperPipeline"; }

  private:
    void run_encoder(const float* mel_data, int32_t mel_bins, int32_t mel_length);
    void setup_cross_attention(int32_t actual_enc_seq_len);
    std::vector<int32_t> run_decoder(const std::vector<int32_t>& initial_tokens,
                                     int32_t max_new_tokens);
    void run_decoder_step(int32_t token_id, std::vector<float>& logits);

    std::unique_ptr<TrtModule> encoder_;
    std::unique_ptr<TrtModule> decoder_;
    std::unique_ptr<IInferenceState> state_;
    WhisperConfig whisper_config_;
    int32_t hidden_size_;
    int32_t num_decoder_layers_;
    std::unique_ptr<MelFilterbank> mel_fb_;
    int32_t mel_n_fft_;
    int32_t mel_hop_length_;
    int32_t mel_chunk_length_;
    int32_t mel_sampling_rate_;
    int32_t mel_win_length_;
    float mel_preemph_;
    bool mel_normalize_per_feature_;
    std::string mel_frontend_;
    cudaStream_t stream_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;

    std::vector<std::vector<uint8_t>> cross_k_host_;
    std::vector<void*> cross_k_ptrs_;
    std::vector<void*> cross_v_ptrs_;
    std::size_t cross_kv_bytes_{0};
};

} // namespace trtmc
