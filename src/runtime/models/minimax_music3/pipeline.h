/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// MiniMax-Music3: text (a caption plus lyrics) to stereo music.
//
// Five engines run per request, in this order:
//
//   language_model    autoregressive over audio frames; emits the first of
//                     eight codebook streams plus the hidden state that the
//                     depth decoder conditions on
//   depth_decoder     the remaining seven residual codebooks for that frame
//   condition_encoder the eight streams -> a latent-rate conditioning signal
//   dit               flow matching, DEFAULT_INFERENCE_STEPS per window, run
//                     over overlapping windows of the conditioning
//   vocoder           latents -> stereo waveform
//
// The window plan, the crop widths and the output rate are not in the
// checkpoint. They are carried in the bundle config by
// engines.bundle_config_overrides() and every one of them was measured
// against a recorded generation before being written down.

#include "trtmc/pipeline.h"
#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

// The facts the runtime cannot re-derive from the checkpoint. Field names
// match the keys engines.bundle_config_overrides() writes.
struct MinimaxMusic3Config {
    int32_t sampling_rate{44100};
    int32_t output_channels{2};
    float frame_rate_hz{25.0F};
    int32_t latent_hop_length{512};
    // Latent frames per autoregressive frame, and the latent length of
    // one window. Both are carried by the bundle rather than rebuilt
    // here: the ratio truncates, and a rebuilt one drifts by a frame.
    float latent_resample_ratio{3.4453125F};
    int32_t chunk_latent_length{689};
    int32_t chunk_frames{200};
    int32_t chunk_hop{100};
    int32_t crop_left_latent{86};
    int32_t crop_right_latent{258};
    int32_t default_inference_steps{30};
    int32_t max_audio_frames{9000};
    int32_t guidance_branches{2};
    int32_t num_codebooks{8};
    int32_t num_residual_codebooks{7};
    int32_t audio_vocab_size{2048};
    int32_t latent_channels{128};
    int32_t condition_dim{1024};
    // One frame's hidden state as the condition encoder reads it:
    // eight streams wide, not the diffusion transformer's condition.
    int32_t frame_hidden_width{32768};
    int32_t condition_streams{8};
    int32_t language_model_hidden_size{4096};
    // Width of one cache row: num_key_value_heads * head_dim.
    int32_t language_model_kv_width{1024};
    // Guidance: the reference runs two branches at this scale.
    float guidance_scale{1.7F};
    int32_t language_model_vocab_size{200000};
    int32_t language_model_layers{28};

    // From the request's music_minimax_music3 namespace.
    std::string caption;
    int32_t max_frames{9000};
    int64_t seed{-1};
    // The checkpoint's draw. Kept here rather than rewriting the request's
    // GenerateConfig, whose top_k default of 1 is indistinguishable from an
    // explicit request for greedy decoding.
    int32_t top_k{50};
    float temperature{1.0F};
};

// The five engines a bundle carries. Ownership is the pipeline's.
struct MinimaxMusic3Engines {
    std::unique_ptr<ITrtModule> language_model;
    std::unique_ptr<ITrtModule> depth_decoder;
    std::unique_ptr<ITrtModule> condition_encoder;
    std::unique_ptr<ITrtModule> dit;
    std::unique_ptr<ITrtModule> vocoder;
};

class MinimaxMusic3TextToMusicPipeline final : public IPipeline {
  public:
    MinimaxMusic3TextToMusicPipeline(MinimaxMusic3Engines engines, MinimaxMusic3Config config,
                                     std::shared_ptr<ITokenizer> tokenizer, std::string model_id);

    ~MinimaxMusic3TextToMusicPipeline() override;

    AudioResult generate_audio(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "MinimaxMusic3TextToMusicPipeline"; }

    // How many autoregressive frames one window advances by. Exposed because
    // the window plan is the part a reader most often wants to check against
    // pipeline_spec.py.
    int32_t chunk_hop() const { return config_.chunk_hop; }

  private:
    //: One emitted frame: where it belongs and what it carries.
    struct EmittedFrame {
        int32_t index{0};
        int32_t total{0};
        int32_t semantic{0};
        const float* hidden{nullptr};
    };

    // Assemble the prompt the checkpoint expects and tokenize it.
    std::vector<int32_t> tokenize_prompt(const std::string& lyrics) const;

    //: What the prompt pass leaves behind, for both guidance branches.
    struct BranchState {
        const float* conditional_hidden{nullptr};
        const float* unconditional_hidden{nullptr};
        const float* conditional_logits{nullptr};
        const float* unconditional_logits{nullptr};
    };

    // Run the prompt through both branches. Returns the next free position.
    int32_t prime_caches(const std::vector<int32_t>& prompt_ids,
                         const std::vector<int32_t>& unconditional_ids, BranchState& state);

    // The classifier-free counterpart of a tokenised prompt.
    static std::vector<int32_t> build_unconditional_ids(const std::vector<int32_t>& prompt_ids);

    // Print what the deterministic prompt pass produced, under TRTMC_MM3_DEBUG.
    void report_prompt_pass(const float* hidden, const float* logits) const;

    // Store one frame's eight conditioning streams and its eight codes.
    void record_frame(const EmittedFrame& frame, const std::vector<int32_t>& residual,
                      std::vector<float>& hidden, std::vector<int32_t>& codes) const;

    // Blend a window's head toward its neighbour's trailing latents. At sigma 1
    // the neighbour's values win outright, which is how the seam is settled
    // once the window is denoised.
    void blend_overlap(std::vector<float>& latents, const std::vector<float>& noise,
                       const std::vector<float>& neighbour, int32_t latent_length,
                       std::size_t carry, float sigma) const;

    // Combine the conditional velocity with an unconditional one, in place.
    void guide_velocity(ITrtModule& dit, TensorMap& inputs, const std::vector<float>& condition,
                        int32_t latent_length, std::vector<float>& guided) const;

    // The frame states the conditioning stage reads: normally the
    // autoregressive stage's, or, under TRTMC_MM3_FRAME_HIDDEN, ones recorded
    // elsewhere so a fault in how codes are drawn can be told apart from a
    // fault in what is done with them.
    // `emitted` returns the frames the model actually produced, which is fewer
    // than `frames` when it draws the audio-end token. Everything downstream
    // must use that count or it denoises and emits the zero-filled tail.
    std::vector<float> collect_frame_states(const std::vector<int32_t>& prompt_ids, int32_t frames,
                                            const GenerateConfig& cfg, int32_t& emitted);

    // The slice of a denoised window that the next one blends its head toward.
    std::vector<float> carry_overlap(const std::vector<float>& latents,
                                     int32_t latent_length) const;

    // Append one window's samples, dropping the crops the seams duplicate.
    void append_window(const std::vector<float>& chunk, std::size_t window,
                       std::size_t window_count, std::vector<float>& samples) const;

    // The parameters the autoregressive draw runs with, defaulted from the
    // checkpoint where the request left them unset.
    GenerateConfig sampling_config(const GenerateConfig& cfg) const;

    // Print the drawn semantic codes under TRTMC_MM3_DEBUG. A degenerate loop
    // shows up here before it shows up in the audio.
    static void report_semantic_codes(const std::vector<int32_t>& codes, int32_t emitted);

    // Report the first and last denoising step's velocity and latents.
    static void report_denoise_step(std::size_t index, std::size_t sigma_count,
                                    const std::vector<float>& guided,
                                    const std::vector<float>& latents);

    // Autoregressive stage. Returns num_codebooks streams of `frames` codes,
    // laid out codebook-major so a window is a contiguous slice per stream,
    // and fills `hidden` with the frame hidden states the condition encoder
    // reads. The language model emits codebook 0 and the depth decoder the
    // seven residual codebooks for the same frame, so one step drives both.
    std::vector<int32_t> generate_codes(const std::vector<int32_t>& prompt_ids, int32_t frames,
                                        const GenerateConfig& cfg, std::vector<float>& hidden,
                                        int32_t& emitted);

    //: The autoregressive stage runs two sequences: the assembled prompt and
    //: its classifier-free counterpart. Each carries its own key/value cache.
    static constexpr int32_t kBranches = 2;
    static constexpr int32_t kConditional = 0;
    static constexpr int32_t kUnconditional = 1;

    // One decode step of one branch. Returns that branch's logits and writes
    // its hidden state, advancing only that branch's cache.
    // `frame_embed` replaces the token id once generation starts: a frame is
    // eight codes and an id carries one. Null for the prompt.
    const float* decode_step(int32_t branch, int32_t token_id, int32_t position,
                             const float** hidden_out, const float* frame_embed = nullptr);

    //: One frame's depth step. The two branches' hidden states and the frame's
    //: semantic code go in; the seven residual codes and the seven hidden
    //: states that follow the language model's come out.
    struct DepthStep {
        const float* conditional_hidden{nullptr};
        const float* unconditional_hidden{nullptr};
        int32_t semantic_code{0};
        int32_t* codes_out{nullptr};
        float* hidden_out{nullptr};
    };

    // The seven residual codebooks for one frame, given its hidden state.
    void sample_residual_codes(const DepthStep& step, const GenerateConfig& cfg,
                               uint64_t& rng_state);

    // One window of frame hidden states -> its conditioning signal.
    std::vector<float> encode_condition(const std::vector<float>& frame_hidden,
                                        int32_t frame_offset, int32_t frames);

    // Flow matching over one window. Returns latent_channels * latent_length.
    // `previous` is the trailing slice of the window before, empty for the
    // first: its frames are blended into the head of this window at every step
    // so neighbouring windows share their boundary.
    std::vector<float> denoise_window(const std::vector<float>& condition, int32_t latent_length,
                                      int32_t steps, uint64_t seed,
                                      const std::vector<float>& previous);

    //: Half the overlap. The reference carries the window's [L - 2h, L - h)
    //: slice forward, which is 172 latent frames of the 344 that neighbouring
    //: windows share.
    static constexpr int32_t kOverlapCarry = 172;

    // Latents -> interleaved stereo samples.
    std::vector<float> decode_waveform(const std::vector<float>& latents, int32_t latent_length);

    int32_t latent_length_for(int32_t frames) const;

    // Cache buffers, one pair per language model layer. They are owned here
    // rather than by the engine so a present_* output can become the next
    // step's cache_* input without a copy.
    void bind_cache();
    void bind_branch(int32_t branch);
    // Copy the decoded row into the cache at `position`.
    void commit_branch(int32_t branch, int32_t position);
    std::size_t slot(int32_t branch, int32_t layer) const;

    // Combine the two branches' scores at the reference's AR scale, then
    // keep only what the conditional branch ranked highest.
    void guide_logits(const float* conditional, const float* unconditional,
                      std::vector<float>& out) const;

    MinimaxMusic3Engines engines_;
    MinimaxMusic3Config config_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    // Widened copies of the engines' half-width outputs, kept as members so
    // a decode step does not reallocate them.
    std::vector<std::vector<float>> hidden_scratch_;
    // Per-branch copies of the logits. The engine reuses one output
    // buffer, so a returned pointer aliases whatever ran last.
    std::vector<std::vector<float>> logits_scratch_;
    std::vector<float> guided_;
    std::vector<float> depth_conditional_;
    std::vector<float> depth_unconditional_;
    // The frame embedding the language model reads back, filled by the
    // depth stage once all eight codes are drawn.
    std::vector<float> frame_embed_;
    // The seven depth states for the frame being drawn.
    std::vector<float> depth_hidden_;
    std::vector<float> depth_scratch_;
    // Rows the engine's cache was compiled for. commit_branch writes by
    // position, so it has to know where the cache ends.
    int32_t cache_rows_{0};
    std::vector<DeviceTensor> cache_k_;
    std::vector<DeviceTensor> cache_v_;
    std::vector<DeviceTensor> present_k_;
    std::vector<DeviceTensor> present_v_;
};

} // namespace trtmc
