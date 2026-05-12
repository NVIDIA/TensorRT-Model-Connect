#pragma once

// PixArtTorchTrtPipeline: Simplified diffusion pipeline for torch-trt engines.
//
// Unlike PixArtPipeline which does CPU-side preprocessing (patch embedding,
// timestep embedding, text projection), the torch-trt engines include all
// preprocessing internally. The pipeline just:
//   1. Tokenize prompt -> run T5 -> text embeddings
//   2. For each step: run DiT(latent, text, timestep) -> noise prediction
//   3. Run VAE(latent) -> image
//
// Bundle sections: text_encoder_0_plan, denoiser_plan, vae_decoder_plan,
//                  config.json (with runtime_strategy="diffusion_pixart_torchtrt")
// No preprocessor_weights needed.

#include "runtime/domains/diffusion/diffusion_types.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

// DPM-Solver++ Multistep Scheduler (matches HF diffusers exactly).
// See pixart_torchtrt_pipeline.cpp for detailed algorithm comments.
struct DpmSolverState {
    std::vector<double> alpha_t;
    std::vector<double> sigma_t;
    std::vector<double> lambda_t;
    std::vector<float> timesteps;
    int32_t num_train_timesteps{1000};
    std::vector<std::vector<float>> model_outputs;
    int32_t lower_order_nums{0};

    void set_timesteps(int32_t num_steps, double beta_start = 0.0001, double beta_end = 0.02);
    void step(const float* eps_pred, const float* sample, float* x_out, std::size_t count,
              int32_t step_index, int32_t num_steps);
};

class PixArtTorchTrtPipeline final : public IPipeline {
  public:
    PixArtTorchTrtPipeline(std::unique_ptr<TrtModule> text_encoder,
                           std::unique_ptr<TrtModule> denoiser, std::unique_ptr<TrtModule> vae,
                           DiffusionConfig config, std::shared_ptr<ITokenizer> tokenizer,
                           std::string model_id_str);

    ~PixArtTorchTrtPipeline() override;

    ImageResult generate_image(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "PixArtTorchTrtPipeline"; }

  private:
    bool run_t5_encoder(const std::vector<int32_t>& input_ids, std::vector<float>& text_embeddings,
                        bool zero_padding = true);
    bool run_denoiser(const std::vector<float>& latent, const std::vector<float>& text_embeddings,
                      int32_t num_real_tokens, float timestep, std::vector<float>& output);
    bool decode_vae(const std::vector<float>& latent, int32_t h_lat, int32_t w_lat,
                    VideoResult& result);
    bool encode_prompt(const std::string& prompt, std::vector<int32_t>& input_ids,
                       std::vector<float>& text_embeddings, std::vector<float>& null_text);
    bool denoise_loop(std::vector<float>& latents, const std::vector<float>& text_embeddings,
                      int32_t num_real_tokens, const std::vector<float>& null_text,
                      float guidance_scale, DpmSolverState& scheduler, int32_t num_steps);
    bool apply_cfg(const std::vector<float>& latents, const std::vector<float>& null_text,
                   float timestep, float guidance_scale, const std::vector<float>& noise_pred,
                   std::vector<float>& noise_uncond, std::vector<float>& eps_out);

    std::unique_ptr<TrtModule> text_encoder_;
    std::unique_ptr<TrtModule> denoiser_;
    std::unique_ptr<TrtModule> vae_;
    DiffusionConfig config_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
};

} // namespace trtmc
