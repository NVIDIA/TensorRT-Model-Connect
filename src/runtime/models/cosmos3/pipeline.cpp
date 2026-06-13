// Cosmos3Pipeline implementation — multi-engine orchestration for the
// Cosmos 3 omni-model. Currently a scaffold: the constructor wires modules,
// generate_diffusion() throws with a clear error listing the remaining work.
//
// Wiring order for a text→video lane (the primary capability):
//
//   1. Tokenize prompt → AR token ids
//   2. prefill_reasoner(prompt_ids): single forward pass through the
//      reasoner engine, populating reasoner_state_ (KvCache).
//   3. Sample noise into a (T_lat, 48, H_lat, W_lat) latent tensor.
//   4. For step in num_inference_steps:
//        a. denoise_step(latent, sigma[step], latent_out):
//             DM generator engine consumes the latent + AR KV cache, returns
//             a velocity / noise prediction (Cosmos 3 uses rectified flow
//             with UniPC scheduler — see python sampler glue).
//        b. UniPC step: latent_in = scheduler.step(velocity, sigma[step])
//   5. decode_video(latent_final, ...) → uint8 RGB frames → MP4 mux.
//
// The two_way joint attention is realized as: during the prefill we mark
// the AR portion of the KV cache as "frozen but visible"; each DM denoising
// step then references that AR KV alongside the DM-local KV via a
// concatenated cache view. This requires KvCache support for a
// "frozen prefix + bidirectional suffix" layout — see Phase-6 follow-up.

#include "runtime/models/cosmos3/pipeline.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

Cosmos3Pipeline::Cosmos3Pipeline(std::unique_ptr<TrtModule> reasoner,
                                 std::unique_ptr<IInferenceState> reasoner_state,
                                 std::unique_ptr<TrtModule> dm_generator,
                                 std::unique_ptr<TrtModule> vae_decoder,
                                 std::unique_ptr<TrtModule> vit_encoder, Cosmos3Config config,
                                 cudaStream_t stream, std::shared_ptr<ITokenizer> tokenizer,
                                 std::string model_id_str)
    : reasoner_(std::move(reasoner)), reasoner_state_(std::move(reasoner_state)),
      dm_generator_(std::move(dm_generator)), vae_decoder_(std::move(vae_decoder)),
      vit_encoder_(std::move(vit_encoder)), config_(std::move(config)), stream_(stream),
      tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id_str)) {}

Cosmos3Pipeline::~Cosmos3Pipeline() = default;

DiffusionResult Cosmos3Pipeline::generate_diffusion(const std::string& /*prompt*/,
                                                    const DiffusionConfig& /*cfg*/) {
    throw std::runtime_error("Cosmos3Pipeline::generate_diffusion not yet implemented.\n"
                             "Required follow-up work in this file:\n"
                             "  - prefill_reasoner(): forward pass through the AR engine and "
                             "populate the shared KV cache that the DM engine will read during "
                             "joint attention.\n"
                             "  - KvCache extension: support a 'frozen prefix + bidirectional "
                             "suffix' layout so the AR tokens stay fixed while DM tokens evolve "
                             "across denoising steps.\n"
                             "  - denoise_step(): pack latent + AR KV into the DM engine inputs; "
                             "execute one TRT engine call; produce a velocity / noise "
                             "prediction.\n"
                             "  - Rectified-flow / UniPC scheduler bindings (Cosmos 3 uses the "
                             "diffusers UniPCMultistepScheduler with shift schedule {256:3, "
                             "480:5, 720:10}).\n"
                             "  - decode_video(): VAE decoder runs once at the end; reuses the "
                             "wan_t2v causal VAE 3D decoder path.\n"
                             "  - Tokenizer integration: Cosmos 3 uses Qwen2TokenizerFast for "
                             "text; the tokenizer is loaded by the pipeline factory and supplied "
                             "via the ITokenizer interface.");
}

void Cosmos3Pipeline::prefill_reasoner(const std::vector<int32_t>& /*prompt_ids*/) {
    // TODO(Phase-6 follow-up): forward through reasoner_ engine, fill
    // reasoner_state_ KV cache. Mirrors VlPipeline / OmniPipeline prefill.
    throw std::runtime_error("Cosmos3Pipeline::prefill_reasoner not implemented");
}

void Cosmos3Pipeline::denoise_step(const float* /*latent_in*/, float /*timestep*/,
                                   float* /*latent_out*/) {
    // TODO(Phase-6 follow-up): execute one dm_generator_ engine call with
    // [latent_in | timestep_embed | AR KV from reasoner_state_] as inputs.
    throw std::runtime_error("Cosmos3Pipeline::denoise_step not implemented");
}

std::vector<uint8_t> Cosmos3Pipeline::decode_video(const float* /*latent_final*/,
                                                   int32_t /*latent_t*/, int32_t /*latent_h*/,
                                                   int32_t /*latent_w*/) {
    // TODO(Phase-6 follow-up): VAE decode → uint8 frame buffer → MP4 mux.
    throw std::runtime_error("Cosmos3Pipeline::decode_video not implemented");
}

} // namespace trtmc
