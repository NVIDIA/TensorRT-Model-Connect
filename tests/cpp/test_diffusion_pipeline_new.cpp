// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-DIFF-CPP-03
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-FAC-01
// Intent:         FluxPipeline, WanPipeline, and ZImagePipeline construction
//                 with null modules; verifies trivial constructors execute
//                 and pipeline_type() returns correct values
// Preconditions:  TRT headers available for type and compile check
// Postconditions: Diffusion pipeline types construct correctly with null
//                 modules and report accurate pipeline type strings
// =============================================================================

// =============================================================================
// Test suite: Diffusion pipeline construction tests
//
// FluxPipeline, WanPipeline, and ZImagePipeline all have trivial constructors
// (no module validation), so they can be constructed with null modules for
// testing the constructor body and pipeline_type() accessor.
//
// FluxPipeline constructor also computes h_latent_, w_latent_, and
// num_img_tokens_ from DiffusionConfig defaults (480x832 / scale_factor=8).
//
// For full E2E validation with real models, see tests/test_e2e.py.
// =============================================================================

#include "runtime/models/flux/pipeline.h"
#include "runtime/models/qwen_image/pipeline.h"
#include "runtime/models/wan/pipeline.h"
#include "runtime/models/z_image/pipeline.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

static int failures = 0;
static void check(bool c, const char* n) {
    if (!c) {
        std::cerr << "FAIL: " << n << '\n';
        ++failures;
    }
}

class FakeTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string& text) const override {
        (void)text;
        return {1, 2};
    }
    std::string decode(const std::vector<int32_t>& ids) const override {
        (void)ids;
        return {};
    }
    int32_t id_for_token(std::string_view token) const override {
        (void)token;
        return 0;
    }
    std::string token_for_id(int32_t id) const override {
        (void)id;
        return {};
    }
};

class FakeModule final : public trtmc::TrtModule {
  public:
    enum class Kind { TextEncoder, Denoiser, Vae };

    explicit FakeModule(Kind kind) : kind_(kind) {}

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++forward_calls;
        if (kind_ == Kind::Denoiser) {
            if (inputs.count("img_patched") != 0U) {
                denoiser_hidden_shapes.push_back(inputs.at("img_patched").shape);
                denoiser_encoder_shapes.push_back(inputs.at("txt_hidden").shape);
                denoiser_timestep_shapes.push_back(inputs.at("timestep").shape);
                const auto& hidden_shape = inputs.at("img_patched").shape;
                const int64_t batch = hidden_shape[0];
                const int64_t patches = hidden_shape[1];
                const int64_t channels = hidden_shape[2];
                output_.assign(static_cast<std::size_t>(batch * patches * channels), 0.0F);
                return {{"noise_patched",
                         trtmc::Tensor{output_.data(), {batch, patches, channels},
                                       trtmc::DType::kFloat32}}};
            }

            denoiser_hidden_shapes.push_back(inputs.at("hidden_states").shape);
            denoiser_encoder_shapes.push_back(inputs.at("encoder_hidden_states").shape);
            if (inputs.count("timestep_embedding") != 0U) {
                denoiser_timestep_shapes.push_back(inputs.at("timestep_embedding").shape);
            } else {
                denoiser_timestep_shapes.push_back(inputs.at("temb").shape);
            }
            denoiser_rope_shapes.push_back(inputs.at("rotary_cos").shape);

            const auto& hidden_shape = inputs.at("hidden_states").shape;
            const int64_t batch = hidden_shape[0];
            const int64_t patches = hidden_shape[1];
            output_.assign(static_cast<std::size_t>(batch * patches * patch_dim), 0.0F);
            return {{"output",
                     trtmc::Tensor{output_.data(), {batch, patches, patch_dim},
                                   trtmc::DType::kFloat32}}};
        }

        if (kind_ == Kind::TextEncoder) {
            text_input_shapes.push_back(inputs.at("input_ids").shape);
            output_.assign(static_cast<std::size_t>(text_seq * text_dim), 0.25F);
            return {{"text_embeddings",
                     trtmc::Tensor{output_.data(), {text_seq, text_dim},
                                   trtmc::DType::kFloat32}},
                    {"last_hidden_state",
                     trtmc::Tensor{output_.data(), {text_seq, text_dim},
                                   trtmc::DType::kFloat32}}};
        }

        if (inputs.count("latent_input") != 0U) {
            vae_input_shapes.push_back(inputs.at("latent_input").shape);
            output_.assign(static_cast<std::size_t>(3 * vae_h * vae_w), 0.0F);
            return {{"decoder_output",
                     trtmc::Tensor{output_.data(), {1, 3, vae_h, vae_w},
                                   trtmc::DType::kFloat32}}};
        }
        if (inputs.count("latent") != 0U) {
            vae_input_shapes.push_back(inputs.at("latent").shape);
            const auto& shape = inputs.at("latent").shape;
            const int64_t h = shape[3];
            const int64_t w = shape[4];
            output_.assign(static_cast<std::size_t>(3 * h * w), 0.0F);
            return {{"image",
                     trtmc::Tensor{output_.data(), {1, 3, 1, h, w},
                                   trtmc::DType::kFloat32}}};
        }

        vae_input_shapes.push_back(inputs.at("latents").shape);
        output_.assign(static_cast<std::size_t>(3 * vae_h * vae_w), 0.0F);
        return {{"image",
                 trtmc::Tensor{output_.data(), {3, vae_h, vae_w}, trtmc::DType::kFloat32}}};
    }

    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap& inputs) override {
        (void)inputs;
        return {};
    }
    void forward_device_async(const trtmc::DeviceTensorMap& inputs) override { (void)inputs; }
    void forward_async(const trtmc::TensorMap& inputs) override { (void)forward(inputs); }
    void sync() override {}

    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override {
        (void)name;
        return true;
    }
    bool has_output(const std::string& name) const override {
        (void)name;
        return true;
    }
    trtmc::DType tensor_dtype(const std::string& name) const override {
        (void)name;
        return trtmc::DType::kFloat32;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        (void)name;
        return {};
    }
    std::vector<int64_t> input_profile_shape(
        const std::string& name, int32_t profile_idx,
        trtmc::ProfileShapeSelector selector) const override {
        (void)profile_idx;
        if (kind_ == Kind::Denoiser && name == "hidden_states" &&
            selector == trtmc::ProfileShapeSelector::kMax) {
            return {2, 1, 128};
        }
        if (kind_ == Kind::Denoiser && name == "img_patched" &&
            selector == trtmc::ProfileShapeSelector::kMax) {
            return {2, 1, 4};
        }
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string& name) const override {
        (void)name;
        return nullptr;
    }
    void bind_external(const std::string& name, void* ptr) override {
        (void)name;
        (void)ptr;
    }
    int32_t input_rank(const std::string& name) const override {
        if (kind_ == Kind::Denoiser && name == "hidden_states")
            return 3;
        if (kind_ == Kind::Denoiser && name == "img_patched")
            return 3;
        return 0;
    }
    bool input_is_dynamic(const std::string& name) const override {
        return kind_ == Kind::Denoiser && name == "hidden_states";
    }
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void> resource) override { (void)resource; }

    int32_t forward_calls{0};
    std::vector<std::vector<int64_t>> text_input_shapes;
    std::vector<std::vector<int64_t>> denoiser_hidden_shapes;
    std::vector<std::vector<int64_t>> denoiser_encoder_shapes;
    std::vector<std::vector<int64_t>> denoiser_timestep_shapes;
    std::vector<std::vector<int64_t>> denoiser_rope_shapes;
    std::vector<std::vector<int64_t>> vae_input_shapes;

    static constexpr int64_t text_seq = 32;
    static constexpr int64_t text_dim = 2;
    static constexpr int64_t patch_dim = 4;
    static constexpr int64_t vae_h = 16;
    static constexpr int64_t vae_w = 16;

  private:
    Kind kind_;
    std::vector<float> output_;
};

static trtmc::ZImagePreprocessorWeights tiny_zimage_weights() {
    trtmc::ZImagePreprocessorWeights weights;
    weights.cap_dim = 2;
    weights.dit_dim = 128;
    weights.freq_dim = 4;
    weights.valid = true;
    weights.cap_norm_weight = {1.0F, 1.0F};
    weights.cap_proj_weight.assign(static_cast<std::size_t>(2 * 128), 0.0F);
    weights.cap_proj_bias.assign(128, 0.0F);
    weights.cap_pad_token.assign(128, 0.0F);
    weights.x_embed_weight.assign(static_cast<std::size_t>(4 * 128), 0.0F);
    weights.x_embed_bias.assign(128, 0.0F);
    weights.t_embedder_mlp_0_weight.assign(static_cast<std::size_t>(4 * 4), 0.0F);
    weights.t_embedder_mlp_0_bias.assign(4, 0.0F);
    weights.t_embedder_mlp_2_weight.assign(static_cast<std::size_t>(4 * 4), 0.0F);
    weights.t_embedder_mlp_2_bias.assign(4, 0.0F);
    return weights;
}

static trtmc::PreprocessorWeights tiny_flux_weights() {
    trtmc::PreprocessorWeights weights;
    constexpr int32_t freq_dim = 4;
    constexpr int32_t dit_dim = 8;
    constexpr int32_t text_dim = 2;
    constexpr int32_t packed_channels = 4;

    weights.time_emb_0_weight.assign(static_cast<std::size_t>(freq_dim * dit_dim), 0.0F);
    weights.time_emb_0_bias.assign(dit_dim, 0.0F);
    weights.time_emb_2_weight.assign(static_cast<std::size_t>(dit_dim * dit_dim), 0.0F);
    weights.time_emb_2_bias.assign(dit_dim, 0.0F);
    weights.context_embed_weight.assign(static_cast<std::size_t>(text_dim * dit_dim), 0.0F);
    weights.context_embed_bias.assign(dit_dim, 0.0F);
    weights.patch_embed_weight.assign(static_cast<std::size_t>(packed_channels * dit_dim), 0.0F);
    weights.patch_embed_bias.assign(dit_dim, 0.0F);
    return weights;
}

static void test_flux_construction() {
    // FluxPipeline constructor computes latent layout from DiffusionConfig defaults:
    //   h_latent = video_height(480) / scale_factor_spatial(8) = 60
    //   w_latent = video_width(832)  / scale_factor_spatial(8) = 104
    //   num_img_tokens = (60/2) * (104/2) = 30 * 52 = 1560
    trtmc::DiffusionConfig cfg;
    trtmc::PreprocessorWeights weights;

    trtmc::FluxPipeline pipeline(
        /*text_encoders=*/{},
        /*denoiser=*/nullptr,
        /*vae=*/nullptr, cfg, weights,
        /*tokenizer=*/nullptr,
        /*clip_tokenizer=*/nullptr,
        /*model_id_str=*/"test-flux");

    check(std::string(pipeline.pipeline_type()) == "FluxPipeline", "FluxPipeline pipeline_type");
    check(std::string(pipeline.model_id()) == "test-flux", "FluxPipeline model_id");
}

static void test_wan_construction() {
    trtmc::DiffusionConfig cfg;
    trtmc::PreprocessorWeights weights;

    trtmc::WanPipeline pipeline(
        /*text_encoder=*/nullptr,
        /*denoiser=*/nullptr,
        /*vae=*/nullptr, cfg, weights,
        /*tokenizer=*/nullptr,
        /*model_id_str=*/"test-wan");

    check(std::string(pipeline.pipeline_type()) == "WanPipeline", "WanPipeline pipeline_type");
    check(std::string(pipeline.model_id()) == "test-wan", "WanPipeline model_id");
}

static void test_zimage_construction() {
    trtmc::DiffusionConfig cfg;
    trtmc::PreprocessorWeights weights;
    trtmc::ZImagePreprocessorWeights z_weights;

    trtmc::ZImagePipeline pipeline(
        /*text_encoder=*/nullptr,
        /*denoiser=*/nullptr,
        /*vae=*/nullptr, cfg, weights, z_weights,
        /*tokenizer=*/nullptr,
        /*model_id_str=*/"test-zimage",
        /*bundle_path=*/"/tmp/test.trtfb");

    check(std::string(pipeline.pipeline_type()) == "ZImagePipeline",
          "ZImagePipeline pipeline_type");
    check(std::string(pipeline.model_id()) == "test-zimage", "ZImagePipeline model_id");
}

static void test_zimage_generate_images_uses_batched_dit_shapes() {
    trtmc::DiffusionConfig cfg;
    cfg.video_height = 2;
    cfg.video_width = 2;
    cfg.num_inference_steps = 1;
    cfg.flow_shift = 3.0F;
    cfg.dit_dim = 128;
    cfg.dit_num_heads = 1;
    cfg.z_dim = 1;
    cfg.scale_factor_spatial = 1;
    cfg.patch_size = {1, 2, 2};
    cfg.freq_dim = 4;
    cfg.text_seq_len = 32;
    cfg.text_encoder_dim = 2;
    cfg.max_batch_size.dit = 2;

    auto text_encoder = std::make_unique<FakeModule>(FakeModule::Kind::TextEncoder);
    auto denoiser = std::make_unique<FakeModule>(FakeModule::Kind::Denoiser);
    auto vae = std::make_unique<FakeModule>(FakeModule::Kind::Vae);
    auto* text_encoder_ptr = text_encoder.get();
    auto* denoiser_ptr = denoiser.get();
    auto* vae_ptr = vae.get();

    trtmc::ZImagePipeline pipeline(
        std::move(text_encoder), std::move(denoiser), std::move(vae), cfg,
        trtmc::PreprocessorWeights{}, tiny_zimage_weights(),
        std::make_shared<FakeTokenizer>(),
        /*model_id_str=*/"test-zimage-batch",
        /*bundle_path=*/"/tmp/test.trtfb");

    trtmc::GenerateConfig generate_cfg;
    generate_cfg.num_steps = 1;
    generate_cfg.seed = 123;
    const auto images = pipeline.generate_images({"a", "b", "c"}, {11, 12, 13}, generate_cfg);

    check(images.size() == 3U, "ZImagePipeline batched result count");
    check(text_encoder_ptr->forward_calls == 3, "ZImagePipeline text encoder remains serial");
    check(denoiser_ptr->forward_calls == 2, "ZImagePipeline denoiser chunks by cap");
    check(vae_ptr->forward_calls == 3, "ZImagePipeline VAE remains sliced");

    check(denoiser_ptr->denoiser_hidden_shapes.size() == 2U, "denoiser captured two chunks");
    if (denoiser_ptr->denoiser_hidden_shapes.size() == 2U) {
        check(denoiser_ptr->denoiser_hidden_shapes[0] == std::vector<int64_t>({2, 1, 128}),
              "first denoiser hidden shape");
        check(denoiser_ptr->denoiser_hidden_shapes[1] == std::vector<int64_t>({1, 1, 128}),
              "second denoiser hidden shape");
        check(denoiser_ptr->denoiser_encoder_shapes[0] ==
                  std::vector<int64_t>({2, 32, 128}),
              "first denoiser encoder shape");
        check(denoiser_ptr->denoiser_timestep_shapes[0] ==
                  std::vector<int64_t>({2, 1, 4}),
              "first denoiser timestep shape");
        check(denoiser_ptr->denoiser_rope_shapes[0] ==
                  std::vector<int64_t>({2, 33, 128}),
              "first denoiser rope shape");
    }
}

static void test_flux_generate_images_uses_batched_dit_shapes() {
    trtmc::DiffusionConfig cfg;
    cfg.video_height = 2;
    cfg.video_width = 2;
    cfg.num_inference_steps = 1;
    cfg.flow_shift = 3.0F;
    cfg.dit_dim = 8;
    cfg.dit_num_heads = 1;
    cfg.z_dim = 1;
    cfg.scale_factor_spatial = 1;
    cfg.patch_size = {1, 2, 2};
    cfg.freq_dim = 4;
    cfg.text_seq_len = 32;
    cfg.text_encoder_dim = 2;
    cfg.axes_dims_rope = {2, 2, 4};
    cfg.max_batch_size.dit = 2;

    std::vector<std::unique_ptr<trtmc::TrtModule>> text_encoders;
    auto text_encoder = std::make_unique<FakeModule>(FakeModule::Kind::TextEncoder);
    auto denoiser = std::make_unique<FakeModule>(FakeModule::Kind::Denoiser);
    auto vae = std::make_unique<FakeModule>(FakeModule::Kind::Vae);
    auto* text_encoder_ptr = text_encoder.get();
    auto* denoiser_ptr = denoiser.get();
    auto* vae_ptr = vae.get();
    text_encoders.push_back(std::move(text_encoder));

    trtmc::FluxPipeline pipeline(
        std::move(text_encoders), std::move(denoiser), std::move(vae), cfg,
        tiny_flux_weights(), std::make_shared<FakeTokenizer>(),
        /*clip_tokenizer=*/nullptr,
        /*model_id_str=*/"test-flux-batch");

    trtmc::GenerateConfig generate_cfg;
    generate_cfg.num_steps = 1;
    generate_cfg.seed = 123;
    const auto images = pipeline.generate_images({"a", "b", "c"}, {11, 12, 13}, generate_cfg);

    check(images.size() == 3U, "FluxPipeline batched result count");
    check(text_encoder_ptr->forward_calls == 3, "FluxPipeline text encoder remains serial");
    check(denoiser_ptr->forward_calls == 2, "FluxPipeline denoiser chunks by cap");
    check(vae_ptr->forward_calls == 3, "FluxPipeline VAE remains sliced");

    check(denoiser_ptr->denoiser_hidden_shapes.size() == 2U,
          "FluxPipeline denoiser captured two chunks");
    if (denoiser_ptr->denoiser_hidden_shapes.size() == 2U) {
        check(denoiser_ptr->denoiser_hidden_shapes[0] == std::vector<int64_t>({2, 1, 8}),
              "FluxPipeline first denoiser hidden shape");
        check(denoiser_ptr->denoiser_hidden_shapes[1] == std::vector<int64_t>({1, 1, 8}),
              "FluxPipeline second denoiser hidden shape");
        check(denoiser_ptr->denoiser_encoder_shapes[0] == std::vector<int64_t>({2, 32, 8}),
              "FluxPipeline first denoiser encoder shape");
        check(denoiser_ptr->denoiser_timestep_shapes[0] == std::vector<int64_t>({2, 8}),
              "FluxPipeline first denoiser temb shape");
        check(denoiser_ptr->denoiser_rope_shapes[0] == std::vector<int64_t>({2, 33, 8}),
              "FluxPipeline first denoiser rope shape");
    }
}

static void test_qwen_image_generate_images_uses_batched_dit_shapes() {
    trtmc::QwenImageConfig cfg;
    cfg.image.default_height = 2;
    cfg.image.default_width = 2;
    cfg.diffusion.default_num_inference_steps = 1;
    cfg.diffusion.default_cfg_scale = 4.0F;
    cfg.diffusion.default_negative_prompt = " ";
    cfg.text_encoder.max_seq_len = 32;
    cfg.denoiser.max_text_tokens = 32;
    cfg.denoiser.text_embed_dim = 2;
    cfg.denoiser.in_channels = 4;
    cfg.denoiser.out_channels = 1;
    cfg.denoiser.patch_size = 2;
    cfg.vae.latent_channels = 1;
    cfg.vae.spatial_scale_factor = 1;

    trtmc::QwenImagePreprocessorWeights preprocessor;
    preprocessor.latents_mean = {0.0F};
    preprocessor.latents_std = {1.0F};
    preprocessor.valid = true;

    auto text_encoder = std::make_unique<FakeModule>(FakeModule::Kind::TextEncoder);
    auto denoiser = std::make_unique<FakeModule>(FakeModule::Kind::Denoiser);
    auto vae = std::make_unique<FakeModule>(FakeModule::Kind::Vae);
    auto* text_encoder_ptr = text_encoder.get();
    auto* denoiser_ptr = denoiser.get();
    auto* vae_ptr = vae.get();

    trtmc::QwenImagePipeline::Construction c;
    c.text_engine = std::move(text_encoder);
    c.denoiser_engine = std::move(denoiser);
    c.vae_decoder_engine = std::move(vae);
    c.tokenizer = std::make_shared<FakeTokenizer>();
    c.config = cfg;
    c.preprocessor = preprocessor;
    c.max_dit_batch_size = 2;
    c.model_id = "test-qwen-image-batch";
    trtmc::QwenImagePipeline pipeline(std::move(c));

    trtmc::GenerateConfig generate_cfg;
    generate_cfg.num_steps = 1;
    generate_cfg.seed = 123;
    const auto images = pipeline.generate_images({"a", "b", "c"}, {11, 12, 13}, generate_cfg);

    check(images.size() == 3U, "QwenImagePipeline batched result count");
    check(text_encoder_ptr->forward_calls == 4,
          "QwenImagePipeline text encoder runs positives plus shared negative");
    check(denoiser_ptr->forward_calls == 4,
          "QwenImagePipeline denoiser runs cond/uncond per chunk");
    check(vae_ptr->forward_calls == 3, "QwenImagePipeline VAE remains sliced");

    check(denoiser_ptr->denoiser_hidden_shapes.size() == 4U,
          "QwenImagePipeline denoiser captured CFG chunk calls");
    if (denoiser_ptr->denoiser_hidden_shapes.size() == 4U) {
        check(denoiser_ptr->denoiser_hidden_shapes[0] == std::vector<int64_t>({2, 1, 4}),
              "QwenImagePipeline first denoiser image shape");
        check(denoiser_ptr->denoiser_hidden_shapes[2] == std::vector<int64_t>({1, 1, 4}),
              "QwenImagePipeline second denoiser image shape");
        check(denoiser_ptr->denoiser_encoder_shapes[0] == std::vector<int64_t>({2, 32, 2}),
              "QwenImagePipeline first denoiser text shape");
        check(denoiser_ptr->denoiser_timestep_shapes[0] == std::vector<int64_t>({2}),
              "QwenImagePipeline first denoiser timestep shape");
    }
}

static void test_flux_with_custom_config() {
    // Test FluxPipeline with non-default config to exercise the latent layout
    // computation path with patch_size override.
    trtmc::DiffusionConfig cfg;
    cfg.video_height = 256;
    cfg.video_width = 256;
    cfg.scale_factor_spatial = 8;
    cfg.patch_size = {1, 2, 2}; // ph=2, pw=2

    trtmc::FluxPipeline pipeline({}, nullptr, nullptr, cfg, trtmc::PreprocessorWeights{}, nullptr,
                                 nullptr, "test-flux-custom");

    check(std::string(pipeline.pipeline_type()) == "FluxPipeline",
          "FluxPipeline custom config pipeline_type");
}

int main() {
    test_flux_construction();
    test_wan_construction();
    test_zimage_construction();
    test_zimage_generate_images_uses_batched_dit_shapes();
    test_flux_generate_images_uses_batched_dit_shapes();
    test_qwen_image_generate_images_uses_batched_dit_shapes();
    test_flux_with_custom_config();
    if (failures > 0)
        std::cerr << failures << " test(s) FAILED\n";
    return failures;
}
