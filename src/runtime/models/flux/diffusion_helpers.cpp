#include "diffusion_helpers.h"

namespace trtmc {

DiffusionConfig make_diffusion_config(const std::string& json) {
    DiffusionConfig dc;
    std::string sched = extract_json_string(json, "scheduler", "flow_match_euler");
    dc.scheduler = sched.empty() ? "flow_match_euler" : sched;
    dc.num_inference_steps = extract_json_int(json, "num_inference_steps", 50);
    dc.guidance_scale = extract_json_float(json, "guidance_scale", 5.0F);
    dc.flow_shift = extract_json_float(json, "flow_shift", 1.0F);
    dc.use_dynamic_shifting = extract_json_int(json, "use_dynamic_shifting", 0) != 0;
    dc.base_shift = extract_json_float(json, "base_shift", 0.5F);
    dc.max_shift = extract_json_float(json, "max_shift", 1.15F);
    dc.base_image_seq_len = extract_json_int(json, "base_image_seq_len", 256);
    dc.max_image_seq_len = extract_json_int(json, "max_image_seq_len", 4096);
    dc.shift_terminal = extract_json_float(json, "shift_terminal", 0.0F);
    dc.video_height = extract_json_int(json, "video_height", 480);
    dc.video_width = extract_json_int(json, "video_width", 832);
    dc.video_num_frames = extract_json_int(json, "video_num_frames", 81);
    dc.z_dim = extract_json_int(json, "z_dim", 16);
    dc.scale_factor_temporal = extract_json_int(json, "scale_factor_temporal", 4);
    dc.scale_factor_spatial = extract_json_int(json, "scale_factor_spatial", 8);
    dc.dit_dim = extract_json_int(json, "dit_dim", 1536);
    dc.dit_num_heads = extract_json_int(json, "dit_num_heads", 12);
    dc.freq_dim = extract_json_int(json, "freq_dim", 256);
    dc.text_seq_len = extract_json_int(json, "text_seq_len", 512);
    dc.text_encoder_dim = extract_json_int(json, "text_encoder_dim", 4096);
    dc.num_vae_caches = extract_json_int(json, "num_vae_caches", 0);
    const auto latent_stat_count = static_cast<std::size_t>(dc.z_dim > 0 ? dc.z_dim : 16);
    dc.latents_mean = extract_json_float_array(json, "latents_mean", latent_stat_count);
    dc.latents_std = extract_json_float_array(json, "latents_std", latent_stat_count);
    dc.patch_size = extract_json_int_array(json, "patch_size");
    dc.axes_dims_rope = extract_json_int_array(json, "axes_dims_rope");
    dc.rope_theta = extract_json_float(json, "rope_theta", 10000.0F);
    dc.vae_model_id = extract_json_string(json, "vae_model_id", "");
    dc.guidance_embeds = extract_json_int(json, "guidance_embeds", 0) != 0;
    dc.use_rope = extract_json_int(json, "use_rope", 1) != 0;
    dc.vae_scaling_factor = extract_json_float(json, "vae_scaling_factor", 0.0F);
    dc.diffusion_backend_type = extract_json_string(json, "diffusion_backend_type", "wan_3d");
    return dc;
}

DiffusionParts load_diffusion_parts(IBackend* backend, const BundleFile& bundle,
                                    const std::string& json, const ModuleCreateOptions& options,
                                    const std::string& denoiser_section_name,
                                    const ModuleCreateOptions* denoiser_options) {
    DiffusionParts parts;

    const ModuleCreateOptions& effective_denoiser_options =
        denoiser_options != nullptr ? *denoiser_options : options;
    parts.denoiser =
        load_trt_module_from_plan(backend, find_section(bundle, denoiser_section_name),
                                  denoiser_section_name.c_str(), effective_denoiser_options);
    parts.vae = load_trt_module_from_plan(backend, find_section(bundle, "vae_decoder_plan"),
                                          "vae_decoder_plan", options);
    parts.vision = try_load_trt_module_from_plan(
        backend, find_section(bundle, "vision_engine_plan"), "vision_engine_plan", options);
    parts.vae_encoder = try_load_trt_module_from_plan(
        backend, find_section(bundle, "vae_encoder_plan"), "vae_encoder_plan", options);

    auto te_plans = find_sections_by_prefix(bundle, "text_encoder_");
    for (std::size_t i = 0; i < te_plans.size(); ++i) {
        std::string label = "text_encoder_" + std::to_string(i);
        parts.text_encoders.push_back(
            load_trt_module_from_plan(backend, te_plans[i], label.c_str(), options));
    }
    if (parts.text_encoders.empty()) {
        auto* plan = find_section(bundle, "engine_plan");
        if (plan && !plan->empty()) {
            parts.text_encoders.push_back(
                load_trt_module_from_plan(backend, plan, "text_encoder_0", options));
        }
    }

    parts.config = make_diffusion_config(json);
    // Carry the engine batch envelope (parsed from the bundle's top-level
    // `max_batch_size` block by ReadBundleFile) into the runtime config so
    // pipelines can clamp/chunk against it. Defaults to {1,1,1} when the
    // block is absent — see design doc Decision C.
    parts.config.max_batch_size.dit = bundle.info.max_batch_size.dit;
    parts.config.max_batch_size.text_encoder = bundle.info.max_batch_size.text_encoder;
    parts.config.max_batch_size.vae = bundle.info.max_batch_size.vae;

    auto* pw = find_section(bundle, "preprocessor_weights");
    if (pw && !pw->empty())
        parts.weights = parse_preprocessor_weights(*pw);

    parts.tokenizer = create_tokenizer_from_bundle(bundle);
    return parts;
}

} // namespace trtmc
