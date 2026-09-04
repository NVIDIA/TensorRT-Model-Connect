/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_format.h"
#include "runtime/models/minimax_h3/pipeline.h"
#include "runtime/models/minimax_h3/ref2va_runtime.h"
#include "trtmc/runtime/pipeline_plugin_loader.h"
#include "trtmc/runtime/pipeline_registry.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* label) {
    if (!condition)
        throw std::runtime_error(label);
}

template <typename Function>
bool rejects(const Function& function) {
    try {
        function();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

class ByteTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string& text) const override {
        std::vector<int32_t> result;
        result.reserve(text.size());
        for (unsigned char value : text)
            result.push_back(static_cast<int32_t>(value) + 1000);
        return result;
    }
    std::string decode(const std::vector<int32_t>& ids) const override {
        std::string result;
        for (int32_t id : ids)
            result.push_back(static_cast<char>(id - 1000));
        return result;
    }
    int32_t id_for_token(std::string_view token) const override {
        return token.empty() ? -1 : static_cast<unsigned char>(token.front()) + 1000;
    }
    std::string token_for_id(int32_t id) const override {
        return std::string(1, static_cast<char>(id - 1000));
    }
};

struct TensorSpec {
    trtmc::DType dtype{trtmc::DType::kFloat32};
    std::vector<int64_t> shape;
    bool input{true};
    bool dynamic{false};
    std::vector<int64_t> minimum;
    std::vector<int64_t> optimum;
    std::vector<int64_t> maximum;
};

enum class ForwardKind { kNone, kText, kAdaln, kDenoiser, kVideoVae, kAudioVae };

class FakeModule final : public trtmc::ITrtModule {
  public:
    explicit FakeModule(ForwardKind kind = ForwardKind::kNone) : kind_(kind) {}

    std::unordered_map<std::string, TensorSpec> tensors;

    void add_dynamic(const std::string& name, trtmc::DType dtype, std::vector<int64_t> minimum,
                     std::vector<int64_t> optimum, std::vector<int64_t> maximum) {
        tensors.emplace(name, TensorSpec{dtype, maximum, true, true, std::move(minimum),
                                         std::move(optimum), maximum});
    }
    void add_static(const std::string& name, trtmc::DType dtype, std::vector<int64_t> shape) {
        tensors.emplace(name, TensorSpec{dtype, std::move(shape), true});
    }
    void add_output(const std::string& name, trtmc::DType dtype, std::vector<int64_t> maximum) {
        tensors.emplace(name, TensorSpec{dtype, std::move(maximum), false});
    }

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        if (kind_ == ForwardKind::kText) {
            require(inputs.size() == 9 && inputs.count("vision_count") == 1,
                    "fake text encoder did not receive its exact inputs");
            require(*static_cast<const int32_t*>(inputs.at("vision_count").data) == 0 &&
                        inputs.at("vision_row_indices").shape == std::vector<int64_t>({1}) &&
                        inputs.at("vision_embeds").shape == std::vector<int64_t>({1, 5120}),
                    "audio-only text path did not bind the dummy vision ABI");
            const int64_t rows = inputs.at("input_ids").shape.at(0);
            text_.assign(static_cast<std::size_t>(rows) * 5120U, 0.125F);
            return {{"encoder_hidden_states",
                     trtmc::Tensor{text_.data(), {rows, 5120}, trtmc::DType::kFloat32}}};
        }
        if (kind_ == ForwardKind::kAdaln) {
            require(inputs.size() == 1 && inputs.count("timestep_features") == 1,
                    "fake AdaLN did not receive its exact input");
            block_.assign(12U * 6U * 5376U, 0);
            final_.assign(4U * 2U * 5376U, 0);
            trtmc::TensorMap outputs;
            for (int32_t layer = 0; layer < 50; ++layer) {
                outputs.emplace(
                    "block_modulation_" + std::to_string(layer),
                    trtmc::Tensor{block_.data(), {12, 6, 5376}, trtmc::DType::kBFloat16});
            }
            outputs.emplace("final_modulation",
                            trtmc::Tensor{final_.data(), {4, 2, 5376}, trtmc::DType::kBFloat16});
            return outputs;
        }
        if (kind_ == ForwardKind::kDenoiser) {
            require(inputs.size() == 60, "fake denoiser did not receive all 60 bindings");
            const auto video_rows = inputs.at("video_hidden_states").shape.at(0);
            const auto audio_rows = inputs.at("audio_hidden_states").shape.at(0);
            video_.assign(static_cast<std::size_t>(video_rows) * 96U, 0.25F);
            audio_.assign(static_cast<std::size_t>(audio_rows) * 32U, -0.5F);
            return {{"video_velocity",
                     trtmc::Tensor{video_.data(), {video_rows, 96}, trtmc::DType::kFloat32}},
                    {"audio_velocity",
                     trtmc::Tensor{audio_.data(), {audio_rows, 32}, trtmc::DType::kFloat32}}};
        }
        if (kind_ == ForwardKind::kVideoVae) {
            require(inputs.size() == 1 && inputs.count("pixel_tile_clip") == 1,
                    "fake VideoVAE did not receive its static clip");
            posterior_.assign(48U * 5U * 16U * 16U, 0.0F);
            return {{"posterior_parameter_tile_clip",
                     trtmc::Tensor{posterior_.data(), {1, 48, 5, 16, 16}, trtmc::DType::kFloat32}}};
        }
        if (kind_ == ForwardKind::kAudioVae) {
            require(inputs.size() == 1 && inputs.count("audio_samples") == 1,
                    "fake AudioVAE did not receive stereo samples");
            const auto latent_frames = inputs.at("audio_samples").shape.at(2) / 800;
            posterior_.assign(static_cast<std::size_t>(2 * 32 * latent_frames), 0.0F);
            return {
                {"posterior_mean",
                 trtmc::Tensor{posterior_.data(), {2, 32, latent_frames}, trtmc::DType::kFloat32}}};
        }
        return {};
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return info(true); }
    std::vector<trtmc::TensorInfo> output_info() const override { return info(false); }
    bool has_input(const std::string& name) const override {
        const auto iterator = tensors.find(name);
        return iterator != tensors.end() && iterator->second.input;
    }
    bool has_output(const std::string& name) const override {
        const auto iterator = tensors.find(name);
        return iterator != tensors.end() && !iterator->second.input;
    }
    trtmc::DType tensor_dtype(const std::string& name) const override {
        return tensors.at(name).dtype;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        return tensors.at(name).shape;
    }
    std::vector<int64_t> input_profile_shape(const std::string& name, int32_t,
                                             trtmc::ProfileShapeSelector selector) const override {
        const auto& spec = tensors.at(name);
        if (selector == trtmc::ProfileShapeSelector::kMin)
            return spec.minimum;
        if (selector == trtmc::ProfileShapeSelector::kOpt)
            return spec.optimum;
        return spec.maximum;
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    int32_t input_rank(const std::string& name) const override {
        return static_cast<int32_t>(tensors.at(name).shape.size());
    }
    bool input_is_dynamic(const std::string& name) const override {
        return tensors.at(name).dynamic;
    }
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    std::vector<trtmc::TensorInfo> info(bool input) const {
        std::vector<trtmc::TensorInfo> result;
        for (const auto& [name, spec] : tensors) {
            if (spec.input == input)
                result.push_back({name, spec.shape, spec.dtype, input});
        }
        return result;
    }

    ForwardKind kind_;
    std::vector<uint16_t> block_;
    std::vector<uint16_t> final_;
    std::vector<float> video_;
    std::vector<float> audio_;
    std::vector<float> posterior_;
    std::vector<float> text_;
};

FakeModule make_adaln_module() {
    FakeModule module(ForwardKind::kAdaln);
    module.add_static("timestep_features", trtmc::DType::kFloat32, {4, 256});
    for (int32_t layer = 0; layer < 50; ++layer)
        module.add_output("block_modulation_" + std::to_string(layer), trtmc::DType::kBFloat16,
                          {12, 6, 5376});
    module.add_output("final_modulation", trtmc::DType::kBFloat16, {4, 2, 5376});
    return module;
}

FakeModule make_denoiser_module() {
    FakeModule module(ForwardKind::kDenoiser);
    module.add_dynamic("video_hidden_states", trtmc::DType::kFloat32, {18870, 96}, {44592, 96},
                       {364608, 96});
    module.add_dynamic("audio_hidden_states", trtmc::DType::kFloat32, {414, 32}, {414, 32},
                       {3558, 32});
    module.add_dynamic("encoder_hidden_states", trtmc::DType::kFloat32, {1, 5120}, {7433, 5120},
                       {262144, 5120});
    module.add_dynamic("position_ids", trtmc::DType::kFloat32, {19285, 3}, {52439, 3}, {630310, 3});
    module.add_dynamic("video_indices", trtmc::DType::kInt32, {18870}, {44592}, {364608});
    module.add_dynamic("audio_indices", trtmc::DType::kInt32, {414}, {414}, {3558});
    module.add_dynamic("text_indices", trtmc::DType::kInt32, {1}, {7433}, {262144});
    for (const char* name : {"adaln_indices", "timestep_indices"})
        module.add_dynamic(name, trtmc::DType::kInt32, {19285}, {52439}, {630310});
    for (int32_t layer = 0; layer < 50; ++layer)
        module.add_static("block_modulation_" + std::to_string(layer), trtmc::DType::kBFloat16,
                          {12, 6, 5376});
    module.add_static("final_modulation", trtmc::DType::kBFloat16, {4, 2, 5376});
    module.add_output("video_velocity", trtmc::DType::kFloat32, {364608, 96});
    module.add_output("audio_velocity", trtmc::DType::kFloat32, {3558, 32});
    return module;
}

FakeModule make_video_vae_module() {
    FakeModule module(ForwardKind::kVideoVae);
    module.add_static("pixel_tile_clip", trtmc::DType::kFloat32, {1, 3, 17, 256, 256});
    module.add_output("posterior_parameter_tile_clip", trtmc::DType::kFloat32, {1, 48, 5, 16, 16});
    return module;
}

FakeModule make_audio_vae_module() {
    FakeModule module(ForwardKind::kAudioVae);
    module.add_dynamic("audio_samples", trtmc::DType::kFloat32, {2, 1, 64000}, {2, 1, 165600},
                       {2, 1, 480000});
    module.add_output("posterior_mean", trtmc::DType::kFloat32, {2, 32, 600});
    return module;
}

FakeModule make_text_module() {
    FakeModule module(ForwardKind::kText);
    module.add_dynamic("input_ids", trtmc::DType::kInt32, {1}, {1144}, {262144});
    module.add_dynamic("mrope_position_ids", trtmc::DType::kInt32, {3, 1}, {3, 1144}, {3, 262144});
    module.add_dynamic("vision_mask", trtmc::DType::kFloat32, {1, 1}, {1144, 1}, {262144, 1});
    module.add_static("vision_count", trtmc::DType::kInt32, {1});
    module.add_dynamic("vision_row_indices", trtmc::DType::kInt32, {1}, {1008}, {262144});
    for (const char* name : {"vision_embeds", "deepstack_0", "deepstack_1", "deepstack_2"})
        module.add_dynamic(name, trtmc::DType::kFloat32, {1, 5120}, {1008, 5120}, {262144, 5120});
    module.add_output("encoder_hidden_states", trtmc::DType::kFloat32, {262144, 5120});
    return module;
}

void test_temporal_and_presentation_contract() {
    const auto schedule = trtmc::minimax_h3::make_ref2va_video_encode_schedule(124);
    require(schedule.snapped_frames == 124 && schedule.clip_count == 8 &&
                schedule.repeated_tail_frames == 12 && schedule.raw_posterior_frames == 40 &&
                schedule.output_latent_frames == 37,
            "Ref2VA VideoVAE 17*n+5 schedule drifted");

    const auto sample = trtmc::minimax_h3::make_ref2va_qwen_video_sample(48);
    require(sample.frame_indices == std::vector<int32_t>({0, 12, 24, 36}),
            "Ref2VA Qwen 24fps-to-2fps sampling drifted");
    require(sample.timestamp_seconds == std::vector<double>({0.25, 1.25}),
            "Ref2VA timestamp pairing drifted");

    trtmc::VideoReferenceInput image;
    image.kind = trtmc::VideoReferenceKind::kImage;
    image.image.height = 32;
    image.image.width = 32;
    image.image.channels = 3;
    trtmc::VideoReferenceInput video;
    video.kind = trtmc::VideoReferenceKind::kVideo;
    video.video.num_frames = 48;
    video.video.height = 32;
    video.video.width = 32;
    video.video.channels = 3;
    video.video.fps_numerator = 24;
    video.video.fps_denominator = 1;
    video.video.soundtrack.samples = {0.0F, 0.0F};
    const auto blueprint =
        trtmc::minimax_h3::make_ref2va_presentation_blueprint("prompt", {image, video});
    require(blueprint.vision_invocations.size() == 3 && blueprint.vision_invocations[0].is_image &&
                blueprint.vision_invocations[1].first_frame == 0 &&
                blueprint.vision_invocations[1].second_frame == 12 &&
                blueprint.vision_invocations[2].first_frame == 24 &&
                blueprint.vision_invocations[2].second_frame == 36,
            "Ref2VA vision invocation order drifted");
    bool saw_audio_before_video = false;
    bool saw_even_timestamp = false;
    for (std::size_t index = 0; index + 1 < blueprint.pieces.size(); ++index) {
        saw_audio_before_video |= blueprint.pieces[index].text == "<Audio 1>: " &&
                                  blueprint.pieces[index + 1].text == "<Video 1>: ";
        saw_even_timestamp |= blueprint.pieces[index].text == "<0.2 seconds>";
    }
    require(saw_audio_before_video && saw_even_timestamp,
            "Ref2VA ordered labels/Python half-even timestamp drifted");

    trtmc::VideoImageInput first_pair;
    first_pair.height = first_pair.width = 32;
    first_pair.channels = 3;
    first_pair.pixels.assign(32U * 32U * 3U, 0.0F);
    auto second_pair = first_pair;
    second_pair.pixels.assign(second_pair.pixels.size(), 1.0F);
    const auto pair_inputs = trtmc::minimax_h3::make_ref2va_vision_inputs(first_pair, second_pair);
    require(pair_inputs.patch_rows == 4 && pair_inputs.pixel_values[0] == -1.0F &&
                pair_inputs.pixel_values[256] == 1.0F,
            "Ref2VA Qwen temporal pair duplicated the first frame");

    ByteTokenizer tokenizer;
    const auto materialized =
        trtmc::minimax_h3::materialize_ref2va_presentation(blueprint, tokenizer);
    require(materialized.vision_rows == 3 &&
                materialized.mrope_position_ids.size() == materialized.input_ids.size() * 3U &&
                materialized.qwen_token_types.size() == materialized.input_ids.size() &&
                materialized.h3_token_tags.size() == materialized.input_ids.size(),
            "Ref2VA materialized Qwen/MRoPE contract drifted");
}

void test_audio_only_dummy_vision_path() {
    trtmc::VideoReferenceInput audio;
    audio.kind = trtmc::VideoReferenceKind::kAudio;
    const auto blueprint = trtmc::minimax_h3::make_ref2va_presentation_blueprint("prompt", {audio});
    require(blueprint.vision_invocations.empty(),
            "audio-only Ref2VA presentation unexpectedly invoked Qwen vision");
    ByteTokenizer tokenizer;
    const auto presentation =
        trtmc::minimax_h3::materialize_ref2va_presentation(blueprint, tokenizer);
    require(presentation.vision_rows == 0 && presentation.vision_row_indices.empty(),
            "audio-only Ref2VA presentation contains visual rows");
    auto text = make_text_module();
    const trtmc::minimax_h3::Ref2vaVisionFeatures no_vision;
    const auto embeddings =
        trtmc::minimax_h3::run_ref2va_text_encoder(text, presentation, no_vision);
    require(embeddings.size() == presentation.input_ids.size() * 5120U,
            "audio-only Ref2VA dummy-vision text path failed");
}

void test_reference_vae_fake_plan_paths() {
    auto audio_module = make_audio_vae_module();
    trtmc::AudioResult audio;
    audio.sample_rate = 32000;
    audio.channels = 2;
    audio.samples.resize(128000);
    audio.num_samples = static_cast<int32_t>(audio.samples.size());
    std::array<float, 32> mean{};
    std::array<float, 32> standard_deviation{};
    standard_deviation.fill(1.0F);
    const auto audio_condition = trtmc::minimax_h3::run_ref2va_audio_vae_encoder(
        audio_module, audio, mean, standard_deviation);
    require(audio_condition.geometry.kind == trtmc::VideoReferenceKind::kAudio &&
                audio_condition.geometry.audio_latents == 80 &&
                audio_condition.audio_hidden_states.size() == 2U * 80U * 32U,
            "Ref2VA fake AudioVAE path drifted");

    auto video_module = make_video_vae_module();
    trtmc::VideoClipInput video;
    video.num_frames = 48;
    video.height = video.width = 256;
    video.channels = 3;
    video.fps_numerator = 24;
    video.fps_denominator = 1;
    video.pixels.resize(48U * 256U * 256U * 3U);
    const auto video_condition =
        trtmc::minimax_h3::run_ref2va_video_vae_encoder(video_module, video);
    require(video_condition.geometry.kind == trtmc::VideoReferenceKind::kVideo &&
                video_condition.geometry.latent_frames == 12 &&
                video_condition.video_hidden_states.size() == 12U * 8U * 8U * 96U,
            "Ref2VA fake VideoVAE clip/global-drop path drifted");
}

void test_scheduler_and_plugin_fail_closed_contract() {
    const auto video = trtmc::make_minimax_h3_schedule(50, 12.0F);
    const auto audio = trtmc::make_minimax_h3_schedule(50, 3.0F);
    require(video.sigmas.size() == 50U && video.timesteps.size() == 49U &&
                audio.sigmas.size() == 50U && audio.timesteps.size() == 49U &&
                video.timesteps.at(1) != audio.timesteps.at(1),
            "Ref2VA 50/49 dual-shift schedule drifted");

    trtmc::load_model_plugin_for_strategy("diffusion_minimax_h3");
    auto* plugin = trtmc::PipelineRegistry::instance().lookup("diffusion_minimax_h3");
    require(plugin != nullptr, "Ref2VA model plugin did not register");
    trtmc::BundleFile bundle;
    trtmc::BaseConfig base;
    const std::string empty;
    const auto rejects_scheduler = [&](const std::string& scheduler) {
        const std::string config =
            "{\"engine_backend\":\"trt_rtx\",\"public_workflows\":[\"t2va\",\"fl2va\",\"ref2va\"],"
            "\"ref2va_schema_version\":2,\"ref2va_supported\":true,"
            "\"ref2va_context_ir_supported\":false,"
            "\"ref2va_regenerate_2k_supported\":false,"
            "\"ref2va_runtime_language\":\"c++/cuda\","
            "\"ref2va_runtime_dependencies\":[\"TensorRT-RTX\"],"
            "\"ref2va_transformer_fallback_allowed\":false,"
            "\"ref2va_scheduler\":" +
            scheduler + "}";
        trtmc::PipelineContext context{bundle, base, config, empty, empty, nullptr, empty, false};
        try {
            (void)plugin->create(context);
        } catch (const std::runtime_error& error) {
            return std::string(error.what()).find("scheduler") != std::string::npos;
        }
        return false;
    };
    require(rejects_scheduler("{\"sigma_grid_points\":5,\"transformer_forwards\":49,"
                              "\"video_shift\":12.0,\"audio_shift\":3.0,"
                              "\"guidance_scale\":1.0,\"guidance_distilled\":true}"),
            "Ref2VA plugin accepted a non-50-point scheduler");
    require(rejects_scheduler("{\"sigma_grid_points\":50,\"transformer_forwards\":4,"
                              "\"video_shift\":12.0,\"audio_shift\":3.0,"
                              "\"guidance_scale\":1.0,\"guidance_distilled\":true}"),
            "Ref2VA plugin accepted a non-49-forward scheduler");
    require(rejects_scheduler("{\"sigma_grid_points\":50,\"transformer_forwards\":49,"
                              "\"video_shift\":11.0,\"audio_shift\":4.0,"
                              "\"guidance_scale\":1.0,\"guidance_distilled\":true}"),
            "Ref2VA plugin accepted incorrect video/audio shifts");
    require(rejects_scheduler("{\"sigma_grid_points\":50,\"transformer_forwards\":49,"
                              "\"video_shift\":12.0,\"audio_shift\":3.0,"
                              "\"guidance_scale\":2.0,\"guidance_distilled\":false}"),
            "Ref2VA plugin accepted guider/negative-prompt semantics");

    const std::string extra_workflow_config =
        "{\"engine_backend\":\"trt_rtx\","
        "\"public_workflows\":[\"t2va\",\"fl2va\",\"ref2va\",\"context-ir\"],"
        "\"ref2va_schema_version\":2,\"ref2va_supported\":true,"
        "\"ref2va_context_ir_supported\":false,"
        "\"ref2va_regenerate_2k_supported\":false,"
        "\"ref2va_runtime_language\":\"c++/cuda\","
        "\"ref2va_runtime_dependencies\":[\"TensorRT-RTX\"],"
        "\"ref2va_transformer_fallback_allowed\":false,"
        "\"ref2va_scheduler\":{\"sigma_grid_points\":50,"
        "\"transformer_forwards\":49,\"video_shift\":12.0,"
        "\"audio_shift\":3.0,\"guidance_scale\":1.0,"
        "\"guidance_distilled\":true}}";
    trtmc::PipelineContext extra_workflow_context{
        bundle, base, extra_workflow_config, empty, empty, nullptr, empty, false};
    bool rejected_extra_workflow = false;
    try {
        (void)plugin->create(extra_workflow_context);
    } catch (const std::runtime_error&) {
        rejected_extra_workflow = true;
    }
    require(rejected_extra_workflow,
            "Ref2VA plugin accepted a Context-IR/2K-style extra public workflow");

    trtmc::BundleFile undeclared_ref_bundle;
    undeclared_ref_bundle.info.sections.push_back({"ref2va_denoiser_plan", 0, 1});
    const std::string undeclared_ref_config = "{\"engine_backend\":\"trt_rtx\","
                                              "\"public_workflows\":[\"t2va\",\"fl2va\"],"
                                              "\"ref2va_supported\":false}";
    trtmc::PipelineContext undeclared_ref_context{
        undeclared_ref_bundle, base, undeclared_ref_config, empty, empty, nullptr, empty, false};
    bool rejected_undeclared_ref_plans = false;
    try {
        (void)plugin->create(undeclared_ref_context);
    } catch (const std::runtime_error& error) {
        rejected_undeclared_ref_plans =
            std::string(error.what()).find("exact public workflow") != std::string::npos;
    }
    require(rejected_undeclared_ref_plans,
            "Ref2VA plugin accepted undeclared Ref plan sections as a legacy bundle");

    const std::string unknown_workflow_config =
        "{\"engine_backend\":\"trt_rtx\","
        "\"public_workflows\":[\"t2va\",\"fl2va\",\"context-ir\"]}";
    trtmc::PipelineContext unknown_workflow_context{
        bundle, base, unknown_workflow_config, empty, empty, nullptr, empty, false};
    bool rejected_unknown_workflow = false;
    try {
        (void)plugin->create(unknown_workflow_context);
    } catch (const std::runtime_error& error) {
        rejected_unknown_workflow =
            std::string(error.what()).find("ordered prefix") != std::string::npos;
    }
    require(rejected_unknown_workflow,
            "MiniMax-H3 plugin accepted an unknown non-Ref public workflow");
}

void test_packed_layout_and_timesteps() {
    using trtmc::VideoReferenceKind;
    using trtmc::minimax_h3::Ref2vaEncodedReferenceGeometry;
    const std::vector<Ref2vaEncodedReferenceGeometry> references = {
        {VideoReferenceKind::kImage, 1, 4, 4, 0},
        {VideoReferenceKind::kVideo, 2, 4, 4, 2},
        {VideoReferenceKind::kAudio, 0, 0, 0, 3},
    };
    const auto layout =
        trtmc::minimax_h3::make_ref2va_packed_layout({1, 0}, references, 2, 4, 4, 2);
    require(layout.sequence_length() == 36 && layout.condition_video_rows == 12 &&
                layout.condition_audio_rows == 10,
            "Ref2VA packed row totals drifted");
    require(layout.text_indices == std::vector<int32_t>({0, 1}) &&
                layout.video_indices.front() == 2 && layout.video_indices[4] == 10 &&
                layout.video_indices.back() == 35 && layout.audio_indices.front() == 6 &&
                layout.audio_indices[4] == 18 && layout.audio_indices.back() == 27,
            "Ref2VA interleaved scatter order drifted");

    const auto rows = trtmc::minimax_h3::make_ref2va_row_timesteps(layout, 0.5F, 0.25F);
    require(rows.unique_timesteps == std::vector<float>({0.25F, 0.5F, 0.999F, 1.0F}),
            "Ref2VA sorted timestep table drifted");
    require(rows.timestep_indices[0] == 1 && rows.adaln_indices[0] == 4 &&
                rows.timestep_indices[2] == 2 && rows.adaln_indices[2] == 6 &&
                rows.timestep_indices[6] == 3 && rows.adaln_indices[6] == 11 &&
                rows.timestep_indices[24] == 0 && rows.adaln_indices[24] == 2,
            "Ref2VA timestep/AdaLN tag mapping drifted");
    const auto padded = trtmc::minimax_h3::pad_ref2va_timesteps({0.25F, 0.5F});
    require(padded.live_count == 2 && padded.values[0] == 0.25F && padded.values[1] == 0.5F &&
                padded.values[2] == 0.5F && padded.values[3] == 0.5F,
            "Ref2VA fixed-four-row timestep padding drifted");
}

void test_request_boundary_rejects_before_plan_execution() {
    trtmc::VideoGenerationRequest request;
    request.mode = trtmc::VideoGenerationMode::kReferenceToVideoAudio;
    trtmc::VideoReferenceInput audio;
    audio.kind = trtmc::VideoReferenceKind::kAudio;
    audio.audio.sample_rate = 32000;
    audio.audio.channels = 1;
    audio.audio.samples.resize(128000);
    audio.audio.num_samples = 128000;
    request.references.push_back(std::move(audio));
    require(rejects([&] { (void)trtmc::minimax_h3::prepare_ref2va_request(request, 124); }),
            "Ref2VA API accepted audio-only conditioning without an image or video");

    request.references.clear();
    trtmc::VideoReferenceInput short_video;
    short_video.kind = trtmc::VideoReferenceKind::kVideo;
    short_video.video.num_frames = 24;
    short_video.video.height = 1;
    short_video.video.width = 1;
    short_video.video.channels = 3;
    short_video.video.fps_numerator = 24;
    short_video.video.fps_denominator = 1;
    short_video.video.pixels.resize(24U * 3U);
    request.references.push_back(std::move(short_video));
    require(rejects([&] { (void)trtmc::minimax_h3::prepare_ref2va_request(request, 124); }),
            "Ref2VA API accepted a sub-two-second video");
}

void test_strict_plan_abi_and_fake_end_to_end() {
    auto adaln = make_adaln_module();
    auto denoiser = make_denoiser_module();
    trtmc::minimax_h3::validate_ref2va_plan(adaln,
                                            trtmc::minimax_h3::Ref2vaPlanKind::kAdalnPrecompute);
    trtmc::minimax_h3::validate_ref2va_plan(denoiser, trtmc::minimax_h3::Ref2vaPlanKind::kDenoiser);

    auto undersized = make_denoiser_module();
    undersized.tensors.at("encoder_hidden_states").maximum = {2641, 5120};
    require(rejects([&] {
                trtmc::minimax_h3::validate_ref2va_plan(
                    undersized, trtmc::minimax_h3::Ref2vaPlanKind::kDenoiser);
            }),
            "Ref2VA accepted an undersized/fallback text profile");

    const auto table = trtmc::minimax_h3::pad_ref2va_timesteps({0.5F, 1.0F});
    auto modulations = trtmc::minimax_h3::run_ref2va_adaln_precompute(adaln, table);
    trtmc::minimax_h3::Ref2vaDenoiserInputs inputs;
    constexpr int32_t text_rows = 1;
    constexpr int32_t audio_rows = 414;
    constexpr int32_t video_rows = 18870;
    constexpr int32_t packed_rows = text_rows + audio_rows + video_rows;
    inputs.layout.position_ids.resize(static_cast<std::size_t>(packed_rows) * 3U);
    inputs.layout.token_tags.resize(packed_rows, 0);
    inputs.layout.token_tags[0] = 1;
    inputs.layout.text_indices = {0};
    inputs.layout.audio_indices.resize(audio_rows);
    std::iota(inputs.layout.audio_indices.begin(), inputs.layout.audio_indices.end(), 1);
    inputs.layout.video_indices.resize(video_rows);
    std::iota(inputs.layout.video_indices.begin(), inputs.layout.video_indices.end(),
              1 + audio_rows);
    std::fill(inputs.layout.token_tags.begin() + 1,
              inputs.layout.token_tags.begin() + 1 + audio_rows, 2);
    inputs.video_hidden_states.resize(static_cast<std::size_t>(video_rows) * 96U);
    inputs.audio_hidden_states.resize(static_cast<std::size_t>(audio_rows) * 32U);
    inputs.encoder_hidden_states.resize(5120);
    inputs.timestep_indices.resize(packed_rows);
    inputs.adaln_indices.resize(packed_rows);
    const auto velocities = trtmc::minimax_h3::run_ref2va_denoiser(denoiser, inputs, modulations);
    require(velocities.video.size() == static_cast<std::size_t>(video_rows) * 96U &&
                velocities.audio.size() == static_cast<std::size_t>(audio_rows) * 32U &&
                velocities.video.front() == 0.25F && velocities.audio.front() == -0.5F,
            "Ref2VA fake native plan path did not complete scatter/gather output");
}

} // namespace

int main() {
    try {
        test_temporal_and_presentation_contract();
        test_audio_only_dummy_vision_path();
        test_scheduler_and_plugin_fail_closed_contract();
        test_reference_vae_fake_plan_paths();
        test_packed_layout_and_timesteps();
        test_request_boundary_rejects_before_plan_execution();
        test_strict_plan_abi_and_fake_end_to_end();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
