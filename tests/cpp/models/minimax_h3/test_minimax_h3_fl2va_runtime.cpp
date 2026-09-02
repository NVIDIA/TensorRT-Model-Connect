/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/conditioning.h"
#include "runtime/models/minimax_h3/fl2va_runtime.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

class FakeTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string& text) const override {
        if (text.rfind("<Picture ", 0) == 0)
            return {101, 102, 103, 104, 105, 106};
        if (text.empty())
            return {};
        return {201, 202};
    }
    std::string decode(const std::vector<int32_t>&) const override { return {}; }
    int32_t id_for_token(std::string_view) const override { return -1; }
    std::string token_for_id(int32_t) const override { return {}; }
};

enum class ModuleKind { kVision, kText, kKeyframeVae };

class FakeModule final : public trtmc::ITrtModule {
  public:
    explicit FakeModule(ModuleKind kind) : kind_(kind) {
        if (kind == ModuleKind::kVision) {
            add_input("pixel_values", trtmc::DType::kFloat32);
            add_input("interp_indices", trtmc::DType::kInt32);
            add_input("interp_weights", trtmc::DType::kFloat32);
            add_input("vision_position_ids", trtmc::DType::kInt32);
            for (const char* name : {"vision_embeds", "deepstack_0", "deepstack_1", "deepstack_2"})
                add_output(name, trtmc::DType::kFloat32);
        } else if (kind == ModuleKind::kText) {
            for (const char* name :
                 {"input_ids", "mrope_position_ids", "vision_count", "vision_row_indices"})
                add_input(name, trtmc::DType::kInt32);
            for (const char* name :
                 {"vision_mask", "vision_embeds", "deepstack_0", "deepstack_1", "deepstack_2"})
                add_input(name, trtmc::DType::kFloat32);
            add_output("encoder_hidden_states", trtmc::DType::kFloat32);
        } else {
            add_input("pixel_tiles", trtmc::DType::kFloat32);
            add_output("posterior_parameter_tiles", trtmc::DType::kFloat32);
        }
    }

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        last_inputs = inputs;
        output_storage_.clear();
        trtmc::TensorMap outputs;
        if (kind_ == ModuleKind::kVision) {
            const int64_t patches = inputs.at("pixel_values").shape.at(0);
            const int64_t rows = patches / 4;
            for (const char* name :
                 {"vision_embeds", "deepstack_0", "deepstack_1", "deepstack_2"}) {
                auto& values = output_storage_[name];
                values.assign(static_cast<std::size_t>(rows) * 5120, static_cast<float>(rows));
                outputs.emplace(name,
                                trtmc::Tensor{values.data(), {rows, 5120}, trtmc::DType::kFloat32});
            }
        } else if (kind_ == ModuleKind::kText) {
            const int64_t rows = inputs.at("input_ids").shape.at(0);
            observed_vision_count = *static_cast<const int32_t*>(inputs.at("vision_count").data);
            auto& values = output_storage_["encoder_hidden_states"];
            values.assign(static_cast<std::size_t>(rows) * 5120, 0.25F);
            outputs.emplace("encoder_hidden_states",
                            trtmc::Tensor{values.data(), {rows, 5120}, trtmc::DType::kFloat32});
        } else {
            const int64_t tiles = inputs.at("pixel_tiles").shape.at(0);
            auto& values = output_storage_["posterior_parameter_tiles"];
            values.assign(static_cast<std::size_t>(tiles) * 48 * 16 * 16, 0.0F);
            outputs.emplace(
                "posterior_parameter_tiles",
                trtmc::Tensor{values.data(), {tiles, 48, 1, 16, 16}, trtmc::DType::kFloat32});
        }
        return outputs;
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override {
        std::vector<trtmc::TensorInfo> result;
        for (const auto& [name, unused] : inputs_) {
            (void)unused;
            result.push_back({name, tensor_shape(name), dtypes_.at(name), true});
        }
        return result;
    }
    std::vector<trtmc::TensorInfo> output_info() const override {
        std::vector<trtmc::TensorInfo> result;
        for (const auto& [name, unused] : outputs_) {
            (void)unused;
            result.push_back({name, tensor_shape(name), dtypes_.at(name), false});
        }
        return result;
    }
    bool has_input(const std::string& name) const override { return inputs_.count(name) != 0; }
    bool has_output(const std::string& name) const override { return outputs_.count(name) != 0; }
    trtmc::DType tensor_dtype(const std::string& name) const override { return dtypes_.at(name); }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (kind_ == ModuleKind::kVision) {
            if (outputs_.count(name) != 0)
                return {(profile_max_override > 0 ? profile_max_override : 4176) / 4, 5120};
            return input_profile_shape(name, 0, trtmc::ProfileShapeSelector::kMax);
        }
        if (kind_ == ModuleKind::kText) {
            if (name == "encoder_hidden_states")
                return {profile_max_override > 0 ? profile_max_override : 2641, 5120};
            if (name == "vision_count")
                return {1};
            return input_profile_shape(name, 0, trtmc::ProfileShapeSelector::kMax);
        }
        if (name == "posterior_parameter_tiles")
            return {33, 48, 1, 16, 16};
        return input_profile_shape(name, 0, trtmc::ProfileShapeSelector::kMax);
    }
    std::vector<int64_t> input_profile_shape(const std::string& name, int32_t,
                                             trtmc::ProfileShapeSelector selector) const override {
        const int32_t profile = selector == trtmc::ProfileShapeSelector::kMin
                                    ? 0
                                    : (selector == trtmc::ProfileShapeSelector::kOpt ? 1 : 2);
        if (kind_ == ModuleKind::kVision) {
            const int64_t rows[3] = {2304, 4032,
                                     profile_max_override > 0 ? profile_max_override : 4176};
            if (name == "pixel_values")
                return {rows[profile], 1536};
            if (name == "vision_position_ids")
                return {rows[profile], 2};
            return {rows[profile], 4};
        }
        if (kind_ == ModuleKind::kText) {
            const int64_t rows[3] = {1, 1144,
                                     profile_max_override > 0 ? profile_max_override : 2641};
            const int64_t vision_rows[3] = {1, 1008,
                                            profile_max_override == 262144 ? 262144 : 2088};
            if (name == "mrope_position_ids")
                return {3, rows[profile]};
            if (name == "vision_mask")
                return {rows[profile], 1};
            if (name == "vision_row_indices")
                return {vision_rows[profile]};
            if (name == "vision_embeds" || name.rfind("deepstack_", 0) == 0)
                return {vision_rows[profile], 5120};
            return {rows[profile]};
        }
        const int64_t rows[3] = {1, 28, 33};
        return {rows[profile], 3, 1, 256, 256};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    int32_t input_rank(const std::string& name) const override {
        return static_cast<int32_t>(tensor_shape(name).size());
    }
    bool input_is_dynamic(const std::string& name) const override {
        return !(kind_ == ModuleKind::kText && name == "vision_count");
    }
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

    trtmc::TensorMap last_inputs;
    int32_t observed_vision_count{-1};
    int64_t profile_max_override{0};

  private:
    void add_input(const std::string& name, trtmc::DType dtype) {
        inputs_[name] = true;
        dtypes_[name] = dtype;
    }
    void add_output(const std::string& name, trtmc::DType dtype) {
        outputs_[name] = true;
        dtypes_[name] = dtype;
    }

    ModuleKind kind_;
    std::unordered_map<std::string, bool> inputs_;
    std::unordered_map<std::string, bool> outputs_;
    std::unordered_map<std::string, trtmc::DType> dtypes_;
    std::unordered_map<std::string, std::vector<float>> output_storage_;
};

trtmc::VideoImageInput make_image(int32_t height, int32_t width) {
    trtmc::VideoImageInput image;
    image.height = height;
    image.width = width;
    image.channels = 3;
    image.pixels.resize(static_cast<std::size_t>(height) * width * 3);
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            for (int32_t channel = 0; channel < 3; ++channel) {
                image.pixels[(static_cast<std::size_t>(y) * width + x) * 3 + channel] =
                    static_cast<float>(y * width + x + channel) /
                    static_cast<float>(height * width + 3);
            }
        }
    }
    return image;
}

void test_official_qwen_presentation_and_mock_plans() {
    FakeTokenizer tokenizer;
    const auto presentation =
        trtmc::minimax_h3::make_fl2va_text_presentation("prompt", 2, 32, 32, tokenizer);
    check(presentation.input_ids.size() == 20 && presentation.vision_row_indices.size() == 2,
          "FL2VA presentation accounts for labels, boundaries, pads, and prompt");
    check(presentation.input_ids[6] == 151652 && presentation.input_ids[7] == 151655 &&
              presentation.input_ids[8] == 151653,
          "FL2VA presentation uses released Qwen vision special IDs");
    check(presentation.vision_row_indices == std::vector<int32_t>({7, 16}),
          "FL2VA scatter rows point only at image_pad tokens");
    check(presentation.token_tags[6] == 0 && presentation.token_tags[7] == 0 &&
              presentation.token_tags.back() == 1,
          "FL2VA text tags distinguish picture blocks from natural text");
    check(presentation.mrope_position_ids.size() == presentation.input_ids.size() * 3,
          "FL2VA presentation emits all three Qwen MRoPE axes");

    const auto image = make_image(32, 32);
    const auto vision_inputs = trtmc::minimax_h3::make_fl2va_vision_inputs(image);
    check(vision_inputs.patch_rows == 4 && vision_inputs.pixel_values.size() == 4U * 1536,
          "FL2VA Qwen image patchifier emits merge-block-major patch rows");
    check(vision_inputs.vision_position_ids == std::vector<int32_t>({0, 0, 0, 1, 1, 0, 1, 1}),
          "FL2VA Qwen position IDs preserve merge-block inner ordering");
    check(vision_inputs.interp_indices.front() == 0 &&
              vision_inputs.interp_indices[12] == 48 * 47 + 47,
          "FL2VA Qwen learned-position interpolation reaches exact corners");
    check(vision_inputs.pixel_values[0] == vision_inputs.pixel_values[256],
          "FL2VA still image is duplicated on Qwen's temporal patch axis");

    FakeModule vision(ModuleKind::kVision);
    auto one_features = trtmc::minimax_h3::run_fl2va_vision_encoder(vision, vision_inputs);
    check(one_features.rows == 1 &&
              vision.last_inputs.at("pixel_values").shape == std::vector<int64_t>({4, 1536}),
          "FL2VA vision mock receives the frozen native plan ABI");
    trtmc::minimax_h3::Fl2vaVisionFeatures both_features;
    both_features.rows = 2;
    for (auto pair : {std::pair{&both_features.vision_embeds, &one_features.vision_embeds},
                      std::pair{&both_features.deepstack_0, &one_features.deepstack_0},
                      std::pair{&both_features.deepstack_1, &one_features.deepstack_1},
                      std::pair{&both_features.deepstack_2, &one_features.deepstack_2}}) {
        pair.first->insert(pair.first->end(), pair.second->begin(), pair.second->end());
        pair.first->insert(pair.first->end(), pair.second->begin(), pair.second->end());
    }
    FakeModule text(ModuleKind::kText);
    const auto embeddings =
        trtmc::minimax_h3::run_fl2va_text_encoder(text, presentation, both_features);
    check(embeddings.size() == presentation.input_ids.size() * 5120 &&
              text.last_inputs.at("mrope_position_ids").shape == std::vector<int64_t>({3, 20}) &&
              text.observed_vision_count == 2,
          "FL2VA unified text mock receives exact MRoPE/scatter ABI");

    FakeModule forged_vision(ModuleKind::kVision);
    forged_vision.profile_max_override = 5000;
    bool rejected_forged_profile = false;
    try {
        trtmc::minimax_h3::validate_fl2va_plan(forged_vision,
                                               trtmc::minimax_h3::Fl2vaPlanKind::kVisionEncoder);
    } catch (const std::runtime_error&) {
        rejected_forged_profile = true;
    }
    check(rejected_forged_profile,
          "FL2VA runtime rejects a merely-larger unauthenticated Qwen profile");

    FakeModule superset_vision(ModuleKind::kVision);
    superset_vision.profile_max_override = 65536;
    trtmc::minimax_h3::validate_fl2va_plan(superset_vision,
                                           trtmc::minimax_h3::Fl2vaPlanKind::kVisionEncoder);
    FakeModule superset_text(ModuleKind::kText);
    superset_text.profile_max_override = 262144;
    trtmc::minimax_h3::validate_fl2va_plan(superset_text,
                                           trtmc::minimax_h3::Fl2vaPlanKind::kTextEncoder);
    check(true, "FL2VA runtime accepts the exact authenticated Ref2VA Qwen superset");
}

void test_keyframe_vae_mock_and_posterior_helpers() {
    FakeModule vae(ModuleKind::kKeyframeVae);
    const auto latent =
        trtmc::minimax_h3::run_fl2va_keyframe_vae_encoder(vae, make_image(768, 768));
    check(latent.size() == 24U * 48 * 48 &&
              vae.last_inputs.at("pixel_tiles").shape == std::vector<int64_t>({16, 3, 1, 256, 256}),
          "FL2VA keyframe VAE mock receives the 16-tile square-canvas ABI");
    bool finite = true;
    for (float value : latent)
        finite = finite && std::isfinite(value);
    check(finite, "FL2VA seed-42 posterior sample and FP16 round stay finite");

    constexpr int32_t latent_height = 2;
    constexpr int32_t latent_width = 2;
    std::vector<float> posterior(24U * latent_height * latent_width * 2, 0.0F);
    std::vector<float> epsilon(24U * latent_height * latent_width, 0.0F);
    const auto normalized = trtmc::minimax_h3::sample_and_normalize_fl2va_posterior(
        posterior, latent_height, latent_width, epsilon);
    const auto rows =
        trtmc::minimax_h3::patchify_fl2va_keyframe_latent(normalized, latent_height, latent_width);
    check(rows.size() == 96 && rows[0] == normalized[0] && rows[4] == normalized[4],
          "FL2VA posterior normalizes channel-major then patchifies in DiT order");
}

void test_structured_request_keyframe_modes() {
    const auto first = make_image(32, 64);
    const auto last = make_image(64, 32);
    FakeTokenizer tokenizer;
    trtmc::VideoGenerationRequest request;
    request.prompt = "prompt";
    request.mode = trtmc::VideoGenerationMode::kFirstLastFrameToVideoAudio;

    const auto execute = [&](const trtmc::VideoGenerationRequest& structured, int32_t frames) {
        std::vector<std::string> sections;
        auto result = trtmc::minimax_h3::run_fl2va_conditioning(
            structured, 768, 768, frames, tokenizer,
            [&](const std::string& section) -> std::unique_ptr<trtmc::ITrtModule> {
                sections.push_back(section);
                if (section == "fl2va_keyframe_vae_encoder_plan")
                    return std::make_unique<FakeModule>(ModuleKind::kKeyframeVae);
                if (section == "vision_encoder_plan")
                    return std::make_unique<FakeModule>(ModuleKind::kVision);
                if (section == "text_encoder_plan")
                    return std::make_unique<FakeModule>(ModuleKind::kText);
                throw std::runtime_error("unexpected mock FL2VA section");
            });
        check(sections == std::vector<std::string>({"fl2va_keyframe_vae_encoder_plan",
                                                    "vision_encoder_plan", "text_encoder_plan"}),
              "FL2VA structured request executes all native conditioning plans in order");
        check(result.text_embeddings.size() == result.text_token_tags.size() * 5120U,
              "FL2VA structured request returns the unified text-plan rows");
        return result;
    };

    request.first_frame = first;
    request.last_frame.reset();
    auto conditioned = execute(request, 124);
    check(conditioned.keyframes.anchors == std::vector<int32_t>({0}) &&
              conditioned.keyframe_latents.size() == 1,
          "FL2VA structured request preserves first-only semantics");

    request.first_frame.reset();
    request.last_frame = last;
    conditioned = execute(request, 345);
    check(conditioned.keyframes.anchors == std::vector<int32_t>({344}) &&
              conditioned.keyframe_latents.size() == 1,
          "FL2VA structured request preserves last-only semantics");

    request.first_frame = first;
    conditioned = execute(request, 345);
    check(conditioned.keyframes.anchors == std::vector<int32_t>({0, 344}) &&
              conditioned.keyframe_latents.size() == 2,
          "FL2VA structured request preserves both endpoint semantics");

    request.config.negative_prompt = "unsupported";
    bool plan_loaded = false;
    bool rejected_negative_prompt = false;
    try {
        (void)trtmc::minimax_h3::run_fl2va_conditioning(
            request, 768, 768, 345, tokenizer,
            [&](const std::string&) -> std::unique_ptr<trtmc::ITrtModule> {
                plan_loaded = true;
                return nullptr;
            });
    } catch (const std::invalid_argument&) {
        rejected_negative_prompt = true;
    }
    check(rejected_negative_prompt && !plan_loaded,
          "FL2VA guidance-distilled contract rejects negative_prompt before plan loading");
}

} // namespace

int main() {
    test_official_qwen_presentation_and_mock_plans();
    test_keyframe_vae_mock_and_posterior_helpers();
    test_structured_request_keyframe_modes();
    if (failures != 0)
        std::cerr << failures << " MiniMax-H3 FL2VA runtime test(s) failed\n";
    return failures == 0 ? 0 : 1;
}
