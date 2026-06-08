// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-VOXCPM2-02
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-VOXCPM2-02
// Intent:         VoxCPM2Pipeline runtime boundary construction and
//                 generate_audio() stage execution contract.
// Preconditions:  No TensorRT SDK required; fake backend-neutral modules.
// Postconditions: Pipeline validates LocEnc->TSLM->RALM->LocDiT->AudioVAE
//                 module order, propagates artifact tensors, leaves the TRT
//                 WAV artifact path to the CLI, and reports exact missing stage
//                 bindings.
// =============================================================================

#include "runtime/models/voxcpm2/pipeline.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace audio = trtmc::runtime::builders::audio;

int failures = 0;
int cfg_binding_hits = 0;
int timestep_binding_hits = 0;
int local_text_feature_binding_hits = 0;
int locdit_aux_binding_hits = 0;
int tslm_text_binding_hits = 0;
float last_cfg_value = 0.0F;
int32_t last_inference_timesteps = 0;
float last_local_text_feature_value = 0.0F;
float last_feat_cond_value = 0.0F;
float last_lm_hidden_value = 0.0F;
float last_residual_hidden_value = 0.0F;
float last_audio_vae_latent_value = 0.0F;
std::vector<float> last_audio_vae_latent_values;
int64_t last_text_token_count = 0;
int64_t last_audio_feat_steps = 0;
int64_t last_feat_cond_rows = 0;
int64_t last_feat_cond_cols = 0;
int64_t last_audio_vae_latent_rows = 0;
int32_t last_first_text_token = 0;
int32_t last_second_text_token = 0;
int32_t last_audio_start_token = 0;
float last_text_mask_value = 0.0F;
float last_audio_mask_value = 0.0F;
std::vector<int32_t> last_text_tokens;
std::vector<float> last_text_mask_values;
std::vector<float> last_audio_mask_values;
std::vector<float> locdit_feat_cond_values;
std::vector<float> locenc_audio_feat_values;
std::vector<int64_t> tslm_text_token_counts;
std::vector<int32_t> tslm_audio_start_values;
std::vector<int64_t> locdit_lm_hidden_rows;
std::vector<int64_t> locdit_residual_hidden_rows;
std::vector<int32_t> position_id_values;
std::vector<int32_t> tslm_position_id_values;
std::vector<int32_t> ralm_position_id_values;
std::string last_tokenizer_input;
int tslm_cache_binding_hits = 0;
int ralm_cache_binding_hits = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

class FakeTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string& text) const override {
        last_tokenizer_input = text;
        if (text == "VoxCPM2 CJK token split") {
            return {500};
        }
        std::vector<int32_t> ids;
        ids.reserve(text.size());
        for (const unsigned char ch : text) {
            ids.push_back(static_cast<int32_t>(ch));
        }
        return ids;
    }

    std::string decode(const std::vector<int32_t>& ids) const override {
        std::string text;
        text.reserve(ids.size());
        for (const int32_t id : ids) {
            text.push_back(static_cast<char>(id));
        }
        return text;
    }

    int32_t id_for_token(std::string_view token) const override {
        if (token == "<|audio_start|>")
            return 101;
        if (token == "\xE4\xBD\xA0")
            return 201;
        if (token == "\xE5\xA5\xBD")
            return 202;
        return -1;
    }

    std::string token_for_id(int32_t id) const override {
        if (id == 500)
            return "\xE2\x96\x81\xE4\xBD\xA0\xE5\xA5\xBD";
        if (id == 201)
            return "\xE4\xBD\xA0";
        if (id == 202)
            return "\xE5\xA5\xBD";
        return std::string(1, static_cast<char>(id));
    }
};

std::shared_ptr<trtmc::ITokenizer> make_fake_tokenizer() {
    return std::make_shared<FakeTokenizer>();
}

class FakeModule final : public trtmc::ITrtModule {
  public:
    struct ExtraOutputSpec {
        std::string name;
        trtmc::DType dtype{trtmc::DType::kFloat32};
        std::vector<float> values;
        std::vector<int64_t> shape;
    };

    FakeModule(std::string input_name = "", std::string output_name = "",
               trtmc::DType output_dtype = trtmc::DType::kFloat32,
               std::vector<float> output_floats = {1.0F, 2.0F},
               std::vector<int64_t> output_shape = {},
               std::vector<std::string> extra_input_names = {},
               std::vector<ExtraOutputSpec> extra_outputs = {},
               std::vector<int64_t> max_profile_shape = {})
        : input_name_(std::move(input_name)), output_name_(std::move(output_name)),
          output_dtype_(output_dtype), extra_input_names_(std::move(extra_input_names)),
          max_profile_shape_(std::move(max_profile_shape)) {
        set_float_output(std::move(output_floats), std::move(output_shape));
        for (auto& output : extra_outputs) {
            add_extra_output(std::move(output));
        }
    }

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        last_inputs_ = inputs;
        record_generation_controls(inputs);
        record_auxiliary_inputs(inputs);
        trtmc::TensorMap outputs;
        if (output_name_.empty())
            return outputs;

        const auto runtime_output = runtime_float_output(inputs);
        trtmc::Tensor tensor;
        tensor.data = runtime_output.first;
        tensor.shape = runtime_output.second;
        tensor.dtype = output_dtype_;
        outputs.emplace(output_name_, tensor);
        for (auto& extra : extra_outputs_) {
            trtmc::Tensor extra_tensor;
            extra_tensor.data = extra.storage.data();
            extra_tensor.shape = extra.shape;
            extra_tensor.dtype = extra.dtype;
            outputs.emplace(extra.name, extra_tensor);
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
        if (input_name_.empty())
            return {};
        return {{input_name_, {-1}, trtmc::DType::kFloat32, true}};
    }
    std::vector<trtmc::TensorInfo> output_info() const override {
        if (output_name_.empty())
            return {};
        return {{output_name_, output_shape_, output_dtype_, false}};
    }
    bool has_input(const std::string& name) const override {
        return name == input_name_ ||
               std::find(extra_input_names_.begin(), extra_input_names_.end(), name) !=
                   extra_input_names_.end();
    }
    bool has_output(const std::string& name) const override {
        if (name == output_name_)
            return true;
        return std::any_of(extra_outputs_.begin(), extra_outputs_.end(),
                           [&](const ExtraOutput& output) { return output.name == name; });
    }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (name == "tslm_past_kv_cache" || name == "ralm_past_kv_cache")
            return {2, 1, 1, 1, 8, 1};
        return {};
    }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        if (!max_profile_shape_.empty())
            return max_profile_shape_;
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

    const trtmc::TensorMap& last_inputs() const { return last_inputs_; }

  private:
    struct ExtraOutput {
        std::string name;
        trtmc::DType dtype{trtmc::DType::kFloat32};
        std::vector<unsigned char> storage;
        std::vector<int64_t> shape;
    };

    void record_generation_controls(const trtmc::TensorMap& inputs) const {
        if (const auto it = inputs.find("cfg_value"); it != inputs.end()) {
            ++cfg_binding_hits;
            last_cfg_value = *static_cast<float*>(it->second.data);
        }
        if (const auto it = inputs.find("inference_timesteps"); it != inputs.end()) {
            ++timestep_binding_hits;
            last_inference_timesteps = *static_cast<int32_t*>(it->second.data);
        }
    }

    void record_auxiliary_inputs(const trtmc::TensorMap& inputs) const {
        if (const auto pos_it = inputs.find("position_id"); pos_it != inputs.end()) {
            const auto position_id = *static_cast<int32_t*>(pos_it->second.data);
            position_id_values.push_back(position_id);
            if (input_name_ == "local_text_features")
                tslm_position_id_values.push_back(position_id);
            if (input_name_ == "semantic_lm_states")
                ralm_position_id_values.push_back(position_id);
        }
        if (input_name_ == "local_text_features") {
            if (inputs.find("tslm_past_kv_cache") != inputs.end())
                ++tslm_cache_binding_hits;
        }
        if (input_name_ == "semantic_lm_states") {
            if (inputs.find("ralm_past_kv_cache") != inputs.end())
                ++ralm_cache_binding_hits;
        }
        if (const auto token_it = inputs.find("text_tokens"); token_it != inputs.end()) {
            if (const auto text_mask_it = inputs.find("text_mask");
                text_mask_it != inputs.end()) {
                if (const auto audio_mask_it = inputs.find("audio_mask");
                    audio_mask_it != inputs.end()) {
                    ++tslm_text_binding_hits;
                    last_text_token_count =
                        token_it->second.shape.empty() ? 0 : token_it->second.shape.front();
                    tslm_text_token_counts.push_back(last_text_token_count);
                    const auto* token_data = static_cast<int32_t*>(token_it->second.data);
                    last_text_tokens.assign(token_data, token_data + last_text_token_count);
                    if (last_text_token_count > 0) {
                        last_first_text_token = token_data[0];
                        last_audio_start_token = token_data[last_text_token_count - 1];
                        tslm_audio_start_values.push_back(last_audio_start_token);
                    }
                    if (last_text_token_count > 1)
                        last_second_text_token = token_data[1];
                    const auto* text_mask_data = static_cast<float*>(text_mask_it->second.data);
                    const auto* audio_mask_data = static_cast<float*>(audio_mask_it->second.data);
                    last_text_mask_values.assign(text_mask_data,
                                                 text_mask_data + last_text_token_count);
                    last_audio_mask_values.assign(audio_mask_data,
                                                  audio_mask_data + last_text_token_count);
                    last_text_mask_value = text_mask_data[0];
                    last_audio_mask_value = audio_mask_data[0];
                }
            }
        }
        if (input_name_ == "audio_feats") {
            if (const auto it = inputs.find("audio_feats"); it != inputs.end()) {
                last_audio_feat_steps =
                    it->second.shape.empty() ? 0 : it->second.shape.front();
                locenc_audio_feat_values.push_back(*static_cast<float*>(it->second.data));
            }
        }
        if (input_name_ != "local_text_features") {
            if (const auto it = inputs.find("local_text_features"); it != inputs.end()) {
                ++local_text_feature_binding_hits;
                last_local_text_feature_value = *static_cast<float*>(it->second.data);
            }
        }
        if (const auto lm_it = inputs.find("lm_hidden"); lm_it != inputs.end()) {
            if (const auto residual_it = inputs.find("residual_hidden");
                residual_it != inputs.end()) {
                if (const auto feat_cond_it = inputs.find("feat_cond");
                    feat_cond_it != inputs.end()) {
                    ++locdit_aux_binding_hits;
                    last_lm_hidden_value = *static_cast<float*>(lm_it->second.data);
                    last_residual_hidden_value = *static_cast<float*>(residual_it->second.data);
                    last_feat_cond_value = *static_cast<float*>(feat_cond_it->second.data);
                    locdit_feat_cond_values.push_back(last_feat_cond_value);
                    locdit_lm_hidden_rows.push_back(lm_it->second.shape.empty()
                                                        ? 0
                                                        : lm_it->second.shape.front());
                    locdit_residual_hidden_rows.push_back(residual_it->second.shape.empty()
                                                              ? 0
                                                              : residual_it->second.shape.front());
                    last_feat_cond_rows = feat_cond_it->second.shape.size() > 0
                                              ? feat_cond_it->second.shape[0]
                                              : 0;
                    last_feat_cond_cols = feat_cond_it->second.shape.size() > 1
                                              ? feat_cond_it->second.shape[1]
                                              : 0;
                }
            }
        }
        if (input_name_ == "audio_vae_latents") {
            if (const auto it = inputs.find("audio_vae_latents"); it != inputs.end()) {
                last_audio_vae_latent_rows =
                    it->second.shape.empty() ? 0 : it->second.shape.front();
                last_audio_vae_latent_value = *static_cast<float*>(it->second.data);
                const std::size_t value_count =
                    it->second.shape.size() == 2
                        ? static_cast<std::size_t>(it->second.shape[0]) *
                              static_cast<std::size_t>(it->second.shape[1])
                        : 0;
                const auto* values = static_cast<float*>(it->second.data);
                last_audio_vae_latent_values.assign(values, values + value_count);
            }
        }
    }

    void set_float_output(std::vector<float> values, std::vector<int64_t> shape) {
        output_values_ = values;
        output_shape_ = shape.empty() ? std::vector<int64_t>{static_cast<int64_t>(values.size())}
                                      : std::move(shape);
        output_storage_.resize(values.size() * sizeof(float));
        if (!values.empty()) {
            std::memcpy(output_storage_.data(), values.data(), output_storage_.size());
        }
    }

    void add_extra_output(ExtraOutputSpec spec) {
        ExtraOutput output;
        output.name = std::move(spec.name);
        output.dtype = spec.dtype;
        output.shape = spec.shape.empty()
                           ? std::vector<int64_t>{static_cast<int64_t>(spec.values.size())}
                           : std::move(spec.shape);
        output.storage.resize(spec.values.size() * sizeof(float));
        if (!spec.values.empty()) {
            std::memcpy(output.storage.data(), spec.values.data(), output.storage.size());
        }
        extra_outputs_.push_back(std::move(output));
    }

    std::pair<void*, std::vector<int64_t>> runtime_float_output(
        const trtmc::TensorMap& inputs) {
        if (output_shape_.empty() || output_shape_[0] != -1)
            return {output_storage_.data(), output_shape_};
        auto input_it = inputs.find(input_name_);
        const int64_t first_dim = input_it == inputs.end() || input_it->second.shape.empty()
                                      ? 1
                                      : input_it->second.shape.front();
        dynamic_output_shape_ = output_shape_;
        dynamic_output_shape_[0] = first_dim;
        std::size_t value_count = 1;
        for (const auto dim : dynamic_output_shape_)
            value_count *= static_cast<std::size_t>(dim);
        std::vector<float> values(value_count, output_values_.empty() ? 0.0F : output_values_[0]);
        for (std::size_t i = 0; i < value_count && !output_values_.empty(); ++i)
            values[i] = output_values_[i % output_values_.size()];
        dynamic_output_storage_.resize(values.size() * sizeof(float));
        if (!values.empty()) {
            std::memcpy(dynamic_output_storage_.data(), values.data(),
                        dynamic_output_storage_.size());
        }
        return {dynamic_output_storage_.data(), dynamic_output_shape_};
    }

    std::string input_name_;
    std::string output_name_;
    trtmc::DType output_dtype_;
    std::vector<std::string> extra_input_names_;
    std::vector<ExtraOutput> extra_outputs_;
    std::vector<float> output_values_;
    std::vector<unsigned char> output_storage_;
    std::vector<unsigned char> dynamic_output_storage_;
    std::vector<int64_t> output_shape_;
    std::vector<int64_t> dynamic_output_shape_;
    std::vector<int64_t> max_profile_shape_;
    trtmc::TensorMap last_inputs_;
};

std::vector<std::string> required_inputs_for_stage(const audio::VoxCPM2GenerationStage& stage) {
    std::vector<std::string> inputs;
    for (std::size_t i = 0; i < stage.required_side_input_count; ++i)
        inputs.emplace_back(stage.required_side_inputs[i]);
    for (std::size_t i = 0; i < stage.required_control_input_count; ++i)
        inputs.emplace_back(stage.required_control_inputs[i]);
    return inputs;
}

std::vector<audio::VoxCPM2LoadedComponent> make_fake_components() {
    std::vector<audio::VoxCPM2LoadedComponent> components;
    components.reserve(audio::kVoxCPM2ComponentSpecs.size());
    for (std::size_t i = 0; i < audio::kVoxCPM2ComponentSpecs.size(); ++i) {
        const auto& spec = audio::kVoxCPM2ComponentSpecs[i];
        const auto& stage = audio::kVoxCPM2GenerationStages[i];
        std::unique_ptr<trtmc::ITrtModule> module =
            std::make_unique<FakeModule>(stage.input_artifact, stage.output_artifact,
                                         trtmc::DType::kFloat32, std::vector<float>{1.0F, 2.0F},
                                         std::vector<int64_t>{}, required_inputs_for_stage(stage));
        components.push_back({spec.name, spec.engine_section, std::move(module)});
    }
    return components;
}

std::vector<audio::VoxCPM2LoadedComponent> make_components_missing_output_binding() {
    auto components = make_fake_components();
    components[0].module =
        std::make_unique<FakeModule>(audio::kVoxCPM2GenerationStages[0].input_artifact, "");
    return components;
}

std::vector<audio::VoxCPM2LoadedComponent> make_components_missing_locdit_side_binding() {
    auto components = make_fake_components();
    const auto& stage = audio::kVoxCPM2GenerationStages[3];
    components[3].module = std::make_unique<FakeModule>(
        stage.input_artifact, stage.output_artifact, trtmc::DType::kFloat32,
        std::vector<float>{1.0F, 2.0F}, std::vector<int64_t>{},
        std::vector<std::string>{"cfg_value", "inference_timesteps"});
    return components;
}

std::vector<float> repeated_values(std::size_t count, float value) {
    return std::vector<float>(count, value);
}

std::vector<audio::VoxCPM2LoadedComponent> make_scripted_components(
    std::size_t latent_patch_count = 2, bool cache_bound_lms = false,
    bool audio_vae_output0 = false, bool stop_logits_stop = false) {
    std::vector<audio::VoxCPM2LoadedComponent> components;
    components.reserve(audio::kVoxCPM2ComponentSpecs.size());
    std::vector<float> locdit_latents;
    for (std::size_t patch = 0; patch < latent_patch_count; ++patch) {
        auto values = repeated_values(4 * 64, 4.0F + static_cast<float>(patch));
        locdit_latents.insert(locdit_latents.end(), values.begin(), values.end());
    }
    const std::vector<std::vector<float>> stage_outputs = {
        repeated_values(2 * 64, 1.0F),  repeated_values(2 * 2048, 2.0F),
        repeated_values(1 * 512, 3.0F), std::move(locdit_latents),
        {0.0F, 0.25F, -0.25F, 0.5F},
    };
    const std::vector<std::vector<int64_t>> stage_shapes = {
        {cache_bound_lms ? -1 : 2, 64},
        {cache_bound_lms ? -1 : 2, 2048},
        {cache_bound_lms ? -1 : 1, 512},
        {static_cast<int64_t>(latent_patch_count * 4), 64},
        {4},
    };
    for (std::size_t i = 0; i < audio::kVoxCPM2ComponentSpecs.size(); ++i) {
        const auto& spec = audio::kVoxCPM2ComponentSpecs[i];
        const auto& stage = audio::kVoxCPM2GenerationStages[i];
        std::vector<std::string> extra_inputs = required_inputs_for_stage(stage);
        const std::string output_name =
            audio_vae_output0 && i == 4 ? "output0" : stage.output_artifact;
        if (i == 1 || i == 2) {
            extra_inputs.push_back("position_id");
        }
        std::vector<FakeModule::ExtraOutputSpec> extra_outputs;
        if (i == 1) {
            if (cache_bound_lms)
                extra_inputs.push_back("tslm_past_kv_cache");
            extra_outputs.push_back(
                {"lm_hidden", trtmc::DType::kFloat32, repeated_values(1 * 2048, 8.0F),
                 {1, 2048}});
            extra_outputs.push_back({"stop_logits", trtmc::DType::kFloat32,
                                     stop_logits_stop ? std::vector<float>{0.0F, 1.0F}
                                                      : std::vector<float>{1.0F, 0.0F},
                                     {1, 2}});
            if (cache_bound_lms) {
                extra_outputs.push_back({"tslm_present_kv_cache", trtmc::DType::kFloat32,
                                         repeated_values(2 * 1 * 1 * 1 * 8 * 1, 9.0F),
                                         {2, 1, 1, 1, 8, 1}});
            }
        }
        if (i == 2 && cache_bound_lms) {
            extra_inputs.push_back("ralm_past_kv_cache");
            extra_outputs.push_back({"ralm_present_kv_cache", trtmc::DType::kFloat32,
                                     repeated_values(2 * 1 * 1 * 1 * 8 * 1, 10.0F),
                                     {2, 1, 1, 1, 8, 1}});
        }
        std::unique_ptr<trtmc::ITrtModule> module = std::make_unique<FakeModule>(
            stage.input_artifact, output_name, trtmc::DType::kFloat32, stage_outputs[i],
            stage_shapes[i], std::move(extra_inputs), std::move(extra_outputs));
        components.push_back({spec.name, spec.engine_section, std::move(module)});
    }
    return components;
}

std::vector<audio::VoxCPM2LoadedComponent> make_components_with_bad_locenc_output_rank() {
    auto components = make_scripted_components();
    const auto& stage = audio::kVoxCPM2GenerationStages[0];
    components[0].module = std::make_unique<FakeModule>(
        stage.input_artifact, stage.output_artifact, trtmc::DType::kFloat32,
        std::vector<float>{1.0F, 2.0F}, std::vector<int64_t>{2});
    return components;
}

void test_constructs_with_loaded_component_contract() {
    trtmc::VoxCPM2Config cfg;
    auto plan = audio::make_voxcpm2_generation_plan(cfg);
    trtmc::VoxCPM2Pipeline pipeline(make_fake_components(), plan, "openbmb/VoxCPM2");

    check(std::string(pipeline.pipeline_type()) == "VoxCPM2Pipeline",
          "voxcpm2 pipeline type is explicit");
    check(std::string(pipeline.model_id()) == "openbmb/VoxCPM2",
          "voxcpm2 pipeline preserves model id");
}

void test_generate_audio_returns_component_waveform_without_hidden_wav_write() {
    const auto original_cwd = std::filesystem::current_path();
    const auto temp_dir =
        std::filesystem::temp_directory_path() / "trtmc_voxcpm2_pipeline_contract";
    std::filesystem::remove_all(temp_dir);
    std::filesystem::create_directories(temp_dir);
    std::filesystem::current_path(temp_dir);

    trtmc::VoxCPM2Config cfg;
    auto plan = audio::make_voxcpm2_generation_plan(cfg);
    trtmc::VoxCPM2Pipeline pipeline(make_scripted_components(), plan, "openbmb/VoxCPM2",
                                    make_fake_tokenizer());

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.cfg_scale = 3.0F;
    gen_cfg.num_steps = 12;
    gen_cfg.seed = 7;
    gen_cfg.max_new_tokens = 2;

    cfg_binding_hits = 0;
    timestep_binding_hits = 0;
    local_text_feature_binding_hits = 0;
    locdit_aux_binding_hits = 0;
    tslm_text_binding_hits = 0;
    last_cfg_value = 0.0F;
    last_inference_timesteps = 0;
    last_local_text_feature_value = 0.0F;
    last_feat_cond_value = 0.0F;
    last_lm_hidden_value = 0.0F;
    last_residual_hidden_value = 0.0F;
    last_audio_vae_latent_value = 0.0F;
    last_text_token_count = 0;
    last_audio_feat_steps = 0;
    last_feat_cond_rows = 0;
    last_feat_cond_cols = 0;
    last_audio_vae_latent_rows = 0;
    last_first_text_token = 0;
    last_second_text_token = 0;
    last_audio_start_token = 0;
    last_text_mask_value = 0.0F;
    last_audio_mask_value = 0.0F;
    last_text_tokens.clear();
    last_text_mask_values.clear();
    last_audio_mask_values.clear();
    last_audio_vae_latent_values.clear();
    locdit_feat_cond_values.clear();
    locenc_audio_feat_values.clear();
    tslm_text_token_counts.clear();
    tslm_audio_start_values.clear();
    locdit_lm_hidden_rows.clear();
    locdit_residual_hidden_rows.clear();
    position_id_values.clear();
    tslm_position_id_values.clear();
    ralm_position_id_values.clear();
    tslm_cache_binding_hits = 0;
    ralm_cache_binding_hits = 0;

    const auto audio = pipeline.generate_audio("VoxCPM2 parity prompt", gen_cfg);

    check(audio.sample_rate == 48000, "voxcpm2 audio sample rate is 48 kHz");
    check(audio.num_samples == 4, "voxcpm2 audio sample count comes from waveform_f32");
    check(audio.samples.size() == 4, "voxcpm2 audio samples are populated");
    check(audio.samples[1] == 0.25F, "voxcpm2 audio preserves waveform samples");
    check(cfg_binding_hits == 2,
          "voxcpm2 forwards cfg_value on each LocDiT autoregressive step");
    check(last_cfg_value == 3.0F, "voxcpm2 cfg_value uses GenerateConfig override");
    check(timestep_binding_hits == 2,
          "voxcpm2 forwards inference_timesteps on each LocDiT autoregressive step");
    check(last_inference_timesteps == 12,
          "voxcpm2 inference_timesteps uses GenerateConfig override");
    check(local_text_feature_binding_hits == 2,
          "voxcpm2 forwards local_text_features to initial and refreshed RALM");
    check(last_local_text_feature_value == 1.0F,
          "voxcpm2 preserved local_text_features retain stage output data");
    check(tslm_text_binding_hits == 2,
          "voxcpm2 calls TSLM for prefill and generated-patch refresh");
    check(!tslm_text_token_counts.empty() && tslm_text_token_counts[0] > 1,
          "voxcpm2 prefill text token tensor is populated");
    check(tslm_text_token_counts.size() == 2 && tslm_text_token_counts[1] == 1,
          "voxcpm2 generated-patch refresh uses one-step TSLM inputs");
    check(!tslm_audio_start_values.empty() && tslm_audio_start_values[0] == 101,
          "voxcpm2 appends upstream audio_start token during prefill");
    check(last_text_mask_values.size() == 1 && last_text_mask_values[0] == 0.0F,
          "voxcpm2 generated-patch refresh clears one-step text mask");
    check(last_audio_mask_values.size() == 1 && last_audio_mask_values[0] == 1.0F,
          "voxcpm2 generated-patch refresh marks one-step audio mask");
    check(locenc_audio_feat_values.size() == 2 && locenc_audio_feat_values[0] == 0.0F,
          "voxcpm2 prefill LocEnc starts from zero-shot audio features");
    check(locenc_audio_feat_values.size() == 2 && locenc_audio_feat_values[1] == 4.0F,
          "voxcpm2 refresh LocEnc receives first generated latent patch");
    check(position_id_values.size() == 2 && position_id_values[0] > 0 &&
              position_id_values[0] == position_id_values[1],
          "voxcpm2 forwards generated-step position_id to TSLM and RALM refresh");
    check(locdit_aux_binding_hits == 2,
          "voxcpm2 calls LocDiT once per generated latent patch");
    check(last_lm_hidden_value == 8.0F, "voxcpm2 LocDiT sees TSLM lm_hidden side tensor");
    check(last_residual_hidden_value == 3.0F,
          "voxcpm2 LocDiT sees RALM residual_hidden primary tensor");
    check(locdit_lm_hidden_rows.size() == 2 && locdit_lm_hidden_rows[0] == 1 &&
              locdit_lm_hidden_rows[1] == 1,
          "voxcpm2 LocDiT receives one TSLM hidden row per autoregressive step");
    check(locdit_residual_hidden_rows.size() == 2 && locdit_residual_hidden_rows[0] == 1 &&
              locdit_residual_hidden_rows[1] == 1,
          "voxcpm2 LocDiT receives one RALM hidden row per autoregressive step");
    check(last_feat_cond_rows == 4 && last_feat_cond_cols == 64,
          "voxcpm2 LocDiT sees feat_cond patch tensors");
    check(locdit_feat_cond_values.size() == 2,
          "voxcpm2 records one feat_cond value per LocDiT step");
    check(locdit_feat_cond_values.size() == 2 && locdit_feat_cond_values[0] == 0.0F,
          "voxcpm2 first LocDiT feat_cond uses zero previous latent");
    check(locdit_feat_cond_values.size() == 2 && locdit_feat_cond_values[1] == 4.0F,
          "voxcpm2 second LocDiT feat_cond uses first generated latent patch");
    check(last_audio_vae_latent_rows == 8,
          "voxcpm2 AudioVAE receives concatenated generated latent patches");
    check(last_audio_vae_latent_value == 4.0F,
          "voxcpm2 AudioVAE latent sequence starts with first generated patch");

    const auto wav_path = temp_dir / "trt_output.wav";
    check(!std::filesystem::exists(wav_path),
          "voxcpm2 pipeline leaves WAV writing to generate-audio --output");

    std::filesystem::current_path(original_cwd);
    std::filesystem::remove_all(temp_dir);
}

void test_generate_audio_maps_audio_vae_output0_to_waveform_artifact() {
    trtmc::VoxCPM2Config cfg;
    auto plan = audio::make_voxcpm2_generation_plan(cfg);
    trtmc::VoxCPM2Pipeline pipeline(make_scripted_components(2, false, true), plan,
                                    "openbmb/VoxCPM2", make_fake_tokenizer());

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 2;
    const auto audio = pipeline.generate_audio("VoxCPM2 parity prompt", gen_cfg);

    check(audio.num_samples == 4,
          "voxcpm2 maps Torch-TensorRT AudioVAE output0 to waveform_f32 samples");
    check(audio.samples.size() == 4 && audio.samples[1] == 0.25F,
          "voxcpm2 preserves waveform data from AudioVAE output0");
}

void test_generate_audio_trims_audio_vae_max_profile_output_to_generated_latents() {
    trtmc::VoxCPM2Config cfg;
    auto plan = audio::make_voxcpm2_generation_plan(cfg);
    auto components = make_scripted_components(2);
    components[4].module = std::make_unique<FakeModule>(
        "audio_vae_latents", "waveform_f32", trtmc::DType::kFloat32,
        repeated_values(16 * 3, 0.5F), std::vector<int64_t>{16 * 3},
        std::vector<std::string>{}, std::vector<FakeModule::ExtraOutputSpec>{},
        std::vector<int64_t>{16, 64});
    trtmc::VoxCPM2Pipeline pipeline(std::move(components), plan, "openbmb/VoxCPM2",
                                    make_fake_tokenizer());

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 2;
    const auto audio = pipeline.generate_audio("VoxCPM2 parity prompt", gen_cfg);

    check(last_audio_vae_latent_rows == 8,
          "voxcpm2 AudioVAE trim test generated eight latent frames");
    check(audio.num_samples == 8 * 3,
          "voxcpm2 trims AudioVAE max-profile waveform to generated latent frames");
    check(audio.samples.size() == 8 * 3,
          "voxcpm2 trimmed AudioVAE waveform sample vector matches num_samples");
}

void test_generate_audio_reads_first_locdit_patch_from_each_static_profile_invocation() {
    trtmc::VoxCPM2Config cfg;
    auto plan = audio::make_voxcpm2_generation_plan(cfg);
    trtmc::VoxCPM2Pipeline pipeline(make_scripted_components(2), plan, "openbmb/VoxCPM2",
                                    make_fake_tokenizer());

    last_audio_vae_latent_values.clear();
    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 2;
    (void)pipeline.generate_audio("VoxCPM2 parity prompt", gen_cfg);

    constexpr std::size_t kSecondPatchFirstValue = 4 * 64;
    check(last_audio_vae_latent_values.size() > kSecondPatchFirstValue,
          "voxcpm2 captures concatenated LocDiT patches for AudioVAE");
    check(last_audio_vae_latent_values[0] == 4.0F,
          "voxcpm2 first generated patch uses first LocDiT output patch");
    check(last_audio_vae_latent_values[kSecondPatchFirstValue] == 4.0F,
          "voxcpm2 static LocDiT profile reads the first output patch on each invocation");
}

void test_generate_audio_uses_tslm_stop_logits_after_upstream_min_len() {
    trtmc::VoxCPM2Config cfg;
    auto plan = audio::make_voxcpm2_generation_plan(cfg);
    trtmc::VoxCPM2Pipeline pipeline(make_scripted_components(8, false, false, true), plan,
                                    "openbmb/VoxCPM2", make_fake_tokenizer());

    locdit_aux_binding_hits = 0;
    last_audio_vae_latent_rows = 0;
    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 8;
    (void)pipeline.generate_audio("VoxCPM2 parity prompt", gen_cfg);

    check(locdit_aux_binding_hits == 4,
          "voxcpm2 consumes TSLM stop_logits after upstream minimum generation length");
    check(last_audio_vae_latent_rows == 16,
          "voxcpm2 AudioVAE receives only generated latent patches before stop");
}

void test_generate_audio_uses_cache_bound_lm_step_contract() {
    trtmc::VoxCPM2Config cfg;
    auto plan = audio::make_voxcpm2_generation_plan(cfg);
    trtmc::VoxCPM2Pipeline pipeline(make_scripted_components(2, true), plan,
                                    "openbmb/VoxCPM2", make_fake_tokenizer());

    tslm_text_binding_hits = 0;
    tslm_text_token_counts.clear();
    tslm_audio_start_values.clear();
    tslm_position_id_values.clear();
    ralm_position_id_values.clear();
    tslm_cache_binding_hits = 0;
    ralm_cache_binding_hits = 0;
    locdit_lm_hidden_rows.clear();
    locdit_residual_hidden_rows.clear();

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 2;
    (void)pipeline.generate_audio("ab", gen_cfg);

    check(tslm_text_binding_hits == 4,
          "voxcpm2 cache-bound TSLM runs one prefill step per text token plus refresh");
    check(tslm_cache_binding_hits == 4,
          "voxcpm2 cache-bound TSLM receives a past KV cache on every step");
    check(ralm_cache_binding_hits == 4,
          "voxcpm2 cache-bound RALM receives a past KV cache on every step");
    check(tslm_text_token_counts == std::vector<int64_t>({1, 1, 1, 1}),
          "voxcpm2 cache-bound TSLM uses one-row token inputs");
    check(tslm_audio_start_values.size() >= 3 && tslm_audio_start_values[2] == 101,
          "voxcpm2 cache-bound prefill still appends audio_start token");
    check(tslm_position_id_values == std::vector<int32_t>({0, 1, 2, 3}),
          "voxcpm2 cache-bound TSLM advances position ids through prefill and refresh");
    check(ralm_position_id_values == std::vector<int32_t>({0, 1, 2, 3}),
          "voxcpm2 cache-bound RALM advances position ids through prefill and refresh");
    check(locdit_lm_hidden_rows.size() == 2 && locdit_lm_hidden_rows[0] == 1 &&
              locdit_lm_hidden_rows[1] == 1,
          "voxcpm2 cache-bound LocDiT receives current TSLM hidden row");
    check(locdit_residual_hidden_rows.size() == 2 && locdit_residual_hidden_rows[0] == 1 &&
              locdit_residual_hidden_rows[1] == 1,
          "voxcpm2 cache-bound LocDiT receives current RALM hidden row");
}

void test_generate_audio_derives_upstream_default_steps_when_max_new_tokens_is_zero() {
    trtmc::VoxCPM2Config cfg;
    auto plan = audio::make_voxcpm2_generation_plan(cfg);
    constexpr std::size_t expected_default_steps = 22;
    trtmc::VoxCPM2Pipeline pipeline(make_scripted_components(expected_default_steps), plan,
                                    "openbmb/VoxCPM2", make_fake_tokenizer());

    cfg_binding_hits = 0;
    timestep_binding_hits = 0;
    locdit_aux_binding_hits = 0;
    last_audio_vae_latent_rows = 0;

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 0;
    (void)pipeline.generate_audio("a", gen_cfg);

    check(cfg_binding_hits == static_cast<int>(expected_default_steps),
          "voxcpm2 max_new_tokens=0 uses upstream text-length default for LocDiT steps");
    check(timestep_binding_hits == static_cast<int>(expected_default_steps),
          "voxcpm2 max_new_tokens=0 forwards timesteps on every default LocDiT step");
    check(locdit_aux_binding_hits == static_cast<int>(expected_default_steps),
          "voxcpm2 default generation length does not depend on hidden-state rows");
    check(last_audio_vae_latent_rows == static_cast<int64_t>(expected_default_steps * 4),
          "voxcpm2 AudioVAE receives all default generated latent patches");
}

void test_generate_audio_normalizes_prompt_before_voxcpm2_text_tokenizer() {
    trtmc::VoxCPM2Config cfg;
    auto plan = audio::make_voxcpm2_generation_plan(cfg);
    trtmc::VoxCPM2Pipeline pipeline(make_scripted_components(), plan, "openbmb/VoxCPM2",
                                    make_fake_tokenizer());

    tslm_text_binding_hits = 0;
    last_text_token_count = 0;
    last_tokenizer_input.clear();

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 1;
    (void)pipeline.generate_audio(
        "Hello, this is the VoxCPM2 TensorRT Model Connect parity test.", gen_cfg);

    const std::string expected =
        "Hello ,  this is the VoxCPM two TensorRT Model Connect parity test .";
    check(last_tokenizer_input == expected,
          "voxcpm2 normalize=true mirrors the upstream text tokenizer input");
    check(last_text_token_count == static_cast<int64_t>(expected.size() + 1),
          "voxcpm2 normalized token tensor appends audio_start after prepared text");
}

void test_generate_audio_can_disable_voxcpm2_text_normalization() {
    trtmc::VoxCPM2Config cfg;
    cfg.normalize = false;
    auto plan = audio::make_voxcpm2_generation_plan(cfg);
    trtmc::VoxCPM2Pipeline pipeline(make_scripted_components(), plan, "openbmb/VoxCPM2",
                                    make_fake_tokenizer());

    last_tokenizer_input.clear();

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 1;
    (void)pipeline.generate_audio("VoxCPM2 parity prompt", gen_cfg);

    check(last_tokenizer_input == "VoxCPM2 parity prompt",
          "voxcpm2 normalize=false preserves raw prompt tokenization");
}

void test_generate_audio_expands_voxcpm2_multichar_cjk_tokens() {
    trtmc::VoxCPM2Config cfg;
    cfg.normalize = false;
    auto plan = audio::make_voxcpm2_generation_plan(cfg);
    trtmc::VoxCPM2Pipeline pipeline(make_scripted_components(), plan, "openbmb/VoxCPM2",
                                    make_fake_tokenizer());

    tslm_text_binding_hits = 0;
    last_text_token_count = 0;
    last_first_text_token = 0;
    last_second_text_token = 0;
    last_audio_start_token = 0;

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 1;
    (void)pipeline.generate_audio("VoxCPM2 CJK token split", gen_cfg);

    check(tslm_text_binding_hits == 1, "voxcpm2 forwards expanded CJK tokens to TSLM");
    check(last_text_token_count == 3,
          "voxcpm2 multi-character CJK token expands before audio_start");
    check(last_first_text_token == 201, "voxcpm2 first CJK character id is preserved");
    check(last_second_text_token == 202, "voxcpm2 second CJK character id is preserved");
    check(last_audio_start_token == 101,
          "voxcpm2 resolves audio_start token through tokenizer");
}

void test_generate_audio_pads_text_tensors_to_engine_steps() {
    trtmc::VoxCPM2Config cfg;
    cfg.max_text_steps = 6;
    auto plan = audio::make_voxcpm2_generation_plan(cfg);
    trtmc::VoxCPM2Pipeline pipeline(make_scripted_components(), plan, "openbmb/VoxCPM2",
                                    make_fake_tokenizer());

    tslm_text_binding_hits = 0;
    last_text_token_count = 0;
    last_audio_feat_steps = 0;
    last_text_tokens.clear();
    last_text_mask_values.clear();
    last_audio_mask_values.clear();

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.max_new_tokens = 1;
    (void)pipeline.generate_audio("abc", gen_cfg);

    check(tslm_text_binding_hits == 1, "voxcpm2 forwards padded text tensors to TSLM");
    check(last_text_token_count == 6, "voxcpm2 pads text_tokens to engine text steps");
    check(last_audio_feat_steps == 6, "voxcpm2 pads audio_feats to engine text steps");
    check(last_text_tokens.size() == 6, "voxcpm2 captures padded token buffer");
    check(last_text_tokens[0] == static_cast<int32_t>('a'),
          "voxcpm2 preserves first prompt token before padding");
    check(last_text_tokens[3] == 101, "voxcpm2 places audio_start before token padding");
    check(last_text_tokens[4] == 0 && last_text_tokens[5] == 0,
          "voxcpm2 zero-pads unused token slots");
    check(last_text_mask_values.size() == 6, "voxcpm2 captures padded text mask");
    check(last_text_mask_values[0] == 1.0F && last_text_mask_values[3] == 1.0F,
          "voxcpm2 marks active prompt/audio_start tokens as text");
    check(last_text_mask_values[4] == 0.0F && last_text_mask_values[5] == 0.0F,
          "voxcpm2 clears padded text mask slots");
    check(last_audio_mask_values.size() == 6, "voxcpm2 captures padded audio mask");
    check(std::all_of(last_audio_mask_values.begin(), last_audio_mask_values.end(),
                      [](float value) { return value == 0.0F; }),
          "voxcpm2 zero-shot audio mask stays empty across padding");
}

void test_generate_audio_rejects_prompt_exceeding_engine_steps() {
    trtmc::VoxCPM2Config cfg;
    cfg.max_text_steps = 2;
    auto plan = audio::make_voxcpm2_generation_plan(cfg);
    trtmc::VoxCPM2Pipeline pipeline(make_scripted_components(), plan, "openbmb/VoxCPM2",
                                    make_fake_tokenizer());

    try {
        trtmc::GenerateConfig gen_cfg;
        (void)pipeline.generate_audio("abc", gen_cfg);
        check(false, "voxcpm2 rejects prompt longer than engine text steps");
    } catch (const std::runtime_error& e) {
        const std::string message = e.what();
        check(message.find("prompt token count 4 exceeds engine text step capacity 2") !=
                  std::string::npos,
              "voxcpm2 reports prompt length and engine text capacity");
    }
}

void test_construct_reports_missing_stage_binding() {
    trtmc::VoxCPM2Config cfg;
    auto plan = audio::make_voxcpm2_generation_plan(cfg);

    try {
        trtmc::VoxCPM2Pipeline pipeline(make_components_missing_output_binding(), plan,
                                        "openbmb/VoxCPM2");
        (void)pipeline;
        check(false, "voxcpm2 pipeline rejects missing stage output binding at construction");
    } catch (const std::runtime_error& e) {
        const std::string message = e.what();
        check(message.find("stage locenc") != std::string::npos,
              "voxcpm2 missing binding error names stage");
        check(message.find("locenc_engine_plan") != std::string::npos,
              "voxcpm2 missing binding error names engine section");
        check(message.find("local_text_features") != std::string::npos,
              "voxcpm2 missing binding error names missing output artifact");
    }
}

void test_construct_reports_missing_required_side_binding() {
    trtmc::VoxCPM2Config cfg;
    auto plan = audio::make_voxcpm2_generation_plan(cfg);

    try {
        trtmc::VoxCPM2Pipeline pipeline(make_components_missing_locdit_side_binding(), plan,
                                        "openbmb/VoxCPM2");
        (void)pipeline;
        check(false, "voxcpm2 pipeline rejects missing required side binding");
    } catch (const std::runtime_error& e) {
        const std::string message = e.what();
        check(message.find("stage locdit") != std::string::npos,
              "voxcpm2 missing side binding error names stage");
        check(message.find("locdit_engine_plan") != std::string::npos,
              "voxcpm2 missing side binding error names engine section");
        check(message.find("required side input binding 'lm_hidden'") != std::string::npos,
              "voxcpm2 missing side binding error names side input");
    }
}

void test_generate_audio_rejects_stage_tensor_contract_mismatch() {
    trtmc::VoxCPM2Config cfg;
    auto plan = audio::make_voxcpm2_generation_plan(cfg);
    trtmc::VoxCPM2Pipeline pipeline(make_components_with_bad_locenc_output_rank(), plan,
                                    "openbmb/VoxCPM2", make_fake_tokenizer());

    try {
        trtmc::GenerateConfig gen_cfg;
        (void)pipeline.generate_audio("VoxCPM2 parity prompt", gen_cfg);
        check(false, "voxcpm2 pipeline rejects wrong stage tensor rank");
    } catch (const std::runtime_error& e) {
        const std::string message = e.what();
        check(message.find("output artifact 'local_text_features' has rank 1") != std::string::npos,
              "voxcpm2 tensor contract error reports observed rank");
        check(message.find("expected 2 for local_text_features:float32|bfloat16") !=
                  std::string::npos,
              "voxcpm2 tensor contract error reports expected contract");
    }
}

void test_rejects_component_order_mismatch() {
    trtmc::VoxCPM2Config cfg;
    auto plan = audio::make_voxcpm2_generation_plan(cfg);
    auto components = make_fake_components();
    components[2].name = "locdit";

    try {
        trtmc::VoxCPM2Pipeline pipeline(std::move(components), plan, "openbmb/VoxCPM2");
        (void)pipeline;
        check(false, "voxcpm2 pipeline rejects component order mismatch");
    } catch (const std::runtime_error& e) {
        const std::string message = e.what();
        check(message.find("loaded component order does not match generation plan") !=
                  std::string::npos,
              "voxcpm2 pipeline reports component order mismatch");
    }
}

} // namespace

int main() {
    test_constructs_with_loaded_component_contract();
    test_generate_audio_returns_component_waveform_without_hidden_wav_write();
    test_generate_audio_maps_audio_vae_output0_to_waveform_artifact();
    test_generate_audio_trims_audio_vae_max_profile_output_to_generated_latents();
    test_generate_audio_reads_first_locdit_patch_from_each_static_profile_invocation();
    test_generate_audio_uses_tslm_stop_logits_after_upstream_min_len();
    test_generate_audio_uses_cache_bound_lm_step_contract();
    test_generate_audio_derives_upstream_default_steps_when_max_new_tokens_is_zero();
    test_generate_audio_normalizes_prompt_before_voxcpm2_text_tokenizer();
    test_generate_audio_can_disable_voxcpm2_text_normalization();
    test_generate_audio_expands_voxcpm2_multichar_cjk_tokens();
    test_generate_audio_pads_text_tensors_to_engine_steps();
    test_generate_audio_rejects_prompt_exceeding_engine_steps();
    test_construct_reports_missing_stage_binding();
    test_construct_reports_missing_required_side_binding();
    test_generate_audio_rejects_stage_tensor_contract_mismatch();
    test_rejects_component_order_mismatch();

    if (failures != 0) {
        std::cerr << failures << " VoxCPM2 pipeline contract test(s) failed\n";
        return 1;
    }
    std::cerr << "All VoxCPM2 pipeline contract tests passed.\n";
    return 0;
}
