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
float last_lm_hidden_value = 0.0F;
float last_residual_hidden_value = 0.0F;
int64_t last_text_token_count = 0;
int64_t last_audio_feat_steps = 0;
int32_t last_first_text_token = 0;
int32_t last_second_text_token = 0;
int32_t last_audio_start_token = 0;
float last_text_mask_value = 0.0F;
float last_audio_mask_value = 0.0F;
std::vector<int32_t> last_text_tokens;
std::vector<float> last_text_mask_values;
std::vector<float> last_audio_mask_values;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

class FakeTokenizer final : public trtmc::ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string& text) const override {
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
               std::vector<ExtraOutputSpec> extra_outputs = {})
        : input_name_(std::move(input_name)), output_name_(std::move(output_name)),
          output_dtype_(output_dtype), extra_input_names_(std::move(extra_input_names)) {
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

        trtmc::Tensor tensor;
        tensor.data = output_storage_.data();
        tensor.shape = output_shape_;
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
    bool has_output(const std::string& name) const override { return name == output_name_; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
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
        if (const auto token_it = inputs.find("text_tokens"); token_it != inputs.end()) {
            if (const auto text_mask_it = inputs.find("text_mask");
                text_mask_it != inputs.end()) {
                if (const auto audio_mask_it = inputs.find("audio_mask");
                    audio_mask_it != inputs.end()) {
                    ++tslm_text_binding_hits;
                    last_text_token_count =
                        token_it->second.shape.empty() ? 0 : token_it->second.shape.front();
                    const auto* token_data = static_cast<int32_t*>(token_it->second.data);
                    last_text_tokens.assign(token_data, token_data + last_text_token_count);
                    if (last_text_token_count > 0) {
                        last_first_text_token = token_data[0];
                        last_audio_start_token = token_data[last_text_token_count - 1];
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
                ++locdit_aux_binding_hits;
                last_lm_hidden_value = *static_cast<float*>(lm_it->second.data);
                last_residual_hidden_value = *static_cast<float*>(residual_it->second.data);
            }
        }
    }

    void set_float_output(std::vector<float> values, std::vector<int64_t> shape) {
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

    std::string input_name_;
    std::string output_name_;
    trtmc::DType output_dtype_;
    std::vector<std::string> extra_input_names_;
    std::vector<ExtraOutput> extra_outputs_;
    std::vector<unsigned char> output_storage_;
    std::vector<int64_t> output_shape_;
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

std::vector<audio::VoxCPM2LoadedComponent> make_scripted_components() {
    std::vector<audio::VoxCPM2LoadedComponent> components;
    components.reserve(audio::kVoxCPM2ComponentSpecs.size());
    const std::vector<std::vector<float>> stage_outputs = {
        repeated_values(2 * 64, 1.0F),  repeated_values(2 * 2048, 2.0F),
        repeated_values(2 * 512, 3.0F), repeated_values(2 * 64, 4.0F),
        {0.0F, 0.25F, -0.25F, 0.5F},
    };
    const std::vector<std::vector<int64_t>> stage_shapes = {
        {2, 64}, {2, 2048}, {2, 512}, {2, 64}, {4},
    };
    for (std::size_t i = 0; i < audio::kVoxCPM2ComponentSpecs.size(); ++i) {
        const auto& spec = audio::kVoxCPM2ComponentSpecs[i];
        const auto& stage = audio::kVoxCPM2GenerationStages[i];
        std::vector<std::string> extra_inputs = required_inputs_for_stage(stage);
        std::vector<FakeModule::ExtraOutputSpec> extra_outputs;
        if (i == 1) {
            extra_outputs.push_back(
                {"lm_hidden", trtmc::DType::kFloat32, repeated_values(2 * 2048, 8.0F), {2, 2048}});
        }
        std::unique_ptr<trtmc::ITrtModule> module = std::make_unique<FakeModule>(
            stage.input_artifact, stage.output_artifact, trtmc::DType::kFloat32, stage_outputs[i],
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

    cfg_binding_hits = 0;
    timestep_binding_hits = 0;
    local_text_feature_binding_hits = 0;
    locdit_aux_binding_hits = 0;
    tslm_text_binding_hits = 0;
    last_cfg_value = 0.0F;
    last_inference_timesteps = 0;
    last_local_text_feature_value = 0.0F;
    last_lm_hidden_value = 0.0F;
    last_residual_hidden_value = 0.0F;
    last_text_token_count = 0;
    last_audio_feat_steps = 0;
    last_first_text_token = 0;
    last_second_text_token = 0;
    last_audio_start_token = 0;
    last_text_mask_value = 0.0F;
    last_audio_mask_value = 0.0F;
    last_text_tokens.clear();
    last_text_mask_values.clear();
    last_audio_mask_values.clear();

    const auto audio = pipeline.generate_audio("VoxCPM2 parity prompt", gen_cfg);

    check(audio.sample_rate == 48000, "voxcpm2 audio sample rate is 48 kHz");
    check(audio.num_samples == 4, "voxcpm2 audio sample count comes from waveform_f32");
    check(audio.samples.size() == 4, "voxcpm2 audio samples are populated");
    check(audio.samples[1] == 0.25F, "voxcpm2 audio preserves waveform samples");
    check(cfg_binding_hits == 1, "voxcpm2 forwards cfg_value when stage declares it");
    check(last_cfg_value == 3.0F, "voxcpm2 cfg_value uses GenerateConfig override");
    check(timestep_binding_hits == 1,
          "voxcpm2 forwards inference_timesteps when stage declares it");
    check(last_inference_timesteps == 12,
          "voxcpm2 inference_timesteps uses GenerateConfig override");
    check(local_text_feature_binding_hits == 1,
          "voxcpm2 forwards preserved local_text_features to RALM");
    check(last_local_text_feature_value == 1.0F,
          "voxcpm2 preserved local_text_features retain stage output data");
    check(tslm_text_binding_hits == 1,
          "voxcpm2 forwards tokenizer-derived text tensors to TSLM");
    check(last_text_token_count > 1, "voxcpm2 text token tensor is populated");
    check(last_audio_start_token == 101, "voxcpm2 appends upstream audio_start token");
    check(last_text_mask_value == 1.0F, "voxcpm2 marks prompt tokens as text");
    check(last_audio_mask_value == 0.0F, "voxcpm2 zero-shot prompt has no audio mask");
    check(locdit_aux_binding_hits == 1,
          "voxcpm2 forwards lm_hidden and residual_hidden tensors to LocDiT");
    check(last_lm_hidden_value == 8.0F, "voxcpm2 LocDiT sees TSLM lm_hidden side tensor");
    check(last_residual_hidden_value == 3.0F,
          "voxcpm2 LocDiT sees RALM residual_hidden primary tensor");

    const auto wav_path = temp_dir / "trt_output.wav";
    check(!std::filesystem::exists(wav_path),
          "voxcpm2 pipeline leaves WAV writing to generate-audio --output");

    std::filesystem::current_path(original_cwd);
    std::filesystem::remove_all(temp_dir);
}

void test_generate_audio_expands_voxcpm2_multichar_cjk_tokens() {
    trtmc::VoxCPM2Config cfg;
    auto plan = audio::make_voxcpm2_generation_plan(cfg);
    trtmc::VoxCPM2Pipeline pipeline(make_scripted_components(), plan, "openbmb/VoxCPM2",
                                    make_fake_tokenizer());

    tslm_text_binding_hits = 0;
    last_text_token_count = 0;
    last_first_text_token = 0;
    last_second_text_token = 0;
    last_audio_start_token = 0;

    trtmc::GenerateConfig gen_cfg;
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
