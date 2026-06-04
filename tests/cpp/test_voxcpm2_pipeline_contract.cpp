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
float last_cfg_value = 0.0F;
int32_t last_inference_timesteps = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

class FakeModule final : public trtmc::ITrtModule {
  public:
    FakeModule(std::string input_name = "", std::string output_name = "",
               trtmc::DType output_dtype = trtmc::DType::kFloat32,
               std::vector<float> output_floats = {1.0F, 2.0F},
               std::vector<std::string> extra_input_names = {})
        : input_name_(std::move(input_name)), output_name_(std::move(output_name)),
          output_dtype_(output_dtype), extra_input_names_(std::move(extra_input_names)) {
        set_float_output(std::move(output_floats));
    }

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        last_inputs_ = inputs;
        record_generation_controls(inputs);
        if (output_name_.empty())
            return {};

        trtmc::Tensor tensor;
        tensor.data = output_storage_.data();
        tensor.shape = output_shape_;
        tensor.dtype = output_dtype_;
        return {{output_name_, tensor}};
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

    void set_float_output(std::vector<float> values) {
        output_shape_ = {static_cast<int64_t>(values.size())};
        output_storage_.resize(values.size() * sizeof(float));
        if (!values.empty()) {
            std::memcpy(output_storage_.data(), values.data(), output_storage_.size());
        }
    }

    std::string input_name_;
    std::string output_name_;
    trtmc::DType output_dtype_;
    std::vector<std::string> extra_input_names_;
    std::vector<unsigned char> output_storage_;
    std::vector<int64_t> output_shape_;
    trtmc::TensorMap last_inputs_;
};

std::vector<audio::VoxCPM2LoadedComponent> make_fake_components() {
    std::vector<audio::VoxCPM2LoadedComponent> components;
    components.reserve(audio::kVoxCPM2ComponentSpecs.size());
    for (std::size_t i = 0; i < audio::kVoxCPM2ComponentSpecs.size(); ++i) {
        const auto& spec = audio::kVoxCPM2ComponentSpecs[i];
        const auto& stage = audio::kVoxCPM2GenerationStages[i];
        std::unique_ptr<trtmc::ITrtModule> module =
            std::make_unique<FakeModule>(stage.input_artifact, stage.output_artifact);
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

std::vector<audio::VoxCPM2LoadedComponent> make_scripted_components() {
    std::vector<audio::VoxCPM2LoadedComponent> components;
    components.reserve(audio::kVoxCPM2ComponentSpecs.size());
    const std::vector<std::vector<float>> stage_outputs = {
        {1.0F, 2.0F}, {3.0F, 4.0F}, {5.0F, 6.0F}, {7.0F, 8.0F}, {0.0F, 0.25F, -0.25F, 0.5F},
    };
    for (std::size_t i = 0; i < audio::kVoxCPM2ComponentSpecs.size(); ++i) {
        const auto& spec = audio::kVoxCPM2ComponentSpecs[i];
        const auto& stage = audio::kVoxCPM2GenerationStages[i];
        std::vector<std::string> extra_inputs;
        if (i == 0)
            extra_inputs = {"cfg_value", "inference_timesteps"};
        std::unique_ptr<trtmc::ITrtModule> module = std::make_unique<FakeModule>(
            stage.input_artifact, stage.output_artifact, trtmc::DType::kFloat32, stage_outputs[i],
            std::move(extra_inputs));
        components.push_back({spec.name, spec.engine_section, std::move(module)});
    }
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
    trtmc::VoxCPM2Pipeline pipeline(make_scripted_components(), plan, "openbmb/VoxCPM2");

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.cfg_scale = 3.0F;
    gen_cfg.num_steps = 12;
    gen_cfg.seed = 7;

    cfg_binding_hits = 0;
    timestep_binding_hits = 0;
    last_cfg_value = 0.0F;
    last_inference_timesteps = 0;

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

    const auto wav_path = temp_dir / "trt_output.wav";
    check(!std::filesystem::exists(wav_path),
          "voxcpm2 pipeline leaves WAV writing to generate-audio --output");

    std::filesystem::current_path(original_cwd);
    std::filesystem::remove_all(temp_dir);
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
    test_construct_reports_missing_stage_binding();
    test_rejects_component_order_mismatch();

    if (failures != 0) {
        std::cerr << failures << " VoxCPM2 pipeline contract test(s) failed\n";
        return 1;
    }
    std::cerr << "All VoxCPM2 pipeline contract tests passed.\n";
    return 0;
}
