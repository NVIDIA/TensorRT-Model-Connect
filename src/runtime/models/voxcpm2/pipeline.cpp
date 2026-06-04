#include "runtime/models/voxcpm2/pipeline.h"

#include "runtime/domains/audio/voxcpm2_config.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trtmc {

namespace audio = runtime::builders::audio;

namespace {

struct OwnedStageTensor {
    std::vector<unsigned char> storage;
    std::vector<int64_t> shape;
    DType dtype{DType::kFloat32};

    Tensor as_tensor() const {
        Tensor tensor;
        tensor.data = storage.empty() ? nullptr : const_cast<unsigned char*>(storage.data());
        tensor.shape = shape;
        tensor.dtype = dtype;
        return tensor;
    }
};

const char* dtype_name(DType dtype) {
    switch (dtype) {
    case DType::kFloat32:
        return "float32";
    case DType::kFloat16:
        return "float16";
    case DType::kBFloat16:
        return "bfloat16";
    case DType::kInt32:
        return "int32";
    case DType::kInt8:
        return "int8";
    }
    return "unknown";
}

OwnedStageTensor copy_stage_output(const Tensor& tensor, const audio::VoxCPM2GenerationStage& stage,
                                   const std::string& component_name) {
    if (tensor.data == nullptr && tensor.nbytes() > 0) {
        throw std::runtime_error("VoxCPM2Pipeline: stage " + component_name +
                                 " returned null data for output artifact '" +
                                 stage.output_artifact + "'");
    }

    OwnedStageTensor owned;
    owned.shape = tensor.shape;
    owned.dtype = tensor.dtype;
    owned.storage.resize(tensor.nbytes());
    if (!owned.storage.empty())
        std::memcpy(owned.storage.data(), tensor.data, owned.storage.size());
    return owned;
}

OwnedStageTensor make_prompt_tensor(const std::string& prompt) {
    OwnedStageTensor tensor;
    tensor.shape = {static_cast<int64_t>(prompt.size())};
    tensor.dtype = DType::kInt8;
    tensor.storage.resize(prompt.size());
    if (!prompt.empty()) {
        std::memcpy(tensor.storage.data(), prompt.data(), prompt.size());
    }
    return tensor;
}

struct RuntimeScalarInputs {
    explicit RuntimeScalarInputs(const VoxCPM2Config& cfg)
        : sample_rate(cfg.sample_rate), reference_sample_rate(cfg.reference_sample_rate),
          cfg_value(cfg.cfg_value), inference_timesteps(cfg.inference_timesteps),
          normalize(cfg.normalize ? 1 : 0), denoise(cfg.denoise ? 1 : 0),
          retry_badcase(cfg.retry_badcase ? 1 : 0),
          retry_badcase_max_times(cfg.retry_badcase_max_times),
          retry_badcase_ratio_threshold(cfg.retry_badcase_ratio_threshold) {
        if (cfg.seed < static_cast<int64_t>(std::numeric_limits<int32_t>::min()) ||
            cfg.seed > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error("VoxCPM2Pipeline: seed is outside int32 runtime range");
        }
        seed = static_cast<int32_t>(cfg.seed);
    }

    void add_to(const ITrtModule& module, TensorMap& inputs) const {
        add_int32(module, inputs, "sample_rate", sample_rate);
        add_int32(module, inputs, "reference_sample_rate", reference_sample_rate);
        add_float32(module, inputs, "cfg_value", cfg_value);
        add_int32(module, inputs, "inference_timesteps", inference_timesteps);
        add_int32(module, inputs, "normalize", normalize);
        add_int32(module, inputs, "denoise", denoise);
        add_int32(module, inputs, "retry_badcase", retry_badcase);
        add_int32(module, inputs, "retry_badcase_max_times", retry_badcase_max_times);
        add_float32(module, inputs, "retry_badcase_ratio_threshold", retry_badcase_ratio_threshold);
        add_int32(module, inputs, "seed", seed);
    }

    int32_t sample_rate;
    int32_t reference_sample_rate;
    float cfg_value;
    int32_t inference_timesteps;
    int32_t normalize;
    int32_t denoise;
    int32_t retry_badcase;
    int32_t retry_badcase_max_times;
    float retry_badcase_ratio_threshold;
    int32_t seed;

  private:
    static void add_int32(const ITrtModule& module, TensorMap& inputs, const char* name,
                          const int32_t& value) {
        if (!module.has_input(name) || inputs.find(name) != inputs.end())
            return;
        Tensor tensor;
        tensor.data = const_cast<int32_t*>(&value);
        tensor.shape = {1};
        tensor.dtype = DType::kInt32;
        inputs.emplace(name, tensor);
    }

    static void add_float32(const ITrtModule& module, TensorMap& inputs, const char* name,
                            const float& value) {
        if (!module.has_input(name) || inputs.find(name) != inputs.end())
            return;
        Tensor tensor;
        tensor.data = const_cast<float*>(&value);
        tensor.shape = {1};
        tensor.dtype = DType::kFloat32;
        inputs.emplace(name, tensor);
    }
};

void validate_stage_bindings(const audio::VoxCPM2LoadedComponent& component,
                             const audio::VoxCPM2GenerationStage& stage) {
    if (!component.module->has_input(stage.input_artifact)) {
        throw std::runtime_error(
            "VoxCPM2Pipeline: stage " + component.name + " (" + component.engine_section +
            ") is missing required input binding '" + stage.input_artifact + "'");
    }
    if (!component.module->has_output(stage.output_artifact)) {
        throw std::runtime_error(
            "VoxCPM2Pipeline: stage " + component.name + " (" + component.engine_section +
            ") is missing required output binding '" + stage.output_artifact + "'");
    }
}

OwnedStageTensor run_stage(const audio::VoxCPM2LoadedComponent& component,
                           const audio::VoxCPM2GenerationStage& stage,
                           const OwnedStageTensor& input, const RuntimeScalarInputs& controls) {
    validate_stage_bindings(component, stage);
    TensorMap inputs;
    inputs.emplace(stage.input_artifact, input.as_tensor());
    controls.add_to(*component.module, inputs);
    auto outputs = component.module->forward(inputs);
    const auto output_it = outputs.find(stage.output_artifact);
    if (output_it == outputs.end()) {
        throw std::runtime_error("VoxCPM2Pipeline: stage " + component.name +
                                 " did not return output artifact '" + stage.output_artifact + "'");
    }

    return copy_stage_output(output_it->second, stage, component.name);
}

AudioResult make_audio_result(const OwnedStageTensor& waveform,
                              const audio::VoxCPM2GenerationPlan& plan) {
    if (waveform.dtype != DType::kFloat32) {
        throw std::runtime_error("VoxCPM2Pipeline: expected waveform_f32 as float32, got " +
                                 std::string(dtype_name(waveform.dtype)));
    }
    if (waveform.storage.size() % sizeof(float) != 0) {
        throw std::runtime_error("VoxCPM2Pipeline: waveform_f32 byte size is not float-aligned");
    }
    const auto sample_count = waveform.storage.size() / sizeof(float);
    if (sample_count > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("VoxCPM2Pipeline: waveform_f32 has too many samples");
    }

    AudioResult out;
    out.samples.resize(sample_count);
    if (!out.samples.empty()) {
        std::memcpy(out.samples.data(), waveform.storage.data(), waveform.storage.size());
    }
    out.num_samples = static_cast<int32_t>(out.samples.size());
    out.sample_rate = plan.config.sample_rate;
    return out;
}

} // namespace

VoxCPM2Pipeline::VoxCPM2Pipeline(std::vector<audio::VoxCPM2LoadedComponent> components,
                                 audio::VoxCPM2GenerationPlan plan, std::string model_id_str)
    : components_(std::move(components)), plan_(std::move(plan)),
      model_id_(std::move(model_id_str)) {
    validate_components();
}

void VoxCPM2Pipeline::validate_components() const {
    if (!audio::voxcpm2_generation_plan_matches_component_contract()) {
        throw std::runtime_error(
            "VoxCPM2Pipeline: generation plan no longer matches component contract");
    }
    if (components_.size() != plan_.stages.size()) {
        throw std::runtime_error(
            "VoxCPM2Pipeline: expected " + std::to_string(plan_.stages.size()) +
            " loaded component modules, got " + std::to_string(components_.size()));
    }

    for (std::size_t i = 0; i < plan_.stages.size(); ++i) {
        const auto& component = components_[i];
        const auto& stage = plan_.stages[i];
        if (component.name != stage.name || component.engine_section != stage.engine_section) {
            throw std::runtime_error(
                "VoxCPM2Pipeline: loaded component order does not match generation plan at stage " +
                std::to_string(i) + " (expected " + stage.name + "/" + stage.engine_section +
                ", got " + component.name + "/" + component.engine_section + ")");
        }
        if (component.module == nullptr || !component.module->ok()) {
            throw std::runtime_error("VoxCPM2Pipeline: invalid loaded module for stage " +
                                     component.name);
        }
        validate_stage_bindings(component, stage);
    }
}

AudioResult VoxCPM2Pipeline::generate_audio(const std::string& prompt, const GenerateConfig& cfg) {
    auto effective_cfg = plan_.config;
    if (cfg.cfg_scale >= 0.0F)
        effective_cfg.cfg_value = cfg.cfg_scale;
    if (cfg.num_steps > 0)
        effective_cfg.inference_timesteps = cfg.num_steps;
    if (cfg.seed >= 0)
        effective_cfg.seed = cfg.seed;

    const auto effective_plan = audio::make_voxcpm2_generation_plan(effective_cfg);
    const RuntimeScalarInputs controls(effective_plan.config);
    OwnedStageTensor current = make_prompt_tensor(prompt);
    for (std::size_t i = 0; i < effective_plan.stages.size(); ++i) {
        current = run_stage(components_[i], effective_plan.stages[i], current, controls);
    }

    auto audio = make_audio_result(current, effective_plan);
    return audio;
}

} // namespace trtmc
