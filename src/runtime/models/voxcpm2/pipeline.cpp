#include "runtime/models/voxcpm2/pipeline.h"

#include "runtime/domains/audio/voxcpm2_config.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
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

using StageArtifacts = std::unordered_map<std::string, OwnedStageTensor>;

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

std::string describe_tensor_contract(const audio::VoxCPM2TensorContract& contract) {
    std::ostringstream os;
    os << contract.name << ":" << audio::voxcpm2_dtype_contract_name(contract.dtype_contract) << "["
       << contract.symbolic_shape << "]";
    return os.str();
}

void validate_tensor_contract(const Tensor& tensor, const audio::VoxCPM2TensorContract& contract,
                              const audio::VoxCPM2GenerationStage& stage,
                              const std::string& component_name, const char* direction) {
    if (!audio::voxcpm2_dtype_matches(contract.dtype_contract, tensor.dtype)) {
        throw std::runtime_error("VoxCPM2Pipeline: stage " + component_name + " (" +
                                 stage.engine_section + ") " + direction + " artifact '" +
                                 contract.name + "' has dtype " + dtype_name(tensor.dtype) +
                                 ", expected " + describe_tensor_contract(contract));
    }
    if (tensor.shape.size() != contract.rank) {
        throw std::runtime_error(
            "VoxCPM2Pipeline: stage " + component_name + " (" + stage.engine_section + ") " +
            direction + " artifact '" + contract.name + "' has rank " +
            std::to_string(tensor.shape.size()) + ", expected " + std::to_string(contract.rank) +
            " for " + describe_tensor_contract(contract));
    }
}

OwnedStageTensor copy_stage_tensor(const Tensor& tensor, const std::string& artifact_name,
                                   const std::string& component_name) {
    if (tensor.data == nullptr && tensor.nbytes() > 0) {
        throw std::runtime_error("VoxCPM2Pipeline: stage " + component_name +
                                 " returned null data for output artifact '" + artifact_name + "'");
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

OwnedStageTensor make_zero_audio_feats_tensor(const std::string& prompt, const VoxCPM2Config& cfg) {
    if (cfg.patch_size <= 0 || cfg.feat_dim <= 0) {
        throw std::runtime_error("VoxCPM2Pipeline: patch_size and feat_dim must be positive");
    }

    OwnedStageTensor tensor;
    const auto text_steps = static_cast<int64_t>(std::max<std::size_t>(1, prompt.size() + 1));
    tensor.shape = {text_steps, cfg.patch_size, cfg.feat_dim};
    tensor.dtype = DType::kFloat32;
    const auto value_count =
        static_cast<std::size_t>(text_steps) * static_cast<std::size_t>(cfg.patch_size) *
        static_cast<std::size_t>(cfg.feat_dim);
    tensor.storage.resize(value_count * sizeof(float));
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
    for (std::size_t i = 0; i < stage.required_side_input_count; ++i) {
        const auto* name = stage.required_side_inputs[i];
        if (!component.module->has_input(name)) {
            throw std::runtime_error("VoxCPM2Pipeline: stage " + component.name + " (" +
                                     component.engine_section +
                                     ") is missing required side input binding '" + name + "'");
        }
    }
    for (std::size_t i = 0; i < stage.required_control_input_count; ++i) {
        const auto* name = stage.required_control_inputs[i];
        if (!component.module->has_input(name)) {
            throw std::runtime_error("VoxCPM2Pipeline: stage " + component.name + " (" +
                                     component.engine_section +
                                     ") is missing required control input binding '" + name + "'");
        }
    }
}

void add_required_artifact_inputs(const audio::VoxCPM2GenerationStage& stage,
                                  const StageArtifacts& artifacts, TensorMap& inputs,
                                  const std::string& component_name) {
    for (std::size_t i = 0; i < stage.required_side_input_count; ++i) {
        const auto* name = stage.required_side_inputs[i];
        if (inputs.find(name) != inputs.end())
            continue;
        const auto artifact_it = artifacts.find(name);
        if (artifact_it == artifacts.end()) {
            throw std::runtime_error("VoxCPM2Pipeline: stage " + component_name +
                                     " is missing required side artifact '" + name + "'");
        }
        inputs.emplace(name, artifact_it->second.as_tensor());
    }
}

void add_declared_artifact_inputs(const ITrtModule& module, const StageArtifacts& artifacts,
                                  TensorMap& inputs) {
    for (const auto& artifact : artifacts) {
        if (inputs.find(artifact.first) != inputs.end())
            continue;
        if (!module.has_input(artifact.first))
            continue;
        inputs.emplace(artifact.first, artifact.second.as_tensor());
    }
}

OwnedStageTensor run_stage(const audio::VoxCPM2LoadedComponent& component,
                           const audio::VoxCPM2GenerationStage& stage, StageArtifacts& artifacts,
                           const RuntimeScalarInputs& controls) {
    validate_stage_bindings(component, stage);
    const auto input_it = artifacts.find(stage.input_artifact);
    if (input_it == artifacts.end()) {
        throw std::runtime_error("VoxCPM2Pipeline: stage " + component.name +
                                 " is missing prepared input artifact '" + stage.input_artifact +
                                 "'");
    }
    const auto& input = input_it->second;
    validate_tensor_contract(input.as_tensor(), stage.input_tensor, stage, component.name, "input");
    TensorMap inputs;
    inputs.emplace(stage.input_artifact, input.as_tensor());
    add_required_artifact_inputs(stage, artifacts, inputs, component.name);
    add_declared_artifact_inputs(*component.module, artifacts, inputs);
    controls.add_to(*component.module, inputs);
    auto outputs = component.module->forward(inputs);
    const auto output_it = outputs.find(stage.output_artifact);
    if (output_it == outputs.end()) {
        throw std::runtime_error("VoxCPM2Pipeline: stage " + component.name +
                                 " did not return output artifact '" + stage.output_artifact + "'");
    }

    validate_tensor_contract(output_it->second, stage.output_tensor, stage, component.name,
                             "output");
    artifacts[stage.output_artifact] =
        copy_stage_tensor(output_it->second, stage.output_artifact, component.name);
    for (const auto& output : outputs) {
        if (output.first == stage.output_artifact)
            continue;
        artifacts[output.first] = copy_stage_tensor(output.second, output.first, component.name);
    }
    return artifacts.at(stage.output_artifact);
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
    StageArtifacts artifacts;
    artifacts.emplace("text_utf8", make_prompt_tensor(prompt));
    artifacts.emplace("audio_feats", make_zero_audio_feats_tensor(prompt, effective_plan.config));
    OwnedStageTensor current = artifacts.at("audio_feats");
    for (std::size_t i = 0; i < effective_plan.stages.size(); ++i) {
        current = run_stage(components_[i], effective_plan.stages[i], artifacts, controls);
    }

    auto audio = make_audio_result(current, effective_plan);
    return audio;
}

} // namespace trtmc
