#include "runtime/models/voxcpm2/pipeline.h"

#include "runtime/domains/audio/voxcpm2_config.h"

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace trtmc {

namespace audio = runtime::builders::audio;

VoxCPM2Pipeline::VoxCPM2Pipeline(
    std::vector<audio::VoxCPM2LoadedComponent> components, audio::VoxCPM2GenerationPlan plan,
    std::string model_id_str)
    : components_(std::move(components)), plan_(std::move(plan)), model_id_(std::move(model_id_str)) {
    validate_components();
}

void VoxCPM2Pipeline::validate_components() const {
    if (!audio::voxcpm2_generation_plan_matches_component_contract()) {
        throw std::runtime_error(
            "VoxCPM2Pipeline: generation plan no longer matches component contract");
    }
    if (components_.size() != plan_.stages.size()) {
        throw std::runtime_error("VoxCPM2Pipeline: expected " +
                                 std::to_string(plan_.stages.size()) +
                                 " loaded component modules, got " +
                                 std::to_string(components_.size()));
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
    }
}

std::string VoxCPM2Pipeline::describe_loaded_components() const {
    std::ostringstream os;
    for (std::size_t i = 0; i < components_.size(); ++i) {
        if (i > 0)
            os << " -> ";
        os << components_[i].name << "(" << components_[i].engine_section << ")";
    }
    return os.str();
}

AudioResult VoxCPM2Pipeline::generate_audio(const std::string& prompt,
                                            const GenerateConfig& cfg) {
    auto effective_cfg = plan_.config;
    if (cfg.cfg_scale >= 0.0F)
        effective_cfg.cfg_value = cfg.cfg_scale;
    if (cfg.num_steps > 0)
        effective_cfg.inference_timesteps = cfg.num_steps;
    if (cfg.seed >= 0)
        effective_cfg.seed = cfg.seed;

    const auto effective_plan = audio::make_voxcpm2_generation_plan(effective_cfg);
    throw std::runtime_error(
        "VoxCPM2Pipeline::generate_audio is not implemented yet. Loaded component pipeline: " +
        describe_loaded_components() + ". Full openbmb/VoxCPM2 support must execute " +
        "LocEnc -> TSLM -> RALM -> LocDiT -> AudioVAE for the prompt, produce waveform_f32, "
        "and write " +
        std::string(effective_plan.output_wav_artifact) +
        " for exact comparison against the Hugging Face VoxCPM reference WAV. prompt_bytes=" +
        std::to_string(prompt.size()) + "; effective generation config: " +
        describe_voxcpm2_config(effective_plan.config) + ". " +
        audio::describe_voxcpm2_generation_plan(effective_plan));
}

} // namespace trtmc
