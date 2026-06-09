#pragma once

#include "runtime/domains/audio/voxcpm2_component_contract.h"
#include "runtime/plugins/shared/plugin_helpers.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::runtime::builders::audio {

struct VoxCPM2LoadedComponent {
    std::string name;
    std::string engine_section;
    std::unique_ptr<::trtmc::ITrtModule> module;
    std::string prefill_engine_section;
    std::unique_ptr<::trtmc::ITrtModule> prefill_module;
};

inline const char* voxcpm2_prefill_engine_section_for(const char* component_name) {
    if (std::string(component_name) == "tslm")
        return "tslm_prefill_engine_plan";
    if (std::string(component_name) == "ralm")
        return "ralm_prefill_engine_plan";
    return nullptr;
}

inline std::vector<VoxCPM2LoadedComponent>
load_voxcpm2_component_modules(::trtmc::IBackend* backend, const ::trtmc::BundleFile& bundle,
                               const ::trtmc::ModuleCreateOptions& options) {
    std::vector<VoxCPM2LoadedComponent> components;
    components.reserve(kVoxCPM2ComponentSpecs.size());

    for (const auto& spec : kVoxCPM2ComponentSpecs) {
        auto loaded = ::trtmc::load_trt_module_from_plan(
            backend, ::trtmc::find_section(bundle, spec.engine_section), spec.engine_section,
            options);
        VoxCPM2LoadedComponent component{spec.name, spec.engine_section, std::move(loaded.module),
                                         "", nullptr};
        if (const auto* prefill_section = voxcpm2_prefill_engine_section_for(spec.name)) {
            if (const auto* plan = ::trtmc::find_section(bundle, prefill_section)) {
                auto prefill_loaded =
                    ::trtmc::load_trt_module_from_plan(backend, plan, prefill_section, options);
                component.prefill_engine_section = prefill_section;
                component.prefill_module = std::move(prefill_loaded.module);
            }
        }
        components.push_back(std::move(component));
    }

    return components;
}

} // namespace trtmc::runtime::builders::audio
