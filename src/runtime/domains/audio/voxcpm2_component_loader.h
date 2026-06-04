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
};

inline std::vector<VoxCPM2LoadedComponent>
load_voxcpm2_component_modules(::trtmc::IBackend* backend, const ::trtmc::BundleFile& bundle,
                               const ::trtmc::ModuleCreateOptions& options) {
    std::vector<VoxCPM2LoadedComponent> components;
    components.reserve(kVoxCPM2ComponentSpecs.size());

    for (const auto& spec : kVoxCPM2ComponentSpecs) {
        auto loaded = ::trtmc::load_trt_module_from_plan(
            backend, ::trtmc::find_section(bundle, spec.engine_section), spec.engine_section,
            options);
        components.push_back({spec.name, spec.engine_section, std::move(loaded.module)});
    }

    return components;
}

} // namespace trtmc::runtime::builders::audio
