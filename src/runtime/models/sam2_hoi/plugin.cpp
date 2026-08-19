/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_format.h"
#include "bundle/bundle_view.h"
#include "runtime/models/sam2_hoi/cuda_stream.h"
#include "runtime/models/sam2_hoi/pafpn_composite.h"
#include "runtime/models/sam2_hoi/pipeline.h"
#include "runtime/models/sam2_hoi/sam2_hoi_video_session.h"
#include "trtmc/models/sam2_hoi_video.h"
#include "trtmc/runtime/pipeline_plugin.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <iomanip>
#include <linux/memfd.h>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/syscall.h>
#include <system_error>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

using PlanSectionMap = std::unordered_map<std::string, BundleSectionInfo>;

std::string pafpn_plan_section(std::size_t ordinal) {
    std::ostringstream name;
    name << "sam2_hoi_pafpn_plan_" << std::setw(3) << std::setfill('0') << ordinal;
    return name.str();
}

std::unordered_set<std::string> phase_a_plan_sections() {
    std::unordered_set<std::string> required{
        "engine_plan",
        "sam2_hoi_detector_engine_plan",
        "sam2_hoi_interaction_engine_plan",
        "sam2_hoi_prompt_tracker_engine_plan",
        "sam2_hoi_recurrent_tracker_engine_plan",
        "sam2_hoi_memory_encoder_engine_plan",
    };
    for (std::size_t ordinal = 0; ordinal < sam2_hoi::kPafpnPlanCount; ++ordinal)
        required.insert(pafpn_plan_section(ordinal));
    return required;
}

PlanSectionMap index_phase_a_plan_sections(const BundleInfo& info) {
    const auto required = phase_a_plan_sections();

    PlanSectionMap indexed;
    std::unordered_set<std::string> all_names;
    for (const auto& section : info.sections) {
        if (!all_names.insert(section.name).second)
            throw std::runtime_error("SAM2 HOI bundle contains duplicate section " + section.name);
        if (section.name.rfind("sam2_hoi_pafpn_plan_", 0) == 0 &&
            required.count(section.name) == 0) {
            throw std::runtime_error("SAM2 HOI bundle contains a non-canonical PAFPN section " +
                                     section.name);
        }
        if (required.count(section.name) != 0) {
            if (section.size == 0)
                throw std::runtime_error("SAM2 HOI bundle contains an empty plan " + section.name);
            indexed.emplace(section.name, section);
        }
    }
    if (indexed.size() != required.size()) {
        for (const auto& name : required) {
            if (indexed.count(name) == 0)
                throw std::runtime_error("SAM2 HOI bundle is missing " + name);
        }
        throw std::runtime_error("SAM2 HOI Phase-A plan inventory is incomplete");
    }
    return indexed;
}

void validate_phase_a_eager_sections(const BundleFile& bundle) {
    const std::unordered_set<std::string> required{
        "config.json",
        "sam2_hoi_pafpn_manifest.json",
        "sam2_hoi_native_plugin_so",
    };
    const auto plans = phase_a_plan_sections();
    std::unordered_set<std::string> actual;
    for (const auto& section : bundle.sections) {
        if (section.data.empty() || !actual.insert(section.name).second)
            throw std::runtime_error("SAM2 HOI eager bundle section contract drift");
        if (plans.count(section.name) != 0)
            throw std::runtime_error("SAM2 HOI plan sections must be loaded lazily");
    }
    for (const auto& name : required) {
        if (actual.count(name) == 0)
            throw std::runtime_error("SAM2 HOI staged bundle is missing eager section " + name);
    }
}

std::unique_ptr<ITrtModule> load_staged_module(IBackend& backend, const std::string& bundle_path,
                                               const PlanSectionMap& sections,
                                               const std::string& section,
                                               const ModuleCreateOptions& options) {
    const auto found = sections.find(section);
    if (found == sections.end() || found->second.size == 0)
        throw std::runtime_error("SAM2 HOI bundle is missing " + section);
    auto bytes = ReadBundleSection(bundle_path, found->second);
    auto module = backend.create_module(bytes.data(), bytes.size(), options);
    if (module == nullptr || !module->ok() || module->stream() == nullptr)
        throw std::runtime_error("SAM2 HOI could not deserialize " + section);
    module->set_timing_label(section);
    return module;
}

std::unique_ptr<ITrtModule> load_legacy_module(const PipelineContext& context,
                                               const std::string& section,
                                               const ModuleCreateOptions& options) {
    const auto* plan = find_section(context.bundle, section);
    if (plan == nullptr || plan->empty())
        throw std::runtime_error("SAM2 HOI bundle is missing " + section);
    if (context.backend == nullptr)
        throw std::runtime_error("SAM2 HOI pipeline has no TensorRT backend");
    auto module = context.backend->create_module(plan->data(), plan->size(), options);
    if (module == nullptr || !module->ok() || module->stream() == nullptr)
        throw std::runtime_error("SAM2 HOI could not deserialize " + section);
    module->set_timing_label(section);
    return module;
}

bool header_has_section(const BundleInfo& info, const std::string& name) {
    return std::any_of(info.sections.begin(), info.sections.end(),
                       [&](const auto& section) { return section.name == name; });
}

struct RetainedNativePlugin {
    int descriptor{-1};
    void* handle{nullptr};
    std::vector<char> bundle_bytes;
    std::string external_path;
};

std::vector<RetainedNativePlugin>& retained_native_plugins() {
    // TensorRT retains creator state from the DSO. Keep both the dlopen handle
    // and any backing memfd alive until process exit.
    static auto* plugins = new std::vector<RetainedNativePlugin>();
    return *plugins;
}

std::mutex& native_plugin_mutex() {
    static auto* mutex = new std::mutex();
    return *mutex;
}

[[noreturn]] void close_descriptor_and_throw(int descriptor, const char* action) {
    const int error = errno;
    if (descriptor >= 0)
        (void)::close(descriptor);
    throw std::system_error(error, std::generic_category(), action);
}

int create_sealed_native_memfd(const std::vector<char>& bytes) {
    const int descriptor = static_cast<int>(
        ::syscall(SYS_memfd_create, "trtmc-sam2-hoi-native", MFD_CLOEXEC | MFD_ALLOW_SEALING));
    if (descriptor < 0)
        throw std::system_error(errno, std::generic_category(),
                                "Unable to create SAM2 HOI native plugin memfd");

    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const auto written = ::write(descriptor, bytes.data() + offset, bytes.size() - offset);
        if (written < 0) {
            if (errno == EINTR)
                continue;
            close_descriptor_and_throw(descriptor, "Unable to write SAM2 HOI native plugin memfd");
        }
        if (written == 0) {
            (void)::close(descriptor);
            throw std::runtime_error("Short write to SAM2 HOI native plugin memfd");
        }
        offset += static_cast<std::size_t>(written);
    }

    constexpr int seals = F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE;
    if (::fcntl(descriptor, F_ADD_SEALS, seals) != 0)
        close_descriptor_and_throw(descriptor, "Unable to seal SAM2 HOI native plugin memfd");
    return descriptor;
}

void verify_and_retain_native_plugin(const std::string& path, int descriptor,
                                     std::vector<char> bundle_bytes, std::string external_path) {
    dlerror();
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        const char* message = dlerror();
        const std::string detail = message != nullptr ? message : path;
        if (descriptor >= 0)
            (void)::close(descriptor);
        throw std::runtime_error("Unable to load SAM2 HOI native plugin: " + detail);
    }

    dlerror();
    using VersionFunction = int (*)();
    auto version =
        reinterpret_cast<VersionFunction>(dlsym(handle, "trtmc_sam2_hoi_native_plugin_version"));
    const char* symbol_error = dlerror();
    if (symbol_error != nullptr || version == nullptr || version() != 1) {
        (void)dlclose(handle);
        if (descriptor >= 0)
            (void)::close(descriptor);
        throw std::runtime_error("SAM2 HOI native plugin has an incompatible ABI");
    }

    try {
        retained_native_plugins().push_back(
            {descriptor, handle, std::move(bundle_bytes), std::move(external_path)});
    } catch (...) {
        (void)dlclose(handle);
        if (descriptor >= 0)
            (void)::close(descriptor);
        throw;
    }
}

void load_embedded_native_plugin(const std::vector<char>& bytes) {
    for (const auto& plugin : retained_native_plugins()) {
        if (plugin.bundle_bytes == bytes)
            return;
    }
    if (!retained_native_plugins().empty()) {
        throw std::runtime_error(
            "A different SAM2 HOI native plugin is already loaded in this process");
    }

    std::vector<char> identity(bytes);
    const int descriptor = create_sealed_native_memfd(identity);
    verify_and_retain_native_plugin("/proc/self/fd/" + std::to_string(descriptor), descriptor,
                                    std::move(identity), {});
}

void load_external_native_plugin(const std::string& path) {
    for (const auto& plugin : retained_native_plugins()) {
        if (plugin.external_path == path)
            return;
    }
    if (!retained_native_plugins().empty()) {
        throw std::runtime_error(
            "A different SAM2 HOI native plugin is already loaded in this process");
    }
    verify_and_retain_native_plugin(path, -1, {}, path);
}

void load_sam2_hoi_native_plugin(const PipelineContext& context) {
    std::lock_guard<std::mutex> lock(native_plugin_mutex());
    const auto* bytes = find_section(context.bundle, "sam2_hoi_native_plugin_so");
    if (bytes != nullptr && !bytes->empty()) {
        load_embedded_native_plugin(*bytes);
        return;
    }

    const char* configured = std::getenv("TRTMC_SAM2_HOI_NATIVE_PLUGIN_LIBRARY");
    const std::string path = configured != nullptr ? configured : std::string{};
    if (path.empty()) {
        throw std::runtime_error(
            "SAM2 HOI bundle is missing its model-owned native plugin section");
    }
    load_external_native_plugin(path);
}

} // namespace

class Sam2HoiPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& context) override {
        const bool phase_a =
            header_has_section(context.bundle.info, "sam2_hoi_pafpn_manifest.json");
        if (!phase_a) {
            // Preserve the existing Python build output until the Phase-A
            // front/leaf packer is integrated. Legacy bundles contain the
            // monolithic image engine and materialize all six plans eagerly.
            load_sam2_hoi_native_plugin(context);
            ModuleCreateOptions options;
            options.runtime_cache_path = context.runtime_cache_path.c_str();
            options.cuda_graphs = false;
            return std::make_unique<sam2_hoi::Sam2HoiPipeline>(
                load_legacy_module(context, "engine_plan", options),
                load_legacy_module(context, "sam2_hoi_detector_engine_plan", options),
                load_legacy_module(context, "sam2_hoi_interaction_engine_plan", options),
                load_legacy_module(context, "sam2_hoi_prompt_tracker_engine_plan", options),
                load_legacy_module(context, "sam2_hoi_recurrent_tracker_engine_plan", options),
                load_legacy_module(context, "sam2_hoi_memory_encoder_engine_plan", options),
                context.bundle.info.model_id);
        }

        validate_phase_a_eager_sections(context.bundle);
        // TensorRT must see the plugin creator before any detector plan is deserialized.
        load_sam2_hoi_native_plugin(context);
        if (context.backend == nullptr)
            throw std::runtime_error("SAM2 HOI pipeline has no TensorRT backend");
        auto plan_sections = index_phase_a_plan_sections(context.bundle.info);
        const auto* manifest_bytes = find_section(context.bundle, "sam2_hoi_pafpn_manifest.json");
        if (manifest_bytes == nullptr || manifest_bytes->empty())
            throw std::runtime_error("SAM2 HOI bundle is missing PAFPN manifest");
        auto manifest =
            sam2_hoi::parse_pafpn_manifest(manifest_bytes->data(), manifest_bytes->size());

        auto image_stream = std::make_shared<sam2_hoi::CudaStream>();
        if (!image_stream->ok())
            throw std::runtime_error("SAM2 HOI could not create image CUDA stream");
        ModuleCreateOptions image_options;
        image_options.stream = image_stream->get();
        image_options.runtime_cache_path = context.runtime_cache_path.c_str();
        image_options.cuda_graphs = false;
        ModuleCreateOptions other_options;
        other_options.runtime_cache_path = context.runtime_cache_path.c_str();
        other_options.cuda_graphs = false;

        auto front = load_staged_module(*context.backend, context.bundle_path, plan_sections,
                                        "engine_plan", image_options);
        auto pafpn = std::make_unique<sam2_hoi::PafpnComposite>(
            std::move(manifest),
            [&](const std::string& section, cudaStream_t stream) {
                if (stream != image_stream->get())
                    throw std::runtime_error("SAM2 HOI PAFPN loader stream drift");
                return load_staged_module(*context.backend, context.bundle_path, plan_sections,
                                          section, image_options);
            },
            image_stream->get());
        auto detector = load_staged_module(*context.backend, context.bundle_path, plan_sections,
                                           "sam2_hoi_detector_engine_plan", image_options);
        return std::make_unique<sam2_hoi::Sam2HoiPipeline>(
            std::static_pointer_cast<void>(image_stream), std::move(front), std::move(pafpn),
            std::move(detector),
            load_staged_module(*context.backend, context.bundle_path, plan_sections,
                               "sam2_hoi_interaction_engine_plan", other_options),
            load_staged_module(*context.backend, context.bundle_path, plan_sections,
                               "sam2_hoi_prompt_tracker_engine_plan", other_options),
            load_staged_module(*context.backend, context.bundle_path, plan_sections,
                               "sam2_hoi_recurrent_tracker_engine_plan", other_options),
            load_staged_module(*context.backend, context.bundle_path, plan_sections,
                               "sam2_hoi_memory_encoder_engine_plan", other_options),
            context.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_sam2_hoi_plugin, Sam2HoiPlugin,
                                       "sam2_hoi_video_tracking");

} // namespace trtmc

extern "C" TrtmcSam2HoiVideoSession*
trtmc_sam2_hoi_video_create_from_bundle_v1(const char* bundle_path, const char* plugin_dir,
                                           const char* backend_dir) noexcept {
    trtmc::sam2_hoi::c_api_internal::clearLastError();
    try {
        if (bundle_path == nullptr || plugin_dir == nullptr || backend_dir == nullptr ||
            *bundle_path == '\0' || *plugin_dir == '\0' || *backend_dir == '\0') {
            throw std::invalid_argument("SAM2 HOI bundle, plugin, and backend paths are required");
        }

        trtmc::LoadOptions options;
        options.model_plugin_search_paths.emplace_back(plugin_dir);
        options.backend_search_paths.emplace_back(backend_dir);
        auto pipeline = trtmc::load(bundle_path, options);
        auto* sam2_hoi_pipeline = dynamic_cast<trtmc::sam2_hoi::Sam2HoiPipeline*>(pipeline.get());
        if (sam2_hoi_pipeline == nullptr) {
            throw std::runtime_error(
                "loaded bundle did not create a SAM2 HOI native-video pipeline");
        }
        std::unique_ptr<trtmc::sam2_hoi::Sam2HoiPipeline> owned_pipeline(
            static_cast<trtmc::sam2_hoi::Sam2HoiPipeline*>(pipeline.release()));
        return trtmc::sam2_hoi::makeVideoSessionHandle(std::move(owned_pipeline));
    } catch (const std::exception& error) {
        trtmc::sam2_hoi::c_api_internal::setLastError(error.what());
    } catch (...) {
        trtmc::sam2_hoi::c_api_internal::setLastError("unknown native exception");
    }
    return nullptr;
}
