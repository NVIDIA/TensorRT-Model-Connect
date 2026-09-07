/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/family_loader.h"

#include "runtime/bundle/bundle_format.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/runtime_root.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <elf.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace trtmc {

namespace {

namespace fs = std::filesystem;

using CreateBackendFn = IBackend* (*)();
using DestroyBackendFn = void (*)(IBackend*);

std::string backend_library_name(const std::string& backend_id) {
    return "libtrtmc_backend_" + backend_id + ".so";
}

std::string family_library_name(const std::string& family_id) {
    return "libtrtmc_model_" + family_id + ".so";
}

constexpr std::uint64_t kMaxElfStringTableSize = 16ULL * 1024ULL * 1024ULL;

bool read_at(std::ifstream& input, std::uintmax_t file_size, std::uint64_t offset,
             void* destination, std::size_t size) {
    if (offset > file_size || size > file_size - offset)
        return false;
    input.clear();
    input.seekg(static_cast<std::streamoff>(offset));
    input.read(static_cast<char*>(destination), static_cast<std::streamsize>(size));
    return input.good();
}

bool has_supported_elf_identity(const Elf64_Ehdr& header) {
    return header.e_ident[EI_MAG0] == ELFMAG0 && header.e_ident[EI_MAG1] == ELFMAG1 &&
           header.e_ident[EI_MAG2] == ELFMAG2 && header.e_ident[EI_MAG3] == ELFMAG3 &&
           header.e_ident[EI_CLASS] == ELFCLASS64 && header.e_ident[EI_DATA] == ELFDATA2LSB;
}

bool has_valid_section_table(const Elf64_Ehdr& header, std::uintmax_t file_size) {
    return header.e_shentsize == sizeof(Elf64_Shdr) && header.e_shnum != 0 &&
           header.e_shstrndx < header.e_shnum && header.e_shoff <= file_size &&
           header.e_shnum <= (file_size - header.e_shoff) / sizeof(Elf64_Shdr);
}

bool read_section_header(std::ifstream& input, std::uintmax_t file_size, const Elf64_Ehdr& header,
                         std::size_t index, Elf64_Shdr& section) {
    const std::uint64_t offset = header.e_shoff + index * sizeof(Elf64_Shdr);
    return read_at(input, file_size, offset, &section, sizeof(section));
}

bool read_string_table(std::ifstream& input, std::uintmax_t file_size, const Elf64_Shdr& section,
                       std::string& contents) {
    if (section.sh_size > kMaxElfStringTableSize)
        return false;
    contents.assign(static_cast<std::size_t>(section.sh_size), '\0');
    return read_at(input, file_size, section.sh_offset, contents.data(), contents.size());
}

bool section_has_name(const Elf64_Shdr& section, const std::string& names, const char* expected) {
    if (section.sh_name >= names.size())
        return false;
    const auto end = names.find('\0', section.sh_name);
    return end != std::string::npos &&
           names.compare(section.sh_name, end - section.sh_name, expected) == 0;
}

bool is_lower_hex(unsigned char character) {
    return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
}

bool read_named_string_table(std::ifstream& input, std::uintmax_t file_size,
                             const Elf64_Ehdr& header, const std::string& section_names,
                             const char* expected_name, std::string& contents) {
    for (std::size_t index = 0; index < header.e_shnum; ++index) {
        Elf64_Shdr section{};
        if (!read_section_header(input, file_size, header, index, section))
            return false;
        if (!section_has_name(section, section_names, expected_name))
            continue;
        return read_string_table(input, file_size, section, contents);
    }
    return false;
}

std::string find_build_cohort(const std::string& strings) {
    static constexpr char prefix[] = "trtmc_build_cohort_";
    static constexpr std::size_t id_size = 32;
    std::size_t position = strings.find(prefix);
    while (position != std::string::npos) {
        const std::size_t id_begin = position + sizeof(prefix) - 1;
        const std::size_t marker_end = id_begin + id_size;
        if (marker_end < strings.size() && strings[marker_end] == '\0' &&
            std::all_of(strings.begin() + static_cast<std::ptrdiff_t>(id_begin),
                        strings.begin() + static_cast<std::ptrdiff_t>(marker_end), is_lower_hex)) {
            return strings.substr(id_begin, id_size);
        }
        position = strings.find(prefix, position + 1);
    }
    return {};
}

bool is_safe_id(const std::string& value) {
    if (value.empty() || value.front() < 'a' || value.front() > 'z')
        return false;
    for (const unsigned char character : value) {
        if ((character >= 'a' && character <= 'z') || (character >= '0' && character <= '9') ||
            character == '_') {
            continue;
        }
        return false;
    }
    return true;
}

void require_safe_id(const std::string& field, const std::string& value) {
    if (!is_safe_id(value)) {
        throw std::runtime_error("Bundle " + field + " must match [a-z][a-z0-9_]*: '" + value +
                                 "'");
    }
}

fs::path explicit_runtime_root(const std::string& runtime_root) {
    if (runtime_root.empty())
        throw std::invalid_argument("runtime_root must be explicit and non-empty");
    std::error_code error;
    fs::path root = fs::absolute(fs::path(runtime_root), error);
    if (error)
        throw std::runtime_error("Unable to resolve runtime_root '" + runtime_root +
                                 "': " + error.message());
    return root.lexically_normal();
}

std::string read_build_cohort(const fs::path& library) {
    std::ifstream input(library, std::ios::binary);
    if (!input)
        return {};

    std::error_code error;
    const std::uintmax_t file_size = fs::file_size(library, error);
    if (error || file_size < sizeof(Elf64_Ehdr))
        return {};

    Elf64_Ehdr header{};
    if (!read_at(input, file_size, 0, &header, sizeof(header)) ||
        !has_supported_elf_identity(header) || !has_valid_section_table(header, file_size)) {
        return {};
    }

    Elf64_Shdr names_header{};
    if (!read_section_header(input, file_size, header, header.e_shstrndx, names_header))
        return {};
    std::string names;
    if (!read_string_table(input, file_size, names_header, names))
        return {};

    std::string strings;
    if (!read_named_string_table(input, file_size, header, names, ".dynstr", strings))
        return {};
    return find_build_cohort(strings);
}

fs::path loaded_library_path(const void* symbol) {
    Dl_info info{};
    if (dladdr(symbol, &info) == 0 || info.dli_fname == nullptr)
        return {};
    std::error_code error;
    fs::path path = fs::absolute(info.dli_fname, error);
    if (error)
        return info.dli_fname;
    fs::path normalized = fs::weakly_canonical(path, error);
    return error ? path.lexically_normal() : normalized;
}

struct LoadedBuild {
    fs::path runtime_library;
    std::string cohort_id;
};

const LoadedBuild& loaded_build() {
    static const LoadedBuild build = [] {
        using InspectBundleFn = BundleInfo (*)(const std::string&);
        using LoadTaskFn = std::unique_ptr<ITask> (*)(const std::string&, const std::string&,
                                                      std::uint64_t, const std::string&, bool);
        const auto inspect_bundle_function = static_cast<InspectBundleFn>(&InspectBundle);
        const auto load_task_function = static_cast<LoadTaskFn>(&load_task);
        const fs::path core_library =
            loaded_library_path(reinterpret_cast<const void*>(inspect_bundle_function));
        const fs::path runtime_library =
            loaded_library_path(reinterpret_cast<const void*>(load_task_function));
        const std::string core_cohort = read_build_cohort(core_library);
        const std::string runtime_cohort = read_build_cohort(runtime_library);
        return LoadedBuild{runtime_library, !core_cohort.empty() && core_cohort == runtime_cohort
                                                ? core_cohort
                                                : std::string{}};
    }();
    return build;
}

class SharedLibrary {
  public:
    explicit SharedLibrary(const fs::path& path) : path_(path.string()) {
        dlerror();
        handle_ = dlopen(path_.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle_ == nullptr) {
            const char* error = dlerror();
            throw std::runtime_error("Unable to load '" + path_ +
                                     "': " + (error != nullptr ? error : "unknown dlopen error"));
        }
    }

    SharedLibrary(const SharedLibrary&) = delete;
    SharedLibrary& operator=(const SharedLibrary&) = delete;

    ~SharedLibrary() {
        if (handle_ != nullptr)
            dlclose(handle_);
    }

    void* require_symbol(const char* name) const {
        dlerror();
        void* symbol = dlsym(handle_, name);
        const char* error = dlerror();
        if (error != nullptr || symbol == nullptr) {
            throw std::runtime_error("Library '" + path_ + "' is missing required symbol '" + name +
                                     "'");
        }
        return symbol;
    }

  private:
    std::string path_;
    void* handle_{nullptr};
};

class BackendLibrary {
  public:
    BackendLibrary(const fs::path& runtime_root, const std::string& backend_id)
        : library_(runtime_root / backend_library_name(backend_id)) {
        const auto create =
            reinterpret_cast<CreateBackendFn>(library_.require_symbol("trtmc_create_backend"));
        destroy_ =
            reinterpret_cast<DestroyBackendFn>(library_.require_symbol("trtmc_destroy_backend"));
        backend_ = create();
        if (backend_ == nullptr)
            throw std::runtime_error("trtmc_create_backend returned nullptr");

        const char* actual_name = backend_->name();
        if (actual_name == nullptr || backend_id != actual_name) {
            const std::string actual = actual_name != nullptr ? actual_name : "<null>";
            destroy_(backend_);
            backend_ = nullptr;
            throw std::runtime_error("Backend identity mismatch: bundle requested '" + backend_id +
                                     "' but DSO created '" + actual + "'");
        }
    }

    BackendLibrary(const BackendLibrary&) = delete;
    BackendLibrary& operator=(const BackendLibrary&) = delete;

    ~BackendLibrary() {
        if (backend_ != nullptr)
            destroy_(backend_);
    }

    IBackend& get() const { return *backend_; }

  private:
    SharedLibrary library_;
    IBackend* backend_{nullptr};
    DestroyBackendFn destroy_{nullptr};
};

class FamilyLibrary {
  public:
    FamilyLibrary(const fs::path& runtime_root, const std::string& family_id)
        : library_(runtime_root / family_library_name(family_id)),
          create_(reinterpret_cast<CreateFamilyFn>(library_.require_symbol(kCreateFamilySymbol))) {}

    FamilyLibrary(const FamilyLibrary&) = delete;
    FamilyLibrary& operator=(const FamilyLibrary&) = delete;

    ITask* create(const FamilyContext& context) const { return create_(context); }

  private:
    SharedLibrary library_;
    CreateFamilyFn create_{nullptr};
};

class RuntimeOptionsBackend final : public IBackend {
  public:
    RuntimeOptionsBackend(IBackend& backend, std::string runtime_cache_path, bool cuda_graphs)
        : backend_(backend), runtime_cache_path_(std::move(runtime_cache_path)),
          cuda_graphs_(cuda_graphs) {}

    std::unique_ptr<ITrtModule> create_module(const void* plan_data, size_t plan_size,
                                              const ModuleCreateOptions& options) override {
        return backend_.create_module(plan_data, plan_size, with_runtime_options(options));
    }

    std::unique_ptr<ITrtModule>
    create_module_prebound(const void* plan_data, size_t plan_size,
                           const ModuleCreateOptions& options,
                           const std::vector<ModuleExternalBinding>& bindings) override {
        return backend_.create_module_prebound(plan_data, plan_size, with_runtime_options(options),
                                               bindings);
    }

    BackendDualProfileModules
    create_dual_profile_modules(const void* plan_data, size_t plan_size,
                                const ModuleCreateOptions& options) override {
        return backend_.create_dual_profile_modules(plan_data, plan_size,
                                                    with_runtime_options(options));
    }

    const char* name() const override { return backend_.name(); }

  private:
    ModuleCreateOptions with_runtime_options(ModuleCreateOptions options) const {
        options.runtime_cache_path = runtime_cache_path_.c_str();
        options.cuda_graphs = cuda_graphs_;
        return options;
    }

    IBackend& backend_;
    std::string runtime_cache_path_;
    bool cuda_graphs_;
};

struct ConfiguredBackendKey {
    IBackend* backend{nullptr};
    std::string runtime_cache_path;
    bool cuda_graphs{false};

    bool operator==(const ConfiguredBackendKey& other) const noexcept {
        return backend == other.backend && runtime_cache_path == other.runtime_cache_path &&
               cuda_graphs == other.cuda_graphs;
    }
};

struct ConfiguredBackendKeyHash {
    std::size_t operator()(const ConfiguredBackendKey& key) const noexcept {
        std::size_t value = std::hash<IBackend*>{}(key.backend);
        value ^= std::hash<std::string>{}(key.runtime_cache_path) + 0x9e3779b9U + (value << 6U) +
                 (value >> 2U);
        value ^= std::hash<bool>{}(key.cuda_graphs) + 0x9e3779b9U + (value << 6U) + (value >> 2U);
        return value;
    }
};

struct RuntimeLibraryCache {
    std::mutex mutex;
    std::unordered_map<std::string, std::unique_ptr<BackendLibrary>> backends;
    std::unordered_map<std::string, std::unique_ptr<FamilyLibrary>> families;
    std::unordered_map<ConfiguredBackendKey, std::unique_ptr<RuntimeOptionsBackend>,
                       ConfiguredBackendKeyHash>
        configured_backends;
};

RuntimeLibraryCache& runtime_library_cache() {
    // The cache deliberately lives until process exit. Family tasks may defer
    // module creation, so their code, backend, and immutable runtime-options
    // adapter must never be unloaded underneath them.
    static RuntimeLibraryCache* cache = new RuntimeLibraryCache();
    return *cache;
}

IBackend& cached_backend(const fs::path& runtime_root, const std::string& backend_id) {
    const std::string path = (runtime_root / backend_library_name(backend_id)).string();
    auto& cache = runtime_library_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    const auto found = cache.backends.find(path);
    if (found != cache.backends.end())
        return found->second->get();

    auto library = std::make_unique<BackendLibrary>(runtime_root, backend_id);
    IBackend& backend = library->get();
    cache.backends.emplace(path, std::move(library));
    return backend;
}

IBackend& cached_configured_backend(IBackend& backend, const std::string& runtime_cache_path,
                                    bool cuda_graphs) {
    ConfiguredBackendKey key{&backend, runtime_cache_path, cuda_graphs};
    auto& cache = runtime_library_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    const auto found = cache.configured_backends.find(key);
    if (found != cache.configured_backends.end())
        return *found->second;

    auto configured =
        std::make_unique<RuntimeOptionsBackend>(backend, runtime_cache_path, cuda_graphs);
    IBackend& result = *configured;
    cache.configured_backends.emplace(std::move(key), std::move(configured));
    return result;
}

FamilyLibrary& cached_family(const fs::path& runtime_root, const std::string& family_id) {
    const std::string path = (runtime_root / family_library_name(family_id)).string();
    auto& cache = runtime_library_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    const auto found = cache.families.find(path);
    if (found != cache.families.end())
        return *found->second;

    auto library = std::make_unique<FamilyLibrary>(runtime_root, family_id);
    FamilyLibrary& family = *library;
    cache.families.emplace(path, std::move(library));
    return family;
}

void require_matching_task(const BundleInfo& info, const ITask& task) {
    const char* actual = task.task();
    if (actual == nullptr || info.task != actual) {
        throw std::runtime_error("Family factory task mismatch: bundle declares '" + info.task +
                                 "' but factory returned '" +
                                 (actual != nullptr ? std::string(actual) : std::string("<null>")) +
                                 "'");
    }
}

} // namespace

std::string loaded_runtime_root() {
    const fs::path& runtime_library = loaded_build().runtime_library;
    if (runtime_library.empty())
        throw std::runtime_error("Unable to locate the active TRTMC runtime loader");
    return runtime_library.parent_path().string();
}

bool runtime_root_matches_loaded_build(const BundleInfo& bundle, const std::string& runtime_root,
                                       bool require_byok) {
    require_safe_id("family", bundle.family);
    require_safe_id("backend", bundle.backend);
    const std::string& cohort_id = loaded_build().cohort_id;
    if (runtime_root.empty() || cohort_id.empty())
        return false;

    std::error_code error;
    fs::path root = fs::absolute(runtime_root, error);
    if (error)
        return false;
    root = root.lexically_normal();

    std::vector<std::string> required{
        "libtrtmc_core.so",
        "libtrtmc_runtime.so",
        backend_library_name(bundle.backend),
        family_library_name(bundle.family),
    };
    if (require_byok)
        required.emplace_back("libtrtmc_byok_tvm_ffi.so");

    return std::all_of(required.begin(), required.end(), [&](const std::string& library) {
        std::error_code library_error;
        const fs::path path = root / library;
        return fs::is_regular_file(path, library_error) && read_build_cohort(path) == cohort_id;
    });
}

std::unique_ptr<ITask> load_task(const std::string& bundle_path, const std::string& runtime_root,
                                 std::uint64_t kv_cache_size_bytes,
                                 const std::string& runtime_cache_path, bool cuda_graphs) {
    const BundleReader reader(bundle_path);
    const BundleInfo& info = reader.info();
    require_safe_id("family", info.family);
    require_safe_id("task", info.task);
    require_safe_id("backend", info.backend);
    if ((!runtime_cache_path.empty() || cuda_graphs) && info.backend != "trt_rtx") {
        throw std::invalid_argument(
            "runtime cache and whole-graph capture require a TensorRT-RTX bundle");
    }

    const fs::path root = explicit_runtime_root(runtime_root);
    IBackend& backend = cached_backend(root, info.backend);
    FamilyLibrary& family = cached_family(root, info.family);
    IBackend& configured_backend =
        cached_configured_backend(backend, runtime_cache_path, cuda_graphs);
    FamilyContext context{reader, configured_backend, kv_cache_size_bytes};
    std::unique_ptr<ITask> task(family.create(context));
    if (task == nullptr)
        throw std::runtime_error("trtmc_create_family returned nullptr");
    require_matching_task(info, *task);
    return task;
}

} // namespace trtmc
