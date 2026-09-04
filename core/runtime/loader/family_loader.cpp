/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/family_loader.h"

#include "runtime/bundle/bundle_format.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <dlfcn.h>
#include <filesystem>
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
        : library_(runtime_root / ("libtrtmc_backend_" + backend_id + ".so")) {
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
        : library_(runtime_root / ("libtrtmc_model_" + family_id + ".so")),
          create_(reinterpret_cast<CreateFamilyFn>(library_.require_symbol(kCreateFamilySymbol))) {}

    FamilyLibrary(const FamilyLibrary&) = delete;
    FamilyLibrary& operator=(const FamilyLibrary&) = delete;

    ITask* create(const FamilyContext& context) const { return create_(context); }

  private:
    SharedLibrary library_;
    CreateFamilyFn create_{nullptr};
};

struct RuntimeLibraryCache {
    std::mutex mutex;
    std::unordered_map<std::string, std::unique_ptr<BackendLibrary>> backends;
    std::unordered_map<std::string, std::unique_ptr<FamilyLibrary>> families;
};

RuntimeLibraryCache& runtime_library_cache() {
    // The cache deliberately lives until process exit. Task objects can be
    // destroyed without a task-specific proxy because their code and backend
    // are never unloaded underneath them.
    static RuntimeLibraryCache* cache = new RuntimeLibraryCache();
    return *cache;
}

IBackend& cached_backend(const fs::path& runtime_root, const std::string& backend_id) {
    const std::string path = (runtime_root / ("libtrtmc_backend_" + backend_id + ".so")).string();
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

FamilyLibrary& cached_family(const fs::path& runtime_root, const std::string& family_id) {
    const std::string path = (runtime_root / ("libtrtmc_model_" + family_id + ".so")).string();
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

std::unique_ptr<ITask> load_task(const std::string& bundle_path, const std::string& runtime_root) {
    const BundleReader reader(bundle_path);
    const BundleInfo& info = reader.info();
    require_safe_id("family", info.family);
    require_safe_id("task", info.task);
    require_safe_id("backend", info.backend);

    const fs::path root = explicit_runtime_root(runtime_root);
    IBackend& backend = cached_backend(root, info.backend);
    FamilyLibrary& family = cached_family(root, info.family);
    FamilyContext context{reader, backend};
    std::unique_ptr<ITask> task(family.create(context));
    if (task == nullptr)
        throw std::runtime_error("trtmc_create_family returned nullptr");
    require_matching_task(info, *task);
    return task;
}

} // namespace trtmc
