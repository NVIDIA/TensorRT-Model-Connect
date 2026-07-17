/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_view.h"
#include "runtime/models/wan2_2_ti2v/pipeline.h"
#include "runtime/models/wan2_2_ti2v/plugin_cache.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/tokenizer.h"
#include "utils/sha256.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

struct Wan22CudaPluginLoadState {
    std::string sha256;
    void* handle{nullptr};
};

std::mutex& cuda_plugin_load_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<std::string, Wan22CudaPluginLoadState>& cuda_plugin_load_states() {
    static std::unordered_map<std::string, Wan22CudaPluginLoadState> states;
    return states;
}

std::string sha256_bytes(const std::vector<char>& bytes) {
    detail::Sha256 digest;
    digest.update(bytes.data(), bytes.size());
    return digest.hex_digest();
}

bool environment_truthy(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr)
        return false;
    const std::string text(value);
    return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON";
}

std::vector<char> read_plugin_file(const std::filesystem::path& path) {
    if (!std::filesystem::is_regular_file(path))
        throw std::runtime_error("Wan2.2 CUDA plugin override is not a regular file: " +
                                 path.string());
    const auto file_size = std::filesystem::file_size(path);
    if (file_size == 0 || file_size > std::vector<char>().max_size())
        throw std::runtime_error("Wan2.2 CUDA plugin override has an invalid size: " +
                                 path.string());
    std::vector<char> bytes(static_cast<std::size_t>(file_size));
    std::ifstream input(path, std::ios::binary);
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!input || input.gcount() != static_cast<std::streamsize>(bytes.size()))
        throw std::runtime_error("Unable to read Wan2.2 CUDA plugin override: " + path.string());
    return bytes;
}

std::filesystem::path cuda_plugin_cache_path(const std::string& sha256, const char* label) {
    const char* configured = std::getenv("TRTMC_WAN22_CUDA_PLUGIN_CACHE_DIR");
    const auto directory = configured != nullptr && configured[0] != '\0'
                               ? std::filesystem::path(configured)
                               : std::filesystem::temp_directory_path() / "trtmc-wan2-2";
    std::ostringstream name;
    name << "libtrtmc_wan2_2_" << label << "_cuda_plugin_" << sha256 << ".so";
    return directory / name.str();
}

bool cuda_plugin_matches(const std::filesystem::path& output, const std::vector<char>& bytes) {
    if (!std::filesystem::is_regular_file(output) ||
        std::filesystem::file_size(output) != bytes.size()) {
        return false;
    }
    std::ifstream existing(output, std::ios::binary);
    std::vector<char> cached(bytes.size());
    existing.read(cached.data(), static_cast<std::streamsize>(cached.size()));
    return existing && cached == bytes;
}

std::filesystem::path cuda_plugin_temporary_path(const std::filesystem::path& output) {
    static std::atomic<std::uint64_t> counter{0};
    return output.string() + ".tmp." + std::to_string(static_cast<long long>(getpid())) + "." +
           std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
}

} // namespace

std::string resolve_wan22_cuda_plugin_override(const char* environment_name) {
    if (environment_name == nullptr || environment_name[0] == '\0')
        throw std::invalid_argument("Wan2.2 CUDA plugin override environment name is empty");
    const char* configured = std::getenv(environment_name);
    if (configured == nullptr || configured[0] == '\0')
        return {};
    if (environment_truthy("TRTMC_MODEL_PLUGIN_STRICT")) {
        throw std::runtime_error(std::string(environment_name) +
                                 " is forbidden when TRTMC_MODEL_PLUGIN_STRICT is enabled");
    }
    if (!environment_truthy("TRTMC_WAN22_ALLOW_DEVELOPMENT_PLUGIN_OVERRIDE")) {
        throw std::runtime_error(std::string(environment_name) +
                                 " requires TRTMC_WAN22_ALLOW_DEVELOPMENT_PLUGIN_OVERRIDE=1");
    }
    return configured;
}

void record_wan22_cuda_plugin_provenance(const std::string& creator_set,
                                         const std::vector<char>& bytes) {
    if (creator_set.empty() || bytes.empty())
        throw std::invalid_argument("Wan2.2 CUDA plugin provenance requires a label and bytes");
    const std::string digest = sha256_bytes(bytes);
    std::lock_guard<std::mutex> lock(cuda_plugin_load_mutex());
    auto [iterator, inserted] = cuda_plugin_load_states().try_emplace(
        creator_set, Wan22CudaPluginLoadState{digest, nullptr});
    if (!inserted && iterator->second.sha256 != digest) {
        throw std::runtime_error("Conflicting Wan2.2 CUDA plugin bytes for creator set " +
                                 creator_set + ": loaded=" + iterator->second.sha256 +
                                 ", requested=" + digest);
    }
}

void publish_wan22_cuda_plugin(const std::filesystem::path& output,
                               const std::vector<char>& bytes) {
    static std::mutex publication_mutex;
    std::lock_guard<std::mutex> lock(publication_mutex);

    std::filesystem::create_directories(output.parent_path());
    if (cuda_plugin_matches(output, bytes))
        return;

    const auto temporary = cuda_plugin_temporary_path(output);
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw std::runtime_error("Unable to create the Wan2.2 CUDA plugin cache file");
    }
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    stream.close();
    if (!stream) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw std::runtime_error("Unable to write the Wan2.2 CUDA plugin cache file");
    }
    try {
        // POSIX rename publishes the complete file atomically and replaces an
        // older cache entry without a remove/open gap. The unique temporary
        // name also permits cooperating processes to publish concurrently.
        std::filesystem::rename(temporary, output);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

namespace {

void validate_wan22_lazy_plan_sections(const PipelineContext& ctx) {
    // Validate the staged contract from header metadata only. This catches an
    // incomplete bundle at load time without materializing any TensorRT plan.
    for (const char* name : {"text_encoder_0_plan", "denoiser_plan", "vae_decoder_plan",
                             "vae_decoder_first_frame_plan"}) {
        bool found = false;
        for (const auto& section : ctx.bundle.info.sections) {
            if (section.name == name) {
                found = section.size != 0;
                break;
            }
        }
        if (!found)
            throw std::runtime_error(std::string("Wan2.2 bundle is missing ") + name);
    }
}

void load_wan22_cuda_plugin(const PipelineContext& ctx, const char* section_name,
                            const char* environment_name, const char* label) {
    std::string path;
    const auto* bytes = find_section(ctx.bundle, section_name);
    if (bytes == nullptr || bytes->empty())
        throw std::runtime_error(std::string("Wan2.2 bundle is missing ") + section_name);

    std::vector<char> selected_bytes;
    const std::string development_override = resolve_wan22_cuda_plugin_override(environment_name);
    if (!development_override.empty()) {
        path = development_override;
        selected_bytes = read_plugin_file(path);
    } else {
        selected_bytes = *bytes;
        const auto cached = cuda_plugin_cache_path(sha256_bytes(selected_bytes), label);
        publish_wan22_cuda_plugin(cached, selected_bytes);
        path = cached.string();
    }

    // Claim the fixed TensorRT creator namespace before dlopen. If another
    // bundle already registered different implementation bytes, fail before
    // any conflicting static registration code can execute.
    record_wan22_cuda_plugin_provenance(label, selected_bytes);
    const std::string selected_digest = sha256_bytes(selected_bytes);
    std::lock_guard<std::mutex> lock(cuda_plugin_load_mutex());
    auto& state = cuda_plugin_load_states().at(label);
    if (state.sha256 != selected_digest) {
        throw std::runtime_error(std::string("Conflicting Wan2.2 CUDA plugin digest for ") + label);
    }
    if (state.handle != nullptr)
        return;
    dlerror();
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        const char* message = dlerror();
        throw std::runtime_error(std::string("Unable to load the Wan2.2 ") + label +
                                 " CUDA plugin: " + (message != nullptr ? message : path));
    }
    // Keep every creator library alive for the complete TensorRT module
    // lifetime.  TensorRT may call plugin methods after pipeline creation.
    state.handle = handle;
}

Wan22ModuleLoader make_staged_module_loader(const PipelineContext& ctx) {
    if (ctx.backend == nullptr)
        throw std::runtime_error("Wan2.2 requires a TensorRT backend");
    if (!ctx.bundle_reader) {
        throw std::runtime_error(
            "Wan2.2 requires the pinned source bundle reader for staged loading");
    }

    // PipelineContext is factory-owned and expires after create(). Capture
    // every value needed by generation by value. Backends are process-cached
    // by BackendLoader, so the backend pointer remains valid for the pipeline.
    // This is the same open file description that materialized ctx.bundle.
    // Retaining it makes both the eager metadata and every later plan read
    // immune to pathname rename/replacement/unlink.
    auto bundle_reader = ctx.bundle_reader;
    const std::string runtime_cache_path = ctx.runtime_cache_path;
    IBackend* const backend = ctx.backend;
    const bool cuda_graphs = ctx.cuda_graphs;
    return [bundle_reader = std::move(bundle_reader), runtime_cache_path, backend,
            cuda_graphs](const std::string& section_name, cudaStream_t stream,
                         const std::vector<ModuleExternalBinding>& external_bindings)
               -> std::unique_ptr<ITrtModule> {
        // Only one plan payload is resident on the host. TensorRT consumes it
        // synchronously in create_module(); this vector dies before the
        // generation stage receives the module.
        auto plan = bundle_reader->read(section_name);
        if (plan.empty())
            throw std::runtime_error("Wan2.2 bundle section is empty: " + section_name);
        ModuleCreateOptions options;
        options.stream = stream;
        options.runtime_cache_path = runtime_cache_path.c_str();
        options.cuda_graphs = cuda_graphs;
        options.external_bindings = external_bindings;
        auto module = backend->create_module(plan.data(), plan.size(), options);
        if (!module || !module->ok())
            throw std::runtime_error("Wan2.2 could not deserialize " + section_name);
        return module;
    };
}

std::shared_ptr<ITokenizer> load_tokenizer(const BundleFile& bundle) {
    const auto* tokenizer_json = find_section(bundle, "tokenizer.json");
    if (tokenizer_json == nullptr || tokenizer_json->empty())
        throw std::runtime_error("Wan2.2 bundle is missing tokenizer.json");
    auto tokenizer = CreateUnigramTokenizer(tokenizer_json->data(), tokenizer_json->size(), false);
    if (!tokenizer)
        throw std::runtime_error("Wan2.2 could not create the native UMT5 tokenizer");
    return std::shared_ptr<ITokenizer>(std::move(tokenizer));
}

} // namespace

class Wan22TI2VPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        validate_wan22_lazy_plan_sections(ctx);
        // TensorRT resolves plugin creators while deserializing plans, so the
        // family-owned CUDA library must be registered first.
        load_wan22_cuda_plugin(ctx, "wan2_2_umt5_cuda_plugin_so",
                               "TRTMC_WAN22_UMT5_CUDA_PLUGIN_LIBRARY", "umt5");
        load_wan22_cuda_plugin(ctx, "wan2_2_dit_cuda_plugin_so",
                               "TRTMC_WAN22_DIT_CUDA_PLUGIN_LIBRARY", "dit");
        load_wan22_cuda_plugin(ctx, "wan2_2_vae_cuda_plugin_so",
                               "TRTMC_WAN22_VAE_CUDA_PLUGIN_LIBRARY", "vae");
        auto tokenizer = load_tokenizer(ctx.bundle);
        return std::make_unique<Wan22TI2VPipeline>(
            make_staged_module_loader(ctx), std::move(tokenizer),
            parse_wan22_options(ctx.config_json), ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_wan2_2_ti2v_plugin, Wan22TI2VPlugin,
                                       "diffusion_wan2_2_ti2v");

} // namespace trtmc
