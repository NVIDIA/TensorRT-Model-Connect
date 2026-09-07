/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen/edge_llm/runtime/adapter.h"

#include "families/qwen/edge_llm/runtime/bridge.h"
#include "families/qwen/runtime/plugin_helpers.h"
#include "trtmc/runtime/family_factory.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::qwen::edge_llm {

namespace {

namespace fs = std::filesystem;

constexpr const char* kSectionPrefix = "edge_llm/";
constexpr const char* kBridgeLibrary = "libtrtmc_qwen_edge_llm.so";
constexpr std::size_t kErrorCapacity = 1024;

struct RuntimeConfig {
    std::int32_t max_cache_length;
};

RuntimeConfig parse_config(const BundleReader& reader) {
    const auto json = nlohmann::json::parse(require_text_section(reader, "edge_llm.json"));
    if (!json.is_object() || json.size() != 3 || !json.contains("max_input_length") ||
        !json.contains("max_cache_length") || !json.contains("max_batch_size")) {
        throw std::runtime_error("qwen Edge-LLM has invalid edge_llm.json fields");
    }
    const auto input = json.at("max_input_length").get<std::int32_t>();
    const auto cache = json.at("max_cache_length").get<std::int32_t>();
    const auto batch = json.at("max_batch_size").get<std::int32_t>();
    if (input <= 0 || cache < input || batch <= 0)
        throw std::runtime_error("qwen Edge-LLM has invalid runtime limits");
    return {cache};
}

bool safe_relative_path(const fs::path& path) {
    if (path.empty() || path.is_absolute())
        return false;
    for (const auto& component : path) {
        if (component.empty() || component == "." || component == "..")
            return false;
    }
    return true;
}

class EngineDirectory {
  public:
    explicit EngineDirectory(const BundleReader& reader) {
        fs::path pattern = fs::temp_directory_path() / "trtmc-qwen-edge-XXXXXX";
        std::string value = pattern.string();
        value.push_back('\0');
        char* created = ::mkdtemp(value.data());
        if (created == nullptr)
            throw std::runtime_error("qwen Edge-LLM could not create its engine directory");
        path_ = created;
        try {
            extract(reader);
        } catch (...) {
            cleanup();
            throw;
        }
    }

    EngineDirectory(const EngineDirectory&) = delete;
    EngineDirectory& operator=(const EngineDirectory&) = delete;
    ~EngineDirectory() { cleanup(); }

    const fs::path& path() const noexcept { return path_; }

  private:
    void extract(const BundleReader& reader) {
        bool found = false;
        for (const auto& section : reader.info().sections) {
            if (section.name.rfind(kSectionPrefix, 0) != 0)
                continue;
            const fs::path relative =
                section.name.substr(std::char_traits<char>::length(kSectionPrefix));
            if (!safe_relative_path(relative))
                throw std::runtime_error("qwen Edge-LLM bundle has an unsafe engine path");
            const fs::path destination = path_ / relative;
            fs::create_directories(destination.parent_path());
            std::ofstream output(destination, std::ios::binary | std::ios::trunc);
            if (!output)
                throw std::runtime_error("qwen Edge-LLM could not open an engine file");
            reader.copy_section(section.name, output);
            found = true;
        }
        if (!found)
            throw std::runtime_error("qwen Edge-LLM bundle has no engine files");
    }

    void cleanup() noexcept {
        if (path_.empty())
            return;
        std::error_code error;
        fs::remove_all(path_, error);
    }

    fs::path path_;
};

fs::path adjacent_bridge() {
    static const char anchor = 0;
    Dl_info info{};
    if (dladdr(&anchor, &info) == 0 || info.dli_fname == nullptr)
        throw std::runtime_error("qwen Edge-LLM could not locate its family DSO");
    return fs::path(info.dli_fname).parent_path() / kBridgeLibrary;
}

class BridgeLibrary {
  public:
    BridgeLibrary() {
        const fs::path path = adjacent_bridge();
        int flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_NODELETE
        flags |= RTLD_NODELETE;
#endif
        dlerror();
        handle_ = dlopen(path.c_str(), flags);
        if (handle_ == nullptr) {
            const char* error = dlerror();
            throw std::runtime_error("qwen Edge-LLM could not load its runtime bridge: " +
                                     std::string(error != nullptr ? error : "unknown error"));
        }
        dlerror();
        auto factory =
            reinterpret_cast<const BridgeApi* (*)() noexcept>(dlsym(handle_, kBridgeSymbol));
        if (const char* error = dlerror(); error != nullptr || factory == nullptr) {
            dlclose(handle_);
            handle_ = nullptr;
            throw std::runtime_error("qwen Edge-LLM runtime bridge has no factory");
        }
        api_ = factory();
        if (api_ == nullptr || api_->abi_version != kBridgeAbiVersion ||
            api_->struct_size < sizeof(BridgeApi) || api_->create == nullptr ||
            api_->destroy == nullptr || api_->generate == nullptr) {
            dlclose(handle_);
            handle_ = nullptr;
            throw std::runtime_error("qwen Edge-LLM runtime bridge has an incompatible ABI");
        }
    }

    BridgeLibrary(const BridgeLibrary&) = delete;
    BridgeLibrary& operator=(const BridgeLibrary&) = delete;
    ~BridgeLibrary() {
        if (handle_ != nullptr)
            dlclose(handle_);
    }

    const BridgeApi& api() const noexcept { return *api_; }

  private:
    void* handle_{nullptr};
    const BridgeApi* api_{nullptr};
};

void validate_config(const TextGenerationConfig& config, std::int32_t max_cache_length) {
    if (config.max_new_tokens <= 0 || config.max_new_tokens > max_cache_length)
        throw std::invalid_argument("qwen Edge-LLM max_new_tokens is outside the bundle limit");
    if (!std::isfinite(config.temperature) || config.temperature < 0.0F ||
        !std::isfinite(config.top_p) || config.top_p < 0.0F || config.top_p > 1.0F ||
        config.top_k < 0 || config.top_k > 1024) {
        throw std::invalid_argument("qwen Edge-LLM has invalid sampling values");
    }
    if (config.source_language_token_id != -1 || config.forced_bos_token_id != -1 ||
        config.min_p != 0.0F || config.seed != -1 || config.repetition_penalty != 1.0F ||
        config.guidance_scale >= 0.0F || config.cfg_scale >= 0.0F || config.num_steps != -1 ||
        config.sde_gamma >= 0.0F || !config.initial_latents.empty() ||
        !config.condition_latents.empty() || !config.condition_mask.empty() ||
        !config.sampling_steps.empty() || !config.sde_noises.empty() || config.eos_token_id != -1 ||
        (config.text_generation_mode != "auto" && config.text_generation_mode != "ar") ||
        config.block_length != 0 || config.confidence_threshold >= 0.0F ||
        config.stop_on_boxed_answer || config.stop_check_interval != 16 ||
        !config.lora_adapter_id.empty()) {
        throw std::invalid_argument("qwen Edge-LLM does not support a requested generation option");
    }
}

class Pipeline final : public ITextGeneration {
  public:
    Pipeline(const BundleReader& reader, RuntimeConfig config)
        : config_(config), engine_(reader), bridge_() {
        std::vector<char> error(kErrorCapacity, '\0');
        handle_ = bridge_.api().create(engine_.path().c_str(), error.data(), error.size());
        if (handle_ == nullptr)
            throw std::runtime_error("qwen Edge-LLM runtime creation failed: " +
                                     std::string(error.data()));
    }

    ~Pipeline() override {
        if (handle_ != nullptr)
            bridge_.api().destroy(handle_);
    }

    std::int32_t default_max_new_tokens() const override {
        return std::min<std::int32_t>(128, config_.max_cache_length);
    }

    TextResult generate(const std::string& prompt, const TextGenerationConfig& config) override {
        validate_config(config, config_.max_cache_length);
        const BridgeRequest request{
            prompt.data(), prompt.size(), config.max_new_tokens,    config.temperature,
            config.top_k,  config.top_p,  config.use_chat_template, config.enable_thinking};
        BridgeResult result{};
        std::vector<char> error(kErrorCapacity, '\0');
        std::lock_guard<std::mutex> lock(mutex_);
        if (!bridge_.api().generate(handle_, &request, &result, error.data(), error.size()))
            throw std::runtime_error("qwen Edge-LLM generation failed: " +
                                     std::string(error.data()));
        if (result.text == nullptr || (result.token_count != 0 && result.token_ids == nullptr))
            throw std::runtime_error("qwen Edge-LLM returned an invalid result");
        std::vector<std::int32_t> token_ids;
        if (result.token_count != 0)
            token_ids.assign(result.token_ids, result.token_ids + result.token_count);
        return {result.text, std::move(token_ids)};
    }

  private:
    RuntimeConfig config_;
    EngineDirectory engine_;
    BridgeLibrary bridge_;
    void* handle_{nullptr};
    std::mutex mutex_;
};

} // namespace

ITask* create(const FamilyOnlyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("qwen Edge-LLM does not support --kv-cache-size");
    return new Pipeline(context.reader, parse_config(context.reader));
}

} // namespace trtmc::qwen::edge_llm
