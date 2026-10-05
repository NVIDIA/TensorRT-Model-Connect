/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "provider_adapter.h"

#include "artifacts.h"
#include "cmake/edge_llm/provider/provider.h"
#include "tokenizer.h"

#include <dlfcn.h>
#include <memory>
#include <mutex>

namespace trtmc::llama::edge_llm {
namespace {
using trtmc::edge_llm::GetProviderV1;
using trtmc::edge_llm::ProviderV1;
struct CloseLibrary {
    void operator()(void* handle) const noexcept {
        if (handle)
            dlclose(handle);
    }
};

/// Family selects the exact provider. The shared runtime knows nothing about Edge versions.
class Provider {
  public:
    explicit Provider(const nlohmann::json& identity) {
        const auto version = identity.at("version").get<std::string>();
        if (identity.at("abi") != 1 || (version != "0.10.0" && version != "0.11.0"))
            throw std::runtime_error("Unsupported Llama Edge provider identity");
        std::string suffix = version;
        for (auto& c : suffix)
            if (c == '.')
                c = '_';
        Dl_info location{};
        if (!dladdr(reinterpret_cast<void*>(&create_provider), &location) || !location.dli_fname)
            throw std::runtime_error("Cannot locate Llama family DSO");
        const auto path = fs::absolute(location.dli_fname).parent_path() /
                          ("libtrtmc_edge_provider_llama_" + suffix + ".so");
        library_.reset(dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL));
        if (!library_)
            throw std::runtime_error("Cannot load Llama Edge provider: " + std::string(dlerror()));
        auto get = reinterpret_cast<GetProviderV1>(dlsym(library_.get(), "trtmc_edge_provider_v1"));
        if (!get)
            throw std::runtime_error("Edge provider factory is missing");
        api_ = get();
        if (!api_ || api_->abi != 1 || api_->size != sizeof(ProviderV1) || !api_->version ||
            version != api_->version || !api_->open || !api_->call || !api_->close || !api_->error)
            throw std::runtime_error("Edge provider ABI/version mismatch");
        const auto root = fs::absolute(location.dli_fname).parent_path();
        const auto descriptor = root / "edge_llm/providers" / (version + ".json");
        const auto worker = root / "edge_llm/llama/provider.py";
        if (!fs::is_regular_file(descriptor))
            throw std::runtime_error("Install the Edge provider descriptor at " +
                                     descriptor.string());
        handle_ = api_->open(worker.c_str(), descriptor.c_str());
        if (!handle_)
            throw std::runtime_error(api_->error());
    }
    ~Provider() {
        if (handle_)
            api_->close(handle_);
    }
    Provider(const Provider&) = delete;
    Provider& operator=(const Provider&) = delete;
    nlohmann::json call(const nlohmann::json& request) {
        const auto encoded = request.dump();
        const char* result = api_->call(handle_, encoded.c_str());
        if (!result)
            throw std::runtime_error(api_->error());
        const auto response = nlohmann::json::parse(result);
        if (response.contains("error"))
            throw std::runtime_error("Llama Edge provider: " +
                                     response.at("error").get<std::string>());
        return response.at("result");
    }

  private:
    std::unique_ptr<void, CloseLibrary> library_;
    const ProviderV1* api_{nullptr};
    void* handle_{nullptr};
};

std::string raw_prefix(const BundleReader& bundle) {
    const auto config = bundle.read_section("edge_llm/engine/tokenizer_config.json");
    const auto tokenizer = bundle.read_section("edge_llm/engine/tokenizer.json");
    return raw_prompt_prefix(nlohmann::json::parse(config.begin(), config.end()),
                             nlohmann::json::parse(tokenizer.begin(), tokenizer.end()));
}

class ProviderTask final : public ITextGeneration {
  public:
    ProviderTask(const BundleReader& bundle, const nlohmann::json& marker)
        : artifacts_(bundle, marker), provider_(marker.at("provider")),
          prefix_(raw_prefix(bundle)) {
        const auto response = provider_.call({{"op", "open"},
                                              {"marker", marker},
                                              {"engine", artifacts_.engine()},
                                              {"checkpoint", artifacts_.checkpoint()}});
        if (!response.value("ready", false))
            throw std::runtime_error("Llama Edge provider failed to initialize");
    }
    std::int32_t default_max_new_tokens() const override { return 128; }
    TextResult generate(const std::string& prompt, const TextGenerationConfig& config) override {
        validate_generation(config);
        const auto length =
            config.max_new_tokens > 0 ? config.max_new_tokens : default_max_new_tokens();
        std::lock_guard<std::mutex> lock(mutex_);
        const auto response =
            provider_.call({{"op", "generate"},
                            {"prompt", config.use_chat_template ? prompt : prefix_ + prompt},
                            {"use_chat_template", config.use_chat_template},
                            {"enable_thinking", config.enable_thinking},
                            {"max_new_tokens", length},
                            {"temperature", config.temperature},
                            {"top_p", config.top_p},
                            {"top_k", config.top_k}});
        auto ids = response.at("token_ids").get<std::vector<std::int32_t>>();
        if (ids.empty() || ids.size() > static_cast<std::size_t>(length))
            throw std::runtime_error("Invalid Llama Edge provider generation result");
        return {response.at("text").get<std::string>(), std::move(ids)};
    }

  private:
    // Provider is destroyed before its engine/checkpoint files.
    Artifacts artifacts_;
    Provider provider_;
    std::string prefix_;
    std::mutex mutex_;
};
} // namespace

ITask* create_provider(const BundleReader& bundle) {
    const auto bytes = bundle.read_section("edge_llm.json");
    const auto marker = nlohmann::json::parse(bytes.begin(), bytes.end());
    if (marker.at("version") != 2 || marker.at("max_sequence_length").get<int>() <= 1 ||
        marker.at("max_input_length").get<int>() <= 0 ||
        marker.at("max_input_length").get<int>() > marker.at("max_sequence_length").get<int>() ||
        marker.at("max_batch_size") != 1 || marker.at("precision") != "fp16" ||
        marker.at("weight_format") != "fp16" ||
        marker.at("execution_variant") != "autoregressive" || !marker.at("artifacts").is_array())
        throw std::runtime_error("Invalid versioned Llama Edge bundle contract");
    return new ProviderTask(bundle, marker);
}
} // namespace trtmc::llama::edge_llm
