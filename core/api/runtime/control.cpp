/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/control.h"

#include "api_internal.h"
#include "trtmc/bundle.h"
#include "trtmc/runtime/family_loader.h"

#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

struct trtmc_bundle {
    explicit trtmc_bundle(std::string path) : reader(std::move(path)) {
        sections.reserve(reader.info().sections.size());
        for (const auto& section : reader.info().sections)
            sections.push_back(
                {trtmc::api::borrowed_string(section.name), section.offset, section.length});
    }
    trtmc::BundleReader reader;
    std::vector<trtmc_bundle_section_v1> sections;
};

namespace trtmc::api {

namespace {

struct BytesStorage final : ResultStorage {
    explicit BytesStorage(std::vector<char> data) : bytes(std::move(data)) {}
    std::vector<char> bytes;
};

struct AdapterListStorage final : ResultStorage {
    explicit AdapterListStorage(std::vector<std::string> values) : names(std::move(values)) {
        views.reserve(names.size());
        for (const auto& name : names)
            views.push_back(borrowed_string(name));
    }
    std::vector<std::string> names;
    std::vector<trtmc_string_view> views;
};

ILoraAdapterManager& require_lora(const trtmc_model* model) {
    auto* metadata = dynamic_cast<internal::IModel*>(&model_family(model));
    auto* adapters = metadata ? metadata->lora_adapters() : nullptr;
    if (!adapters)
        throw ApiFailure{TRTMC_UNSUPPORTED, "loaded model does not support LoRA adapters"};
    return *adapters;
}

trtmc_status TRTMC_CALL lora_load(trtmc_model* model, trtmc_string_view id, trtmc_string_view path,
                                  trtmc_error** error) noexcept {
    return guarded(error, [&] {
        const std::lock_guard<std::mutex> lock(model_mutex(model));
        require_model_idle(model);
        require_lora(model).load_lora_adapter(std::string(string_view(id)), path_string(path));
    });
}

trtmc_status TRTMC_CALL lora_unload(trtmc_model* model, trtmc_string_view id,
                                    trtmc_error** error) noexcept {
    return guarded(error, [&] {
        const std::lock_guard<std::mutex> lock(model_mutex(model));
        require_model_idle(model);
        require_lora(model).unload_lora_adapter(std::string(string_view(id)));
    });
}

trtmc_status TRTMC_CALL lora_list(trtmc_model* model, trtmc_result** out,
                                  trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(out != nullptr, "adapter list output is null");
        const std::lock_guard<std::mutex> lock(model_mutex(model));
        require_model_idle(model);
        *out = make_result<AdapterListStorage>(require_lora(model).loaded_lora_adapters());
    });
}

trtmc_status TRTMC_CALL lora_list_view(const trtmc_result* result, trtmc_strings_view* out,
                                       trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out != nullptr, "adapter list view output is null");
        const auto& values = require_result<AdapterListStorage>(result).views;
        *out = {values.data(), values.size()};
    });
}

const trtmc_lora_api_v1 lora_api = {
    {1, 0, sizeof(trtmc_lora_api_v1)}, lora_load, lora_unload, lora_list, lora_list_view};

} // namespace

trtmc_status TRTMC_CALL bundle_open(trtmc_string_view path, trtmc_bundle** out,
                                    trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(out != nullptr, "bundle output is null");
        *out = new trtmc_bundle(path_string(path));
    });
}

void TRTMC_CALL bundle_release(trtmc_bundle* bundle) noexcept {
    delete bundle;
}

trtmc_status TRTMC_CALL bundle_info(const trtmc_bundle* bundle, trtmc_bundle_info_v1* out,
                                    trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(bundle != nullptr && out != nullptr, "bundle or metadata output is null");
        const auto& info = bundle->reader.info();
        *out = {info.format,
                borrowed_string(info.family),
                borrowed_string(info.task),
                borrowed_string(info.backend),
                bundle->sections.data(),
                bundle->sections.size()};
    });
}

trtmc_status TRTMC_CALL bundle_read_section(const trtmc_bundle* bundle, trtmc_string_view name,
                                            trtmc_result** out, trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(bundle != nullptr && out != nullptr, "bundle or section output is null");
        *out = make_result<BytesStorage>(bundle->reader.read_section(string_view(name)));
    });
}

trtmc_status TRTMC_CALL bytes_result_view(const trtmc_result* result, trtmc_bytes_view* out,
                                          trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out != nullptr, "byte view output is null");
        const auto& bytes = require_result<BytesStorage>(result).bytes;
        *out = {reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()};
    });
}

trtmc_status TRTMC_CALL byok_load(const trtmc_byok_options_v1* options,
                                  trtmc_error** error) noexcept {
    return guarded(error, [&] {
        require(options != nullptr && options->struct_size >= sizeof(*options),
                "BYOK options are null or smaller than v1");
        auto root = path_string(options->runtime_root);
        if (root.empty())
            root = default_runtime_root();
        preload_byok_kernel(root, path_string(options->library), path_string(options->function),
                            path_string(options->kernel_name));
    });
}

trtmc_status TRTMC_CALL model_get_lora_api(const trtmc_model* model, std::uint32_t major,
                                           std::uint32_t minor, const trtmc_lora_api_v1** out,
                                           trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(out != nullptr, "LoRA API output is null");
        if (major != 1 || minor != 0)
            throw ApiFailure{TRTMC_VERSION_MISMATCH, "requested LoRA version is unavailable"};
        const std::lock_guard<std::mutex> lock(model_mutex(model));
        (void)require_lora(model);
        *out = &lora_api;
    });
}

} // namespace trtmc::api
