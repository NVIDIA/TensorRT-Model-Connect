/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/core.hpp"

namespace trtmc {

struct BundleSection {
    std::string name;
    std::uint64_t offset;
    std::uint64_t length;
};

struct BundleMetadata {
    std::int32_t format;
    std::string family;
    std::string task;
    std::string backend;
    std::vector<BundleSection> sections;
};

class BytesResult {
  public:
    BytesResult(const BytesResult&) = delete;
    BytesResult& operator=(const BytesResult&) = delete;
    BytesResult(BytesResult&&) noexcept = default;
    BytesResult& operator=(BytesResult&&) noexcept = default;

    // Borrowed from this result; remains valid after the source bundle closes.
    Span<const std::uint8_t> bytes() const {
        if (!owner_.get())
            return {};
        trtmc_bytes_view view{};
        trtmc_error* error = nullptr;
        const auto status = owner_.api().bytes_result_view(owner_.get(), &view, &error);
        detail::check(owner_.api(), status, error);
        return {view.data, static_cast<std::size_t>(view.size)};
    }

  private:
    friend class Bundle;
    explicit BytesResult(detail::ResultOwner owner) noexcept : owner_(std::move(owner)) {}
    detail::ResultOwner owner_;
};

class Bundle {
  public:
    static Bundle open(std::string_view path) {
        const auto& api = detail::core_api();
        trtmc_bundle* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api.bundle_open(detail::c_string(path), &raw, &error);
        Bundle bundle(api, raw);
        detail::check(api, status, error);
        return bundle;
    }

    Bundle(const Bundle&) = delete;
    Bundle& operator=(const Bundle&) = delete;
    Bundle(Bundle&&) noexcept = default;
    Bundle& operator=(Bundle&&) noexcept = default;

    BundleMetadata info() const {
        trtmc_bundle_info_v1 view{};
        trtmc_error* error = nullptr;
        const auto status = api_->bundle_info(handle_.get(), &view, &error);
        detail::check(*api_, status, error);
        BundleMetadata output{view.format,
                              std::string(detail::string_view(view.family)),
                              std::string(detail::string_view(view.task)),
                              std::string(detail::string_view(view.backend)),
                              {}};
        for (std::uint64_t i = 0; i < view.section_count; ++i) {
            const auto& section = view.sections[i];
            output.sections.push_back(
                {std::string(detail::string_view(section.name)), section.offset, section.length});
        }
        return output;
    }

    BytesResult read_section(std::string_view name) const {
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status =
            api_->bundle_read_section(handle_.get(), detail::c_string(name), &raw, &error);
        detail::ResultOwner owner(*api_, raw);
        detail::check(*api_, status, error);
        return BytesResult(std::move(owner));
    }

  private:
    struct Release {
        const trtmc_core_api_v1* api;
        void operator()(trtmc_bundle* bundle) const noexcept { api->bundle_release(bundle); }
    };
    Bundle(const trtmc_core_api_v1& api, trtmc_bundle* handle) noexcept
        : api_(&api), handle_(handle, Release{&api}) {}
    const trtmc_core_api_v1* api_;
    std::unique_ptr<trtmc_bundle, Release> handle_;
};

// Load before Model::load. The optional extension retains its registered kernel
// module for the process lifetime, so there is no misleading unload handle.
inline void load_byok_kernel(std::string_view library, std::string_view function,
                             std::string_view kernel_name, std::string_view runtime_root = {}) {
    const auto& api = detail::core_api();
    const trtmc_byok_options_v1 options{sizeof(trtmc_byok_options_v1),
                                        detail::c_string(runtime_root), detail::c_string(library),
                                        detail::c_string(function), detail::c_string(kernel_name)};
    trtmc_error* error = nullptr;
    const auto status = api.byok_load(&options, &error);
    detail::check(api, status, error);
}

class LoraManager {
  public:
    void load(std::string_view id, std::string_view path) const {
        trtmc_error* error = nullptr;
        const auto status =
            api_->load(state_->handle, detail::c_string(id), detail::c_string(path), &error);
        detail::check(state_->api, status, error);
    }

    void unload(std::string_view id) const {
        trtmc_error* error = nullptr;
        const auto status = api_->unload(state_->handle, detail::c_string(id), &error);
        detail::check(state_->api, status, error);
    }

    std::vector<std::string> list() const {
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        auto status = api_->list(state_->handle, &raw, &error);
        detail::ResultOwner owner(state_, raw);
        detail::check(state_->api, status, error);
        trtmc_strings_view view{};
        error = nullptr;
        status = api_->list_view(raw, &view, &error);
        detail::check(state_->api, status, error);
        std::vector<std::string> names;
        for (std::uint64_t i = 0; i < view.size; ++i)
            names.emplace_back(detail::string_view(view.data[i]));
        return names;
    }

  private:
    friend class Model;
    LoraManager(std::shared_ptr<detail::ModelState> state, const trtmc_lora_api_v1* table)
        : state_(std::move(state)), api_(table) {
        if (!api_ || api_->header.major != 1 || api_->header.minor != 0 ||
            api_->header.byte_size < sizeof(*api_))
            throw Error(TRTMC_VERSION_MISMATCH, "incompatible LoRA API table");
    }
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_lora_api_v1* api_;
};

inline LoraManager Model::lora_adapters() const {
    const trtmc_lora_api_v1* table = nullptr;
    trtmc_error* error = nullptr;
    const auto status = state_->api.model_get_lora_api(state_->handle, 1, 0, &table, &error);
    detail::check(state_->api, status, error);
    return LoraManager(state_, table);
}

} // namespace trtmc
