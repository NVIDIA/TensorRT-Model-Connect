/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/bundle.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>

namespace trtmc::app {
// Core-owned optional metadata. Never inspect family-private runtime sections.
inline nlohmann::json read_provenance(const std::string& path) {
    constexpr std::string_view section_name = "trtmc.provenance.json";
    const BundleReader reader(path);
    const auto* section = reader.find_section(section_name);
    if (section == nullptr)
        return nullptr;
    if (section->length > (1U << 20U))
        throw std::runtime_error("bundle provenance exceeds 1 MiB");
    const auto bytes = reader.read_section(section_name);
    return nlohmann::json::parse(bytes.begin(), bytes.end());
}
} // namespace trtmc::app
