/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "lexical_table.h"

#include "ple_rows.h"

#include <limits>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>

namespace trtmc::qwen3_8_flash_next {
namespace fs = std::filesystem;
LexicalTable::LexicalTable(const BundleReader& bundle) {
    const auto* section = bundle.find_section("ngram_e4m3.bin");
    const auto raw = bundle.read_section("ngram_layout.json");
    const auto count =
        nlohmann::json::parse(raw.begin(), raw.end()).at("table_rows").get<std::int64_t>();
    if (!section || count <= 0 ||
        static_cast<std::uint64_t>(count) > std::numeric_limits<std::uint64_t>::max() / 160 ||
        section->length != static_cast<std::uint64_t>(count) * 160)
        throw std::runtime_error("Invalid Flash-Next lexical table section length");
    bytes_ = section->length;
    const auto parent = fs::temp_directory_path();
    if (fs::space(parent).available < bytes_)
        throw std::runtime_error(
            "Not enough temporary disk space for the Flash-Next lexical table");
    std::random_device random;
    for (int i = 0; i < 8 && directory_.empty(); ++i) {
        const auto trial = parent / ("trtmc-flash-ngram-" + std::to_string(random()) + "-" +
                                     std::to_string(random()));
        if (fs::create_directory(trial))
            directory_ = trial;
    }
    if (directory_.empty())
        throw std::runtime_error("Cannot allocate a private Flash-Next lexical table directory");
    try {
        fs::permissions(directory_, fs::perms::owner_all, fs::perm_options::replace);
        const auto path = directory_ / "rows.bin";
        std::ofstream out(path, std::ios::binary);
        if (!out)
            throw std::runtime_error("Cannot open the Flash-Next lexical table for extraction");
        bundle.copy_section("ngram_e4m3.bin", out);
        out.close();
        if (!out || fs::file_size(path) != bytes_)
            throw std::runtime_error("Incomplete Flash-Next lexical table extraction");
        stream_.open(path, std::ios::binary);
        if (!stream_)
            throw std::runtime_error("Cannot open the extracted Flash-Next lexical table");
    } catch (...) {
        cleanup();
        throw;
    }
}
LexicalTable::~LexicalTable() {
    cleanup();
}
void LexicalTable::cleanup() noexcept {
    if (stream_.is_open())
        stream_.close();
    std::error_code ignored;
    if (!directory_.empty())
        fs::remove_all(directory_, ignored);
}
std::vector<std::uint8_t> LexicalTable::rows(const std::vector<std::int64_t>& ids) {
    stream_.clear();
    return read_ple_rows(stream_, 0, bytes_, ids);
}
} // namespace trtmc::qwen3_8_flash_next
