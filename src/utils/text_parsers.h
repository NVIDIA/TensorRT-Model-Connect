/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trtmc {

struct SourceLine {
    int number{0};
    std::string text;
};

bool starts_with(std::string_view value, std::string_view prefix);
bool ends_with(std::string_view value, std::string_view suffix);
std::string to_lower_ascii(std::string value);
std::string trim(std::string value);
std::string strip_inline_comment(std::string line);
std::string read_file(const std::filesystem::path& path);
std::vector<SourceLine> read_clean_lines(const std::filesystem::path& path);
std::vector<std::string> load_vocab(const std::filesystem::path& path);
std::vector<std::pair<std::string, std::string>> load_transitions(const std::filesystem::path& path);
std::vector<std::string> split_words(const std::string& line);
int32_t parse_int(const std::string& text, const std::filesystem::path& path, int line_number, const char* field);
float parse_float(const std::string& text, const std::filesystem::path& path, int line_number, const char* field);
bool iequals_ascii(std::string_view a, std::string_view b);

} // namespace trtmc
