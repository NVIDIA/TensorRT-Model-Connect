/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/text_parsers.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace trtmc {

bool starts_with(std::string_view value, std::string_view prefix)
{
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(std::string_view value, std::string_view suffix)
{
    return value.size() >= suffix.size()
        && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string to_lower_ascii(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string trim(std::string value)
{
    const auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [&](unsigned char c) { return !is_space(c); }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [&](unsigned char c) { return !is_space(c); }).base(), value.end());
    return value;
}

std::string strip_inline_comment(std::string line)
{
    const std::size_t hash = line.find('#');
    if (hash != std::string::npos)
    {
        line.erase(hash);
    }
    return trim(std::move(line));
}

std::string read_file(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in)
    {
        throw std::runtime_error("Failed to open file: " + path.string());
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

std::vector<SourceLine> read_clean_lines(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in)
    {
        throw std::runtime_error("Failed to open file: " + path.string());
    }

    std::vector<SourceLine> lines;
    std::string line;
    int line_number = 0;
    while (std::getline(in, line))
    {
        ++line_number;
        line = strip_inline_comment(std::move(line));
        if (!line.empty())
        {
            lines.push_back(SourceLine{line_number, std::move(line)});
        }
    }
    return lines;
}

std::vector<std::string> load_vocab(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in)
    {
        throw std::runtime_error("Failed to open vocab file: " + path.string());
    }

    std::vector<std::string> vocab;
    std::string line;
    while (std::getline(in, line))
    {
        line = strip_inline_comment(std::move(line));
        if (!line.empty())
        {
            vocab.push_back(std::move(line));
        }
    }

    if (vocab.empty())
    {
        throw std::runtime_error("Vocabulary is empty: " + path.string());
    }
    return vocab;
}

std::vector<std::pair<std::string, std::string>> load_transitions(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in)
    {
        throw std::runtime_error("Failed to open transitions file: " + path.string());
    }

    std::vector<std::pair<std::string, std::string>> transitions;
    std::string line;
    int line_number = 0;
    while (std::getline(in, line))
    {
        ++line_number;
        line = strip_inline_comment(std::move(line));
        if (line.empty())
        {
            continue;
        }

        std::istringstream iss(line);
        std::string from;
        std::string to;
        iss >> from >> to;
        if (from.empty() || to.empty())
        {
            throw std::runtime_error(
                "Invalid transition at " + path.string() + ":" + std::to_string(line_number));
        }
        transitions.emplace_back(std::move(from), std::move(to));
    }

    if (transitions.empty())
    {
        throw std::runtime_error("Transitions file is empty: " + path.string());
    }
    return transitions;
}

std::vector<std::string> split_words(const std::string& line)
{
    std::istringstream iss(line);
    std::vector<std::string> out;
    std::string token;
    while (iss >> token)
    {
        out.push_back(std::move(token));
    }
    return out;
}

int32_t parse_int(const std::string& text, const std::filesystem::path& path, int line_number, const char* field)
{
    try
    {
        std::size_t parsed = 0;
        const int value = std::stoi(text, &parsed);
        if (parsed != text.size())
        {
            throw std::runtime_error("invalid integer suffix");
        }
        return static_cast<int32_t>(value);
    }
    catch (const std::exception&)
    {
        throw std::runtime_error("Invalid integer for " + std::string(field) + " at " + path.string() + ":"
            + std::to_string(line_number) + ": " + text);
    }
}

float parse_float(const std::string& text, const std::filesystem::path& path, int line_number, const char* field)
{
    try
    {
        std::size_t parsed = 0;
        const float value = std::stof(text, &parsed);
        if (parsed != text.size())
        {
            throw std::runtime_error("invalid float suffix");
        }
        return value;
    }
    catch (const std::exception&)
    {
        throw std::runtime_error("Invalid float for " + std::string(field) + " at " + path.string() + ":"
            + std::to_string(line_number) + ": " + text);
    }
}

bool iequals_ascii(std::string_view a, std::string_view b)
{
    if (a.size() != b.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const unsigned char ac = static_cast<unsigned char>(a[i]);
        const unsigned char bc = static_cast<unsigned char>(b[i]);
        if (std::tolower(ac) != std::tolower(bc))
        {
            return false;
        }
    }
    return true;
}

} // namespace trtmc
