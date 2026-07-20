/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_format.h"

#include "utils/json_helpers.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace trtmc {

namespace {

using BundleSectionLocation = std::pair<std::uint64_t, std::uint64_t>;
using BundleSectionEntry = std::pair<std::string, BundleSectionLocation>;
using BundleSectionTable = std::vector<BundleSectionEntry>;

uint64_t read_u64_le(std::ifstream& in) {
    unsigned char bytes[8];
    in.read(reinterpret_cast<char*>(bytes), 8);
    if (!in) {
        throw std::runtime_error("Failed to read uint64 from bundle file");
    }
    uint64_t value = 0;
    for (int i = 7; i >= 0; --i) {
        value = (value << 8) | bytes[i];
    }
    return value;
}

std::size_t find_matching_object_end(const std::string& json, std::size_t brace_start) {
    int depth = 1;
    std::size_t pos = brace_start + 1;
    while (pos < json.size() && depth > 0) {
        if (json[pos] == '{') {
            ++depth;
        } else if (json[pos] == '}') {
            --depth;
        }
        ++pos;
    }
    return pos;
}

[[noreturn]] void throw_malformed_section_field(const std::string& key) {
    throw std::runtime_error("Bundle section has malformed '" + key + "'");
}

std::size_t skip_json_whitespace(const std::string& text, std::size_t position) {
    while (position < text.size() &&
           std::isspace(static_cast<unsigned char>(text[position])) != 0) {
        ++position;
    }
    return position;
}

std::size_t require_section_value_start(const std::string& inner, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const auto key_pos = inner.find(needle);
    if (key_pos == std::string::npos)
        throw std::runtime_error("Bundle section is missing '" + key + "'");

    const std::size_t colon = skip_json_whitespace(inner, key_pos + needle.size());
    if (colon == inner.size() || inner[colon] != ':')
        throw_malformed_section_field(key);

    const std::size_t start = skip_json_whitespace(inner, colon + 1);
    if (start == inner.size())
        throw_malformed_section_field(key);
    if (inner[start] == '-')
        throw std::runtime_error("Bundle section has negative '" + key + "'");
    if (!std::isdigit(static_cast<unsigned char>(inner[start])))
        throw_malformed_section_field(key);
    return start;
}

std::size_t find_decimal_end(const std::string& inner, std::size_t start, const std::string& key) {
    std::size_t end = start;
    while (end < inner.size() && std::isdigit(static_cast<unsigned char>(inner[end])) != 0)
        ++end;
    if (inner[start] == '0' && end != start + 1)
        throw_malformed_section_field(key);
    return end;
}

std::uint64_t parse_decimal_u64(const std::string& inner, std::size_t start, std::size_t end,
                                const std::string& key) {
    std::uint64_t value = 0;
    const auto result = std::from_chars(inner.data() + start, inner.data() + end, value);
    if (result.ec == std::errc::result_out_of_range)
        throw std::runtime_error("Bundle section has overflowing '" + key + "'");
    if (result.ec != std::errc{} || result.ptr != inner.data() + end)
        throw_malformed_section_field(key);
    return value;
}

void require_section_value_terminator(const std::string& inner, std::size_t end,
                                      const std::string& key) {
    end = skip_json_whitespace(inner, end);
    if (end == inner.size() || (inner[end] != ',' && inner[end] != '}'))
        throw_malformed_section_field(key);
}

std::uint64_t parse_section_size_field(const std::string& inner, const std::string& key) {
    const std::size_t start = require_section_value_start(inner, key);
    const std::size_t end = find_decimal_end(inner, start, key);
    const std::uint64_t value = parse_decimal_u64(inner, start, end, key);
    require_section_value_terminator(inner, end, key);
    return value;
}

bool parse_section_entry(const std::string& sections_json, std::size_t& search_pos,
                         BundleSectionEntry& entry) {
    const auto quote_start = sections_json.find('"', search_pos);
    if (quote_start == std::string::npos) {
        return false;
    }
    const auto quote_end = sections_json.find('"', quote_start + 1);
    if (quote_end == std::string::npos) {
        return false;
    }

    const auto inner_brace = sections_json.find('{', quote_end + 1);
    if (inner_brace == std::string::npos) {
        return false;
    }
    const auto inner_brace_end = sections_json.find('}', inner_brace + 1);
    if (inner_brace_end == std::string::npos) {
        return false;
    }

    const std::string section_name =
        sections_json.substr(quote_start + 1, quote_end - quote_start - 1);
    const std::string inner = sections_json.substr(inner_brace, inner_brace_end - inner_brace + 1);
    const std::uint64_t offset_val = parse_section_size_field(inner, "offset");
    const std::uint64_t size_val = parse_section_size_field(inner, "size");
    entry = {section_name, {offset_val, size_val}};

    search_pos = inner_brace_end + 1;
    return true;
}

void parse_sections_table(const std::string& json, BundleSectionTable& sections_out) {
    const std::string sections_key = "\"sections\"";
    const auto sections_pos = json.find(sections_key);
    if (sections_pos == std::string::npos) {
        return;
    }

    const auto brace_start = json.find('{', sections_pos + sections_key.size());
    if (brace_start == std::string::npos) {
        return;
    }

    const std::size_t brace_end = find_matching_object_end(json, brace_start);
    const std::string sections_json = json.substr(brace_start, brace_end - brace_start);
    std::size_t search_pos = 0;
    while (search_pos < sections_json.size()) {
        BundleSectionEntry entry;
        if (!parse_section_entry(sections_json, search_pos, entry)) {
            break;
        }
        sections_out.push_back(std::move(entry));
    }
}

BundleInfo BundleInfoFromJson(const std::string& json, BundleSectionTable& sections_out) {
    BundleInfo info;
    info.model_id = extract_json_string(json, "model_id", "");
    info.source_model_id = extract_json_string(json, "source_model_id", "");
    info.source_revision = extract_json_string(json, "source_revision", "");
    info.model_type = extract_json_string(json, "model_type", "");
    info.family = extract_json_string(json, "family", "");
    info.precision = extract_json_string(json, "precision", "");
    info.trt_version = extract_json_string(json, "trt_version", "");
    info.trt_abi = extract_json_string(json, "trt_abi", "");
    info.gpu_name = extract_json_string(json, "gpu_name", "");
    info.created_at = extract_json_string(json, "created_at", "");
    info.vocab_size = extract_json_int(json, "vocab_size", 0);
    info.hidden_size = extract_json_int(json, "hidden_size", 0);
    info.num_layers = extract_json_int(json, "num_layers", 0);
    info.num_attention_heads = extract_json_int(json, "num_attention_heads", 1);
    info.num_key_value_heads = extract_json_int(json, "num_key_value_heads", 1);
    info.max_cache_length = extract_json_int(json, "max_cache_length", 32);
    info.runtime_strategy = extract_json_string(json, "runtime_strategy", "");
    const int32_t tokenizer_add_special =
        extract_json_int(json, "tokenizer_add_special_tokens", -1);
    if (tokenizer_add_special >= 0) {
        info.tokenizer_add_special_tokens = (tokenizer_add_special != 0);
        info.tokenizer_add_special_tokens_present = true;
    }

    // Per-component diffusion batch caps (see design doc Decision C).
    // Absent => leave the default {1, 1, 1} so legacy bundles run unchanged.
    const std::string mbs_text = extract_json_object_text(json, "max_batch_size");
    if (!mbs_text.empty()) {
        info.max_batch_size.dit = extract_json_int(mbs_text, "dit", 1);
        info.max_batch_size.text_encoder = extract_json_int(mbs_text, "text_encoder", 1);
        info.max_batch_size.vae = extract_json_int(mbs_text, "vae", 1);
    }

    sections_out.clear();
    parse_sections_table(json, sections_out);
    info.sections.clear();
    info.sections.reserve(sections_out.size());
    for (const auto& [name, offset_size] : sections_out) {
        const auto& [offset, size] = offset_size;
        info.sections.push_back(BundleSectionInfo{name, offset, size});
    }

    return info;
}

std::uint64_t read_bundle_data_start(std::ifstream& in, const std::string& path,
                                     std::uint64_t file_size) {
    unsigned char magic[8];
    in.read(reinterpret_cast<char*>(magic), sizeof(magic));
    if (!in || std::memcmp(magic, kBundleMagic, sizeof(kBundleMagic)) != 0)
        throw std::runtime_error("Invalid bundle magic in: " + path);

    const std::uint64_t header_length = read_u64_le(in);
    if (header_length > 100 * 1024 * 1024)
        throw std::runtime_error("Bundle header too large: " + path);
    if (header_length > std::numeric_limits<std::uint64_t>::max() - kBundleHeaderOffset)
        throw std::runtime_error("Bundle header offset overflow: " + path);

    const std::uint64_t data_start = kBundleHeaderOffset + header_length;
    if (data_start > file_size)
        throw std::runtime_error("Bundle data offset extends outside file: " + path);
    return data_start;
}

std::uint64_t checked_section_file_offset(const BundleSectionInfo& section,
                                          std::uint64_t data_start, std::uint64_t file_size,
                                          const std::string& path) {
    if (section.offset > file_size - data_start ||
        section.size > file_size - data_start - section.offset) {
        throw std::runtime_error("Bundle section '" + section.name +
                                 "' extends outside file: " + path);
    }
    if (section.size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        section.size > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("Bundle section '" + section.name +
                                 "' is too large to read: " + path);
    }
    return data_start + section.offset;
}

std::ifstream open_bundle_section(const std::string& path, const BundleSectionInfo& section) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
        throw std::runtime_error("Failed to open bundle file: " + path);

    const auto file_end = in.tellg();
    if (file_end < 0)
        throw std::runtime_error("Failed to determine bundle size: " + path);
    const auto file_size = static_cast<std::uint64_t>(file_end);
    in.seekg(0);

    const std::uint64_t data_start = read_bundle_data_start(in, path, file_size);
    const std::uint64_t file_offset =
        checked_section_file_offset(section, data_start, file_size, path);
    in.seekg(static_cast<std::streamoff>(file_offset));
    if (!in) {
        throw std::runtime_error("Failed to seek to bundle section '" + section.name +
                                 "' in: " + path);
    }
    return in;
}

} // namespace

BundleFile ReadBundleFile(const std::string& path) {
    BundleSectionReader reader(path);
    return ReadBundleFile(reader);
}

BundleFile ReadBundleFile(BundleSectionReader& reader) {
    return reader.read_all();
}

BundleSectionReader::BundleSectionReader(const std::string& path)
    : path_(path), stream_(path, std::ios::binary) {
    if (!stream_) {
        throw std::runtime_error("Failed to open bundle file: " + path_);
    }

    unsigned char magic[8];
    stream_.read(reinterpret_cast<char*>(magic), sizeof(magic));
    if (!stream_ || std::memcmp(magic, kBundleMagic, sizeof(kBundleMagic)) != 0) {
        throw std::runtime_error("Invalid bundle magic in: " + path_);
    }

    const uint64_t header_length = read_u64_le(stream_);
    if (header_length > 100 * 1024 * 1024) {
        throw std::runtime_error("Bundle header too large: " + path_);
    }

    std::string header_json(static_cast<std::size_t>(header_length), '\0');
    stream_.read(header_json.data(), static_cast<std::streamsize>(header_length));
    if (!stream_) {
        throw std::runtime_error("Failed to read bundle header: " + path_);
    }

    BundleSectionTable section_table;
    info_ = BundleInfoFromJson(header_json, section_table);
    data_start_ = static_cast<std::uint64_t>(kBundleHeaderOffset) + header_length;

    stream_.seekg(0, std::ios::end);
    const auto end_position = stream_.tellg();
    if (end_position < 0) {
        throw std::runtime_error("Failed to determine bundle size: " + path_);
    }
    file_size_ = static_cast<std::uint64_t>(end_position);
    if (data_start_ > file_size_)
        throw std::runtime_error("Bundle data starts outside file bounds in: " + path_);
}

std::vector<char> BundleSectionReader::read(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto section =
        std::find_if(info_.sections.begin(), info_.sections.end(),
                     [&name](const BundleSectionInfo& entry) { return entry.name == name; });
    if (section == info_.sections.end()) {
        throw std::runtime_error("Bundle section '" + name + "' was not found in: " + path_);
    }

    return read_locked(*section);
}

void BundleSectionReader::for_each_chunk(
    const std::string& name, std::size_t chunk_size,
    const std::function<void(const char*, std::size_t)>& visitor) {
    if (chunk_size == 0)
        throw std::invalid_argument("Bundle section chunk size must be non-zero");
    if (!visitor)
        throw std::invalid_argument("Bundle section chunk visitor must be callable");

    std::lock_guard<std::mutex> lock(mutex_);
    const auto section =
        std::find_if(info_.sections.begin(), info_.sections.end(),
                     [&name](const BundleSectionInfo& entry) { return entry.name == name; });
    if (section == info_.sections.end()) {
        throw std::runtime_error("Bundle section '" + name + "' was not found in: " + path_);
    }

    const auto section_offset = section->offset;
    const auto section_size = section->size;
    if (section_offset > file_size_ - data_start_ ||
        section_size > file_size_ - data_start_ - section_offset) {
        throw std::runtime_error("Bundle section '" + section->name +
                                 "' is outside file bounds in: " + path_);
    }

    const std::size_t buffer_size = static_cast<std::size_t>(
        std::min<std::uint64_t>(section_size, static_cast<std::uint64_t>(chunk_size)));
    std::vector<char> buffer(buffer_size);
    std::uint64_t consumed = 0;
    stream_.clear();
    stream_.seekg(static_cast<std::streamoff>(data_start_ + section_offset), std::ios::beg);
    if (!stream_) {
        throw std::runtime_error("Failed to seek bundle section '" + section->name +
                                 "' in: " + path_);
    }
    while (consumed < section_size) {
        const std::size_t requested = static_cast<std::size_t>(std::min<std::uint64_t>(
            section_size - consumed, static_cast<std::uint64_t>(buffer.size())));
        stream_.read(buffer.data(), static_cast<std::streamsize>(requested));
        if (!stream_ || stream_.gcount() != static_cast<std::streamsize>(requested)) {
            throw std::runtime_error("Failed to read complete bundle section '" + section->name +
                                     "' from: " + path_);
        }
        visitor(buffer.data(), requested);
        consumed += requested;
    }
}

BundleFile BundleSectionReader::read_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    BundleFile bundle;
    bundle.info = info_;
    bundle.sections.reserve(info_.sections.size());
    for (const auto& section_info : info_.sections) {
        BundleSection section;
        section.name = section_info.name;
        section.data = read_locked(section_info);
        bundle.sections.push_back(std::move(section));
    }
    return bundle;
}

std::vector<char> BundleSectionReader::read_locked(const BundleSectionInfo& section) {
    const auto section_offset = section.offset;
    const auto section_size = section.size;
    if (section_offset > file_size_ - data_start_ ||
        section_size > file_size_ - data_start_ - section_offset) {
        throw std::runtime_error("Bundle section '" + section.name +
                                 "' is outside file bounds in: " + path_);
    }
    if (section_size > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()) ||
        section_size > static_cast<std::uint64_t>(std::vector<char>().max_size())) {
        throw std::runtime_error("Bundle section '" + section.name +
                                 "' is too large to read from: " + path_);
    }

    std::vector<char> data(static_cast<std::size_t>(section_size));
    stream_.clear();
    stream_.seekg(static_cast<std::streamoff>(data_start_ + section_offset), std::ios::beg);
    if (!stream_) {
        throw std::runtime_error("Failed to seek bundle section '" + section.name +
                                 "' in: " + path_);
    }
    if (!data.empty()) {
        stream_.read(data.data(), static_cast<std::streamsize>(data.size()));
        if (!stream_ || stream_.gcount() != static_cast<std::streamsize>(data.size())) {
            throw std::runtime_error("Failed to read complete bundle section '" + section.name +
                                     "' from: " + path_);
        }
    }
    return data;
}

std::vector<char> ReadBundleSection(const std::string& path, const std::string& name) {
    BundleSectionReader reader(path);
    return reader.read(name);
}

BundleInfo ReadBundleHeader(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open bundle file: " + path);
    }

    unsigned char magic[8];
    in.read(reinterpret_cast<char*>(magic), sizeof(magic));
    if (!in || std::memcmp(magic, kBundleMagic, sizeof(kBundleMagic)) != 0) {
        throw std::runtime_error("Invalid bundle magic in: " + path);
    }

    const uint64_t header_length = read_u64_le(in);
    if (header_length > 100 * 1024 * 1024) {
        throw std::runtime_error("Bundle header too large: " + path);
    }

    std::string header_json(static_cast<std::size_t>(header_length), '\0');
    in.read(header_json.data(), static_cast<std::streamsize>(header_length));
    if (!in) {
        throw std::runtime_error("Failed to read bundle header: " + path);
    }

    BundleSectionTable sections_ignored;
    return BundleInfoFromJson(header_json, sections_ignored);
}

std::vector<char> ReadBundleSection(const std::string& path, const BundleSectionInfo& section) {
    std::ifstream in = open_bundle_section(path, section);
    std::vector<char> data(static_cast<std::size_t>(section.size));
    if (!data.empty()) {
        in.read(data.data(), static_cast<std::streamsize>(data.size()));
    }
    if (!in) {
        throw std::runtime_error("Failed to read bundle section '" + section.name +
                                 "' from: " + path);
    }
    return data;
}

void CopyBundleSection(const std::string& path, const BundleSectionInfo& section,
                       std::ostream& output) {
    std::ifstream in = open_bundle_section(path, section);
    std::array<char, 1024 * 1024> buffer{};
    std::uint64_t remaining = section.size;
    while (remaining != 0) {
        const auto chunk_size =
            static_cast<std::streamsize>(std::min<std::uint64_t>(remaining, buffer.size()));
        in.read(buffer.data(), chunk_size);
        if (in.gcount() != chunk_size) {
            throw std::runtime_error("Failed to read bundle section '" + section.name +
                                     "' from: " + path);
        }
        output.write(buffer.data(), chunk_size);
        if (!output) {
            throw std::runtime_error("Failed to write materialized bundle section '" +
                                     section.name + "'");
        }
        remaining -= static_cast<std::uint64_t>(chunk_size);
    }
}

bool HasBundleMagic(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }

    unsigned char magic[8];
    in.read(reinterpret_cast<char*>(magic), sizeof(magic));
    if (!in) {
        return false;
    }

    return std::memcmp(magic, kBundleMagic, sizeof(kBundleMagic)) == 0;
}

// Public API implementations from bundle.h

bool IsBundle(const std::string& path) {
    return HasBundleMagic(path);
}

BundleInfo InspectBundle(const std::string& bundle_path) {
    return ReadBundleHeader(bundle_path);
}

} // namespace trtmc
