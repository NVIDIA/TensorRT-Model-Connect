/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_format.h"

#include "utils/json_helpers.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace trtmc {

namespace {

using BundleSectionLocation = std::pair<std::size_t, std::size_t>;
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

int64_t parse_int64_field(const std::string& inner, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const auto key_pos = inner.find(needle);
    if (key_pos == std::string::npos) {
        return 0;
    }

    const auto colon = inner.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        return 0;
    }

    const auto start = inner.find_first_of("-0123456789", colon + 1);
    if (start == std::string::npos) {
        return 0;
    }

    try {
        return std::stoll(inner.substr(start));
    } catch (...) {
        return 0;
    }
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
    const int64_t offset_val = parse_int64_field(inner, "offset");
    const int64_t size_val = parse_int64_field(inner, "size");
    entry = {section_name,
             {static_cast<std::size_t>(offset_val), static_cast<std::size_t>(size_val)}};

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
        info.sections.push_back(BundleSectionInfo{name, static_cast<std::uint64_t>(offset),
                                                  static_cast<std::uint64_t>(size)});
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

std::ifstream open_bundle_section(const std::string& path, const BundleSectionInfo& section,
                                  std::uint64_t& file_offset_out) {
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
    file_offset_out = file_offset;
    in.seekg(static_cast<std::streamoff>(file_offset));
    if (!in) {
        throw std::runtime_error("Failed to seek to bundle section '" + section.name +
                                 "' in: " + path);
    }
    return in;
}

#if defined(__linux__)
struct FileCacheRange {
    off_t offset;
    off_t size;
};

std::optional<FileCacheRange> aligned_file_cache_range(std::uint64_t file_offset,
                                                       std::uint64_t size) noexcept {
    const long raw_page_size = sysconf(_SC_PAGESIZE);
    if (size == 0 || raw_page_size <= 0 ||
        file_offset > std::numeric_limits<std::uint64_t>::max() - size) {
        return std::nullopt;
    }

    const auto page_size = static_cast<std::uint64_t>(raw_page_size);
    const std::uint64_t aligned_offset = file_offset - (file_offset % page_size);
    std::uint64_t aligned_end = file_offset + size;
    const std::uint64_t end_remainder = aligned_end % page_size;
    if (end_remainder != 0) {
        const std::uint64_t padding = page_size - end_remainder;
        if (aligned_end > std::numeric_limits<std::uint64_t>::max() - padding)
            return std::nullopt;
        aligned_end += padding;
    }

    const std::uint64_t aligned_size = aligned_end - aligned_offset;
    const auto max_advice_value = static_cast<std::uint64_t>(std::numeric_limits<off_t>::max());
    if (aligned_offset > max_advice_value || aligned_size > max_advice_value)
        return std::nullopt;
    return FileCacheRange{static_cast<off_t>(aligned_offset), static_cast<off_t>(aligned_size)};
}
#endif

void release_bundle_section_cache(const std::string& path, std::uint64_t file_offset,
                                  std::uint64_t size) noexcept {
#if defined(__linux__)
    const auto range = aligned_file_cache_range(file_offset, size);
    if (!range)
        return;

    const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0)
        return;

    // Bundle data has already been copied into an owned buffer (or output
    // stream), so retaining a second copy in the Linux page cache only raises
    // peak memory. This is especially costly when GPU framebuffer is exposed
    // as an OS-managed NUMA node.
    (void)posix_fadvise(descriptor, range->offset, range->size, POSIX_FADV_DONTNEED);
    close(descriptor);
#else
    (void)path;
    (void)file_offset;
    (void)size;
#endif
}

} // namespace

BundleFile ReadBundleFile(const std::string& path) {
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

    BundleSectionTable section_table;
    BundleFile bundle;
    bundle.info = BundleInfoFromJson(header_json, section_table);

    in.seekg(0, std::ios::end);
    const auto file_end = in.tellg();
    if (file_end < 0)
        throw std::runtime_error("Failed to determine bundle size: " + path);
    const std::uint64_t file_size = static_cast<std::uint64_t>(file_end);
    const std::uint64_t data_start = kBundleHeaderOffset + header_length;

    for (const auto& [name, offset_size] : section_table) {
        const auto& [offset, size] = offset_size;
        BundleSection section;
        section.name = name;
        const BundleSectionInfo section_info{name, offset, size};
        const std::uint64_t file_offset =
            checked_section_file_offset(section_info, data_start, file_size, path);
        section.data.resize(static_cast<std::size_t>(size));

        in.seekg(static_cast<std::streamoff>(file_offset));
        if (!section.data.empty())
            in.read(section.data.data(), static_cast<std::streamsize>(section.data.size()));
        if (!in) {
            throw std::runtime_error("Failed to read bundle section '" + name + "' from: " + path);
        }
        bundle.sections.push_back(std::move(section));
    }

    in.close();
    release_bundle_section_cache(path, 0, file_size);
    return bundle;
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
    std::uint64_t file_offset = 0;
    std::ifstream in = open_bundle_section(path, section, file_offset);
    std::vector<char> data(static_cast<std::size_t>(section.size));
    if (!data.empty()) {
        in.read(data.data(), static_cast<std::streamsize>(data.size()));
    }
    if (!in) {
        throw std::runtime_error("Failed to read bundle section '" + section.name +
                                 "' from: " + path);
    }
    in.close();
    release_bundle_section_cache(path, file_offset, section.size);
    return data;
}

void CopyBundleSection(const std::string& path, const BundleSectionInfo& section,
                       std::ostream& output) {
    std::uint64_t file_offset = 0;
    std::ifstream in = open_bundle_section(path, section, file_offset);
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
    in.close();
    release_bundle_section_cache(path, file_offset, section.size);
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
