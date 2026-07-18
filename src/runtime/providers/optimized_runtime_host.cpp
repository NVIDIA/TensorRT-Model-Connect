/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/providers/optimized_runtime_host.h"

#include "bundle/bundle_format.h"
#include "runtime/providers/optimized_runtime_factory.h"
#include "utils/sha256.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <string_view>
#include <sys/file.h>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

namespace fs = std::filesystem;

constexpr const char* kDescriptorSection = "optimized_runtime.json";
constexpr std::uint64_t kMaxDescriptorSize = 1024ULL * 1024ULL;
constexpr std::uint64_t kMaxImplementationMetadataSize = 16ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t kMaxArtifactEntries = 65536;
constexpr std::uint64_t kMaxArtifactPathBytes = 4096;
constexpr std::uint64_t kMaxArtifactTotalSize = 1ULL << 40;
constexpr std::size_t kErrorCapacity = 4096;

struct RuntimeIdentity {
    std::string name;
    std::string version;
    std::string commit;
};

struct ArtifactDescriptor {
    std::string section_prefix;
    std::vector<std::string> directories;
    std::uint64_t file_count{0};
    std::uint64_t total_size{0};
    std::string tree_sha256;
};

struct OptimizedRuntimeDescriptor {
    int32_t schema_version{0};
    std::string implementation_id;
    std::string model_id;
    std::string profile_id;
    std::string runtime_library;
    int32_t factory_abi{0};
    std::string implementation_metadata_section;
    RuntimeIdentity runtime;
    ArtifactDescriptor artifact;
};

const BundleSectionInfo* find_section(const BundleInfo& info, const std::string& name) {
    for (const auto& section : info.sections) {
        if (section.name == name)
            return &section;
    }
    return nullptr;
}

std::string read_text_section(const std::string& bundle_path, const BundleInfo& info,
                              const std::string& name, std::uint64_t max_size) {
    const auto* section = find_section(info, name);
    if (section == nullptr)
        throw std::runtime_error("Optimized-runtime bundle is missing required section '" + name +
                                 "': " + bundle_path);
    if (section->size > max_size) {
        throw std::runtime_error("Optimized-runtime section '" + name +
                                 "' exceeds its size limit: " + bundle_path);
    }
    const auto bytes = ReadBundleSection(bundle_path, *section);
    return std::string(bytes.begin(), bytes.end());
}

nlohmann::json parse_json_document(const std::string& text, const std::string& context) {
    std::vector<std::unordered_set<std::string>> object_keys;
    nlohmann::json::parser_callback_t callback = [&](int, nlohmann::json::parse_event_t event,
                                                     nlohmann::json& parsed) {
        if (event == nlohmann::json::parse_event_t::object_start)
            object_keys.emplace_back();
        if (event == nlohmann::json::parse_event_t::key) {
            const std::string key = parsed.get<std::string>();
            if (object_keys.empty() || !object_keys.back().insert(key).second) {
                throw std::runtime_error("Duplicate JSON object key in " + context + ": " + key);
            }
        }
        if (event == nlohmann::json::parse_event_t::object_end) {
            if (object_keys.empty())
                throw std::runtime_error("Unbalanced JSON object in " + context);
            object_keys.pop_back();
        }
        return true;
    };
    try {
        return nlohmann::json::parse(text, callback);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("Invalid " + context + ": " + error.what());
    }
}

void require_exact_keys(const nlohmann::json& object, std::initializer_list<const char*> expected,
                        const std::string& context) {
    if (!object.is_object())
        throw std::runtime_error(context + " must be a JSON object");
    std::set<std::string> expected_keys;
    for (const char* key : expected)
        expected_keys.insert(key);
    std::set<std::string> actual_keys;
    for (auto it = object.begin(); it != object.end(); ++it)
        actual_keys.insert(it.key());
    if (actual_keys == expected_keys)
        return;
    for (const auto& key : actual_keys) {
        if (expected_keys.count(key) == 0)
            throw std::runtime_error(context + " has unknown field '" + key + "'");
    }
    for (const auto& key : expected_keys) {
        if (actual_keys.count(key) == 0)
            throw std::runtime_error(context + " is missing required field '" + key + "'");
    }
    throw std::runtime_error(context + " has an invalid field set");
}

std::string require_string(const nlohmann::json& object, const char* field,
                           const std::string& context) {
    const auto value = object.find(field);
    if (value == object.end() || !value->is_string() ||
        value->get_ref<const std::string&>().empty())
        throw std::runtime_error(context + " requires non-empty string field '" + field + "'");
    return value->get<std::string>();
}

std::uint64_t require_uint64(const nlohmann::json& object, const char* field,
                             const std::string& context) {
    const auto value = object.find(field);
    if (value == object.end() || !value->is_number_unsigned())
        throw std::runtime_error(context + " requires unsigned integer field '" + field + "'");
    return value->get<std::uint64_t>();
}

int32_t require_positive_int32(const nlohmann::json& object, const char* field,
                               const std::string& context) {
    const std::uint64_t value = require_uint64(object, field, context);
    if (value == 0 || value > static_cast<std::uint64_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error(context + " field '" + field + "' is outside the int32 range");
    return static_cast<int32_t>(value);
}

bool is_safe_identifier(std::string_view value) {
    if (value.empty() || value.size() > 255 || value == "." || value == "..")
        return false;
    return std::all_of(value.begin(), value.end(), [](unsigned char character) {
        return std::isalnum(character) || character == '-' || character == '_' || character == '.';
    });
}

bool is_lower_sha256(const std::string& value) {
    return value.size() == 64 &&
           std::all_of(value.begin(), value.end(), [](unsigned char character) {
               return std::isdigit(character) || (character >= 'a' && character <= 'f');
           });
}

bool unsafe_relative_path_syntax(const std::string& value) {
    return value.empty() || value.size() > kMaxArtifactPathBytes || value.front() == '/' ||
           value.back() == '/' || value.find('\\') != std::string::npos ||
           value.find('\0') != std::string::npos;
}

std::string validated_relative_path(const std::string& value, const std::string& label) {
    if (unsafe_relative_path_syntax(value))
        throw std::runtime_error("Unsafe " + label + " path: " + value);
    const fs::path path(value);
    const std::string normalized = path.lexically_normal().generic_string();
    const bool unsafe_component = std::any_of(
        path.begin(), path.end(), [](const fs::path& part) { return part == "." || part == ".."; });
    if (path.is_absolute() || normalized != value || normalized == "." || unsafe_component)
        throw std::runtime_error("Unsafe " + label + " path: " + value);
    return normalized;
}

void validate_runtime_library(const std::string& value) {
    const bool valid_prefix = value.rfind("libtrtmc_impl_", 0) == 0;
    if (fs::path(value).filename() != fs::path(value) || !valid_prefix || value.size() <= 3 ||
        value.substr(value.size() - 3) != ".so" ||
        !std::all_of(value.begin(), value.end(), [](unsigned char character) {
            return std::isalnum(character) || character == '-' || character == '_' ||
                   character == '.';
        })) {
        throw std::runtime_error("optimized_runtime.json runtime_library must be an exact safe "
                                 "libtrtmc_impl_*.so basename");
    }
}

std::vector<std::string> parse_string_array(const nlohmann::json& object, const char* field,
                                            const std::string& context, bool allow_empty) {
    const auto value = object.find(field);
    if (value == object.end() || !value->is_array())
        throw std::runtime_error(context + " field '" + field + "' must be an array");
    if ((!allow_empty && value->empty()) || value->size() > kMaxArtifactEntries)
        throw std::runtime_error(context + " field '" + field + "' has unsupported cardinality");
    std::vector<std::string> result;
    std::unordered_set<std::string> seen;
    result.reserve(value->size());
    for (const auto& item : *value) {
        if (!item.is_string() || item.get_ref<const std::string&>().empty())
            throw std::runtime_error(context + " field '" + field +
                                     "' must contain non-empty strings");
        const std::string text = item.get<std::string>();
        if (!seen.insert(text).second)
            throw std::runtime_error(context + " field '" + field + "' contains duplicate value '" +
                                     text + "'");
        result.push_back(text);
    }
    return result;
}

RuntimeIdentity parse_runtime_identity(const nlohmann::json& object) {
    const std::string context = "optimized_runtime.json runtime";
    require_exact_keys(object, {"name", "version", "commit"}, context);
    return RuntimeIdentity{require_string(object, "name", context),
                           require_string(object, "version", context),
                           require_string(object, "commit", context)};
}

ArtifactDescriptor parse_artifact_descriptor(const nlohmann::json& object) {
    const std::string context = "optimized_runtime.json artifact";
    require_exact_keys(object,
                       {"section_prefix", "directories", "file_count", "total_size", "tree_sha256"},
                       context);
    ArtifactDescriptor result;
    result.section_prefix = validated_relative_path(
        require_string(object, "section_prefix", context), "optimized-runtime artifact prefix");
    result.directories = parse_string_array(object, "directories", context, true);
    for (auto& directory : result.directories)
        directory = validated_relative_path(directory, "optimized-runtime artifact directory");
    result.file_count = require_uint64(object, "file_count", context);
    result.total_size = require_uint64(object, "total_size", context);
    result.tree_sha256 = require_string(object, "tree_sha256", context);
    if (result.file_count == 0 || result.file_count > kMaxArtifactEntries)
        throw std::runtime_error("optimized_runtime.json artifact.file_count is out of range");
    if (result.total_size > kMaxArtifactTotalSize)
        throw std::runtime_error("optimized_runtime.json artifact.total_size exceeds 1 TiB");
    if (!is_lower_sha256(result.tree_sha256)) {
        throw std::runtime_error(
            "optimized_runtime.json artifact.tree_sha256 must be 64 lowercase hexadecimal "
            "digits");
    }
    if (result.file_count + result.directories.size() > kMaxArtifactEntries)
        throw std::runtime_error("optimized-runtime artifact entry count exceeds 65536");
    return result;
}

OptimizedRuntimeDescriptor parse_descriptor(const std::string& descriptor_json,
                                            const BundleInfo& bundle_info) {
    const nlohmann::json root = parse_json_document(descriptor_json, "optimized_runtime.json");
    const std::string context = "optimized_runtime.json";
    require_exact_keys(root,
                       {"schema_version", "implementation_id", "model_id", "profile_id",
                        "runtime_library", "factory_abi", "implementation_metadata_section",
                        "runtime", "artifact"},
                       context);

    OptimizedRuntimeDescriptor descriptor;
    descriptor.schema_version = require_positive_int32(root, "schema_version", context);
    if (descriptor.schema_version != 2) {
        throw std::runtime_error("Unsupported optimized_runtime.json schema_version " +
                                 std::to_string(descriptor.schema_version) + "; expected 2");
    }
    descriptor.implementation_id = require_string(root, "implementation_id", context);
    descriptor.model_id = require_string(root, "model_id", context);
    descriptor.profile_id = require_string(root, "profile_id", context);
    descriptor.runtime_library = require_string(root, "runtime_library", context);
    descriptor.factory_abi = require_positive_int32(root, "factory_abi", context);
    descriptor.implementation_metadata_section =
        validated_relative_path(require_string(root, "implementation_metadata_section", context),
                                "optimized-runtime implementation metadata section");
    descriptor.runtime = parse_runtime_identity(root.at("runtime"));
    descriptor.artifact = parse_artifact_descriptor(root.at("artifact"));

    if (!is_safe_identifier(descriptor.implementation_id)) {
        throw std::runtime_error(
            "optimized_runtime.json implementation_id must be a safe path component");
    }
    if (!is_safe_identifier(descriptor.profile_id)) {
        throw std::runtime_error("optimized_runtime.json profile_id must be a safe path component");
    }
    validate_runtime_library(descriptor.runtime_library);
    if (descriptor.factory_abi !=
        static_cast<int32_t>(internal::kOptimizedRuntimeFactoryAbiVersionV1)) {
        throw std::runtime_error("optimized_runtime.json factory_abi must be 1");
    }
    if (descriptor.implementation_metadata_section == kDescriptorSection) {
        throw std::runtime_error(
            "optimized_runtime.json implementation_metadata_section must be capsule-owned");
    }
    const std::string artifact_prefix = descriptor.artifact.section_prefix + "/";
    if (descriptor.implementation_metadata_section.rfind(artifact_prefix, 0) == 0) {
        throw std::runtime_error(
            "optimized-runtime implementation metadata may not be inside its artifact tree");
    }
    if (bundle_info.model_id != descriptor.model_id) {
        throw std::runtime_error("Optimized-runtime model mismatch: bundle header declares '" +
                                 bundle_info.model_id + "', descriptor declares '" +
                                 descriptor.model_id + "'");
    }
    return descriptor;
}

struct EmbeddedArtifactFile {
    const BundleSectionInfo* section{nullptr};
    std::string relative_path;
};

struct EmbeddedArtifacts {
    std::vector<EmbeddedArtifactFile> files;
};

void validate_artifact_paths(const ArtifactDescriptor& descriptor,
                             const std::unordered_set<std::string>& file_paths) {
    std::unordered_set<std::string> directories;
    for (const auto& directory : descriptor.directories) {
        if (!directories.insert(directory).second)
            throw std::runtime_error("Duplicate optimized-runtime artifact directory: " +
                                     directory);
        if (file_paths.count(directory) != 0)
            throw std::runtime_error("Optimized-runtime artifact file/directory collision: " +
                                     directory);
        fs::path parent = fs::path(directory).parent_path();
        while (!parent.empty()) {
            if (file_paths.count(parent.generic_string()) != 0) {
                throw std::runtime_error(
                    "Optimized-runtime artifact directory/ancestor collision: " + directory);
            }
            parent = parent.parent_path();
        }
    }
    for (const auto& file : file_paths) {
        fs::path parent = fs::path(file).parent_path();
        while (!parent.empty()) {
            if (file_paths.count(parent.generic_string()) != 0) {
                throw std::runtime_error("Optimized-runtime artifact file/ancestor collision: " +
                                         file);
            }
            parent = parent.parent_path();
        }
    }
}

EmbeddedArtifacts find_embedded_artifacts(const BundleInfo& bundle_info,
                                          const ArtifactDescriptor& descriptor) {
    const std::string prefix = descriptor.section_prefix + "/";
    EmbeddedArtifacts embedded;
    std::unordered_set<std::string> paths;
    std::uint64_t total_size = 0;
    for (const auto& section : bundle_info.sections) {
        if (section.name.rfind(prefix, 0) != 0)
            continue;
        const std::string relative = validated_relative_path(section.name.substr(prefix.size()),
                                                             "optimized-runtime artifact");
        if (!paths.insert(relative).second)
            throw std::runtime_error("Duplicate optimized-runtime artifact section: " + relative);
        if (section.size > kMaxArtifactTotalSize - total_size)
            throw std::runtime_error("Embedded optimized-runtime artifacts exceed 1 TiB");
        total_size += section.size;
        embedded.files.push_back(EmbeddedArtifactFile{&section, relative});
    }
    if (embedded.files.size() != descriptor.file_count || total_size != descriptor.total_size) {
        throw std::runtime_error(
            "Embedded optimized-runtime artifact manifest does not match bundle sections");
    }
    validate_artifact_paths(descriptor, paths);
    std::sort(embedded.files.begin(), embedded.files.end(),
              [](const auto& left, const auto& right) {
                  return left.relative_path < right.relative_path;
              });
    return embedded;
}

using Sha256Digest = std::array<std::uint8_t, 32>;

class HashingStreamBuffer final : public std::streambuf {
  public:
    explicit HashingStreamBuffer(std::streambuf* destination) : destination_(destination) {}
    Sha256Digest digest() const { return digest_.digest(); }

  protected:
    std::streamsize xsputn(const char* source, std::streamsize count) override {
        const std::streamsize written = destination_->sputn(source, count);
        if (written > 0)
            digest_.update(source, static_cast<std::size_t>(written));
        return written;
    }

    int_type overflow(int_type character) override {
        if (traits_type::eq_int_type(character, traits_type::eof()))
            return traits_type::not_eof(character);
        const char value = traits_type::to_char_type(character);
        if (traits_type::eq_int_type(destination_->sputc(value), traits_type::eof()))
            return traits_type::eof();
        digest_.update(&value, 1);
        return character;
    }

    int sync() override { return destination_->pubsync(); }

  private:
    std::streambuf* destination_;
    internal::Sha256 digest_;
};

Sha256Digest copy_and_hash_artifact(const std::string& bundle_path,
                                    const EmbeddedArtifactFile& file, const fs::path& root) {
    const fs::path output_path = root / fs::path(file.relative_path);
    std::error_code error;
    fs::create_directories(output_path.parent_path(), error);
    if (error)
        throw std::runtime_error("Failed to create optimized-runtime artifact directory: " +
                                 error.message());
    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("Failed to create optimized-runtime artifact: " +
                                 output_path.string());
    HashingStreamBuffer buffer(output.rdbuf());
    std::ostream hashing_output(&buffer);
    CopyBundleSection(bundle_path, *file.section, hashing_output);
    hashing_output.flush();
    if (!hashing_output)
        throw std::runtime_error("Failed to write optimized-runtime artifact: " +
                                 output_path.string());
    output.close();
    if (!output)
        throw std::runtime_error("Failed to finalize optimized-runtime artifact: " +
                                 output_path.string());
    fs::permissions(output_path, fs::perms::owner_read | fs::perms::owner_write,
                    fs::perm_options::replace, error);
    if (error)
        throw std::runtime_error("Failed to secure optimized-runtime artifact: " + error.message());
    return buffer.digest();
}

int open_regular_file(const fs::path& path, std::uint64_t expected_size) {
    struct stat metadata{};
    const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (descriptor < 0)
        return -1;
    if (fstat(descriptor, &metadata) != 0 || !S_ISREG(metadata.st_mode) || metadata.st_size < 0 ||
        static_cast<std::uint64_t>(metadata.st_size) != expected_size) {
        (void)close(descriptor);
        return -1;
    }
    return descriptor;
}

bool hash_open_file(int descriptor, internal::Sha256& hash) {
    std::array<char, 1024 * 1024> buffer{};
    while (true) {
        const ssize_t count = read(descriptor, buffer.data(), buffer.size());
        if (count == 0)
            return true;
        if (count < 0) {
            if (errno == EINTR)
                continue;
            return false;
        }
        hash.update(buffer.data(), static_cast<std::size_t>(count));
    }
}

bool hash_file(const fs::path& path, std::uint64_t expected_size, Sha256Digest& digest) {
    const int descriptor = open_regular_file(path, expected_size);
    if (descriptor < 0)
        return false;
    internal::Sha256 hash;
    const bool read_success = hash_open_file(descriptor, hash);
    const bool close_success = close(descriptor) == 0;
    if (!read_success || !close_success)
        return false;
    digest = hash.digest();
    return true;
}

void update_tree_record(internal::Sha256& tree, std::string_view kind,
                        const std::string& relative_path) {
    constexpr char terminator = '\0';
    tree.update(kind);
    tree.update(&terminator, 1);
    tree.update(relative_path);
    tree.update(&terminator, 1);
}

std::string artifact_tree_sha256(const ArtifactDescriptor& descriptor,
                                 const EmbeddedArtifacts& embedded,
                                 const std::vector<Sha256Digest>& file_digests) {
    if (file_digests.size() != embedded.files.size())
        throw std::logic_error("Optimized-runtime artifact digest count mismatch");
    internal::Sha256 tree;
    std::vector<std::string> directories = descriptor.directories;
    std::sort(directories.begin(), directories.end());
    for (const auto& directory : directories)
        update_tree_record(tree, "directory", directory);
    for (std::size_t index = 0; index < embedded.files.size(); ++index) {
        const auto& file = embedded.files[index];
        update_tree_record(tree, "file", file.relative_path);
        tree.update(std::to_string(file.section->size));
        constexpr char terminator = '\0';
        tree.update(&terminator, 1);
        tree.update(file_digests[index].data(), file_digests[index].size());
    }
    return tree.hex_digest();
}

struct MaterializedPaths {
    std::set<std::string> directories;
    std::set<std::string> files;
};

void add_materialized_path(MaterializedPaths& paths, const fs::path& relative,
                           const fs::file_status& status) {
    if (fs::is_symlink(status)) {
        throw std::runtime_error("Materialized optimized-runtime tree contains a symlink: " +
                                 relative.string());
    }
    if (fs::is_directory(status)) {
        paths.directories.insert(relative.generic_string());
        return;
    }
    if (fs::is_regular_file(status)) {
        paths.files.insert(relative.generic_string());
        return;
    }
    throw std::runtime_error("Materialized optimized-runtime tree contains a special file: " +
                             relative.string());
}

MaterializedPaths inspect_materialized_tree(const fs::path& root) {
    std::error_code error;
    const fs::file_status root_status = fs::symlink_status(root, error);
    if (error || fs::is_symlink(root_status) || !fs::is_directory(root_status)) {
        throw std::runtime_error(
            "Materialized optimized-runtime artifact root is not a real directory");
    }
    MaterializedPaths paths;
    fs::recursive_directory_iterator iterator(root, fs::directory_options::none, error);
    if (error)
        throw std::runtime_error("Failed to inspect optimized-runtime artifacts: " +
                                 error.message());
    const fs::recursive_directory_iterator end;
    while (iterator != end) {
        const fs::path relative = iterator->path().lexically_relative(root);
        const fs::file_status status = iterator->symlink_status(error);
        if (error)
            throw std::runtime_error("Failed to inspect optimized-runtime artifacts: " +
                                     error.message());
        add_materialized_path(paths, relative, status);
        iterator.increment(error);
        if (error)
            throw std::runtime_error("Failed to inspect optimized-runtime artifacts: " +
                                     error.message());
    }
    return paths;
}

void validate_materialized_tree_layout(const fs::path& root, const ArtifactDescriptor& descriptor,
                                       const EmbeddedArtifacts& embedded) {
    MaterializedPaths expected;
    expected.directories.insert(descriptor.directories.begin(), descriptor.directories.end());
    for (const auto& file : embedded.files)
        expected.files.insert(file.relative_path);
    const MaterializedPaths actual = inspect_materialized_tree(root);
    if (actual.directories != expected.directories || actual.files != expected.files) {
        throw std::runtime_error(
            "Materialized optimized-runtime tree does not exactly match its manifest");
    }
}

void validate_materialized_tree(const fs::path& root, const ArtifactDescriptor& descriptor,
                                const EmbeddedArtifacts& embedded) {
    validate_materialized_tree_layout(root, descriptor, embedded);
    std::vector<Sha256Digest> hashes;
    hashes.reserve(embedded.files.size());
    for (const auto& file : embedded.files) {
        Sha256Digest digest{};
        if (!hash_file(root / file.relative_path, file.section->size, digest)) {
            throw std::runtime_error("Invalid materialized optimized-runtime artifact: " +
                                     (root / file.relative_path).string());
        }
        hashes.push_back(digest);
    }
    const std::string actual_hash = artifact_tree_sha256(descriptor, embedded, hashes);
    if (actual_hash != descriptor.tree_sha256) {
        throw std::runtime_error("Optimized-runtime artifact tree SHA-256 mismatch: expected " +
                                 descriptor.tree_sha256 + ", got " + actual_hash);
    }
}

fs::path default_cache_root() {
    const char* xdg_cache = std::getenv("XDG_CACHE_HOME");
    if (xdg_cache != nullptr && xdg_cache[0] != '\0' && fs::path(xdg_cache).is_absolute())
        return fs::path(xdg_cache) / "trtmc";
    const char* home = std::getenv("HOME");
    if (home != nullptr && home[0] != '\0' && fs::path(home).is_absolute())
        return fs::path(home) / ".cache" / "trtmc";
    return fs::temp_directory_path() / ("trtmc-" + std::to_string(getuid()));
}

fs::path artifact_cache_path(const OptimizedRuntimeDescriptor& descriptor,
                             const std::string& requested_cache) {
    const fs::path root =
        requested_cache.empty() ? default_cache_root() : fs::path(requested_cache);
    return root / "optimized-runtimes" / descriptor.implementation_id /
           (descriptor.profile_id + "-" + descriptor.artifact.tree_sha256);
}

void ensure_directory(const fs::path& path) {
    std::error_code error;
    fs::create_directories(path, error);
    if (error)
        throw std::runtime_error("Failed to create optimized-runtime cache directory: " +
                                 error.message());
    fs::permissions(path, fs::perms::owner_all, fs::perm_options::replace, error);
    if (error)
        throw std::runtime_error("Failed to secure optimized-runtime cache directory: " +
                                 error.message());
}

class FileLock {
  public:
    explicit FileLock(const fs::path& path) {
        descriptor_ = open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (descriptor_ < 0 || flock(descriptor_, LOCK_EX) != 0) {
            if (descriptor_ >= 0)
                (void)close(descriptor_);
            descriptor_ = -1;
            throw std::runtime_error("Unable to lock optimized-runtime artifact cache: " +
                                     path.string());
        }
    }
    ~FileLock() {
        if (descriptor_ >= 0) {
            (void)flock(descriptor_, LOCK_UN);
            (void)close(descriptor_);
        }
    }
    FileLock(const FileLock&) = delete;
    FileLock& operator=(const FileLock&) = delete;

  private:
    int descriptor_{-1};
};

class StagingDirectory {
  public:
    explicit StagingDirectory(fs::path path) : path_(std::move(path)) {}
    ~StagingDirectory() {
        if (!path_.empty()) {
            std::error_code ignored;
            fs::remove_all(path_, ignored);
        }
    }
    const fs::path& path() const { return path_; }
    void release() { path_.clear(); }
    StagingDirectory(const StagingDirectory&) = delete;
    StagingDirectory& operator=(const StagingDirectory&) = delete;

  private:
    fs::path path_;
};

fs::path create_staging_directory(const fs::path& output_root) {
    static std::atomic<std::uint64_t> counter{0};
    const fs::path parent = output_root.parent_path();
    for (int attempt = 0; attempt < 32; ++attempt) {
        const fs::path candidate =
            parent / ("." + output_root.filename().string() + ".staging." +
                      std::to_string(static_cast<long long>(getpid())) + "." +
                      std::to_string(counter.fetch_add(1, std::memory_order_relaxed)));
        std::error_code error;
        if (fs::create_directory(candidate, error) && !error) {
            fs::permissions(candidate, fs::perms::owner_all, fs::perm_options::replace, error);
            if (error)
                throw std::runtime_error("Failed to secure artifact staging directory: " +
                                         error.message());
            return candidate;
        }
    }
    throw std::runtime_error("Unable to reserve optimized-runtime artifact staging directory");
}

fs::path materialize_artifacts(const std::string& bundle_path, const BundleInfo& bundle_info,
                               const OptimizedRuntimeDescriptor& descriptor,
                               const std::string& requested_cache) {
    const EmbeddedArtifacts embedded = find_embedded_artifacts(bundle_info, descriptor.artifact);
    const bool contains_runtime_library =
        std::any_of(embedded.files.begin(), embedded.files.end(), [&](const auto& file) {
            return file.relative_path == descriptor.runtime_library;
        });
    if (!contains_runtime_library) {
        throw std::runtime_error("Optimized-runtime artifact tree does not contain declared "
                                 "runtime_library '" +
                                 descriptor.runtime_library + "'");
    }
    const fs::path output_root = artifact_cache_path(descriptor, requested_cache);
    ensure_directory(output_root.parent_path());
    FileLock lock(output_root.parent_path() / ("." + output_root.filename().string() + ".lock"));

    std::error_code error;
    if (fs::exists(output_root, error) && !error) {
        validate_materialized_tree(output_root, descriptor.artifact, embedded);
        return output_root;
    }
    if (error)
        throw std::runtime_error("Failed to inspect optimized-runtime cache: " + error.message());

    StagingDirectory staging(create_staging_directory(output_root));
    for (const auto& directory : descriptor.artifact.directories) {
        fs::create_directories(staging.path() / directory, error);
        if (error)
            throw std::runtime_error("Failed to create optimized-runtime artifact directory: " +
                                     error.message());
    }
    std::vector<Sha256Digest> hashes;
    hashes.reserve(embedded.files.size());
    for (const auto& file : embedded.files)
        hashes.push_back(copy_and_hash_artifact(bundle_path, file, staging.path()));
    const std::string actual_hash = artifact_tree_sha256(descriptor.artifact, embedded, hashes);
    if (actual_hash != descriptor.artifact.tree_sha256) {
        throw std::runtime_error("Optimized-runtime artifact tree SHA-256 mismatch: expected " +
                                 descriptor.artifact.tree_sha256 + ", got " + actual_hash);
    }
    // copy_and_hash_artifact already authenticated every byte written to this
    // private staging tree. Re-check its exact shape without immediately
    // reading and hashing the full artifact payload a second time.
    validate_materialized_tree_layout(staging.path(), descriptor.artifact, embedded);
    fs::rename(staging.path(), output_root, error);
    if (error)
        throw std::runtime_error("Failed to publish optimized-runtime artifact tree: " +
                                 error.message());
    staging.release();
    // The content-addressed tree is now atomically published and validated.
    // Release the cooperative publication lock before any DSO or downstream
    // runtime code executes.
    return output_root;
}

class DsoHandle {
  public:
    explicit DsoHandle(void* handle) : handle_(handle) {}
    ~DsoHandle() {
        if (handle_ != nullptr)
            (void)dlclose(handle_);
    }
    void* get() const { return handle_; }
    void* release() {
        void* result = handle_;
        handle_ = nullptr;
        return result;
    }
    DsoHandle(const DsoHandle&) = delete;
    DsoHandle& operator=(const DsoHandle&) = delete;

  private:
    void* handle_{nullptr};
};

void* open_provider_dso(const fs::path& artifact_path,
                        const OptimizedRuntimeDescriptor& descriptor) {
    // Generic bundles are self-contained. Once the integrity-checked descriptor
    // names an embedded implementation DSO, a load failure is terminal; using
    // an installed same-name library would violate artifact provenance.
    const fs::path candidate = artifact_path / descriptor.runtime_library;
    dlerror();
    void* handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle != nullptr)
        return handle;
    const char* error = dlerror();
    throw std::runtime_error("Unable to load embedded optimized-runtime DSO '" +
                             candidate.string() + "' for implementation '" +
                             descriptor.implementation_id +
                             "': " + (error == nullptr ? "unknown dlopen error" : error));
}

std::string factory_string(const char* value) {
    return value == nullptr ? std::string{} : std::string(value);
}

bool toolchain_abi_matches(const internal::OptimizedRuntimeToolchainAbiV1& left,
                           const internal::OptimizedRuntimeToolchainAbiV1& right) {
    const auto fields = [](const internal::OptimizedRuntimeToolchainAbiV1& abi) {
        return std::array{
            abi.compiler_family,
            abi.compiler_major_version,
            abi.cxx_abi_family,
            abi.cxx_abi_version,
            abi.cxx_language_standard,
            abi.standard_library_family,
            abi.standard_library_version,
            abi.standard_library_abi,
            abi.pointer_size,
            abi.string_size,
            abi.string_alignment,
        };
    };
    return fields(left) == fields(right);
}

const internal::OptimizedRuntimeFactoryV1*
resolve_factory(void* dso, const OptimizedRuntimeDescriptor& descriptor) {
    dlerror();
    void* symbol = dlsym(dso, internal::kOptimizedRuntimeFactoryEntrypointV1);
    const char* error = dlerror();
    if (symbol == nullptr || error != nullptr) {
        throw std::runtime_error(descriptor.runtime_library + " is missing " +
                                 std::string(internal::kOptimizedRuntimeFactoryEntrypointV1));
    }
    const auto getter = reinterpret_cast<internal::GetOptimizedRuntimeFactoryV1>(symbol);
    const internal::OptimizedRuntimeFactoryV1* factory = getter();
    if (factory == nullptr)
        throw std::runtime_error(descriptor.runtime_library +
                                 " returned a null optimized-runtime factory");
    return factory;
}

void validate_factory(const internal::OptimizedRuntimeFactoryV1& factory,
                      const OptimizedRuntimeDescriptor& descriptor) {
    if (factory.abi_version != internal::kOptimizedRuntimeFactoryAbiVersionV1 ||
        factory.struct_size < sizeof(internal::OptimizedRuntimeFactoryV1) ||
        factory.create == nullptr) {
        throw std::runtime_error("Optimized-runtime DSO has an invalid factory v1 table");
    }
    if (factory.pipeline_abi_version != internal::kOptimizedRuntimePipelineAbiVersionV1) {
        throw std::runtime_error(
            "Optimized-runtime IPipeline ABI version mismatch for implementation '" +
            descriptor.implementation_id + "': expected " +
            std::to_string(internal::kOptimizedRuntimePipelineAbiVersionV1) + ", got " +
            std::to_string(factory.pipeline_abi_version));
    }
    if (!toolchain_abi_matches(factory.toolchain_abi,
                               internal::kCurrentOptimizedRuntimeToolchainAbiV1)) {
        throw std::runtime_error(
            "Optimized-runtime C++ toolchain ABI mismatch for implementation '" +
            descriptor.implementation_id +
            "'; rebuild the provider with the Model Connect compiler and standard library");
    }
    if (factory_string(factory.implementation_id) != descriptor.implementation_id) {
        throw std::runtime_error(
            "Optimized-runtime factory implementation identity mismatch for '" +
            descriptor.implementation_id + "'");
    }
    if (factory_string(factory.runtime_name) != descriptor.runtime.name) {
        throw std::runtime_error("Optimized-runtime name mismatch for implementation '" +
                                 descriptor.implementation_id + "'");
    }
    if (factory_string(factory.runtime_version) != descriptor.runtime.version) {
        throw std::runtime_error("Optimized-runtime version mismatch for implementation '" +
                                 descriptor.implementation_id + "'");
    }
    if (factory_string(factory.runtime_commit) != descriptor.runtime.commit) {
        throw std::runtime_error("Optimized-runtime source revision mismatch for implementation '" +
                                 descriptor.implementation_id + "'");
    }
}

std::unique_ptr<IPipeline> create_pipeline(const internal::OptimizedRuntimeFactoryV1& factory,
                                           const OptimizedRuntimeDescriptor& descriptor,
                                           const std::string& bundle_path,
                                           const fs::path& artifact_path,
                                           const std::string& implementation_metadata,
                                           const LoadOptions& options) {
    const std::string artifact_path_text = artifact_path.string();
    internal::OptimizedRuntimePipelineCreateRequestV1 request{};
    request.abi_version = internal::kOptimizedRuntimeFactoryAbiVersionV1;
    request.struct_size = sizeof(request);
    request.implementation_id = descriptor.implementation_id.c_str();
    request.model_id = descriptor.model_id.c_str();
    request.profile_id = descriptor.profile_id.c_str();
    request.bundle_path = bundle_path.c_str();
    request.artifact_path = artifact_path_text.c_str();
    request.implementation_metadata = implementation_metadata.data();
    request.implementation_metadata_size = implementation_metadata.size();
    request.load_options = &options;

    std::array<char, kErrorCapacity> error{};
    IPipeline* raw = factory.create(&request, error.data(), error.size());
    error.back() = '\0';
    if (raw == nullptr) {
        std::string message = "Optimized-runtime implementation '" + descriptor.implementation_id +
                              "' failed to create its pipeline";
        if (error[0] != '\0')
            message += ": " + std::string(error.data());
        throw std::runtime_error(message);
    }

    std::unique_ptr<IPipeline> pipeline(raw);
    const char* actual_model = pipeline->model_id();
    const char* actual_pipeline_type = pipeline->pipeline_type();
    if (actual_model == nullptr || std::string(actual_model) != descriptor.model_id) {
        throw std::runtime_error(
            "Optimized-runtime pipeline model identity mismatch for implementation '" +
            descriptor.implementation_id + "'");
    }
    if (actual_pipeline_type == nullptr || actual_pipeline_type[0] == '\0') {
        throw std::runtime_error("Optimized-runtime pipeline type is empty for implementation '" +
                                 descriptor.implementation_id + "'");
    }
    return pipeline;
}

bool retain_dso_for_process_lifetime(void* dso) {
    // IPipeline's virtual methods and destructor are implemented by the model-
    // owned DSO. Intentionally leak one reference per dynamic-loader identity
    // to avoid both premature dlclose and static-destruction-order hazards.
    // The loader handle is safer than a path identity: repeated dlopen calls
    // for one loaded object return the same handle, while a replaced file may
    // be loaded as a distinct object even when its canonical path is unchanged.
    static auto* mutex = new std::mutex;
    static auto* handles = new std::unordered_set<void*>;
    std::lock_guard<std::mutex> lock(*mutex);
    return handles->insert(dso).second;
}

} // namespace

std::unique_ptr<IPipeline> try_make_optimized_runtime_pipeline(const std::string& bundle_path,
                                                               const BundleInfo& bundle_info,
                                                               const LoadOptions& options) {
    if (find_section(bundle_info, kDescriptorSection) == nullptr)
        return nullptr;

    // Presence of optimized_runtime.json claims the generic capsule path. No
    // error below may be converted into native or alternate-runtime fallback.
    const std::string descriptor_json =
        read_text_section(bundle_path, bundle_info, kDescriptorSection, kMaxDescriptorSize);
    const OptimizedRuntimeDescriptor descriptor = parse_descriptor(descriptor_json, bundle_info);
    const std::string implementation_metadata =
        read_text_section(bundle_path, bundle_info, descriptor.implementation_metadata_section,
                          kMaxImplementationMetadataSize);
    // Private metadata validity and target/profile compatibility belong to the
    // model-owned capsule DSO. The host deliberately treats these bytes as
    // opaque and only owns their bounded transport.
    const fs::path artifacts =
        materialize_artifacts(bundle_path, bundle_info, descriptor, options.runtime_cache_path);
    DsoHandle dso(open_provider_dso(artifacts, descriptor));
    const internal::OptimizedRuntimeFactoryV1* factory = resolve_factory(dso.get(), descriptor);
    validate_factory(*factory, descriptor);
    auto pipeline = create_pipeline(*factory, descriptor, bundle_path, artifacts,
                                    implementation_metadata, options);
    if (retain_dso_for_process_lifetime(dso.get()))
        (void)dso.release();
    std::cerr << "[trtmc] Optimized-runtime implementation initialized during load (model="
              << pipeline->model_id() << ", implementation=" << descriptor.implementation_id
              << ", pipeline_type=" << pipeline->pipeline_type() << ")" << std::endl;
    return pipeline;
}

} // namespace trtmc
