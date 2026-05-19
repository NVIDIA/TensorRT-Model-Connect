#include "runtime/deployment/artifact_store.h"

#include "bundle/bundle_view.h"

#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace trtmc::deployment {

namespace {

std::string sanitize_component(std::string text) {
    for (char& c : text) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.';
        if (!ok)
            c = '_';
    }
    if (text.empty())
        text = "artifact";
    return text;
}

void reject_unsafe_relative_path(const std::filesystem::path& rel) {
    if (rel.empty() || rel.is_absolute())
        throw std::runtime_error("Unsafe deployment artifact path: " + rel.string());
    for (const auto& part : rel) {
        if (part == "..") {
            throw std::runtime_error("Unsafe deployment artifact path: " + rel.string());
        }
    }
}

void write_file(const std::filesystem::path& path, const std::vector<char>& data) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("Failed to create deployment artifact: " + path.string());
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
    if (!out)
        throw std::runtime_error("Failed to write deployment artifact: " + path.string());
}

} // namespace

std::filesystem::path default_artifact_cache_root(const std::string& bundle_path,
                                                  const std::string& variant_id) {
    std::ostringstream key;
    key << std::hex << std::hash<std::string>{}(bundle_path + "|" + variant_id);
    return std::filesystem::temp_directory_path() / "trtmc_artifacts" / key.str();
}

ArtifactStore::ArtifactStore(const BundleFile& bundle, std::string bundle_path,
                             std::string cache_root)
    : bundle_(bundle)
    , bundle_path_(std::move(bundle_path))
    , cache_root_(cache_root.empty() ? default_artifact_cache_root(bundle_path_, "")
                                     : std::filesystem::path(cache_root)) {}

const std::vector<char>* ArtifactStore::read(const std::string& name) const {
    return find_section(bundle_, name);
}

std::filesystem::path ArtifactStore::artifact_path(const std::string& section_name) const {
    return cache_root_ / sanitize_component(section_name);
}

std::filesystem::path ArtifactStore::materialize(const std::string& name) const {
    const auto* data = read(name);
    if (data == nullptr)
        throw std::runtime_error("Deployment artifact section not found: " + name);
    const auto path = artifact_path(name);
    write_file(path, *data);
    return path;
}

std::filesystem::path ArtifactStore::materialize_directory(
    const std::string& section_prefix) const {
    if (section_prefix.empty())
        throw std::runtime_error("Deployment directory artifact requires a section prefix");

    const std::filesystem::path out_dir =
        cache_root_ / sanitize_component(section_prefix);
    bool wrote_any = false;
    for (const auto& section : bundle_.sections) {
        if (section.name.rfind(section_prefix, 0) != 0)
            continue;
        std::filesystem::path rel(section.name.substr(section_prefix.size()));
        reject_unsafe_relative_path(rel);
        write_file(out_dir / rel, section.data);
        wrote_any = true;
    }
    if (!wrote_any) {
        throw std::runtime_error(
            "Deployment directory artifact had no matching sections: " + section_prefix);
    }
    return out_dir;
}

} // namespace trtmc::deployment
