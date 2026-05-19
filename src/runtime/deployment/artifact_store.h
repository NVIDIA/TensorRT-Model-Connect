#pragma once

#include "bundle/bundle_format.h"

#include <filesystem>
#include <string>
#include <vector>

namespace trtmc::deployment {

class ArtifactStore {
  public:
    ArtifactStore(const BundleFile& bundle, std::string bundle_path, std::string cache_root);

    const std::vector<char>* read(const std::string& name) const;
    std::filesystem::path materialize(const std::string& name) const;
    std::filesystem::path materialize_directory(const std::string& section_prefix) const;

    const std::filesystem::path& cache_root() const {
        return cache_root_;
    }

  private:
    const BundleFile& bundle_;
    std::string bundle_path_;
    std::filesystem::path cache_root_;

    std::filesystem::path artifact_path(const std::string& section_name) const;
};

std::filesystem::path default_artifact_cache_root(const std::string& bundle_path,
                                                  const std::string& variant_id);

} // namespace trtmc::deployment
