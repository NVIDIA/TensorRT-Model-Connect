/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_view.h"

#include <algorithm>

namespace trtmc {

const std::vector<char>* find_section(const BundleFile& bundle, const std::string& name)
{
    for (const auto& section : bundle.sections)
    {
        if (section.name == name)
            return &section.data;
    }
    return nullptr;
}

std::vector<const std::vector<char>*> find_sections_by_prefix(
    const BundleFile& bundle, const std::string& prefix)
{
    // Collect matching sections with their names for sorting
    struct Match {
        std::string name;
        const std::vector<char>* data;
    };
    std::vector<Match> matches;

    for (const auto& section : bundle.sections)
    {
        if (section.name.size() >= prefix.size()
            && section.name.compare(0, prefix.size(), prefix) == 0)
        {
            matches.push_back({section.name, &section.data});
        }
    }

    std::sort(matches.begin(), matches.end(),
        [](const Match& a, const Match& b) { return a.name < b.name; });

    std::vector<const std::vector<char>*> result;
    result.reserve(matches.size());
    for (const auto& m : matches)
        result.push_back(m.data);
    return result;
}

} // namespace trtmc
