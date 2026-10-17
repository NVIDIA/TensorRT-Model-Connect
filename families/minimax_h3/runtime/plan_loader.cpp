/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/plan_loader.h"

#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <stdexcept>
#include <sys/stat.h>
#include <utility>

namespace trtmc {
namespace {

struct FileIdentity {
    struct stat value {};

    explicit FileIdentity(const std::string& path) {
        if (::stat(path.c_str(), &value) != 0 || !S_ISREG(value.st_mode))
            throw std::runtime_error("MiniMax-H3 cannot stat regular bundle file: " + path);
    }

    bool operator==(const FileIdentity& other) const {
        const auto& a = value;
        const auto& b = other.value;
        return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size &&
               a.st_mtim.tv_sec == b.st_mtim.tv_sec && a.st_mtim.tv_nsec == b.st_mtim.tv_nsec &&
               a.st_ctim.tv_sec == b.st_ctim.tv_sec && a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
    }
};

bool same_metadata(const BundleInfo& a, const BundleInfo& b) {
    if (a.format != b.format || a.family != b.family || a.task != b.task ||
        a.backend != b.backend || a.sections.size() != b.sections.size())
        return false;
    for (const auto& section : a.sections) {
        const auto found =
            std::find_if(b.sections.begin(), b.sections.end(),
                         [&](const auto& other) { return section.name == other.name; });
        if (found == b.sections.end() || section.offset != found->offset ||
            section.length != found->length)
            return false;
    }
    return true;
}

} // namespace

struct MiniMaxH3PlanSource::State {
    FileIdentity identity;
    BundleReader bundle;

    explicit State(const BundleReader& reader) : identity(reader.path()), bundle(reader.path()) {
        verify();
        if (!same_metadata(reader.info(), bundle.info()))
            throw std::runtime_error("MiniMax-H3 bundle metadata changed before family creation");
    }

    void verify() const {
        if (!(identity == FileIdentity(bundle.path())))
            throw std::runtime_error("MiniMax-H3 bundle changed after family creation: " +
                                     bundle.path());
    }
};

MiniMaxH3PlanSource::MiniMaxH3PlanSource(const BundleReader& reader)
    : state_(std::make_unique<State>(reader)) {}

MiniMaxH3PlanSource::~MiniMaxH3PlanSource() = default;

void MiniMaxH3PlanSource::require_nonempty(std::string_view name) const {
    state_->verify();
    const auto* section = state_->bundle.find_section(name);
    if (!section || section->length == 0)
        throw std::runtime_error("MiniMax-H3 bundle section is missing or empty: " +
                                 std::string(name));
}

std::vector<char> MiniMaxH3PlanSource::read_required(std::string_view name) const {
    require_nonempty(name);
    auto result = state_->bundle.read_section(name);
    state_->verify();
    return result;
}

MiniMaxH3ModuleLoader make_minimax_h3_plan_loader(IBackend& backend,
                                                  std::shared_ptr<const MiniMaxH3PlanSource> source,
                                                  std::vector<std::string> plan_names) {
    if (!source || plan_names.empty())
        throw std::invalid_argument("MiniMax-H3 plan loader requires a source and plan names");
    std::sort(plan_names.begin(), plan_names.end());
    if (std::adjacent_find(plan_names.begin(), plan_names.end()) != plan_names.end())
        throw std::invalid_argument("MiniMax-H3 plan loader has duplicate plan names");
    for (const auto& name : plan_names)
        source->require_nonempty(name);
    return [&backend, source = std::move(source),
            names = std::move(plan_names)](const std::string& name, cudaStream_t stream) {
        if (!std::binary_search(names.begin(), names.end(), name))
            throw std::runtime_error("MiniMax-H3 requested undeclared plan: " + name);
        // Both supported backends synchronously deserialize this input. The
        // local payload is released on return, including every failure path.
        const auto plan = source->read_required(name);
        ModuleCreateOptions options{};
        options.stream = stream;
        auto module = backend.create_module(plan.data(), plan.size(), options);
        if (!module || !module->ok())
            throw std::runtime_error("MiniMax-H3 failed to load plan: " + name);
        source->require_nonempty(name);
        return module;
    };
}

} // namespace trtmc
