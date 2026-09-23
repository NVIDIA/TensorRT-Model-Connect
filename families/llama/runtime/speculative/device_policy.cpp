/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/llama/runtime/speculative/device_policy.h"

#include "families/llama/runtime/plugin_helpers.h"

#include <algorithm>
#include <stdexcept>

namespace trtmc::llama::speculative {
namespace {
void checked(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
std::shared_ptr<void> pinned(std::size_t bytes) {
    void* pointer = nullptr;
    checked(cudaMallocHost(&pointer, bytes));
    return {pointer, [](void* p) { cudaFreeHost(p); }};
}
} // namespace

DeviceEagle3::DeviceEagle3(const FamilyContext& context, Engine& target, Engine& draft, int depth)
    : target_(target), draft_(draft), max_depth_(depth) {
    auto load = [&](const std::string& name) {
        auto module =
            load_engine(context.backend, require_section(context.reader, name), name.c_str());
        return module;
    };
    mapping_ = load("policy_mapping.plan");
    features_ = load("policy_features.plan");
    kv_ = load("policy_kv.plan");
    for (int width : {1, 2}) {
        for (int level = 0; level <= depth; ++level) {
            const auto suffix = std::to_string(width) + "_" + std::to_string(level) + ".plan";
            prepare_[width - 1].push_back(load("policy_prepare_" + suffix));
            accept_[width - 1].push_back(load("policy_accept_" + suffix));
        }
    }
    const auto stream = mapping_->stream();
    starts_ = DeviceTensor({depth + 1}, DType::kInt32, stream);
    tokens_ = DeviceTensor({2 * depth + 1}, DType::kInt32, stream);
    valid_ = DeviceTensor({depth}, DType::kInt32, stream);
    limits_ = DeviceTensor({2}, DType::kInt32, stream);
    root_ = DeviceTensor({1}, DType::kInt32, stream);
    if (!starts_.ok() || !tokens_.ok() || !valid_.ok() || !limits_.ok() || !root_.ok())
        throw std::runtime_error("device policy allocation failed");
    upload_ = pinned((depth + 4) * sizeof(std::int32_t));
    readback_ = pinned((7 * depth + 16) * sizeof(std::int32_t));
    cudaEvent_t event = nullptr;
    checked(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    event_ = {event, [](void* p) { cudaEventDestroy(static_cast<cudaEvent_t>(p)); }};
}

DeviceEagle3::~DeviceEagle3() {
    // No borrowed buffer may be freed while queued model/policy work reads it.
    cudaStreamSynchronize(target_.stream());
    cudaStreamSynchronize(draft_.stream());
    mapping_->sync();
    features_->sync();
    kv_->sync();
    for (int width = 0; width < 2; ++width) {
        for (auto& module : prepare_[width])
            module->sync();
        for (auto& module : accept_[width])
            module->sync();
    }
}

void DeviceEagle3::order(cudaStream_t consumer, cudaStream_t producer) {
    if (consumer == producer)
        return;
    auto event = static_cast<cudaEvent_t>(event_.get());
    checked(cudaEventRecord(event, producer));
    checked(cudaStreamWaitEvent(consumer, event, 0));
}

void DeviceEagle3::begin(FeatureView features) {
    root_on_device_ = false;
    adopt(features);
}

void DeviceEagle3::adopt(FeatureView features) {
    draft_result_ = draft_.device_result(features);
}

ITrtModule& DeviceEagle3::prepare(int width, int depth, int offset, cudaStream_t ready) {
    auto& module = *prepare_[width - 1][depth];
    order(module.stream(), ready);
    module.bind_external("start", static_cast<std::int32_t*>(starts_.data()) + offset);
    module.forward_device_async({});
    return module;
}

void DeviceEagle3::propose_and_verify(int root, int committed, int remaining, int width,
                                      bool ignore_eos) {
    width_ = width;
    depth_ =
        std::min({max_depth_, remaining, (target_.contract().capacity - committed - 1) / width});
    rows_ = 1 + width * depth_;
    auto* upload = static_cast<std::int32_t*>(upload_.get());
    for (int offset = 0; offset <= max_depth_; ++offset)
        upload[offset] = committed + offset;
    upload[max_depth_ + 1] = root;
    upload[max_depth_ + 2] = remaining;
    upload[max_depth_ + 3] = ignore_eos;
    const auto stream = mapping_->stream();
    // The prior feedback must consume the old start values before replacing
    // them. This stream dependency does not block the CPU drafting loop.
    order(stream, draft_result_.ready);
    order(stream, target_.stream()); // finish any previous tree commit before reusing starts_
    checked(cudaMemcpyAsync(starts_.data(), upload, (max_depth_ + 1) * sizeof(std::int32_t),
                            cudaMemcpyHostToDevice, stream));
    checked(cudaMemcpyAsync(limits_.data(), upload + max_depth_ + 2, 2 * sizeof(std::int32_t),
                            cudaMemcpyHostToDevice, stream));
    checked(cudaMemcpyAsync(
        tokens_.data(), root_on_device_ ? root_.data() : upload + max_depth_ + 1,
        sizeof(std::int32_t), root_on_device_ ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice,
        stream));
    for (int step = 0; step < depth_; ++step) {
        order(stream, draft_result_.ready);
        mapping_->bind_external("selection", const_cast<std::int32_t*>(draft_result_.selection));
        mapping_->forward_device_async({});
        auto* candidate = static_cast<std::int32_t*>(tokens_.data()) + 1 + step * width;
        checked(cudaMemcpyAsync(candidate, mapping_->device_ptr("tokens"),
                                width * sizeof(std::int32_t), cudaMemcpyDeviceToDevice, stream));
        checked(cudaMemcpyAsync(static_cast<std::int32_t*>(valid_.data()) + step,
                                mapping_->device_ptr("valid"), sizeof(std::int32_t),
                                cudaMemcpyDeviceToDevice, stream));
        if (step + 1 < depth_) {
            auto& metadata = prepare(1, 0, step, stream);
            order(draft_.stream(), metadata.stream());
            draft_result_ = draft_.run_device(
                candidate, 1, metadata, static_cast<std::int32_t*>(starts_.data()) + step, false,
                {}, draft_result_.features.slice(draft_result_.features.rows - 1, 1));
        }
    }
    auto& metadata = prepare(width, depth_, 0, stream);
    order(target_.stream(), metadata.stream());
    target_result_ = target_.run_device(static_cast<std::int32_t*>(tokens_.data()), rows_, metadata,
                                        static_cast<std::int32_t*>(starts_.data()), true);
}

std::pair<CandidateTree, StepResult> DeviceEagle3::read_verification() {
    auto* host = static_cast<std::int32_t*>(readback_.get());
    const auto stream = mapping_->stream();
    order(stream, target_result_.ready);
    checked(cudaMemcpyAsync(host, tokens_.data(), rows_ * sizeof(std::int32_t),
                            cudaMemcpyDeviceToHost, stream));
    checked(cudaMemcpyAsync(host + rows_, target_result_.selection,
                            2 * rows_ * sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream));
    if (depth_)
        checked(cudaMemcpyAsync(host + 3 * rows_, valid_.data(), depth_ * sizeof(std::int32_t),
                                cudaMemcpyDeviceToHost, stream));
    checked(cudaStreamSynchronize(stream));
    for (int step = 0; step < depth_; ++step) {
        if (host[3 * rows_ + step] != 1)
            throw std::runtime_error("non-finite draft logits");
    }
    CandidateTree tree;
    tree.tokens.assign(host, host + rows_);
    tree.parents.push_back(-1);
    for (int step = 0; step < depth_; ++step)
        for (int sibling = 0; sibling < width_; ++sibling)
            tree.parents.push_back(step == 0 ? 0 : 1 + (step - 1) * width_);
    StepResult result;
    result.selection.assign(host + rows_, host + 3 * rows_);
    result.vocab = target_.contract().vocab;
    result.ranks = 1;
    result.features = target_result_.features;
    return {std::move(tree), std::move(result)};
}

DeviceAcceptance DeviceEagle3::accept(int committed) {
    auto& module = *accept_[width_ - 1][depth_];
    order(module.stream(), target_result_.ready);
    module.bind_external("tokens", tokens_.data());
    module.bind_external("selection", const_cast<std::int32_t*>(target_result_.selection));
    module.bind_external("limits", limits_.data());
    if (depth_)
        module.bind_external("draft_valid", valid_.data());
    module.forward_device_async({});
    auto* host = static_cast<std::int32_t*>(readback_.get());
    checked(cudaMemcpyAsync(root_.data(), module.device_ptr("bonus"), sizeof(std::int32_t),
                            cudaMemcpyDeviceToDevice, module.stream()));
    checked(cudaMemcpyAsync(host, module.device_ptr("status"), 4 * sizeof(std::int32_t),
                            cudaMemcpyDeviceToHost, module.stream()));
    checked(cudaMemcpyAsync(host + 4, module.device_ptr("tokens_out"),
                            (depth_ + 1) * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                            module.stream()));
    module.sync();
    if (host[3] != 1)
        throw std::runtime_error("non-finite logits consumed by device policy");
    if (host[0] < 1 || host[0] > depth_ + 1 || host[1] < 1 || host[1] > host[0])
        throw std::runtime_error("invalid device acceptance status");
    root_on_device_ = true;
    if (width_ == 2) {
        order(kv_->stream(), module.stream());
        target_.commit_device(committed, static_cast<std::int32_t*>(starts_.data()),
                              static_cast<std::int32_t*>(module.device_ptr("path")), host[0], *kv_);
        order(target_.stream(), kv_->stream());
    }
    return {host[0], host[2] != 0, {host + 4, host + 4 + host[1]}};
}

void DeviceEagle3::feedback(int length) {
    auto& accepted = *accept_[width_ - 1][depth_];
    order(features_->stream(), accepted.stream());
    features_->bind_external("features", const_cast<std::uint16_t*>(target_result_.features.data),
                             {rows_, target_.contract().feature_width});
    features_->bind_external("path", accepted.device_ptr("path"), {length});
    features_->forward_device_async({});
    auto& metadata = prepare(1, length - 1, 0, features_->stream());
    order(draft_.stream(), metadata.stream());
    draft_result_ =
        draft_.run_device(static_cast<std::int32_t*>(accepted.device_ptr("tokens_out")), length,
                          metadata, static_cast<std::int32_t*>(starts_.data()), false,
                          {static_cast<const std::uint16_t*>(features_->device_ptr("features_out")),
                           length, target_.contract().feature_width});
}

void DeviceEagle3::finish() {
    checked(cudaStreamSynchronize(target_.stream()));
}

} // namespace trtmc::llama::speculative
