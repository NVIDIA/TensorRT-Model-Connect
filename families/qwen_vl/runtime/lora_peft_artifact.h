/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {
namespace qwen_vl {

enum class PeftTensorDType {
    kFloat16,
    kBFloat16,
    kFloat32,
};

struct PeftTensorInfo {
    std::string name;
    PeftTensorDType dtype{PeftTensorDType::kFloat32};
    std::vector<int64_t> shape;
    std::size_t element_count{0};
};

struct PeftLoraConfig {
    int rank{0};
    double alpha{0.0};
    std::vector<std::string> target_modules;

    double scale() const { return alpha / static_cast<double>(rank); }
};

// Qwen-VL's reader for the standard PEFT adapter_config.json and
// adapter_model.safetensors pair.
class PeftLoraArtifact {
  public:
    static PeftLoraArtifact load(const std::string& adapter_dir);

    ~PeftLoraArtifact();
    PeftLoraArtifact(PeftLoraArtifact&&) noexcept;
    PeftLoraArtifact& operator=(PeftLoraArtifact&&) noexcept;
    PeftLoraArtifact(const PeftLoraArtifact&) = delete;
    PeftLoraArtifact& operator=(const PeftLoraArtifact&) = delete;

    const PeftLoraConfig& config() const;
    const std::vector<PeftTensorInfo>& tensors() const;
    float read_float(const std::string& tensor_name, std::size_t index) const;

  private:
    struct Impl;
    explicit PeftLoraArtifact(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace qwen_vl
} // namespace trtmc
