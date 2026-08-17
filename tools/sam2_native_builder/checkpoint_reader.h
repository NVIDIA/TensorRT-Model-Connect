/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::sam2::native {

inline constexpr char kSupportedCheckpointSha256[] =
    "89fd676560809c8504411b574cea305c86db1f65bda790ec7fe16cedc6c6ff73";

enum class DType {
    kFloat32,
    kInt64,
};

const char* dtypeName(DType dtype) noexcept;
std::size_t elementSize(DType dtype) noexcept;

class CheckpointError final : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

struct TensorInfo {
    std::string name;
    DType dtype{DType::kFloat32};
    std::string storage_key;
    std::uint64_t storage_offset{0}; // Elements, not bytes.
    std::uint64_t storage_elements{0};
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> strides;
    std::uint64_t logical_elements{0};
    std::size_t logical_bytes{0};
    std::size_t storage_span_bytes{0};
    bool contiguous{false};
};

// data points at the tensor's first addressed storage element and remains
// valid until its CheckpointReader is destroyed. bytes is the logical byte
// count. For a non-contiguous tensor, storage_span_bytes includes holes and
// callers must honor strides or use CheckpointReader::copyTensor().
struct WeightView {
    const void* data{nullptr};
    std::size_t bytes{0};
    DType dtype{DType::kFloat32};
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> strides;
    bool contiguous{false};
    std::size_t storage_span_bytes{0};
};

struct ReaderLimits {
    std::uint64_t max_archive_bytes{UINT32_MAX};
    std::size_t max_archive_members{16384};
    std::size_t max_pickle_bytes{16U * 1024U * 1024U};
    std::size_t max_pickle_stack{65536};
    std::size_t max_pickle_memo{131072};
    std::size_t max_string_bytes{1024U * 1024U};
    std::size_t max_tensors{16384};
    std::size_t max_dimensions{16};
    std::uint64_t max_tensor_logical_bytes{UINT32_MAX};
};

class CheckpointReader final {
  public:
    // The one-argument production path authenticates the delivered SAM2
    // checkpoint before inspecting any ZIP or pickle bytes.
    static CheckpointReader open(const std::filesystem::path& path, ReaderLimits limits = {});
    static CheckpointReader open(const std::filesystem::path& path,
                                 std::string_view expected_sha256, ReaderLimits limits = {});

    // Exposed so tests and controlled tooling can pin a different synthetic
    // archive without weakening the production default above.
    static std::string checkpointSha256(const std::filesystem::path& path,
                                        std::uint64_t max_archive_bytes = UINT32_MAX);

    ~CheckpointReader();
    CheckpointReader(CheckpointReader&&) noexcept;
    CheckpointReader& operator=(CheckpointReader&&) noexcept;

    CheckpointReader(const CheckpointReader&) = delete;
    CheckpointReader& operator=(const CheckpointReader&) = delete;

    std::size_t tensorCount() const noexcept;
    std::size_t storageCount() const noexcept;
    std::vector<std::string> tensorNames() const;

    const TensorInfo& tensorInfo(std::string_view name) const;
    WeightView tensor(std::string_view name) const;
    WeightView requireTensor(std::string_view name, DType dtype,
                             const std::vector<std::int64_t>& shape) const;
    WeightView requireTensor(std::string_view name, DType dtype,
                             std::initializer_list<std::int64_t> shape) const;

    // Returns a tightly packed, row-major copy. This is a memcpy for
    // contiguous tensors and a checked strided gather otherwise.
    std::vector<std::uint8_t> copyTensor(std::string_view name) const;

  private:
    struct Impl;
    explicit CheckpointReader(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> impl_;
};

} // namespace trtmc::sam2::native
