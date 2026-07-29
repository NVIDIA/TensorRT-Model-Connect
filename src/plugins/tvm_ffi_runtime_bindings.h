/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#if TRTMC_HAS_TVM_FFI

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

namespace trtmc {

// Owns the TVM-FFI module and function handles for one runtime-bound slot.
class TvmFfiBoundFunction {
  public:
    // Adopts owned TVM-FFI handles. This header is internal to the runtime.
    TvmFfiBoundFunction(void* module, void* function) noexcept
        : module_(module), function_(function) {}
    ~TvmFfiBoundFunction();

    TvmFfiBoundFunction(const TvmFfiBoundFunction&) = delete;
    TvmFfiBoundFunction& operator=(const TvmFfiBoundFunction&) = delete;

    void* handle() const noexcept { return function_; }

  private:
    void* module_{nullptr};
    void* function_{nullptr};
};

using TvmFfiBoundFunctionPtr = std::shared_ptr<const TvmFfiBoundFunction>;

// Immutable mapping from serialized TensorRT plugin names to loaded TVM-FFI functions.
class TvmFfiBindingSet {
  public:
    ~TvmFfiBindingSet();

    static std::shared_ptr<const TvmFfiBindingSet> Load(std::string_view slot_descriptor_json,
                                                        const std::string& bindings_path);

    TvmFfiBoundFunctionPtr find(std::string_view kernel_name) const noexcept;
    std::size_t size() const noexcept;
    bool was_captured() const noexcept { return captured_.load(); }

  private:
    struct Impl;
    explicit TvmFfiBindingSet(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
    mutable std::atomic<bool> captured_{false};
};

// A slot name is reserved only when it begins with this prefix and has a non-empty ID.
bool is_runtime_tvm_ffi_kernel_name(std::string_view kernel_name) noexcept;

// Look up a slot while TensorRT is synchronously deserializing an engine.
TvmFfiBoundFunctionPtr active_tvm_ffi_binding(std::string_view kernel_name) noexcept;

// Preserve the legacy global-registry path while giving its owned handle RAII lifetime.
TvmFfiBoundFunctionPtr resolve_global_tvm_ffi_function(std::string_view name) noexcept;

// Publishes one immutable binding set only for the pipeline-load operation. The
// recursive process-wide lock serializes concurrent engine loads; plugin clones
// retain their own shared handles after the scope exits.
class ScopedTvmFfiBindings {
  public:
    explicit ScopedTvmFfiBindings(std::shared_ptr<const TvmFfiBindingSet> bindings);
    ~ScopedTvmFfiBindings() noexcept;

    ScopedTvmFfiBindings(const ScopedTvmFfiBindings&) = delete;
    ScopedTvmFfiBindings& operator=(const ScopedTvmFfiBindings&) = delete;

  private:
    std::unique_lock<std::recursive_mutex> lock_;
    bool active_{false};
};

} // namespace trtmc

#endif // TRTMC_HAS_TVM_FFI
