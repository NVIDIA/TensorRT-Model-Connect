/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>

namespace trtmc {

template <typename T>
class Span {
  public:
    constexpr Span() noexcept = default;
    constexpr Span(T* data, std::size_t size) noexcept : data_(data), size_(size) {}

    template <std::size_t Size>
    constexpr Span(T (&data)[Size]) noexcept : data_(data), size_(Size) {}

    constexpr T* data() const noexcept { return data_; }
    constexpr std::size_t size() const noexcept { return size_; }
    constexpr bool empty() const noexcept { return size_ == 0; }
    constexpr T* begin() const noexcept { return data_; }
    constexpr T* end() const noexcept { return size_ == 0 ? data_ : data_ + size_; }
    constexpr T& operator[](std::size_t index) const noexcept { return data_[index]; }

  private:
    T* data_{nullptr};
    std::size_t size_{0};
};

} // namespace trtmc
