/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace trtmc::cli {

// Convert one UTF-16 command-line value to UTF-8. The Windows implementation
// rejects malformed surrogate sequences instead of replacing them.
std::string utf8_from_utf16(std::wstring_view value);

// Own the UTF-8 storage and the mutable pointer array expected by the existing
// platform-neutral CLI parser. argv()[argc()] is always null.
class Utf8CommandLine {
  public:
    Utf8CommandLine(int argc, wchar_t* const* argv);

    int argc() const noexcept { return argc_; }
    char** argv() noexcept { return pointers_.data(); }

  private:
    int argc_{0};
    std::vector<std::string> storage_;
    std::vector<char*> pointers_;
};

} // namespace trtmc::cli
