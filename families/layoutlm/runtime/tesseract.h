/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>

namespace trtmc::layoutlm {

// Tesseract OCR through its C API. The library is user-installed and loaded on
// first use (libtesseract.so.5); the family does not link or ship it. A missing
// library or language file is an error, never a fallback.
class TesseractOcr {
  public:
    // libtesseract.so.5, loaded on first use and kept for the life of the process.
    static const TesseractOcr& load();
    // The named library, unloaded with this object; the CPU test names a
    // missing one to check the error.
    explicit TesseractOcr(const char* library);
    ~TesseractOcr();
    TesseractOcr(const TesseractOcr&) = delete;
    TesseractOcr& operator=(const TesseractOcr&) = delete;

    // The TSV the tesseract command line writes for
    // `tesseract IMAGE OUT -c tessedit_create_tsv=1` (what pytesseract runs),
    // header line included, for 8-bit RGB pixels in row-major HWC order.
    std::string tsv(const std::uint8_t* rgb, std::int32_t width, std::int32_t height,
                    const std::string& language, std::int32_t page_segmentation_mode) const;

    const std::string& version() const noexcept { return version_; }

  private:
    struct Api;
    const Api* api_{nullptr};
    std::string version_;
};

} // namespace trtmc::layoutlm
