/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/layoutlm/runtime/tesseract.h"

#include <dlfcn.h>
#include <memory>
#include <stdexcept>

namespace trtmc::layoutlm {
namespace {

// The TSV renderer of the tesseract command line writes this header line before
// the rows that TessBaseAPIGetTsvText returns.
constexpr const char* kTsvHeader = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_"
                                   "num\tleft\ttop\twidth\theight\tconf\ttext\n";

} // namespace

// Declarations of the Tesseract 5 C API (tesseract/capi.h) that this family calls.
struct TesseractOcr::Api {
    void* handle{nullptr};
    const char* (*version)(){nullptr};
    void* (*create)(){nullptr};
    void (*destroy)(void*){nullptr};
    int (*init)(void*, const char*, const char*){nullptr};
    int (*set_variable)(void*, const char*, const char*){nullptr};
    void (*set_page_segmentation_mode)(void*, int){nullptr};
    void (*set_image)(void*, const unsigned char*, int, int, int, int){nullptr};
    int (*recognize)(void*, void*){nullptr};
    char* (*tsv_text)(void*, int){nullptr};
    void (*delete_text)(const char*){nullptr};
    void (*end)(void*){nullptr};
};

namespace {

template <typename Function>
void resolve(void* handle, const char* name, Function& function) {
    function = reinterpret_cast<Function>(dlsym(handle, name));
    if (function == nullptr)
        throw std::runtime_error(std::string("libtesseract.so.5 does not export ") + name);
}

} // namespace

TesseractOcr::TesseractOcr(const char* library) {
    auto* api = new Api();
    api->handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (api->handle == nullptr) {
        const char* reason = dlerror();
        delete api;
        throw std::runtime_error(
            std::string("layoutlm reads the document with Tesseract OCR and could not load ") +
            library +
            " (install Tesseract 5 and its eng language data, for example the tesseract-ocr "
            "and tesseract-ocr-eng packages): " +
            (reason == nullptr ? "unknown error" : reason));
    }
    try {
        resolve(api->handle, "TessVersion", api->version);
        resolve(api->handle, "TessBaseAPICreate", api->create);
        resolve(api->handle, "TessBaseAPIDelete", api->destroy);
        resolve(api->handle, "TessBaseAPIInit3", api->init);
        resolve(api->handle, "TessBaseAPISetVariable", api->set_variable);
        resolve(api->handle, "TessBaseAPISetPageSegMode", api->set_page_segmentation_mode);
        resolve(api->handle, "TessBaseAPISetImage", api->set_image);
        resolve(api->handle, "TessBaseAPIRecognize", api->recognize);
        resolve(api->handle, "TessBaseAPIGetTsvText", api->tsv_text);
        resolve(api->handle, "TessDeleteText", api->delete_text);
        resolve(api->handle, "TessBaseAPIEnd", api->end);
    } catch (...) {
        dlclose(api->handle);
        delete api;
        throw;
    }
    api_ = api;
    version_ = api->version();
}

TesseractOcr::~TesseractOcr() {
    if (api_ != nullptr) {
        dlclose(api_->handle);
        delete api_;
    }
}

const TesseractOcr& TesseractOcr::load() {
    // Never destroyed: no unload at process exit while another thread may still recognize.
    static const TesseractOcr* const instance = new TesseractOcr("libtesseract.so.5");
    return *instance;
}

std::string TesseractOcr::tsv(const std::uint8_t* rgb, std::int32_t width, std::int32_t height,
                              const std::string& language,
                              std::int32_t page_segmentation_mode) const {
    if (rgb == nullptr || width <= 0 || height <= 0)
        throw std::invalid_argument("Tesseract needs a nonempty RGB image");
    const auto deleter = [this](void* handle) {
        api_->end(handle);
        api_->destroy(handle);
    };
    // One recognizer per document, as each command-line run starts fresh.
    std::unique_ptr<void, decltype(deleter)> handle(api_->create(), deleter);
    if (!handle)
        throw std::runtime_error("Tesseract could not create a recognizer");
    if (api_->init(handle.get(), nullptr, language.c_str()) != 0)
        throw std::runtime_error("Tesseract could not load the '" + language +
                                 "' language data; set TESSDATA_PREFIX to its tessdata directory");
    // Diagnostics such as the estimated resolution go nowhere instead of stderr.
    api_->set_variable(handle.get(), "debug_file", "/dev/null");
    api_->set_variable(handle.get(), "tessedit_create_tsv", "1");
    // The C API starts in single-block mode; the command line defaults to automatic layout.
    api_->set_page_segmentation_mode(handle.get(), page_segmentation_mode);
    api_->set_image(handle.get(), rgb, width, height, 3, 3 * width);
    if (api_->recognize(handle.get(), nullptr) != 0)
        throw std::runtime_error("Tesseract failed to recognize the document image");
    char* rows = api_->tsv_text(handle.get(), 0);
    if (rows == nullptr)
        throw std::runtime_error("Tesseract returned no TSV text");
    std::string result(kTsvHeader);
    result += rows;
    api_->delete_text(rows);
    return result;
}

} // namespace trtmc::layoutlm
