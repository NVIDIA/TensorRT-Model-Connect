/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/backend/backend_loader.h"
#include "runtime/platform/dynamic_library.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

static int failures = 0;

static void check(bool cond, const char* name) {
    if (!cond) {
        std::cerr << "FAIL: " << name << std::endl;
        ++failures;
    }
}

int main() {
    const std::string backend_name = "nonexistent_backend_xyz";
    const auto missing_dir = std::filesystem::temp_directory_path() / "trtmc-missing-backends";
    const std::string library_name =
        trtmc::internal::dynamic_library_filename("trtmc_backend_" + backend_name);
    const std::string missing_path = (missing_dir / library_name).string();

    // Loading a nonexistent backend should throw
    bool threw = false;
    try {
        trtmc::BackendLoader::load(backend_name, {missing_dir.string()});
    } catch (const std::runtime_error& e) {
        threw = true;
        std::string msg = e.what();
        check(msg.find(backend_name) != std::string::npos, "error mentions backend name");
        check(msg.find(library_name) != std::string::npos, "error mentions DSO name");
        check(msg.find(missing_path) != std::string::npos,
              "error mentions explicit backend search dir");
        check(msg.find("TRTMC_BACKEND_DIR") == std::string::npos, "error does not mention env var");
        check(msg.find("backend_search_paths") != std::string::npos,
              "error mentions explicit search paths");
    }
    check(threw, "missing backend throws runtime_error");

    std::cerr << (failures == 0 ? "ALL PASSED" : "SOME FAILED") << std::endl;
    return failures;
}
