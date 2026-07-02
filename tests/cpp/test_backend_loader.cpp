/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/backend/backend_loader.h"

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
    // Loading a nonexistent backend should throw
    bool threw = false;
    try {
        trtmc::BackendLoader::load("nonexistent_backend_xyz", {"/tmp/trtmc-missing-backends"});
    } catch (const std::runtime_error& e) {
        threw = true;
        std::string msg = e.what();
        check(msg.find("nonexistent_backend_xyz") != std::string::npos,
              "error mentions backend name");
        check(msg.find("libtrtmc_backend_nonexistent_backend_xyz.so") != std::string::npos,
              "error mentions DSO name");
        check(msg.find("/tmp/trtmc-missing-backends/libtrtmc_backend_nonexistent_backend_xyz.so") !=
                  std::string::npos,
              "error mentions explicit backend search dir");
        check(msg.find("TRTMC_BACKEND_DIR") == std::string::npos, "error does not mention env var");
        check(msg.find("backend_search_paths") != std::string::npos,
              "error mentions explicit search paths");
    }
    check(threw, "missing backend throws runtime_error");

    std::cerr << (failures == 0 ? "ALL PASSED" : "SOME FAILED") << std::endl;
    return failures;
}
