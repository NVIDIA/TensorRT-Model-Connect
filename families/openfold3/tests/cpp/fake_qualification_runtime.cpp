/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/openfold3/structure_prediction.h"
#include "trtmc/runtime/family_loader.h"

#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>

extern "C" int openfold3_test_build_id();

namespace {

class Prediction final : public trtmc::openfold3::IStructurePrediction {
  public:
    trtmc::openfold3::StructurePredictionResult predict_structure(const std::string&) override {
        return {"data_test\n", "{\"build_id\":" + std::to_string(TEST_BUILD_ID) + "}"};
    }
};

} // namespace

namespace trtmc {

std::unique_ptr<ITask> load_task(const std::string&, const std::string& runtime_root, std::uint64_t,
                                 const std::string&, bool) {
    const auto backend = std::filesystem::path(runtime_root) / "libtrtmc_backend_trt.so";
    void* library = dlopen(backend.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr)
        throw std::runtime_error(dlerror());
    const auto backend_build =
        reinterpret_cast<int (*)()>(dlsym(library, "openfold3_test_build_id"));
    const bool matches = backend_build != nullptr && backend_build() == TEST_BUILD_ID &&
                         openfold3_test_build_id() == TEST_BUILD_ID;
    dlclose(library);
    if (!matches)
        throw std::runtime_error("qualification runtime/core/backend product build mismatch");
    return std::make_unique<Prediction>();
}

} // namespace trtmc
