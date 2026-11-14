/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/parakeet_tdt/tests/cpp/fake_module.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <iostream>
#include <stdexcept>

class FakeBackend final : public IBackend {
  public:
    int calls{0};
    int fail_at{0};
    std::unique_ptr<ITrtModule> create_module(const void* data, size_t size,
                                              const ModuleCreateOptions&) override {
        const std::string payload(static_cast<const char*>(data), size);
        const char* expected[] = {"encoder", "predictor", "joint"};
        if (calls >= 3 || payload != expected[calls])
            throw std::runtime_error("incorrect engine section order or content");
        ++calls;
        return calls == fail_at ? nullptr : std::make_unique<FakeModule>(calls - 1);
    }
    std::unique_ptr<ITrtModule>
    create_module_prebound(const void*, size_t, const ModuleCreateOptions&,
                           const std::vector<ModuleExternalBinding>&) override {
        throw std::runtime_error("unexpected prebound call");
    }
    BackendDualProfileModules create_dual_profile_modules(const void*, size_t,
                                                          const ModuleCreateOptions&) override {
        throw std::runtime_error("unexpected dual profile call");
    }
    const char* name() const override { return "trt"; }
};

int main(int argc, char** argv) {
    if (argc != 5)
        return 1;
    FakeBackend backend;
    const std::string mode = argv[4];
    backend.fail_at = mode == "fail-engine"      ? 1
                      : mode == "fail-predictor" ? 2
                      : mode == "fail-joint"     ? 3
                                                 : 0;
    const auto kv = std::string(argv[4]) == "kv" ? 1U : 0U;
    const std::string expected_error = argv[2];
    try {
        BundleReader reader(argv[1]);
        std::unique_ptr<ITask> task(trtmc_create_family({reader, backend, kv}));
        if (!expected_error.empty())
            throw std::logic_error("factory accepted invalid bundle");
        auto* model = dynamic_cast<internal::IModel*>(task.get());
        if (!model || model->task_bindings().size() != 1 ||
            model->task_bindings()[0].key.id != "speech_transcription")
            return 2;
    } catch (const std::exception& error) {
        if (expected_error.empty() ||
            std::string(error.what()).find(expected_error) == std::string::npos) {
            std::cerr << error.what() << '\n';
            return 3;
        }
    }
    if (backend.calls != std::stoi(argv[3]))
        return 4;
    if (FakeModule::live_instances != 0)
        return 5;
    std::cout << "factory bundle contract passed\n";
}
