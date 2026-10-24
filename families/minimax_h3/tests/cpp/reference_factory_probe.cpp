/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/internal/model.h"
#include "trtmc/internal/video.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <array>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace {
using namespace trtmc;
using namespace trtmc::internal;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

// A boundary guard, not an engine or inference double. Every attempted
// deserialization fails after checking the actual lazy file-loader payload.
class NoInferenceBackend final : public IBackend {
  public:
    int calls{0};
    const char* name() const override { return "trt"; }
    std::unique_ptr<ITrtModule> create_module(const void* data, std::size_t bytes,
                                              const ModuleCreateOptions& options) override {
        ++calls;
        require(data && std::string(static_cast<const char*>(data), bytes) == "fixture:vision.plan",
                "first lazy plan must be the declared vision fixture");
        require(options.stream == nullptr && !options.cuda_graphs, "unexpected module options");
        throw std::runtime_error("intentional native factory stop before deserialization");
    }
    std::unique_ptr<ITrtModule>
    create_module_prebound(const void*, std::size_t, const ModuleCreateOptions&,
                           const std::vector<ModuleExternalBinding>&) override {
        throw std::runtime_error("unexpected prebound deserialization");
    }
    BackendDualProfileModules create_dual_profile_modules(const void*, std::size_t,
                                                          const ModuleCreateOptions&) override {
        throw std::runtime_error("unexpected dual-profile deserialization");
    }
};

template <class Error, class Call>
void rejects(Call call) {
    try {
        call();
    } catch (const Error&) {
        return;
    }
    throw std::runtime_error("expected native rejection did not occur");
}

void validate_model(ITask& task, NoInferenceBackend& backend) {
    require(std::string(task.task()) == "references_text_to_audio_video", "wrong primary task");
    auto* model = dynamic_cast<IModel*>(&task);
    require(model != nullptr, "factory did not return the semantic model contract");
    const auto bindings = model->task_bindings();
    auto* reference = dynamic_cast<IReferencesTextToAudioVideo*>(&task);
    require(reference && bindings.size() == 1 && bindings[0].key.id == reference->kTask &&
                bindings[0].key.major == 1 && bindings[0].key.minor == 0 &&
                bindings[0].implementation == reference,
            "reference task binding pointer/version mismatch across the DSO");
    require(dynamic_cast<IInitialImageTextToAudioVideo*>(&task) == nullptr,
            "image identity reference must not advertise initial-frame semantics");
    require(bindings[0].fields.size() == 1 && bindings[0].fields[0].name == "seed" &&
                bindings[0].fields[0].kind == ConfigKind::I64,
            "unexpected reference runtime configuration surface");
    const auto initial = backend.calls;
    rejects<UnsupportedTask>([&] { (void)reference->run({{}, "prompt"}, {}); });
    const ConfigEntry steps[] = {{"steps", int64_t{4}}};
    rejects<ConfigError>([&] { (void)reference->run({{}, "prompt"}, steps); });
    const VideoReferenceItem audio[] = {AudioView{}};
    rejects<UnsupportedTask>([&] { (void)reference->run({audio, "prompt"}, {}); });
    require(backend.calls == initial, "invalid input reached engine loading");
    const std::array<uint8_t, 3> rgb{127, 64, 32};
    const VideoReferenceItem images[] = {
        ImageView{rgb.data(), rgb.size(), 1, 1, 3, ImageFormat::UInt8}};
    for (int repeat = 0; repeat < 2; ++repeat) {
        try {
            (void)reference->run({images, "walk and wave"}, {});
        } catch (const std::runtime_error& error) {
            require(std::string(error.what()) ==
                        "intentional native factory stop before deserialization",
                    "valid request failed before the guarded lazy plan boundary");
        }
        require(backend.calls == initial + repeat + 1,
                "failed request did not recover native state");
    }
}
} // namespace

int main(int argc, char** argv) {
    void* handle = nullptr;
    try {
        require(argc == 3, "usage: reference_factory_probe <family-dso> <fixture-manifest>");
        const auto* visibility = std::getenv("CUDA_VISIBLE_DEVICES");
        require(visibility && std::string(visibility).empty(),
                "this factory-only proof requires hidden GPUs");
        std::ifstream input(argv[2]);
        const auto manifest = nlohmann::json::parse(input);
        handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (!handle)
            throw std::runtime_error(dlerror());
        const auto factory = reinterpret_cast<CreateFamilyFn>(dlsym(handle, kCreateFamilySymbol));
        require(factory != nullptr, "family factory export is missing");
        nlohmann::json results = nlohmann::json::array();
        for (const auto& item : manifest) {
            NoInferenceBackend backend;
            BundleReader reader(item.at("bundle").get<std::string>());
            bool constructed = false;
            std::string error;
            try {
                std::unique_ptr<ITask> task(factory({reader, backend, 0}));
                require(task != nullptr, "factory returned null");
                constructed = true;
                validate_model(*task, backend);
            } catch (const std::exception& failure) {
                error = failure.what();
            }
            const bool expected = item.at("constructs").get<bool>();
            require(constructed == expected,
                    "factory fixture construction disagrees with manifest");
            require(expected ? error.empty() && backend.calls == 2
                             : !error.empty() && backend.calls == 0,
                    "factory contract or preflight failed");
            results.push_back({{"case", item.at("case")},
                               {"constructed", constructed},
                               {"guarded_lazy_loads", backend.calls},
                               {"expected_rejection", error}});
        }
        dlclose(handle);
        handle = nullptr;
        std::cout << nlohmann::json({{"cases", results},
                                     {"engine_execution_count", 0},
                                     {"factory_contract_only", true},
                                     {"full_ref2va_qualified", false}})
                         .dump()
                  << '\n';
    } catch (const std::exception& error) {
        if (handle)
            dlclose(handle);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
