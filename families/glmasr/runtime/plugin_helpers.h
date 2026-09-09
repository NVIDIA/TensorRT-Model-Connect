#pragma once
#include "families/whisper/runtime/tokenizer.h"
#include "trtmc/bundle.h"
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <vector>
namespace trtmc {
struct LoadedModule {
    std::unique_ptr<ITrtModule> module;
};
struct DualProfileModules {
    std::unique_ptr<ITrtModule> prefill;
    std::unique_ptr<ITrtModule> decode;
};
struct MelFilterbank {
    std::vector<float> data;
    std::int32_t n_freq_bins{0};
    std::int32_t n_mel_bins{0};
};
LoadedModule load_trt_module_from_plan(IBackend*, const std::vector<char>*, const char*,
                                       const ModuleCreateOptions& = {});
DualProfileModules load_dual_profile_modules(IBackend*, const std::vector<char>*, const char*,
                                             const ModuleCreateOptions& = {});
std::shared_ptr<ITokenizer> create_tokenizer_from_bundle(const BundleReader&);
MelFilterbank load_mel_filterbank(const BundleReader&);
void load_ffi_kernels_from_bundle(const BundleReader&);
} // namespace trtmc
