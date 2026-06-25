#pragma once
#include "plugin_helpers.h"
#include "runtime/models/personaplex/speech_config.h"
#include "utils/json_helpers.h"

namespace trtmc {

std::unique_ptr<PersonaplexKvCache> make_coarse_kv_cache(const std::string& json,
                                                         const BaseConfig& base,
                                                         cudaStream_t stream,
                                                         DType cache_dtype = DType::kFloat32);

SpeechConfig build_speech_config_from_bundle(const BundleFile& bundle, const std::string& json,
                                             const BaseConfig& base, const std::string& hf_python);

void infer_speech_vocab_sizes(SpeechConfig& sc, const std::string& json, const BaseConfig& base);

BaseConfig make_depth_engine_config(const std::string& json, const BaseConfig& base);

std::vector<std::unique_ptr<TrtModule>> load_depth_engines(IBackend* backend,
                                                           const BundleFile& bundle,
                                                           const ModuleCreateOptions& options = {});

int32_t compute_kv_dim_kv_heads(const BaseConfig& base, int32_t default_dim);

int32_t safe_embed_dim(const std::vector<float>& data, int32_t divisor);

} // namespace trtmc
