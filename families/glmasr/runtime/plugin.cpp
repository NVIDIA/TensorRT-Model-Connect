#include "families/glmasr/runtime/glmasr_config.h"
#include "families/glmasr/runtime/pipeline.h"
#include "families/glmasr/runtime/plugin_helpers.h"
#include "trtmc/runtime/family_factory.h"

#include <nlohmann/json.hpp>
#include <stdexcept>

namespace {
std::vector<char> required(const trtmc::BundleReader& b, const char* name) {
    const auto* s = b.find_section(name);
    if (!s || !s->length)
        throw std::runtime_error(std::string("missing bundle section: ") + name);
    return b.read_section(name);
}
} // namespace

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    const auto runtime_data = required(context.reader, "runtime.json");
    const auto json = nlohmann::json::parse(runtime_data.begin(), runtime_data.end());
    ModuleCreateOptions options;
    const auto decoder_plan = required(context.reader, "engine.plan");
    auto decoder =
        load_trt_module_from_plan(&context.backend, &decoder_plan, "engine.plan", options);
    const auto encoder_plan = required(context.reader, "encoder.plan");
    auto encoder =
        load_trt_module_from_plan(&context.backend, &encoder_plan, "encoder.plan", options);
    GlmAsrConfig config;
    config.max_cache_length = json.value("max_cache_length", 384);
    config.vocab_size = json.value("vocab_size", config.vocab_size);
    config.audio_embedding_size = json.value("hidden_size", config.audio_embedding_size);
    config.mel_num_bins = json.value("mel_num_bins", config.mel_num_bins);
    config.mel_n_fft = json.value("mel_n_fft", config.mel_n_fft);
    config.mel_hop_length = json.value("mel_hop_length", config.mel_hop_length);
    config.mel_sampling_rate = json.value("mel_sampling_rate", config.mel_sampling_rate);
    config.mel_chunk_length = json.value("mel_chunk_length", config.mel_chunk_length);
    config.eos_token_id = json.value("eot_token_id", config.eos_token_id);
    config.transcription_prompt = json.value("transcription_prompt", config.transcription_prompt);
    const auto cache_shape = decoder.module->tensor_shape("cache_k_0");
    const auto kv_dim = cache_shape.empty() ? 0 : static_cast<int32_t>(cache_shape.back());
    auto state = std::make_unique<GlmAsrKvCache>(
        json.value("num_layers", 1), config.max_cache_length, kv_dim, decoder.module->stream(),
        decoder.module->tensor_dtype("cache_k_0"));
    auto mel = load_mel_filterbank(context.reader);
    auto tokenizer = create_tokenizer_from_bundle(context.reader);
    return new GlmAsrPipeline(std::move(encoder.module), std::move(decoder.module),
                              std::move(state), std::move(config), std::move(mel),
                              decoder.module->stream(), std::move(tokenizer), "");
}
