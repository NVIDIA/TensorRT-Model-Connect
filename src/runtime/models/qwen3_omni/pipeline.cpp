#include "runtime/models/qwen3_omni/pipeline.h"

#include "runtime/models/qwen3_omni/omni_audio_plan.h"
#include "trtmc/tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace trtmc {

// ─── OmniPipeline (TrtModule-based) ───

OmniPipeline::OmniPipeline(std::unique_ptr<TrtModule> thinker,
                           std::unique_ptr<Qwen3OmniInferenceState> thinker_state,
                           std::unique_ptr<TrtModule> talker,
                           std::unique_ptr<Qwen3OmniInferenceState> talker_state,
                           std::unique_ptr<TrtModule> code2wav, OmniConfig config,
                           cudaStream_t stream, std::shared_ptr<ITokenizer> tokenizer,
                           std::string model_id_str)
    : thinker_(std::move(thinker)), thinker_state_(std::move(thinker_state)),
      talker_(std::move(talker)), talker_state_(std::move(talker_state)),
      code2wav_(std::move(code2wav)), config_(std::make_unique<OmniConfig>(std::move(config))),
      stream_(stream), tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id_str)) {
    if (!thinker_ || !thinker_->ok())
        throw std::runtime_error("OmniPipeline: invalid thinker module");
    if (!thinker_state_ || !thinker_state_->ok())
        throw std::runtime_error("OmniPipeline: invalid thinker cache");
}

OmniPipeline::~OmniPipeline() = default;

void OmniPipeline::run_thinker_step(int32_t token_id, std::vector<float>& logits,
                                    std::vector<float>* hidden_state) {
    Tensor token_tensor;
    token_tensor.data = &token_id;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;

    TensorMap inputs;
    inputs["token_id"] = token_tensor;
    thinker_state_->prepare_step(inputs);

    TensorMap outputs = thinker_->forward(inputs);

    auto it = outputs.find("logits");
    if (it == outputs.end())
        throw std::runtime_error("OmniPipeline thinker: no 'logits' output");

    const auto& lt = it->second;
    auto n = lt.numel();
    logits.resize(static_cast<std::size_t>(n));
    std::memcpy(logits.data(), lt.data, n * sizeof(float));

    if (hidden_state != nullptr) {
        auto hs_it = outputs.find("hidden_state");
        if (hs_it == outputs.end())
            throw std::runtime_error("OmniPipeline thinker: no 'hidden_state' output");

        const auto& ht = hs_it->second;
        auto hn = ht.numel();
        hidden_state->resize(static_cast<std::size_t>(hn));
        std::memcpy(hidden_state->data(), ht.data, hn * sizeof(float));
    }

    thinker_state_->advance();
}

void OmniPipeline::run_talker_embed_step(const float* embed_ptr, int32_t embed_size,
                                         std::vector<float>& logits) {
    float use_input_embed = 1.0F;

    std::vector<float> embed_buf(embed_ptr, embed_ptr + embed_size);

    Tensor token_tensor;
    int32_t dummy_token = 0;
    token_tensor.data = &dummy_token;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;

    Tensor embed_tensor;
    embed_tensor.data = embed_buf.data();
    embed_tensor.shape = {static_cast<int64_t>(embed_size)};
    embed_tensor.dtype = DType::kFloat32;

    Tensor use_embed_tensor;
    use_embed_tensor.data = &use_input_embed;
    use_embed_tensor.shape = {1};
    use_embed_tensor.dtype = DType::kFloat32;

    TensorMap inputs;
    inputs["token_id"] = token_tensor;
    inputs["input_embed"] = embed_tensor;
    inputs["use_input_embed"] = use_embed_tensor;
    talker_state_->prepare_step(inputs);

    TensorMap outputs = talker_->forward(inputs);

    auto it = outputs.find("logits");
    if (it == outputs.end())
        throw std::runtime_error("OmniPipeline talker: no 'logits' output");

    const auto& lt = it->second;
    auto n = lt.numel();
    logits.resize(static_cast<std::size_t>(n));
    std::memcpy(logits.data(), lt.data, n * sizeof(float));

    talker_state_->advance();
}

static int32_t omni_argmax(const std::vector<float>& logits) {
    if (logits.empty())
        return 0;
    return static_cast<int32_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

std::vector<int32_t> OmniPipeline::run_thinker(const std::vector<int32_t>& input_ids,
                                               int32_t max_tokens,
                                               std::vector<float>& hidden_states_out) {
    thinker_state_->reset();
    thinker_state_->bind_to(*thinker_);
    hidden_states_out.clear();

    std::vector<float> logits;

    for (std::size_t i = 0; i + 1 < input_ids.size(); ++i)
        run_thinker_step(input_ids[i], logits);

    if (!input_ids.empty())
        run_thinker_step(input_ids.back(), logits);

    std::vector<int32_t> output_ids;
    output_ids.reserve(static_cast<std::size_t>(max_tokens));

    for (int32_t step = 0; step < max_tokens; ++step) {
        if (logits.empty())
            break;
        int32_t token = omni_argmax(logits);
        if (token == 0)
            break;
        output_ids.push_back(token);
        std::vector<float> hidden_state;
        run_thinker_step(token, logits, &hidden_state);
        hidden_states_out.insert(hidden_states_out.end(), hidden_state.begin(), hidden_state.end());
    }

    std::cerr << "[trtmc] Omni Thinker: generated " << output_ids.size() << " text tokens"
              << std::endl;
    return output_ids;
}

std::vector<int32_t> OmniPipeline::run_talker(const std::vector<float>& hidden_states,
                                              int32_t num_tokens) {
    if (!talker_ || !talker_state_) {
        std::cerr << "[trtmc] Omni: no Talker engine" << std::endl;
        return {};
    }

    talker_state_->reset();
    talker_state_->bind_to(*talker_);

    const int32_t n_codebooks = config_->talker_n_codebooks;
    const int32_t codebook_size = config_->talker_codebook_size;
    const int32_t talker_hidden = config_->talker_hidden_size;

    const OmniTalkerDecodePlan decode_plan =
        make_omni_talker_decode_plan(n_codebooks, codebook_size, num_tokens);

    std::vector<int32_t> all_codes;
    all_codes.reserve(static_cast<std::size_t>(num_tokens) * n_codebooks);

    std::vector<float> logits;
    for (int32_t t = 0; t < num_tokens; ++t) {
        const float* ep = hidden_states.data() + static_cast<std::size_t>(t) * talker_hidden;
        run_talker_embed_step(ep, talker_hidden, logits);
        append_omni_talker_codes_from_logits(logits, decode_plan, all_codes);
    }

    std::cerr << "[trtmc] Omni Talker: generated " << all_codes.size() << " codec tokens ("
              << num_tokens << " frames x " << n_codebooks << " codebooks)" << std::endl;
    return all_codes;
}

std::vector<float> OmniPipeline::run_code2wav(const std::vector<int32_t>& codec_tokens,
                                              int32_t n_codebooks, int32_t n_frames) {
    if (!code2wav_) {
        std::cerr << "[trtmc] Omni: no Code2Wav engine, generating simple waveform" << std::endl;
        const int32_t samples_per_frame = config_->sample_rate / 75;
        const int32_t total_samples = n_frames * samples_per_frame;
        std::vector<float> waveform(static_cast<std::size_t>(total_samples), 0.0F);
        for (int32_t f = 0; f < n_frames; ++f) {
            const float freq = 200.0F + static_cast<float>(codec_tokens[f * n_codebooks]) * 800.0F /
                                            static_cast<float>(config_->talker_codebook_size);
            const float amp = 0.3F;
            for (int32_t s = 0; s < samples_per_frame; ++s) {
                const auto idx = static_cast<std::size_t>(f) * samples_per_frame + s;
                const float t = static_cast<float>(s) / static_cast<float>(config_->sample_rate);
                waveform[idx] = amp * std::sin(2.0F * 3.14159265F * freq * t);
            }
        }
        return waveform;
    }

    const int32_t max_frames = config_->code2wav_max_frames;
    const int32_t actual_frames = std::min(n_frames, max_frames);
    const int32_t upsample = config_->code2wav_upsample_factor;

    std::vector<int32_t> input_codes =
        build_omni_code2wav_input_codes(codec_tokens, n_codebooks, max_frames, actual_frames);

    Tensor codes_tensor;
    codes_tensor.data = input_codes.data();
    codes_tensor.shape = {static_cast<int64_t>(n_codebooks), static_cast<int64_t>(max_frames)};
    codes_tensor.dtype = DType::kInt32;

    TensorMap inputs;
    inputs["codec_tokens"] = codes_tensor;

    TensorMap outputs = code2wav_->forward(inputs);

    auto it = outputs.find("waveform");
    if (it == outputs.end()) {
        std::cerr << "[trtmc] Omni Code2Wav: no 'waveform' output" << std::endl;
        return {};
    }

    const auto& wt = it->second;
    const auto total_out = wt.numel();
    const auto trimmed =
        static_cast<std::size_t>(actual_frames) * static_cast<std::size_t>(upsample);
    const auto copy_n = std::min(total_out, trimmed);

    std::vector<float> waveform(static_cast<std::size_t>(copy_n));
    std::memcpy(waveform.data(), wt.data, copy_n * sizeof(float));

    std::cerr << "[trtmc] Omni Code2Wav: " << actual_frames << " frames -> " << waveform.size()
              << " samples" << std::endl;
    return waveform;
}

AudioResult OmniPipeline::generate_audio(const std::string& prompt, const GenerateConfig& cfg) {
    std::vector<int32_t> input_ids;
    if (tokenizer_)
        input_ids = tokenizer_->encode(prompt);

    int32_t max_tokens = cfg.max_new_tokens > 0 ? cfg.max_new_tokens : 768;

    AudioResult result;
    result.sample_rate = config_->sample_rate;

    std::cerr << "[trtmc] Omni: starting pipeline with " << input_ids.size() << " input tokens"
              << std::endl;

    std::vector<float> hidden_states;
    auto text_tokens = run_thinker(input_ids, max_tokens, hidden_states);
    if (text_tokens.empty()) {
        std::cerr << "[trtmc] Omni: Thinker produced no tokens" << std::endl;
        return result;
    }

    const OmniTalkerPlan talker_plan =
        make_omni_talker_plan(text_tokens.size(), hidden_states.size(), talker_ != nullptr);
    if (talker_plan.should_run_talker) {
        auto codec_tokens = run_talker(hidden_states, talker_plan.num_tokens);

        const OmniCodecPlan codec_plan = make_omni_codec_plan(*config_, codec_tokens.size());
        if (codec_plan.should_run_codec) {
            auto waveform = run_code2wav(codec_tokens, codec_plan.n_codebooks, codec_plan.n_frames);
            if (!waveform.empty()) {
                result.samples = std::move(waveform);
                result.num_samples = static_cast<int32_t>(result.samples.size());
            }
        }
    }

    std::cerr << "[trtmc] Omni: generated " << result.num_samples << " samples ("
              << (result.num_samples > 0
                      ? static_cast<float>(result.num_samples) / result.sample_rate
                      : 0.0F)
              << "s @ " << result.sample_rate << " Hz)" << std::endl;

    return result;
}

} // namespace trtmc
