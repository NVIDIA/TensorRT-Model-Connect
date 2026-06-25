#include "runtime/models/personaplex/pipeline.h"

#include "runtime/models/personaplex/decode_runtime.h"
#include "runtime/models/personaplex/speech_decode_stop_policy.h"
#include "runtime/models/personaplex/speech_delay_cache.h"
#include "runtime/models/personaplex/speech_depth_plan.h"
#include "runtime/models/personaplex/speech_generation_policy.h"
#include "runtime/models/personaplex/speech_mimi_decode_plan.h"
#include "runtime/models/personaplex/speech_runtime_plan.h"
#include "runtime/models/personaplex/speech_temporal_embed_plan.h"
#include "runtime/models/personaplex/speech_waveform_postprocess.h"
#include "runtime/models/personaplex/subprocess_runner.h"
#include "utils/wav_reader.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

namespace trtmc {

// ─── PosixSubprocessRunner (moved from speech_backend.cpp) ───

namespace {

// ---------------------------------------------------------------------------
// Subprocess pipe helpers (extracted from PosixSubprocessRunner::run)
// ---------------------------------------------------------------------------

bool create_subprocess_pipes(int (&stdin_pipe)[2], int (&stdout_pipe)[2], int (&stderr_pipe)[2]) {
    return pipe(stdin_pipe) == 0 && pipe(stdout_pipe) == 0 && pipe(stderr_pipe) == 0;
}

void write_all_to_fd(int fd, const void* data, std::size_t size) {
    const auto* p = static_cast<const char*>(data);
    std::size_t remaining = size;
    while (remaining > 0) {
        auto written = write(fd, p, remaining);
        if (written <= 0)
            break;
        p += written;
        remaining -= static_cast<std::size_t>(written);
    }
}

void read_all_from_fd(int fd, std::vector<char>& out) {
    out.clear();
    char buf[65536];
    for (;;) {
        auto n = read(fd, buf, sizeof(buf));
        if (n <= 0)
            break;
        out.insert(out.end(), buf, buf + n);
    }
}

void read_all_string_from_fd(int fd, std::string& out) {
    out.clear();
    char buf[65536];
    for (;;) {
        auto n = read(fd, buf, sizeof(buf));
        if (n <= 0)
            break;
        out.append(buf, static_cast<std::size_t>(n));
    }
}

void exec_child_process(int (&stdin_pipe)[2], int (&stdout_pipe)[2], int (&stderr_pipe)[2],
                        const std::vector<const char*>& c_argv) {
    dup2(stdin_pipe[0], STDIN_FILENO);
    dup2(stdout_pipe[1], STDOUT_FILENO);
    dup2(stderr_pipe[1], STDERR_FILENO);
    close(stdin_pipe[0]);
    close(stdin_pipe[1]);
    close(stdout_pipe[0]);
    close(stdout_pipe[1]);
    close(stderr_pipe[0]);
    close(stderr_pipe[1]);
    execvp(c_argv[0], const_cast<char* const*>(c_argv.data()));
    _exit(127);
}

class PosixSubprocessRunner final : public ISubprocessRunner {
  public:
    int run(const std::vector<std::string>& argv, const void* input_data, std::size_t input_size,
            std::vector<char>& out_stdout, std::string& out_stderr) override {
        std::vector<const char*> c_argv;
        for (const auto& arg : argv)
            c_argv.push_back(arg.c_str());
        c_argv.push_back(nullptr);

        int stdin_pipe[2] = {-1, -1};
        int stdout_pipe[2] = {-1, -1};
        int stderr_pipe[2] = {-1, -1};

        if (!create_subprocess_pipes(stdin_pipe, stdout_pipe, stderr_pipe)) {
            out_stderr = "pipe() failed";
            return -1;
        }

        pid_t pid = fork();
        if (pid < 0) {
            out_stderr = "fork() failed";
            return -1;
        }

        if (pid == 0)
            exec_child_process(stdin_pipe, stdout_pipe, stderr_pipe, c_argv);

        close(stdin_pipe[0]);
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);

        if (input_data && input_size > 0)
            write_all_to_fd(stdin_pipe[1], input_data, input_size);
        close(stdin_pipe[1]);

        read_all_from_fd(stdout_pipe[0], out_stdout);
        close(stdout_pipe[0]);

        read_all_string_from_fd(stderr_pipe[0], out_stderr);
        close(stderr_pipe[0]);

        int status = 0;
        waitpid(pid, &status, 0);
        return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }
};

} // namespace

std::shared_ptr<ISubprocessRunner> CreateDefaultSubprocessRunner() {
    static std::shared_ptr<ISubprocessRunner> runner = std::make_shared<PosixSubprocessRunner>();
    return runner;
}

// ─── SpeechPipeline (TrtModule-based) ───

SpeechPipeline::SpeechPipeline(std::unique_ptr<TrtModule> mimi_encoder,
                               std::unique_ptr<TrtModule> temporal,
                               std::unique_ptr<PersonaplexInferenceState> temporal_state,
                               std::vector<std::unique_ptr<TrtModule>> depth_engines,
                               std::unique_ptr<PersonaplexInferenceState> depth_state,
                               std::unique_ptr<TrtModule> mimi_decoder, SpeechConfig config,
                               cudaStream_t stream,
                               std::shared_ptr<ISubprocessRunner> subprocess_runner,
                               std::string model_id_str)
    : mimi_encoder_(std::move(mimi_encoder)), temporal_(std::move(temporal)),
      temporal_state_(std::move(temporal_state)), depth_engines_(std::move(depth_engines)),
      depth_state_(std::move(depth_state)), mimi_decoder_(std::move(mimi_decoder)), stream_(stream),
      config_(std::move(config)), subprocess_runner_(std::move(subprocess_runner)),
      model_id_(std::move(model_id_str)) {
    if (!temporal_ || !temporal_->ok())
        throw std::runtime_error("SpeechPipeline: invalid temporal module");
    if (!temporal_state_ || !temporal_state_->ok())
        throw std::runtime_error("SpeechPipeline: invalid temporal cache");

    if (!subprocess_runner_)
        subprocess_runner_ = CreateDefaultSubprocessRunner();

    // Fixed deterministic seed for reproducible audio output.
    // PersonaPlex depth sampling (temperature=0.8, top_k=250) is sensitive
    // to the RNG sequence; a fixed seed ensures identical output across runs
    // and between the old and new pipeline paths.
    rng_state_ = 0x5EEDC0DECAFE1234ULL;
}

SpeechPipeline::~SpeechPipeline() = default;

// ---------------------------------------------------------------------------
// Mimi Encoder: audio waveform -> codec tokens
// ---------------------------------------------------------------------------

namespace {

struct MimiEncoderShapes {
    int32_t engine_input_samples{0};
    int32_t enc_codebooks{0};
    int32_t enc_frames{0};
};

MimiEncoderShapes query_mimi_encoder_shapes(const TrtModule& module) {
    MimiEncoderShapes s;
    for (const auto& info : module.input_info()) {
        if (info.name == "audio_input" && !info.shape.empty())
            s.engine_input_samples = static_cast<int32_t>(info.shape.back());
    }
    for (const auto& info : module.output_info()) {
        if (info.name == "codec_tokens" && info.shape.size() >= 2) {
            s.enc_codebooks = static_cast<int32_t>(info.shape[0]);
            s.enc_frames = static_cast<int32_t>(info.shape[1]);
        }
    }
    return s;
}

std::vector<int32_t> transpose_codec_tokens_to_frame_major(const float* data, int32_t codebooks,
                                                           int32_t frames) {
    const auto output_elems = static_cast<std::size_t>(codebooks) * frames;
    std::vector<int32_t> tokens(output_elems);
    for (int32_t cb = 0; cb < codebooks; ++cb) {
        for (int32_t frame = 0; frame < frames; ++frame) {
            const auto src = static_cast<std::size_t>(cb) * frames + frame;
            const auto dst = static_cast<std::size_t>(frame) * codebooks + cb;
            tokens[dst] = static_cast<int32_t>(std::round(data[src]));
        }
    }
    return tokens;
}

void log_first_n_tokens(const char* label, const std::vector<int32_t>& tokens, int32_t n = 16) {
    std::cerr << label;
    for (int32_t i = 0; i < std::min(n, static_cast<int32_t>(tokens.size())); ++i)
        std::cerr << tokens[static_cast<std::size_t>(i)] << " ";
    std::cerr << std::endl;
}

} // anonymous namespace

std::vector<int32_t> SpeechPipeline::run_mimi_encode(const float* samples, int32_t num_samples) {
    last_encode_frames_ = 0;
    last_encode_codebooks_ = 0;

    if (!mimi_encoder_ || !mimi_encoder_->ok()) {
        std::cerr << "[speech] No Mimi TRT encoder available" << std::endl;
        return {};
    }

    const auto shapes = query_mimi_encoder_shapes(*mimi_encoder_);

    if (num_samples != shapes.engine_input_samples) {
        std::cerr << "[speech] WARNING: input samples " << num_samples << " != engine expects "
                  << shapes.engine_input_samples << ", using engine size" << std::endl;
    }

    std::cerr << "[speech] Mimi encoder TRT: input [1,1," << shapes.engine_input_samples
              << "], output [" << shapes.enc_codebooks << "," << shapes.enc_frames << "]"
              << std::endl;

    // Prepare input: pad or truncate to match engine size.
    const auto input_elems = static_cast<std::size_t>(shapes.engine_input_samples);
    std::vector<float> input_buf(input_elems, 0.0F);
    const auto copy_n = std::min(static_cast<std::size_t>(num_samples), input_elems);
    std::memcpy(input_buf.data(), samples, copy_n * sizeof(float));

    Tensor audio_input_tensor;
    audio_input_tensor.data = input_buf.data();
    audio_input_tensor.shape = {1, 1, static_cast<int64_t>(shapes.engine_input_samples)};
    audio_input_tensor.dtype = DType::kFloat32;

    TensorMap inputs;
    inputs["audio_input"] = audio_input_tensor;

    TensorMap outputs = mimi_encoder_->forward(inputs);

    auto it = outputs.find("codec_tokens");
    if (it == outputs.end()) {
        std::cerr << "[speech] Mimi encoder: no 'codec_tokens' output" << std::endl;
        return {};
    }

    const auto& out_tensor = it->second;
    auto tokens = transpose_codec_tokens_to_frame_major(static_cast<const float*>(out_tensor.data),
                                                        shapes.enc_codebooks, shapes.enc_frames);

    std::cerr << "[speech] Mimi encode (TRT): " << num_samples << " samples -> "
              << shapes.enc_frames << " frames x " << shapes.enc_codebooks << " codebooks"
              << std::endl;

    last_encode_frames_ = shapes.enc_frames;
    last_encode_codebooks_ = shapes.enc_codebooks;

    log_first_n_tokens("[speech] Encoder tokens [0:16]: ", tokens);

    return tokens;
}

// ---------------------------------------------------------------------------
// Temporal step with PersonaplexKvCache: input_embed -> logits (+ hidden_state)
// ---------------------------------------------------------------------------

void SpeechPipeline::run_temporal_embed_step(const float* embed_ptr, int32_t embed_size,
                                             std::vector<float>& logits,
                                             std::vector<float>& hidden_out) {
    temporal_state_->bind_to(*temporal_);

    float use_input_embed = 1.0F;
    int32_t dummy_token = 0;

    // Copy embed to mutable buffer (Tensor requires non-const pointer)
    std::vector<float> embed_buf(embed_ptr, embed_ptr + embed_size);

    Tensor token_tensor;
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
    temporal_state_->prepare_step(inputs);

    TensorMap outputs = temporal_->forward(inputs);

    // Extract logits
    auto logits_it = outputs.find("logits");
    if (logits_it == outputs.end())
        throw std::runtime_error("SpeechPipeline temporal: no 'logits' output");

    const auto& lt = logits_it->second;
    auto n = lt.numel();
    logits.resize(static_cast<std::size_t>(n));
    std::memcpy(logits.data(), lt.data, n * sizeof(float));

    // Extract hidden_state if available
    auto hidden_it = outputs.find("hidden_state");
    if (hidden_it != outputs.end()) {
        const auto& ht = hidden_it->second;
        auto hn = ht.numel();
        hidden_out.resize(static_cast<std::size_t>(hn));
        std::memcpy(hidden_out.data(), ht.data, hn * sizeof(float));
    } else {
        // Fallback: use logits as hidden representation (truncated/padded to hidden_size)
        hidden_out.clear();
    }

    temporal_state_->advance();
}

// ---------------------------------------------------------------------------
// Depth step: generate num_codebooks tokens
// ---------------------------------------------------------------------------

namespace {

int32_t speech_clamp_token(int32_t token, int32_t vocab_size) {
    return clamp_speech_depth_token(token, vocab_size);
}

bool speech_is_sampling_enabled(const SpeechConfig& cfg) {
    return cfg.depth_temperature > 0.0F && cfg.depth_top_k > 0;
}

int32_t speech_select_depth_token_greedy(const std::vector<float>& logits, const SpeechConfig&,
                                         uint64_t&) {
    return personaplex_select_argmax_token(logits);
}

int32_t speech_select_depth_token_sampled(const std::vector<float>& logits, const SpeechConfig& cfg,
                                          uint64_t& rng_state) {
    return personaplex_sample_token_topk(logits, cfg.depth_temperature, cfg.depth_top_k, rng_state);
}

using SpeechDepthTokenSelectFn = int32_t (*)(const std::vector<float>&, const SpeechConfig&,
                                             uint64_t&);

SpeechDepthTokenSelectFn speech_select_depth_token_dispatch(const SpeechConfig& cfg) {
    return speech_is_sampling_enabled(cfg) ? speech_select_depth_token_sampled
                                           : speech_select_depth_token_greedy;
}

int32_t speech_sample_temporal_text_token(const std::vector<float>& logits, int32_t text_pad_id,
                                          const SpeechConfig& cfg, uint64_t& rng_state) {
    if (logits.empty())
        return text_pad_id;
    if (speech_is_sampling_enabled(cfg)) {
        constexpr float kTextTemp = 0.7F;
        constexpr int32_t kTextTopK = 25;
        return personaplex_sample_token_topk(logits, kTextTemp, kTextTopK, rng_state);
    }
    return personaplex_select_argmax_token(logits);
}

void speech_maybe_log_depth_debug(int32_t cb, int32_t depth_call_idx,
                                  const std::vector<float>& logits, int32_t best) {
    if (cb != 0 || depth_call_idx >= 12 || logits.empty())
        return;
    int32_t top1 = -1;
    int32_t top2 = -1;
    float v1 = -1.0e30F;
    float v2 = -1.0e30F;
    for (int32_t i = 0; i < static_cast<int32_t>(logits.size()); ++i) {
        const float v = logits[static_cast<std::size_t>(i)];
        if (v > v1) {
            v2 = v1;
            top2 = top1;
            v1 = v;
            top1 = i;
        } else if (v > v2) {
            v2 = v;
            top2 = i;
        }
    }
    std::cerr << "[speech] DepthDbg frame=" << depth_call_idx << " cb0 top1=" << top1
              << " v1=" << v1 << " top2=" << top2 << " v2=" << v2 << " margin=" << (v1 - v2)
              << " sampled=" << best << std::endl;
}

} // anonymous namespace

std::vector<int32_t> SpeechPipeline::run_depth(const float* temporal_hidden, int32_t hidden_dim,
                                               int32_t text_token,
                                               const int32_t* forced_audio_tokens,
                                               const uint8_t* forced_audio_provided) {
    (void)hidden_dim;

    const auto& cfg = config_;
    const int32_t num_cb = cfg.num_codebooks;
    const int32_t depth_hidden = cfg.depth_hidden_size;

    if (depth_engines_.empty())
        return std::vector<int32_t>(static_cast<std::size_t>(num_cb), 0);

    // Reset shared depth cache for this frame
    depth_state_->reset();

    std::vector<int32_t> codebook_tokens;
    codebook_tokens.reserve(static_cast<std::size_t>(num_cb));
    const int32_t depth_call_idx = depth_debug_call_count_++;

    std::vector<float> logits;
    const auto projection_view = make_depth_projection_view(cfg, temporal_hidden);

    std::vector<float> depth_embed(static_cast<std::size_t>(depth_hidden), 0.0F);
    const auto select_token = speech_select_depth_token_dispatch(cfg);

    int32_t prev_token = 0;

    for (int32_t cb = 0; cb < num_cb; ++cb) {
        const auto cb_idx = static_cast<std::size_t>(cb);
        auto* engine = (cb_idx < depth_engines_.size() && depth_engines_[cb_idx])
                           ? depth_engines_[cb_idx].get()
                           : depth_engines_[0].get();

        build_depth_input_embedding(cfg, projection_view, cb, text_token, prev_token, depth_hidden,
                                    depth_embed);

        // Bind depth cache and run with input_embed
        depth_state_->bind_to(*engine);

        float use_input_embed = 1.0F;
        int32_t dummy_token = 0;

        Tensor token_tensor;
        token_tensor.data = &dummy_token;
        token_tensor.shape = {1};
        token_tensor.dtype = DType::kInt32;

        Tensor embed_tensor;
        embed_tensor.data = depth_embed.data();
        embed_tensor.shape = {static_cast<int64_t>(depth_hidden)};
        embed_tensor.dtype = DType::kFloat32;

        Tensor use_embed_tensor;
        use_embed_tensor.data = &use_input_embed;
        use_embed_tensor.shape = {1};
        use_embed_tensor.dtype = DType::kFloat32;

        TensorMap inputs;
        inputs["token_id"] = token_tensor;
        inputs["input_embed"] = embed_tensor;
        inputs["use_input_embed"] = use_embed_tensor;
        depth_state_->prepare_step(inputs);

        TensorMap outputs = engine->forward(inputs);

        auto logits_it = outputs.find("logits");
        if (logits_it == outputs.end()) {
            std::cerr << "[speech] Depth step cb=" << cb << " failed: no logits" << std::endl;
            break;
        }

        const auto& lt = logits_it->second;
        auto n = lt.numel();
        logits.resize(static_cast<std::size_t>(n));
        std::memcpy(logits.data(), lt.data, n * sizeof(float));

        depth_state_->advance();

        int32_t best = select_token(logits, cfg, rng_state_);
        best = std::max(0, std::min(best, cfg.codebook_size - 1));
        codebook_tokens.push_back(best);
        prev_token =
            resolve_depth_prev_token(cb, best, cfg, forced_audio_tokens, forced_audio_provided);
        speech_maybe_log_depth_debug(cb, depth_call_idx, logits, best);
    }

    // Pad if we stopped early
    while (static_cast<int32_t>(codebook_tokens.size()) < num_cb)
        codebook_tokens.push_back(0);

    return codebook_tokens;
}

// ---------------------------------------------------------------------------
// Mimi Decoder: codec tokens -> waveform
// ---------------------------------------------------------------------------

namespace {

struct MimiDecoderShapes {
    int32_t dec_codebooks{0};
    int32_t dec_frames{0};
    std::vector<int32_t> output_dims;
};

MimiDecoderShapes query_mimi_decoder_shapes(const TrtModule& module) {
    MimiDecoderShapes s;
    for (const auto& info : module.input_info()) {
        if (info.name == "codec_tokens" && info.shape.size() >= 2) {
            s.dec_codebooks = static_cast<int32_t>(info.shape[0]);
            s.dec_frames = static_cast<int32_t>(info.shape[1]);
        }
    }
    for (const auto& info : module.output_info()) {
        if (info.name == "audio_output") {
            s.output_dims.reserve(info.shape.size());
            for (auto d : info.shape)
                s.output_dims.push_back(static_cast<int32_t>(d));
        }
    }
    return s;
}

} // anonymous namespace

std::vector<float> SpeechPipeline::run_mimi_decode(const std::vector<int32_t>& codec_tokens,
                                                   int32_t num_frames) {
    if (num_frames <= 0)
        return {};

    int32_t actual_codebooks = 0;
    if (!codec_tokens.empty())
        actual_codebooks = static_cast<int32_t>(codec_tokens.size()) / num_frames;

    if (!mimi_decoder_ || !mimi_decoder_->ok()) {
        std::cerr << "[speech] No Mimi TRT decoder available" << std::endl;
        return {};
    }

    const auto shapes = query_mimi_decoder_shapes(*mimi_decoder_);
    const auto layout =
        build_mimi_decode_layout(shapes.dec_codebooks, shapes.dec_frames, shapes.output_dims);

    std::cerr << "[speech] Mimi decoder TRT: input [" << layout.dec_codebooks << ","
              << layout.dec_frames << "], output " << layout.total_output_elems << " samples"
              << std::endl;

    auto input_tokens = build_mimi_decoder_input(codec_tokens, num_frames, actual_codebooks,
                                                 layout.dec_frames, layout.dec_codebooks);

    // Debug: print first few input tokens
    std::cerr << "[speech] Decoder input tokens [0:16]: ";
    for (int32_t i = 0; i < std::min(16, static_cast<int32_t>(layout.input_elems)); ++i)
        std::cerr << input_tokens[static_cast<std::size_t>(i)] << " ";
    std::cerr << std::endl;

    Tensor codec_tensor;
    codec_tensor.data = input_tokens.data();
    codec_tensor.shape = {static_cast<int64_t>(shapes.dec_codebooks),
                          static_cast<int64_t>(shapes.dec_frames)};
    codec_tensor.dtype = DType::kFloat32;

    TensorMap inputs;
    inputs["codec_tokens"] = codec_tensor;

    TensorMap outputs = mimi_decoder_->forward(inputs);

    auto it = outputs.find("audio_output");
    if (it == outputs.end()) {
        std::cerr << "[speech] Mimi decoder: no 'audio_output' output" << std::endl;
        return {};
    }

    const auto& out_tensor = it->second;
    const auto total_elems = static_cast<std::size_t>(layout.total_output_elems);
    std::vector<float> waveform(total_elems);
    std::memcpy(waveform.data(), out_tensor.data, total_elems * sizeof(float));

    float rms = 0.0F;
    float mx = 0.0F;
    waveform_stats(waveform, layout.total_output_elems, rms, mx);
    std::cerr << "[speech] Mimi decode (TRT): " << layout.dec_frames << " frames -> "
              << layout.total_output_elems << " samples (RMS=" << rms << ", Max=" << mx << ")"
              << std::endl;
    return waveform;
}

// ---------------------------------------------------------------------------
// Text Prompt Injection
// ---------------------------------------------------------------------------

namespace {

// Resolve text prompt tokens: use pre-tokenized, runtime-tokenize, or empty.
// Returns true if tokens were resolved; false means skip text prompt.
bool resolve_text_prompt_tokens(const SpeechConfig& cfg, ISubprocessRunner& subprocess_runner,
                                std::vector<int32_t>& text_tokens) {
    text_tokens = cfg.text_prompt_ids;
    if (!text_tokens.empty()) {
        std::cerr << "[speech] Injecting pre-tokenized text prompt (" << text_tokens.size()
                  << " tokens)" << std::endl;
        return true;
    }
    if (!cfg.system_prompt.empty() && !cfg.hf_python.empty()) {
        auto tokenization =
            TokenizeSpeechPromptRuntime(cfg.hf_python, cfg.system_prompt, subprocess_runner);
        if (tokenization.rc != 0 || tokenization.tokens.empty()) {
            std::cerr << "[speech] Text prompt tokenization failed (rc=" << tokenization.rc
                      << "): " << tokenization.stderr_data << std::endl;
            return false;
        }
        text_tokens = std::move(tokenization.tokens);
        std::cerr << "[speech] Injecting runtime-tokenized text prompt: \"" << cfg.system_prompt
                  << "\" (" << text_tokens.size() << " tokens)" << std::endl;
        return true;
    }
    return false;
}

void compute_text_prompt_frame_embed(const SpeechConfig& cfg, int32_t text_token_id, int32_t hidden,
                                     float* summed_embed) {
    std::fill(summed_embed, summed_embed + hidden, 0.0F);
    const int32_t text_tok = speech_clamp_token(text_token_id, cfg.temporal_text_vocab);
    const auto text_offset = static_cast<std::size_t>(text_tok) * hidden;
    add_speech_embedding_row(cfg.temporal_text_embedding, text_offset, hidden, summed_embed);

    const int32_t audio_vocab = cfg.audio_vocab_size;
    const int32_t bos = speech_clamp_token(cfg.codebook_size, audio_vocab);
    const auto emb_stride_cb = static_cast<std::size_t>(audio_vocab) * hidden;
    for (int32_t cb = 0; cb < cfg.num_codebooks; ++cb) {
        const auto emb_offset =
            static_cast<std::size_t>(cb) * emb_stride_cb + static_cast<std::size_t>(bos) * hidden;
        add_speech_embedding_row(cfg.audio_embeddings, emb_offset, hidden, summed_embed);
    }
}

} // anonymous namespace

void SpeechPipeline::run_text_prompt() {
    const auto& cfg = config_;
    const int32_t hidden = cfg.temporal_hidden_size;

    if (cfg.temporal_text_embedding.empty() || cfg.temporal_text_vocab <= 0 ||
        !temporal_->has_input("input_embed")) {
        std::cerr << "[speech] Cannot inject text prompt: missing embeddings" << std::endl;
        return;
    }

    std::vector<int32_t> text_tokens;
    if (!resolve_text_prompt_tokens(cfg, *subprocess_runner_, text_tokens))
        return;

    std::vector<float> summed_embed(static_cast<std::size_t>(hidden));
    std::vector<float> logits;
    std::vector<float> hidden_out;

    for (std::size_t t = 0; t < text_tokens.size(); ++t) {
        compute_text_prompt_frame_embed(cfg, text_tokens[t], hidden, summed_embed.data());
        run_temporal_embed_step(summed_embed.data(), hidden, logits, hidden_out);
    }

    std::cerr << "[speech] Text prompt injection complete (" << text_tokens.size()
              << " temporal steps)" << std::endl;
}

// ---------------------------------------------------------------------------
// Interleaved generation helpers (free functions, SpeechPipeline-specific)
// ---------------------------------------------------------------------------

namespace {

void speech_log_depth_mode(const SpeechConfig& cfg) {
    if (speech_is_sampling_enabled(cfg)) {
        std::cerr << "[speech] Depth sampling: temperature=" << cfg.depth_temperature
                  << " top_k=" << cfg.depth_top_k << std::endl;
        return;
    }
    std::cerr << "[speech] Depth decoding: greedy (argmax)" << std::endl;
}

void speech_log_stop_configuration(const SpeechConfig& cfg, int32_t extra_tail) {
    if (cfg.text_eos_token_id >= 0) {
        std::cerr << "[speech] Text EOS early-stop enabled: eos_token_id=" << cfg.text_eos_token_id
                  << " (min_streak=" << kSpeechMinConsecutiveTextEos << ")" << std::endl;
    }
    if (extra_tail <= 0)
        return;
    std::cerr << "[speech] Text PAD fallback stop enabled after input "
                 "(pad_id="
              << cfg.text_padding_id << ", min_streak=" << kSpeechMinConsecutiveTextPadAfterInput
              << ")" << std::endl;
    std::cerr << "[speech] Post-input continuation cap: " << kSpeechMaxContinuationFramesAfterInput
              << " frames" << std::endl;
}

void speech_maybe_log_stop_decision(SpeechDecodeStopReason reason,
                                    const SpeechDecodeStopState& stop_state, int32_t offset) {
    switch (reason) {
    case SpeechDecodeStopReason::kNone:
        return;
    case SpeechDecodeStopReason::kTextEos:
        std::cerr << "[speech] Text EOS detected at offset " << offset
                  << " (streak=" << stop_state.text_eos_streak
                  << "), draining delayed frames until offset "
                  << stop_state.stop_collect_until_offset << std::endl;
        return;
    case SpeechDecodeStopReason::kTextPadFallback:
        std::cerr << "[speech] Text PAD fallback stop at offset " << offset
                  << " (streak=" << stop_state.text_pad_streak
                  << "), draining delayed frames until offset "
                  << stop_state.stop_collect_until_offset << std::endl;
        return;
    case SpeechDecodeStopReason::kContinuationCap:
        std::cerr << "[speech] Continuation cap reached at offset " << offset
                  << ", draining delayed frames until offset "
                  << stop_state.stop_collect_until_offset << std::endl;
        return;
    }
}

void speech_maybe_log_interleaved_debug(int32_t offset, int32_t hidden,
                                        const std::vector<float>& frame_hidden, int32_t text_input,
                                        int32_t sampled_text_token,
                                        const std::vector<int32_t>& frame_codes) {
    if (offset <= 0 || offset > 5)
        return;
    float l2 = 0.0F;
    for (int32_t d = 0; d < hidden; ++d)
        l2 += frame_hidden[static_cast<std::size_t>(d)] * frame_hidden[static_cast<std::size_t>(d)];
    l2 = std::sqrt(l2);
    std::cerr << "[speech] Offset " << offset << " hidden L2=" << l2 << " text_in=" << text_input
              << " text_out=" << sampled_text_token << " depth:";
    for (int32_t cb = 0; cb < std::min(4, static_cast<int32_t>(frame_codes.size())); ++cb)
        std::cerr << " " << frame_codes[static_cast<std::size_t>(cb)];
    std::cerr << "..." << std::endl;
}

void speech_log_output_frames_debug(const std::vector<int32_t>& output_codes,
                                    int32_t generated_frames, int32_t mimi_cb) {
    if (output_codes.empty())
        return;
    for (int32_t frame = 0; frame < generated_frames; ++frame) {
        std::cerr << "[speech] Output frame " << frame << ":";
        for (int32_t cb = 0; cb < mimi_cb; ++cb) {
            const auto idx = static_cast<std::size_t>(frame) * mimi_cb + cb;
            if (idx < output_codes.size())
                std::cerr << " " << output_codes[idx];
        }
        std::cerr << std::endl;
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// speak(): Full speech-to-speech pipeline
// ---------------------------------------------------------------------------

bool SpeechPipeline::speak_validate_dual_stream() const {
    const bool has_audio_emb = !config_.audio_embeddings.empty() && config_.audio_vocab_size > 0;
    const bool has_input_embed = temporal_->has_input("input_embed");
    (void)temporal_->has_output("hidden_state");
    if (!has_audio_emb || !has_input_embed) {
        std::cerr << "[speech] ERROR: dual-stream requires audio_embeddings "
                     "and input_embed support"
                  << std::endl;
        return false;
    }
    return true;
}

void SpeechPipeline::speak_run_generation_loop(const SpeechGenerationSettings& settings,
                                               const SpeechOutputPlan& plan,
                                               DelayCacheState& delay_state,
                                               const std::vector<int32_t>& codec_tokens,
                                               std::vector<int32_t>& output_codes,
                                               int32_t& frames_collected) {
    const int32_t hidden = settings.hidden;
    SpeechDecodeStopState stop_state;
    speech_log_stop_configuration(config_, plan.extra_tail);

    std::vector<float> summed_embed(static_cast<std::size_t>(hidden));
    std::vector<float> frame_hidden(static_cast<std::size_t>(hidden));
    std::vector<float> logits;
    std::vector<float> hidden_out;
    std::vector<int32_t> moshi_input(static_cast<std::size_t>(settings.stream_cb));
    std::vector<int32_t> user_input(static_cast<std::size_t>(settings.stream_cb));
    std::vector<int32_t> target_audio_tokens(static_cast<std::size_t>(settings.num_cb));
    std::vector<uint8_t> target_audio_provided(static_cast<std::size_t>(settings.num_cb));

    frames_collected = 0;
    for (int32_t offset = 0; offset < plan.total_iters && frames_collected < plan.output_frames;
         ++offset) {
        write_user_tokens_to_delay_cache(delay_state, codec_tokens, offset, settings.stream_cb,
                                         settings.num_frames, settings.encode_codebooks,
                                         settings.audio_bos);
        fill_initial_delay_tokens(delay_state, offset, settings.text_bos, settings.audio_bos);
        if (offset == 0) {
            seed_delay_offset_zero(delay_state, settings.text_bos, settings.audio_bos);
            continue;
        }

        const int32_t model_input_pos = offset - 1;
        const int32_t target_pos = offset;

        int32_t text_input = settings.text_pad_id;
        read_model_inputs_from_delay_cache(delay_state, model_input_pos, settings.stream_cb,
                                           text_input, moshi_input, user_input);
        compute_dual_stream_summed_embed(config_, settings.hidden, settings.stream_cb,
                                         moshi_input.data(), user_input.data(), text_input,
                                         summed_embed.data());

        run_temporal_embed_step(summed_embed.data(), settings.hidden, logits, hidden_out);

        if (!hidden_out.empty()) {
            frame_hidden.resize(static_cast<std::size_t>(settings.hidden));
            const auto copy_sz =
                std::min(hidden_out.size(), static_cast<std::size_t>(settings.hidden));
            std::memcpy(frame_hidden.data(), hidden_out.data(), copy_sz * sizeof(float));
        } else {
            fill_hidden_from_logits(frame_hidden, logits, settings.hidden);
        }

        const int32_t sampled_text_token =
            speech_sample_temporal_text_token(logits, settings.text_pad_id, config_, rng_state_);
        const auto text_target_idx = delay_cache_index(delay_state, 0, target_pos);
        const bool text_provided = delay_state.provided[text_target_idx] != 0;
        const int32_t next_text_token =
            text_provided ? delay_state.cache[text_target_idx] : sampled_text_token;

        build_target_audio_arrays(delay_state, target_pos, settings.num_cb, settings.audio_bos,
                                  target_audio_tokens, target_audio_provided);
        auto frame_codes = run_depth(frame_hidden.data(), settings.hidden, next_text_token,
                                     target_audio_tokens.data(), target_audio_provided.data());

        clear_provided_flags_at_pos(delay_state, model_input_pos);
        write_generated_tokens_to_delay_cache(delay_state, target_pos, sampled_text_token,
                                              text_provided, frame_codes, settings.num_cb);
        if (collect_output_codes_from_delay_cache(delay_state, offset, delay_state.max_delay,
                                                  settings.mimi_cb, output_codes)) {
            ++frames_collected;
        }

        SpeechDecodeStopInput stop_input;
        stop_input.text_eos_token_id = config_.text_eos_token_id;
        stop_input.text_padding_id = config_.text_padding_id;
        stop_input.effective_frames = plan.effective_frames;
        stop_input.extra_tail = plan.extra_tail;
        stop_input.target_pos = target_pos;
        stop_input.sampled_text_token = sampled_text_token;
        stop_input.offset = offset;
        stop_input.max_delay = delay_state.max_delay;
        stop_input.text_provided = text_provided;
        const auto stop_decision = UpdateSpeechDecodeStopState(stop_state, stop_input);
        stop_state = stop_decision.state;
        speech_maybe_log_stop_decision(stop_decision.reason, stop_state, offset);
        speech_maybe_log_interleaved_debug(offset, settings.hidden, frame_hidden, text_input,
                                           sampled_text_token, frame_codes);
        if (stop_decision.should_break)
            break;
    }
}

void SpeechPipeline::speak_postprocess_waveform(std::vector<float>& waveform,
                                                int32_t generated_frames) const {
    const auto trim_result = trim_speech_waveform_to_generated_frames(
        config_.sample_rate, config_.frame_rate, generated_frames, waveform);
    if (trim_result.trimmed) {
        std::cerr << "[speech] Trimmed decoded waveform to " << trim_result.expected_samples
                  << " samples (" << generated_frames << " generated frames)" << std::endl;
    }

    const auto normalize_result = peak_normalize_speech_waveform(waveform);
    if (normalize_result.normalized) {
        std::cerr << "[speech] Peak-normalized: peak=" << normalize_result.peak
                  << " scale=" << normalize_result.scale << std::endl;
    }
}

AudioResult SpeechPipeline::speak(const float* audio_in, int32_t num_samples,
                                  const GenerateConfig& cfg, int32_t input_sample_rate) {
    AudioResult result;
    result.sample_rate = config_.sample_rate;

    depth_debug_call_count_ = 0;

    const int32_t max_output_frames = cfg.max_new_tokens > 0 ? cfg.max_new_tokens : 375;

    // Resample if the input sample rate differs from the model's expected rate.
    const float* samples_ptr = audio_in;
    int32_t samples_count = num_samples;
    std::vector<float> resampled_buf;
    const int32_t target_rate = config_.sample_rate;

    if (input_sample_rate > 0 && target_rate > 0 && input_sample_rate != target_rate) {
        std::cerr << "[speech] Resampling audio from " << input_sample_rate << " Hz to "
                  << target_rate << " Hz" << std::endl;
        resampled_buf = resample_linear(audio_in, num_samples, input_sample_rate, target_rate);
        samples_ptr = resampled_buf.data();
        samples_count = static_cast<int32_t>(resampled_buf.size());
        input_sample_rate = target_rate;
    }

    std::cerr << "[speech] Starting pipeline with " << samples_count << " input samples"
              << std::endl;

    speech_log_depth_mode(config_);

    // Stage 1: Encode input audio via Mimi
    auto codec_tokens = run_mimi_encode(samples_ptr, samples_count);

    const auto encoder_shape = resolve_encoder_shape_without_engine(
        config_, last_encode_codebooks_, last_encode_frames_, codec_tokens.size());
    const int32_t num_frames = encoder_shape.num_frames;

    std::cerr << "[speech] Encoder output: " << codec_tokens.size() << " tokens = " << num_frames
              << " frames x " << encoder_shape.encode_codebooks << " codebooks" << std::endl;

    if (num_frames <= 0) {
        std::cerr << "[speech] Encoder produced no frames" << std::endl;
        return result;
    }

    if (!speak_validate_dual_stream())
        return result;

    temporal_state_->reset();

    if (should_run_text_prompt_injection(config_))
        run_text_prompt();

    const int32_t num_cb = config_.num_codebooks;
    const int32_t hidden = config_.temporal_hidden_size;
    auto delay_state = make_delay_cache_state(config_.delays, num_cb);
    SpeechOutputPlanInput plan_input;
    plan_input.sample_rate = config_.sample_rate;
    plan_input.frame_rate = config_.frame_rate;
    plan_input.num_frames = num_frames;
    plan_input.num_input_samples = samples_count;
    plan_input.input_sample_rate = input_sample_rate;
    plan_input.tail_frames = cfg.tail_frames;
    plan_input.max_output_frames = max_output_frames;
    plan_input.max_delay = delay_state.max_delay;
    const auto plan = ComputeSpeechOutputPlan(plan_input);
    const int32_t mimi_cb = config_.mimi_decode_codebooks;
    std::vector<int32_t> output_codes;
    output_codes.reserve(static_cast<std::size_t>(mimi_cb) * plan.output_frames);

    std::cerr << "[speech] Interleaved temporal+depth with delay pattern: " << plan.output_frames
              << " output frames, " << plan.total_iters
              << " total iterations (max_delay=" << delay_state.max_delay
              << ", input_effective=" << plan.effective_frames
              << ", tail_frames=" << plan.extra_tail << ")" << std::endl;

    const SpeechGenerationSettings settings =
        make_speech_generation_settings(config_, hidden, encoder_shape);

    int32_t frames_collected = 0;
    speak_run_generation_loop(settings, plan, delay_state, codec_tokens, output_codes,
                              frames_collected);

    const int32_t generated_frames = frames_collected;
    std::cerr << "[speech] Depth: generated " << generated_frames << " frames x " << num_cb
              << " codebooks (decoding first " << mimi_cb << ")" << std::endl;
    speech_log_output_frames_debug(output_codes, generated_frames, mimi_cb);

    // Stage 4: Decode output tokens to audio via Mimi decoder
    auto waveform = run_mimi_decode(output_codes, generated_frames);
    speak_postprocess_waveform(waveform, generated_frames);

    result.samples = std::move(waveform);
    result.num_samples = static_cast<int32_t>(result.samples.size());
    std::cerr << "[speech] Generated " << result.num_samples << " samples ("
              << static_cast<float>(result.num_samples) / result.sample_rate << "s @ "
              << result.sample_rate << " Hz)" << std::endl;
    return result;
}

} // namespace trtmc
