#!/usr/bin/env python3
"""Staged Canary ASR parity: TRT engine (pure Python) vs NeMo reference.

Pure-Python, no C++ binary. Builds the Canary encoder + decoder TRT engines via
the family plugin, runs them with a self-contained TrtRunner, and compares
against the NeMo reference *stage by stage* so the first point of divergence is
localized:

  Stage 1 (encoder): feed the *same* NeMo-preprocessor mel into the TRT encoder
    and compare its output against NeMo `model.encoder(...)`. This isolates the
    encoder graph (subsampling conv, FastConformer rel-pos attention, baked
    rel-PE) from the mel frontend and from the decoder.

  Stage 2 (decoder): feed the *same* encoder output (NeMo's) as cross-attention
    K/V into the TRT decoder, run greedy decoding from the Canary prompt, and
    compare per-step logits / argmax against a NeMo decoder reference driven the
    same way. This isolates the decoder graph + KV cache + position/mask.

The disciplined flow is: run Stage 1 first. Only if the encoder matches does a
decoder-logit comparison mean anything. `--stage all` runs Stage 2 only when
Stage 1 passes.

Usage (run on p2021 inside the NeMo venv):

    python3 tools/diff_canary.py \
      --model /path/to/canary_model_dir_or.nemo \
      --audio /path/to/sample0000.wav \
      --stage all --max-new-tokens 64

    # Separate paths for TRT build vs NeMo reference if needed:
    python3 tools/diff_canary.py \
      --model models/canary --nemo-model nvidia/canary-1b-v2 \
      --audio sample0000.wav --stage encoder
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

# Engine execution helpers (TRT + cuda-python) live in debug_runner.
from tensorrt_model_connect.debug_runner import (  # noqa: E402
    _check_cuda,
    _require_trt_runtime,
    cudart,
    trt,
)


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

def read_audio_mono(path: str) -> tuple[np.ndarray, int]:
    """Read an audio file (wav/flac/...) as float32 mono in [-1, 1].

    Prefers soundfile (handles flac); falls back to the stdlib wave reader
    for plain PCM WAV when soundfile is unavailable."""
    try:
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        data = np.asarray(data, dtype=np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return np.ascontiguousarray(data), int(sr)
    except ImportError:
        pass
    with wave.open(str(path), "rb") as wf:
        n_ch = wf.getnchannels()
        sr = wf.getframerate()
        width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if width == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {width} bytes")
    if n_ch > 1:
        data = data.reshape(-1, n_ch).mean(axis=1)
    return np.ascontiguousarray(data), sr


def resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x
    duration = x.shape[0] / float(src_sr)
    dst_n = int(round(duration * dst_sr))
    src_t = np.linspace(0.0, duration, num=x.shape[0], endpoint=False)
    dst_t = np.linspace(0.0, duration, num=dst_n, endpoint=False)
    return np.interp(dst_t, src_t, x).astype(np.float32)


# ---------------------------------------------------------------------------
# TRT engine build (via plugin)
# ---------------------------------------------------------------------------

def _rebake_rel_pe(weights, build_mel_length: int):
    """Re-derive enc_seq + per-layer rel-PE projections for a chosen mel_length.

    Diagnostic only: the encoder bakes rel-PE for a FIXED enc_seq derived from
    mel_length. This lets us rebuild the encoder for an enc_seq close to the
    actual audio's valid length, to test whether the fixed-length rel-PE is the
    source of encoder divergence."""
    from tensorrt_model_connect.families.canary.plugin import (
        _compute_enc_seq_len, _relative_pe)
    hidden = int(weights["_hidden"])
    enc_heads = int(weights["_enc_heads"])
    head_dim = int(weights["_head_dim"])
    enc_layers = int(weights["_enc_layers"])
    es_new = _compute_enc_seq_len(build_mel_length)
    weights["_mel_length"] = int(build_mel_length)
    weights["_enc_seq"] = es_new
    rpe = _relative_pe(es_new, hidden)
    for i in range(enc_layers):
        proj = rpe @ weights[f"el.{i}.w_pos"]
        weights[f"el.{i}.rpe_proj"] = proj.reshape(2 * es_new - 1, enc_heads, head_dim)
    print(f"[diff-canary] OVERRIDE mel_length={build_mel_length} -> es={es_new} "
          f"(rel-PE rebaked)", file=sys.stderr)


def build_engines(model_path: str, max_cache_length: int, verbose: bool,
                  build_mel_length: int | None = None):
    """Build Canary encoder + decoder engines. Returns (enc_plan, dec_plan, weights, config, meta)."""
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.engine_builder import _resolve_model
    from tensorrt_model_connect.families import find_plugin

    model_dir = _resolve_model(model_path)
    config = ModelConfig.from_dir(model_dir)
    plugin = find_plugin(config.model_type)
    if plugin is None:
        raise ValueError(f"No plugin for model_type={config.model_type!r}")
    if not plugin.matches("canary"):
        print(f"[diff-canary] WARNING: plugin for {config.model_type!r} is not Canary",
              file=sys.stderr)

    print(f"[diff-canary] Loading weights ({config.model_type}) from {model_dir} ...",
          file=sys.stderr)
    weights = plugin.load_weights(model_dir, config)
    if build_mel_length is not None:
        _rebake_rel_pe(weights, build_mel_length)

    print(f"[diff-canary] Building decoder engine (cache={max_cache_length}) ...",
          file=sys.stderr)
    dec_plan = plugin.build_engine(config, weights, max_cache_length, verbose=verbose)
    print(f"[diff-canary] Decoder built ({len(dec_plan) / 1e6:.1f} MB)", file=sys.stderr)

    print("[diff-canary] Building encoder engine ...", file=sys.stderr)
    enc_plan = plugin.build_vision_engine(model_dir, config, weights, verbose=verbose)
    if enc_plan is None:
        raise RuntimeError("Canary encoder build returned None")
    print(f"[diff-canary] Encoder built ({len(enc_plan) / 1e6:.1f} MB)", file=sys.stderr)

    overrides = plugin.get_bundle_config_overrides(config) or {}
    meta = {
        "es": int(weights["_enc_seq"]),
        "mel_bins": int(weights["_mel_bins"]),
        "mel_length": int(weights["_mel_length"]),
        "hidden": int(weights["_hidden"]),
        "dec_layers": int(weights["_dec_layers"]),
        "vocab": int(weights["_vocab"]),
        "prompt_ids": list(overrides.get("decoder_start_token_ids", [])),
        "stop_token_ids": list(overrides.get("stop_token_ids", [3, 2])),
        "max_cache_length": max_cache_length,
    }
    return enc_plan, dec_plan, weights, config, meta


# ---------------------------------------------------------------------------
# Self-contained TRT runners
# ---------------------------------------------------------------------------

class _Ctx:
    """Minimal RAII-ish holder for a deserialized engine + context + stream."""

    def __init__(self, plan: bytes):
        _require_trt_runtime()
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(plan)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TRT engine")
        self.context = self.engine.create_execution_context()
        err, self.stream = cudart.cudaStreamCreate()
        _check_cuda(err)

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.engine.get_tensor_shape(name))


def _malloc(nbytes: int):
    err, ptr = cudart.cudaMalloc(max(nbytes, 1))
    _check_cuda(err)
    return ptr


def _h2d(dst, host: np.ndarray, stream):
    host = np.ascontiguousarray(host)
    cudart.cudaMemcpyAsync(dst, host.ctypes.data, host.nbytes,
                           cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)


def _d2h(host: np.ndarray, src, stream):
    cudart.cudaMemcpyAsync(host.ctypes.data, src, host.nbytes,
                           cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)


def run_trt_encoder(enc_plan: bytes, mel_padded: np.ndarray, enc_mask: np.ndarray,
                    es: int, hidden: int) -> np.ndarray:
    """Run the Canary encoder once. Returns encoder_output [es, hidden]."""
    ctx = _Ctx(enc_plan)
    stream = ctx.stream

    mel_shape = ctx.shape("mel_features")
    mask_shape = ctx.shape("encoder_mask")
    out_shape = ctx.shape("encoder_output")

    if tuple(mel_padded.shape) != tuple(mel_shape):
        raise ValueError(f"mel shape {mel_padded.shape} != engine {mel_shape}")
    if tuple(enc_mask.shape) != tuple(mask_shape):
        raise ValueError(f"mask shape {enc_mask.shape} != engine {mask_shape}")

    d_mel = _malloc(mel_padded.astype(np.float32).nbytes)
    d_mask = _malloc(enc_mask.astype(np.float32).nbytes)
    out_host = np.zeros(out_shape, dtype=np.float32)
    d_out = _malloc(out_host.nbytes)

    _h2d(d_mel, mel_padded.astype(np.float32), stream)
    _h2d(d_mask, enc_mask.astype(np.float32), stream)
    ctx.context.set_tensor_address("mel_features", d_mel)
    ctx.context.set_tensor_address("encoder_mask", d_mask)
    ctx.context.set_tensor_address("encoder_output", d_out)
    ctx.context.execute_async_v3(stream)
    _d2h(out_host, d_out, stream)
    cudart.cudaStreamSynchronize(stream)

    for ptr in (d_mel, d_mask, d_out):
        cudart.cudaFree(ptr)
    return out_host.reshape(es, hidden)


class CanaryDecoderRunner:
    """Single-step KV-cache decoder runner with externally-supplied cross K/V."""

    def __init__(self, dec_plan: bytes, num_layers: int, max_cache_length: int,
                 es: int, hidden: int):
        self.ctx = _Ctx(dec_plan)
        self.num_layers = num_layers
        self.max_cache_length = max_cache_length
        self.es = es
        self.hidden = hidden
        self.cache_length = 0

        cache_shape = self.ctx.shape("cache_k_0")  # (max_cache, attn_size)
        self.attn_size = int(cache_shape[1])
        row_bytes = self.attn_size * 4
        self.row_bytes = row_bytes

        self._d_cache_k = [_malloc(max_cache_length * row_bytes) for _ in range(num_layers)]
        self._d_cache_v = [_malloc(max_cache_length * row_bytes) for _ in range(num_layers)]
        self._d_present_k = [_malloc(row_bytes) for _ in range(num_layers)]
        self._d_present_v = [_malloc(row_bytes) for _ in range(num_layers)]
        cross_bytes = es * hidden * 4
        self._d_cross_k = [_malloc(cross_bytes) for _ in range(num_layers)]
        self._d_cross_v = [_malloc(cross_bytes) for _ in range(num_layers)]

        for i in range(num_layers):
            cudart.cudaMemsetAsync(self._d_cache_k[i], 0, max_cache_length * row_bytes, self.ctx.stream)
            cudart.cudaMemsetAsync(self._d_cache_v[i], 0, max_cache_length * row_bytes, self.ctx.stream)

        self._d_token = _malloc(4)
        self._d_pos = _malloc(4)
        self.attn_window = max_cache_length + 1
        self._d_mask = _malloc(self.attn_window * 4)

        logits_shape = self.ctx.shape("logits")
        self.logits_numel = int(np.prod(logits_shape))
        self._h_logits = np.zeros(self.logits_numel, dtype=np.float32)
        self._d_logits = _malloc(self.logits_numel * 4)
        cudart.cudaStreamSynchronize(self.ctx.stream)

    def set_cross(self, encoder_output: np.ndarray):
        """Load identical raw encoder output into every layer's cross K/V."""
        eo = np.ascontiguousarray(encoder_output.astype(np.float32))
        if eo.shape != (self.es, self.hidden):
            raise ValueError(f"encoder_output {eo.shape} != ({self.es},{self.hidden})")
        for i in range(self.num_layers):
            _h2d(self._d_cross_k[i], eo, self.ctx.stream)
            _h2d(self._d_cross_v[i], eo, self.ctx.stream)
        cudart.cudaStreamSynchronize(self.ctx.stream)

    def step(self, token_id: int) -> np.ndarray:
        stream = self.ctx.stream
        ctx = self.ctx.context

        position_id = min(self.cache_length, self.max_cache_length)
        valid = min(self.cache_length, self.max_cache_length)
        mask = np.full((1, self.attn_window), -1.0e4, dtype=np.float32)
        mask[0, :valid] = 0.0
        mask[0, -1] = 0.0

        _h2d(self._d_token, np.array([token_id], dtype=np.int32), stream)
        _h2d(self._d_pos, np.array([position_id], dtype=np.int32), stream)
        _h2d(self._d_mask, mask, stream)

        ctx.set_tensor_address("token_id", self._d_token)
        ctx.set_tensor_address("position_id", self._d_pos)
        ctx.set_tensor_address("attention_mask", self._d_mask)
        ctx.set_tensor_address("logits", self._d_logits)
        for i in range(self.num_layers):
            ctx.set_tensor_address(f"cache_k_{i}", self._d_cache_k[i])
            ctx.set_tensor_address(f"cache_v_{i}", self._d_cache_v[i])
            ctx.set_tensor_address(f"present_k_{i}", self._d_present_k[i])
            ctx.set_tensor_address(f"present_v_{i}", self._d_present_v[i])
            ctx.set_tensor_address(f"cross_k_{i}", self._d_cross_k[i])
            ctx.set_tensor_address(f"cross_v_{i}", self._d_cross_v[i])
        ctx.execute_async_v3(stream)

        D2D = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        for i in range(self.num_layers):
            for cbuf, pbuf in ((self._d_cache_k[i], self._d_present_k[i]),
                               (self._d_cache_v[i], self._d_present_v[i])):
                if self.cache_length < self.max_cache_length:
                    off = self.cache_length * self.row_bytes
                    cudart.cudaMemcpyAsync(cbuf + off, pbuf, self.row_bytes, D2D, stream)
                else:
                    cudart.cudaMemcpyAsync(cbuf, cbuf + self.row_bytes,
                                           (self.max_cache_length - 1) * self.row_bytes, D2D, stream)
                    off = (self.max_cache_length - 1) * self.row_bytes
                    cudart.cudaMemcpyAsync(cbuf + off, pbuf, self.row_bytes, D2D, stream)

        _d2h(self._h_logits, self._d_logits, stream)
        cudart.cudaStreamSynchronize(stream)
        self.cache_length = min(self.cache_length + 1, self.max_cache_length)
        return self._h_logits.copy()


# ---------------------------------------------------------------------------
# NeMo reference
# ---------------------------------------------------------------------------

def load_nemo_model(nemo_model: str, device: str):
    import nemo.collections.asr as nemo_asr
    if nemo_model.endswith(".nemo") and Path(nemo_model).exists():
        model = nemo_asr.models.ASRModel.restore_from(nemo_model, map_location=device)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(nemo_model, map_location=device)
    try:
        model = model.to(device)
    except Exception:
        pass
    model.eval()
    return model


def nemo_mel_and_encoder(model, audio: np.ndarray, device: str):
    """Return (mel [mel_bins,T], encoder_out [T',D], valid_len, enc_states [T',D])."""
    import torch
    sig = torch.tensor(audio, dtype=torch.float32, device=device).unsqueeze(0)
    sig_len = torch.tensor([audio.shape[0]], dtype=torch.long, device=device)
    with torch.no_grad():
        processed, processed_len = model.preprocessor(input_signal=sig, length=sig_len)
        encoded, encoded_len = model.encoder(audio_signal=processed, length=processed_len)
    mel = processed[0].detach().cpu().numpy()                 # [mel_bins, T]
    enc = encoded[0].detach().cpu().numpy().transpose(1, 0)   # [T', D]
    valid = int(encoded_len[0].item())
    return mel, enc, valid, encoded, encoded_len


# ---------------------------------------------------------------------------
# Comparison reporting
# ---------------------------------------------------------------------------

def _row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return num / den


def compare_encoder(trt_enc: np.ndarray, nemo_enc: np.ndarray, valid: int,
                    atol: float, cos_thresh: float) -> bool:
    n = min(valid, trt_enc.shape[0], nemo_enc.shape[0])
    a = trt_enc[:n]
    b = nemo_enc[:n]
    abs_diff = np.abs(a - b)
    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())
    cos = _row_cosine(a, b)
    min_cos = float(cos.min())
    worst_row = int(abs_diff.max(axis=1).argmax())

    print("\n=== Stage 1: ENCODER parity (valid rows = %d) ===" % n)
    print(f"  shape TRT={trt_enc.shape}  NeMo={nemo_enc.shape}")
    print(f"  max|Δ|={max_abs:.6g}  mean|Δ|={mean_abs:.6g}  min row-cosine={min_cos:.6f}")
    print(f"  worst row={worst_row} (max|Δ|={float(abs_diff[worst_row].max()):.6g}, "
          f"cosine={float(cos[worst_row]):.6f})")
    # First row where cosine drops noticeably.
    bad = np.where(cos < 0.9999)[0]
    if bad.size:
        print(f"  first row with cosine<0.9999: row {int(bad[0])} (cosine={float(cos[bad[0]]):.6f})")
    # Verdict is cosine-based: a real encoder graph bug shows up as valid rows
    # whose direction diverges (cosine << 1). Pure fp32 accumulation keeps
    # cosine ~1 even when max|Δ| exceeds a tight atol, so cosine is the robust
    # signal; max|Δ| vs atol is reported as supplementary info.
    ok = min_cos >= cos_thresh
    print(f"  RESULT: {'PASS' if ok else 'FAIL'} "
          f"(min row-cosine {min_cos:.6f} {'>=' if ok else '<'} {cos_thresh}; "
          f"max|Δ|={max_abs:.4g} vs atol={atol})")
    return ok


def nemo_decoder_logits(model, encoded, encoded_len, prompt_ids, gen_ids, device):
    """Best-effort NeMo decoder logits for the next token given prompt+gen so far.

    NOTE: the NeMo decoder API is version-sensitive. This drives
    transf_decoder + log_softmax directly. Adjust here if your NeMo version
    differs. Returns a numpy vector of size [vocab] (log-probs)."""
    import torch
    ids = torch.tensor([prompt_ids + gen_ids], dtype=torch.long, device=device)
    L = ids.shape[1]
    dec_mask = torch.ones((1, L), dtype=torch.float32, device=device)
    Tp = encoded.shape[-1]
    enc_states = encoded.transpose(1, 2)  # [B, T', D]
    enc_mask = torch.zeros((1, Tp), dtype=torch.float32, device=device)
    enc_mask[0, :int(encoded_len[0].item())] = 1.0
    with torch.no_grad():
        hidden = model.transf_decoder(
            input_ids=ids, decoder_mask=dec_mask,
            encoder_embeddings=enc_states, encoder_mask=enc_mask)
        logp = model.log_softmax(hidden_states=hidden)
    return logp[0, -1].detach().cpu().numpy()


def compare_decoder(model, encoded, encoded_len, nemo_enc_for_trt,
                    dec_runner: CanaryDecoderRunner, meta, max_new_tokens,
                    topk, device) -> bool:
    prompt_ids = meta["prompt_ids"]
    stop_ids = set(meta["stop_token_ids"])
    if not prompt_ids:
        print("[diff-canary] No prompt_ids available; skipping decoder stage", file=sys.stderr)
        return False

    # Feed identical (NeMo) encoder output as cross K/V to isolate the decoder.
    dec_runner.set_cross(nemo_enc_for_trt)

    # Prefill prompt; the logits after the last prompt token select gen token 0.
    last_logits = None
    for tok in prompt_ids:
        last_logits = dec_runner.step(int(tok))

    print("\n=== Stage 2: DECODER per-step logits (greedy) ===")
    print(f"  prompt_ids={prompt_ids}")
    header = f"  {'step':>4} {'trt_tok':>8} {'nemo_tok':>9} {'top{}_overlap'.format(topk):>12} {'logp_max|Δ|':>12}"
    print(header)

    gen_ids: list[int] = []
    first_div = -1
    ok = True
    for step in range(max_new_tokens):
        trt_logits = last_logits
        trt_tok = int(np.argmax(trt_logits))
        # NeMo reference logits for the same context.
        nemo_logp = nemo_decoder_logits(model, encoded, encoded_len, prompt_ids, gen_ids, device)
        nemo_tok = int(np.argmax(nemo_logp))

        trt_topk = set(np.argsort(trt_logits)[-topk:].tolist())
        nemo_topk = set(np.argsort(nemo_logp)[-topk:].tolist())
        overlap = len(trt_topk & nemo_topk)
        # Compare on a common normalization (log-softmax of TRT logits).
        trt_logp = trt_logits - np.log(np.exp(trt_logits - trt_logits.max()).sum()) - trt_logits.max()
        common = min(trt_logp.shape[0], nemo_logp.shape[0])
        logp_dmax = float(np.abs(trt_logp[:common] - nemo_logp[:common]).max())

        flag = "" if trt_tok == nemo_tok else "  <-- DIVERGE"
        if trt_tok != nemo_tok and first_div < 0:
            first_div = step
            ok = False
        print(f"  {step:>4} {trt_tok:>8} {nemo_tok:>9} {overlap:>12} {logp_dmax:>12.4g}{flag}")

        # Advance greedily by the TRT token (we are auditing the TRT trajectory).
        gen_ids.append(trt_tok)
        if trt_tok in stop_ids:
            print(f"  TRT stop token {trt_tok} at step {step}")
            break
        last_logits = dec_runner.step(trt_tok)

    if first_div >= 0:
        print(f"  RESULT: FAIL (first argmax divergence at gen step {first_div})")
    else:
        print("  RESULT: PASS (argmax matched for all compared steps)")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Path/dir for TRT engine build (.nemo or model dir)")
    ap.add_argument("--nemo-model", default=None, help="NeMo model id or .nemo for reference (default: --model)")
    ap.add_argument("--audio", required=True, help="WAV file (sample 0000)")
    ap.add_argument("--stage", choices=["encoder", "decoder", "all"], default="all")
    ap.add_argument("--max-cache-length", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--enc-atol", type=float, default=1e-2)
    ap.add_argument("--enc-cos-thresh", type=float, default=0.999)
    ap.add_argument("--mask-value", type=float, default=-1.0e4)
    ap.add_argument("--target-sr", type=int, default=16000)
    ap.add_argument("--build-mel-length", type=int, default=None,
                    help="Diagnostic: rebuild encoder for this mel_length (rebakes rel-PE) "
                         "to test the fixed-enc_seq rel-PE hypothesis")
    ap.add_argument("--dump-dir", default=None,
                    help="If set, save trt/nemo encoder outputs as .npy here")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    nemo_model = args.nemo_model or args.model

    # 1) Build engines.
    enc_plan, dec_plan, weights, config, meta = build_engines(
        args.model, args.max_cache_length, args.verbose, args.build_mel_length)
    es, mel_bins, mel_length = meta["es"], meta["mel_bins"], meta["mel_length"]
    hidden = meta["hidden"]
    print(f"[diff-canary] meta: es={es} mel_bins={mel_bins} mel_length={mel_length} "
          f"hidden={hidden} dec_layers={meta['dec_layers']} vocab={meta['vocab']}",
          file=sys.stderr)

    # 2) NeMo reference: mel + encoder output.
    print(f"[diff-canary] Loading NeMo model {nemo_model} ...", file=sys.stderr)
    model = load_nemo_model(nemo_model, args.device)
    audio, sr = read_audio_mono(args.audio)
    audio = resample_linear(audio, sr, args.target_sr)
    nemo_mel, nemo_enc, valid_len, encoded, encoded_len = nemo_mel_and_encoder(
        model, audio, args.device)
    print(f"[diff-canary] NeMo: mel={nemo_mel.shape} enc={nemo_enc.shape} valid_len={valid_len}",
          file=sys.stderr)
    from tensorrt_model_connect.families.canary.plugin import _compute_enc_seq_len
    print(f"[diff-canary] _compute_enc_seq_len(T={nemo_mel.shape[1]})={_compute_enc_seq_len(nemo_mel.shape[1])} "
          f"(NeMo encoded_len={valid_len}, engine es={es})", file=sys.stderr)

    # Pad/truncate NeMo mel to the engine's fixed mel_length.
    T = nemo_mel.shape[1]
    if T > mel_length:
        print(f"[diff-canary] WARNING: mel T={T} > mel_length={mel_length}; truncating", file=sys.stderr)
        mel_padded = nemo_mel[:, :mel_length].copy()
    else:
        mel_padded = np.zeros((mel_bins, mel_length), dtype=np.float32)
        mel_padded[:, :T] = nemo_mel
    valid = min(valid_len, es)
    enc_mask = np.full((1, 1, es), args.mask_value, dtype=np.float32)
    enc_mask[0, 0, :valid] = 0.0

    encoder_ok = True
    if args.stage in ("encoder", "all"):
        trt_enc = run_trt_encoder(enc_plan, mel_padded, enc_mask, es, hidden)
        if args.dump_dir:
            dd = Path(args.dump_dir)
            dd.mkdir(parents=True, exist_ok=True)
            np.save(dd / "trt_encoder.npy", trt_enc[:valid])
            np.save(dd / "nemo_encoder.npy", nemo_enc[:valid])
            print(f"[diff-canary] dumped encoder arrays to {dd}", file=sys.stderr)
        encoder_ok = compare_encoder(trt_enc, nemo_enc, valid, args.enc_atol,
                                     args.enc_cos_thresh)

    if args.stage == "encoder":
        return 0 if encoder_ok else 1

    if args.stage == "all" and not encoder_ok:
        print("\n[diff-canary] Encoder diverged -> the root cause is upstream of the "
              "decoder. Fix the encoder before trusting decoder-logit comparison. "
              "Skipping Stage 2.", file=sys.stderr)
        return 1

    # 3) Decoder stage. Feed NeMo encoder output (padded to es) as cross K/V.
    nemo_enc_padded = np.zeros((es, hidden), dtype=np.float32)
    nemo_enc_padded[:min(valid, nemo_enc.shape[0])] = nemo_enc[:min(valid, nemo_enc.shape[0])]
    dec_runner = CanaryDecoderRunner(
        dec_plan, meta["dec_layers"], args.max_cache_length, es, hidden)
    try:
        dec_ok = compare_decoder(model, encoded, encoded_len, nemo_enc_padded,
                                 dec_runner, meta, args.max_new_tokens, args.topk, args.device)
    except Exception as exc:  # NeMo decoder API is version-sensitive.
        print(f"\n[diff-canary] Decoder reference failed: {exc!r}", file=sys.stderr)
        print("[diff-canary] The TRT decoder ran; only the NeMo reference call needs "
              "adjusting for your NeMo version (see nemo_decoder_logits()).", file=sys.stderr)
        return 2

    return 0 if (encoder_ok and dec_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
