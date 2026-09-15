# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental offline Python TTS composition with prepared voice features.

No learned reference-audio frontend is hidden here. The NPZ is an explicit
input contract, not a runtime call to the official PyTorch/ONNX pipeline.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import re
import time

import numpy as np

from .llm import END_OF_PROMPT, LLMEngine, pack_prompt


def text_tokenizer(model_dir):
    """Local Qwen BPE for plain text and the instruction boundary only.

    No text normalization, phoneme overrides or paralinguistic tags are claimed.
    Tokenization is CPU string processing, not a neural inference fallback.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(Path(model_dir) / "CosyVoice-BlankEN",
                                              local_files_only=True, trust_remote_code=False)
    tokenizer.add_special_tokens({"eos_token": "<|endoftext|>", "pad_token": "<|endoftext|>",
                                  "additional_special_tokens": ["<|im_start|>", "<|im_end|>", "<|endofprompt|>"]})
    if tokenizer.encode("<|endofprompt|>", add_special_tokens=False) != [END_OF_PROMPT]:
        raise ValueError("Tokenizer does not have the published CosyVoice3 instruction boundary")
    return tokenizer


def encode_request(model_dir, text, *, instruction="You are a helpful assistant.", prompt_text="", prompt_tokens=()):
    for name, value in (("text", text), ("instruction", instruction), ("prompt_text", prompt_text)):
        if not isinstance(value, str) or re.search(r"<[^>]*>|\[[^\]]*\]", value):
            raise ValueError(f"{name}: this entry point accepts plain text, not control/phoneme tags")
    if not text.strip() or not instruction.strip():
        raise ValueError("Text and instruction must not be empty")
    if prompt_text and (not prompt_text.strip() or len(prompt_tokens) == 0):
        raise ValueError("Zero-shot prompt text requires a nonempty transcript and reference speech tokens")
    tokenizer = text_tokenizer(model_dir)
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    prompt_ids = tokenizer.encode(instruction + "<|endofprompt|>" + prompt_text, add_special_tokens=False)
    packed = pack_prompt(text_ids, prompt_ids, prompt_tokens if prompt_text else ())
    return packed, len(text_ids)


def read_voice(path):
    """NPZ: prompt_tokens INT32[1,N], prompt_features FP32[1,2N,80], speaker FP32[1,192]."""
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"prompt_tokens", "prompt_features", "speaker"}:
            raise ValueError("Voice NPZ requires exactly prompt_tokens, prompt_features and speaker")
        voice = {k: arrays[k].copy() for k in arrays.files}
    tokens, features, speaker = (voice[k] for k in ("prompt_tokens", "prompt_features", "speaker"))
    if tokens.dtype != np.int32 or tokens.ndim != 2 or tokens.shape[0] != 1 or np.any((tokens < 0) | (tokens >= 6561)):
        raise ValueError("Invalid prompt speech token array")
    if features.shape != (1, tokens.shape[1] * 2, 80) or features.dtype != np.float32 or not np.isfinite(features).all():
        raise ValueError("Prompt features must align exactly with prompt speech tokens (2 frames/token)")
    if speaker.shape != (1, 192) or speaker.dtype != np.float32 or not np.isfinite(speaker).all():
        raise ValueError("Invalid speaker embedding")
    if np.linalg.norm(speaker.astype(np.float64)) == 0:
        raise ValueError("A real, nonzero speaker embedding is required; no placeholder voice")
    return voice


def reference_features(audio):
    """Non-learned signal processing, matching the pinned official frontend.

    SoundFile decodes, torchaudio resamples/computes Kaldi filterbanks, and
    Whisper's log-mel function computes features only (no Whisper model).
    The 24 kHz mel equations are family-owned and need no official checkout.
    """
    import soundfile
    import torch
    import torchaudio
    import whisper
    from librosa.filters import mel as mel_filters
    from torchaudio.compliance.kaldi import fbank

    info = soundfile.info(audio)
    if info.samplerate < 16000 or not 0.1 <= info.frames / info.samplerate <= 30:
        raise ValueError("Reference audio requires 0.1–30 seconds at >= 16 kHz")
    samples, sr = soundfile.read(audio, dtype="float32", always_2d=True)
    if not np.isfinite(samples).all():
        raise ValueError("Reference audio contains NaN/Inf")
    waveform = torch.from_numpy(samples.T.copy()).mean(0, keepdim=True)
    # Match upstream's cached FP32 resampling kernel. The functional API
    # computes its kernel differently and is not bit-identical here.
    speech16 = torchaudio.transforms.Resample(sr, 16000)(waveform) if sr != 16000 else waveform
    speech24 = torchaudio.transforms.Resample(sr, 24000)(waveform) if sr != 24000 else waveform
    bank = fbank(speech16, num_mel_bins=80, dither=0, sample_frequency=16000)
    speaker = (bank - bank.mean(0, keepdim=True))[None].contiguous()
    tokens = whisper.log_mel_spectrogram(speech16, n_mels=128).contiguous()
    padded = torch.nn.functional.pad(speech24[:, None], (720, 720), mode="reflect")[:, 0]
    spectrum = torch.view_as_real(torch.stft(padded, 1920, hop_length=480, win_length=1920,
                                            window=torch.hann_window(1920), center=False,
                                            normalized=False, onesided=True, return_complex=True))
    magnitude = torch.sqrt(spectrum.pow(2).sum(-1) + 1e-9)
    basis = torch.from_numpy(mel_filters(sr=24000, n_fft=1920, n_mels=80, fmin=0, fmax=None))
    mel = torch.log(torch.clamp(basis @ magnitude, min=1e-5)).transpose(1, 2).contiguous()
    return {"speaker_features": speaker, "token_features": tokens, "mel": mel}


def prepare_voice(args):
    """Reference WAV -> native TensorRT speaker/tokens -> explicit voice NPZ."""
    import torch
    from .frontend import FrontendEngine

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    features = reference_features(args.audio)
    components, values = {}, {}
    for component, path, feature_name in (("campplus", args.campplus, "speaker_features"),
                                           ("speech_tokenizer", args.speech_tokenizer, "token_features")):
        engine = FrontendEngine(path, component)
        result = engine.extract(features[feature_name].to(engine.device))
        values.update({key: tensor.cpu().numpy().copy() for key, tensor in result.items()})
        components[component] = engine.manifest["component"]
        del result, engine
        gc.collect()
        torch.cuda.empty_cache()
    count = min(values["tokens"].shape[1], features["mel"].shape[1] // 2)
    if count < 1 or not np.linalg.norm(values["speaker"].astype(np.float64)):
        raise ValueError("Reference audio produced no usable voice conditioning")
    if np.any((values["tokens"] < 0) | (values["tokens"] >= 6561)):
        raise RuntimeError("Speech tokenizer produced an out-of-vocabulary token")
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "voice.npz", prompt_tokens=values["tokens"][:, :count],
             prompt_features=features["mel"][:, :count * 2].numpy(), speaker=values["speaker"])
    read_voice(output / "voice.npz")
    report = {"status": "completed_unqualified", "learned_execution": "native_tensorrt",
              "source_audio": str(args.audio), "components": components,
              "voice": "voice.npz", "prompt_tokens": count,
              "prompt_mel_frames": 2 * count, "alignment": "official_24khz_crop_2_frames_per_token"}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def synthesize(args):
    """Unload each stage before the next, so this development path fits an 8 GB GPU."""
    import torch
    import soundfile as sf
    from .conditioning import ConditioningEngine
    from .flow_runtime import FlowEngine
    from .hift import HiFTEngine
    from .flow import OfflineFlow

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if type(args.seed) is not int or not 0 <= args.seed < 2**64:
        raise ValueError("seed must be an integer in [0, 2**64)")
    if type(args.max_tokens) is not int or args.max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    voice = read_voice(args.voice)
    packed, text_length = encode_request(args.model_dir, args.text, instruction=args.instruction,
                                         prompt_text=args.prompt_text, prompt_tokens=voice["prompt_tokens"][0].tolist())
    manifests = {key: json.loads((path / "manifest.json").read_text(encoding="utf-8")) for key, path in
                 (("llm", args.llm), ("flow", args.flow), ("conditioner", args.conditioner), ("hift", args.hift))}
    prompt_count = voice["prompt_tokens"].shape[1]
    max_tokens = min(args.max_tokens, text_length * 20)
    limits = (manifests["conditioner"]["profile"]["max_tokens"] - prompt_count,
              manifests["flow"]["profile"]["max_frames"] // 2 - prompt_count,
              manifests["hift"]["profile"]["max_frames"] // 2,
              manifests["llm"]["max_context"] - len(packed) + 1)
    if type(args.max_tokens) is not int or max_tokens < 1 or max_tokens > min(limits):
        raise ValueError(f"Requested max_tokens={max_tokens} exceeds component capacity {min(limits)}; rebuild larger profiles")
    output.mkdir(parents=True)
    report = {"status": "running", "text": args.text, "instruction": args.instruction, "prompt_text": args.prompt_text,
              "seed": args.seed, "sampling": "greedy" if args.greedy else "ras_top_p_0.8_top_k_25",
              "sampling_rng": "numpy_pcg64_not_bit_identical_to_torch_rng", "packed_ids": packed,
              "target_text_token_count": text_length,
              "max_tokens": max_tokens, "voice": str(args.voice),
              "tokenizer_files": ["vocab.json", "merges.txt", "tokenizer_config.json"],
              "components": {k: m["component"] for k, m in manifests.items()},
              "qualification": "not_registered_not_audio_quality_qualified", "stages_seconds": {}}

    def save():
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        start = time.perf_counter()
        llm = LLMEngine(args.llm)
        generated = llm.generate(packed, max_tokens=max_tokens, min_tokens=min(text_length * 2, max_tokens),
                                 seed=args.seed, greedy=args.greedy)
        report.update(generated)
        report["stages_seconds"]["llm_including_load"] = time.perf_counter() - start
        save()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        if generated["finish_reason"] != "stop_token":
            raise RuntimeError("LLM reached its length limit without a stop token; tokens saved, refusing to label truncated audio complete")
        if not generated["tokens"]:
            raise RuntimeError("LLM stopped without producing speech tokens")
        start = time.perf_counter()
        conditioner, estimator = ConditioningEngine(args.conditioner), FlowEngine(args.flow)
        flow = OfflineFlow(conditioner, estimator)
        tokens = torch.tensor([generated["tokens"]], dtype=torch.int32, device=estimator.device)
        request = {k: torch.from_numpy(v).to(estimator.device) for k, v in voice.items()}
        frames = (tokens.shape[1] + prompt_count) * 2
        rng = torch.Generator(device=estimator.device).manual_seed(args.seed)
        noise = torch.randn((1, 80, frames), dtype=torch.float32, device=estimator.device, generator=rng)
        mel = flow(tokens=tokens, noise=noise, **request).cpu()
        np.save(output / "mel.npy", mel.numpy(), allow_pickle=False)
        report["stages_seconds"]["flow_including_load"] = time.perf_counter() - start
        save()
        del flow, conditioner, estimator, request, noise, tokens
        gc.collect()
        torch.cuda.empty_cache()
        start = time.perf_counter()
        hift = HiFTEngine(args.hift)
        # CPU generator avoids global random state and unnecessary 300-second source allocation.
        uniform = np.random.default_rng(args.seed).random((1, 9, mel.shape[2] * 480), dtype=np.float32)
        result = hift(mel.to(hift.device), noise=torch.from_numpy(uniform).to(hift.device))
        waveform = result["audio"][0].cpu().numpy()
        sf.write(output / "audio.wav", waveform, 24000, subtype="PCM_16")
        report.update(status="completed_unqualified", sample_rate=24000, samples=len(waveform),
                      duration_seconds=len(waveform) / 24000, audio="audio.wav")
        report["stages_seconds"]["hift_including_load"] = time.perf_counter() - start
        save()
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2))
