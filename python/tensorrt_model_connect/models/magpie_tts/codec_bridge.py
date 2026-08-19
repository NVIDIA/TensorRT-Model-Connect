#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NanoCodec decoding bridge for MagpieTTS.

Subprocess bridge that loads NeMo's AudioCodecModel and decodes codec
tokens (8 codebooks x T frames) to a WAV file.

Usage:
    python3 -m tensorrt_model_connect.models.magpie_tts.codec_bridge \
        --codec-model /path/to/nanocodec.nemo \
        --tokens-file /tmp/codes.json \
        --output /tmp/output.wav \
        [--device cpu|cuda]

Tokens JSON format:
    {"codes": [[c0_f0, c0_f1, ...], [c1_f0, c1_f1, ...], ...], "num_frames": 82}
    (8 arrays of length T, one per codebook)

Also accepts --tokens-npy for loading directly from a .npy file (shape [1, 8, T]
or [8, T]).
"""

import argparse
import json
import os
import sys
import wave

import numpy as np
import torch

# Prevent NeMo from trying to download speaker encoder at load time.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def resolve_device(requested: str) -> str:
    """Return the requested device if available, else fall back to cpu."""
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to cpu", file=sys.stderr)
        return "cpu"
    return requested


def load_codec(codec_path: str, device: str):
    """Load NanoCodec from a .nemo checkpoint.

    Disables use_scl_loss in the config override to avoid downloading
    the speaker encoder model from HuggingFace at load time (not needed
    for decoding).
    """
    import tempfile

    from nemo.collections.tts.models import AudioCodecModel
    from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
    from omegaconf import OmegaConf

    # Extract config, disable speaker encoder loss, save override.
    connector = SaveRestoreConnector()
    tmpdir = tempfile.mkdtemp()
    connector._unpack_nemo_file(codec_path, tmpdir)
    cfg = OmegaConf.load(os.path.join(tmpdir, "model_config.yaml"))
    cfg.use_scl_loss = False
    override_path = os.path.join(tmpdir, "override_config.yaml")
    OmegaConf.save(cfg, override_path)

    codec = AudioCodecModel.restore_from(
        codec_path,
        override_config_path=override_path,
        map_location=device,
        strict=False,
    )
    codec.eval()
    return codec


def load_tokens_json(path: str) -> np.ndarray:
    """Load codec tokens from JSON file. Returns shape [8, T] int64 array."""
    with open(path, "r") as f:
        data = json.load(f)
    codes = np.array(data["codes"], dtype=np.int64)
    if codes.ndim != 2 or codes.shape[0] != 8:
        raise ValueError(
            f"Expected codes shape [8, T], got {codes.shape}"
        )
    return codes


def load_tokens_npy(path: str) -> np.ndarray:
    """Load codec tokens from .npy file. Returns shape [8, T] int64 array."""
    codes = np.load(path)
    if codes.ndim == 3:
        # [1, 8, T] -> [8, T]
        if codes.shape[0] == 1:
            codes = codes[0]
        else:
            raise ValueError(
                f"Expected batch dim 1, got shape {codes.shape}"
            )
    if codes.ndim != 2 or codes.shape[0] != 8:
        raise ValueError(
            f"Expected codes shape [8, T], got {codes.shape}"
        )
    return codes.astype(np.int64)


def decode_tokens(codec, codes: np.ndarray, device: str) -> np.ndarray:
    """Decode [8, T] codec tokens to float32 audio waveform.

    Returns 1-D float32 numpy array.
    """
    tokens = torch.tensor(codes, device=device, dtype=torch.long).unsqueeze(0)  # [1, 8, T]
    tokens_len = torch.tensor([tokens.shape[2]], dtype=torch.long, device=device)
    with torch.no_grad():
        wav, wav_len = codec.decode(tokens=tokens, tokens_len=tokens_len)
    # wav shape: [1, 1, num_samples] or [1, num_samples]
    wav_np = wav.squeeze().cpu().numpy().astype(np.float32)
    length = wav_len[0].item()
    return wav_np[:length]


def write_wav(path: str, audio: np.ndarray, sample_rate: int = 22050):
    """Write float32 audio to int16 PCM WAV."""
    # Clip and convert to int16
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def main():
    parser = argparse.ArgumentParser(description="NanoCodec decoding bridge")
    parser.add_argument(
        "--codec-model", required=True,
        help="Path to NanoCodec .nemo checkpoint",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--tokens-file",
        help="Path to JSON file with codec tokens",
    )
    group.add_argument(
        "--tokens-npy",
        help="Path to .npy file with codec tokens (shape [1,8,T] or [8,T])",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output WAV file path",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device for decoding (default: cuda)",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=22050,
        help="Output sample rate (default: 22050)",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)

    # Load tokens
    if args.tokens_file:
        codes = load_tokens_json(args.tokens_file)
    else:
        codes = load_tokens_npy(args.tokens_npy)
    print(f"Loaded codes: shape={codes.shape}, range=[{codes.min()}, {codes.max()}]",
          file=sys.stderr)

    # Load codec
    print(f"Loading NanoCodec from {args.codec_model}...", file=sys.stderr)
    codec = load_codec(args.codec_model, device)
    print("Codec loaded.", file=sys.stderr)

    # Decode
    print("Decoding...", file=sys.stderr)
    audio = decode_tokens(codec, codes, device)
    print(f"Decoded audio: {len(audio)} samples, "
          f"{len(audio) / args.sample_rate:.3f}s", file=sys.stderr)

    # Write WAV
    write_wav(args.output, audio, args.sample_rate)
    print(f"Wrote {args.output}", file=sys.stderr)

    # Print summary to stdout for programmatic consumption
    result = {
        "output": args.output,
        "num_samples": len(audio),
        "duration_s": len(audio) / args.sample_rate,
        "sample_rate": args.sample_rate,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
