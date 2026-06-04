#!/usr/bin/env python3
"""Compare two WAV files with exact sample-rate and waveform-byte equality."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any


def read_wav_payload(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as f:
            if f.read(4) != b"RIFF":
                return None
            f.read(4)
            if f.read(4) != b"WAVE":
                return None

            data_bytes = b""
            audio_format = 1
            num_channels = 1
            sample_rate = 0
            bits_per_sample = 16
            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    break
                chunk_size_raw = f.read(4)
                if len(chunk_size_raw) < 4:
                    break
                chunk_size = struct.unpack("<I", chunk_size_raw)[0]
                if chunk_id == b"fmt ":
                    fmt_data = f.read(chunk_size)
                    if len(fmt_data) >= 16:
                        audio_format = struct.unpack("<H", fmt_data[0:2])[0]
                        num_channels = struct.unpack("<H", fmt_data[2:4])[0]
                        sample_rate = struct.unpack("<I", fmt_data[4:8])[0]
                        bits_per_sample = struct.unpack("<H", fmt_data[14:16])[0]
                elif chunk_id == b"data":
                    data_bytes = f.read(chunk_size)
                else:
                    f.read(chunk_size)
                if chunk_size % 2 == 1:
                    f.read(1)
    except OSError:
        return None

    return {
        "audio_format": audio_format,
        "bits_per_sample": bits_per_sample,
        "num_channels": num_channels,
        "sample_rate": sample_rate,
        "data": data_bytes,
    }


def compare_wavs(trt_wav: Path, ref_wav: Path) -> dict[str, Any]:
    trt = read_wav_payload(trt_wav)
    ref = read_wav_payload(ref_wav)
    if trt is None or ref is None:
        return {
            "passed": False,
            "trt_wav": str(trt_wav),
            "ref_wav": str(ref_wav),
            "error": "could not read TRT or reference WAV",
        }

    metrics = {
        "sample_rate_exact": trt["sample_rate"] == ref["sample_rate"],
        "channel_count_exact": trt["num_channels"] == ref["num_channels"],
        "sample_format_exact": (
            trt["audio_format"] == ref["audio_format"]
            and trt["bits_per_sample"] == ref["bits_per_sample"]
        ),
        "waveform_length_exact": len(trt["data"]) == len(ref["data"]),
        "waveform_exact_match": trt["data"] == ref["data"],
    }
    return {
        "passed": all(metrics.values()),
        "trt_wav": str(trt_wav),
        "ref_wav": str(ref_wav),
        "metrics": metrics,
        "trt": {
            "sample_rate": trt["sample_rate"],
            "num_channels": trt["num_channels"],
            "audio_format": trt["audio_format"],
            "bits_per_sample": trt["bits_per_sample"],
            "data_bytes": len(trt["data"]),
        },
        "ref": {
            "sample_rate": ref["sample_rate"],
            "num_channels": ref["num_channels"],
            "audio_format": ref["audio_format"],
            "bits_per_sample": ref["bits_per_sample"],
            "data_bytes": len(ref["data"]),
        },
    }


_read_wav_payload = read_wav_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trt_wav", type=Path)
    parser.add_argument("ref_wav", type=Path)
    args = parser.parse_args(argv)

    result = compare_wavs(args.trt_wav, args.ref_wav)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
