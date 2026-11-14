#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare complete, pinned LibriSpeech WAV clips for the published AIPerf client."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import wave


def parquet_rows(path: str):
    import pyarrow.parquet as pq

    # Read a local pinned shard; no Python network callbacks survive early exit.
    with pq.ParquetFile(path) as source:
        for batch in source.iter_batches(batch_size=32, use_threads=False):
            yield from batch.to_pylist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("count must be positive")

    from huggingface_hub import hf_hub_download
    import soundfile as sf

    # Reuse the qualification application's canonical dataset pin.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugins"))
    from trtmc_aiperf_plugins.datasets import LIBRISPEECH_REVISION

    args.output_dir.mkdir(parents=True, exist_ok=False)
    audit = {"dataset": "openslr/librispeech_asr", "revision": LIBRISPEECH_REVISION,
             "subset": "clean", "split": "test", "selection": "first eligible clips in dataset order",
             "requested": args.count, "selected": [], "skipped": [], "failed": []}
    shard = hf_hub_download(audit["dataset"], "clean/test/0000.parquet", repo_type="dataset",
                            revision=LIBRISPEECH_REVISION)
    audit["source_file"] = "clean/test/0000.parquet"
    rows = parquet_rows(shard)
    try:
        for row in rows:
            sample_id = str(row["id"])
            try:
                raw = row["audio"]["bytes"]
                if not isinstance(raw, bytes):
                    raise ValueError("dataset did not provide embedded audio bytes")
                info = sf.info(io.BytesIO(raw))
                if info.channels != 1 or info.subtype != "PCM_16" or info.duration > 30:
                    audit["skipped"].append({"id": sample_id, "reason": "outside mono PCM16 / 30-second input contract",
                                             "duration_seconds": info.duration, "channels": info.channels})
                    continue
                samples, rate = sf.read(io.BytesIO(raw), dtype="int16")
                pcm = samples.astype("<i2").tobytes()
                path = args.output_dir.resolve() / f"clip-{len(audit['selected']):03d}.wav"
                with wave.open(str(path), "wb") as audio:
                    audio.setparams((1, 2, rate, len(samples), "NONE", "not compressed"))
                    audio.writeframes(pcm)
                audit["selected"].append({"id": sample_id, "speaker_id": str(row["speaker_id"]),
                    "text": row["text"], "audio_path": str(path), "sample_rate": rate,
                    "sample_count": len(samples), "duration_seconds": len(samples) / rate,
                    "source_sha256": hashlib.sha256(raw).hexdigest(),
                    "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                    "wav_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            except Exception as error:  # noqa: BLE001 - failed rows must stay in the selection audit
                audit["failed"].append({"id": sample_id, "reason": str(error)})
                raise
            if len(audit["selected"]) == args.count:
                break
        if len(audit["selected"]) != args.count:
            raise ValueError("dataset ended before the requested selection was complete")
        (args.output_dir / "inputs.jsonl").write_text("".join(json.dumps({
            "session_id": item["id"], "audio": item["audio_path"]}) + "\n" for item in audit["selected"]))
    finally:
        rows.close()
        (args.output_dir / "selection.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Prepared {len(audit['selected'])} complete WAV clips in {args.output_dir}")


if __name__ == "__main__":
    main()
