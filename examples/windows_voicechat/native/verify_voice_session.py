# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay recorded microphone speech through the real native RTX bridge.

Requires numpy and soundfile. This is an explicit model/GPU check, not a demo
mode or a unit-test double. The input is paced in 20 ms blocks like the desktop.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import queue
import subprocess
import threading
import time
import wave

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Verification JSON receipt")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--system-prompt", help="Override the native session system prompt")
    parser.add_argument("--runtime-cache", type=Path, help="Use the application's TensorRT-RTX runtime cache")
    parser.add_argument("--lead-silence-seconds", type=float, default=0)
    parser.add_argument("--tail-silence-seconds", type=float, default=0)
    args = parser.parse_args()
    import soundfile as sf

    source, rate = sf.read(args.input, dtype="float32", always_2d=True)
    if rate != 16000 or source.shape[1] != 1:
        raise ValueError("Verification fixture must be mono 16000 Hz audio")
    source = source[:, 0]
    original_samples = len(source)
    for value in (args.lead_silence_seconds, args.tail_silence_seconds):
        if not np.isfinite(value) or not 0 <= value <= 120:
            raise ValueError("Leading and trailing silence must be between 0 and 120 seconds")
    source = np.concatenate((np.zeros(round(rate * args.lead_silence_seconds), dtype=np.float32),
                             source, np.zeros(round(rate * args.tail_silence_seconds), dtype=np.float32)))
    messages: queue.Queue = queue.Queue()
    events, audio, transcript = [], [], []
    errors = []
    finished = threading.Event()
    started = time.monotonic()
    reader = sender = None
    log_path = args.output.with_suffix(".stderr.log")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as diagnostics:
        command = [str(args.bridge), "--bundle", str(args.bundle), "--runtime-root", str(args.bridge.parent)]
        if args.system_prompt is not None:
            command.extend(("--system-prompt", args.system_prompt))
        if args.runtime_cache is not None:
            command.extend(("--runtime-cache", str(args.runtime_cache)))
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=diagnostics,
            text=True, encoding="utf-8", bufsize=1,
        )

        def read_events() -> None:
            try:
                for line in process.stdout:
                    messages.put(json.loads(line))
            except Exception as error:
                messages.put({"type": "error", "fatal": True, "message": str(error)})
            finally:
                messages.put({"type": "pipe_eof"})

        def send_audio() -> None:
            try:
                deadline = time.monotonic()
                for offset in range(0, len(source), 320):
                    if finished.is_set():
                        return
                    encoded = base64.b64encode(source[offset:offset + 320].astype("<f4").tobytes()).decode("ascii")
                    process.stdin.write(json.dumps({"type": "audio", "sampleRate": 16000, "audio": encoded}) + "\n")
                    process.stdin.flush()
                    deadline += 0.02
                    if finished.wait(max(0, deadline - time.monotonic())):
                        return
                process.stdin.write('{"type":"finish"}\n')
                process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as error:
                if not finished.is_set():
                    messages.put({"type": "error", "fatal": True, "message": str(error)})

        try:
            reader = threading.Thread(target=read_events, daemon=True)
            reader.start()
            while time.monotonic() - started < args.timeout:
                try:
                    message = messages.get(timeout=0.25)
                except queue.Empty:
                    continue
                kind = message.get("type")
                if kind == "pipe_eof":
                    break
                if kind == "error":
                    errors.append(message["message"])
                    if message.get("fatal", True):
                        break
                elif kind == "ready":
                    if message.get("backend") != "trt_rtx":
                        raise RuntimeError("Bridge ready message does not confirm TensorRT-RTX")
                    print(json.dumps(message), flush=True)
                    sender = threading.Thread(target=send_audio, daemon=True)
                    sender.start()
                elif kind == "event":
                    event = dict(message)
                    if event.get("kind") == "agent_audio":
                        if event["sampleRate"] != 48000:
                            raise RuntimeError("Bridge emitted an unexpected sample rate")
                        decoded = np.frombuffer(base64.b64decode(event.pop("audio"), validate=True), dtype="<f4").copy()
                        if len(decoded) != event["sampleCount"] or not np.isfinite(decoded).all():
                            raise RuntimeError("Bridge emitted invalid float32 audio")
                        audio.append(decoded)
                    if event.get("text") and event.get("kind") in {"agent_text", "user_transcript"}:
                        print(json.dumps(event), flush=True)
                        transcript.append({key: event[key] for key in ("kind", "epoch", "text", "isFinal")})
                    events.append(event)
                elif kind == "stopped":
                    break
            else:
                errors.append(f"Bridge exceeded the {args.timeout:g}-second verification timeout")
        finally:
            finished.set()
            if sender:
                sender.join(timeout=2)
            if process.poll() is None:
                try:
                    process.stdin.write('{"type":"stop"}\n')
                    process.stdin.flush()
                except (BrokenPipeError, OSError, ValueError):
                    pass
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            if reader:
                reader.join(timeout=2)
            process.stdin.close()
            process.stdout.close()

    samples = np.concatenate(audio) if audio else np.empty(0, dtype=np.float32)
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if samples.size else 0
    if process.returncode:
        errors.append(f"Bridge exited with status {process.returncode}")
    if samples.size == 0 or rms < 0.0001:
        errors.append("No audible model-generated speech was produced")
    if not any(event["kind"] == "user_transcript" and event["text"].strip() for event in transcript):
        errors.append("The input fixture produced no user transcript")
    if not any(event["kind"] == "agent_text" and event["text"].strip() for event in transcript):
        errors.append("The model produced no response text")
    audio_path = args.output.with_suffix(".wav")
    if samples.size:
        with wave.open(str(audio_path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(48000)
            output.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    result = {
        "passed": not errors, "backend": "trt_rtx", "bundle": str(args.bundle),
        "input": str(args.input), "inputSamples": original_samples,
        "streamedInputSamples": len(source),
        "leadSilenceSeconds": args.lead_silence_seconds,
        "tailSilenceSeconds": args.tail_silence_seconds,
        "systemPrompt": args.system_prompt,
        "runtimeCache": str(args.runtime_cache) if args.runtime_cache else None,
        "elapsedSeconds": time.monotonic() - started,
        "outputSamples": int(samples.size), "outputRms": rms,
        "outputAudio": str(audio_path) if samples.size else None,
        "transcript": transcript, "eventCount": len(events),
        "errors": errors, "diagnostics": str(log_path),
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
