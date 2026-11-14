# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the pinned upstream generation routines without adapting their semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor, LFMModality


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model = LFM2AudioModel.from_pretrained(Path(args.checkpoint)).eval()
    processor = LFM2AudioProcessor.from_pretrained(Path(args.checkpoint)).eval()
    results = []
    for case in json.loads(Path(args.cases).read_text()):
        torch.manual_seed(case.get("seed", 0))
        mode = case["mode"]
        voice = {
            "us_male": "US male",
            "us_female": "US female",
            "uk_male": "UK male",
            "uk_female": "UK female",
        }[case.get("voice", "us_male")]
        system = "Perform ASR." if mode == "asr" else f"Perform TTS. Use the {voice} voice."
        if mode in {"chat", "speech_response"}:
            system = "Respond with interleaved text and audio."
        if mode in {"text", "mixed"}:
            system = "Respond with text."
        chat = ChatState(processor)
        chat.new_turn("system")
        chat.add_text(system)
        chat.end_turn()
        turns = case["turns"] if mode == "chat" else [case]
        outputs = []
        for i, turn in enumerate(turns):
            chat.new_turn("user")
            if "audio" in turn:
                wave = torch.from_numpy(np.fromfile(turn["audio"], np.float32))[None]
                chat.add_audio(wave, case.get("sample_rate", 16000))
            if "text" in turn:
                chat.add_text(turn["text"])
            chat.end_turn()
            chat.new_turn("assistant")
            text = []
            audio = []
            modalities = []
            generate = (
                model.generate_interleaved
                if mode in {"chat", "speech_response"}
                else model.generate_sequential
            )
            for token in generate(
                **chat,
                max_new_tokens=case.get("max_new_tokens", 128),
                text_temperature=case.get("text_temperature", 0.0),
                text_top_k=case.get("text_top_k", 1),
                audio_temperature=case.get("audio_temperature", 0.0),
                audio_top_k=case.get("audio_top_k", 1),
            ):
                if token.numel() == 1:
                    text.append(token)
                    modalities.append(LFMModality.TEXT)
                else:
                    audio.append(token)
                    modalities.append(LFMModality.AUDIO_OUT)
            ids = torch.cat(text).tolist() if text else []
            codes = (
                torch.stack(audio, 1)
                if audio
                else torch.empty(
                    (8, 0), device=model.lfm.embed_tokens.weight.device, dtype=torch.long
                )
            )
            # Completed card examples end with EOA and decode audio_out[:-1].
            # Preserve ordinary frames when an explicit generation limit
            # truncates a response; such a response is not completion evidence.
            ended_audio = bool(codes.shape[1] and (codes[:, -1] == 2048).all())
            valid = codes[:, :-1] if ended_audio else codes
            waveform = (
                processor.decode(valid[None]).cpu().numpy()[0]
                if valid.numel()
                else np.empty(0, np.float32)
            )
            record = {
                "prompt_tokens": int(chat.modality_flag.numel()),
                "prompt_text_tokens": chat.text[0].tolist(),
                "prompt_modalities": chat.modality_flag.tolist(),
                "text": processor.text.decode(ids, skip_special_tokens=True),
                "raw_text": processor.text.decode(ids),
                "tokens": ids,
                "codes": codes.cpu().tolist(),
                "modalities": [int(value) for value in modalities],
                "samples": int(waveform.size),
                "ended_audio": ended_audio,
            }
            path = Path(args.output).parent / (case["name"] + f"-{i}.reference.f32")
            waveform.tofile(path)
            record["waveform"] = str(path)
            outputs.append(record)
            if mode == "chat":
                chat.append(
                    text=torch.stack(text, 1) if text else chat.text.new_empty(1, 0),
                    audio_out=codes,
                    modality_flag=torch.tensor(modalities, device=chat.text.device),
                )
                chat.end_turn()
        result = {"name": case["name"], "turns": outputs}
        results.append(result)
        print(json.dumps(result), flush=True)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
