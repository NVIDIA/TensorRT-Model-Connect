# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hold original backbone activations fixed while testing every depth codebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor

from families.lfm2_audio.tests.component_parity import Runner, metrics


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    model = LFM2AudioModel.from_pretrained(Path(args.checkpoint)).eval()
    processor = LFM2AudioProcessor.from_pretrained(Path(args.checkpoint)).eval()
    runner = Runner(args.engine)
    report = []
    for voice, frame_index in (("US male", 12), ("US female", 4), ("UK male", 4), ("UK female", 5)):
        chat = ChatState(processor)
        chat.new_turn("system")
        chat.add_text(f"Perform TTS. Use the {voice} voice.")
        chat.end_turn()
        chat.new_turn("user")
        chat.add_text("Hello there. How can I help you today?")
        chat.end_turn()
        chat.new_turn("assistant")
        frames = []
        original = model._sample_audio_frame

        def capture(embedding, **kwargs):
            codes = original(embedding, **kwargs)
            frames.append((embedding.detach().clone(), codes.detach().clone()))
            return codes

        model._sample_audio_frame = capture
        try:
            for token in model.generate_sequential(
                **chat,
                max_new_tokens=128,
                text_temperature=0.0,
                text_top_k=1,
                audio_temperature=0.0,
                audio_top_k=1,
            ):
                if len(frames) > frame_index:
                    break
        finally:
            model._sample_audio_frame = original
        hidden, codes = frames[frame_index]
        projected = model.depth_linear(hidden).reshape(8, 1024)
        previous = torch.zeros(1024, device="cuda", dtype=torch.bfloat16)
        cache = None
        runner.reset()
        entries = []
        for codebook in range(8):
            ref, cache = model.depthformer.forward_cached(
                (projected[codebook] + previous)[None, None], cache
            )
            expected = model.depth_embeddings[codebook].get_logits(ref.squeeze())
            actual = runner.run(
                input_embedding=hidden[None].float(), previous_embedding=previous[None].float()
            )["logits"].flatten()
            order = torch.argsort(expected, descending=True, stable=True)
            entry = {
                "codebook": codebook,
                "reference_token": int(expected.argmax()),
                "native_token": int(actual.argmax()),
                "reference_margin": float(expected[order[0]] - expected[order[1]]),
                "metrics": metrics(actual, expected),
                "reference_native_choice_logit": float(expected[actual.argmax()]),
                "reference_best_logit": float(expected.max()),
            }
            entries.append(entry)
            previous = model.depth_embeddings[codebook](codes[codebook : codebook + 1]).squeeze()
        report.append({"voice": voice, "frame": frame_index, "codebooks": entries})
        print(json.dumps(report[-1]), flush=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
