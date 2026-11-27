# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Separate original-prefill drift from same-state native one-row decode drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor

from families.lfm2_audio.tests.component_parity import Runner, metrics


def inject(runner, cache, layer_types):
    position = cache.get_seq_length()
    attention = convolution = 0
    for index, kind in enumerate(layer_types):
        if kind == "conv":
            runner.tensors[f"conv_state_{convolution}"].copy_(cache.conv_cache[index][0])
            convolution += 1
        else:
            runner.tensors[f"cache_k_{attention}"][:, :, :position].copy_(cache.key_cache[index])
            runner.tensors[f"cache_v_{attention}"][:, :, :position].copy_(cache.value_cache[index])
            attention += 1
    runner.position = position


def run_decode(runner, embedding):
    return runner.run(
        input_embedding=embedding.reshape(1, -1).float(),
        decode_phase=torch.ones((1, 1), device="cuda", dtype=torch.int32),
    )


def frame(runner, model, hidden):
    runner.reset()
    previous = torch.zeros((1, 1024), device="cuda", dtype=torch.float32)
    codes = []
    for codebook in range(8):
        out = runner.run(input_embedding=hidden.reshape(1, -1).float(), previous_embedding=previous)
        token = out["logits"].argmax().reshape(1)
        codes.append(int(token))
        previous = model.depth_embeddings[codebook](token).reshape(1, 1024).float()
    return torch.tensor([2048] * 8 if codes[0] == 2048 else codes, device="cuda", dtype=torch.long)


def audio_embedding(model, codes):
    return model.audio_embedding(codes + model.codebook_offsets).sum(0)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--depth", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    model = LFM2AudioModel.from_pretrained(Path(args.checkpoint)).eval()
    processor = LFM2AudioProcessor.from_pretrained(Path(args.checkpoint)).eval()
    native = Runner(args.backbone)
    fresh = Runner(args.backbone)
    depth = Runner(args.depth)
    references = {
        item["name"]: item["turns"][0]
        for item in json.loads(Path(args.reference).read_text())
        if item["name"].startswith("tts-")
    }
    report = []
    for voice in ("us_male", "us_female", "uk_male", "uk_female"):
        title = {
            "us_male": "US male",
            "us_female": "US female",
            "uk_male": "UK male",
            "uk_female": "UK female",
        }[voice]
        chat = ChatState(processor)
        chat.new_turn("system")
        chat.add_text(f"Perform TTS. Use the {title} voice.")
        chat.end_turn()
        chat.new_turn("user")
        chat.add_text("Hello there. How can I help you today?")
        chat.end_turn()
        chat.new_turn("assistant")
        inputs = model._prefill(**chat)
        initial = model.lfm(inputs_embeds=inputs, use_cache=True)
        cache = initial.past_key_values
        native.reset()
        inject(native, cache, model.conf.lfm.layer_types)
        hidden = initial.last_hidden_state[0, -1]
        logits = torch.nn.functional.linear(hidden, model.lfm.embed_tokens.weight)
        audio = False
        own_codes = []
        own_text = []
        for _ in range(128):
            if not audio:
                token = int(logits.argmax())
                own_text.append(token)
                if token == 7:
                    break
                if token == 128:
                    audio = True
                embedding = model.lfm.embed_tokens(torch.tensor(token, device="cuda"))
            else:
                codes = frame(depth, model, hidden)
                own_codes.append(codes.cpu().tolist())
                if int(codes[0]) == 2048:
                    audio = False
                embedding = audio_embedding(model, codes)
            outputs = run_decode(native, embedding)
            hidden, logits = outputs["hidden_state"].flatten(), outputs["logits"].flatten()
        expected = references["tts-" + voice]
        expected_codes = torch.tensor(expected["codes"], device="cuda").T
        assert expected["tokens"] == [128, 7], (
            "this diagnostic requires the complete single-span TTS reference"
        )
        own_codes_by_book = (
            torch.tensor(own_codes).T.tolist() if own_codes else [[] for _ in range(8)]
        )
        native.reset()
        inject(native, cache, model.conf.lfm.layer_types)
        fresh.reset()
        rows = []
        embedding = model.lfm.embed_tokens(torch.tensor(128, device="cuda"))
        for index, expected_frame in enumerate(expected_codes):
            inject(fresh, cache, model.conf.lfm.layer_types)
            exact_state = run_decode(fresh, embedding)["hidden_state"].flatten()
            continuous = run_decode(native, embedding)["hidden_state"].flatten()
            reference = model.lfm(
                inputs_embeds=embedding[None, None], past_key_values=cache, use_cache=True
            )
            cache = reference.past_key_values
            ref_hidden = reference.last_hidden_state[0, -1]
            ref_codes = model._sample_audio_frame(ref_hidden, temperature=0.0, top_k=1)
            if int(ref_codes[0]) == 2048:
                ref_codes[:] = 2048
            assert torch.equal(ref_codes, expected_frame), (
                voice,
                index,
                ref_codes.tolist(),
                expected_frame.tolist(),
            )
            row = {
                "frame": index,
                "same_state_hidden": metrics(exact_state, ref_hidden),
                "continuous_hidden": metrics(continuous, ref_hidden),
            }
            for name, value in (("same_state", exact_state), ("continuous", continuous)):
                sampled = model._sample_audio_frame(
                    value.to(torch.bfloat16), temperature=0.0, top_k=1
                )
                if int(sampled[0]) == 2048:
                    sampled[:] = 2048
                row[name + "_original_depth_codes_match"] = bool(
                    torch.equal(sampled, expected_frame)
                )
            rows.append(row)
            embedding = audio_embedding(model, expected_frame)
        item = {
            "voice": voice,
            "original_prefill_native_generation_text_match": own_text == expected["tokens"],
            "original_prefill_native_generation_codes_match": own_codes_by_book
            == expected["codes"],
            "native_frames": len(own_codes),
            "reference_frames": int(expected_codes.shape[0]),
            "steps": rows,
        }
        report.append(item)
        print(json.dumps({key: value for key, value in item.items() if key != "steps"}), flush=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
