"""Logit diff hooks owned by the Whisper family."""

from __future__ import annotations

import sys

import numpy as np


def handles_model_type(model_type: str) -> bool:
    return model_type.lower() == "whisper"


def attach_additional_plans(plugin, model_dir, config, weights, *, verbose: bool) -> None:
    build_vision = getattr(plugin, "build_vision_engine", None)
    if build_vision is None:
        return

    print("[diff] Building Whisper encoder engine ...", file=sys.stderr)
    encoder_plan = build_vision(model_dir, config, weights, verbose=verbose)
    if encoder_plan is None:
        return
    print(f"[diff] Encoder built ({len(encoder_plan) / 1e6:.1f} MB)",
          file=sys.stderr)
    config._encoder_plan = encoder_plan
    config._weights = weights


def make_trt_runner(engine_plan, config, max_cache_length):
    from tools.families.whisper.debug_runner import WhisperTrtRunner

    encoder_plan = getattr(config, "_encoder_plan", None)
    if encoder_plan is None:
        raise RuntimeError("Whisper encoder plan not built")

    raw = config.raw
    num_layers = raw.get("decoder_layers", config.num_hidden_layers)
    max_source = raw.get("max_source_positions", 1500)
    num_mel_bins = raw.get("num_mel_bins", 80)
    runner = WhisperTrtRunner(
        decoder_plan=engine_plan,
        encoder_plan=encoder_plan,
        num_layers=num_layers,
        max_cache_length=max_cache_length,
        max_source_positions=max_source,
        hidden_size=config.hidden_size,
    )

    mel_length = max_source * 2
    mel_features = np.zeros((num_mel_bins, mel_length), dtype=np.float32)
    print(f"  Running TRT encoder (mel={num_mel_bins}x{mel_length}) ...",
          file=sys.stderr)
    runner.run_encoder(mel_features)
    return runner


def load_hf_model(model_dir):
    import torch
    from transformers import WhisperForConditionalGeneration

    print("[diff] Loading Whisper model via WhisperForConditionalGeneration ...",
          file=sys.stderr)
    return WhisperForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.float32)


def run_hf(model, config, input_ids, max_new_tokens):
    """Run HF Whisper encoder-decoder with dummy mel."""
    import torch
    from transformers.modeling_outputs import BaseModelOutput

    raw = config.raw
    max_source = raw.get("max_source_positions", 1500)
    num_mel_bins = raw.get("num_mel_bins", 80)
    mel_length = max_source * 2

    mel_features = torch.zeros(1, num_mel_bins, mel_length, dtype=torch.float32)

    all_logits = []
    with torch.no_grad():
        encoder_outputs = model.model.encoder(mel_features)
        encoder_hidden = encoder_outputs.last_hidden_state
        enc_out = BaseModelOutput(last_hidden_state=encoder_hidden)

        past_key_values = None
        gen_ids = list(input_ids)

        for step_idx in range(len(input_ids) + max_new_tokens):
            if step_idx < len(input_ids):
                token = input_ids[step_idx]
            else:
                token = int(np.argmax(all_logits[-1]))
                gen_ids.append(token)

            ids_tensor = torch.tensor([[token]], dtype=torch.long)

            outputs = model(
                decoder_input_ids=ids_tensor,
                encoder_outputs=enc_out,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[0, -1].numpy()
            all_logits.append(logits)

    return all_logits


def prompt_cases(_prompts):
    start_ids = [50258, 50259, 50359, 50363]
    return [(
        "whisper-decode",
        f"[decoder start tokens: {start_ids}]",
        start_ids,
    )]
