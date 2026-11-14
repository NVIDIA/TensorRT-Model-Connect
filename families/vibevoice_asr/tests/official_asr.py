# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official VibeVoice-ASR transcription of one audio file, greedy like the official demo.

Runs under the reference interpreter that ``prepare_reference.py`` creates
(transformers 4.57, the checkpoint's own ``vibevoice`` source), never under the
repository environment. It mirrors ``demo/vibevoice_asr_inference_from_file.py``:
``VibeVoiceASRProcessor`` reads and resamples the file, the model generates
with ``do_sample=False`` until the tokenizer's end-of-text token, and the ids
are decoded with special tokens skipped.

``--acoustic mean`` replaces the sigma-VAE sampling of the acoustic latents
(``VibeVoiceTokenizerEncoderOutput.sample``, which adds noise with a random
scale on every call) by its mean, the deterministic mode transformers exposes
as ``sample=False``. ``--acoustic sampled`` keeps the official noise under
``--seed``.
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--context-info", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--acoustic", choices=("mean", "sampled"), default="mean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.source_dir)
    import torch
    import transformers
    from vibevoice.modular import modular_vibevoice_tokenizer
    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

    if args.acoustic == "mean":

        def mean(self, dist_type="fix"):
            return self.mean, self.std

        modular_vibevoice_tokenizer.VibeVoiceTokenizerEncoderOutput.sample = mean
    torch.manual_seed(args.seed)

    processor = VibeVoiceASRProcessor.from_pretrained(
        args.model_dir, language_model_pretrained_name=args.tokenizer_dir
    )
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    model.eval()
    inputs = processor(
        audio=[args.audio],
        sampling_rate=None,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
        context_info=args.context_info,
    )
    inputs = {
        key: value.to("cuda") if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    eos = processor.tokenizer.eos_token_id
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=processor.pad_id,
            eos_token_id=eos,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
        )
    prompt = inputs["input_ids"][0].tolist()
    generated = output.sequences[0, len(prompt) :].tolist()
    # The five best logits of every step, so a parity check can tell a tie
    # (equal logits that greedy argmax breaks by token id) from a real difference.
    top = [torch.topk(step[0].float(), 5) for step in output.logits]
    top_logits = [{"ids": item.indices.tolist(), "values": item.values.tolist()} for item in top]
    stopped = eos in generated
    if stopped:
        generated = generated[: generated.index(eos)]
    text = processor.decode(generated, skip_special_tokens=True)
    result = {
        "prompt_ids": prompt,
        "generated_ids": generated,
        "top_logits": top_logits[: len(generated) + 1],
        "stopped_at_eos": stopped,
        "eos_id": eos,
        "text": text,
        "segments": processor.post_process_transcription(text),
        "speech_samples": int(inputs["speech_tensors"].shape[-1]),
        "acoustic": args.acoustic,
        "seed": args.seed,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(0),
        },
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
