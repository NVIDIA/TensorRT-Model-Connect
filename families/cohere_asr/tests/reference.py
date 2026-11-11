# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent oracle from the exact checkpoint's supplied PyTorch implementation."""

import importlib.util
import json
from pathlib import Path
import sys
import types
import numpy as np
import torch
from safetensors.torch import load_file
from transformers.modeling_outputs import BaseModelOutput


def load_reference(directory: Path):
    package_name = "cohere_asr_checkpoint_reference"
    package = types.ModuleType(package_name)
    package.__path__ = [str(directory)]
    sys.modules[package_name] = package
    modules = {}
    for name in [
        "configuration_cohere_asr",
        "modeling_cohere_asr",
        "processing_cohere_asr",
        "tokenization_cohere_asr",
    ]:
        spec = importlib.util.spec_from_file_location(
            package_name + "." + name, directory / (name + ".py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    raw = json.loads((directory / "config.json").read_text())
    config = modules["configuration_cohere_asr"].CohereAsrConfig(**raw)
    with torch.device("meta"):
        model = modules["modeling_cohere_asr"].CohereAsrForConditionalGeneration(config)
    weights = load_file(str(directory / "model.safetensors"))
    model.load_state_dict(
        {k: v for k, v in weights.items() if not k.startswith("preprocessor.")},
        strict=True,
        assign=True,
    )
    model = model.float().eval()
    frontend = modules["processing_cohere_asr"].FilterbankFeatures(
        n_window_size=400, n_window_stride=160, nfilt=128, n_fft=512, pad_to=0, device="cpu"
    )
    frontend.window = weights["preprocessor.featurizer.window"].float()
    frontend.fb = weights["preprocessor.featurizer.fb"].float()
    tokenizer = modules["tokenization_cohere_asr"].CohereAsrTokenizer.from_pretrained(
        str(directory)
    )
    return model, frontend, tokenizer


def transcribe(model, frontend, tokenizer, audio: np.ndarray):
    prompt = tokenizer(model.build_prompt("ar"), add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        features, length = frontend(
            torch.from_numpy(audio.copy())[None], torch.tensor([len(audio)])
        )
        encoded, encoded_length = model.encoder(features, length)
        prefix = list(prompt)
        generated = []
        # The native decoder recomputes the prefix; the reference follows the
        # same greedy equation independently, without using native outputs.
        for _ in range(256 - len(prompt)):
            values = model(
                input_ids=torch.tensor([prefix]),
                encoder_outputs=BaseModelOutput(last_hidden_state=encoded),
                length=length,
            )
            next_id = int(values.logits[0, -1].argmax())
            if next_id == 3:
                break
            generated.append(next_id)
            prefix.append(next_id)
        else:
            raise AssertionError("Official reference did not reach EOS")
    return {
        "text": tokenizer.decode(generated, skip_special_tokens=True),
        "token_ids": generated,
        "prompt_ids": prompt,
        "features": features[0, :, : int(length.item())].numpy(),
        "encoded": encoded[:, : int(encoded_length.item())].numpy(),
    }
