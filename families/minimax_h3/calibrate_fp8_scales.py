# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate the static FP8 activation scales of the MiniMax-H3 DiT.

Not run by users. It runs real prompts through the PyTorch (Diffusers ModularPipeline, ``t2va``
workflow) bf16 pipeline at the native 1344x768x124 shape, records ``max(abs(input))`` of every
quantized projection of the 50 transformer blocks over all denoising steps and prompts, and
stores ``max / 448`` in ``fp8_activation_scales.json`` under the model-config key. The three
projection kinds are the fused query/key/value GEMM (input of ``to_q``, shared by ``to_k`` and
``to_v``), the gate/up GEMM ``ff.net.0.proj`` and the down GEMM ``ff.net.2``. The attention
output projection ``attn.to_out.0`` is only reported (it stays bf16 on TensorRT 11.4).

Usage::

    python -m families.minimax_h3.calibrate_fp8_scales --model-dir <MiniMax-H3 snapshot> \\
        --report report.json

The weights (about 140 GB with the text encoder) need one GPU with enough memory for the
transformer, the text encoder and the VAE (or ``--cpu-offload``).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .quantization import (
    ACTIVATION_SCALES_FILENAME,
    FP8_E4M3_MAX,
    QUANTIZED,
    scale_key,
)

HEIGHT, WIDTH, NUM_FRAMES, NUM_STEPS = 768, 1344, 124, 50

# module-name suffix recorded for each engine projection name stem; the extra entry is only reported
HOOKED = {
    "attn.qkv": "attn.to_q",
    "ff.net.0.proj": "ff.net.0.proj",
    "ff.net.2": "ff.net.2",
    "attn.to_out.0": "attn.to_out.0",
}

_BLOCK_MODULE = re.compile(r"^transformer_blocks\.(\d+)\.(.+)$")


def _shot(first: str, second: str, sound: str, music: str) -> str:
    return (
        f"integrated_multimodal_description: [Shot 1] {first} [Shot 2] {second}\n"
        f"overall_soundscape: {sound}\nnon_diegetic_music: {music}"
    )


# Eight scenes in the H3-Context-IR prompt format (two shots, soundscape, music).
CALIBRATION_PROMPTS = [
    _shot(
        "Wide cinematic shot of a caravan of camels crossing endless orange dunes under a pale "
        "morning sky, riders wrapped in blue scarves, wind lifting thin sand veils from the crests.",
        "At 00:05.000, the camera cuts to a close-up of a rider's weathered face squinting at the "
        "horizon as sand grains blow past and a hawk circles far above.",
        "Wind over sand, the soft padding of hooves, jingling harness bells and a distant hawk cry.",
        "Slow hand-drum rhythm with a haunting reed flute.",
    ),
    _shot(
        "Medium shot inside a cozy rainy-day cafe, a man in a wool sweater types on a laptop beside "
        "a fogged window, steaming coffee cup in front of him and blurred umbrellas passing outside.",
        "At 00:04.000, the barista places a fresh croissant on his table, and he looks up and smiles "
        "as raindrops streak down the glass behind him.",
        "Rain against the window, soft cafe chatter, a coffee machine hissing and the gentle clink "
        "of ceramics.",
        "Mellow jazz trio with brushed drums and upright bass.",
    ),
    _shot(
        "Close-up of a red fox in tall golden grass at sunrise, breath fogging in the cold, its ears "
        "twitching as dew sparkles on every blade.",
        "At 00:04.000, the fox pounces high into the grass in slow motion, and the camera follows its "
        "arc as seeds scatter in the backlight before it lands and trots away.",
        "Quiet morning birdsong, rustling grass, a soft thump on landing and a distant stream.",
        "Gentle piano with a light string ostinato.",
    ),
    _shot(
        "Time-lapse style aerial shot over a futuristic neon city at night, glowing traffic ribbons "
        "flowing between mirrored towers while holographic billboards flicker in the rain.",
        "At 00:05.000, the camera dives between two skyscrapers toward a crowded elevated walkway, "
        "umbrellas and neon reflections streaking across wet glass.",
        "Deep city hum, the whoosh of passing vehicles, rain hiss and faint electronic advertisements.",
        "Pulsing synthwave with a driving arpeggio and heavy sub-bass.",
    ),
    _shot(
        "Dramatic low-angle shot in a dim boxing gym, a fighter wraps her hands with white tape under "
        "a single hanging bulb, sweat on her forearms and a heavy bag swaying behind her.",
        "At 00:04.000, she strikes the bag with a rapid combination, the camera shaking slightly with "
        "each hit as chalk dust explodes into the light.",
        "Heavy thuds on leather, controlled breathing, the squeak of shoes and the creak of the chain.",
        "Tense minimal drums with a low brooding synth drone.",
    ),
    _shot(
        "Gentle macro shot of a honeybee landing on a purple lavender flower in a sunny cottage "
        "garden, pollen dusting its legs while soft focus roses sway in the background.",
        "At 00:05.000, the bee lifts off and the camera follows it past a stone birdbath where water "
        "droplets glitter, then settles on an old wooden gate with a watering can.",
        "Buzzing wings, gentle breeze in leaves, a trickle of water and a distant songbird.",
        "Light airy acoustic guitar with soft handclaps.",
    ),
    _shot(
        "Steady wide shot from a platform as a vintage steam train pulls into a misty station at "
        "dawn, thick white steam billowing over the tracks and lamps glowing amber.",
        "At 00:04.500, passengers in long coats step down onto the platform, the camera panning to a "
        "conductor checking a pocket watch as the whistle blows.",
        "Heavy chuffing of the engine, the hiss of steam, a long whistle and murmurs of passengers.",
        "Nostalgic accordion waltz with a soft clarinet.",
    ),
    _shot(
        "Slow orbital shot of an astronaut in a white suit floating outside a space station, Earth "
        "curving blue and white below, the visor reflecting the glowing planet.",
        "At 00:05.000, the camera moves close to the visor as the astronaut tightens a bolt with a "
        "handheld tool, a small spark drifting away into the dark.",
        "Muffled breathing inside a helmet, faint radio static and the dull tap of a tool on metal.",
        "Vast cinematic pads with a slow heartbeat-like pulse.",
    ),
]


class _Recorder:
    """Forward pre-hooks that keep the running ``max(abs(input))`` of each hooked module."""

    def __init__(self, transformer, num_layers: int):
        self.amax: dict[str, float] = {}
        self.per_call: dict[str, list[float]] = {}
        self.handles = []
        names = {
            f"transformer_blocks.{i}.{module}": f"transformer_blocks.{i}.{stem}"
            for i in range(num_layers)
            for stem, module in HOOKED.items()
        }
        found = set()
        for module_name, module in transformer.named_modules():
            if module_name in names:
                found.add(module_name)
                self.handles.append(
                    module.register_forward_pre_hook(self._hook(names[module_name]))
                )
        missing = sorted(set(names) - found)
        if missing:
            raise RuntimeError(
                f"transformer has no module for {missing[:4]} (and {len(missing)} more)"
            )

    def _hook(self, key: str):
        def hook(_module, args):
            value = float(args[0].detach().abs().amax().float())
            self.amax[key] = max(self.amax.get(key, 0.0), value)
            self.per_call.setdefault(key, []).append(value)

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def calibrate(model_dir: Path, *, num_prompts: int, steps: int, cpu_offload: bool, device: str):
    import torch
    from diffusers import ComponentsManager, ModularPipeline

    manager = None
    if cpu_offload:
        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(device=device)
    pipeline = ModularPipeline.from_pretrained(
        str(model_dir), workflow="t2va", components_manager=manager, local_files_only=True
    )
    pipeline.load_components(
        dtype=torch.bfloat16, pretrained_model_name_or_path=str(model_dir), local_files_only=True
    )
    if manager is None:
        pipeline = pipeline.to(device)
    transformer = pipeline.transformer
    num_layers = int(transformer.config.num_layers)
    recorder = _Recorder(transformer, num_layers)
    try:
        for index, prompt in enumerate(CALIBRATION_PROMPTS[:num_prompts]):
            print(f"[calibrate] prompt {index + 1}/{num_prompts}", flush=True)
            pipeline(
                prompt=prompt,
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                num_inference_steps=steps,
                generator=torch.Generator().manual_seed(1000 + index),
                output="videos",
                output_type="np",
            )
    finally:
        recorder.close()
    config = transformer.config
    return recorder, int(config.hidden_size), num_layers, int(config.ffn_dim)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--report", type=Path, help="write per-projection maxima and statistics")
    parser.add_argument("--num-prompts", type=int, default=len(CALIBRATION_PROMPTS))
    parser.add_argument("--steps", type=int, default=NUM_STEPS)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    recorder, hidden, layers, ffn = calibrate(
        args.model_dir,
        num_prompts=args.num_prompts,
        steps=args.steps,
        cpu_offload=args.cpu_offload,
        device=args.device,
    )
    scales = {
        name: value / FP8_E4M3_MAX
        for name, value in sorted(recorder.amax.items())
        if name.rsplit(".", 1)[0] != "x" and any(name.endswith(stem) for stem in QUANTIZED)
    }
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text()) if path.exists() else {}
    table[scale_key(hidden, layers, ffn)] = scales
    path.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    print(f"[calibrate] wrote {len(scales)} scales to {path}")
    if args.report:
        args.report.write_text(
            json.dumps(
                {
                    "amax": recorder.amax,
                    "calls": {name: len(values) for name, values in recorder.per_call.items()},
                    "prompts": args.num_prompts,
                    "steps": args.steps,
                },
                indent=1,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
