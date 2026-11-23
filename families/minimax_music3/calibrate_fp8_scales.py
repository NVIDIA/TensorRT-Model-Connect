# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate the static FP8 activation scales of MiniMax-Music3.

Not used when building an engine. Run it once per supported checkpoint and commit the updated
``fp8_activation_scales.json``::

    python -m families.minimax_music3.calibrate_fp8_scales fetch --out prompts.jsonl
    python -m families.minimax_music3.calibrate_fp8_scales calibrate \\
        --model-dir <MiniMax-Music3 snapshot> --prompts prompts.jsonl

``fetch`` writes music descriptions (``google/MusicCaps`` captions) paired with song lyrics
(``theelderemo/genius-lyrics-cleaned``) and needs network access. ``calibrate`` loads the
language model, the RVQ depth decoder, the condition encoder and the diffusion transformer in
PyTorch bf16 (the reference precision), runs ModelOpt ``FP8_DEFAULT_CFG`` (max calibration)
while every prompt goes through the whole generation -- the autoregressive stage with
classifier-free guidance and top-k sampling, then every denoising window with all Euler steps
and both guidance branches -- and stores ``input_quantizer.amax / 448`` of every quantized
projection under the key of its engine (see ``quantization.scale_key``).

The loops below mirror the reference pipeline (diffusers ``modular_pipelines/minimax_music3``)
and the native runtime: the same prompt format, the same guidance and sampling, the same
window plan and sigma schedule. They exist so the calibration sees the activations of a real
generation, including the frame embeddings the language model reads after the prompt.

A projection whose input scale is more than 20 times the median of the same projection in the
other layers is stored as ``null`` and stays unquantized; a projection missing from the file is
an error at build time.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

from .quantization import ACTIVATION_SCALES_FILENAME, ENGINE_PROJECTIONS, FP8_E4M3_MAX, scale_key

_CAPTIONS = "google/MusicCaps"
_LYRICS = "theelderemo/genius-lyrics-cleaned"
_LYRIC_TAGS = ("pop", "rock", "country", "rb")
_PAGE = 20
_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset={dataset}"
    "&config=default&split=train&offset={offset}&length={length}"
)
_SECTION = re.compile(r"^\[([A-Za-z\- ]+?)(?:\s*\d+)?(?::[^\]]*)?\]\s*$")
_KNOWN_SECTIONS = ("intro", "verse", "pre-chorus", "chorus", "bridge", "outro", "interlude", "hook")
_MAX_LYRIC_LINES = 24
_OUTLIER_RATIO = 20.0

# Module-name patterns of the PyTorch reference, per engine, with the engine's own name for the
# projection (the weight names the TensorRT builders pass).
_MODULE_PATTERNS = {
    "language_model": (
        re.compile(
            r"^language_model\.model\.layers\.(\d+)\.(self_attn\.[qkvo]_proj|mlp\.(?:gate|up|down)_proj)$"
        ),
        lambda layer, stem: f"layer.{layer}.{_LM_STEMS[stem]}",
    ),
    "rvq_depth_decoder": (
        re.compile(
            r"^rvq_depth_decoder\.layers\.(\d+)\.(attn\.to_[qkv]|attn\.to_out|gate_proj|up_proj|down_proj)$"
        ),
        lambda layer, stem: f"layers.{layer}.{stem}",
    ),
    "transformer": (
        re.compile(
            r"^transformer\.transformer_blocks\.(\d+)\.(attn\.to_[qkv]|attn\.to_out\.0|ff_in|ff_out)$"
        ),
        lambda layer, stem: f"transformer_blocks.{layer}.{stem}",
    ),
}
_LM_STEMS = {
    "self_attn.q_proj": "w_q",
    "self_attn.k_proj": "w_k",
    "self_attn.v_proj": "w_v",
    "self_attn.o_proj": "w_o",
    "mlp.gate_proj": "w_gate",
    "mlp.up_proj": "w_up",
    "mlp.down_proj": "w_down",
}


def engine_projection_name(module_name: str) -> tuple[str, str] | None:
    """Map a reference module name to ``(engine, projection name)``, or None if not quantized."""
    for engine, (pattern, rename) in _MODULE_PATTERNS.items():
        match = pattern.match(module_name)
        if match is not None:
            return engine, rename(int(match.group(1)), match.group(2))
    return None


def split_outlier_scales(scales: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    """Split off projections whose input scale is far above the median of the same projection."""

    def stem(name: str) -> str:
        return re.sub(r"\.\d+\.", ".N.", name, count=1)

    by_stem: dict[str, list[float]] = {}
    for name, value in scales.items():
        by_stem.setdefault(stem(name), []).append(value)
    median = {key: statistics.median(values) for key, values in by_stem.items()}
    kept, excluded = {}, []
    for name, value in scales.items():
        if value > _OUTLIER_RATIO * median[stem(name)]:
            excluded.append(name)
        else:
            kept[name] = value
    return kept, sorted(excluded)


# --- prompts ---------------------------------------------------------------------------------


def _get_rows(dataset: str, offset: int) -> list:
    url = _ROWS_URL.format(dataset=dataset, offset=offset, length=_PAGE)
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                return [item["row"] for item in json.load(response)["rows"]]
        except urllib.error.HTTPError as error:
            if error.code != 429 and error.code < 500:
                raise
        except urllib.error.URLError:
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError("rows API kept failing")


def clean_lyrics(text: str) -> str | None:
    """Keep the first verses of a song with its section tags in the checkpoint's form."""
    lines, sections = [], 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _SECTION.match(line)
        if match:
            name = match.group(1).strip().lower()
            name = next((known for known in _KNOWN_SECTIONS if name.startswith(known)), None)
            if name is None:
                continue
            sections += 1
            lines.append(f"[{name}]")
        elif line.startswith("["):
            continue
        else:
            lines.append(line)
        if len(lines) >= _MAX_LYRIC_LINES:
            break
    if sections == 0 or len(lines) < 6 or not all(line.isascii() for line in lines):
        return None
    return "\n".join(lines)


def fetch_prompts(out: Path, count: int, start: int) -> None:
    """Write ``count`` caption and lyrics pairs as JSONL, from row ``start`` of both sets."""
    captions, offset = [], start
    while len(captions) < count:
        captions += [
            row["caption"].strip() for row in _get_rows(_CAPTIONS, offset) if row.get("caption")
        ]
        offset += _PAGE
    lyrics, offset = [], start
    while len(lyrics) < count:
        for row in _get_rows(_LYRICS, offset):
            if row.get("tag") in _LYRIC_TAGS:
                cleaned = clean_lyrics(row.get("lyrics") or "")
                if cleaned:
                    lyrics.append(cleaned)
        offset += _PAGE * 5  # spread over the corpus
    rows = [
        {"caption": caption, "lyrics": lyric, "seed": start + index}
        for index, (caption, lyric) in enumerate(zip(captions[:count], lyrics[:count]))
    ]
    out.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} prompts to {out}")


# --- reference generation (mirrors the reference pipeline) ----------------------------------


def load_reference(model_dir: Path, device: str = "cuda"):
    """Load the PyTorch components in bf16, as the reference pipeline runs them."""
    from types import SimpleNamespace

    import torch
    from diffusers import (
        MiniMaxMusic3ConditionEncoder,
        MiniMaxMusic3RVQDepthDecoder,
        MiniMaxMusic3Transformer1DModel,
        MiniMaxMusic3Vocoder,
    )
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    def load(cls, sub):
        return (
            cls.from_pretrained(str(Path(model_dir) / sub), torch_dtype=torch.bfloat16)
            .to(device)
            .eval()
        )

    return SimpleNamespace(
        tokenizer=AutoTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer")),
        language_model=Qwen3ForCausalLM.from_pretrained(
            str(Path(model_dir) / "language_model"), dtype=torch.bfloat16
        )
        .to(device)
        .eval(),
        rvq_depth_decoder=load(MiniMaxMusic3RVQDepthDecoder, "rvq_depth_decoder"),
        condition_encoder=load(MiniMaxMusic3ConditionEncoder, "condition_encoder"),
        transformer=load(MiniMaxMusic3Transformer1DModel, "transformer"),
        vocoder=load(MiniMaxMusic3Vocoder, "vocoder"),
        device=device,
    )


def _top_k_sample(logits, generator, top_k: int):
    import torch

    values = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    threshold = torch.topk(values, min(top_k, values.shape[-1]), dim=-1).values[..., -1, None]
    values = values.masked_fill(values < threshold, -float("inf"))
    probs = torch.softmax(values, dim=-1)
    return torch.multinomial(probs.cpu(), 1, generator=generator).squeeze(-1).to(logits.device)


def generate_frames(ref, caption: str, lyrics: str, frames: int, seed: int, record=None):
    """Autoregressive stage: return ``frame_hiddens`` ``[1, frames, 8 * 4096]``.

    ``record(step, **tensors)``, when given, receives every step's conditional-branch tensors:
    the language model input, its logits and hidden state, the frame codes and the depth
    decoder logits and hidden states.
    """
    import torch

    from . import prompt_format as pf
    from .depth_decoder import AUDIO_VOCAB_SIZE, NUM_CODEBOOKS

    lm, depth = ref.language_model, ref.rvq_depth_decoder
    ids = ref.tokenizer(pf.assemble_prompt(caption, lyrics), return_tensors="pt")["input_ids"]
    pf.check_prompt_length(int(ids.shape[1]))
    uncond = ids.clone()
    uncond[:, 1:-2] = pf.AUDIO_CFG_TOKEN_ID
    text_ids = torch.cat((ids, uncond)).to(ref.device)
    generator = torch.Generator().manual_seed(seed)
    vocab_mask = torch.ones(lm.config.vocab_size, dtype=torch.bool, device=ref.device)
    vocab_mask[pf.AUDIO_CODE_OFFSET : pf.AUDIO_CODE_OFFSET + pf.SEMANTIC_VOCAB_SIZE] = False
    vocab_mask[pf.AUDIO_END_TOKEN_ID] = False
    offsets = torch.arange(NUM_CODEBOOKS - 1, device=ref.device) * AUDIO_VOCAB_SIZE

    with torch.no_grad():
        output = lm.model(inputs_embeds=lm.model.embed_tokens(text_ids), use_cache=True)
        cache, last_hidden = output.past_key_values, output.last_hidden_state[:, -1]
        if record is not None:
            record(-1, prompt_ids=ids[0])
        hiddens = []
        for step in range(frames + 1):
            logits = lm.lm_head(last_hidden).float()
            masked = logits.masked_fill(vocab_mask, -float("inf"))
            cond, base = masked[0:1], masked[1:2]
            guided = base + (cond - base) * pf.AR_CFG_SCALE
            threshold = torch.topk(cond, pf.AR_CFG_TOP_K, dim=-1).values[..., -1, None]
            guided = guided.masked_fill(cond < threshold, -float("inf")).masked_fill(
                vocab_mask, -float("inf")
            )
            sampled = _top_k_sample(guided, generator, pf.AR_SAMPLING_TOP_K)
            if int(sampled.item()) == pf.AUDIO_END_TOKEN_ID:
                break
            semantic = (sampled - pf.AUDIO_CODE_OFFSET).repeat(2)
            # depth decoder: residual codes one codebook at a time
            sequence = [
                depth.projection(last_hidden).unsqueeze(1),
                depth.projection(lm.model.embed_tokens(semantic + pf.AUDIO_CODE_OFFSET)).unsqueeze(
                    1
                ),
            ]
            codes, parts, depth_logits = [semantic], [], []
            for index in range(1, NUM_CODEBOOKS):
                hidden = depth(torch.cat(sequence, dim=1))[:, -1]
                parts.append(hidden[:1])
                head = depth.audio_heads[index - 1](hidden).float()
                depth_logits.append(head[:1])
                code = _top_k_sample(
                    head[1:2] + (head[:1] - head[1:2]) * pf.AR_CFG_SCALE,
                    generator,
                    pf.AR_SAMPLING_TOP_K,
                ).repeat(2)
                codes.append(code)
                if index < NUM_CODEBOOKS - 1:
                    embed = depth.audio_embeddings(code + (index - 1) * AUDIO_VOCAB_SIZE)
                    sequence.append(depth.projection(embed).unsqueeze(1))
            frame_codes = torch.stack(codes, dim=1)
            depth_hidden = torch.cat(parts, dim=-1)
            feedback = lm.model.embed_tokens(frame_codes[:, :1] + pf.AUDIO_CODE_OFFSET)
            extra = depth.audio_embeddings(frame_codes[:, 1:] + offsets.unsqueeze(0)).sum(
                dim=1, keepdim=True
            )
            feedback = (feedback + extra.to(feedback.dtype)) * NUM_CODEBOOKS**-0.5
            if record is not None:
                record(
                    step,
                    logits=logits[0],
                    last_hidden=last_hidden[0],
                    codes=frame_codes[0],
                    depth_logits=torch.cat(depth_logits),
                    depth_hidden=depth_hidden[0],
                    feedback=feedback[0, 0],
                )
            if step > 0:
                hiddens.append(torch.cat((last_hidden[:1], depth_hidden), dim=-1))
                if len(hiddens) >= frames:
                    break
            output = lm.model(inputs_embeds=feedback, past_key_values=cache, use_cache=True)
            cache, last_hidden = output.past_key_values, output.last_hidden_state[:, -1]
    if not hiddens:
        raise RuntimeError("the prompt ended generation before the first frame")
    return torch.stack(hiddens, dim=1)


def timestep_schedule(steps: int) -> list[float]:
    """The timesteps the reference scheduler runs (``invert_sigmas``), then the final 1.0."""
    if steps == 1:
        return [0.0, 1.0]
    last = 1.0 / steps
    return [1.0 - (1.0 + (last - 1.0) / (steps - 1) * index) for index in range(steps)] + [1.0]


def denoise_window(velocity, condition, noise, *, steps: int, previous=None):
    """One window's flow-matching Euler loop with guidance, as the reference runs it.

    ``velocity(latents, timestep, condition)`` is the transformer; the unconditional branch
    conditions on zeros. ``previous`` is the carried tail of the previous window.
    """
    import torch

    from .pipeline_spec import GUIDANCE_SCALE

    latents = noise.clone()
    overlap = 0 if previous is None else min(previous.shape[-1], latents.shape[-1])
    prompt = latents[..., :overlap].clone()
    times = timestep_schedule(steps)
    silent = torch.zeros_like(condition)
    for index in range(steps):
        t, following = times[index], times[index + 1]
        if overlap:
            latents[..., :overlap] = (1.0 - (1.0 - 1e-6) * t) * prompt + t * previous[..., :overlap]
        timestep = torch.full((latents.shape[0],), t, device=latents.device, dtype=latents.dtype)
        cond = velocity(latents, timestep, condition)
        base = velocity(latents, timestep, silent)
        latents = latents + (following - t) * (base + GUIDANCE_SCALE * (cond - base))
    if overlap:
        latents[..., :overlap] = previous[..., :overlap]
    return latents


def denoise_all(ref, frame_hiddens, seed: int, steps: int = 30):
    """Denoising stage over every window of the reference window plan (PyTorch)."""
    import torch

    from .pipeline_spec import CHUNK_FRAMES, chunk_starts

    carry = 172  # reference _OVERLAP_LATENT_LENGTH
    generator = torch.Generator().manual_seed(seed)
    previous, previous_condition, chunks = None, None, []

    def velocity(latents, timestep, condition):
        return ref.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=condition,
            return_dict=False,
        )[0]

    with torch.no_grad():
        for start in chunk_starts(frame_hiddens.shape[1]).starts:
            condition = ref.condition_encoder(frame_hiddens[:, start : start + CHUNK_FRAMES]).to(
                torch.bfloat16
            )
            if previous is not None:
                overlap = min(previous.shape[-1], condition.shape[1])
                condition[:, :overlap] = previous_condition[:, :overlap]
            noise = torch.randn((1, 128, condition.shape[1]), generator=generator).to(
                ref.device, torch.bfloat16
            )
            latents = denoise_window(velocity, condition, noise, steps=steps, previous=previous)
            begin = max(0, latents.shape[-1] - 2 * carry)
            end = max(begin, latents.shape[-1] - carry)
            previous, previous_condition = latents[..., begin:end], condition[:, begin:end]
            chunks.append(latents)
    return chunks


# --- calibration -----------------------------------------------------------------------------


def read_fingerprint_tensor(model_dir: Path, component: str) -> dict:
    """Read the one stored tensor the scale key fingerprints, at its stored precision."""
    import torch
    from safetensors import safe_open

    from .quantization import _KEY_PARTS

    name = _KEY_PARTS[component][1]
    for shard in sorted((Path(model_dir) / component).glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:
            if name in handle.keys():
                return {name: handle.get_tensor(name).to(torch.float64).numpy()}
    raise FileNotFoundError(f"{component} has no {name}")


def calibrate(
    model_dir: Path, prompts: Path, frames: int, steps: int, report: Path | None = None
) -> None:
    """Run ModelOpt FP8 calibration over whole generations and store the activation scales."""
    import modelopt.torch.quantization as mtq
    import torch

    rows = [json.loads(line) for line in prompts.read_text(encoding="utf-8").splitlines() if line]
    ref = load_reference(model_dir)
    container = torch.nn.ModuleDict(
        {
            "language_model": ref.language_model,
            "rvq_depth_decoder": ref.rvq_depth_decoder,
            "transformer": ref.transformer,
        }
    )

    def forward_loop(model) -> None:
        ref.language_model = model["language_model"]
        ref.rvq_depth_decoder = model["rvq_depth_decoder"]
        ref.transformer = model["transformer"]
        for index, row in enumerate(rows):
            started = time.time()
            hiddens = generate_frames(ref, row["caption"], row["lyrics"], frames, int(row["seed"]))
            denoise_all(ref, hiddens, int(row["seed"]), steps)
            print(
                f"prompt {index + 1}/{len(rows)}: {hiddens.shape[1]} frames, {time.time() - started:.0f} s",
                flush=True,
            )

    mtq.quantize(container, mtq.FP8_DEFAULT_CFG, forward_loop)

    tensors = {engine: {} for engine in ENGINE_PROJECTIONS}
    for name, module in container.named_modules():
        mapped = engine_projection_name(name)
        if mapped is None:
            continue
        quantizer = getattr(module, "input_quantizer", None)
        amax = None if quantizer is None else quantizer.amax
        if amax is None or not float(amax) > 0:
            raise RuntimeError(f"{name} has no calibrated activation amax")
        tensors[mapped[0]][mapped[1]] = float(amax) / FP8_E4M3_MAX

    if report is not None:
        # every calibrated projection, including the ones the build does not quantize
        report.write_text(json.dumps(tensors, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    for engine, scales in tensors.items():
        expected = ENGINE_PROJECTIONS[engine]()
        missing = sorted(set(expected) - set(scales))
        if missing:
            raise RuntimeError(
                f"{engine}: no scale for {len(missing)} projections, e.g. {missing[0]}"
            )
        kept, excluded = split_outlier_scales({name: scales[name] for name in expected})
        for name in excluded:
            print(
                f"{engine}: leaving {name} unquantized, input scale {scales[name]:.4g} is a massive activation"
            )
        key = scale_key(engine, read_fingerprint_tensor(Path(model_dir), engine))
        table[key] = dict(sorted({**kept, **dict.fromkeys(excluded)}.items()))
        print(f"{key}: {len(kept)} scales, {len(excluded)} left unquantized")
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    """Parse the command line and run ``fetch`` or ``calibrate``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--out", type=Path, required=True)
    fetch.add_argument("--count", type=int, default=32)
    fetch.add_argument("--start", type=int, default=0, help="first row (held-out sets start later)")
    run = commands.add_parser("calibrate")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--prompts", type=Path, required=True)
    run.add_argument(
        "--frames", type=int, default=300, help="autoregressive frames per prompt (25 per second)"
    )
    run.add_argument("--steps", type=int, default=30, help="Euler steps per denoising window")
    run.add_argument(
        "--report", type=Path, default=None, help="also write every raw scale to this JSON file"
    )
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_prompts(args.out, args.count, args.start)
    else:
        calibrate(args.model_dir, args.prompts, args.frames, args.steps, args.report)


if __name__ == "__main__":
    main()
