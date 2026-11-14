# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate the FP8 activation scales of the Lance understanding decoder.

Not run by users. It runs real image-text and text-only prompts through a BF16
PyTorch copy of the understanding decoder (the same graph the engine builds:
Qwen2 attention with QKV bias, per-head QK-norm, M-RoPE, SwiGLU) and records the
absolute maximum of the input of every projection that the FP8 build
quantizes. The scale of a projection is ``amax / 448`` (static, per tensor).
A projection whose scale is above ``OUTLIER_RATIO`` times the median of its
kind is written to ``excluded`` and stays unquantized.

Image features come from the Qwen2.5-VL ViT that ships with Lance, so the
decoder sees the same kind of input as at run time. The output is
``fp8_activation_scales.json``, keyed by model config (``scale_key``).

    python -m families.lance.calibrate_fp8_scales calibrate \
        --model-dir <Lance checkpoint> --image-parquet <flickr30k shard> \
        --text-prompts <jsonl with chat messages> --output <json>
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
from pathlib import Path

from .quantization import ACTIVATION_SCALES_FILENAME, FP8_E4M3_MAX, PROJECTIONS, scale_key

OUTLIER_RATIO = 20.0
IMAGE_SIZE = 448
MAX_TOKENS = 380
IMAGE_TOKEN = "<|video_pad|>"
IMAGE_TOKEN_ID = 151656
IMAGE_PAD_COUNT = 256
GRID = 16
LLM_PREFIX = "language_model.model"
LM_HEAD_KEY = "language_model.lm_head.weight"
# Short instructions an image-text user might give; the caption of the image is
# appended as the assistant answer so decode-time activations are covered.
QUESTIONS = (
    "Describe this image in detail.",
    "What is shown in this image?",
    "Write a short caption for this image.",
    "What color is the main object in this image? Answer in one word.",
    "How many people are in the image?",
    "Where was this photo taken?",
    "What is happening in this picture?",
    "Describe the scene briefly.",
)
PROMPT_TEMPLATE = (
    "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n"
    "<|vision_start|>{image_pads}<|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"
)
TEXT_TEMPLATE = (
    "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
)


def mrope_positions(input_ids, image_token_id=IMAGE_TOKEN_ID, grid=GRID):
    """Per-token (t, h, w) positions, the same rule as the native runtime."""
    n = len(input_ids)
    if image_token_id not in input_ids:
        return [(i, i, i) for i in range(n)]
    begin = input_ids.index(image_token_id)
    end = begin + grid * grid
    out = [(i, i, i) for i in range(begin)]
    out += [(begin, begin + i // grid, begin + i % grid) for i in range(grid * grid)]
    pos = begin + grid
    for _ in range(end, n):
        out.append((pos, pos, pos))
        pos += 1
    return out


class LanceReferenceDecoder:
    """BF16 PyTorch copy of the understanding decoder with projection-input hooks."""

    def __init__(self, model_dir: Path, device: str = "cuda"):
        import torch
        from safetensors import safe_open

        self.torch = torch
        self.device = device
        config = json.loads((model_dir / "llm_config.json").read_text(encoding="utf-8"))
        self.config = config
        self.layers = int(config["num_hidden_layers"])
        self.heads = int(config["num_attention_heads"])
        self.kv_heads = int(config["num_key_value_heads"])
        self.hidden = int(config["hidden_size"])
        self.head_dim = self.hidden // self.heads
        self.eps = float(config["rms_norm_eps"])
        self.theta = float(config["rope_theta"])
        self.mrope_section = list(config["rope_scaling"]["mrope_section"])
        self.weights = {}
        with safe_open(str(model_dir / "model.safetensors"), framework="pt", device=device) as f:
            wanted = [k for k in f.keys() if k.startswith(LLM_PREFIX) and "moe_gen" not in k]
            wanted.append(LM_HEAD_KEY)
            for key in wanted:
                self.weights[key[len(LLM_PREFIX) + 1 :] if key.startswith(LLM_PREFIX) else key] = (
                    f.get_tensor(key)
                )
        self.amax: dict[str, float] = {}
        self.recording = False

    def _norm(self, x, w):
        torch = self.torch
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * w

    def _linear(self, x, name, weight_key, bias_key=None):
        if self.recording:
            value = float(x.detach().abs().max())
            self.amax[name] = max(self.amax.get(name, 0.0), value)
        w = self.weights[weight_key]
        b = self.weights.get(bias_key) if bias_key else None
        return self.torch.nn.functional.linear(x, w, b)

    def _rope(self, x, cos, sin):
        torch = self.torch
        half = x.shape[-1] // 2
        rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * cos + rot * sin

    def _cos_sin(self, positions):
        torch = self.torch
        pos = torch.tensor(positions, device=self.device, dtype=torch.float32)  # [T, 3]
        inv = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.head_dim, 2, device=self.device, dtype=torch.float32)
                / self.head_dim
            )
        )
        axis = torch.cat(
            [
                torch.full((s,), a, device=self.device, dtype=torch.long)
                for a, s in enumerate(self.mrope_section)
            ]
        )
        freqs = pos[:, axis] * inv[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[:, None, :], emb.sin()[:, None, :]

    @property
    def embedding(self):
        return self.weights["embed_tokens.weight"]

    def forward(self, embeds, positions):
        """embeds [T, hidden] bf16, positions [T][3]; returns logits [T, vocab] float32."""
        torch = self.torch
        t = embeds.shape[0]
        cos, sin = self._cos_sin(positions)
        cos, sin = cos.to(embeds.dtype), sin.to(embeds.dtype)
        x = embeds
        for i in range(self.layers):
            p = f"layers.{i}."
            w = self.weights
            h = self._norm(x, w[p + "input_layernorm.weight"])
            q = self._linear(
                h, f"layer.{i}.w_q", p + "self_attn.q_proj.weight", p + "self_attn.q_proj.bias"
            )
            k = self._linear(
                h, f"layer.{i}.w_k", p + "self_attn.k_proj.weight", p + "self_attn.k_proj.bias"
            )
            v = self._linear(
                h, f"layer.{i}.w_v", p + "self_attn.v_proj.weight", p + "self_attn.v_proj.bias"
            )
            q = self._norm(q.view(t, self.heads, self.head_dim), w[p + "self_attn.q_norm.weight"])
            k = self._norm(
                k.view(t, self.kv_heads, self.head_dim), w[p + "self_attn.k_norm.weight"]
            )
            v = v.view(t, self.kv_heads, self.head_dim)
            q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
            a = (
                torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(0, 1)[None],
                    k.transpose(0, 1)[None],
                    v.transpose(0, 1)[None],
                    is_causal=True,
                    enable_gqa=True,
                )[0]
                .transpose(0, 1)
                .reshape(t, self.heads * self.head_dim)
            )
            x = x + self._linear(a, f"layer.{i}.w_o", p + "self_attn.o_proj.weight")
            h = self._norm(x, w[p + "post_attention_layernorm.weight"])
            g = self._linear(h, f"layer.{i}.w_gate", p + "mlp.gate_proj.weight")
            u = self._linear(h, f"layer.{i}.w_up", p + "mlp.up_proj.weight")
            x = x + self._linear(
                torch.nn.functional.silu(g) * u, f"layer.{i}.w_down", p + "mlp.down_proj.weight"
            )
        x = self._norm(x, self.weights["norm.weight"])
        return torch.nn.functional.linear(
            x, self.weights.get("lm_head.weight", self.weights["embed_tokens.weight"])
        ).float()


def load_vision_tower(model_dir: Path, device: str = "cuda"):
    """Return ``encode(pil_image) -> [256, hidden]`` bf16 features from the Lance ViT."""
    import torch
    from safetensors.torch import load_file
    from transformers import Qwen2VLImageProcessor
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionTransformerPretrainedModel,
    )

    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionConfig
    except ImportError:  # newer transformers keep the config in its own module
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
            Qwen2_5_VLVisionConfig,
        )
    vision_dir = model_dir.parent / "Qwen2.5-VL-ViT"
    config = Qwen2_5_VLVisionConfig.from_pretrained(vision_dir)
    model = Qwen2_5_VisionTransformerPretrainedModel(config)
    model.load_state_dict(load_file(vision_dir / "vit.safetensors"), strict=True)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    processor = Qwen2VLImageProcessor()

    def encode(image):
        from PIL import Image

        image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
        enc = processor(images=[image], return_tensors="pt")
        with torch.no_grad():
            feats = model(
                enc["pixel_values"].to(device=device, dtype=torch.bfloat16),
                enc["image_grid_thw"].to(device=device),
            )
        feats = getattr(feats, "pooler_output", feats)  # transformers 5 wraps the output
        return feats.to(torch.bfloat16)

    return encode


def load_image_samples(parquet: Path, count: int, offset: int = 0):
    """Return (PIL image, caption) pairs from a flickr30k-style parquet shard."""
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(parquet, columns=["image", "caption"])
    out = []
    for i in range(offset, min(offset + count, table.num_rows)):
        row = table.slice(i, 1).to_pylist()[0]
        caption = row["caption"]
        caption = caption[0] if isinstance(caption, list) else str(caption)
        out.append((Image.open(io.BytesIO(row["image"]["bytes"])), caption))
    return out


def load_text_prompts(path: Path, count: int):
    """Return ``(user, assistant)`` text pairs from a jsonl file of chat messages."""
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        messages = json.loads(line)["messages"]
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        assistant = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        if len(user) + len(assistant) > 8:
            pairs.append((user, assistant))
    return pairs[:count]


def embed_image_text(ref, tokenizer, features, question, answer):
    """Token ids, embeds and M-RoPE positions of one image-text sample."""
    text = PROMPT_TEMPLATE.format(image_pads=IMAGE_TOKEN * IMAGE_PAD_COUNT, prompt=question)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    ids += tokenizer(answer, add_special_tokens=False)["input_ids"][: max(0, MAX_TOKENS - len(ids))]
    return _embed(ref, ids, features)


def embed_text(ref, tokenizer, user, assistant):
    """Token ids, embeds and positions of one text-only sample."""
    ids = tokenizer(TEXT_TEMPLATE.format(prompt=user), add_special_tokens=False)["input_ids"]
    ids = ids[: MAX_TOKENS // 2]
    ids += tokenizer(assistant, add_special_tokens=False)["input_ids"][: MAX_TOKENS - len(ids)]
    return _embed(ref, ids, None)


def _embed(ref, ids, features):
    import torch

    t = torch.tensor(ids, device=ref.device)
    embeds = ref.embedding[t].clone()
    if features is not None:
        mask = t == IMAGE_TOKEN_ID
        embeds[mask] = features.to(embeds.dtype)
    return ids, embeds, mrope_positions(ids)


def scales_from_amax(amax: dict[str, float], layers: int) -> tuple[dict, list]:
    """Turn recorded amax values into scales and the list of outlier projections to exclude."""
    scales, excluded = {}, []
    for _, stem in PROJECTIONS:
        names = [f"layer.{i}.{stem}" for i in range(layers)]
        missing = [n for n in names if amax.get(n, 0.0) <= 0.0]
        if missing:
            raise RuntimeError(f"no calibration data for {missing[:3]} ... ({len(missing)} total)")
        median = statistics.median(amax[n] for n in names)
        for n in names:
            scales[n] = float(amax[n] / FP8_E4M3_MAX)
            if amax[n] > OUTLIER_RATIO * median:
                excluded.append(n)
    return scales, sorted(excluded, key=lambda n: (int(n.split(".")[1]), n))


def load_tokenizer(model_dir: Path):
    """The Lance_3B tokenizer.json (the directory has no config.json for AutoTokenizer)."""
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(tokenizer_file=str(model_dir / "tokenizer.json"))


def calibrate(args) -> None:
    import torch

    model_dir = (
        Path(args.model_dir) / "Lance_3B"
        if (Path(args.model_dir) / "Lance_3B").is_dir()
        else Path(args.model_dir)
    )
    tokenizer = load_tokenizer(model_dir)
    ref = LanceReferenceDecoder(model_dir)
    encode = load_vision_tower(model_dir)
    ref.recording = True
    n_image = n_text = 0
    with torch.no_grad():
        for image, caption in load_image_samples(Path(args.image_parquet), args.image_samples):
            feats = encode(image)
            question = QUESTIONS[n_image % len(QUESTIONS)]
            _, embeds, pos = embed_image_text(ref, tokenizer, feats, question, caption)
            ref.forward(embeds, pos)
            n_image += 1
        for user, assistant in load_text_prompts(Path(args.text_prompts), args.text_samples):
            _, embeds, pos = embed_text(ref, tokenizer, user, assistant)
            ref.forward(embeds, pos)
            n_text += 1
    scales, excluded = scales_from_amax(ref.amax, ref.layers)

    class _Cfg:
        model_type = "lance"
        hidden_size = ref.hidden
        num_hidden_layers = ref.layers
        intermediate_size = int(ref.config["intermediate_size"])

    out = Path(args.output)
    table = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    table[scale_key(_Cfg)] = {
        "calibration": f"{n_image} image-text and {n_text} text samples, max calibration",
        "input_scales": scales,
        "excluded": excluded,
    }
    out.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(scales)} scales, excluded {excluded}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--model-dir", required=True)
    cal.add_argument("--image-parquet", required=True)
    cal.add_argument("--text-prompts", required=True)
    cal.add_argument("--image-samples", type=int, default=320)
    cal.add_argument("--text-samples", type=int, default=192)
    cal.add_argument("--output", default=str(Path(__file__).parent / ACTIVATION_SCALES_FILENAME))
    cal.set_defaults(func=calibrate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
