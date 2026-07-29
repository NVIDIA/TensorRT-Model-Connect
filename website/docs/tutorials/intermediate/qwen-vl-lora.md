---
title: Qwen2.5-VL LoRA Adapters
---

This tutorial builds one Qwen2.5-VL bundle that can run either the base
model or a PEFT LoRA adapter selected at runtime. It uses the public
[ComicsPAP adapter](https://huggingface.co/VLR-CVC/Qwen2.5-VL-3B-Instruct-lora-ComicsPAP),
which is trained from `Qwen/Qwen2.5-VL-3B-Instruct` to select the next
comic panel from four candidates.

The adapter has rank 8 and targets `q_proj` and `v_proj`. Those properties
determine the LoRA inputs compiled into the TensorRT decoder.

## Download the adapter

Install the sample dependencies and download the adapter checkpoint:

```bash
python3 -m pip install -U datasets huggingface_hub pillow

hf download \
  VLR-CVC/Qwen2.5-VL-3B-Instruct-lora-ComicsPAP \
  --local-dir /tmp/comicspap-lora
```

The base model can remain a Hugging Face model ID. TMC resolves and
snapshots it during the build.

## Prepare a public sample

The following script selects public validation sample `dhgdlkfamu`, creates
the four-panel contact sheet expected by the adapter, and writes its prompt:

```bash
python3 - <<'PY'
from pathlib import Path

from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont


sample = next(
    item
    for item in load_dataset(
        "VLR-CVC/ComicsPAP",
        "caption_relevance",
        split="val",
        streaming=True,
    )
    if item["sample_id"] == "dhgdlkfamu"
)

output = Path("/tmp/comicspap-sample")
output.mkdir(parents=True, exist_ok=True)

panels = []
for image in sample["options"]:
    panel = image.convert("RGB")
    ratio = min(1.0, 500.0 / max(panel.size))
    panels.append(
        panel.resize(
            (int(panel.width * ratio), int(panel.height * ratio)),
            Image.Resampling.LANCZOS,
        )
    )

widths = sorted(panel.width for panel in panels)
heights = sorted(panel.height for panel in panels)
panel_width = (widths[1] + widths[2]) // 2
panel_height = (heights[1] + heights[2]) // 2
panels = [
    panel.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
    for panel in panels
]

margin = 10
label_height = 28
sheet = Image.new(
    "RGB",
    (panel_width * 4 + margin * 3, panel_height + label_height),
    "white",
)
draw = ImageDraw.Draw(sheet)
label_font = ImageFont.load_default(size=20)
number_font = ImageFont.load_default(size=24)
label_box = draw.textbbox((0, 0), "Options", font=label_font)
draw.text(
    ((sheet.width - (label_box[2] - label_box[0])) // 2, 2),
    "Options",
    fill="black",
    font=label_font,
)

for index, panel in enumerate(panels):
    x = index * (panel_width + margin)
    y = label_height
    sheet.paste(panel, (x, y))
    number_box = draw.textbbox((0, 0), str(index), font=number_font)
    number_width = number_box[2] - number_box[0]
    number_height = number_box[3] - number_box[1]
    number_x = x + panel_width - number_width - 4
    number_y = y + panel_height - number_height - 4
    draw.rectangle(
        (
            number_x - 2,
            number_y - 2,
            number_x + number_width + 2,
            number_y + number_height + 2,
        ),
        fill="white",
    )
    draw.text(
        (number_x, number_y),
        str(index),
        fill="black",
        font=number_font,
    )

sheet.save(output / "options_contact_sheet.png")

prompt = (
    "Pick A Panel Task: In the image you have a row of comic panels. "
    "From the options pick the panel that best follows the context caption. "
    "You must return your final answer as a number with "
    "'answer: <your answer here>'\n\n"
    f"context: {sample['previous_panel_caption']}"
)
(output / "prompt.txt").write_text(prompt, encoding="utf-8")
print(f"sample={sample['sample_id']} expected={sample['solution_index']}")
PY
```

This sample uses zero-based option labels and has expected answer `0`.

## Build a LoRA-capable bundle

The example profile accepts smart-resized images from 448 × 448 pixels up
to about one megapixel. Its 1,536-token decoder capacity covers this sample,
including 1,045 merged image tokens.

```bash
TRTMC=./build/trtmc

$TRTMC build Qwen/Qwen2.5-VL-3B-Instruct \
  -o /tmp/qwen25vl-3b-comicspap.trtfb \
  --precision fp16 \
  --max-cache-length 1536 \
  --decoder-engine-layout split \
  --set qwen_vl_decoder.decode_attention=decomposed \
  --set qwen_vl_lora.enabled=true \
  --set qwen_vl_lora.max_rank=8 \
  --set qwen_vl_lora.target_modules=q_proj,v_proj \
  --set qwen_vl_vision.dynamic_resolution=true \
  --set qwen_vl_vision.min_pixels=200704 \
  --set qwen_vl_vision.opt_pixels=819280 \
  --set qwen_vl_vision.max_pixels=1003520
```

The split layout builds a native-attention prefill engine and an independent
decode engine. Selecting decomposed decode uses FP32 attention scores only
for the single-token decode graph, without materializing decomposed attention
scores for the full prompt.

Dynamic image resolution preserves the aspect ratio and Qwen smart-resize
layout used to train the adapter. Choose `max_pixels` and
`max-cache-length` from the largest input that the application must serve.
Larger profiles increase build time and memory use.

## Compare the base model and adapter

First run the LoRA-capable bundle without selecting an adapter. LoRA inputs
default to zero, so this executes the base model:

```bash
PROMPT="$(cat /tmp/comicspap-sample/prompt.txt)"

$TRTMC run /tmp/qwen25vl-3b-comicspap.trtfb \
  --prompt "$PROMPT" \
  --image /tmp/comicspap-sample/options_contact_sheet.png \
  --max-new-tokens 16 \
  --temperature 0
```

Select the adapter at runtime without rebuilding the bundle:

```bash
$TRTMC run /tmp/qwen25vl-3b-comicspap.trtfb \
  --prompt "$PROMPT" \
  --image /tmp/comicspap-sample/options_contact_sheet.png \
  --max-new-tokens 16 \
  --temperature 0 \
  --lora-adapter /tmp/comicspap-lora \
  --lora-adapter-id comicspap
```

For this public sample, greedy decoding produces `answer: 0` with the
adapter. The base and adapter runs use the same TensorRT bundle; only the
runtime-bound LoRA weights change.

## Adapter compatibility

A runtime adapter must satisfy the contract compiled into the bundle:

| Adapter property | Bundle requirement |
| --- | --- |
| Base model | Same Qwen2.5-VL architecture and layer dimensions |
| Rank | Less than or equal to `qwen_vl_lora.max_rank` |
| Target modules | A subset of `qwen_vl_lora.target_modules` |
| Format | PEFT adapter config plus safetensors weights |

Building with a larger maximum rank or more target modules makes one bundle
accept more adapters, but also adds more runtime input buffers. Prefer the
smallest contract that covers the adapters the application will load.
