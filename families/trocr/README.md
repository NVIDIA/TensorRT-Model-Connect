# TrOCR

This family builds TrOCR (`microsoft/trocr-base-printed`): a ViT image encoder
and the TrOCR text decoder, published as `VisionEncoderDecoderModel`. It reads
one line of text from one image through the `images_text_to_text` Task. The
bundle holds two TensorRT engines: the encoder, which also projects its output
through every decoder layer's cross-attention keys and values once per image,
and a one-token decoder over a self-attention cache the runtime fills. Builds
are float32, the precision the checkpoint is stored in.

```bash
python -m tensorrt_model_connect build microsoft/trocr-base-printed \
  --revision 93450be3f1ed40a930690d951ef3932687cc1892 -o trocr.bundle
trtmc run trocr.bundle --image line.png --prompt ""
```

## Contract

| Item | Behavior |
|---|---|
| Input | exactly one image (RGB, RGBA or grayscale; uint8, or float32 in [0, 1]) and empty text parts. TrOCR takes no instruction; a non-empty text part, a second image, another role, reasoning or tool parts are refused. |
| Preprocessing | the checkpoint's ViT image processor: the whole image squashed to 384 x 384 with Pillow's bilinear filter, rescaled and normalized. Reproduced bit for bit for 8-bit pixels. Float32 pixels are first rounded to the nearest 8-bit level: exact for values k / 255 (a decoded 8-bit image); other float32 values are not compared with Transformers. |
| Decoding | greedy from `decoder_start_token_id` until `eos_token_id`; the end token stays in `token_ids`, special tokens are left out of `text`. |
| Parameter | `max_new_tokens` (1 to the bundle's `max_sequence_length`, default: until the end token or that capacity). Transformers' `generate()` stops at `max_length=20` unless told otherwise. |
| Build options | `--max-sequence-length` (2 to 512, default 512) sizes the decoder cache; the image size is the checkpoint's. Other precisions, batch, parallelism, quantization and `--kv-cache-size` are refused. |

## Checkpoints

Claimed: `VisionEncoderDecoderModel` with a `vit` encoder and a `trocr` decoder.
The builder refuses, by name, the TrOCR checkpoints with the other decoder graph
(`trocr-large-printed`, `trocr-*-stage1`: sinusoidal positions, ReLU, scaled
embeddings, untied output projection). The `trocr-small-*` checkpoints use a
DeiT encoder and are not claimed. `microsoft/trocr-base-printed` is validated
end to end. `microsoft/trocr-base-handwritten` (the same configuration) was only
measured with the graph-parity helpers on the printed case's images: the same
teacher-forced tokens and logits within the gate, and cross-attention states as
far from its GPU reference as that reference's own CPU run (up to 1.5e-3, above
the 1e-3 gate calibrated on the printed checkpoint).

## Validation

`tests/test_e2e.py` compares the native CLI on six single-line images, and the
public C and C++ SDK on three requests (two of those images) served by one
loaded model, with Transformers in float32: token ids and text must be equal.
The same module then compares the engines' cross-attention states and
teacher-forced logits with the reference (`tests/graph_parity.py`); injected defects such as
a tanh GELU or a wrong layer-norm epsilon keep the greedy tokens on every image,
so these gates catch what token equality cannot. An fp16 graph was measured
and not shipped: it kept every token but moved the logits two to three times
as far as the reference's own float16 run.
