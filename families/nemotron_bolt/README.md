# Nemotron Bolt-74B

The `nemotron_bolt` family builds the prepared [Bolt-74B-A9B-24GB target](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB) for text, RGB images and timestamped video. Bolt uses a Puzzle decoder and a RADIO vision encoder. It is not the Nemotron Nano Omni family and does not accept audio. The optional [LiLiCorr companion](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr) is a draft, not a standalone model.

## Build

Build on the machine where inference will run. Set `TARGET` and `DRAFT` to the corresponding local checkpoint directories. BF16 selects model compute; it does **not** replace the target's prepared IQ2, Q8 and NVFP4 weights with a BF16 checkpoint.

```sh
trtmc nemotron_bolt build "$TARGET" --task vision_language_generation \
  --precision bf16 --max-sequence-length 8192 -o bolt.bundle

trtmc nemotron_bolt build "$TARGET" --task vision_language_generation \
  --precision bf16 --max-sequence-length 8192 \
  --execution-variant lilicorr --companion "draft=$DRAFT" -o bolt-lilicorr.bundle
```

Use `--task text_generation` when no visual encoder is needed. Both recipes use one device and one request at a time. The context limit includes the rendered prompt, visual tokens and requested output.

## How the backend is selected

- Optional Edge offload uses a matching 0.12.0 native SDK and checkpoint-direct builder. When the builder is in a separate Python environment, pass `--edge-builder-python /absolute/path/to/python`. A Python wheel alone does not supply Model Connect's linked C++ SDK.
- The Edge LiLiCorr route targets SM120. Other devices use the native implementation for this pair. If Edge preparation is attempted and fails, the builder warns and retries the same request with native TensorRT. It does not silently change the precision, drop the draft, or switch runtimes after an inference error.
- The native implementation constructs TensorRT graphs using the Python API. C++ owns tokenization, visual preprocessing, decoding and speculative verification. A LiLiCorr bundle includes the target-feature projection, five proposal layers, vocabulary projection and learned candidate lattice. Accepted tokens update the target state; rejected proposals do not.
- For native builds, install the dependencies in this family's `requirements.txt`. Build storage must accommodate staged sections **and** the final bundle. Edge bundles also need temporary space at inference to extract the plans and their external weights.

## Validation boundary

The ordinary and paired builds ran through both native and Edge. All four returned the exact first ten primes; the two seeded requests on each paired route produced identical token IDs. The premerge case covers the ordinary prime-list request, not visual quality. The separately selectable media and LiLiCorr cases check the full common names and the order of the animals. The same image and video requests executed through Model Connect and the released Edge runtime: both returned an underspecified image label and misidentified the second animal in the video. Those routes remain executable; this does **not** establish visual accuracy. Component comparisons and a single exact text answer do not qualify other prompts, modalities or hardware.

## Third-party material

Native builds use `gguf==0.19.0` to decode the IQ2 magnitude table; the native bundle includes its MIT notice. No checkpoint weights ship with Model Connect. The published target and companion each identify [OpenMDW-1.1](https://openmdw.ai/license/1-1/) in their model cards. A generated bundle includes the model license and any LICENSE, NOTICE and model card found in the supplied target and draft directories. Keep terms and notices with model-derived assets when redistributing.
