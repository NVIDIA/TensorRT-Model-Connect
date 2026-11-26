# PatchTST

This family builds PatchTST time-series models from Hugging Face checkpoints
(`ibm-granite/granite-timeseries-patchtst`, a 7-channel 96-step forecaster, and
`ibm-research/patchtst-etth1-regression-distribution`, a 6-channel regression model
with a normal distribution head, in the manifests) as one TensorRT engine: batch 1,
the context length of the checkpoint, fp16 in the manifests (fp32 is also supported).
The encoder runs once per channel row; every linear adds one weight constant that all
channel rows share.

## FP8 (`quantization=fp8`)

No FP8 or NVFP4 checkpoint of PatchTST exists on the Hub, so `quantization=fp8`
quantizes the original checkpoint at build time (fp16 precision only):

| Part | Precision |
|---|---|
| Encoder `q_proj`, `k_proj`, `v_proj` (one shared input scale), `ff.0`, `ff.3` | FP8 e4m3: weights per tensor (amax / 448), activations per tensor (static scale) |
| Attention output projection, patch embedding, positional encoding, norms, attention, head | fp16 |

The attention output projection stays fp16 because an FP8 Quantize right after the
fused attention does not build on TensorRT 11.4. The activation scales were
calibrated once with ModelOpt `FP8_DEFAULT_CFG` (max calibration, 512 windows from the
ETTh1 and ETTh2 training split) and are stored in `fp8_activation_scales.json`, keyed
by model config and a weight fingerprint. An already quantized checkpoint is refused.

Measured at batch 1, TensorRT 11.4, CUDA graph, median of 21 interleaved rounds, for
`ibm-granite/granite-timeseries-patchtst`:

| GPU | fp16 ms | FP8 ms | FP8 vs fp16 | Plan size fp16 to FP8 |
|---|---|---|---|---|
| B200 | 0.326 | 0.285 | 1.14x | 3.17 to 2.63 MiB |
| B300 | 0.313 | 0.266 | 1.18x | 3.14 to 2.63 MiB |

On 1213 held-out ETT test windows the forecast relative L2 error against the Hugging
Face fp32 model is 0.005 to 0.023 (fp16: 1e-4 to 5e-4), and MSE and MASE change by
less than 0.1%. On TensorRT 11.1 the FP8 engine is slower than fp16 (0.93x).

Only the granite forecaster is supported in FP8. The regression checkpoint
(`ibm-research/patchtst-etth1-regression-distribution`) has no committed scales, so
an FP8 build of it raises the missing-scale error: in FP8 its predicted location
moves by 7% to 12% relative L2 from the Hugging Face model (ETTh1 0.097, ETTh2 0.066,
ETTm1 0.116; fp16 about 0.0013), and 4 of 242 ETTh2 windows flip the sign of the
location. Its 1.14x (B200) to 1.19x (B300) speedup did not justify that error. To try
it anyway, calibrate it and rebuild:

```
python -m families.patchtst.calibrate_fp8_scales \
    --model-dir <regression checkpoint> --ett-dir <ETDataset>/ETT-small
```

This adds an entry for that model config to `fp8_activation_scales.json`. The same
command is needed for any other PatchTST checkpoint.
