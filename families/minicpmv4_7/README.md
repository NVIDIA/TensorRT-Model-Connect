# MiniCPM-V 4.7

This family owns image-and-text generation for the MiniCPM-V 4.7 checkpoint identity.
The 35B-A3B path uses a Qwen3.5-MoE decoder; the 1B path uses a dense Qwen3.5 decoder.
Both use the existing vision tower, canvas M-RoPE and explicit recurrent/KV state contract.
The builder and native tokenizer validate the separate released tokenizer profiles.

The new 1B path is experimental and has not been GPU-qualified. Its manifest pins
`openbmb/MiniCPM-V-4.7-1B` at `ad9bf379feb71204c888b22923d63322452407f9`,
builds FP32, and checks the provider host contract and teacher-forced FP32 argmax
with a strict zero logit-gap gate. The 35B BF16 near-tie thresholds are unchanged.
CPU configuration, checkpoint-header and graph-math checks do not establish engine correctness.

The E2E selector accepts either manifest name, individual case names or `minicpmv4_7`.
Checkpoint artifacts must already be in the local Hugging Face cache; the E2E never downloads weights.
BF16 qualification of the 1B path and calibration of any nonzero tolerance remain outstanding.
