# SAM2-HOI Phase-A build

Phase-A is opt-in. The default command continues to build the legacy
monolithic eight-section bundle.

Build the front-only image plan, 137 PAFPN leaf plans, and the other five
model plans with:

```bash
trtmc build /path/to/reviewed/hoi \
  --precision bf16 \
  --max-cache-length 7 \
  --set sam2_hoi.phase_a_pafpn=true \
  -o sam2-hoi-phase-a.bundle
```

The Phase-A builder uses TensorRT's Network Definition API directly. It emits
146 sections: 143 lazy plans plus eager `config.json`,
`sam2_hoi_pafpn_manifest.json`, and `sam2_hoi_native_plugin_so`. It does not
use ONNX or prebuilt plan/report artifacts. The resulting bundle must still
pass the model-owned five-frame `sam2-hoi-full-chain-accuracy-v2` gate before
benchmark publication.
