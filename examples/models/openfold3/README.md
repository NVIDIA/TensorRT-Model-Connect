# OpenFold3 native structure prediction

This example builds OpenFold3 with direct TensorRT Python APIs, then runs the
resulting bundle in native C++. Python, PyTorch, OpenFold3, ONNX, and external
preprocessing are not loaded after the bundle is built.

## Qualified profile

- OpenFold3 `v0.5.0` at
  `c4771653c5d0a3ebb0b3af71b05efd64bc44ee86` (Apache-2.0)
- public, ungated `openbind-2025-06-30-174k` checkpoint (Apache-2.0), SHA-256
  `bd43301c011d5f87580d3e8b548658869433e4488399feb03035ba248f8e29e4`
- `components.bcif`, SHA-256
  `473d845c8b250b188dbed9bf505ae206692a178a2a7c4869bf8f9de707ffcc0c`
- mixed FP16 by default: learned projections use FP16 while normalization,
  attention, pair/MSA contractions, reductions, diffusion state, confidence
  expectations, and external I/O remain FP32
- one protein chain containing the 20 standard amino acids, 1–128 tokens,
  query-only MSA depth 1, disabled template search, no ligand, 3 recycles
  (4 trunk passes), 1 diffusion sample, 200 steps, and seed 42

The 76-residue [ubiquitin request](query_ubiquitin.json) produces 601 real atoms
and 608 internally padded atom rows. Bundles are exact-request and exact-shape;
prepare and build a new bundle for another supported sequence length. Mixed BF16
remains available with `--precision bf16`; it uses BF16 triangle attention while
retaining FP32 pair/MSA contractions and other sensitive operations.

## Build

Run from the repository root in a CUDA/TensorRT environment. OpenFold3 is needed
only to prepare features and load the checkpoint during engine construction.

```bash
PACKAGE=/tmp/openfold3-ubiquitin
BUNDLE=/tmp/openfold3-ubiquitin.bundle
mkdir -p "$PACKAGE"

curl -fL \
  https://openfold3-data.s3.amazonaws.com/openfold3-parameters/of3-ob-2025-06-30-174k.pt \
  -o "$PACKAGE/of3-ob-2025-06-30-174k.pt"
curl -fL https://openfold3-data.s3.amazonaws.com/components.bcif \
  -o "$PACKAGE/components.bcif"

python -m pip install openfold3==0.5.0
python -m tensorrt_model_connect.families.openfold3.prepare_model_dir \
  --query examples/models/openfold3/query_ubiquitin.json \
  --components "$PACKAGE/components.bcif" \
  --output-dir "$PACKAGE"

# Mixed FP16 is the family default.
trtmc build "$PACKAGE" -o "$BUNDLE"
```

Preparation validates the pinned component dictionary, request envelope, dummy
template convention, and every feature shape. It writes a deterministic,
pickle-free feature archive. The isolated build profile verifies that the
OpenFold3 0.5.0 model source matches the pinned revision; the builder separately
verifies checkpoint and component sizes and hashes before compiling any plan.

The bundle contains input and atom embedding, the learned two-block dummy-template
path used when search is disabled, the four-block query-MSA module, all 48
Pairformer blocks, four recycling passes, diffusion conditioning, both atom
transformers, all 24 diffusion token blocks, the 200-step sampler, and all
confidence heads. Family-owned plans are composed on one CUDA stream.

## Run natively

```bash
trtmc predict-structure "$BUNDLE" \
  --input examples/models/openfold3/query_ubiquitin.json \
  --output /tmp/openfold3-ubiquitin.cif \
  --output-json /tmp/openfold3-ubiquitin.json
```

The standards-compliant mmCIF stores per-atom pLDDT in
`_atom_site.B_iso_or_equiv`. JSON metadata contains pLDDT, PAE, PDE, average
pLDDT, gPDE, pTM, sampling controls, precision, request digest, and rank. Ranking
score is not applicable because this profile emits exactly one sample.

The direct C++ API example is built as follows:

```bash
cmake -S examples/models/openfold3/native_structure_prediction \
  -B /tmp/openfold3-native -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/openfold3-native --target trtmc_openfold3_native -j

/tmp/openfold3-native/trtmc_openfold3_native "$BUNDLE" \
  --request examples/models/openfold3/query_ubiquitin.json \
  --output /tmp/openfold3-ubiquitin.cif \
  --backend-dir /tmp/openfold3-native \
  --model-plugin-dir /tmp/openfold3-native/models/openfold3
```

## Reproduce parity and performance

Generate the BF16 eager accuracy oracle from the exact prepared feature archive.
This is important because upstream preprocessing randomly rigid-transforms
`ref_pos`. OpenFold3's eager FP16 path is not used as an oracle: on this
qualification stack its diffusion rollout produces non-finite coordinates. The
TensorRT mixed-FP16 graph retains its sensitive normalization, attention,
reduction, diffusion-state, and output operations in FP32 and remains finite.

```bash
python examples/models/openfold3/generate_reference.py \
  --query "$PACKAGE/query.json" \
  --components "$PACKAGE/components.bcif" \
  --features "$PACKAGE/openfold3_features.npz" \
  --checkpoint "$PACKAGE/of3-ob-2025-06-30-174k.pt" \
  --precision bf16 \
  --output-dir /tmp/openfold3-reference
```

Run `qualify.py` against that output and the fixed thresholds under
`qualification/thresholds/`. For warmed native timing, use
`predict-structure --warmup 3 --benchmark 10`. The aligned eager and compiled
baselines use the same prepared inputs and exclude preprocessing, checkpoint
loading, engine construction, and compilation:

```bash
for MODE in eager compile; do
  python examples/models/openfold3/benchmark_reference.py \
    --query "$PACKAGE/query.json" \
    --components "$PACKAGE/components.bcif" \
    --features "$PACKAGE/openfold3_features.npz" \
    --checkpoint "$PACKAGE/of3-ob-2025-06-30-174k.pt" \
    --precision bf16 --mode "$MODE" --warmup 2 --iterations 5 \
    --output "/tmp/openfold3-${MODE}.json"
done
```

### GB300 qualification results

The 76-token ubiquitin profile was qualified on one NVIDIA GB300 with
TensorRT 11.2.1.2. Latency excludes preparation and engine construction. FP16
combines two 10-run trials; BF16 reports one 10-run trial after three warmups.

| Profile | Mean / p50 latency | Throughput | 1UBQ CA RMSD | BF16-oracle CA RMSD | pLDDT Pearson |
| --- | ---: | ---: | ---: | ---: | ---: |
| mixed FP16 (default) | 1050.7 / 981.5 ms | 0.952 samples/s | 2.671 Å | 2.589 Å | 0.937 |
| mixed BF16 | 808.0 / 807.4 ms | 1.238 samples/s | 0.919 Å | 1.051 Å | 0.986 |

Against the aligned BF16 PyTorch oracle (19,737.4 ms eager and 11,078.4 ms
compiled), native FP16 is 18.8x / 10.5x faster and native BF16 is 24.4x /
13.7x faster. Startup peak-memory deltas were 2,163 MiB for FP16 and 2,189 MiB
for BF16. The upstream FP16 rollout is non-finite, so the FP16 accuracy gate
uses experimental CA RMSD, local pair distances, and confidence correlations
rather than rigid trajectory agreement.

Seeded diffusion is deterministic for a serialized bundle, but numerically
equivalent TensorRT tactics can select a different valid diffusion basin.
Consequently, strict BF16 trajectory evidence is bound to the recorded bundle
SHA, and every rebuild must be requalified. Exact quality, validity, timing,
memory, artifact hashes, and software-stack evidence is recorded in the
adjacent `qualification/` directory.

## Limits

This profile does not qualify external or paired MSAs, searched templates,
nucleic acids, ligands, modified residues with unsupported atom layouts,
multiple chains or samples, runtime preprocessing, runtime-variable shapes,
CUDA graphs, training, or sequences longer than 128 tokens. Inputs outside the
recorded request and sampling contract are rejected instead of running silently.
